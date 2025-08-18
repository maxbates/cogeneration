import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from sympy import Float

from cogeneration.config.base import (
    InterpolantSteeringConfig,
    ModelESMKey,
    ProteinMPNNRunnerConfig,
)
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.tools.protein_mpnn_runner import ProteinMPNNRunner
from cogeneration.data.trajectory import SamplingStep
from cogeneration.dataset.interaction import DIST_INTERACTION_BACKBONE
from cogeneration.models.esm_frozen import FrozenEsmModel
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures


def _batch_repeat_feat(v: Any, K: int) -> Any:
    """Duplicate the feature K times along batch-dim (dim 0 for tensors)."""
    if torch.is_tensor(v):
        return v.repeat_interleave(K, dim=0)
    elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        # assume len(v) == B, repeat each element K times
        return [x for x in v for _ in range(K)]
    else:
        # primitive
        return [v] * K


def _batch_select_feat(v: Any, idx: torch.Tensor) -> Any:
    """Pick batch elements at `idx`."""
    if torch.is_tensor(v):
        return v.index_select(0, idx)
    elif isinstance(v, Sequence) and not isinstance(v, (str, bytes)):
        idx_list = idx.tolist()
        return [v[i] for i in idx_list]
    else:
        # primitive
        return v


# PotentialEnergy is energy computed by potential: (B,) tensor scaled to [0, 1]
PotentialEnergy = torch.Tensor


@dataclass
class PotentialField:
    """
    Vector fields / logits guidance, e.g. produced by a potential.
    These will be added to the model prediction in the interpolant.

    The fields/logits should be scaled appropriately and masked when used by interpolant.
    The interpolant will perform euler step `d_t` using modified fields.

    Potentials are cached by the FKSteeringCalculator, and may be reused on steps without resampling.
    This is so that the same potential can easily be used across multiple steps,
    rather than imposing large jumps every re-sampling step.
    (Also, for aatype logits, unmasking prob is proportional to `d_t`,
    and not possible to scale logits to influence this, so seems like the simplest workaround.)
    """

    # trans VF (B, N, 3)
    trans: Optional[torch.Tensor] = None
    # rotmats VF tangent to current frame (B, N, 3)
    rotmats: Optional[torch.Tensor] = None
    # logits for aatypes (B, N, S)
    logits: Optional[torch.Tensor] = None

    def decayed(self, scale: float) -> "PotentialField":
        """Return a new PotentialField with each component scaled by `scale`."""
        if scale == 1.0:
            return self
        if scale == 0.0:
            return PotentialField()

        return PotentialField(
            trans=self.trans * scale if self.trans is not None else None,
            rotmats=self.rotmats * scale if self.rotmats is not None else None,
            logits=self.logits * scale if self.logits is not None else None,
        )

    def mask(self, mask: torch.Tensor):
        """Apply (B, N) mask to each field (broadcast on last dim)."""
        if self.trans is not None:
            self.trans = self.trans * mask[..., None]
        if self.rotmats is not None:
            self.rotmats = self.rotmats * mask[..., None]
        if self.logits is not None:
            self.logits = self.logits * mask[..., None]

    def __getitem__(self, idx) -> "PotentialField":
        """Slice the potential field to a subset of the batch."""
        return PotentialField(
            trans=self.trans[idx] if self.trans is not None else None,
            rotmats=self.rotmats[idx] if self.rotmats is not None else None,
            logits=self.logits[idx] if self.logits is not None else None,
        )

    def __add__(self, other: "PotentialField") -> "PotentialField":
        """Elementwise add two potential fields, treating None as identity."""
        if other is None:
            return self
        if not isinstance(other, PotentialField):
            raise ValueError(f"Cannot add PotentialField and {type(other)}")

        trans = (
            other.trans
            if self.trans is None
            else self.trans if other.trans is None else self.trans + other.trans
        )
        rotmats = (
            other.rotmats
            if self.rotmats is None
            else self.rotmats if other.rotmats is None else self.rotmats + other.rotmats
        )
        logits = (
            other.logits
            if self.logits is None
            else self.logits if other.logits is None else self.logits + other.logits
        )
        return PotentialField(trans=trans, rotmats=rotmats, logits=logits)


@dataclass
class Potential(ABC):
    """Base class for Feynman-Kac steering potentials.

    Potentials compute zero-bottomed energy (rather than rewards) for each member of a batch.
    Energy should be scaled to be in the range [0, 1], where 0 is the best score.

    Potentials can also compute guidance, a vector field to guide translations, rotations, or aatypes logits.

    Each potential receives:
    - `protein_state` the current protein state
    - `model_pred` the raw model prediction
    - `protein_pred` the modified prediction
    """

    energy_scale: float = 1.0  # compute() energy * energy_scale is the final energy
    guidance_scale: float = (
        1.0  # compute() guidance * guidance_scale is the final guidance
    )

    @abstractmethod
    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, Optional[PotentialField]]:
        """Return the energy (B,) scaled to [0, 1] and optional guidance per sample in the batch."""


@dataclass
class FKStepMetric:
    """
    Metrics for each resampling step using FK steering. lists are len `num_batch` * `num_particles`
    """

    step: int
    energy: List[float]
    log_G: List[float]
    log_G_delta: List[float]
    weights: List[float]
    effective_sample_size: Float
    keep: List[int]
    guidance: Optional[PotentialField] = None

    def log(self) -> str:
        return dedent(
            f"""
        Step {self.step} | energy = {self.energy}
        Step {self.step} | ∆G     = {self.log_G_delta}
        Step {self.step} | Log G  = {self.log_G}
        Step {self.step} | ESS    = {self.effective_sample_size}
        Step {self.step} | keep   = {self.keep}
        """
        )


@dataclass
class FKSteeringTrajectory:
    """
    A trajectory of Feynman-Kac steering metrics across sampling steps.

    Stores FK steering metrics for each resampling step during inference,
    allowing analysis of particle energy, weights, and effective sample size over time.
    """

    num_batch: int
    num_particles: int
    metrics: List[FKStepMetric] = field(default_factory=list)

    def append(self, metric: Optional[FKStepMetric]):
        """
        Append an FK step metric. If metric is None (steering disabled), skip.
        """
        if metric is not None:
            self.metrics.append(metric)

    @property
    def num_steps(self):
        return len(self.metrics)

    def batch_sample_slice(self, batch_idx: int) -> "FKSteeringTrajectory":
        """
        Extract FK steering trajectory for a specific batch member.

        Args:
            batch_idx: Index of the batch member (0-based, in original batch before particle expansion)

        Returns:
            FKSteeringTrajectory containing only metrics & guidance for the specified batch member (metrics for all particles)
        """
        if batch_idx >= self.num_batch:
            raise ValueError(f"batch_idx {batch_idx} >= num_batch {self.num_batch}")

        start_idx = batch_idx * self.num_particles
        end_idx = start_idx + self.num_particles

        sliced_metrics = []
        for metric in self.metrics:
            # weights is always a list of lists (one per batch member) when FK steering is enabled
            # since resampling_weights has shape (num_batch, num_particles)
            weights_slice = metric.weights[batch_idx]

            sliced_metric = FKStepMetric(
                step=metric.step,
                energy=metric.energy[start_idx:end_idx],
                log_G=metric.log_G[start_idx:end_idx],
                log_G_delta=metric.log_G_delta[start_idx:end_idx],
                weights=weights_slice,
                effective_sample_size=metric.effective_sample_size,  # scalar, keep as-is
                keep=metric.keep[start_idx:end_idx],
                guidance=metric.guidance[start_idx:end_idx],
            )
            sliced_metrics.append(sliced_metric)

        return FKSteeringTrajectory(
            num_batch=1,
            num_particles=self.num_particles,
            metrics=sliced_metrics,
        )


@dataclass
class ChainBreakPotential(Potential):
    """
    Penalize unnatural backbone chain breaks by checking residue Euclidean distances,
    while accounting for true chain breaks or non-sequential residues.
    """

    allowed_backbone_dist: float = 4.0  # allowed distance, ideal is 3.8Å
    maximum_backbone_dist: float = 12.0  # upper bound

    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, Optional[PotentialField]]:
        num_batch, num_res = batch[bp.res_idx].size()  # (B, N)
        chain_idx = batch[bp.chain_idx]
        res_idx = batch[bp.res_idx]

        # use protein_pred, e.g. for inpainting after fixing motifs
        trans = protein_pred.trans  # (B, N, 3)

        # dists from one residue to the next
        dists = torch.linalg.norm(
            trans[:, :-1, :] - trans[:, 1:, :], dim=-1
        )  # (B, N-1)
        # excess penalty is >0Å above allowed distance
        excess = torch.relu(dists - self.allowed_backbone_dist)
        # normalize penalties to [0, 1] range
        span = max(1e-6, self.maximum_backbone_dist - self.allowed_backbone_dist)
        penalty_pair = excess.clamp(max=span) / span  # (B, N-1)

        # continuity masks
        same_chain = chain_idx[:, 1:].eq(chain_idx[:, :-1])
        consecutive = res_idx[:, 1:].eq(res_idx[:, :-1] + 1)
        need_continuity = same_chain & consecutive
        penalty_pair = penalty_pair * need_continuity.float()

        # aggregate and logistic ramp
        penalty_sum = penalty_pair.sum(dim=1)
        energy = 1.0 - torch.exp(-1 * penalty_sum)  # in [0, 1]
        return energy, None


@dataclass
class HotSpotPotential(Potential):
    """
    Potential that encourages residues marked as hot spots to be in contact across chains.
    Each hot spot residue should have at least one residue from another chain within contact distance.
    Only applied to multimer structures with hot spots defined.

    TODO - this is basically the same as the `LossCalculator.loss_hot_spots`, should de-duplicate.
    """

    contact_distance_ang: float = DIST_INTERACTION_BACKBONE + 0.5
    max_distance_penalty: float = (
        10.0  # maximum distance penalty, beyond contact_distance threshold
    )

    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, Optional[PotentialField]]:
        num_batch, num_res = batch[bp.res_idx].size()  # (B, N)
        device = batch[bp.res_idx].device

        # Skip if no hot spots marked
        if bp.hot_spots not in batch:
            return torch.zeros(num_batch, device=device), None

        hot_spots_mask = batch[bp.hot_spots].bool()  # (B, N) - convert to bool
        hot_spots_present = hot_spots_mask.sum(-1) > 0  # (B,)
        if not hot_spots_present.any():
            return torch.zeros(num_batch, device=device), None

        # Only apply to multimers
        chain_idx = batch[bp.chain_idx]  # (B, N)
        multi_chain_mask = chain_idx.max(-1).values != chain_idx.min(-1).values  # (B,)
        if not multi_chain_mask.any():
            return torch.zeros(num_batch, device=device), None

        # Use frame translations
        pred_trans = protein_pred.trans  # (B, N, 3)

        # Calculate pairwise distances between CA atoms
        pairwise_ca = pred_trans.unsqueeze(2) - pred_trans.unsqueeze(1)  # (B, N, N, 3)
        pairwise_dists = torch.norm(pairwise_ca, dim=-1)  # (B, N, N)

        # Mask for inter-chain residue pairs
        chain_i = chain_idx.unsqueeze(2)  # (B, N, 1)
        chain_j = chain_idx.unsqueeze(1)  # (B, 1, N)
        inter_chain = chain_i != chain_j  # (B, N, N)

        # Valid residue pairs (both residues must be valid)
        res_mask = batch[bp.res_mask].bool()  # (B, N) - convert to bool
        res_i_mask = res_mask.unsqueeze(2)  # (B, N, 1)
        res_j_mask = res_mask.unsqueeze(1)  # (B, 1, N)
        valid_pairs = res_i_mask & res_j_mask  # (B, N, N)

        # Mask for valid inter-chain pairs
        valid_inter_chain = inter_chain & valid_pairs  # (B, N, N)

        # For each hot spot residue, find minimum distance to any residue on another chain
        # Set distances to invalid pairs to a large value so they don't affect the min
        masked_dists = pairwise_dists.clone()
        masked_dists[~valid_inter_chain] = float("inf")

        # Find minimum distance to any other-chain residue for each residue
        min_inter_chain_dist, _ = masked_dists.min(dim=2)  # (B, N)

        # Only consider hot spot residues for energy calculation
        hot_spot_min_dists = min_inter_chain_dist * hot_spots_mask.float()  # (B, N)

        # Replace inf values (no valid inter-chain residues) with contact distance to avoid large energies
        hot_spot_min_dists = torch.where(
            torch.isinf(hot_spot_min_dists),
            torch.tensor(self.contact_distance_ang, device=device),
            hot_spot_min_dists,
        )

        # Apply relu penalty: penalize when minimum distance exceeds contact threshold
        contact_penalty = torch.relu(
            hot_spot_min_dists - self.contact_distance_ang
        )  # (B, N)
        contact_penalty = contact_penalty.clamp(
            max=self.max_distance_penalty
        )  # cap distance

        # Only apply penalty to actual hot spots
        hot_spot_penalties = contact_penalty * hot_spots_mask.float()  # (B, N)

        # Sum over hot spots and normalize by number of hot spots
        penalty_per_batch = hot_spot_penalties.sum(dim=1)  # (B,)
        num_hot_spots = hot_spots_mask.float().sum(dim=1).clamp_min(1.0)  # (B,)
        penalty_per_batch = penalty_per_batch / num_hot_spots  # (B,)

        # Only apply to relevant batches
        penalty_per_batch = (
            penalty_per_batch * hot_spots_present.float() * multi_chain_mask.float()
        )

        # Normalize energy to [0, 1] range
        # Energy should be 0 when contacts are satisfied, 1 when maximally violated
        energy = penalty_per_batch / self.max_distance_penalty
        energy = energy.clamp(min=0.0, max=1.0)

        return energy, None


@dataclass
class ContactConditioningPotential(Potential):
    """
    Potential that encourages adherence to contact conditioning constraints.
    Uses ReLU penalty for distances that exceed the defined contact thresholds.
    Normalizes energy over the number of defined contacts.
    Optionally ignores contacts in motifs and upweights inter-chain contacts.

    TODO - consider supporting a schedule to scale guidance over time
    """

    tolerance_dist: float = 0.5  # tolerance distance for contact conditioning
    inter_chain_weight: float = 2.0  # upweight factor for inter-chain contacts
    max_distance_penalty: float = 10.0  # maximum distance penalty
    ignore_motif_motif_in_energy: bool = True
    potential_max_force_angstroms: float = 1.0  # maximum force magnitude in angstroms

    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, Optional[PotentialField]]:
        B, N = batch[bp.res_idx].size()
        chain_idx = batch[bp.chain_idx]  # (B, N)
        device = batch[bp.res_idx].device

        if bp.contact_conditioning not in batch:
            return torch.zeros(B, device=device), None

        contact_target_dist = batch[bp.contact_conditioning]  # (B, N, N)
        contact_cond_mask = contact_target_dist > 0  # (B, N, N)

        if not contact_cond_mask.any():
            return torch.zeros(B, device=device), None

        # Energy, calculated using predicted structure

        pred_trans = protein_pred.trans  # (B, N, 3)
        pred_dists = torch.norm(
            pred_trans.unsqueeze(2) - pred_trans.unsqueeze(1), dim=-1
        )  # (B, N, N)

        # tolerance band penalties (ReLU outside [targets +/- tolerance_dist])
        shortfall = torch.relu(contact_target_dist - pred_dists - self.tolerance_dist)
        excess = torch.relu(pred_dists - contact_target_dist - self.tolerance_dist)
        penalties = (shortfall + excess).clamp(max=self.max_distance_penalty)

        # If inpainting, drop motif-motif contacts since they are subject to motif guidance
        motif_mask = batch.get(bp.motif_mask, None)
        if motif_mask is not None and self.ignore_motif_motif_in_energy:
            mm = motif_mask.bool()
            motif_mask_2d = (
                mm.unsqueeze(2) & mm.unsqueeze(1)
            ).float()  # both residues motif
            penalties = penalties * (1.0 - motif_mask_2d)  # zero motif-motif only

        # only defined contacts contribute
        penalties = penalties * contact_cond_mask.float()

        # optional inter-chain upweighting
        if self.inter_chain_weight != 1.0:
            inter_chain_mask = (
                chain_idx.unsqueeze(2) != chain_idx.unsqueeze(1)
            ) & contact_cond_mask  # (B, N, N)
            weights = torch.ones_like(penalties)
            weights[inter_chain_mask] = self.inter_chain_weight
            penalties = penalties * weights

        # normalize energy to [0, 1]
        num_contacts = contact_cond_mask.float().sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
        energy = (penalties.sum(dim=(1, 2)) / num_contacts) / self.max_distance_penalty
        energy = energy.clamp(0.0, 1.0)

        # Guidance vector field, calculated using the current state:
        # compute current distances, determine errors from conditioning target,
        # compute force on each residue from other residues, sum, normalize, scale

        trans_t = protein_state.trans  # (B, N, 3) current coords
        dists_t = trans_t.unsqueeze(2) - trans_t.unsqueeze(1)  # (B, N, N, 3)
        pairwise_dist = torch.linalg.norm(dists_t, dim=-1)  # (B, N, N)

        # active outside the tolerance band
        too_far = pairwise_dist > (
            contact_target_dist + self.tolerance_dist
        )  # => shorten
        too_close = pairwise_dist < (
            contact_target_dist - self.tolerance_dist
        )  # => lengthen
        active = (too_far | too_close) & contact_cond_mask

        # error magnitude beyond tolerance, ReLU capped to max_distance_penalty
        err_far = torch.relu(pairwise_dist - contact_target_dist - self.tolerance_dist)
        err_close = torch.relu(
            contact_target_dist - pairwise_dist - self.tolerance_dist
        )
        mag = (err_far + err_close).clamp(max=self.max_distance_penalty)  # (B, N, N)

        # VF u_ij points from i -> j
        # too far => negative => pull res i toward target res j
        # too close => positive => push res i away from target res j
        sign = (-1.0) * too_far.float() + (1.0) * too_close.float()  # (B, N, N)
        coef = (mag * sign) * active.float()  # signed strength per edge

        # optional inter-chain upweighting on guidance too
        if self.inter_chain_weight != 1.0:
            inter_chain_mask = (
                chain_idx.unsqueeze(2) != chain_idx.unsqueeze(1)
            ) & active
            coef = torch.where(inter_chain_mask, coef * self.inter_chain_weight, coef)

        # compute unit directions and per-edge forces (on each residue from other residues)
        u = dists_t / (pairwise_dist.clamp_min(1e-8).unsqueeze(-1))  # (B, N, N, 3)
        force = coef.unsqueeze(-1) * u  # (B, N, N, 3) force on res i from res j

        # accumulate per-residue forces symmetrically
        trans_vf = force.sum(dim=2) - force.sum(dim=1)  # (B, N, 3)

        # normalize by number of active interactions to stabilize scale
        cnt = active.float().sum(dim=2) + active.float().sum(dim=1)  # (B, N)
        trans_vf = trans_vf / cnt.clamp_min(1.0).unsqueeze(-1)

        # if motif_mask exists, only apply VF to scaffolds
        if motif_mask is not None:
            scaffold = (~motif_mask.bool()).float().unsqueeze(-1)  # (B, N, 1)
            trans_vf = trans_vf * scaffold

        # global guidance scale
        trans_vf = trans_vf * float(self.guidance_scale)

        # cap max force, clip by norm (equivariant)
        max_force = self.potential_max_force_angstroms
        norm = trans_vf.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        trans_vf = trans_vf * (max_force / norm).clamp(max=1.0)

        return energy, PotentialField(trans=trans_vf)


@dataclass
class InverseFoldPotential(Potential):
    """
    Potential that inverse folds predicted structure using ProteinMPNN,
    and compares predicted sequence to ProteinMPNN logits.

    For now, we always generate a single sequence per sample.
    """

    inverse_fold_logits_temperature: float = 1.0
    inverse_fold_guidance_scale: float = 1.0
    inverse_fold_logits_cap: float = 4.0
    protein_mpnn_cfg: ProteinMPNNRunnerConfig = field(
        default_factory=ProteinMPNNRunnerConfig
    )

    def __post_init__(self):
        # initialize logger for optional warnings
        self.log = logging.getLogger(__name__)
        self.logged_warning = False

        # Runner won't load model until run inference.
        self.protein_mpnn_runner = ProteinMPNNRunner(cfg=self.protein_mpnn_cfg)

    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, Optional[PotentialField]]:
        if not self.logged_warning:
            self.log.warning(
                f"🐌 InverseFoldPotential is enabled, generating 1 sequence per particle + timestep, and will increase sampling time."
            )
            if not self.protein_mpnn_cfg.use_native_runner:
                self.log.warning(
                    "🐌 InverseFoldPotential is enabled, but using subprocess runner. This will increase sampling time."
                )
            self.logged_warning = True

        trans = protein_pred.trans  # (B, N, 3)
        rotmats = protein_pred.rotmats  # (B, N, 3, 3)
        aatypes = protein_pred.aatypes  # (B, N)
        torsions = protein_pred.torsions  # Optional (B, N, 7, 2)
        res_mask = batch[bp.res_mask]  # (B, N)
        diffuse_mask = batch[bp.diffuse_mask]  # (B, N)
        chain_idx = batch[bp.chain_idx]  # (B, N)

        # Run ProteinMPNN to get sequence logits for the predicted structure
        mpnn_result = self.protein_mpnn_runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            torsions=torsions,
            num_passes=1,
            sequences_per_pass=1,
        )

        # In practice we only have 1 pass / 1 seq, but use mean over logits
        logits = mpnn_result.averaged_logits  # (B, N, S)
        logits = logits.to(aatypes.device)

        # Targets are the predicted amino acids (ensure device matches logits)
        targets = aatypes.long()  # (B, N)

        # Compute per-position negative log likelihood
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, N, S)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(
            -1
        )  # (B, N)

        # Only score valid, designed residues (move mask to logits device)
        valid_mask = (res_mask > 0) & (diffuse_mask > 0)  # (B, N)
        nll = nll * valid_mask.float()

        # Normalize per batch item
        denom = valid_mask.float().sum(dim=-1).clamp_min(1.0)  # (B,)
        avg_nll = nll.sum(dim=-1) / denom  # (B,)

        # Scale closer to [0, 1] by dividing by log(medium-high-perplexity)
        perplexity_cap = 7.0  # less than vocab_size = logits.shape[-1]
        max_ce = float(torch.log(torch.tensor(perplexity_cap, device=logits.device)))
        energy = avg_nll / max_ce

        # Set up ProteinMPNNlogits for aatypes guidance

        # Ensure logits shapes are in agreement with model prediction
        if logits.shape != model_pred.logits.shape:
            # add mask logit to ProteinMPNN logits
            if logits.shape[-1] == 20 and model_pred.logits.shape[-1] == 21:
                pad = torch.zeros(
                    logits.shape[:-1] + (1,), device=logits.device, dtype=logits.dtype
                )
                logits = torch.cat([logits, pad], dim=-1)
            else:
                raise ValueError(
                    f"logits.shape {logits.shape} incompatible with protein_pred.aatypes.shape {model_pred.logits.shape}"
                )

        # Apply temperature
        T = max(self.inverse_fold_logits_temperature, 1e-4)  # avoid div-by-zero
        logits = logits / T

        # Scale by logits guidance factor
        logits = logits * self.guidance_scale

        # Scale and clamp very strong logits
        cap = float(self.inverse_fold_logits_cap)
        logits_f = logits.float()  # compute in fp32 for stability
        max_abs = logits_f.abs().amax(dim=-1, keepdim=True)  # (B, N, 1)
        scale_down = (cap / (max_abs + 1e-6)).clamp(max=1.0)
        logits_f = (logits_f * scale_down).clamp(min=-cap, max=cap)
        logits = logits_f.to(logits.dtype)

        guidance = PotentialField(
            logits=logits,
        )

        return energy, guidance


@dataclass
class ESMLogitsPotential(Potential):
    """
    Potential that runs a frozen ESM model on the current sequence and produces
    amino-acid logits guidance. Energy is the average negative log likelihood of
    the current sequence under ESM logits, normalized to [0, 1].

    Only residues with `res_mask & diffuse_mask` and that are not masked tokens
    contribute to energy. Guidance logits are returned for all residues; they
    will be masked by FK steering to designed positions.
    """

    esm_model_key: ModelESMKey = ModelESMKey.esm2_t30_150M_UR50D
    esm_logits_temperature: float = 1.0
    esm_logits_cap: float = 4.0

    def __post_init__(self):
        # Do not compute pair attentions; only logits are required
        self.esm = FrozenEsmModel(model_key=self.esm_model_key, use_esm_attn_map=False)

    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, Optional[PotentialField]]:
        # Inputs for ESM
        aatypes = protein_pred.aatypes  # (B, N)
        chain_idx = batch[bp.chain_idx]  # (B, N)
        res_mask = batch[bp.res_mask]  # (B, N)
        diffuse_mask = batch[bp.diffuse_mask]  # (B, N)

        # Run ESM to get logits (B, N, 21)
        _, _, logits = self.esm(
            aatypes=aatypes,
            chain_index=chain_idx,
            res_mask=res_mask,
        )

        # Energy: average NLL over valid positions
        targets = aatypes.long()  # (B, N)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  # (B, N, 21)
        nll = -log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

        # nll valid if real residue and not masked token
        valid_mask = (res_mask > 0) & (aatypes != MASK_TOKEN_INDEX)  # (B, N)
        nll = nll * valid_mask.float()

        denom = valid_mask.float().sum(dim=-1).clamp_min(1.0)  # (B,)
        avg_nll = nll.sum(dim=-1) / denom  # (B,)

        perplexity_cap = 7.0
        max_ce = float(torch.log(torch.tensor(perplexity_cap, device=logits.device)))
        energy = (avg_nll / max_ce).clamp(min=0.0, max=1.0)

        # Guidance: temperature scale and cap logits magnitude
        T = max(self.esm_logits_temperature, 1e-4)
        logits = logits / T
        logits = logits * float(self.guidance_scale)

        cap = float(self.esm_logits_cap)
        logits_f = logits.float()
        max_abs = logits_f.abs().amax(dim=-1, keepdim=True)
        scale_down = (cap / (max_abs + 1e-6)).clamp(max=1.0)
        logits_f = (logits_f * scale_down).clamp(min=-cap, max=cap)
        logits = logits_f.to(logits.dtype)

        guidance = PotentialField(logits=logits)

        return energy, guidance


@dataclass
class FKSteeringCalculator:
    """
    Helper to implement simplified Feynman-Kac steering for inference.

    Several potentials only compute energies for particle resampling,
    but some also compute "PotentialFields" to guide trajectories.
    For translations and rotations, these are vector fields, and for aatypes, these are logits.

    Feynman-Kac steering is a general framework for diffusion models for tilting sampling at inference,
    and which lends itself nicely to flow matching with stochastic paths.

    One nice characteristic of FK steering is that no fine tuning is required,
    and so new potentials can easily be added to tilt sampling towards desired properties.

    In this relatively simple implementation, similar to Boltz-1, we use potentials to compute energy
    of the model's t=1 prediction (where low energy => high reward).

    During sampling, we sample K particles per sample,
    which particles are resampled according to their energy.
    """

    cfg: InterpolantSteeringConfig
    potentials: Optional[List[Potential]] = None

    def __post_init__(self):
        # Instantiate requested potentials from cfg if not provided explicitly.
        if not self.potentials:
            self.potentials = FKSteeringCalculator.default_potentials(self.cfg)

    @staticmethod
    def default_potentials(cfg: InterpolantSteeringConfig) -> List[Potential]:
        return [
            ChainBreakPotential(
                energy_scale=cfg.chain_break_energy_scale,
                allowed_backbone_dist=cfg.chain_break_allowed_backbone_dist,
                maximum_backbone_dist=cfg.chain_break_maximum_backbone_dist,
            ),
            HotSpotPotential(
                energy_scale=cfg.hot_spot_energy_scale,
                contact_distance_ang=cfg.hot_spot_contact_distance_ang,
                max_distance_penalty=cfg.hot_spot_max_distance_penalty,
            ),
            ContactConditioningPotential(
                energy_scale=cfg.contact_conditioning_energy_scale,
                inter_chain_weight=cfg.contact_conditioning_inter_chain_weight,
                max_distance_penalty=cfg.contact_conditioning_max_distance_penalty,
            ),
            InverseFoldPotential(
                energy_scale=cfg.inverse_fold_energy_scale,
                guidance_scale=cfg.inverse_fold_guidance_scale,
                inverse_fold_logits_temperature=cfg.inverse_fold_logits_temperature,
                inverse_fold_logits_cap=cfg.inverse_fold_logits_cap,
                protein_mpnn_cfg=cfg.protein_mpnn,
            ),
            ESMLogitsPotential(
                energy_scale=cfg.esm_logits_energy_scale,
                guidance_scale=cfg.esm_logits_guidance_scale,
                esm_logits_temperature=cfg.esm_logits_temperature,
                esm_logits_cap=cfg.esm_logits_cap,
                esm_model_key=cfg.esm_model_key,
            ),
        ]

    def compute(
        self,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[PotentialEnergy, PotentialField]:
        """Weighted sum of all configured potentials."""
        num_batch, num_res = batch[bp.res_idx].size()  # (B, N)
        device = batch[bp.res_idx].device

        total_energy = torch.zeros(num_batch, device=device)
        guidances = []
        for potential in self.potentials:
            if potential.energy_scale == 0.0:
                continue

            energy, guidance = potential.compute(
                batch=batch,
                protein_state=protein_state,
                model_pred=model_pred,
                protein_pred=protein_pred,
            )

            energy = energy.to(device)
            energy = energy.clamp(min=0.0, max=1.0)
            total_energy = total_energy + potential.energy_scale * energy

            guidances.append(guidance)

        # Accumulate guidance fields and mask to valid residues
        guidance = sum(guidances, start=PotentialField())
        guidance.mask((batch[bp.res_mask] > 0) & (batch[bp.diffuse_mask] > 0))

        return total_energy, guidance


@dataclass
class FKSteeringResampler:
    """
    Stateful Feynman-Kac Steering Sequential Monte Carlo driver.

    If enabled, creates K particles per sample in the batch,
    and manages energy calculation and resampling during interpolant sampling.
    """

    cfg: InterpolantSteeringConfig
    # Optional override for number of particles; if None, cfg.num_particles is used
    num_particles: Optional[int] = None

    def __post_init__(self):
        self.calculator = FKSteeringCalculator(cfg=self.cfg)
        self._log = logging.getLogger(__name__)

        if self.num_particles is None:
            self.num_particles = self.cfg.num_particles

        # track energy trajectory for each particle
        self._energy_trajectory: Optional[torch.Tensor] = None  # (B * K, 0->T)
        # track accumulated log G
        self._log_G: Optional[torch.Tensor] = None  # (B * K,)
        # track log G delta (difference at previous step)
        self._log_G_delta: Optional[torch.Tensor] = None  # (B * K,)

        # cache last computed guidance to reuse between resampling intervals
        self._cached_guidance: Optional[PotentialField] = None

    @property
    def enabled(self) -> bool:
        return self.num_particles > 1

    def log_G_score(self) -> torch.Tensor:
        """
        log G score is a weighted sum of the absolute energy and the difference from previous step
        expects `self._log_G` and `self._log_G_delta` to be set to prev/accumulated values.
        """
        return (self.cfg.energy_weight_difference * self._log_G_delta) + (
            self.cfg.energy_weight_absolute * self._log_G
        )

    def init_particles(self, batch: NoisyFeatures) -> NoisyFeatures:
        if not self.enabled:
            return batch

        self._log.debug(f"Init FK Steering with {self.num_particles} particles")

        num_batch, num_res = batch[bp.res_mask].shape  # (B, N)
        device = batch[bp.res_mask].device

        self._energy_trajectory = torch.empty(
            num_batch * self.num_particles, 0, device=device
        )
        self._log_G = torch.zeros(num_batch * self.num_particles, device=device)
        self._log_G_delta = torch.zeros(num_batch * self.num_particles, device=device)

        batch = {k: _batch_repeat_feat(v, self.num_particles) for k, v in batch.items()}

        return batch

    def on_step(
        self,
        step_idx: int,
        batch: NoisyFeatures,
        protein_state: SamplingStep,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
    ) -> Tuple[
        NoisyFeatures,  # batch with selected particles, if enabled
        Optional[torch.Tensor],  # reindexing, if enabled
        FKStepMetric,  # metrics for this step
        PotentialField,  # guidance for this step
    ]:
        """
        If enabled, resample particles in a batch according to their energy.
        Updates the batch's particles and returns the resampled indices.

        Batch is updated to new particles, but caller must update out of scope
        predictions / state with the resampled indices.

        We output sample indices and metrics if this is a resampling step.

        However, guidance potential field may be output on all steps, if enabled.
        Guidance is cached and reused (with optional decay) on non-resampling steps.
        """
        # escape if disabled
        if not self.enabled:
            return batch, None, None, PotentialField()

        # If not resampling at this step, return decayed guidance
        if (step_idx % self.cfg.resampling_interval) != 0:
            if self._cached_guidance is None:
                return batch, None, None, PotentialField()

            steps_since_resampling = step_idx % self.cfg.resampling_interval
            decay = self.cfg.guidance_cache_decay**steps_since_resampling
            decayed_guidance = self._cached_guidance.decayed(decay)
            return batch, None, None, decayed_guidance

        assert self._energy_trajectory is not None, "FK Steering not initialized."

        batch_size, num_res = batch[bp.res_mask].shape  # (B * K, N)
        num_batch = batch_size // self.num_particles  # B (original batch size)
        device = batch[bp.res_mask].device

        # calculate and track energy
        energy, guidance = self.calculator.compute(
            batch=batch,
            protein_state=protein_state,
            model_pred=model_pred,
            protein_pred=protein_pred,
        )  # (B * K,)
        self._energy_trajectory = torch.cat(
            [self._energy_trajectory, energy.unsqueeze(1)], dim=1
        )

        # cache guidance for use on non-resampling steps
        self._cached_guidance = guidance

        # Compute log G values, weighing particles by improvement
        # Calculate both the running total, and difference from previous step.
        if step_idx == 0:
            # step 0: initialize to -energy (no difference yet)
            log_G_delta = -1 * energy
        else:
            # step > 0: log_G is the difference between the previous and current energy
            log_G_delta = (
                self._energy_trajectory[:, -2] - self._energy_trajectory[:, -1]
            )
        # scale log_G by lambda
        log_G_delta *= self.cfg.fk_lambda
        # cache log G delta
        self._log_G_delta = log_G_delta
        # accumulate log G values
        self._log_G += log_G_delta

        # calculate log G score
        log_G_score = self.log_G_score()
        logG_mat = log_G_score.view(num_batch, self.num_particles)  # (B, K)

        # softmax-normalize resampling weights per sample
        # temperature scale log G; <1 = sharpen, 1 = no effect.
        # TODO(cfg) fksteering temp scale
        tau = 0.5
        logG_mat = logG_mat / tau
        resampling_weights = (logG_mat - logG_mat.max(dim=1, keepdim=True).values).exp()
        resampling_weights = resampling_weights / resampling_weights.sum(
            dim=1, keepdim=True
        )

        # track metric effective sample size as diagnostic of weight degeneracy
        effective_sample_size = 1.0 / (resampling_weights**2).sum(dim=1)

        # draw surviving particle indices
        row_idx = [
            torch.multinomial(w_i, self.num_particles, replacement=True)
            for w_i in resampling_weights
        ]
        idx = (
            torch.stack(row_idx)
            + torch.arange(num_batch, device=device).unsqueeze(1) * self.num_particles
        ).flatten()

        step_metric = FKStepMetric(
            step=step_idx,
            energy=energy.tolist(),
            log_G=log_G_score.tolist(),
            log_G_delta=log_G_delta.tolist(),
            weights=resampling_weights.tolist(),
            effective_sample_size=effective_sample_size.mean().item(),
            keep=idx.tolist(),
            guidance=guidance,
        )
        self._log.debug(step_metric.log())

        # reindex batch + internal state using the sampled indices
        batch = {k: _batch_select_feat(v, idx) for k, v in batch.items()}
        self._energy_trajectory = self._energy_trajectory.index_select(0, idx)
        self._log_G = self._log_G.index_select(0, idx)
        self._log_G_delta = self._log_G_delta.index_select(0, idx)

        return batch, idx, step_metric, guidance

    def best_particle_in_batch(
        self, batch: NoisyFeatures
    ) -> Tuple[NoisyFeatures, Optional[torch.Tensor]]:
        """
        Pick the best particle per sample in the batch by log G score.
        Converts batch from (B * K, ...) to (B, ...), and returns indices of the best particles.
        """
        if not self.enabled:
            return batch, None

        batch_size, num_res = batch[bp.res_mask].shape  # (B * K, N)
        num_batch = batch_size // self.num_particles  # B (original batch size)
        device = batch[bp.res_mask].device

        log_G_score = self.log_G_score()
        logG_mat = log_G_score.view(num_batch, self.num_particles)  # (B, K)
        idx_best = logG_mat.argmax(dim=1)  # (B,)
        idx_best += torch.arange(num_batch, device=device) * self.num_particles  # (B,)
        batch_best = {k: _batch_select_feat(v, idx_best) for k, v in batch.items()}

        return batch_best, idx_best
