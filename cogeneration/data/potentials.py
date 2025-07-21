import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cogeneration.config.base import InterpolantSteeringConfig
from cogeneration.data.trajectory import SamplingStep
from cogeneration.dataset.interaction import DIST_INTERACTION_BACKBONE
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyFeatures

# TODO(fksteering) reconsider what states to provide to compute_energy().
#   model output for sure...
#   Current protein state vs. modified protein prediction...

# TODO(fksteering) additional potentials, e.g.:
#   - van der waals clashes and interactions
#   - run ProteinMPNN on to predict the sequence to bias towards designable sequences


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


@dataclass
class Potential(ABC):
    """Base class for Feynman-Kac steering potentials.

    Potentials compute zero-bottomed energy (rather than rewards) for each member of a batch.
    Energy should be scaled to be in the range [0, 1], where 0 is the best score.

    Each potential receives the clean model prediction `model_pred`,
    and the modified prediction `protein_pred`, and can calculate using the appropriate one.

    Fow now, does not support computing gradients for guidance.
    """

    scale: float = 1.0  # compute_energy() * scale is the final energy

    @abstractmethod
    def compute_energy(
        self,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> torch.Tensor:  # (B,)
        """Return the energy per predicted sample in the batch, scaled to [0, 1]."""


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
    effective_sample_size: float
    keep: List[int]

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
            FKSteeringTrajectory containing only metrics for the specified batch member's particles
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

    def compute_energy(
        self,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> torch.Tensor:
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
        return energy


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

    def compute_energy(
        self,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> torch.Tensor:
        num_batch, num_res = batch[bp.res_idx].size()  # (B, N)
        device = batch[bp.res_idx].device

        # Skip if no hot spots marked
        if bp.hot_spots not in batch:
            return torch.zeros(num_batch, device=device)

        hot_spots_mask = batch[bp.hot_spots].bool()  # (B, N) - convert to bool
        hot_spots_present = hot_spots_mask.sum(-1) > 0  # (B,)
        if not hot_spots_present.any():
            return torch.zeros(num_batch, device=device)

        # Only apply to multimers
        chain_idx = batch[bp.chain_idx]  # (B, N)
        multi_chain_mask = chain_idx.max(-1).values != chain_idx.min(-1).values  # (B,)
        if not multi_chain_mask.any():
            return torch.zeros(num_batch, device=device)

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

        return energy


@dataclass
class ContactConditioningPotential(Potential):
    """
    Potential that encourages adherence to contact conditioning constraints.
    Uses ReLU penalty for distances that exceed the defined contact thresholds.
    Normalizes energy over the number of defined contacts.
    Optionally ignores contacts in motifs and upweights inter-chain contacts.
    """

    tolerance_dist: float = 0.5  # tolerance distance for contact conditioning
    inter_chain_weight: float = 2.0  # upweight factor for inter-chain contacts
    max_distance_penalty: float = 10.0  # maximum distance penalty

    def compute_energy(
        self,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> torch.Tensor:
        num_batch, num_res = batch[bp.res_idx].size()  # (B, N)
        device = batch[bp.res_idx].device

        if bp.contact_conditioning not in batch:
            return torch.zeros(num_batch, device=device)

        contact_conditioning = batch[bp.contact_conditioning]  # (B, N, N)
        contact_mask = contact_conditioning > 0  # (B, N, N)

        if not contact_mask.any():
            return torch.zeros(num_batch, device=device)

        # pairwse distances
        pred_trans = protein_pred.trans  # (B, N, 3)
        pairwise_dists = torch.norm(
            pred_trans.unsqueeze(2) - pred_trans.unsqueeze(1),
            dim=-1,
        )  # (B, N, N)

        # clamped ReLU penalty
        shortfall = torch.relu(
            contact_conditioning - pairwise_dists - self.tolerance_dist
        )
        excess = torch.relu(pairwise_dists - contact_conditioning - self.tolerance_dist)
        contact_penalties = excess + shortfall
        contact_penalties = contact_penalties.clamp(max=self.max_distance_penalty)

        # Only for defined contacts
        contact_penalties = contact_penalties * contact_mask.float()  # (B, N, N)

        # Ignore contacts in motifs, if defined
        motif_mask = batch.get(bp.motif_mask, None)
        if motif_mask is not None:
            # Create motif edge mask: both residues must NOT be in motifs
            motif_edge_mask = (~motif_mask.bool()).unsqueeze(2) & (
                ~motif_mask.bool()
            ).unsqueeze(1)
            contact_penalties = contact_penalties * motif_edge_mask.float()

        # reweight inter-chain contacts if specified
        if self.inter_chain_weight != 1.0:
            chain_idx = batch[bp.chain_idx]  # (B, N)
            chain_i = chain_idx.unsqueeze(2)  # (B, N, 1)
            chain_j = chain_idx.unsqueeze(1)  # (B, 1, N)
            inter_chain_mask = chain_i != chain_j  # (B, N, N)

            contact_weights = torch.ones_like(contact_penalties)
            contact_weights = torch.where(
                inter_chain_mask & contact_mask,
                self.inter_chain_weight,
                contact_weights,
            )
            contact_penalties = contact_penalties * contact_weights

        # Sum penalties and normalize by number of defined contacts
        penalty_per_batch = contact_penalties.sum(dim=(1, 2))  # (B,)
        num_contacts = contact_mask.float().sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
        penalty_per_batch = penalty_per_batch / num_contacts  # (B,)

        # Normalize energy to [0, 1] range
        energy = penalty_per_batch / self.max_distance_penalty
        energy = energy.clamp(min=0.0, max=1.0)

        return energy


@dataclass
class FKSteeringCalculator:
    """
    Helper to implement Feynman-Kac steering potentials for inference.

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
                scale=cfg.chain_break_scale,
                allowed_backbone_dist=cfg.chain_break_allowed_backbone_dist,
                maximum_backbone_dist=cfg.chain_break_maximum_backbone_dist,
            ),
            HotSpotPotential(
                scale=cfg.hot_spot_scale,
                contact_distance_ang=cfg.hot_spot_contact_distance_ang,
                max_distance_penalty=cfg.hot_spot_max_distance_penalty,
            ),
            ContactConditioningPotential(
                scale=cfg.contact_conditioning_scale,
                inter_chain_weight=cfg.contact_conditioning_inter_chain_weight,
                max_distance_penalty=cfg.contact_conditioning_max_distance_penalty,
            ),
        ]

    def compute_energy(
        self,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> torch.Tensor:  # (B,)
        """Weighted sum of all configured potentials."""
        num_batch, num_res = batch[bp.res_idx].size()  # (B, N)
        device = batch[bp.res_idx].device

        total = torch.zeros(num_batch, device=device)
        for potential in self.potentials:
            if potential.scale == 0.0:
                continue
            energy = potential.compute_energy(
                batch=batch,
                model_pred=model_pred,
                protein_pred=protein_pred,
                protein_state=protein_state,
            )
            energy = energy.clamp(min=0.0, max=1.0)
            total = total + potential.scale * energy

        return total


@dataclass
class FKSteeringResampler:
    """
    Stateful Feynman-Kac Steering Sequential Monte Carlo driver.

    If enabled, creates K particles per sample in the batch,
    and manages energy calculation and resampling during interpolant sampling.
    """

    cfg: InterpolantSteeringConfig

    def __post_init__(self):
        self.calculator = FKSteeringCalculator(cfg=self.cfg)
        self._log = logging.getLogger(__name__)

        # track energy trajectory for each particle
        self._energy_trajectory: Optional[torch.Tensor] = None  # (B * K, 0->T)
        # track accumulated log G
        self._log_G: Optional[torch.Tensor] = None  # (B * K,)
        # track log G delta (difference at previous step)
        self._log_G_delta: Optional[torch.Tensor] = None  # (B * K,)

    @property
    def enabled(self) -> bool:
        return self.cfg.num_particles > 1

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

        self._log.debug(f"Init FK Steering with {self.cfg.num_particles} particles")

        num_batch, num_res = batch[bp.res_mask].shape  # (B, N)
        num_particles = self.cfg.num_particles  # K
        device = batch[bp.res_mask].device

        self._energy_trajectory = torch.empty(
            num_batch * num_particles, 0, device=device
        )
        self._log_G = torch.zeros(num_batch * num_particles, device=device)
        self._log_G_delta = torch.zeros(num_batch * num_particles, device=device)

        batch = {k: _batch_repeat_feat(v, num_particles) for k, v in batch.items()}

        return batch

    def on_step(
        self,
        step_idx: int,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> Tuple[NoisyFeatures, Optional[torch.Tensor], FKStepMetric]:
        """
        If enabled, resample particles in a batch according to their energy.
        Updates the batch's particles and returns the resampled indices.

        Batch is updated to new particles, but caller must update out of scope
        predictions / state with the resampled indices.
        """
        # escape if disabled or not resampling on this step
        if not self.enabled:
            return batch, None, None
        if (step_idx % self.cfg.resampling_interval) != 0:
            return batch, None, None

        assert self._energy_trajectory is not None, "FK Steering not initialized."

        batch_size, num_res = batch[bp.res_mask].shape  # (B * K, N)
        num_particles = self.cfg.num_particles  # K
        num_batch = batch_size // num_particles  # B (original batch size)
        device = batch[bp.res_mask].device

        # calculate and track energy
        energy = self.calculator.compute_energy(
            batch=batch,
            model_pred=model_pred,
            protein_pred=protein_pred,
            protein_state=protein_state,
        )  # (B * K,)
        self._energy_trajectory = torch.cat(
            [self._energy_trajectory, energy.unsqueeze(1)], dim=1
        )

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
        logG_mat = log_G_score.view(num_batch, num_particles)  # (B, K)

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
        # TODO(metrics) expose ESS and energy trajectory
        effective_sample_size = 1.0 / (resampling_weights**2).sum(dim=1)

        # draw surviving particle indices
        row_idx = [
            torch.multinomial(w_i, num_particles, replacement=True)
            for w_i in resampling_weights
        ]
        idx = (
            torch.stack(row_idx)
            + torch.arange(num_batch, device=device).unsqueeze(1) * num_particles
        ).flatten()

        step_metric = FKStepMetric(
            step=step_idx,
            energy=energy.tolist(),
            log_G=log_G_score.tolist(),
            log_G_delta=log_G_delta.tolist(),
            weights=resampling_weights.tolist(),
            effective_sample_size=effective_sample_size.mean().item(),
            keep=idx.tolist(),
        )
        self._log.debug(step_metric.log())

        # reindex batch + internal state using the sampled indices
        batch = {k: _batch_select_feat(v, idx) for k, v in batch.items()}
        self._energy_trajectory = self._energy_trajectory.index_select(0, idx)
        self._log_G = self._log_G.index_select(0, idx)
        self._log_G_delta = self._log_G_delta.index_select(0, idx)

        return batch, idx, step_metric

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
        num_particles = self.cfg.num_particles  # K
        num_batch = batch_size // num_particles  # B (original batch size)
        device = batch[bp.res_mask].device

        log_G_score = self.log_G_score()
        logG_mat = log_G_score.view(num_batch, num_particles)  # (B, K)
        idx_best = logG_mat.argmax(dim=1)  # (B,)
        idx_best += torch.arange(num_batch, device=device) * num_particles  # (B,)
        batch_best = {k: _batch_select_feat(v, idx_best) for k, v in batch.items()}

        return batch_best, idx_best
