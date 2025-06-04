from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from cogeneration.config.base import InterpolantSteeringConfig
from cogeneration.data.trajectory import SamplingStep
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyFeatures

# TODO reconsider what states to provide to compute_energy().
#   model output for sure...
#   Current protein state vs. modified protein prediction...

# TODO additional potentials, e.g.:
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

    scale: float = 1.0  # energy * scale is the final energy

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
class ChainBreakPotential(Potential):
    """
    Penalize unnatural backbone chain breaks by checking residue Euclidean distances,
    but accounting for true chain breaks or non-sequential residues.
    """

    max_backbone_dist: float = 4.0

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

        # continuity masks
        same_chain = chain_idx[:, 1:].eq(chain_idx[:, :-1])
        consecutive = res_idx[:, 1:].eq(res_idx[:, :-1] + 1)
        need_continuity = same_chain & consecutive  # (B, N-1)

        # dists from one residue to the next
        dists = torch.linalg.norm(trans[:, :-1, :] - trans[:, 1:, :], dim=-1)
        breaks = need_continuity & (dists > self.max_backbone_dist)
        num_breaks = breaks.sum(dim=1).float()  # (B,)

        # logistic ramp, so even 1 break is penalized heavily
        energy = 1.0 - torch.exp(-1 * num_breaks)
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
                max_backbone_dist=cfg.chain_break_max_backbone_dist,
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

        # track energy trajectory for each particle
        self._energy_trajectory: Optional[torch.Tensor] = None

        # track running total log G
        self._log_G: Optional[torch.Tensor] = None

    @property
    def enabled(self) -> bool:
        return self.cfg.num_particles > 1

    def init_particles(self, batch: NoisyFeatures) -> NoisyFeatures:
        if not self.enabled:
            return batch

        num_batch, num_res = batch[bp.res_mask].shape  # (B, N)
        num_particles = self.cfg.num_particles  # K
        device = batch[bp.res_mask].device

        self._energy_trajectory = torch.empty(
            num_batch * num_particles, 0, device=device
        )  # (B * K, 0)
        self._log_G = torch.zeros(num_batch * num_particles, device=device)  # (B * K,)

        batch = {k: _batch_repeat_feat(v, num_particles) for k, v in batch.items()}

        return batch

    def on_step(
        self,
        step_idx: int,
        batch: NoisyFeatures,
        model_pred: SamplingStep,
        protein_pred: SamplingStep,
        protein_state: SamplingStep,
    ) -> Tuple[NoisyFeatures, Optional[torch.Tensor]]:
        """
        If enabled, resample particles in a batch according to their energy.
        Updates the batch's particles and returns the resampled indices.

        Batch is updated to new particles, but caller must update out of scope
        predictions / state with the resampled indices.
        """
        # escape if disabled or not resampling on this step
        if not self.enabled:
            return batch, None
        if (step_idx % self.cfg.resampling_interval) != 0:
            return batch, None

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
        # accumulate log G values
        self._log_G += log_G_delta

        # log G is a weighted sum of the absolute energy and the difference from previous step
        log_G_score = (self.cfg.energy_weight_difference * log_G_delta) + (
            self.cfg.energy_weight_absolute * self._log_G
        )

        # softmax-normalize resampling weights per sample
        logG_mat = log_G_score.view(num_batch, num_particles)  # (B, K)
        resampling_weights = (logG_mat - logG_mat.max(dim=1, keepdim=True).values).exp()
        resampling_weights = resampling_weights / resampling_weights.sum(
            dim=1, keepdim=True
        )

        # draw surviving particle indices
        row_idx = [
            torch.multinomial(w_i, num_particles, replacement=True)
            for w_i in resampling_weights
        ]
        idx = (
            torch.stack(row_idx)
            + torch.arange(num_batch, device=device).unsqueeze(1) * num_particles
        ).flatten()

        # reindex batch + internal state using the sampled indices
        batch = {k: _batch_select_feat(v, idx) for k, v in batch.items()}
        self._energy_trajectory = self._energy_trajectory.index_select(0, idx)
        self._log_G = self._log_G.index_select(0, idx)

        # track metric effective sample size as diagnostic of weight degeneracy
        # TODO expose ESS and energy trajectory
        effective_sample_size = 1.0 / (resampling_weights**2).sum(dim=1)

        return batch, idx

    def best_particle_in_batch(
        self, batch: NoisyFeatures
    ) -> Tuple[NoisyFeatures, Optional[torch.Tensor]]:
        """
        Pick the best particle per sample in the batch, by accumulated log G values.
        Converts batch from (B * K, ...) to (B, ...), and returns indices of the best particles.
        """
        if not self.enabled:
            return batch, None

        batch_size, num_res = batch[bp.res_mask].shape  # (B * K, N)
        num_particles = self.cfg.num_particles  # K
        num_batch = batch_size // num_particles  # B (original batch size)
        device = batch[bp.res_mask].device

        logG_mat = self._log_G.view(num_batch, num_particles)  # (B, K)
        idx_best = logG_mat.argmax(dim=1)  # (B,)
        idx_best += torch.arange(num_batch, device=device) * num_particles  # (B,)
        batch_best = {k: _batch_select_feat(v, idx_best) for k, v in batch.items()}

        return batch_best, idx_best
