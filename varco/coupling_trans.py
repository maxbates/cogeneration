import math
from dataclasses import dataclass
from typing import Optional

import torch

from cogeneration.data.const import NM_TO_ANG_SCALE
from cogeneration.data.noise_mask import centered_gaussian
from varco.config import VarcoInterpolantTransCouplerConfig
from varco.coupling import Coupler, Coupling
from varco.tree_plan import BatchedTreePlan


@dataclass
class TranslationCoupling(Coupling):
    """Coupling for translations using Brownian bridge."""

    pass


class TranslationCoupler(Coupler[TranslationCoupling]):
    def __init__(self, cfg: VarcoInterpolantTransCouplerConfig):
        self.cfg = cfg

    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool
        x1: torch.Tensor,  # (B, N, 3)
        device: torch.device,
    ) -> torch.Tensor:
        """Sample gaussian translations for all positions at t=0."""
        B, N = motif_mask.shape
        return centered_gaussian(B, N, n_bb_atoms=3, device=device) * NM_TO_ANG_SCALE

    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, 3)
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Weighted average of child translation anchors."""
        wsum = child_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = child_weights.float() / wsum.float()  # (N_valid, 2)
        return (child_anchors * weights.unsqueeze(-1)).sum(dim=1)  # (N_valid, 3)

    def sample_bridge(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        s: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Brownian-bridge marginal at time s from (x_start at t0) to (x_end at 1)."""
        original_shape = x_start.shape
        flat_start = x_start.view(-1, 3)
        flat_end = x_end.view(-1, 3)
        flat_s = s.reshape(-1)
        flat_t0 = t0.reshape(-1)

        denom = (1.0 - flat_t0).clamp_min(1e-6)
        u = ((flat_s - flat_t0) / denom).clamp(0.0, 1.0)
        mean = flat_start + u.unsqueeze(-1) * (flat_end - flat_start)

        if self.cfg.noise_scale == 0.0:
            return mean.view(original_shape)

        var = (
            (flat_s - flat_t0).clamp_min(0.0) * (1.0 - flat_s).clamp_min(0.0) / denom
        ).clamp_min(0.0)
        std = (var.sqrt() * self.cfg.noise_scale).to(mean.dtype)
        eps = torch.randn_like(mean)
        return (mean + std.unsqueeze(-1) * eps).view(original_shape)

    def post_process(
        self,
        x_t: torch.Tensor,
        present_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Zero out non-present nodes."""
        return torch.where(
            present_mask.unsqueeze(-1),
            x_t,
            torch.zeros_like(x_t),
        )

    def _make_coupling(
        self,
        anchors: torch.Tensor,  # (B, A, 3)
        tree: BatchedTreePlan,
        creation_state: torch.Tensor,  # (B, A, 3)
    ) -> TranslationCoupling:
        return TranslationCoupling(
            tree=tree,
            anchors=anchors,
            creation_state=creation_state,
        )

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, 3)
        x1_pred: torch.Tensor,  # (B, P, 3)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, 3) guidance VF
    ) -> torch.Tensor:  # (B, P, 3)
        """Euler step for translation updates (all in angstroms)."""
        trans_pred = x1_pred
        if x_t.shape != trans_pred.shape:
            raise ValueError(
                f"x_t and x1_pred must match shape; got {tuple(x_t.shape)} vs {tuple(trans_pred.shape)}"
            )
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(
                f"Expected x_t to have shape (B, P, 3); got {tuple(x_t.shape)}"
            )
        if potential is not None:
            if potential.shape != x_t.shape:
                raise ValueError(
                    f"potential and x_t must match shape; got {tuple(potential.shape)} vs {tuple(x_t.shape)}"
                )

        B, P, _ = x_t.shape
        device = x_t.device

        valid_fmask = (
            (birth_time <= t[:, None]).bool().float().unsqueeze(-1)
        )  # (B, P, 1)

        denom = (1.0 - t).clamp_min(1e-4).view(B, 1, 1)
        v = (trans_pred - x_t) / denom

        if potential is not None:
            v = v + potential

        # cap drift jump
        drift_step = v * float(dt)
        if float(self.cfg.drift_step_cap_ang) > 0.0:
            cap = float(self.cfg.drift_step_cap_ang)
            step_norm = drift_step.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            drift_step = drift_step * (cap / step_norm).clamp(max=1.0)

        x_next = x_t + drift_step

        # add brownian noise
        if float(self.cfg.noise_scale) > 0.0 and float(dt) > 0.0:
            scale = torch.full_like(t, float(self.cfg.noise_scale))
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=scale,
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            ).to(dtype=x_next.dtype, device=x_next.device)
            x_next = x_next + torch.randn_like(x_next) * (
                sigma_t.view(B, 1, 1) * math.sqrt(float(dt))
            )

        return x_next * valid_fmask + x_t * (1.0 - valid_fmask)
