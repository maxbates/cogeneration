import math
from dataclasses import dataclass
from typing import Optional

import torch

from cogeneration.data import so3_utils
from cogeneration.data.noise_mask import uniform_so3
from varco.config import VarcoInterpolantRotationCouplerConfig
from varco.coupling import Coupler, Coupling
from varco.tree_plan import BatchedTreePlan


@dataclass
class RotationCoupling(Coupling):
    """Coupling for SO(3) rotations using geodesic bridge with IGSO3 noise."""

    pass


class RotationCoupler(Coupler[RotationCoupling]):
    """
    Coupler for SO(3) rotations using geodesic interpolation with IGSO3 noise.

    Implements stochastic bridges on SO(3) using the IGSO3 distribution for
    intermediate noise. Supports geodesic interpolation between rotation matrices.
    """

    def __init__(self, cfg: VarcoInterpolantRotationCouplerConfig):
        self.cfg = cfg
        self._igso3: Optional[so3_utils.SampleIGSO3] = None
        self._device: torch.device = torch.device("cpu")

    def set_device(self, device: torch.device):
        """Set device for IGSO3 sampler. Must be called before using."""
        self._device = device
        # Move IGSO3 to GPU for CUDA, but keep on CPU for MPS (VonMises issues)
        if self._igso3 is not None and device.type != "mps":
            self._igso3.to(device)

    @property
    def igso3(self) -> so3_utils.SampleIGSO3:
        """Lazy initialization of IGSO3 sampler."""
        if self._igso3 is None:
            steps = 1000
            sigma_grid = torch.logspace(
                math.log10(self.cfg.igso3_sigma_min),
                math.log10(self.cfg.igso3_sigma_max),
                steps=steps,
                dtype=torch.float64,
            )
            self._igso3 = so3_utils.SampleIGSO3(steps, sigma_grid, cache_dir=".cache")
            if self._device is not None and self._device.type != "mps":
                self._igso3.to(self._device)
            self._igso3 = self._igso3.float()
        return self._igso3

    def _ensure_igso3_device(self, target_device: torch.device):
        """Ensure IGSO3 is on correct device before sampling."""
        igso3_device = self.igso3.sigma_grid.device
        if target_device.type == "mps":
            # IGSO3 stays on CPU for MPS, results moved after
            return
        if igso3_device != target_device:
            self._igso3.to(target_device)

    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool
        x1: torch.Tensor,  # (B, N, 3, 3) rotation matrices
        device: torch.device,
    ) -> torch.Tensor:
        """Sample uniform SO(3) rotations for scaffolds, keep motif rotations."""
        B, N = motif_mask.shape
        # Sample uniform SO(3) for all positions
        return uniform_so3(B, N, device=device)

    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, 3, 3)
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Weighted geodesic average of two child rotation anchors."""
        # Normalize weights
        wsum = child_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = child_weights.float() / wsum.float()  # (N_valid, 2)

        # For two rotation matrices R0 and R1 with weights w0 and w1:
        # Use geodesic interpolation: geodesic_t(w1, R1, R0)
        # where w1 is the weight of the second child (first child has weight w0 = 1 - w1)
        R0 = child_anchors[:, 0]  # (N_valid, 3, 3)
        R1 = child_anchors[:, 1]  # (N_valid, 3, 3)
        # w1 needs shape (N_valid, 1) to broadcast with rot_vf (N_valid, 3) -> (N_valid, 3)
        w1 = weights[:, 1].unsqueeze(-1)  # (N_valid, 1)

        # Geodesic from R0 toward R1, parameterized by w1
        return so3_utils.geodesic_t(t=w1, mat=R1, base_mat=R0)  # (N_valid, 3, 3)

    def sample_bridge(
        self,
        x_start: torch.Tensor,  # (..., 3, 3)
        x_end: torch.Tensor,  # (..., 3, 3)
        s: torch.Tensor,  # (...) current time
        t0: torch.Tensor,  # (...) birth time
    ) -> torch.Tensor:
        """Sample SO(3) bridge marginal at time s from (x_start at t0) to (x_end at 1)."""
        original_shape = x_start.shape  # (..., 3, 3)

        # Flatten to (N, 3, 3) for processing
        flat_start = x_start.reshape(-1, 3, 3)  # (N, 3, 3)
        flat_end = x_end.reshape(-1, 3, 3)  # (N, 3, 3)
        flat_s = s.reshape(-1)  # (N,)
        flat_t0 = t0.reshape(-1)  # (N,)
        N = flat_start.shape[0]

        # Compute interpolation parameter u in [0, 1]
        denom = (1.0 - flat_t0).clamp_min(1e-6)
        u = ((flat_s - flat_t0) / denom).clamp(0.0, 1.0)  # (N,)

        # Deterministic geodesic interpolation
        # u needs shape (N, 1) to broadcast with rot_vf (N, 3) -> (N, 3)
        mean = so3_utils.geodesic_t(
            t=u.unsqueeze(-1),  # (N, 1)
            mat=flat_end,
            base_mat=flat_start,
        )  # (N, 3, 3)

        if self.cfg.noise_scale == 0.0:
            return mean.view(original_shape)

        # Stochastic bridge: add IGSO3 noise scaled by bridge variance
        # Variance for Brownian bridge: (s - t0)(1 - s) / (1 - t0)
        var = (
            (flat_s - flat_t0).clamp_min(0.0) * (1.0 - flat_s).clamp_min(0.0) / denom
        ).clamp_min(
            0.0
        )  # (N,)

        # Scale variance by cfg.sigma and compute std for IGSO3
        std = (var.sqrt() * self.cfg.noise_scale).clamp_min(1e-6)  # (N,)

        # Sample IGSO3 noise and apply
        self._ensure_igso3_device(mean.device)
        sigma_for_igso3 = std.clamp(self.cfg.igso3_sigma_min, self.cfg.igso3_sigma_max)

        # Only sample noise where std > min threshold
        apply_mask = std > self.cfg.igso3_sigma_min
        if apply_mask.any():
            identity_noise = (
                torch.eye(3, device=mean.device).unsqueeze(0).expand(N, -1, -1)
            )
            sigma_sel = sigma_for_igso3[apply_mask]
            sigma_sel = sigma_sel.to(self.igso3.sigma_grid.device)
            noise_sel = self.igso3.sample(sigma_sel, 1).to(mean.device)
            noise_sel = noise_sel.squeeze(1)  # (N_apply, 3, 3)
            intermediate_noise = identity_noise.clone()
            intermediate_noise[apply_mask] = noise_sel
            mean = torch.einsum("...ij,...jk->...ik", mean, intermediate_noise)

        return mean.view(original_shape)

    def post_process(
        self,
        x_t: torch.Tensor,  # (B, P, 3, 3)
        present_mask: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        anchors: torch.Tensor,  # (B, P, 3, 3)
    ) -> torch.Tensor:
        """Set non-present nodes to identity rotation."""
        identity = torch.eye(3, device=x_t.device, dtype=x_t.dtype)
        return torch.where(
            present_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
            x_t,
            identity.unsqueeze(0).unsqueeze(0).expand_as(x_t),
        )

    def _make_coupling(
        self,
        tree: BatchedTreePlan,
        anchors: torch.Tensor,  # (B, A, 3, 3)
        creation_state: torch.Tensor,  # (B, A, 3, 3)
    ) -> RotationCoupling:
        return RotationCoupling(
            tree=tree,
            anchors=anchors,
            creation_state=creation_state,
        )

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, 3, 3)
        x1_pred: torch.Tensor,  # (B, P, 3, 3)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[
            torch.Tensor
        ] = None,  # (B, P, 3) rotation tangent vector field
    ) -> torch.Tensor:
        """
        Single Euler step for rotation sampling using geodesic flow.

        Uses calc_rot_vf to compute vector field and geodesic_t to step.
        Optionally adds IGSO3 noise for stochastic sampling.
        """
        if x_t.shape != x1_pred.shape:
            raise ValueError(
                f"x_t and x1_pred must match shape; got {tuple(x_t.shape)} vs {tuple(x1_pred.shape)}"
            )
        if x_t.ndim != 4 or x_t.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected x_t to have shape (B, P, 3, 3); got {tuple(x_t.shape)}"
            )
        if potential is not None and potential.shape != (*x_t.shape[:2], 3):
            raise ValueError(
                f"potential must have shape (B, P, 3); got {tuple(potential.shape)}"
            )

        B, P = x_t.shape[:2]
        device = x_t.device

        # Mask for valid (born) positions
        valid_mask = (
            (birth_time <= t[:, None]).unsqueeze(-1).unsqueeze(-1)
        )  # (B, P, 1, 1)

        # VF scaling: 1 / (1 - t), clamped
        # Using exponential schedule if configured
        if self.cfg.exp_rate > 0:
            r = self.cfg.exp_rate
            denom = 1.0 - torch.exp(-r * (1.0 - t))
            scaling = (r / denom.clamp_min(1e-8)).clamp_min(1e-4)
        else:
            scaling = (1.0 / (1.0 - t).clamp_min(1e-4)).clamp_min(1e-4)

        # Compute rotation vector field: Log_{x_t}(x1_pred)
        rot_vf = so3_utils.calc_rot_vf(mat_t=x_t, mat_1=x1_pred)  # (B, P, 3)

        # Add guidance potential if provided
        if potential is not None:
            rot_vf = rot_vf + potential

        # Clamp per-residue rotation step.
        drift_step_cap = float(self.cfg.drift_step_cap_rad)
        if drift_step_cap > 0.0:
            rot_step = rot_vf * (scaling.view(B, 1, 1) * float(dt))
            step_norm = rot_step.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rot_step = rot_step * (drift_step_cap / step_norm).clamp(max=1.0)
            rot_vf = rot_step / (scaling.view(B, 1, 1) * float(dt) + 1e-8)

        # Geodesic step: geodesic_t(scaling * dt, x1_pred, x_t, rot_vf)
        geodesic_time = (scaling * dt)[:, None, None]  # (B, 1, 1)
        x_next = so3_utils.geodesic_t(
            t=geodesic_time,
            mat=x1_pred,
            base_mat=x_t,
            rot_vf=rot_vf,
        )  # (B, P, 3, 3)

        # Optionally add IGSO3 noise for stochastic sampling
        if float(self.cfg.noise_scale) > 0.0 and float(dt) > 0.0:
            self._ensure_igso3_device(device)

            # Compute sigma_t scaled by sqrt(dt)
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=torch.full_like(t, float(self.cfg.noise_scale)),
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            ).clamp_max(float(self.cfg.igso3_sigma_max))
            sqrt_dt = math.sqrt(float(dt))
            sigma_t = (sigma_t * sqrt_dt).to(self.igso3.sigma_grid.device)

            apply_mask = sigma_t > self.cfg.igso3_sigma_min
            if apply_mask.any():
                sigma_sel = sigma_t[apply_mask]
                identity_noise = (
                    torch.eye(3, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, P, -1, -1)
                )
                noise_sel = self.igso3.sample(sigma_sel, P).to(device)
                noise_sel = noise_sel.reshape(apply_mask.sum(), P, 3, 3)
                intermediate_noise = identity_noise.clone()
                intermediate_noise[apply_mask] = noise_sel
                x_next = torch.einsum("...ij,...jk->...ik", x_next, intermediate_noise)

        # Keep unborn positions unchanged
        x_next = torch.where(valid_mask.expand(-1, -1, 3, 3), x_next, x_t)

        return x_next
