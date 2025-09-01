import math
from typing import Optional

import torch

from cogeneration.config.base import InterpolantTorsionsConfig
from cogeneration.data.fm.flow_matcher import FlowMatcher
from cogeneration.data.noise_mask import (
    angles_noise,
    fill_torsions,
    mask_blend_3d,
    torsions_noise,
)


class FlowMatcherTorsions(FlowMatcher):
    """
    Flow matcher for torsion angles.

    Uses Von Mises noise for angles and interpolates in angle space.
    Euler updates are performed on angles, then converted back to (sin, cos).
    """

    def __init__(self, cfg: InterpolantTorsionsConfig):
        self.cfg = cfg
        self._device: Optional[torch.device] = None

    def set_device(self, device: torch.device):
        self._device = device

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        """Generate t=0 torsion angles noise (B, N, 7, 2)."""
        num_batch, num_res = res_mask.shape
        return torsions_noise(
            sigma=torch.ones((num_batch,), device=self._device),
            num_samples=num_res,
            num_angles=7,
        )

    def corrupt(
        self,
        torsions_1: torch.Tensor,  # (B, N, 7, 2)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Corrupt torsions from t=1 to t using noise.
        """
        num_batch, num_res = res_mask.shape
        torsions_0 = self.sample_base(res_mask=res_mask)

        # interpolate in angle space using linear schedule
        angles_1 = torch.atan2(torsions_1[..., 0], torsions_1[..., 1])  # (B, N, 7)
        angles_0 = torch.atan2(torsions_0[..., 0], torsions_0[..., 1])  # (B, N, 7)
        t_broadcast = t.view(num_batch, 1, 1)  # (B, 1, 1)
        angles_t = (1.0 - t_broadcast) * angles_0 + t_broadcast * angles_1

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            sigma_t = self._compute_sigma_t(
                t,  # (B,)
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            angles_t += angles_noise(sigma=sigma_t, num_samples=num_res, num_angles=7)

        # wrap to keep angles in (-π,π]
        angles_t = (angles_t + math.pi) % (2.0 * math.pi) - math.pi

        # angles -> (sin, cos)
        torsions_t = torch.stack((torch.sin(angles_t), torch.cos(angles_t)), dim=-1)

        # Fix non-diffused residues to t=1
        torsions_t = mask_blend_3d(torsions_t, torsions_1, diffuse_mask)

        return torsions_t * res_mask[..., None, None]

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        torsions_1: torch.Tensor,  # (B, N, 7, 2)
        torsions_t: torch.Tensor,  # (B, N, K, 2)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:  # (B, N, 7, 2)
        """
        Perform an Euler step in angle space to update torsion angles.

        `K` can be 1 or 7, depending on the number of torsions predicted by the model.
        However, torsions_1 is always 7, and torsions_t is filled to 7 if K < 7.
        """
        B, N, K = torsions_1.shape[:3]

        # (B, N, K, 2) -> (B, N, 7, 2)
        torsions_t = fill_torsions(
            shape=torsions_1.shape,
            torsions=torsions_t,
            device=torsions_1.device,
        )

        angles_1 = torch.atan2(torsions_1[..., 0], torsions_1[..., 1])  # (B, N, 7)
        angles_t = torch.atan2(torsions_t[..., 0], torsions_t[..., 1])  # (B, N, 7)

        t = t.to(self._device)
        d_t = d_t.to(self._device)
        # Broadcast per-batch t to (B, 1, 1) for (B, N, K) angles tensors; d_t is scalar
        angles_vf = (angles_1 - angles_t) / (1.0 - t.view(-1, 1, 1))
        angles_next = angles_t + angles_vf * d_t

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # Brownian increment: sigma_t * sqrt(dt)
            sigma_t = self._compute_sigma_t(
                t,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            sqrt_dt = torch.sqrt(d_t.to(sigma_t.device))
            sigma_t = sigma_t * sqrt_dt
            angles_next += angles_noise(
                sigma=sigma_t,
                num_samples=angles_next.shape[1],
                num_angles=angles_next.shape[2],
            )

        # wrap to keep angles in (-π,π]
        angles_next = (angles_next + math.pi) % (2.0 * math.pi) - math.pi

        # (sin, cos)
        torsions_next = torch.stack(
            (torch.sin(angles_next), torch.cos(angles_next)), dim=-1
        )
        return torsions_next  # (B, N, 7, 2)
