import math
from dataclasses import dataclass
from typing import Optional, Union

import torch

from cogeneration.config.base import (
    InterpolantRotationsConfig,
    InterpolantRotationsScheduleEnum,
)
from cogeneration.data import so3_utils
from cogeneration.data.fm.flow_matcher import FlowMatcher
from cogeneration.data.noise_mask import mask_blend_3d, uniform_so3


@dataclass
class FlowMatcherRotations(FlowMatcher):
    """
    Flow matcher for rotations (SO(3)).

    Owns an IGSO(3) sampler table and implements corruption and Euler updates
    using geodesics on SO(3). Base sampling returns uniform SO(3) rotations.
    """

    cfg: InterpolantRotationsConfig

    _igso3: Optional[so3_utils.SampleIGSO3] = None

    def set_device(self, device: torch.device):
        super().set_device(device)

        # on CUDA, move to GPU, so can be broadcasted.
        # on MPS, leave on CPU, issues with VonMises.
        if self._igso3 is not None and device.type != "mps":
            self._igso3.to(device)

    @property
    def igso3(self):
        if self._igso3 is None:
            steps = 1000
            sigma_grid = torch.logspace(
                math.log10(self.cfg.igso3_sigma_min),
                math.log10(self.cfg.igso3_sigma),
                steps=steps,
                dtype=torch.float64,
            )
            self._igso3 = so3_utils.SampleIGSO3(steps, sigma_grid, cache_dir=".cache")
            if self._device is not None and self._device.type != "mps":
                self._igso3.to(self._device)
            self._igso3 = self._igso3.float()
        return self._igso3

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        """Generate t=0 SO(3) rotation matrices (B, N, 3, 3) from uniform SO(3)."""
        num_batch, num_res = res_mask.shape
        return uniform_so3(num_batch, num_res, device=self._device)

    def _so3_time(
        self, t: torch.Tensor, schedule: InterpolantRotationsScheduleEnum
    ) -> torch.Tensor:
        """Map t to SO(3) schedule time, using cfg's schedule"""
        if schedule == InterpolantRotationsScheduleEnum.linear:
            return t
        elif schedule == InterpolantRotationsScheduleEnum.exp:
            rate = self.cfg.exp_rate
            return (1 - torch.exp(-t * rate)) / (1 - math.exp(-rate))
        else:
            raise ValueError(f"Invalid schedule: {schedule}")

    def _vf_scaling(self, t: torch.Tensor) -> torch.Tensor:
        schedule = self.cfg.sample_schedule
        """Euler step scaling for vector field under sample schedule."""
        if schedule == InterpolantRotationsScheduleEnum.linear:
            return (1 / (1 - t)).clamp(min=1e-4)
        elif schedule == InterpolantRotationsScheduleEnum.exp:
            t = t.to(self._device)
            r = torch.tensor(
                self.cfg.exp_rate, dtype=torch.get_default_dtype(), device=t.device
            )
            denom = 1.0 - torch.exp(-r * (1.0 - t))
            denom = torch.clamp(denom, min=1e-8)
            scale = r / denom
            return scale.clamp(min=1e-4)
        else:
            raise ValueError(f"Unknown sample schedule {schedule}")

    def time_training(self, t: torch.Tensor) -> torch.Tensor:
        return self._so3_time(t, schedule=self.cfg.train_schedule)

    def time_sampling(self, t: torch.Tensor) -> torch.Tensor:
        return self._so3_time(t, schedule=self.cfg.sample_schedule)

    def corrupt(
        self,
        rotmats_1: torch.Tensor,  # (B, N, 3, 3)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
    ) -> torch.Tensor:
        """Corrupt rotations from t=1 to t using IGSO(3)."""
        num_batch, num_res = res_mask.shape
        tau = self.time_training(t)

        # sample IGSO(3) for t=0
        sigma = torch.tensor(
            [self.cfg.igso3_sigma], device=self.igso3.sigma_grid.device
        )
        noisy_rotmats = self.igso3.sample(sigma, num_batch * num_res).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)

        # compose with reference frames to get rotmats_0
        # applying noise as composition ensures noise is relative to reference frame + stay on SO(3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        # interpolate on geodesic between rotmats_0 and rotmats_1
        rotmats_t = so3_utils.geodesic_t(tau[:, None, None], rotmats_1, rotmats_0)

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        # stochastic intermediate noise
        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            sigma_t = self._compute_sigma_t(
                tau,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            # Only apply intermediate noise for samples with positive sigma
            # ensure exact t in {0,1} receive no noise, e.g. if domain's t is fixed
            apply_mask = sigma_t > 0
            if apply_mask.any():
                # Prepare identity noise for all, then overwrite selected batch items
                identity_noise = torch.eye(3, device=self._device)[None, None].repeat(
                    num_batch, num_res, 1, 1
                )
                # Sample IGSO3 only for the subset needing noise
                sigma_sel = sigma_t[apply_mask].to(self.igso3.sigma_grid.device)
                noise_sel = self.igso3.sample(sigma_sel, num_res).to(self._device)
                noise_sel = noise_sel.reshape(apply_mask.sum(), num_res, 3, 3)
                intermediate_noise = identity_noise
                intermediate_noise[apply_mask] = noise_sel
                rotmats_t = torch.einsum(
                    "...ij,...jk->...ik", rotmats_t, intermediate_noise
                )

        # set residues not in res_mask to identity
        identity = torch.eye(3, device=self._device)
        rotmats_t = mask_blend_3d(rotmats_t, identity[None, None], res_mask)

        # only corrupt residues in diffuse_mask
        return mask_blend_3d(rotmats_t, rotmats_1, diffuse_mask)

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        rotmats_1: torch.Tensor,  # (B, N, 3, 3)
        rotmats_t: torch.Tensor,  # (B, N, 3, 3)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, 3)
    ) -> torch.Tensor:
        # VF scaling applies schedule properly, dont give tau.
        scaling = self._vf_scaling(t)
        # tau required for noise sigma_t
        tau = self.time_sampling(t)

        rot_vf = so3_utils.calc_rot_vf(mat_t=rotmats_t, mat_1=rotmats_1)

        if potential is not None:
            assert (
                potential.shape == rot_vf.shape
            ), f"potential {potential.shape} != rot_vf {rot_vf.shape}"
            rot_vf += potential

        # scaled time along geodesic `t` -> `1`, broadcast over (N,3)
        geodesic_time = (scaling * d_t)[:, None, None]
        rotmats_next = so3_utils.geodesic_t(
            t=geodesic_time,
            mat=rotmats_1,
            base_mat=rotmats_t,
            rot_vf=rot_vf,
        )

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            # Sample IGSO(3) noise with a time-independent sigma_t, scaled by sqrt(dt)
            # Add IGSO(3) noise to keep rotmats_next on SO(3).
            num_batch, num_res, _, _ = rotmats_t.shape

            sigma_t = self._compute_sigma_t(
                tau,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            # Per-batch Brownian increment: scale sigma_t by sqrt(dt)
            sqrt_dt = torch.sqrt(d_t).to(sigma_t.device)
            sigma_t = (sigma_t * sqrt_dt).to(self.igso3.sigma_grid.device)

            # Only apply noise where sigma_t > 0
            apply_mask = sigma_t > 0
            if apply_mask.any():
                sigma_sel = sigma_t[apply_mask].to(self.igso3.sigma_grid.device)
                # safety check sigma range for selected only
                if sigma_sel.min() < self.igso3.sigma_grid.min():
                    raise ValueError(
                        f"rots sigma_t < igso3 grid min, noise will be larger than desired. Lower igso3_sigma_min."
                    )

                identity_noise = torch.eye(3, device=self._device)[None, None].repeat(
                    num_batch, num_res, 1, 1
                )
                noise_sel = self.igso3.sample(sigma_sel, num_res).to(self._device)
                noise_sel = noise_sel.reshape(apply_mask.sum(), num_res, 3, 3)
                intermediate_noise = identity_noise
                intermediate_noise[apply_mask] = noise_sel
                rotmats_next = torch.einsum(
                    "...ij,...jk->...ik", rotmats_next, intermediate_noise
                )

        return rotmats_next
