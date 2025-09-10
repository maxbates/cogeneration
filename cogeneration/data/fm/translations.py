import math
from dataclasses import dataclass
from typing import Optional, Union

import torch
from scipy.optimize import linear_sum_assignment  # noqa

from cogeneration.config.base import (
    InterpolantTranslationsConfig,
    InterpolantTranslationsNoiseTypeEnum,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data.const import NM_TO_ANG_SCALE
from cogeneration.data.fm.flow_matcher import FlowMatcher
from cogeneration.data.noise_mask import (
    centered_gaussian,
    centered_harmonic,
    mask_blend_2d,
)
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass


@dataclass
class FlowMatcherTrans(FlowMatcher):
    """
    Flow matcher for translations domain (R^3).

    Supports two guassian priors, and the base / intermediate distributions may be different:
    - centered_gaussian: centered gaussian prior
    - centered_harmonic: centered harmonic prior

    Handles base noise sampling, corruption, and Euler updates with optional stochasticity and potentials.
    """

    cfg: InterpolantTranslationsConfig

    def sample_base(
        self, chain_idx: torch.Tensor, is_intermediate: bool
    ) -> torch.Tensor:
        """Sample t=0 translations noise (Å) using Gaussian or harmonic prior."""
        if is_intermediate:
            noise_type = self.cfg.intermediate_noise_type
        else:
            noise_type = self.cfg.initial_noise_type

        if noise_type == InterpolantTranslationsNoiseTypeEnum.centered_gaussian:
            return (
                centered_gaussian(*chain_idx.shape, device=self._device)
                * NM_TO_ANG_SCALE
            )
        elif noise_type == InterpolantTranslationsNoiseTypeEnum.centered_harmonic:
            return (
                centered_harmonic(chain_idx=chain_idx, device=self._device)
                * NM_TO_ANG_SCALE
            )
        else:
            raise ValueError(f"Unknown translation noise type {noise_type}")

    def batch_ot(
        self,
        trans_0: torch.Tensor,
        trans_1: torch.Tensor,
        res_mask: torch.Tensor,
        center: bool = False,
    ) -> torch.Tensor:
        """
        Compute optimal transport between two batches of translations.
        Returns OT mapping of trans_0 structures to trans_1 structures.
        Will force translations are centered if `center==True`.
        Does not re-order the translations within a structure.
        """
        num_batch, num_res = trans_0.shape[:2]

        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_0 = trans_0[noise_idx]
        batch_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]

        batch_0, batch_1, _ = batch_align_structures(
            batch_0, batch_1, mask=batch_mask, center=center
        )
        batch_0 = batch_0.reshape(num_batch, num_batch, num_res, 3)  # (B, B, N, 3)
        batch_1 = batch_1.reshape(num_batch, num_batch, num_res, 3)  # (B, B, N, 3)
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)  # (B, B, N)

        distances = torch.linalg.norm(batch_0 - batch_1, dim=-1)  # (B, B, N)
        cost_matrix = torch.sum(distances, dim=-1) / torch.sum(batch_mask, dim=-1)

        noise_perm, gt_perm = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return batch_0[(tuple(gt_perm), tuple(noise_perm))]

    def vector_field(
        self,
        t: torch.Tensor,  # (B,)
        trans_1: torch.Tensor,  # (B, N, 3)
        trans_t: torch.Tensor,  # (B, N, 3)
    ) -> torch.Tensor:
        if self.cfg.sample_schedule == InterpolantTranslationsScheduleEnum.linear:
            return (trans_1 - trans_t) / (1 - t.view(-1, 1, 1))
        elif self.cfg.sample_schedule == InterpolantTranslationsScheduleEnum.vpsde:
            bmin = self.cfg.vpsde_bmin
            bmax = self.cfg.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1 - t)
            alpha_t = torch.exp(-bmin * (1 - t) - 0.5 * (1 - t) ** 2 * (bmax - bmin))
            bt_v = bt.view(-1, 1, 1)
            alpha_t_v = alpha_t.view(-1, 1, 1)
            return 0.5 * bt_v * trans_t + 0.5 * bt_v * (
                torch.sqrt(alpha_t_v) * trans_1 - trans_t
            ) / (1 - alpha_t_v)
        else:
            raise ValueError(f"Invalid sample schedule: {self.cfg.sample_schedule}")

    def corrupt(
        self,
        trans_1: torch.Tensor,  # (B, N, 3)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        chain_idx: torch.Tensor,  # (B, N)
        stochasticity_scale: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        tau = self.time_training(t)
        trans_0 = self.sample_base(chain_idx=chain_idx, is_intermediate=False)

        if self.cfg.batch_ot:
            trans_0 = self.batch_ot(
                trans_0,
                trans_1,
                res_mask=res_mask,
                center=False,
            )
        elif self.cfg.batch_align:
            trans_0, _, _ = batch_align_structures(
                pos_1=trans_0,
                pos_2=trans_1,
                mask=res_mask,
                center=False,
            )

        if self.cfg.train_schedule == InterpolantTranslationsScheduleEnum.linear:
            trans_t = (1 - tau.view(-1, 1, 1)) * trans_0 + tau.view(-1, 1, 1) * trans_1
        else:
            raise ValueError(f"Unknown trans schedule {self.cfg.train_schedule}")

        stochasticity_scale = stochasticity_scale.to(t.device).view(-1)  # (B,)
        if (stochasticity_scale > 0).any():
            sigma_t = self._compute_sigma_t(
                tau,
                scale=stochasticity_scale,
            )
            intermediate_noise = self.sample_base(
                chain_idx=chain_idx, is_intermediate=True
            )
            intermediate_noise = intermediate_noise * sigma_t[..., None, None]
            trans_t += intermediate_noise

        trans_t = mask_blend_2d(trans_t, trans_1, diffuse_mask)
        trans_t -= batch_center_of_mass(trans_t, mask=res_mask)[:, None]

        return trans_t * res_mask[..., None]

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        trans_1: torch.Tensor,  # (B, N, 3)
        trans_t: torch.Tensor,  # (B, N, 3)
        chain_idx: torch.Tensor,  # (B, N)
        stochasticity_scale: torch.Tensor,  # (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, 3) VF
    ) -> torch.Tensor:
        tau = self.time_sampling(t)
        trans_vf = self.vector_field(t=tau, trans_1=trans_1, trans_t=trans_t)

        # optionally add intermediate noise
        stochasticity_scale = stochasticity_scale.to(t.device).view(-1)  # (B,)
        if (stochasticity_scale > 0).any():
            intermediate_noise = self.sample_base(
                chain_idx=chain_idx,
                is_intermediate=True,
            )
            sigma_t = self._compute_sigma_t(
                tau,
                scale=stochasticity_scale,
            )
            sigma_t = sigma_t.to(trans_t.device)
            # Per-batch Brownian increment scales with sqrt(dt)
            sqrt_dt = torch.sqrt(d_t).to(trans_t.device)
            intermediate_noise = intermediate_noise * sqrt_dt * sigma_t[..., None, None]
        else:
            intermediate_noise = torch.zeros_like(trans_t)

        if potential is not None:
            assert (
                potential.shape == trans_vf.shape
            ), f"potential {potential.shape} != trans_vf {trans_vf.shape}"
            trans_vf += potential

        trans_next = trans_t + trans_vf * d_t + intermediate_noise
        return trans_next
