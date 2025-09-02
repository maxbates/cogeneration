import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from cogeneration.config.base import (
    InterpolantAATypesConfig,
    InterpolantAATypesScheduleEnum,
)
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.fm.aatypes import FlowMatcherAATypes
from cogeneration.data.fm.flow_matcher import FlowMatcher
from cogeneration.data.logits import combine_logits
from cogeneration.data.noise_mask import (
    mask_blend_1d,
    masked_categorical,
    uniform_categorical,
)

# TODO - configurable peak_mass and leave_mass_cap, or disable


class FlowMatcherAATypesUniformNoisy(FlowMatcherAATypes):
    """
    Noisy interpolant over 20 amino acid tokens (no MASK token).

    Uses Multiflow-style probability view and torch.multinomial() for jumps,
    with explicit drift vs noise handling
    """

    # intensity multiplier for noise => E[#noise jumps] ~ 0.5 when scale=1
    # Expected number of noise jumps per position over the full path:
    #   E ≈ NOISE_GAIN * eta_base * ∫_0^1 t(1 - t) dt
    #     = 3 * eta_base * (1/6)
    #     = 0.5 * eta_base
    # where:
    #   - NOISE_GAIN = 3 (by design),
    #   - eta_base = stochastic_noise_intensity * stochasticity_scale,
    #   - assuming min_sigma = 0.
    NOISE_GAIN: int = 3

    @property
    def num_tokens(self):
        return 20

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        num_batch, num_res = res_mask.shape
        return uniform_categorical(
            num_batch, num_res, num_tokens=self.num_tokens, device=self._device
        )

    def uniform_probs(
        self, aatypes: torch.Tensor, change_only: bool = True, eps: float = 1e-8
    ) -> torch.Tensor:
        """Uniform probability, optionally only outside aatypes states for change-only kernel"""
        B, N = aatypes.shape
        noise = torch.ones(B, N, self.num_tokens, device=self._device)
        if change_only:
            noise.scatter_(-1, aatypes.unsqueeze(-1).long(), 0.0)
        noise = noise / noise.sum(-1, keepdim=True).clamp_min(eps)
        return noise

    def _cumulative_hazard_rho(
        self,
        t: torch.Tensor,  # (B,)
        scale: float = 1.0,  # like _compute_sigma_t(), e.g. stochastic_scale
        min_sigma: float = 0.0,  # like _compute_sigma_t()
        kappa: float = 4.1589,  # hazard gain/steepness; larger -> quicker corruption
        eps: float = 1e-4,
        normalize: bool = False,  # => ρ(0)=1, ρ(1)=0 i.e. ignore scale
    ) -> torch.Tensor:
        """
        Returns rho(t), cumulative hazard from uniform / noise to t (or tau)
        cf. _compute_sigma_t() -> sigma_t which is instantaneous at `t`
        σ(t)^2 = scale^2 * t(1-t) + min_sigma^2

        rho(t) is built from sigma_t using a Poisson map
        we pull out `scale` from integral and multiply in linearly.

        - When normalize=False
            rho = 1 - exp(-kappa * scale * I_shape(t))
        - When normalize=True
            rho = [1 - exp(-kappa * I_shape(t))] / [1 - exp(-kappa * I_shape(0))]
            (scale is ignored so endpoints are fixed for all 'scale')

        where I_shape(t) = ∫_t^1 [1·u(1-u) + min_sigma^2] du
                         = (1 - 3t^2 + 2t^3)/6 + min_sigma^2 * (1 - t).

        Default kappa is ln(2)/(1/6) so ρ(t=0, scale=1) = 0.5
        """
        t = t.clamp(eps, 1.0 - eps)
        scale = max(scale, 0.0)  # >= 0
        scale *= self.NOISE_GAIN

        # shape-only closed form integral of σ^2 at scale=1
        I_shape = (1.0 - 3.0 * t**2 + 2.0 * t**3) / 6.0 + (min_sigma**2) * (
            1.0 - t
        )  # (B,)

        if not normalize:
            # linear-in-scale hazard
            rho = 1.0 - torch.exp(-kappa * scale * I_shape)
            return rho.clamp(0.0, 1.0)

        # normalize so endpoints are exact irrespective of scale
        I0_shape = (1.0 / 6.0) + (min_sigma**2)  # I_shape at t=0
        num = 1.0 - torch.exp(-kappa * I_shape)  # (B,)
        den = 1.0 - math.exp(-kappa * I0_shape)  # scalar
        rho = (num / (den + eps)).clamp(0.0, 1.0)
        return rho

    def corrupt(
        self,
        aatypes_1: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Corrupt AA residues from t=1 to t using uniform base (no mask),
        with optional additional stochasticity.
        """
        B, N = res_mask.shape
        assert aatypes_1.shape == (B, N)

        # scale t -> tau according to schedule (linear, exp)
        tau = self.aatypes_schedule(t=t)

        # drift: simple interpolation between current and base
        # t=1 aatypes as one-hot logits
        current_aatypes = F.one_hot(
            aatypes_1, num_classes=self.num_tokens
        ).float()  # (B, N, S)
        # t=0 base distribution is uniform across states
        base = self.uniform_probs(aatypes=aatypes_1, change_only=False)  # (B, N, S)
        # combine linearly in tau
        probs = (1.0 - tau.view(B, 1, 1)) * current_aatypes + tau.view(
            B, 1, 1
        ) * base  # (B, N, S)

        # stochasticity: add-on "rate" rho via Poisson map over ∫σ^2 hazard
        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # `rho` represents cumulative hazard
            rho = self._cumulative_hazard_rho(
                t=tau,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            rho = rho.view(B, 1, 1)

            # noise is change-only kernel
            noise = self.uniform_probs(aatypes=aatypes_1, change_only=True)

            # combine with drift probs using cumulative hazard
            probs = (1.0 - rho) * probs + rho * noise

        # regularize probabilities
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)

        # sample residues according to probs
        aatypes_t = torch.multinomial(probs.view(-1, self.num_tokens), num_samples=1)
        aatypes_t = aatypes_t.view(B, N)

        # residues outside `res_mask` are set to mask regardless; these should be excluded downstream
        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _poisson_noise_weight(
        self,
        t: torch.Tensor,  # (B,)
        d_t: torch.Tensor,  # scalar
        scale: float = 1.0,  # like _compute_sigma_t(), e.g. stochastic_scale
        min_sigma: float = 0.0,  # like _compute_sigma_t()
        peak_mass: float = 0.05,  # calibrated peak noise mass (@ t=0.5) for scale=1
    ) -> torch.Tensor:
        """
        Compute Poisson mapped noise weight nu_t (B,) from sigma_t:
        ν(t) = 1 - exp(-k * d_t * scale * σ(t)^2)
        where k is calibrated so ν(τ=0.5, scale=1) = peak_mass

        note that training's one-shot mix-in of noise is not exact (multiple jumps possible):
        difference cf. sampling is on the order of sqrt(k * nu * integral(σ(t)^2))
        but with peak_mass ~0.05 and d_t ~0.01, the difference is quite small
        """
        # get σ(t)^2 shape (~ scale=1.0)
        sigma_t_squared = t * (1.0 - t) + (min_sigma**2)  # (B,)
        # calibrate k using σ^2_peak = 0.25 + min_sigma^2 at t=0.5 when scale=1
        SIGMA_SQUARED_PEAK = 0.25 + min_sigma**2  # scale=1

        k = -math.log(1 - peak_mass) / d_t * SIGMA_SQUARED_PEAK + 1e-12

        # scale linearly by `scale`
        lambda_t = k * d_t * scale * sigma_t_squared * self.NOISE_GAIN
        # Poisson map
        nu_t = 1 - torch.exp(-lambda_t)
        return nu_t

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=20)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, S=20)
    ) -> torch.Tensor:
        B, N = aatypes_t.shape
        assert logits_1.shape == (B, N, self.num_tokens)

        # combine potential with predicted logits
        if potential is not None:
            assert (
                potential.shape == logits_1.shape
            ), f"guidance {potential.shape} != logits_1 {logits_1.shape}"
            logits_1 = combine_logits([logits_1, potential])

        # compute probabilities from softmaxed logits
        probs = F.softmax(logits_1 / self.cfg.drift_temp, dim=-1)  # (B, N, S)

        # uncertainty gate ~ 1 - p(current)
        uncertainty = self._uncertainty_gate(aatypes_t, probs).unsqueeze(
            -1
        )  # (B, N, 1)

        # compute drift step probs (off-diagonal mass)
        step_probs = d_t * probs * uncertainty

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            tau = self.aatypes_schedule(t=t)

            # compute noise leave , scaled by sigma_t
            nu_t = self._poisson_noise_weight(
                t=tau,
                d_t=d_t,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
                peak_mass=0.05,  # TODO configurable, or clearly tied to stochastic_noise_intensity
            )
            nu_t = nu_t.clamp_min(0.0).view(B, 1, 1)

            # change-only leave mass
            noise = self.uniform_probs(aatypes_t, change_only=True)

            # scale noise leave mass
            noise_probs = nu_t * noise

            # combine with drift probs
            step_probs += noise_probs

        leave_mass_cap = 0.25  # TODO configurable
        off = step_probs.clone()
        off.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
        row_sum = off.sum(-1, keepdim=True).clamp_min(1e-12)
        cap_scale = (leave_mass_cap / row_sum).clamp_max(1.0)
        step_probs = off * cap_scale

        # regularize step probs
        step_probs = self._regularize_step_probs(
            probs=step_probs,
            aatypes_t=aatypes_t,
        )

        # sample new residues from step_probs
        aatypes_t = torch.multinomial(
            step_probs.view(-1, self.num_tokens), num_samples=1
        )
        aatypes_t = aatypes_t.view(B, N)

        return aatypes_t
