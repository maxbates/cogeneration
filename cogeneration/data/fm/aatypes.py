import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from cogeneration.config.base import (
    InterpolantAATypesConfig,
    InterpolantAATypesScheduleEnum,
)
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.fm.flow_matcher import FlowMatcher
from cogeneration.data.logits import combine_logits
from cogeneration.data.noise_mask import (
    mask_blend_1d,
    masked_categorical,
    uniform_categorical,
)


@dataclass
class FlowMatcherAATypes(FlowMatcher, ABC):
    """
    Noisy aatypes interpolant

    Uses Multiflow-style probability view and torch.multinomial() for jumps,
    with explicit drift vs noise handling
    """

    cfg: InterpolantAATypesConfig

    # TODO - configurable leave_mass_cap, or disable
    # maximum leave mass during sampling
    leave_mass_cap: Optional[float] = 0.25

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
    @abstractmethod
    def num_tokens(self) -> int:
        """Number of amino acid tokens."""
        raise NotImplementedError

    @abstractmethod
    def sample_base(
        self,
        res_mask: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """Sample from an aatypes (B, N) base distribution (t=0)."""
        raise NotImplementedError

    @abstractmethod
    def corrupt(
        self,
        aatypes_1: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
    ) -> torch.Tensor:
        """Corrupt aatypes (B, N) from t=1 to t"""
        raise NotImplementedError

    @abstractmethod
    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, S)
    ) -> torch.Tensor:
        """Perform aatypes single Euler step update, returning new aatypes (B, N)"""
        raise NotImplementedError

    def _aatypes_schedule(
        self,
        t: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Schedule for aatypes drift rate per-component scales

        Returns `tau` (mapped time) (B,)
        """
        if self.cfg.schedule == InterpolantAATypesScheduleEnum.linear:
            y = t
        elif self.cfg.schedule == InterpolantAATypesScheduleEnum.exp:
            # map t -> y in [0,1): y = (1 - e^{-rt}) / (1 - e^{-r})
            r = float(abs(self.cfg.schedule_exp_rate))
            y = (1.0 - torch.exp(-r * t)) / (1.0 - math.e ** (-r))
        else:
            raise ValueError(f"Unknown aatypes schedule {self.cfg.schedule}")

        return y.clamp(min=0.0, max=1.0)

    def time_training(self, t: torch.Tensor) -> torch.Tensor:
        return self._aatypes_schedule(t)

    def time_sampling(self, t: torch.Tensor) -> torch.Tensor:
        return self._aatypes_schedule(t)

    def _uncertainty_gate(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        drift_probs: torch.Tensor,  # (B, N, S)
    ) -> torch.Tensor:
        """
        Uncertainty gate (i.e. `1.0 - p_current`) multiplier (B, N). Ones if disabled.

        May wish to .clamp_min(floor) e.g. for noise on very certain positions.
        """
        is_mask = aatypes_t == MASK_TOKEN_INDEX
        is_aa = ~is_mask

        # uncertainty gating: for AA rows use 1 - p(current), for MASK rows set to 1
        if self.cfg.uncertainty_gating:
            p_current = (
                drift_probs.gather(-1, aatypes_t.long().unsqueeze(-1))
                .squeeze(-1)
                .clamp(0.0, 1.0)
            )  # (B, N)
            # (1 - p_current) ** gamma
            uncertainty = (1.0 - p_current) ** self.cfg.uncertainty_gating_gamma
            # mask to AA rows
            uncertainty = uncertainty * is_aa.float() + (
                1.0 * is_mask.float()
            )  # (B, N)
        else:
            uncertainty = torch.ones_like(
                is_aa, dtype=torch.float, device=self._device
            )  # (B, N)

        return uncertainty  # (B, N)

    def _regularize_step_probs(
        self,
        probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:  # (B, N, S)
        """
        Regularize the softmax probabilities to build a per-step probability row for euler discrete sampling:

        - rows sum to 1
        - current state is set to 1 - sum of all other values

        Has the effect of injecting "stay" mass for the current state,
        so that if the probability of the current residue is low, it will likely jump,
        but if it is high relative to other residues, it will likely stay
        """
        num_batch, num_res, S = probs.shape
        assert aatypes_t.shape == (num_batch, num_res)

        # clone to avoid in-place modification
        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        probs = probs.clone().clamp(min=0.0, max=1.0)

        # zero current state
        probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)

        # set diagonal (current state) to 1 - sum(off-diagonal)
        row_sums = probs.sum(-1, keepdim=True)
        probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 1.0 - row_sums)

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        # in case negative or out-of-bound values appear after the diagonal assignment.
        probs = torch.clamp(probs, min=0.0, max=1.0)

        return probs

    def _uniform_kernel(
        self,
        aatypes: torch.Tensor,  # (B, N)
        change_only: bool = True,
        forbid_cols: Tuple[
            int, ...
        ] = (),  # e.g. (MASK_TOKEN_INDEX,) in masking for change kernel
        eps: float = 1e-12,
    ) -> torch.Tensor:
        B, N = aatypes.shape
        S = self.num_tokens
        K = torch.ones(B, N, S, device=self._device)
        for col in forbid_cols:
            K[..., col] = 0.0
        if change_only:
            K.scatter_(-1, aatypes.long().unsqueeze(-1), 0.0)
        K = K / K.sum(-1, keepdim=True).clamp_min(eps)
        return K

    def _cumulative_hazard_rho(
        self,
        t: torch.Tensor,  # (B,) tau
        scale: torch.Tensor,  # (B,) like _compute_sigma_t(), e.g. stochastic_scale
        min_sigma: float = 0.0,  # like _compute_sigma_t()
        kappa: float = 4.1589,  # hazard gain/steepness; larger -> quicker corruption
        eps: float = 1e-4,
        normalize: bool = False,  # => ρ(0)=1, ρ(1)=0 i.e. ignore scale
    ) -> torch.Tensor:
        """
        Returns rho(t), cumulative hazard from uniform / noise to t (or tau)
        cf. _compute_sigma_t() -> sigma_t which is instantaneous at `t`
        sigma_t^2 = scale^2 * t(1-t) + min_sigma^2

        rho(t) is built from sigma_t using a Poisson map.
        we pull out `scale` from integral and multiply in linearly.

        - When normalize=False
            rho = 1 - exp(-kappa * scale * I_shape(t))
        - When normalize=True
            rho = [1 - exp(-kappa * I_shape(t))] / [1 - exp(-kappa * I_shape(0))]
            (scale is ignored so endpoints are fixed for all 'scale')

        where I_shape(t) = ∫ t^1 [1·u(1-u) + min_sigma^2] du
                         = (1 - 3t^2 + 2t^3)/6 + min_sigma^2 * (1 - t).

        Default kappa is ln(2)/(1/6) so rho(t=0, scale=1) = 0.5
        """
        t = t.clamp(0.0, 1.0)
        # ensure non-negative tensor scale (B,)
        scale = scale.to(t.device).clamp_min(0.0)
        scale = scale * self.NOISE_GAIN

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

    def _sample_aatypes(
        self,
        probs: torch.Tensor,  # (B, N, S)
        aatypes_1: torch.Tensor,  # (B, N)
        res_mask: Optional[torch.Tensor],  # (B, N)
        diffuse_mask: Optional[torch.Tensor],  # (B, N)
    ) -> torch.Tensor:
        """
        Sample residues according to probs
        """
        B, N = aatypes_1.shape
        assert probs.shape == (B, N, self.num_tokens)

        # sample residues according to probs
        aatypes_t = torch.multinomial(probs.view(-1, self.num_tokens), num_samples=1)
        aatypes_t = aatypes_t.view(B, N)

        # residues outside `res_mask` are set to mask regardless; these should be excluded downstream
        if res_mask is not None:
            aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        # limit to diffuse_mask
        if diffuse_mask is not None:
            aatypes_t = mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

        return aatypes_t

    def _poisson_noise_weight(
        self,
        t: torch.Tensor,  # (B,) tau
        d_t: torch.Tensor,  # scalar
        scale: torch.Tensor,  # (B,) like _compute_sigma_t(), e.g. stochastic_scale
        min_sigma: float = 0.0,  # like _compute_sigma_t()
    ) -> torch.Tensor:
        """
        Compute Poisson mapped noise weight nu_t (B,) from sigma_t:
        nu_t = 1 - exp(d_t * sigma_t^2 * (scale * NOISE_GAIN))

        note that training's one-shot mix-in of noise is not exact (multiple jumps possible):
        difference cf. sampling is on the order of sqrt(k * nu * integral(sigma_t^2))
        but with scale OOM~1 and d_t ~0.01, the difference is quite small
        """
        # get sigma_t^2 shape (~ scale=1.0)
        sigma_t_squared = t * (1.0 - t) + (min_sigma**2)  # (B,)
        # scale linearly by `scale`, and include NOISE_GAIN
        scale = scale.to(t.device)
        lambda_t = d_t * sigma_t_squared * (scale * self.NOISE_GAIN)
        # Poisson map
        nu_t = 1 - torch.exp(-lambda_t)
        return nu_t

    def _cap_leave_mass(
        self,
        step_probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """
        Scale step_probs to limit leave mass
        """
        if self.leave_mass_cap is None:
            return step_probs

        off = step_probs.clone()
        off.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
        row_sum = off.sum(-1, keepdim=True).clamp_min(1e-12)
        shrink = (self.leave_mass_cap / row_sum).clamp_max(1.0)
        return off * shrink


class FlowMatcherAATypesUniform(FlowMatcherAATypes):
    """
    Noisy interpolant over 20 amino acid tokens (no MASK token).

    Uses Multiflow-style probability view and torch.multinomial() for jumps,
    with explicit drift vs noise handling
    """

    @property
    def num_tokens(self):
        return 20

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns uniform (random AA) t=0 base distribution (B, N)
        """
        B, N = res_mask.shape
        return uniform_categorical(
            B, N, num_tokens=self.num_tokens, device=self._device
        )

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

        tau = self.time_training(t)

        # drift: simple interpolation between current and base
        # t=1 aatypes as one-hot logits
        current_aatypes = F.one_hot(
            aatypes_1, num_classes=self.num_tokens
        ).float()  # (B, N, S)
        # t=0 base distribution is uniform across states
        base = self._uniform_kernel(aatypes=aatypes_1, change_only=False)  # (B, N, S)
        # combine linearly
        tau_b11 = tau.view(B, 1, 1)
        probs = tau_b11 * current_aatypes + (1.0 - tau_b11) * base  # (B, N, S)

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        # stochasticity: add-on "rate" rho via Poisson map over ∫σ^2 hazard
        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            # `rho` represents cumulative hazard
            rho = self._cumulative_hazard_rho(
                t=tau,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            rho = rho.view(B, 1, 1)

            # change-only kernel
            kernel_change = self._uniform_kernel(aatypes=aatypes_1, change_only=True)

            # combine with drift probs using cumulative hazard
            probs = (1.0 - rho) * probs + rho * kernel_change

        # regularize probabilities
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)

        # sample from probs, limiting to res_mask and diffuse_mask
        return self._sample_aatypes(
            probs=probs,
            aatypes_1=aatypes_1,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
        )

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=20)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, S=20)
    ) -> torch.Tensor:
        B, N = aatypes_t.shape
        assert logits_1.shape == (B, N, self.num_tokens)
        tau = self.time_sampling(t)

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

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            # compute noise leave mass scale
            nu_t = self._poisson_noise_weight(
                t=tau,
                d_t=d_t,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            nu_t = nu_t.clamp_min(0.0).view(B, 1, 1)

            # change-only kernel
            kernel_change = self._uniform_kernel(aatypes_t, change_only=True)

            # combine scaled noise kernel with drift probs
            step_probs += nu_t * kernel_change

        # cap leave mass
        step_probs = self._cap_leave_mass(
            step_probs=step_probs,
            aatypes_t=aatypes_t,
        )

        # regularize step probs
        step_probs = self._regularize_step_probs(
            probs=step_probs,
            aatypes_t=aatypes_t,
        )

        return self._sample_aatypes(
            probs=step_probs,
            aatypes_1=aatypes_t,
            res_mask=None,
            diffuse_mask=None,
        )


class FlowMatcherAATypesMasking(FlowMatcherAATypes):
    """
    Noisy aatypes interpolant over 21 amino acid tokens (including MASK token).

    Uses Multiflow-style probability view and torch.multinomial() for jumps,
    with explicit drift vs noise handling
    """

    @property
    def num_tokens(self):
        return 21

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns all-mask t=0 base distribution (B, N)
        """
        B, N = res_mask.shape
        return masked_categorical(B, N, device=self._device)

    def _masking_kernels(
        self,
        aatypes: torch.Tensor,  # (B, N)
        eps: float = 1e-12,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Masking kernel for noisy aatypes interpolant
        """
        is_mask = (aatypes == MASK_TOKEN_INDEX).unsqueeze(-1)  # (B, N, 1)
        is_aa = ~is_mask  # (B, N, 1)

        # unmask kernel: uniform over AA columns only (exclude MASK)
        kernel_unmask = (
            self._uniform_kernel(
                aatypes=aatypes,
                change_only=False,
                forbid_cols=(MASK_TOKEN_INDEX,),
            )
            * is_mask.float()
        )

        # change kernel: exclude self and MASK column
        kernel_change = (
            self._uniform_kernel(
                aatypes=aatypes,
                change_only=True,
                forbid_cols=(MASK_TOKEN_INDEX,),
            )
            * is_aa.float()
        )

        return kernel_unmask, kernel_change

    def corrupt(
        self,
        aatypes_1: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
    ) -> torch.Tensor:
        """
        Linear drift to mask base distribution, one-shot Poisson noise mixed between unmask and change
        """
        B, N = res_mask.shape
        assert aatypes_1.shape == (B, N)

        # scale t -> tau according to schedule (linear, exp)
        tau = self._aatypes_schedule(t=t)

        # drift: simple interpolation between current and base
        # t=1 aatypes as one-hot logits
        current_aatypes = F.one_hot(
            aatypes_1, num_classes=self.num_tokens
        ).float()  # (B, N, S)
        # t=0 base distribution is all mask
        mask_onehot = torch.zeros(
            B, N, self.num_tokens, device=self._device
        )  # (B, N, S)
        mask_onehot[..., MASK_TOKEN_INDEX] = 1.0
        # combine linearly
        tau_b11 = tau.view(B, 1, 1)
        probs = tau_b11 * current_aatypes + (1.0 - tau_b11) * mask_onehot  # (B, N, S)

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            # `rho` represents cumulative hazard
            rho = self._cumulative_hazard_rho(
                t=tau,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            rho = rho.view(B, 1, 1)

            # get unmask and change kernels
            kernel_unmask, kernel_change = self._masking_kernels(
                aatypes=aatypes_1
            )  # (B, N, S)

            # split the noise mass by current mask probability under drift
            p_mask = probs[..., MASK_TOKEN_INDEX].unsqueeze(-1)  # (B, N, 1)
            kernel_noise = (
                p_mask * kernel_unmask + (1.0 - p_mask) * kernel_change
            )  # (B, N, S)

            # combine with drift probs using cumulative hazard
            probs = (1.0 - rho) * probs + rho * kernel_noise

        # regularize probabilities
        probs = probs / probs.sum(-1, keepdim=True).clamp_min(1e-12)

        # sample from probs, limiting to res_mask and diffuse_mask
        return self._sample_aatypes(
            probs=probs,
            aatypes_1=aatypes_1,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
        )

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, S=21)
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

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        # Poisson noise per step (using unmask/change kernels dependent on current state)
        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            # t -> tau according to schedule
            tau = self._aatypes_schedule(t=t)

            # compute noise leave mass scale
            nu_t = self._poisson_noise_weight(
                t=tau,
                d_t=d_t,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )
            nu_t = nu_t.clamp_min(0.0).view(B, 1, 1)

            kernel_unmask, kernel_change = self._masking_kernels(
                aatypes=aatypes_t
            )  # (B, N, S)

            # split the noise mass by whether position is mask or AA
            is_mask = (aatypes_t == MASK_TOKEN_INDEX).unsqueeze(-1).float()  # (B, N, 1)
            kernel_noise = (
                is_mask * kernel_unmask + (1.0 - is_mask) * kernel_change
            )  # (B, N, S)

            # combine scaled noise kernel with drift probs
            step_probs += nu_t * kernel_noise

        # cap leave mass
        step_probs = self._cap_leave_mass(step_probs, aatypes_t)

        # regularize step probs
        step_probs = self._regularize_step_probs(
            probs=step_probs,
            aatypes_t=aatypes_t,
        )

        return self._sample_aatypes(
            probs=step_probs,
            aatypes_1=aatypes_t,
            res_mask=None,
            diffuse_mask=None,
        )
