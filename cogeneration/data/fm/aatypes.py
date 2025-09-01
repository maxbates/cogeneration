import math
from abc import ABC, abstractmethod
from typing import Optional, Tuple

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


class FlowMatcherAATypes(FlowMatcher, ABC):
    """
    Flow matcher for amino acid types (categorical domain).
    """

    def __init__(self, cfg: InterpolantAATypesConfig):
        self.cfg = cfg
        self._device: Optional[torch.device] = None

    def set_device(self, device: torch.device):
        self._device = device

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
        stochasticity_scale: float = 1.0,
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
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, S)
    ) -> torch.Tensor:
        """Perform aatypes single Euler step update, returning new aatypes (B, N)"""
        raise NotImplementedError

    def _aatypes_schedule(
        self,
        t: torch.Tensor,  # (B,)
        kappa: float = 1.0,
        eps: float = 0.05,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Schedule for aatypes drift rate per-component scales

        Returns `sched` and `tau` (mapped t -> y),
        where `sched` is (1 + kappa * tau + eps) / (1 - tau + eps)

        eps smooths blowup at t=1
        kappa > 0 adds some mass early on
        """
        if self.cfg.schedule == InterpolantAATypesScheduleEnum.linear:
            y = t
        elif self.cfg.schedule == InterpolantAATypesScheduleEnum.exp:
            # map t -> y in [0,1): y = (1 - e^{-rt}) / (1 - e^{-r})
            r = float(abs(self.cfg.schedule_exp_rate))
            y = (1.0 - torch.exp(-r * t)) / (1.0 - math.e ** (-r))
        else:
            raise ValueError(f"Unknown aatypes schedule {self.cfg.schedule}")

        sched = (1.0 + kappa * y + eps) / (1.0 - y + eps)
        sched = sched.clamp_min(0.0)

        return sched, y

    def _regularize_step_probs(
        self,
        probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:  # (B, N, S)
        """
        Regularize the softmax probabilities to build a per-step probability row for euler discrete sampling:

        - rows sum to 1
        - current state is set to 1 - sum of all other values

        So that if the probability of the current residue is low, it will likely jump,
        but if it is high relative to other residues, it will likely stay
        """
        num_batch, num_res, S = probs.shape
        device = probs.device
        assert aatypes_t.shape == (num_batch, num_res)

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        probs = torch.clamp(probs, min=0.0, max=1.0)

        batch_idx = torch.arange(num_batch, device=device).repeat_interleave(num_res)
        residue_idx = torch.arange(num_res, device=device).repeat(num_batch)
        curr_states = aatypes_t.long().flatten()

        # set the probabilities corresponding to the current amino acid types to 0.0
        probs[batch_idx, residue_idx, curr_states] = 0.0

        # adjust the probabilities at the current positions to be 1 - sum of all other values in the row
        row_sums = torch.sum(probs, dim=-1).flatten()
        probs[batch_idx, residue_idx, curr_states] = 1.0 - row_sums

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        # in case negative or out-of-bound values appear after the diagonal assignment.
        probs = torch.clamp(probs, min=0.0, max=1.0)

        return probs

    def _aatypes_jump_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: Optional[torch.Tensor],  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        CTMC jump for aatypes. pass no logits for stochastic jump.

        Uses logits -> rate matrix to allow sequnce to explore neighboring states
        in proportion to the rates the network thinks are possible.
        So notice that unlike in training, where jumps are to a uniform-random sampled state,
        here the jumps are to states the network thinks are plausible.
        The thinking is that during training, the model must learn a broad "restoring drift" and uniform sampling is class balanced.
        (We can't use logits during training without simulation, they are just a one-hot of the sequence.)
        However, this training may slow convergence, or the model may only learn to fix very wrong residues.

        This is different than the `cfg.noise` term,
        which adds noise to the rate matrix in determinsitic interpolation.
        """
        B, N = aatypes_t.shape
        t = t.to(self._device)
        d_t = d_t.to(self._device)

        if logits_1 is None:
            # uniform rates excluding self
            S = self.num_tokens
            prob_rows = torch.ones(B, N, S, device=self._device)
            current_idx = aatypes_t.unsqueeze(-1).long()
            prob_rows.scatter_(-1, current_idx, 0.0)
            prob_rows = prob_rows / prob_rows.sum(-1, keepdim=True).clamp_min(1e-8)
        else:
            S = logits_1.shape[-1]
            prob_rows = F.softmax(logits_1 / self.cfg.drift_temp, dim=-1)
            prob_rows = prob_rows.clamp(min=1e-8)

        # Build rate matrix (positive exits)
        current_idx = aatypes_t.unsqueeze(-1).long()  # (B,N,1)
        exit_rates = prob_rows.scatter_(-1, current_idx, 0.0)
        exit_sums = exit_rates.sum(-1, keepdim=True)
        step_rates = exit_rates.clone()
        step_rates.scatter_(-1, current_idx, -exit_sums)  # current = neg sum of others

        # Multiply by sigma_t so amount of noise mirrors other domains.
        # Multiplying the whole rate matrix by the same value preserves valid rate matrix.
        sigma_t = self._compute_sigma_t(
            t,
            scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
        )
        step_rates = step_rates * sigma_t[..., None, None]  # (B, N, S)

        # decide whether each residue jumps during d_t
        # total exit rate λ_i = − q_{i, current_state}
        lambda_step = -step_rates.gather(-1, current_idx).squeeze(-1)  # (B, N), >0
        p_jump = 1.0 - torch.exp(-lambda_step * d_t.view(-1, 1))  # (B, N)
        jump_mask = torch.rand_like(p_jump) < p_jump  # bool (B, N)

        if not jump_mask.any():
            return aatypes_t

        # set current state to 0, then normalize by exit rate
        jump_aa_probs = step_rates.clone()
        jump_aa_probs.scatter_(-1, current_idx, 0.0)
        jump_aa_probs = jump_aa_probs / lambda_step.clamp_min(1e-8).unsqueeze(-1)

        # sample new aa only for residues that jump
        jumped_states = (
            torch.multinomial(jump_aa_probs[jump_mask].reshape(-1, S), 1)
            .squeeze(-1)
            .to(aatypes_t.dtype)
        )

        # set new aa for jumped residues
        jumped_aatypes_t = aatypes_t.clone()
        jumped_aatypes_t[jump_mask] = jumped_states
        return jumped_aatypes_t


class FlowMatcherAATypesUniform(FlowMatcherAATypes):
    """
    Uniform interpolant over 20 amino acid tokens (no MASK token).

    - t=0 base sampling: uniform over 20 tokens
    - corruption: blends toward base according to t with optional stochastic jumps
    - euler_step: discrete flow update without a MASK token
    """

    @property
    def num_tokens(self):
        return 20

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        num_batch, num_res = res_mask.shape
        return uniform_categorical(
            num_batch, num_res, num_tokens=self.num_tokens, device=self._device
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
        Corrupt AA residues from t=1 to t using uniform base sampling (no MASK token).
        Optionally add stochastic CTMC-style jumps controlled by sigma_t.
        """
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch,), f"t.shape: {t.shape} != (num_batch,)"
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        u = torch.rand(num_batch, num_res, device=self._device)
        corruption_mask = (u < (1 - t.view(-1, 1))).int()
        aatypes_0 = self.sample_base(res_mask=res_mask)
        aatypes_t = mask_blend_1d(aatypes_0, aatypes_1, corruption_mask)

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            sigma_t = self._compute_sigma_t(
                t,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            ).unsqueeze(
                1
            )  # (B, 1)
            p_jump = sigma_t.clamp(max=1.0)
            jump_mask = torch.rand(num_batch, num_res, device=self._device) < p_jump
            if jump_mask.any():
                jump_noise = self.sample_base(res_mask=res_mask)
                aatypes_t = mask_blend_1d(jump_noise, aatypes_t, jump_mask.int())

        # residues outside `res_mask` are set to mask regardless; these should be excluded downstream
        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _euler_step_uniform(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=20)
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        num_batch, num_res, num_states = logits_1.shape
        assert aatypes_t.shape == (num_batch, num_res)
        assert num_states == 20
        assert (
            aatypes_t.max() < 20
        ), "No UNK tokens allowed in the uniform sampling step!"

        temp = self.cfg.drift_temp
        noise = self.cfg.stochastic_noise_intensity

        # convert logits to probabilities
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

        # probability of x1 matching xt exactly, used for uncertainty scaling (noise * prob_eq_xt)
        pt_x1_eq_xt_prob = torch.gather(
            pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)
        )  # (B, N, 1)
        assert pt_x1_eq_xt_prob.shape == (num_batch, num_res, 1)

        # broadcast `t` to per-site
        t_b = t.to(self._device).view(-1, 1, 1)

        # compute step probabilities (scaled by d_t), with noise and time factoring.
        # encourages transitions with a uncertainty-scaling 'noise' term for the current residue.
        step_probs = d_t * (
            pt_x1_probs * ((1.0 + noise + noise * (num_states - 1) * t_b) / (1.0 - t_b))
            + noise * pt_x1_eq_xt_prob
        )

        # force valid rate matrix
        step_probs = self._regularize_step_probs(step_probs, aatypes_t)

        # sample new residues from step_probs
        new_aatypes = torch.multinomial(step_probs.view(-1, num_states), num_samples=1)
        return new_aatypes.view(num_batch, num_res)

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=20)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, S=20)
    ) -> torch.Tensor:
        """
        Perform a single Euler update step for the uniform interpolant (S=20).
        Adds optional jump step if stochastic is enabled.
        """
        if potential is not None:
            assert (
                potential.shape == logits_1.shape
            ), f"Guidance logits shape {potential.shape} does not match logits_1 shape {logits_1.shape}"
            logits_1 = combine_logits([logits_1, potential])

        aatypes_t = self._euler_step_uniform(
            d_t=d_t, t=t, logits_1=logits_1, aatypes_t=aatypes_t
        )

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            aatypes_t = self._aatypes_jump_step(
                d_t,
                t=t,
                logits_1=logits_1,
                aatypes_t=aatypes_t,
                stochasticity_scale=stochasticity_scale,
            )
        return aatypes_t


class FlowMatcherAATypesMasking(FlowMatcherAATypes):
    """
    Masking interpolant with optional purity-based unmasking (21 tokens including MASK).

    - t=0 base sampling: all MASK
    - corruption: blends toward base according to t with optional stochastic jumps
    - euler_step: discrete flow update that excludes MASK in softmax; optionally
      uses purity-based unmasking schedule
    """

    @property
    def num_tokens(self):
        return 21

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        num_batch, num_res = res_mask.shape
        return masked_categorical(num_batch, num_res, device=self._device)

    def corrupt(
        self,
        aatypes_1: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Corrupt AA residues from t=1 to t using masking base (all MASK at t=0).
        Optionally add stochastic CTMC-style jumps controlled by sigma_t.
        """
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch,), f"t.shape: {t.shape} != (num_batch,)"
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        u = torch.rand(num_batch, num_res, device=self._device)
        corruption_mask = (u < (1 - t.view(-1, 1))).int()
        aatypes_0 = self.sample_base(res_mask=res_mask)
        aatypes_t = mask_blend_1d(aatypes_0, aatypes_1, corruption_mask)

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            sigma_t = self._compute_sigma_t(
                t,
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            ).unsqueeze(1)
            p_jump = sigma_t.clamp(max=1.0)
            jump_mask = torch.rand(num_batch, num_res, device=self._device) < p_jump
            if jump_mask.any():
                jump_noise = self.sample_base(res_mask=res_mask)
                aatypes_t = mask_blend_1d(jump_noise, aatypes_t, jump_mask.int())

        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)
        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _euler_step_masking(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        num_batch, num_res, num_states = logits_1.shape
        assert num_states == 21
        assert aatypes_t.shape == (num_batch, num_res)

        device = logits_1.device
        temp = self.cfg.drift_temp
        noise = self.cfg.stochastic_noise_intensity  # used to be its own value, ~20.0

        # set mask to small negative so won't be picked in softmax
        logits_1[:, :, MASK_TOKEN_INDEX] = -1e9

        # convert logits to probabilities
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

        # prepare a (0,0,...1) mask vector to help add masking transitions.
        mask_one_hot = torch.zeros((num_states,), device=device)
        mask_one_hot[MASK_TOKEN_INDEX] = 1.0

        # identify which positions are currently mask
        aatypes_t_is_mask = (
            (aatypes_t == MASK_TOKEN_INDEX).view(num_batch, num_res, 1).float()
        )

        # broadcast t and d_t to (B, 1, 1) for (B, N, S) tensors
        t = t.to(self._device).view(-1, 1, 1)
        d_t = d_t.to(self._device)

        # compute step probabilities (scaled by d_t), with noise and time factoring
        step_probs = d_t * pt_x1_probs * ((1.0 + noise * t) / (1.0 - t))  # (B, N, S)
        # add transitions from non-mask to mask as noise
        step_probs += (
            d_t * (1.0 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * noise
        )

        # force valid rate matrix
        step_probs = self._regularize_step_probs(step_probs, aatypes_t)

        # sample new residues from step_probs
        new_aatypes = torch.multinomial(step_probs.view(-1, num_states), num_samples=1)
        return new_aatypes.view(num_batch, num_res)

    def _euler_step_purity(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
    ):
        """
        Perform an Euler step with purity sampling scheme, which decides which tokens to unmask
        based on their log-probabilities.

        This function identifies masked residues, ranks them by max log prob,
        picks some number to unmask, and re-masks some positions proportional to (d_t, noise, t).

        On average more dimensions are unmasked at each step, with the probability of ultimately
        selecting the correct token increasing as noise increases.

        This function is designed specifically for the "masking" interpolant type, where S = 21.
        """
        num_batch, num_res, num_states = logits_1.shape

        assert aatypes_t.shape == (num_batch, num_res)
        assert (
            num_states == 21
        ), "Purity-based unmasking only works with masking interpolant type"

        temp = self.cfg.drift_temp
        noise = self.cfg.stochastic_noise_intensity
        t = t.to(self._device)
        d_t = d_t.to(self._device)

        # remove mask dimension to handle only the 20 valid aa states. mask is 21.
        logits_1_wo_mask = logits_1[:, :, : (num_states - 1)]  # (B, N, S-1)
        # convert logits to probabilities for the non-mask states
        pt_x1_probs = F.softmax(logits_1_wo_mask / temp, dim=-1)  # (B, N, S-1)
        # find maximum log prob among valid states
        max_logprob = torch.log(pt_x1_probs).max(dim=-1)[0]  # (B, N)
        # bias so only currently masked are favored for unmasking
        max_logprob -= (aatypes_t != MASK_TOKEN_INDEX).float() * 1e9
        # sort positions by descending maximum log-prob (highest to lowest)
        sorted_max_logprob_indices = torch.argsort(
            max_logprob, dim=-1, descending=True
        )  # (B, N)

        # determine how many masked residues to unmask
        # probability of unmasking is scaled by d_t, noise, and time
        unmask_probs = (
            (d_t * ((1.0 + noise * t) / (1.0 - t))).clamp(max=1.0).to(self._device)
        )  # scalar
        masked_counts = (aatypes_t == MASK_TOKEN_INDEX).sum(dim=-1).float()  # (B,)
        number_to_unmask = torch.binomial(
            count=masked_counts, prob=unmask_probs
        )  # (B,)

        # sample new residues for unmasked positions from pt_x1_probs
        # we'll assign these new residues to top 'number_to_unmask' positions
        unmasked_samples = torch.multinomial(
            pt_x1_probs.view(-1, num_states - 1), num_samples=1
        )
        unmasked_samples = unmasked_samples.view(num_batch, num_res)

        # Next lines are vectorized version of per-batch top-k replacement

        D_grid = (
            torch.arange(num_res, device=self._device)
            .unsqueeze(0)
            .expand(num_batch, -1)
        )  # (B, N)
        # 'mask1' is 1 where d < number_to_unmask[b], 0 otherwise
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        # if 0 to unmask, we set sorted_max_logprob_indices[:, 0] for all positions
        fallback = sorted_max_logprob_indices[:, 0].view(-1, 1).repeat(1, num_res)
        selected_indices = (mask1 * sorted_max_logprob_indices) + (
            (1.0 - mask1) * fallback
        )
        selected_indices = selected_indices.long()

        # 'mask2' indicates precisely which positions in aatypes_t we replace
        mask2 = torch.zeros((num_batch, num_res), device=self._device)
        mask2.scatter_(
            dim=1,
            index=selected_indices,
            src=torch.ones((num_batch, num_res), device=self._device),
        )
        # if zero are unmasked, we skip altogether
        none_unmasked_mask = (number_to_unmask == 0).unsqueeze(-1).float()
        mask2 *= 1.0 - none_unmasked_mask

        # 'mask2' is 1 where we replace with unmasked_samples, 0 otherwise
        aatypes_t = aatypes_t * (1.0 - mask2) + unmasked_samples * mask2

        # re-mask some positions as noise with probability (d_t * noise)
        re_mask_prob = d_t.view(-1, 1) * noise
        rand_vals = torch.rand(num_batch, num_res, device=self._device)
        re_mask_mask = (rand_vals < re_mask_prob).float()
        aatypes_t = aatypes_t * (1.0 - re_mask_mask) + (MASK_TOKEN_INDEX * re_mask_mask)

        return aatypes_t

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, S=21)
    ) -> torch.Tensor:
        """
        Perform a single Euler update step for the masking interpolant (S=21).
        If `purity_selection` is enabled, use purity-based unmasking; otherwise
        use masking-based rate update. Adds optional jump step if stochastic is enabled.
        """
        if potential is not None:
            assert (
                potential.shape == logits_1.shape
            ), f"Guidance logits shape {potential.shape} does not match logits_1 shape {logits_1.shape}"
            logits_1 = combine_logits([logits_1, potential])

        if self.cfg.purity_selection:
            aatypes_t = self._euler_step_purity(
                d_t=d_t, t=t, logits_1=logits_1, aatypes_t=aatypes_t
            )
        else:
            aatypes_t = self._euler_step_masking(
                d_t=d_t, t=t, logits_1=logits_1, aatypes_t=aatypes_t
            )

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            aatypes_t = self._aatypes_jump_step(
                d_t,
                t=t,
                logits_1=logits_1,
                aatypes_t=aatypes_t,
                stochasticity_scale=stochasticity_scale,
            )
        return aatypes_t
