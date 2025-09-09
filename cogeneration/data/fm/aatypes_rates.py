"""
WIP (i.e. not yet correct) rate-matrix formulation for CTMC jump aatypes flow matcher
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

from cogeneration.config.base import (
    InterpolantAATypesConfig,
    InterpolantAATypesInterpolantTypeEnum,
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


class FlowMatcherAATypesCTMC(FlowMatcherAATypes):
    """
    AAtypes flow matching using CTMC drift rate matrix + optional noise
    """

    def __init__(self, cfg: InterpolantAATypesConfig):
        self.cfg = cfg
        self._device: Optional[torch.device] = None

    def set_device(self, device: torch.device):
        self._device = device

    @property
    def num_tokens(self) -> int:
        return (
            21
            if self.cfg.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
            else 20
        )

    def sample_base(self, res_mask: torch.Tensor) -> torch.Tensor:
        num_batch, num_res = res_mask.shape
        if self.cfg.interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            return masked_categorical(num_batch, num_res, device=self._device)
        elif self.cfg.interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            return uniform_categorical(
                num_batch, num_res, num_tokens=self.num_tokens, device=self._device
            )
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.interpolant_type}"
            )

    def _aatypes_rate_schedule(
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
        tau = self._aatypes_schedule(t=t)

        sched = (1.0 + kappa * tau + eps) / (1.0 - tau + eps)
        sched = sched.clamp_min(0.0)

        return sched, tau

    def _aatypes_component_scales(
        self,
        t: torch.Tensor,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Time-dependent aatypes drift rate per-component scales

        Returns:
        change_scale (AA->AA')  : (B, 1, 1)  ~ cfg.change_rate * s_sched(t)
        unmask_scale (mask->AA) : (B, 1, 1)  ~ cfg.unmask_rate * s_sched(t)
        remask_scale (AA->MASK) : (B, 1, 1)  ~ cfg.remask_rate * 1/s_sched(t) - eps [->0 @ t=1]
        """
        t = t.to(self._device)

        # aggressively smooth out near t=1
        eps = 0.25
        sched, _ = self._aatypes_rate_schedule(t=t, eps=eps)
        sched = sched.view(-1, 1, 1)
        inv_sched = (1.0 / sched).view(-1, 1, 1)

        # multiply base component weight * schedule
        change_scale = self.cfg.change_rate * sched
        unmask_scale = self.cfg.unmask_rate * sched
        remask_scale = self.cfg.remask_rate * inv_sched - eps

        return (
            change_scale.clamp_min(0.0),
            unmask_scale.clamp_min(0.0),
            remask_scale.clamp_min(0.0),
        )

    def corrupt(
        self,
        aatypes_1: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: Union[torch.Tensor, float] = 1.0,  # (B,)
    ):
        """
        Corrupt AA residues from t=1 to t, using masking or uniform sampling.

        If `self.cfg.stochastic` is True, we also corrupt with unmasks/remasks/changes.
        AA -> AA' vs mask (change vs remask) proportion uses fixed split.
        """
        num_batch, num_res = res_mask.shape
        interpolant_type = self.cfg.interpolant_type

        # check shapes
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch,), f"t.shape: {t.shape} != (B,)"
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        # aatypes_t = aatypes_1 with masked fraction of residues based on t
        tau = self._aatypes_schedule(t=t)
        u = torch.rand(num_batch, num_res, device=self._device)
        corruption_mask = (u < (1.0 - tau)).int()
        aatypes_base = self.sample_base(res_mask=res_mask)
        aatypes_t = mask_blend_1d(aatypes_base, aatypes_1, corruption_mask)

        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            # For stochasticity, instead of specifying rates and using CTMC jump,
            # we simply introduce changes, dependent on t and interpolant type.

            sigma_t = self._compute_sigma_t(
                tau,  # (B,)
                scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
            )

            # probability a residue jumps
            p_jump = (
                sigma_t.unsqueeze(1).expand(num_batch, num_res).clamp(max=1.0)
            )  # (B, N)
            jump_mask = torch.rand(num_batch, num_res, device=self._device) < p_jump

            if jump_mask.any():
                # For uniform interpolant, convert AA -> AA' (not allowing self)
                if interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
                    assert self.num_tokens == 20
                    K = jump_mask.sum().item()
                    # disallow self and normalize probs
                    probs = torch.ones(K, self.num_tokens, device=self._device)
                    probs.scatter_(1, aatypes_t.long()[jump_mask].view(-1, 1), 0.0)
                    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    # pick and set new aatypes
                    new_aatypes = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    aatypes_t[jump_mask] = new_aatypes.to(aatypes_t.dtype)

                # For masking interpolant, additional unmask / remask / change, weighted by component scales
                elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
                    # unmask unneeded - any mask that is jumping will unmask
                    # pass t, not tau, will be scaled in function
                    change_scale, unmask_scale, remask_scale = (
                        self._aatypes_component_scales(t=t)
                    )
                    change_scale = change_scale.view(num_batch)
                    remask_scale = remask_scale.view(num_batch)

                    is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
                    is_aa = ~is_mask
                    aa_col_mask = torch.ones(
                        self.num_tokens, dtype=torch.bool, device=self._device
                    )
                    aa_col_mask[MASK_TOKEN_INDEX] = False

                    # mask -> AA uniformly
                    select_mask = is_mask & jump_mask
                    if select_mask.any():
                        K = select_mask.sum().item()
                        probs = torch.zeros(K, self.num_tokens, device=self._device)
                        probs[:, aa_col_mask] = 1.0 / aa_col_mask.sum()  # 1/20
                        new_states = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        aatypes_t[select_mask] = new_states.to(aatypes_t.dtype)

                    # AA -> AA' or mask (depending on interpolant type)
                    select_aa = is_aa & jump_mask
                    if select_aa.any():
                        # relative rates of change vs remasking are independent of drift ratio
                        # the propertion could be a function of t, but it can also just be a constant
                        prop_change = 0.5
                        select_change = (
                            torch.rand(num_batch, num_res, device=self._device)
                            < prop_change
                        ) & select_aa
                        select_remask = select_aa & (~select_change)

                        # AA->AA' uniformly (exclude self and MASK)
                        if select_change.any():
                            K = select_change.sum().item()
                            probs = torch.ones(K, self.num_tokens, device=self._device)
                            probs.scatter_(
                                1, aatypes_t.long()[select_change].view(-1, 1), 0.0
                            )
                            probs[:, MASK_TOKEN_INDEX] = 0.0
                            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(
                                1e-8
                            )
                            new_states = torch.multinomial(
                                probs, num_samples=1
                            ).squeeze(-1)
                            aatypes_t[select_change] = new_states.to(aatypes_t.dtype)

                        # AA->MASK (remask)
                        if select_remask.any():
                            aatypes_t[select_remask] = torch.as_tensor(
                                MASK_TOKEN_INDEX,
                                device=self._device,
                                dtype=aatypes_t.dtype,
                            )

                else:
                    raise ValueError(
                        f"Unknown aatypes interpolant type {self.cfg.interpolant_type}"
                    )

        # residues outside `res_mask` are set to mask regardless of aatype noise/interpolant strategy.
        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        # only corrupt residues in `diffuse_mask`
        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _aatypes_build_rates_drift(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        logits_1: torch.Tensor,  # (B, N, S)
        t: torch.Tensor,  # (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate drift rate vector from current state (B, N, S) and change probabilities (B, N, S).

        The probabilities:
        - are per-site perference over targets (no "stay" entry)
        - sum to 1
        - can be converted to Multiflow-style rate matrix by _regularize_step_probs() for categorical / euler discrete sampling

        The rates:
        - represent per-site off-diagonal exit rates with units 1/time
        - is a rate vector that can be used directly by CTMC jump sampling
        - are scaled per `t` schedule, and change rates specified in config
        - the "stay" rate is the negative sum of the other rates; stay is implicit

        In contrast to the Multiflow-style rate matrix construction, which regularizes the step change probabilities
        by setting the current state to the negative sum of the other rates (i.e. `d_t * rate`), CTMC jump uses one-jump
        hazard rate `1 - exp(-lambda * d_t)`, where `lambda` is the exit rate per site.
        They are approximately equal on the first order for low lambda and low `d_t`.
        One benefit, since we support stochastic paths, is that the rate matrices can be added, to perform a single CTMC jump.

        Purity gating should be applied to the returned rates (after adding noise).
        """
        B, N, S = logits_1.shape
        dev = logits_1.device
        interpolant_type = self.cfg.interpolant_type

        # combine model logits and potential logits
        if potential is not None:
            # Note that the potential logits is already the combination of all the potentials,
            # and has been centered. Combining the potential logits with the predicted logits
            # yields the same softmax probs as if they had been added all at once, but the logits
            # themselves differ by a row-wise constant. So, if in the future we are interested in
            # some logits-loss, the logits should be normalized, or use probabilities instead.
            drift_logits = combine_logits(logits=[logits_1, potential])
        else:
            drift_logits = logits_1

        # in masking, forbid mask as drift target
        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            assert S == 21
            drift_logits = drift_logits.clone()
            drift_logits[..., MASK_TOKEN_INDEX] = -1e9
        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            assert S == 20
            assert aatypes_t.lt(S).all()
        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        # compute probabilities from logits, scaled by temperature
        drift_temp = self.cfg.drift_temp
        drift_probs = torch.softmax(drift_logits / drift_temp, dim=-1)  # (B, N, S)

        # per-component scales as some function of t
        change_scale, unmask_scale, remask_scale = self._aatypes_component_scales(t)

        # masks for mask vs AA, depending on interpolant type
        aa_cols = torch.ones(S, dtype=torch.bool, device=dev)
        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            aa_cols[MASK_TOKEN_INDEX] = False
        is_mask = aatypes_t == MASK_TOKEN_INDEX
        is_aa = ~is_mask

        # uncertainty gating: for AA rows use 1 - p(current), for MASK rows set to 1
        # TODO consider scaling (1 - p(current)) ** 1/gamma (configurable) to sharpen
        if self.cfg.uncertainty_gating:
            p_current = (
                drift_probs.gather(-1, aatypes_t.long().unsqueeze(-1))
                .squeeze(-1)
                .clamp(0.0, 1.0)
            )  # (B, N)
            # (1 - p_current) ** gamma
            uncert = (1.0 - p_current) ** self.cfg.uncertainty_gating_gamma
            # mask to AA rows
            uncert = uncert * is_aa.float() + (1.0 * is_mask.float())  # (B, N)
        else:
            uncert = torch.ones_like(is_aa, dtype=torch.float, device=dev)  # (B, N)

        # off-diagonal step mass defined below depending on interpolant type + rates
        rates_off = torch.zeros_like(drift_probs)

        # for masked positions, all mass goes to unmasking
        # (implicitly, this must be masking interpolant)
        if is_mask.any():
            # disallow mask -> mask
            probs_aa = drift_probs.masked_fill(~aa_cols.view(1, 1, -1), 0.0)
            # normalize probs over AA columns only
            z = probs_aa.sum(-1, keepdim=True).clamp_min(1e-8)
            probs_aa = probs_aa / z
            # put unmask mass on AA columns
            rates_unmask = (
                unmask_scale * uncert.unsqueeze(-1) * probs_aa
            ) * is_mask.unsqueeze(-1).float()
            rates_off += rates_unmask

        # for AA positions, mass is split between change and remask
        if is_aa.any():
            # disallow -> self
            change_probs = drift_probs.clone()
            change_probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
            # disallow -> mask here (remasking handled separately below)
            if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
                change_probs[..., MASK_TOKEN_INDEX] = 0.0
            z = change_probs.sum(-1, keepdim=True).clamp_min(1e-8)
            change_probs = change_probs / z
            rates_change = (
                change_scale
                * uncert.unsqueeze(-1)
                * change_probs
                * is_aa.unsqueeze(-1).float()
            )  # (B, N, S)
            rates_off += rates_change

            # remasking has single mask target
            if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
                rates_remask = (remask_scale * uncert.unsqueeze(-1)).squeeze(
                    -1
                )  # (B, N)
                rates_off[..., MASK_TOKEN_INDEX] += rates_remask * is_aa.float()

        # convert to off-diagonal rates r_off for jump sampling
        # defensively ensure self column is zero
        rates_off.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)

        return rates_off, drift_probs

    def _aatypes_build_rates_noise(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        stochasticity_scale: torch.Tensor,  # (B,)
    ) -> torch.Tensor:
        """
        Build logits-free corruption-matching noise rate vector (B, N, S).

        - uniform: AA -> AA' uniformly over S-1 targets (excludes self)
        - masking: mask -> AA uniformly over 20 AAs; AA -> mask or AA' (excludes self) with fixed split

        per-row noise mass is sigma_t * stochastic_noise_intensity * stochasticity_scale

        Returns rates (in units 1/time) independent of d_t. Caller adds to drift rates and does a single CTMC jump.
        """
        B, N = aatypes_t.shape
        device = aatypes_t.device
        interpolant_type = self.cfg.interpolant_type

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            S = 21
            aa_cols = torch.ones(S, dtype=torch.bool, device=device)
            aa_cols[MASK_TOKEN_INDEX] = False
            aa_count = int(aa_cols.sum().item())  # 20
        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            S = 20
            assert aatypes_t.lt(S).all(), "unexpected mask token"
        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        # Initialize rates
        rates_noise = torch.zeros(B, N, S, device=device)

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        # if all entries are zero, skip
        if not (stochasticity_scale > 0).any():
            return rates_noise

        # sigma_t per batch (B,) scaled by config + stochasticity_scale
        sigma = self._compute_sigma_t(
            t=t,
            scale=self.cfg.stochastic_noise_intensity * stochasticity_scale,
        )
        sigma = sigma.clamp_min(0.0)  # (B,)

        # Broadcasts
        sigma_BN = sigma.view(B, 1).expand(B, N)  # (B, N)
        sigma_BN1 = sigma_BN.unsqueeze(-1)  # (B, N, 1)

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            # AA -> AA' uniformly over S-1 (excludes self)
            rates_noise += sigma_BN1 / (S - 1)  # (B, N, S)
            rates_noise.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)

        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
            is_aa = ~is_mask

            # MASK rows: split sigma_t uniformly across AA columns
            if is_mask.any():
                masked_view = rates_noise[is_mask]
                masked_view[..., aa_cols] = (sigma_BN[is_mask] / aa_count).unsqueeze(-1)
                rates_noise[is_mask] = masked_view

            # AA rows: split sigma between AA->AA' and AA->MASK
            if is_aa.any():
                prop_change = 0.5
                prop_remask = 1.0 - prop_change

                # AA -> mask (remask)
                aa_view = rates_noise[is_aa]
                aa_view[..., MASK_TOKEN_INDEX] = sigma_BN[is_aa] * prop_remask
                rates_noise[is_aa] = aa_view

                # AA -> AA' uniformly (exclude self and mask)
                change_probs = torch.zeros(B, N, S, device=device)  # (B, N, S)
                change_probs[
                    is_aa.unsqueeze(-1).expand(-1, -1, S) & aa_cols.view(1, 1, -1)
                ] = 1.0
                change_probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
                zc = change_probs.sum(-1, keepdim=True).clamp_min(1e-8)
                change_probs = change_probs / zc
                # lambda_change is per-row mass
                lambda_change = sigma_BN * prop_change * is_aa.float()  # (B, N)
                rates_noise += lambda_change.unsqueeze(-1) * change_probs

        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        return rates_noise

    def _aatypes_purity_gate(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        rates: torch.Tensor,  # (B, N, S) drift rates
        drift_probs: torch.Tensor,  # (B, N, S) drift softmax, used for scoring
        t: torch.Tensor,  # (B,)
        d_t: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """
        Build a purity gate (B, N, S) for the masking interpolant that prevents unmasking on
        masked rows not selected by purity in this step. This mask never touches AA rows
        or the mask column; it only zeros mask -> AA transitions for non-selected masked rows.

        The gate is built by:
        - score masked rows by confidence (max over AA probs)
        - compute per-row jump probabilities from drift
        - sample Bernoulli for masked rows, then sum -> K_b (Poisson–binomial)
        - select top-K masked rows by confidence; other masked rows are gated

        Use the returned gate to zero out combined (drift + noise)rates:
        ```
        purity_gate = self._aatypes_purity_gate(...)
        rates = rates.masked_fill(purity_gate, 0.0)
        ```
        """
        B, N, S = rates.shape
        device = rates.device
        assert S == 21, "Purity is only defined for masking (21 tokens)."

        # rows that are currently MASK
        is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
        masked_counts = is_mask.sum(dim=1)  # (B,)
        if masked_counts.max() == 0:
            return torch.zeros_like(rates, dtype=torch.bool)

        # mask for mask vs AA
        aa_cols = torch.ones(S, dtype=torch.bool, device=device)
        aa_cols[MASK_TOKEN_INDEX] = False

        # calculate exit prob per masked row this step to determine number to unmask
        # per-row confidence = max over AA columns of predicted probabilities
        # p_row: (B, N) with p_bi = 1 - exp(-lambda_bi * d_t)
        lam = rates.sum(dim=-1)  # (B, N)
        p_row = 1.0 - torch.exp(-lam * d_t)  # (B, N)
        p_row = p_row.clamp(0.0, 1.0)
        # sample independent Bernoulli for masked rows, then sum -> K_b (Poisson–binomial)
        u = torch.rand_like(p_row)
        unmask_draw = (u < p_row) & is_mask  # (B, N) 1 if that row would unmask
        number_to_unmask = unmask_draw.sum(dim=1).to(torch.long)  # (B,)

        # build gate where True = allowed to unmask
        # select top-K masked rows per batch (vectorized via ranks)
        # ranks: 0 for largest score, 1 for second, ...
        aa_max = drift_probs[..., aa_cols].amax(dim=-1)  # (B, N)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=aa_max.dtype)
        masked_scores = torch.where(
            is_mask, aa_max, neg_inf
        )  # (B, N), non-masked = -inf
        sort_idx = torch.argsort(masked_scores, dim=1, descending=True)  # (B, N)
        ranks = torch.empty_like(sort_idx)
        D_grid = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        ranks.scatter_(1, sort_idx, D_grid)
        gate = (ranks < number_to_unmask.view(-1, 1)) & is_mask  # (B, N)

        # zero mask->AA rates in rows not selected
        to_zero = ((~gate) & is_mask).unsqueeze(-1) & aa_cols.view(
            1, 1, -1
        )  # (B, N, S)

        return to_zero

    def _aatypes_regularize_rates(
        self,
        rates: torch.Tensor,  # (B, N, S)
        drift_probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B,)
        d_t: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """
        Regularize rates by capping them.

        Computes a per-row jump probability cap `pmax` and converts to a smooth, row-wise squash.
        Small rows are ~unchanged, large rows are soft-capped so still allow confident changes.
        """
        B, N, S = rates.shape
        device = rates.device
        t = t.to(device)

        # uncertainty only for AA rows
        is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
        p_current = (
            drift_probs.gather(-1, aatypes_t.long().unsqueeze(-1))
            .squeeze(-1)
            .clamp(0.0, 1.0)
        )
        uncert = (1.0 - p_current) * (~is_mask).float()  # (B, N)

        # simple per-step probability caps
        # mask: allow aggressive late unmasking (ramps to 0.6)
        pmax_mask = (
            torch.clamp(0.1 + 0.6 * t, 0.05, 0.6).view(B, 1).expand(B, N)
        )  # (B, N)
        # AA: allow flips mostly when uncertain & mid-trajectory
        pmax_aa = torch.clamp(
            0.4 * (t * (1 - t)).view(B, 1) * uncert, 0.0, 0.25
        )  # (B, N)

        # pmax depends on whether currently masked or AA
        # avoid 1.0 (would give inf rate)
        pmax = torch.where(is_mask, pmax_mask, pmax_aa).clamp(0.0, 0.999)

        # convert probability to rate cap
        # CTMC relation: p = 1 - exp(-lambda * d_t) -> lambda = -log(1-p) / d_t
        lambda_cap = -torch.log1p(-pmax) / d_t

        # soft-cap λ_eff = λ_cap * (1 - exp(-λ_raw / λ_cap))
        # monotone, identity for small rows, bounded for large rows
        # total row hazard λ_raw = sum_j r_ij (off-diagonal rates)
        lambda_raw = rates.sum(dim=-1)
        # λ_eff = λ_cap * (1 - exp(-λ_raw / λ_cap))
        lambda_eff = lambda_cap * (1.0 - torch.exp(-lambda_raw / (lambda_cap + 1e-6)))

        # scale = λ_eff / λ_raw preserves destination mix but limits movement per step
        scale = (lambda_eff / (lambda_raw + 1e-6)).unsqueeze(-1)  # (B, N, 1)

        return rates * scale

    def _aatypes_ctmc_jump(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        rates: torch.Tensor,  # (B, N, S)
        d_t: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """
        Single CTMC jump step using rate matrix (i.e. rates from current state), scaled by d_t.

        rates (hazard densities) are functions of t with units 1/time.
        per-step jump probabilities are the rates multiplied by d_t:
        p_jump = 1.0 - exp(-lambda_step * d_t)
        where lambda_step = sum rates at that position
        (and so sum rates -> >1 => p_jump -> >d_t)
        """
        B, N = aatypes_t.shape
        S = rates.shape[-1]
        device = rates.device

        # decide whether each residue jumps during d_t
        lambda_step = rates.sum(dim=-1)  # exit rate per site (B, N)
        p_jump = 1.0 - torch.exp(-lambda_step * d_t)  # (B, N)
        jump_mask = torch.rand_like(p_jump) < p_jump  # (B, N)
        if not jump_mask.any():
            return aatypes_t

        # sample new aa for jumped residues
        jump_rates = rates.clone()
        current_aatypes_idx = aatypes_t.long().unsqueeze(-1)
        jump_rates.scatter_(-1, current_aatypes_idx, 0.0)  # zero out current col
        # normalize, conditional over targets
        jump_probs = jump_rates / lambda_step.clamp_min(1e-8).unsqueeze(-1)

        # sample new aa only for residues that jump
        new_states = torch.multinomial(jump_probs[jump_mask].reshape(-1, S), 1).squeeze(
            -1
        )
        new_aatypes = aatypes_t.clone()
        new_aatypes[jump_mask] = new_states.to(aatypes_t.dtype)
        return new_aatypes

    def euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # (B,)
        logits_1: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: torch.Tensor,  # (B,)
        potential: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # get drift rates + probs
        rates_drift, drift_probs = self._aatypes_build_rates_drift(
            aatypes_t=aatypes_t, logits_1=logits_1, t=t, potential=potential
        )

        # If purity sampling is enabled, not only do we avoid AA -> AA' transitions,
        # but also limit rates to top-K masked sites by ranked confidence
        # TODO - match Multiflow behavior of guaranteeing jump?
        if (
            self.cfg.purity_selection
            and self.cfg.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            # Purity is determined by the drift rates
            purity_gate = self._aatypes_purity_gate(
                aatypes_t=aatypes_t,
                rates=rates_drift,
                drift_probs=drift_probs,
                t=t,
                d_t=d_t,
            )
            rates_drift = rates_drift.masked_fill(purity_gate, 0.0)

        # soft-cap regularize rates to prevent blowups as t->1
        rates_drift = self._aatypes_regularize_rates(
            rates=rates_drift,
            drift_probs=drift_probs,
            aatypes_t=aatypes_t,
            t=t,
            d_t=d_t,
        )

        stochasticity_scale = self._stochasticity_scale_tensor(
            scale=stochasticity_scale, t=t
        )  # (B,)

        # optionally add logits-free noise
        rates_noise = torch.zeros_like(rates_drift)
        if (
            self.cfg.stochastic
            and self.cfg.stochastic_noise_intensity > 0.0
            and (stochasticity_scale > 0).any()
        ):
            rates_noise = self._aatypes_build_rates_noise(
                aatypes_t=aatypes_t, t=t, stochasticity_scale=stochasticity_scale
            )

        rates = rates_drift + rates_noise

        # single CTMC jump using combined rates
        aatypes_next = self._aatypes_ctmc_jump(
            aatypes_t=aatypes_t, rates=rates, d_t=d_t
        )

        return aatypes_next
