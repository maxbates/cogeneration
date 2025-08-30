"""
Plot noise and drift curves for training and sampling, across translations, rotations, and amino acid types.

Plots the values defined in the Config for mock values.

AI-generated.
"""

import argparse
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from cogeneration.config.base import Config
from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX, NM_TO_ANG_SCALE
from cogeneration.data.fm.rotations import FlowMatcherRotations
from cogeneration.data.fm.translations import FlowMatcherTrans
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import uniform_so3
from cogeneration.data.rigid import batch_center_of_mass
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp


@dataclass
class TrainingNoiseResult:
    t: List[float] = field(default_factory=list)
    trans_noise_mean: List[float] = field(default_factory=list)
    trans_noise_std: List[float] = field(default_factory=list)
    rots_noise_mean_rad: List[float] = field(default_factory=list)
    rots_noise_std_rad: List[float] = field(default_factory=list)
    aa_noise_mean_frac: List[float] = field(default_factory=list)
    aa_noise_std_frac: List[float] = field(default_factory=list)
    trans_sigma: List[float] = field(default_factory=list)
    rots_sigma: List[float] = field(default_factory=list)
    aa_sigma: List[float] = field(default_factory=list)


@dataclass
class SamplingNoiseDriftResult:
    t: List[float] = field(default_factory=list)
    # noise
    trans_noise_mean: List[float] = field(default_factory=list)
    trans_noise_std: List[float] = field(default_factory=list)
    rots_noise_mean_rad: List[float] = field(default_factory=list)
    rots_noise_std_rad: List[float] = field(default_factory=list)
    aa_noise_mean_frac: List[float] = field(default_factory=list)
    aa_noise_std_frac: List[float] = field(default_factory=list)
    # aatypes noise breakdown
    aa_noise_unmask_frac: List[float] = field(default_factory=list)
    aa_noise_remask_frac: List[float] = field(default_factory=list)
    aa_noise_change_frac: List[float] = field(default_factory=list)
    # drift
    trans_drift_mean: List[float] = field(default_factory=list)
    trans_drift_std: List[float] = field(default_factory=list)
    rots_drift_mean_rad: List[float] = field(default_factory=list)
    rots_drift_std_rad: List[float] = field(default_factory=list)
    aa_drift_mean_frac: List[float] = field(default_factory=list)
    aa_drift_std_frac: List[float] = field(default_factory=list)
    # aatypes drift breakdown
    aa_drift_unmask_frac: List[float] = field(default_factory=list)
    aa_drift_remask_frac: List[float] = field(default_factory=list)
    aa_drift_change_frac: List[float] = field(default_factory=list)
    # ratios
    trans_noise_to_drift: List[float] = field(default_factory=list)
    rots_noise_to_drift: List[float] = field(default_factory=list)
    aa_noise_to_drift: List[float] = field(default_factory=list)
    # velocity-level ratios
    trans_noise_to_drift_vel: List[float] = field(default_factory=list)
    rots_noise_to_drift_vel: List[float] = field(default_factory=list)
    # theory sigma*sqrt(dt)
    trans_sigma_step: List[float] = field(default_factory=list)
    rots_sigma_step: List[float] = field(default_factory=list)
    # aatypes jump rates
    aa_jump_rate_theory: List[float] = field(default_factory=list)
    aa_jump_rate_empirical: List[float] = field(default_factory=list)
    aa_sigma_step: List[float] = field(default_factory=list)
    # aatypes theoretical component rates (1/time)
    aa_drift_rate_unmask: List[float] = field(default_factory=list)
    aa_drift_rate_remask: List[float] = field(default_factory=list)
    aa_drift_rate_change: List[float] = field(default_factory=list)
    aa_noise_rate_unmask: List[float] = field(default_factory=list)
    aa_noise_rate_remask: List[float] = field(default_factory=list)
    aa_noise_rate_change: List[float] = field(default_factory=list)
    # aatypes per-component scales (dimensionless)
    aa_scale_unmask: List[float] = field(default_factory=list)
    aa_scale_remask: List[float] = field(default_factory=list)
    aa_scale_change: List[float] = field(default_factory=list)
    # cumulative RSS curves
    trans_noise_cum_rss: List[float] = field(default_factory=list)
    rots_noise_cum_rss: List[float] = field(default_factory=list)
    trans_theory_cum_rss: List[float] = field(default_factory=list)
    rots_theory_cum_rss: List[float] = field(default_factory=list)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int):
    torch.manual_seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except Exception:
        pass


@torch.no_grad()
def prepare_base_inputs(
    interpolant: Interpolant,
    num_batch: int,
    num_res: int,
    trans1_scale: float = NM_TO_ANG_SCALE,
) -> Tuple[
    torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
]:
    device = interpolant._device

    res_mask = torch.ones(num_batch, num_res, device=device)
    diffuse_mask = torch.ones_like(res_mask)
    chain_idx = torch.zeros(num_batch, num_res, device=device).long()
    res_idx = torch.arange(num_res, device=device).view(1, -1).repeat(num_batch, 1)

    # trans_1: random centered structure (Å)
    trans_1 = torch.randn(num_batch, num_res, 3, device=device)
    trans_1 -= batch_center_of_mass(trans_1, mask=res_mask)[:, None]
    trans_1 = trans_1 * float(trans1_scale)

    # rotmats_1: uniform SO(3)
    rotmats_1 = uniform_so3(num_batch, num_res, device=device)

    # aatypes_1: random tokens in [0, num_tokens)
    aatypes_1 = torch.randint(
        low=0, high=interpolant.num_tokens, size=(num_batch, num_res), device=device
    )

    return res_mask, diffuse_mask, chain_idx, res_idx, trans_1, rotmats_1, aatypes_1


@torch.no_grad()
def measure_training_noise(
    interpolant: Interpolant,
    num_batch: int,
    num_res: int,
    num_timesteps: int,
    step_stride: int,
    trans1_scale: float,
) -> TrainingNoiseResult:
    device = interpolant._device
    res_mask, diffuse_mask, chain_idx, _, trans_1, rotmats_1, aatypes_1 = (
        prepare_base_inputs(interpolant, num_batch, num_res, trans1_scale)
    )

    ts = torch.linspace(interpolant.cfg.min_t, 1.0, num_timesteps, device=device)
    eval_indices = list(range(0, num_timesteps, step_stride))

    # calibration constants so theory matches units and aggregation used for empirical curves
    # translations: mean ||noise|| for one unit sigma
    cal_trans = interpolant._trans_noise(chain_idx=chain_idx, is_intermediate=True)
    cal_trans_mean_norm = torch.linalg.norm(cal_trans, dim=-1).mean().item()
    # rotations: mean angle (rad) for IGSO3 sigma=1
    rots_fm = FlowMatcherRotations(cfg=interpolant.cfg.rots)
    rots_fm.set_device(device)
    sigma_one = torch.ones((num_batch,), device=rots_fm.igso3.sigma_grid.device)
    cal_rot_noise = (
        rots_fm.igso3.sample(sigma_one, num_res)
        .to(device)
        .reshape(num_batch, num_res, 3, 3)
    )
    cal_rot_mean_angle = so3_utils.angle_from_rotmat(cal_rot_noise)[0].mean().item()

    # storage
    results = TrainingNoiseResult()

    # translations flow matcher
    trans_fm = FlowMatcherTrans(cfg=interpolant.cfg.trans)
    trans_fm.set_device(device)

    for i in eval_indices:
        t = ts[i]
        t_b = t.view(1).repeat(num_batch).view(num_batch, 1)

        # translations
        seed_all(12340 + i)
        trans_no = trans_fm.corrupt(
            trans_1=trans_1,
            t=t_b,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=0.0,
        )
        seed_all(12340 + i)
        trans_st = trans_fm.corrupt(
            trans_1=trans_1,
            t=t_b,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=1.0,
        )
        trans_delta = (trans_st - trans_no) * res_mask[..., None]
        trans_delta_norm = torch.linalg.norm(trans_delta, dim=-1)  # (B, N)
        results.trans_noise_mean.append(trans_delta_norm.mean().item())
        results.trans_noise_std.append(trans_delta_norm.std().item())

        # rotations: measure geodesic angle between with/without stochasticity
        seed_all(22340 + i)
        rot_no = rots_fm.corrupt(
            rotmats_1=rotmats_1,
            t=t_b,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=0.0,
        )
        seed_all(22340 + i)
        rot_st = rots_fm.corrupt(
            rotmats_1=rotmats_1,
            t=t_b,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=1.0,
        )
        # delta rotation = rot_st @ rot_no^T
        rot_delta = torch.matmul(rot_st, rot_no.transpose(-1, -2))
        angles, _, _ = so3_utils.angle_from_rotmat(rot_delta)
        results.rots_noise_mean_rad.append(angles.mean().item())
        results.rots_noise_std_rad.append(angles.std().item())

        # aatypes: fraction of positions changed by stochastic jump
        seed_all(32340 + i)
        aa_no = interpolant._corrupt_aatypes(
            aatypes_1=aatypes_1,
            t=t_b,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=0.0,
        )
        seed_all(32340 + i)
        aa_st = interpolant._corrupt_aatypes(
            aatypes_1=aatypes_1,
            t=t_b,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=1.0,
        )
        aa_changed = (aa_st != aa_no).float()
        results.aa_noise_mean_frac.append(aa_changed.mean().item())
        results.aa_noise_std_frac.append(aa_changed.float().std().item())

        # theoretical sigma(t)
        sigma_t_trans = interpolant.trans_fm._compute_sigma_t(
            t=t_b.squeeze(1),
            scale=interpolant.cfg.trans.stochastic_noise_intensity,
        )
        # rotations use schedule-transformed time for sigma
        if interpolant.cfg.rots.train_schedule.name == "linear":
            so3_t = t_b.squeeze(1)
        elif interpolant.cfg.rots.train_schedule.name == "exp":
            rate = interpolant.cfg.rots.exp_rate
            so3_t = (1 - torch.exp(-t_b.squeeze(1) * rate)) / (1 - math.exp(-rate))
        else:
            raise ValueError("Unknown rotations train schedule")
        sigma_t_rots = interpolant.rots_fm._compute_sigma_t(
            t=so3_t,
            scale=interpolant.cfg.rots.stochastic_noise_intensity,
        )
        sigma_t_aa = interpolant.aatypes_fm._compute_sigma_t(
            t=t_b.squeeze(1),
            scale=interpolant.cfg.aatypes.stochastic_noise_intensity,
        )

        # scale theory by calibration constants
        results.trans_sigma.append((sigma_t_trans.mean().item()) * cal_trans_mean_norm)
        results.rots_sigma.append((sigma_t_rots.mean().item()) * cal_rot_mean_angle)
        results.aa_sigma.append(sigma_t_aa.mean().item())
        results.t.append(t.item())

    return results


@torch.no_grad()
def measure_sampling_noise_and_drift(
    interpolant: Interpolant,
    num_batch: int,
    num_res: int,
    num_timesteps: int,
    step_stride: int,
    trans1_scale: float,
) -> SamplingNoiseDriftResult:
    device = interpolant._device
    res_mask, diffuse_mask, chain_idx, res_idx, trans_1, rotmats_1, aatypes_1 = (
        prepare_base_inputs(interpolant, num_batch, num_res, trans1_scale)
    )

    # initialize states at t=0 priors
    trans_fm = FlowMatcherTrans(cfg=interpolant.cfg.trans)
    trans_fm.set_device(device)
    trans_t = trans_fm.sample_base(chain_idx=chain_idx, is_intermediate=False)
    rots_fm = FlowMatcherRotations(cfg=interpolant.cfg.rots)
    rots_fm.set_device(device)
    rotmats_t = rots_fm.sample_base(res_mask=res_mask)
    aatypes_t = interpolant.aatypes_fm.sample_base(res_mask=res_mask)

    ts = torch.linspace(interpolant.cfg.min_t, 1.0, num_timesteps, device=device)

    # calibration constants: same as in training
    cal_trans = trans_fm.sample_base(chain_idx=chain_idx, is_intermediate=True)
    cal_trans_mean_norm = torch.linalg.norm(cal_trans, dim=-1).mean().item()
    cal_trans_mean_norm2 = torch.linalg.norm(cal_trans, dim=-1).square().mean().item()
    sigma_one = torch.ones((num_batch,), device=rots_fm.igso3.sigma_grid.device)
    cal_rot_noise = (
        rots_fm.igso3.sample(sigma_one, num_res)
        .to(device)
        .reshape(num_batch, num_res, 3, 3)
    )
    cal_rot_mean_angle = so3_utils.angle_from_rotmat(cal_rot_noise)[0].mean().item()
    # not used directly; rotations use per-step MC to capture nonlinearity

    out = SamplingNoiseDriftResult()

    # cumulative trackers for expected squared increments
    cum_trans_e2 = 0.0
    cum_rots_e2 = 0.0
    cum_theory_trans_sigma2 = 0.0
    cum_theory_rots_sigma2 = 0.0

    for i in range(0, num_timesteps - 1, step_stride):
        t1 = ts[i]
        t2 = ts[i + 1]
        dt = t2 - t1
        t1_f = float(t1.item())
        dt_f = float(dt.item())

        # ----- translations -----
        trans_next_drift = trans_fm.euler_step(
            d_t=dt,
            t=torch.ones(num_batch, device=device) * t1,
            trans_1=trans_1,
            trans_t=trans_t,
            chain_idx=chain_idx,
            stochasticity_scale=0.0,
            potential=None,
        )
        trans_vf = trans_fm.vector_field(t=t1, trans_1=trans_1, trans_t=trans_t)
        trans_drift_step = torch.linalg.norm(trans_vf * dt_f, dim=-1)
        out.trans_drift_mean.append(trans_drift_step.mean().item())
        out.trans_drift_std.append(trans_drift_step.std().item())

        seed_all(41340 + i)
        trans_next_noise = trans_fm.euler_step(
            d_t=dt,
            t=torch.ones(num_batch, device=device) * t1,
            trans_1=trans_1,
            trans_t=trans_t,
            chain_idx=chain_idx,
            stochasticity_scale=1.0,
            potential=None,
        )
        trans_noise_comp = trans_next_noise - (trans_t + trans_vf * dt_f)
        trans_noise_norm = torch.linalg.norm(trans_noise_comp, dim=-1)
        out.trans_noise_mean.append(trans_noise_norm.mean().item())
        out.trans_noise_std.append(trans_noise_norm.std().item())
        # accumulate mean squared norm per step
        cum_trans_e2 += trans_noise_norm.square().mean().item()
        out.trans_noise_cum_rss.append(math.sqrt(cum_trans_e2))
        # ratio noise/drift
        out.trans_noise_to_drift.append(
            (trans_noise_norm.mean().item()) / (trans_drift_step.mean().item() + 1e-8)
        )
        # velocity-level ratio
        trans_noise_vel = trans_noise_norm.mean().item() / max(math.sqrt(dt_f), 1e-12)
        trans_drift_vel = trans_drift_step.mean().item() / (dt_f + 1e-12)
        out.trans_noise_to_drift_vel.append(trans_noise_vel / (trans_drift_vel + 1e-12))

        # update state (drift-only) for next iteration
        trans_t = trans_next_drift

        # theory
        sigma_t_trans = interpolant.trans_fm._compute_sigma_t(
            t=torch.ones(num_batch, device=device) * t1_f,
            scale=interpolant.cfg.trans.stochastic_noise_intensity,
        ) * math.sqrt(dt_f)
        sig_step_trans = sigma_t_trans.mean().item()
        out.trans_sigma_step.append(sig_step_trans * cal_trans_mean_norm)
        # theory cumulative (sum sigma^2 dt scaled to expected squared norm)
        cum_theory_trans_sigma2 += (sig_step_trans**2) * cal_trans_mean_norm2
        out.trans_theory_cum_rss.append(math.sqrt(cum_theory_trans_sigma2))

        # ----- rotations -----
        rot_next_drift = rots_fm.euler_step(
            d_t=dt,
            t=t1,
            rotmats_1=rotmats_1,
            rotmats_t=rotmats_t,
            stochasticity_scale=0.0,
            potential=None,
        )
        # drift angle from current state
        delta_drift = torch.matmul(rot_next_drift, rotmats_t.transpose(-1, -2))
        drift_angles, _, _ = so3_utils.angle_from_rotmat(delta_drift)
        out.rots_drift_mean_rad.append(drift_angles.mean().item())
        out.rots_drift_std_rad.append(drift_angles.std().item())

        seed_all(51340 + i)
        rot_next_noise = rots_fm.euler_step(
            d_t=dt,
            t=t1,
            rotmats_1=rotmats_1,
            rotmats_t=rotmats_t,
            stochasticity_scale=1.0,
            potential=None,
        )
        # noise angle between noise and drift-only outputs
        delta_noise = torch.matmul(rot_next_noise, rot_next_drift.transpose(-1, -2))
        noise_angles, _, _ = so3_utils.angle_from_rotmat(delta_noise)
        out.rots_noise_mean_rad.append(noise_angles.mean().item())
        out.rots_noise_std_rad.append(noise_angles.std().item())
        cum_rots_e2 += noise_angles.square().mean().item()
        out.rots_noise_cum_rss.append(math.sqrt(cum_rots_e2))
        # ratio noise/drift
        out.rots_noise_to_drift.append(
            (noise_angles.mean().item()) / (drift_angles.mean().item() + 1e-8)
        )
        # velocity-level ratio (angles)
        noise_ang_vel = noise_angles.mean().item() / max(math.sqrt(dt_f), 1e-12)
        drift_ang_vel = drift_angles.mean().item() / (dt_f + 1e-12)
        out.rots_noise_to_drift_vel.append(noise_ang_vel / (drift_ang_vel + 1e-12))

        # update state (drift-only)
        rotmats_t = rot_next_drift

        # theory
        sigma_t_rots = interpolant.rots_fm._compute_sigma_t(
            t=torch.ones(num_batch, device=device) * t1_f,
            scale=interpolant.cfg.rots.stochastic_noise_intensity,
        ) * math.sqrt(dt_f)
        # Monte Carlo IGSO3 to estimate E[angle] and E[angle^2] for current sigma
        # vectorized per-batch sampling
        K = 32
        sig_vec = sigma_t_rots.to(rots_fm.igso3.sigma_grid.device)
        # sample K noises per residue and batch
        mc_noise = rots_fm.igso3.sample(sig_vec, num_res * K).to(device)
        mc_noise = mc_noise.reshape(num_batch, K, num_res, 3, 3)
        mc_angles, _, _ = so3_utils.angle_from_rotmat(mc_noise)
        e_angle = mc_angles.mean().item()
        e_angle2 = mc_angles.square().mean().item()
        out.rots_sigma_step.append(e_angle)  # step magnitude in radians
        cum_theory_rots_sigma2 += e_angle2
        out.rots_theory_cum_rss.append(math.sqrt(cum_theory_rots_sigma2))

        # ----- aatypes -----
        seed_all(61340 + i)
        aa_next_drift = interpolant._aatypes_euler_step(
            d_t=dt,
            t=t1,
            logits_1=torch.zeros(
                num_batch, num_res, interpolant.num_tokens, device=device
            ),
            aatypes_t=aatypes_t,
            stochasticity_scale=0.0,
            potential=None,
        )
        # drift fraction changed (relative to previous state)
        aa_drift_changed = (aa_next_drift != aatypes_t).float()
        out.aa_drift_mean_frac.append(aa_drift_changed.mean().item())
        out.aa_drift_std_frac.append(aa_drift_changed.std().item())

        # drift breakdown: unmask/remask/change
        prev_is_mask = aatypes_t == MASK_TOKEN_INDEX
        prev_is_aa = ~prev_is_mask
        drift_unmask = (prev_is_mask & (aa_next_drift != MASK_TOKEN_INDEX)).float()
        drift_remask = (prev_is_aa & (aa_next_drift == MASK_TOKEN_INDEX)).float()
        drift_change = (
            prev_is_aa
            & (aa_next_drift != MASK_TOKEN_INDEX)
            & (aa_next_drift != aatypes_t)
        ).float()
        out.aa_drift_unmask_frac.append(drift_unmask.mean().item())
        out.aa_drift_remask_frac.append(drift_remask.mean().item())
        out.aa_drift_change_frac.append(drift_change.mean().item())

        seed_all(61340 + i)
        aa_next_noise = interpolant._aatypes_euler_step(
            d_t=dt,
            t=t1,
            logits_1=torch.zeros(
                num_batch, num_res, interpolant.num_tokens, device=device
            ),
            aatypes_t=aatypes_t,
            stochasticity_scale=1.0,
            potential=None,
        )
        # noise-only fraction changed (difference between with and without extra CTMC jump)
        aa_noise_changed = (aa_next_noise != aa_next_drift).float()
        out.aa_noise_mean_frac.append(aa_noise_changed.mean().item())
        out.aa_noise_std_frac.append(aa_noise_changed.std().item())
        # noise breakdown: compare to drift baseline
        base_is_mask = aa_next_drift == MASK_TOKEN_INDEX
        base_is_aa = ~base_is_mask
        noise_unmask = (base_is_mask & (aa_next_noise != MASK_TOKEN_INDEX)).float()
        noise_remask = (base_is_aa & (aa_next_noise == MASK_TOKEN_INDEX)).float()
        noise_change = (
            base_is_aa
            & (aa_next_noise != MASK_TOKEN_INDEX)
            & (aa_next_noise != aa_next_drift)
        ).float()
        out.aa_noise_unmask_frac.append(noise_unmask.mean().item())
        out.aa_noise_remask_frac.append(noise_remask.mean().item())
        out.aa_noise_change_frac.append(noise_change.mean().item())
        # ratio noise/drift
        out.aa_noise_to_drift.append(
            (aa_noise_changed.mean().item()) / (aa_drift_changed.mean().item() + 1e-8)
        )

        # update state (drift-only)
        aatypes_t = aa_next_drift

        sigma_t_aa = interpolant.aatypes_fm._compute_sigma_t(
            t=torch.ones(num_batch, device=device) * t1_f,
            scale=interpolant.cfg.aatypes.stochastic_noise_intensity,
        ) * math.sqrt(dt_f)
        # Use CTMC jump expectation to form theory: E[fraction changed] = mean p_jump
        # replicate _aatype_jump_step rate construction with uniform logits
        # IMPORTANT: sigma_t here is NOT scaled by sqrt(dt); dt enters in the exp
        sigma_no_sqrt = interpolant.aatypes_fm._compute_sigma_t(
            t=torch.ones(num_batch, device=device) * t1_f,
            scale=interpolant.cfg.aatypes.stochastic_noise_intensity,
        )  # (B,)
        S = interpolant.num_tokens
        logits_uniform = torch.zeros(num_batch, num_res, S, device=device)
        prob_rows = torch.softmax(logits_uniform, dim=-1).clamp(min=1e-8)
        current_idx = aatypes_t.unsqueeze(-1).long()
        exit_rates = prob_rows.scatter(-1, current_idx, 0.0)
        exit_sums = exit_rates.sum(-1, keepdim=True)
        step_rates = exit_rates.clone()
        step_rates.scatter_(-1, current_idx, -exit_sums)
        # scale entire rate matrix by sigma_t (broadcast over residues and states)
        step_rates = step_rates * sigma_no_sqrt.view(-1, 1, 1)
        lam = -step_rates.gather(-1, current_idx).squeeze(-1)  # (B,N)
        p_jump = 1.0 - torch.exp(-lam * dt_f)
        out.aa_jump_rate_theory.append(p_jump.mean().item())
        out.aa_sigma_step.append(p_jump.mean().item())

        # empirical jump rate by calling jump-step directly
        aa_after_jump = interpolant.aatypes_fm._aatype_jump_step(
            d_t=dt,
            t=t1,
            logits_1=torch.zeros(
                num_batch, num_res, interpolant.num_tokens, device=device
            ),
            aatypes_t=aatypes_t,
            stochasticity_scale=1.0,
        )
        out.aa_jump_rate_empirical.append(
            (aa_after_jump != aatypes_t).float().mean().item()
        )

        # theoretical drift/noise rate components (mean over sites)
        logits_zeros = torch.zeros(
            num_batch, num_res, interpolant.num_tokens, device=device
        )
        rates_drift, _ = interpolant._aatypes_build_rates_drift(
            aatypes_t=aatypes_t, logits_1=logits_zeros, t=t1, potential=None
        )
        rates_noise = interpolant._aatypes_build_rates_noise(
            aatypes_t=aatypes_t, t=t1, stochasticity_scale=1.0
        )
        S = rates_drift.shape[-1]
        has_mask_col = S > MASK_TOKEN_INDEX
        aa_cols = torch.ones(S, dtype=torch.bool, device=device)
        if has_mask_col:
            aa_cols[MASK_TOKEN_INDEX] = False

        # masks relative to current state aatypes_t
        is_mask_now = aatypes_t == MASK_TOKEN_INDEX
        is_aa_now = ~is_mask_now

        # Drift
        if is_mask_now.any():
            drift_unmask_rate = (
                rates_drift[is_mask_now][..., aa_cols].sum(dim=-1).mean().item()
            )
        else:
            drift_unmask_rate = 0.0
        if has_mask_col and is_aa_now.any():
            drift_remask_rate = (
                rates_drift[is_aa_now][..., MASK_TOKEN_INDEX].mean().item()
            )
        else:
            drift_remask_rate = 0.0
        if is_aa_now.any():
            drift_change_rate = (
                rates_drift[is_aa_now][..., aa_cols].sum(dim=-1).mean().item()
            )
        else:
            drift_change_rate = 0.0
        out.aa_drift_rate_unmask.append(drift_unmask_rate)
        out.aa_drift_rate_remask.append(drift_remask_rate)
        out.aa_drift_rate_change.append(drift_change_rate)

        # Noise
        if is_mask_now.any():
            noise_unmask_rate = (
                rates_noise[is_mask_now][..., aa_cols].sum(dim=-1).mean().item()
            )
        else:
            noise_unmask_rate = 0.0
        if has_mask_col and is_aa_now.any():
            noise_remask_rate = (
                rates_noise[is_aa_now][..., MASK_TOKEN_INDEX].mean().item()
            )
        else:
            noise_remask_rate = 0.0
        if is_aa_now.any():
            noise_change_rate = (
                rates_noise[is_aa_now][..., aa_cols].sum(dim=-1).mean().item()
            )
        else:
            noise_change_rate = 0.0
        out.aa_noise_rate_unmask.append(noise_unmask_rate)
        out.aa_noise_rate_remask.append(noise_remask_rate)
        out.aa_noise_rate_change.append(noise_change_rate)

        # Per-component scales as a function of time (dimensionless)
        scale_change, scale_unmask, scale_remask = (
            interpolant._aatypes_component_scales(t=t1)
        )
        out.aa_scale_change.append(scale_change.mean().item())
        out.aa_scale_unmask.append(scale_unmask.mean().item())
        out.aa_scale_remask.append(scale_remask.mean().item())

        out.t.append(t1.item())

    return out


def plot_results(train_res: TrainingNoiseResult, samp_res: SamplingNoiseDriftResult):
    # Convert angles to degrees for readability
    rots_train_mean_deg = np.array(train_res.rots_noise_mean_rad) * (180.0 / math.pi)
    rots_train_std_deg = np.array(train_res.rots_noise_std_rad) * (180.0 / math.pi)
    rots_samp_mean_deg = np.array(samp_res.rots_noise_mean_rad) * (180.0 / math.pi)
    rots_samp_std_deg = np.array(samp_res.rots_noise_std_rad) * (180.0 / math.pi)
    rots_samp_drift_mean_deg = np.array(samp_res.rots_drift_mean_rad) * (
        180.0 / math.pi
    )
    rots_samp_drift_std_deg = np.array(samp_res.rots_drift_std_rad) * (180.0 / math.pi)

    # Normalize t to [0,1] so plots start at 0 and end at 1
    t_train = np.array(train_res.t, dtype=float)
    t_samp = np.array(samp_res.t, dtype=float)

    def _norm_t(t):
        if len(t) < 2:
            return t
        return (t - t[0]) / (t[-1] - t[0] + 1e-12)

    t_train_n = _norm_t(t_train)
    t_samp_n = _norm_t(t_samp)

    # Colors for left (training) and right (sampling) axes
    color_left = "#1f77b4"  # blue
    color_right1 = "#ff7f0e"  # orange (sampling noise/theory)
    color_right2 = "#2ca02c"  # green (sampling drift)

    # Two-column layout with an extra top row for sigma_t plots
    fig, axes = plt.subplots(5, 2, figsize=(14, 40), sharex="col")

    # Top row: sigma_t curves grouped by variable type
    # Left: AATypes (fraction) — training σ(t) vs sampling σ√dt
    ax = axes[0, 0]
    ax.set_title("AATypes σ(t): training vs sampling (fraction)")
    if len(train_res.aa_sigma) > 0:
        ax.plot(t_train_n, train_res.aa_sigma, label="aa train σ(t)", color=color_left)
    if len(samp_res.aa_sigma_step) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_sigma_step,
            linestyle="--",
            color=color_right1,
            label="aa samp σ√dt",
        )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Right: Translations [Å] (left axis) and Rotations [deg] (right axis) — training vs sampling
    ax = axes[0, 1]
    ax.set_title("Trans/Rot σ(t): training vs sampling")
    ax_r = ax.twinx()
    # Translations on left axis (Å)
    if len(train_res.trans_sigma) > 0:
        ax.plot(
            t_train_n,
            train_res.trans_sigma,
            label="trans train σ(t) [Å]",
            color=color_left,
        )
    if len(samp_res.trans_sigma_step) > 0:
        ax.plot(
            t_samp_n,
            samp_res.trans_sigma_step,
            linestyle="--",
            color=color_left,
            alpha=0.8,
            label="trans samp σ√dt [Å]",
        )
    ax.tick_params(axis="y", colors=color_left)
    # Rotations on right axis (degrees)
    if len(train_res.rots_sigma) > 0:
        ax_r.plot(
            t_train_n,
            np.array(train_res.rots_sigma) * (180.0 / math.pi),
            label="rots train σ(t) [deg]",
            color=color_right1,
        )
    if len(samp_res.rots_sigma_step) > 0:
        ax_r.plot(
            t_samp_n,
            np.array(samp_res.rots_sigma_step) * (180.0 / math.pi),
            linestyle="--",
            color=color_right1,
            alpha=0.8,
            label="rots samp σ√dt [deg]",
        )
    ax_r.tick_params(axis="y", colors=color_right1)
    # Legend combining both axes
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    # Right column, row 1: Translations (Å)
    ax = axes[1, 1]
    ax.set_title("Translations noise and drift (Å)")
    ax_r = ax.twinx()
    # training (left axis)
    ax.plot(
        t_train_n,
        train_res.trans_noise_mean,
        label="train noise mean [L]",
        color=color_left,
    )
    ax.fill_between(
        t_train_n,
        np.array(train_res.trans_noise_mean) - np.array(train_res.trans_noise_std),
        np.array(train_res.trans_noise_mean) + np.array(train_res.trans_noise_std),
        color=color_left,
        alpha=0.2,
    )
    ax.plot(
        t_train_n,
        train_res.trans_sigma,
        linestyle="--",
        color=color_left,
        alpha=0.7,
        label="theory σ(t) train [L]",
    )
    ax.tick_params(axis="y", colors=color_left)
    # sampling (right axis)
    ax_r.plot(
        t_samp_n,
        samp_res.trans_noise_mean,
        label="sample noise mean [R]",
        color=color_right1,
    )
    ax_r.fill_between(
        t_samp_n,
        np.array(samp_res.trans_noise_mean) - np.array(samp_res.trans_noise_std),
        np.array(samp_res.trans_noise_mean) + np.array(samp_res.trans_noise_std),
        color=color_right1,
        alpha=0.2,
    )
    ax_r.plot(
        t_samp_n,
        samp_res.trans_sigma_step,
        linestyle="--",
        color=color_right1,
        alpha=0.7,
        label="theory σ(t)√dt sample [R]",
    )
    ax_r.plot(
        t_samp_n,
        samp_res.trans_drift_mean,
        label="sample drift mean [R]",
        color=color_right2,
    )
    ax_r.tick_params(axis="y", colors=color_right1)
    # legends
    ax.grid(True, alpha=0.3)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    # Right column, row 2: Rotations (deg)
    ax = axes[2, 1]
    ax.set_title("Rotations noise and drift (degrees)")
    ax_r = ax.twinx()
    # training (left)
    ax.plot(
        t_train_n, rots_train_mean_deg, label="train noise mean [L]", color=color_left
    )
    ax.fill_between(
        t_train_n,
        rots_train_mean_deg - rots_train_std_deg,
        rots_train_mean_deg + rots_train_std_deg,
        color=color_left,
        alpha=0.2,
    )
    ax.plot(
        t_train_n,
        np.array(train_res.rots_sigma) * (180.0 / math.pi),
        linestyle="--",
        color=color_left,
        alpha=0.7,
        label="theory σ(t) train [L]",
    )
    ax.tick_params(axis="y", colors=color_left)
    # sampling (right)
    ax_r.plot(
        t_samp_n, rots_samp_mean_deg, label="sample noise mean [R]", color=color_right1
    )
    ax_r.fill_between(
        t_samp_n,
        rots_samp_mean_deg - rots_samp_std_deg,
        rots_samp_mean_deg + rots_samp_std_deg,
        color=color_right1,
        alpha=0.2,
    )
    ax_r.plot(
        t_samp_n,
        rots_samp_drift_mean_deg,
        label="sample drift mean [R]",
        color=color_right2,
    )
    ax_r.plot(
        t_samp_n,
        np.array(samp_res.rots_sigma_step) * (180.0 / math.pi),
        linestyle="--",
        color=color_right1,
        alpha=0.7,
        label="theory σ(t)√dt sample [R]",
    )
    ax_r.tick_params(axis="y", colors=color_right1)
    ax.grid(True, alpha=0.3)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    # Left column, row 1: AATypes (fraction)
    ax = axes[1, 0]
    ax.set_title("AATypes noise and drift (fraction changed)")
    ax_r = ax.twinx()
    # training (left)
    ax.plot(
        t_train_n,
        train_res.aa_noise_mean_frac,
        label="train noise mean [L]",
        color=color_left,
    )
    ax.fill_between(
        t_train_n,
        np.array(train_res.aa_noise_mean_frac) - np.array(train_res.aa_noise_std_frac),
        np.array(train_res.aa_noise_mean_frac) + np.array(train_res.aa_noise_std_frac),
        color=color_left,
        alpha=0.2,
    )
    ax.plot(
        t_train_n,
        train_res.aa_sigma,
        linestyle="--",
        color=color_left,
        alpha=0.7,
        label="theory σ(t) train [L]",
    )
    ax.tick_params(axis="y", colors=color_left)
    # sampling (right)
    ax_r.plot(
        t_samp_n,
        samp_res.aa_noise_mean_frac,
        label="sample noise mean [R]",
        color=color_right1,
    )
    ax_r.fill_between(
        t_samp_n,
        np.array(samp_res.aa_noise_mean_frac) - np.array(samp_res.aa_noise_std_frac),
        np.array(samp_res.aa_noise_mean_frac) + np.array(samp_res.aa_noise_std_frac),
        color=color_right1,
        alpha=0.2,
    )
    ax_r.plot(
        t_samp_n,
        samp_res.aa_drift_mean_frac,
        label="sample drift mean [R]",
        color=color_right2,
    )
    if len(samp_res.aa_jump_rate_theory) > 0:
        ax_r.plot(
            t_samp_n,
            samp_res.aa_jump_rate_theory,
            linestyle="--",
            color=color_right1,
            alpha=0.7,
            label="theory jump rate [R]",
        )
    if len(samp_res.aa_jump_rate_empirical) > 0:
        ax_r.plot(
            t_samp_n,
            samp_res.aa_jump_rate_empirical,
            linestyle=":",
            color=color_right1,
            alpha=0.9,
            label="empirical jump rate [R]",
        )
    ax_r.tick_params(axis="y", colors=color_right1)
    ax.grid(True, alpha=0.3)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")

    # Left column, row 2: AATypes per-step breakdown (fractions): drift (solid) vs noise (dashed)
    ax = axes[2, 0]
    ax.set_title("AATypes per-step transitions: drift (solid) vs noise (dashed)")
    c_unmask = "#1f77b4"
    c_remask = "#ff7f0e"
    c_change = "#2ca02c"
    if len(samp_res.aa_drift_unmask_frac) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_drift_unmask_frac,
            label="drift unmask",
            color=c_unmask,
        )
        ax.plot(
            t_samp_n,
            samp_res.aa_drift_remask_frac,
            label="drift remask",
            color=c_remask,
        )
        ax.plot(
            t_samp_n,
            samp_res.aa_drift_change_frac,
            label="drift change",
            color=c_change,
        )
    if len(samp_res.aa_noise_unmask_frac) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_noise_unmask_frac,
            linestyle="--",
            color=c_unmask,
            alpha=0.8,
            label="noise unmask",
        )
        ax.plot(
            t_samp_n,
            samp_res.aa_noise_remask_frac,
            linestyle="--",
            color=c_remask,
            alpha=0.8,
            label="noise remask",
        )
        ax.plot(
            t_samp_n,
            samp_res.aa_noise_change_frac,
            linestyle="--",
            color=c_change,
            alpha=0.8,
            label="noise change",
        )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Left column, row 3: AATypes component scales vs t (dimensionless)
    ax = axes[3, 0]
    ax.set_title("AATypes component scales vs t (dimensionless)")
    if len(samp_res.aa_scale_unmask) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_scale_unmask,
            color=c_unmask,
            label="unmask scale",
        )
    if len(samp_res.aa_scale_remask) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_scale_remask,
            color=c_remask,
            label="remask scale",
        )
    if len(samp_res.aa_scale_change) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_scale_change,
            color=c_change,
            label="change scale",
        )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Right column, row 3: Cumulative RSS reconciliation (sampling vs theory)
    ax = axes[3, 1]
    ax.set_title("Cumulative noise (RSS): trans [L] vs rots [R]")
    ax_r = ax.twinx()
    # translations on left
    if len(samp_res.trans_noise_cum_rss) > 0:
        ax.plot(
            t_samp_n,
            samp_res.trans_noise_cum_rss,
            label="trans RSS (sample) [L]",
            color=color_left,
        )
    if len(samp_res.trans_theory_cum_rss) > 0:
        ax.plot(
            t_samp_n,
            samp_res.trans_theory_cum_rss,
            label="trans RSS (theory) [L]",
            linestyle="--",
            color=color_left,
            alpha=0.7,
        )
    ax.tick_params(axis="y", colors=color_left)
    # rotations on right (degrees)
    if len(samp_res.rots_noise_cum_rss) > 0:
        ax_r.plot(
            t_samp_n,
            np.array(samp_res.rots_noise_cum_rss) * (180.0 / math.pi),
            label="rots RSS deg (sample) [R]",
            color=color_right1,
        )
    if len(samp_res.rots_theory_cum_rss) > 0:
        ax_r.plot(
            t_samp_n,
            np.array(samp_res.rots_theory_cum_rss) * (180.0 / math.pi),
            label="rots RSS deg (theory) [R]",
            linestyle="--",
            color=color_right1,
            alpha=0.7,
        )
    ax_r.tick_params(axis="y", colors=color_right1)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")
    ax.grid(True, alpha=0.3)

    # Left column, row 4: Noise-to-drift ratio for AATypes (step-level)
    ax = axes[4, 0]
    ax.set_title("AATypes noise-to-drift ratio (step)")
    if len(samp_res.aa_noise_to_drift) > 0:
        ax.plot(
            t_samp_n,
            samp_res.aa_noise_to_drift,
            label="aa step",
            color=color_right2,
        )
    ax.tick_params(axis="y")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Right column, row 4: Noise-to-drift ratios for Trans/Rot (step [L] vs velocity [R])
    ax = axes[4, 1]
    ax.set_title("Trans/Rot noise-to-drift: step [L] vs velocity [R]")
    ax_r = ax.twinx()
    if len(samp_res.trans_noise_to_drift) > 0:
        ax.plot(
            t_samp_n,
            samp_res.trans_noise_to_drift,
            label="trans step [L]",
            color=color_left,
        )
    if len(samp_res.rots_noise_to_drift) > 0:
        ax.plot(
            t_samp_n,
            samp_res.rots_noise_to_drift,
            label="rots step [L]",
            color=color_right1,
        )
    ax.tick_params(axis="y", colors=color_left)
    # velocity ratios on right
    if len(samp_res.trans_noise_to_drift_vel) > 0:
        ax_r.plot(
            t_samp_n,
            samp_res.trans_noise_to_drift_vel,
            linestyle="--",
            color=color_left,
            alpha=0.6,
            linewidth=3.0,
            zorder=3,
            label="trans vel [R]",
        )
    if len(samp_res.rots_noise_to_drift_vel) > 0:
        ax_r.plot(
            t_samp_n,
            samp_res.rots_noise_to_drift_vel,
            linestyle="--",
            color=color_right1,
            alpha=0.6,
            linewidth=3.0,
            zorder=3,
            label="rots vel [R]",
        )
    ax_r.tick_params(axis="y", colors=color_right1)
    lines_l, labels_l = ax.get_legend_handles_labels()
    lines_r, labels_r = ax_r.get_legend_handles_labels()
    ax.legend(lines_l + lines_r, labels_l + labels_r, loc="upper left")
    ax.grid(True, alpha=0.3)

    # X labels on bottom row for both columns
    axes[4, 0].set_xlabel("t (normalized)")
    axes[4, 1].set_xlabel("t (normalized)")

    # Set consistent x-limits for used primary axes
    used_axes = [
        axes[0, 0],
        axes[1, 0],
        axes[2, 0],
        axes[3, 0],
        axes[4, 0],
        axes[0, 1],
        axes[1, 1],
        axes[2, 1],
        axes[3, 1],
        axes[4, 1],
    ]
    for axx in used_axes:
        axx.set_xlim(0.0, 1.0)
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot stochastic noise scales for training and sampling."
    )
    parser.add_argument("--num_batch", type=int, default=10)
    parser.add_argument("--num_res", type=int, default=128)
    parser.add_argument(
        "--num_timesteps",
        type=int,
        default=None,
        help="Override number of timesteps; defaults to cfg.interpolant.sampling.num_timesteps",
    )
    parser.add_argument(
        "--step_stride", type=int, default=5, help="Evaluate every N steps"
    )
    parser.add_argument(
        "--trans1_scale",
        type=float,
        default=NM_TO_ANG_SCALE,
        help="Scale for synthetic trans_1 amplitude (Å)",
    )
    args = parser.parse_args()

    cfg = Config().interpolate()

    interpolant = Interpolant(cfg=cfg.interpolant)
    device = get_device()
    interpolant.set_device(device)

    num_timesteps = (
        args.num_timesteps or cfg.inference.interpolant.sampling.num_timesteps
    )

    train_res = measure_training_noise(
        interpolant=interpolant,
        num_batch=args.num_batch,
        num_res=args.num_res,
        num_timesteps=num_timesteps,
        step_stride=args.step_stride,
        trans1_scale=args.trans1_scale,
    )

    samp_res = measure_sampling_noise_and_drift(
        interpolant=interpolant,
        num_batch=args.num_batch,
        num_res=args.num_res,
        num_timesteps=num_timesteps,
        step_stride=args.step_stride,
        trans1_scale=args.trans1_scale,
    )

    plot_results(train_res, samp_res)


if __name__ == "__main__":
    main()
