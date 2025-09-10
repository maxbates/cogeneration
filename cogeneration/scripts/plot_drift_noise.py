import math
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm

from cogeneration.config.base import Config
from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.interpolant import Interpolant
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp


def get_device() -> torch.device:
    """return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_all(seed: int):
    """seed python, numpy, and torch for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


@torch.no_grad()
def prepare_base_inputs(
    interpolant: Interpolant,
    num_batch: int,
    num_res: int,
):
    """prepare masks and sample base distributions for metrics."""
    device = interpolant._device

    res_mask = torch.ones(num_batch, num_res, device=device)
    diffuse_mask = torch.ones_like(res_mask)
    chain_idx = torch.zeros(num_batch, num_res, device=device).long()
    res_idx = torch.arange(num_res, device=device).view(1, -1).repeat(num_batch, 1)

    # trans_1 / rotmats_1 / aatypes_1: use flow matchers base distributions
    trans_1 = interpolant.trans_fm.sample_base(
        chain_idx=chain_idx, is_intermediate=True
    )
    rotmats_1 = interpolant.rots_fm.sample_base(res_mask=res_mask)
    aatypes_1 = interpolant.aatypes_fm.sample_base(res_mask=res_mask)

    return res_mask, diffuse_mask, chain_idx, res_idx, trans_1, rotmats_1, aatypes_1


@dataclass
class DomainResults:
    """per-domain time series for plotting and comparison."""

    t: List[float] = field(default_factory=list)
    theory_train: List[float] = field(default_factory=list)
    theory_sample: List[float] = field(default_factory=list)
    corr_noise_mean: List[float] = field(default_factory=list)
    corr_noise_std: List[float] = field(default_factory=list)
    samp_noise_mean: List[float] = field(default_factory=list)
    samp_noise_std: List[float] = field(default_factory=list)
    samp_drift_mean: List[float] = field(default_factory=list)
    samp_drift_std: List[float] = field(default_factory=list)
    step_ratio: List[float] = field(default_factory=list)
    vel_ratio: List[float] = field(default_factory=list)
    # theoretical drift and ratios (calibrated to first block)
    theory_drift_step: List[float] = field(default_factory=list)
    theory_drift_vel: List[float] = field(default_factory=list)
    step_ratio_theory: List[float] = field(default_factory=list)
    vel_ratio_theory: List[float] = field(default_factory=list)
    samp_cum_rss: List[float] = field(default_factory=list)
    theory_cum_rss: List[float] = field(default_factory=list)
    steps: List[float] = field(default_factory=list)


@dataclass
class AllResults:
    """aggregated results across translations, rotations, torsions, and aatypes."""

    trans: DomainResults = field(default_factory=DomainResults)
    rots: DomainResults = field(default_factory=DomainResults)
    aatypes: DomainResults = field(default_factory=DomainResults)
    tors: DomainResults = field(default_factory=DomainResults)


@dataclass
class MetricSnapshot:
    """normalized per-domain sampling metrics for one time block."""

    noise_mean: float
    noise_std: float
    drift_mean: float
    drift_std: float
    theory_sample: float
    step_ratio: float
    vel_ratio: float
    # per-block average theoretical scales (pre-calibration)
    drift_step_scale: float
    drift_vel_scale: float
    mean_dt: float


@dataclass
class MetricAccum:
    """accumulates per-step metrics within a block for one domain."""

    noise_mean: float = 0.0
    noise_std: float = 0.0
    drift_mean: float = 0.0
    drift_std: float = 0.0
    sigma_step: float = 0.0
    noise_to_drift: float = 0.0
    noise_to_drift_vel: float = 0.0
    drift_step_scale: float = 0.0
    drift_vel_scale: float = 0.0
    dt_sum: float = 0.0

    def add_noise(self, mean: float, std: float) -> None:
        self.noise_mean += mean
        self.noise_std += std

    def add_drift(self, mean: float, std: float) -> None:
        self.drift_mean += mean
        self.drift_std += std

    def add_sigma(self, value: float) -> None:
        self.sigma_step += value

    def add_step_ratio(self, ratio: float) -> None:
        self.noise_to_drift += ratio

    def add_vel_ratio(self, ratio: float) -> None:
        self.noise_to_drift_vel += ratio

    def add_theory_drift_scales(
        self, step_scale: float, vel_scale: float, dt_value: float
    ) -> None:
        self.drift_step_scale += step_scale
        self.drift_vel_scale += vel_scale
        self.dt_sum += dt_value

    def snapshot(self, c: float) -> MetricSnapshot:
        return MetricSnapshot(
            noise_mean=self.noise_mean / c,
            noise_std=self.noise_std / c,
            drift_mean=self.drift_mean / c,
            drift_std=self.drift_std / c,
            theory_sample=self.sigma_step / c,
            step_ratio=self.noise_to_drift / c,
            vel_ratio=self.noise_to_drift_vel / c,
            drift_step_scale=self.drift_step_scale / c,
            drift_vel_scale=self.drift_vel_scale / c,
            mean_dt=(self.dt_sum / c) if c > 0 else 0.0,
        )


@dataclass
class CumulativeRSS:
    """cumulative root-sum-of-squares (second-moment sums) up to the current sampling time."""

    trans_emp: float
    trans_theory: float
    rots_emp: float
    rots_theory: float
    tors_emp: float
    tors_theory: float


@dataclass
class DriftCalibration:
    """per-domain constants mapping theoretical drift scales to empirical magnitudes."""

    trans: Optional[float] = None
    rots: Optional[float] = None
    tors: Optional[float] = None
    aatypes: Optional[float] = None


@dataclass
class SamplingBlock:
    """accumulates per-step sampling statistics over a stride and flushes normalized snapshots."""

    trans: MetricAccum = field(default_factory=MetricAccum)
    rots: MetricAccum = field(default_factory=MetricAccum)
    tors: MetricAccum = field(default_factory=MetricAccum)
    aatypes: MetricAccum = field(default_factory=MetricAccum)

    count: int = 0
    t: Optional[float] = None

    def _flush_domain(
        self,
        res: DomainResults,
        snap: MetricSnapshot,
        cum_emp: Optional[float],
        cum_theory: Optional[float],
        drift_cal_const: Optional[float],
        include_vel: bool,
        include_cum: bool,
        t_value: float,
        steps: float,
    ) -> float:
        """append a normalized snapshot into domain results and update calibration."""
        res.samp_drift_mean.append(snap.drift_mean)
        res.samp_drift_std.append(snap.drift_std)
        res.samp_noise_mean.append(snap.noise_mean)
        res.samp_noise_std.append(snap.noise_std)
        res.theory_sample.append(snap.theory_sample)
        # Empirical ratios should reflect the plotted mean noise/drift, not
        # the average of per-step ratios. Compute from block-averaged means.
        step_ratio_emp = snap.noise_mean / max(snap.drift_mean, 1e-12)
        res.step_ratio.append(step_ratio_emp)
        if include_vel:
            noise_vel_emp = snap.noise_mean / max(math.sqrt(snap.mean_dt), 1e-12)
            drift_vel_emp = snap.drift_mean / max(snap.mean_dt, 1e-12)
            vel_ratio_emp = noise_vel_emp / max(drift_vel_emp, 1e-12)
            res.vel_ratio.append(vel_ratio_emp)
        res.steps.append(steps)

        if drift_cal_const is None and (t_value >= 0.5) and snap.drift_step_scale > 0:
            drift_cal_const = snap.drift_mean / max(snap.drift_step_scale, 1e-12)

        const = drift_cal_const
        if const is None:
            th_step = float("nan")
            th_vel = float("nan")
            res.theory_drift_step.append(th_step)
            if include_vel:
                res.theory_drift_vel.append(th_vel)
            res.step_ratio_theory.append(float("nan"))
            if include_vel:
                res.vel_ratio_theory.append(float("nan"))
        else:
            th_step = const * snap.drift_step_scale
            th_vel = const * snap.drift_vel_scale
            res.theory_drift_step.append(th_step)
            if include_vel:
                res.theory_drift_vel.append(th_vel)
            res.step_ratio_theory.append(snap.theory_sample / max(th_step, 1e-12))
            if include_vel:
                th_noise_vel = snap.theory_sample / max(math.sqrt(snap.mean_dt), 1e-12)
                res.vel_ratio_theory.append(th_noise_vel / max(th_vel, 1e-12))

        if include_cum:
            res.samp_cum_rss.append(math.sqrt(cum_emp))
            res.theory_cum_rss.append(math.sqrt(cum_theory))

        return drift_cal_const

    def flush_into(
        self, out: AllResults, c: float, t_value: float, cum: CumulativeRSS, drift_cal
    ) -> None:
        """normalize accumulated metrics and append per-domain results for this block."""
        # append time for all domains
        out.trans.t.append(t_value)
        out.rots.t.append(t_value)
        out.tors.t.append(t_value)
        out.aatypes.t.append(t_value)

        # translations
        snap = self.trans.snapshot(c)
        drift_cal.trans = self._flush_domain(
            res=out.trans,
            snap=snap,
            cum_emp=cum.trans_emp,
            cum_theory=cum.trans_theory,
            drift_cal_const=drift_cal.trans,
            include_vel=True,
            include_cum=True,
            t_value=t_value,
            steps=c,
        )

        # rotations
        snap = self.rots.snapshot(c)
        drift_cal.rots = self._flush_domain(
            res=out.rots,
            snap=snap,
            cum_emp=cum.rots_emp,
            cum_theory=cum.rots_theory,
            drift_cal_const=drift_cal.rots,
            include_vel=True,
            include_cum=True,
            t_value=t_value,
            steps=c,
        )

        # torsions
        snap = self.tors.snapshot(c)
        drift_cal.tors = self._flush_domain(
            res=out.tors,
            snap=snap,
            cum_emp=cum.tors_emp,
            cum_theory=cum.tors_theory,
            drift_cal_const=drift_cal.tors,
            include_vel=True,
            include_cum=True,
            t_value=t_value,
            steps=c,
        )

        # aatypes (no vel-based quantities, no cumulative RSS tracked)
        snap = self.aatypes.snapshot(c)
        drift_cal.aatypes = self._flush_domain(
            res=out.aatypes,
            snap=snap,
            cum_emp=None,
            cum_theory=None,
            drift_cal_const=drift_cal.aatypes,
            include_vel=False,
            include_cum=False,
            t_value=t_value,
            steps=c,
        )


def _angles_from_rotmats(mats: torch.Tensor) -> torch.Tensor:
    """return geodesic angles from rotation matrices."""
    ang, _, _ = so3_utils.angle_from_rotmat(mats)
    return ang


class TrainingMeasurer:
    """compute per-domain training corruption metrics and theory scales."""

    def __init__(self, cfg: Config, interpolant: Interpolant) -> None:
        """store config and interpolant for measuring training metrics."""
        self.cfg = cfg
        self.interpolant = interpolant

    @torch.no_grad()
    def measure(self) -> AllResults:
        """measure training-theory scales and empirical corruption noise per domain."""
        num_batch = self.cfg.noise_plot.num_batch
        num_res = self.cfg.noise_plot.num_res
        num_timesteps = self.cfg.inference.interpolant.sampling.num_timesteps
        step_stride = self.cfg.noise_plot.step_stride
        device = self.interpolant._device
        res_mask, diffuse_mask, chain_idx, _, trans_1, rotmats_1, aatypes_1 = (
            prepare_base_inputs(self.interpolant, num_batch, num_res)
        )
        torsions_1 = self.interpolant.torsions_fm.sample_base(res_mask=res_mask)

        ts = torch.linspace(
            self.interpolant.cfg.min_t, 1.0, num_timesteps, device=device
        )
        eval_idx = range(0, num_timesteps, max(1, step_stride))

        out = AllResults()

        for i in tqdm(
            eval_idx, total=len(eval_idx), desc="Training metrics", leave=False
        ):
            t_scalar = ts[i]
            t = torch.ones(num_batch, device=device) * t_scalar

            self._measure_translations_train(
                i=i,
                t=t,
                chain_idx=chain_idx,
                trans_1=trans_1,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                out=out,
            )
            self._measure_rotations_train(
                i=i,
                t=t,
                rotmats_1=rotmats_1,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                out=out,
            )
            self._measure_aatypes_train(
                i=i,
                t=t,
                aatypes_1=aatypes_1,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                out=out,
            )
            self._measure_torsions_train(
                i=i,
                t=t,
                torsions_1=torsions_1,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                out=out,
            )

            for dom in (out.trans, out.rots, out.aatypes, out.tors):
                dom.t.append(t_scalar.item())

        return out

    def _measure_translations_train(
        self,
        i: int,
        t: torch.Tensor,
        chain_idx: torch.Tensor,
        trans_1: torch.Tensor,
        res_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        out: AllResults,
    ) -> None:
        """measure translation corruption noise and theory at time t."""
        cal_trans = self.interpolant.trans_fm.sample_base(
            chain_idx=chain_idx, is_intermediate=True
        )
        cal_trans_mean_norm = torch.linalg.norm(cal_trans, dim=-1).mean().item()
        seed_all(12000 + i)
        trans_no = self.interpolant.trans_fm.corrupt(
            trans_1=trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=torch.zeros_like(t),
        )
        seed_all(12000 + i)
        trans_st = self.interpolant.trans_fm.corrupt(
            trans_1=trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=torch.ones_like(t),
        )
        d = (trans_st - trans_no) * res_mask[..., None]
        dnorm = torch.linalg.norm(d, dim=-1)
        out.trans.corr_noise_mean.append(dnorm.mean().item())
        out.trans.corr_noise_std.append(dnorm.std().item())
        sig = (
            self.interpolant.trans_fm._compute_sigma_t(
                t=self.interpolant.trans_fm.time_training(t),
                scale=self.interpolant.cfg.trans.stochastic_noise_intensity,
            )
            .mean()
            .item()
        )
        out.trans.theory_train.append(sig * cal_trans_mean_norm)

    def _measure_rotations_train(
        self,
        i: int,
        t: torch.Tensor,
        rotmats_1: torch.Tensor,
        res_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        out: AllResults,
    ) -> None:
        """measure rotation corruption noise and theory at time t."""
        sigma_one = torch.ones(
            (res_mask.shape[0],),
            device=self.interpolant.rots_fm.igso3.sigma_grid.device,
        )
        cal_rot_noise = (
            self.interpolant.rots_fm.igso3.sample(sigma_one, res_mask.shape[1])
            .to(t.device)
            .reshape(res_mask.shape[0], res_mask.shape[1], 3, 3)
        )
        cal_rot_mean_angle = _angles_from_rotmats(cal_rot_noise).mean().item()
        seed_all(22000 + i)
        rot_no = self.interpolant.rots_fm.corrupt(
            rotmats_1=rotmats_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.zeros_like(t),
        )
        seed_all(22000 + i)
        rot_st = self.interpolant.rots_fm.corrupt(
            rotmats_1=rotmats_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.ones_like(t),
        )
        delta = torch.matmul(rot_st, rot_no.transpose(-1, -2))
        ang = _angles_from_rotmats(delta)
        out.rots.corr_noise_mean.append(ang.mean().item())
        out.rots.corr_noise_std.append(ang.std().item())
        sig_r = (
            self.interpolant.rots_fm._compute_sigma_t(
                t=self.interpolant.rots_fm.time_training(t),
                scale=self.interpolant.cfg.rots.stochastic_noise_intensity,
            )
            .mean()
            .item()
        )
        out.rots.theory_train.append(sig_r * cal_rot_mean_angle)

    def _measure_aatypes_train(
        self,
        i: int,
        t: torch.Tensor,
        aatypes_1: torch.Tensor,
        res_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        out: AllResults,
    ) -> None:
        """measure aatypes corruption fraction and theory at time t."""
        seed_all(32000 + i)
        aa_no = self.interpolant.aatypes_fm.corrupt(
            aatypes_1=aatypes_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.zeros_like(t),
        )
        seed_all(32000 + i)
        aa_st = self.interpolant.aatypes_fm.corrupt(
            aatypes_1=aatypes_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.ones_like(t),
        )
        frac = (aa_st != aa_no).float().mean().item()
        out.aatypes.corr_noise_mean.append(frac)
        out.aatypes.corr_noise_std.append(0.0)
        tau = self.interpolant.aatypes_fm.time_training(t)
        rho = (
            self.interpolant.aatypes_fm._cumulative_hazard_rho(
                t=tau,
                scale=torch.full_like(
                    tau, float(self.interpolant.cfg.aatypes.stochastic_noise_intensity)
                ),
            )
            .mean()
            .item()
        )
        out.aatypes.theory_train.append(rho)

    def _measure_torsions_train(
        self,
        i: int,
        t: torch.Tensor,
        torsions_1: torch.Tensor,
        res_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        out: AllResults,
    ) -> None:
        """measure torsion corruption noise and theory at time t."""
        seed_all(42000 + i)
        tor_no = self.interpolant.torsions_fm.corrupt(
            torsions_1=torsions_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.zeros_like(t),
        )
        seed_all(42000 + i)
        tor_st = self.interpolant.torsions_fm.corrupt(
            torsions_1=torsions_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.ones_like(t),
        )
        ang_no = torch.atan2(tor_no[..., 0], tor_no[..., 1])
        ang_st = torch.atan2(tor_st[..., 0], tor_st[..., 1])
        da = torch.atan2(torch.sin(ang_st - ang_no), torch.cos(ang_st - ang_no))
        out.tors.corr_noise_mean.append(da.abs().mean().item())
        out.tors.corr_noise_std.append(da.std().item())
        sig_tor = (
            self.interpolant.torsions_fm._compute_sigma_t(
                t=t, scale=self.interpolant.cfg.torsions.stochastic_noise_intensity
            )
            .mean()
            .item()
        )
        out.tors.theory_train.append(sig_tor)


class SamplingMeasurer:
    """compute per-domain sampling noise/drift metrics and theory references."""

    def __init__(self, cfg: Config, interpolant: Interpolant) -> None:
        """store config and interpolant for measuring sampling metrics."""
        self.cfg = cfg
        self.interpolant = interpolant

    @torch.no_grad()
    def measure(self) -> AllResults:
        """measure empirical noise and drift during sampling and compute theory references."""
        num_batch = self.cfg.noise_plot.num_batch
        num_res = self.cfg.noise_plot.num_res
        num_timesteps = self.cfg.inference.interpolant.sampling.num_timesteps
        step_stride = self.cfg.noise_plot.step_stride
        device = self.interpolant._device
        res_mask, diffuse_mask, chain_idx, res_idx, trans_1, rotmats_1, aatypes_1 = (
            prepare_base_inputs(self.interpolant, num_batch, num_res)
        )

        trans_fm = self.interpolant.trans_fm
        rots_fm = self.interpolant.rots_fm
        torsions_fm = self.interpolant.torsions_fm
        aatypes_fm = self.interpolant.aatypes_fm

        ts = torch.linspace(
            self.interpolant.cfg.min_t, 1.0, num_timesteps, device=device
        )

        # calibration is computed within per-domain step functions

        out = AllResults()

        cum_trans_e2 = 0.0
        cum_rots_e2 = 0.0
        cum_tors_e2 = 0.0
        cum_theory_trans_sigma2 = 0.0
        cum_theory_rots_sigma2 = 0.0
        cum_theory_tors_sigma2 = 0.0

        trans_t = trans_fm.sample_base(chain_idx=chain_idx, is_intermediate=False)
        rotmats_t = rots_fm.sample_base(res_mask=res_mask)
        aatypes_t = aatypes_fm.sample_base(res_mask=res_mask)
        torsions_t = torsions_fm.sample_base(res_mask=res_mask)

        t1 = ts[0]
        step_i = 0

        block = SamplingBlock()
        drift_cal = DriftCalibration()

        for t2 in tqdm(ts[1:], total=len(ts) - 1, desc="Sampling metrics", leave=False):
            dt = t2 - t1
            t = torch.ones(num_batch, device=device) * t1
            dt_f = float(dt.item())
            t_mean = float(t1.item())

            trans_t, add_emp, add_th = self._step_translations(
                step_i=step_i,
                dt=dt,
                t=t,
                trans_1=trans_1,
                trans_t=trans_t,
                chain_idx=chain_idx,
                block=block,
                t_mean=t_mean,
            )
            cum_trans_e2 += add_emp
            cum_theory_trans_sigma2 += add_th

            rotmats_t, add_emp, add_th = self._step_rotations(
                step_i=step_i,
                dt=dt,
                t=t,
                rotmats_1=rotmats_1,
                rotmats_t=rotmats_t,
                block=block,
            )
            cum_rots_e2 += add_emp
            cum_theory_rots_sigma2 += add_th

            torsions_t, add_emp, add_th = self._step_torsions(
                step_i=step_i,
                dt=dt,
                t=t,
                torsions_t=torsions_t,
                block=block,
                t_mean=t_mean,
            )
            cum_tors_e2 += add_emp
            cum_theory_tors_sigma2 += add_th

            aatypes_t = self._step_aatypes(
                step_i=step_i,
                dt=dt,
                t=t,
                aatypes_t=aatypes_t,
                block=block,
                num_batch=num_batch,
                num_res=num_res,
            )

            block.count += 1
            block.t = t1.item() if block.t is None else block.t

            if block.count == step_stride:
                c = float(block.count)
                block.flush_into(
                    out=out,
                    c=c,
                    t_value=block.t,
                    cum=CumulativeRSS(
                        trans_emp=cum_trans_e2,
                        trans_theory=cum_theory_trans_sigma2,
                        rots_emp=cum_rots_e2,
                        rots_theory=cum_theory_rots_sigma2,
                        tors_emp=cum_tors_e2,
                        tors_theory=cum_theory_tors_sigma2,
                    ),
                    drift_cal=drift_cal,
                )
                block = SamplingBlock()

            t1 = t2
            step_i += 1

        return out

    def _step_translations(
        self,
        step_i: int,
        dt: torch.Tensor,
        t: torch.Tensor,
        trans_1: torch.Tensor,
        trans_t: torch.Tensor,
        chain_idx: torch.Tensor,
        block: SamplingBlock,
        t_mean: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """compute and accumulate translation sampling metrics for one step; returns next state, empirical e2 increment, theory e2 increment."""
        trans_fm = self.interpolant.trans_fm
        dt_f = float(dt.item())
        cal_trans = trans_fm.sample_base(chain_idx=chain_idx, is_intermediate=True)
        cal_trans_mean_norm = torch.linalg.norm(cal_trans, dim=-1).mean().item()
        cal_trans_mean_norm2 = (
            torch.linalg.norm(cal_trans, dim=-1).square().mean().item()
        )
        trans_next_drift = trans_fm.euler_step(
            d_t=dt,
            t=t,
            trans_1=trans_1,
            trans_t=trans_t,
            chain_idx=chain_idx,
            stochasticity_scale=torch.zeros_like(t),
            potential=None,
        )
        trans_vf = trans_fm.vector_field(t=t, trans_1=trans_1, trans_t=trans_t)
        trans_drift_step = torch.linalg.norm(trans_next_drift - trans_t, dim=-1)
        block.trans.add_drift(
            trans_drift_step.mean().item(), trans_drift_step.std().item()
        )

        seed_all(51000 + step_i)
        trans_next_noise = trans_fm.euler_step(
            d_t=dt,
            t=t,
            trans_1=trans_1,
            trans_t=trans_t,
            chain_idx=chain_idx,
            stochasticity_scale=trans_fm.effective_stochastic_scale(
                t=t, stochastic_scale=torch.ones_like(t)
            ),
            potential=None,
        )
        trans_noise_comp = trans_next_noise - (trans_t + trans_vf * dt_f)
        trans_noise_norm = torch.linalg.norm(trans_noise_comp, dim=-1)
        block.trans.add_noise(
            trans_noise_norm.mean().item(), trans_noise_norm.std().item()
        )
        add_emp = trans_noise_norm.square().mean().item()

        sig_t = self.interpolant.trans_fm._compute_sigma_t(
            t=self.interpolant.trans_fm.time_sampling(t),
            scale=self.interpolant.cfg.trans.stochastic_noise_intensity,
        ).mean().item() * math.sqrt(dt_f)
        block.trans.add_sigma(sig_t * cal_trans_mean_norm)
        add_th = (sig_t**2) * cal_trans_mean_norm2

        block.trans.add_step_ratio(
            trans_noise_norm.mean().item() / (trans_drift_step.mean().item() + 1e-8)
        )
        noise_vel = trans_noise_norm.mean().item() / max(math.sqrt(dt_f), 1e-12)
        drift_vel = trans_drift_step.mean().item() / (dt_f + 1e-12)
        block.trans.add_vel_ratio(noise_vel / (drift_vel + 1e-12))

        scale_vel = 1.0 / max(1.0 - t_mean, 1e-12)
        block.trans.add_theory_drift_scales(
            step_scale=scale_vel * dt_f, vel_scale=scale_vel, dt_value=dt_f
        )

        return trans_next_drift, add_emp, add_th

    def _step_rotations(
        self,
        step_i: int,
        dt: torch.Tensor,
        t: torch.Tensor,
        rotmats_1: torch.Tensor,
        rotmats_t: torch.Tensor,
        block: SamplingBlock,
    ) -> Tuple[torch.Tensor, float, float]:
        """compute and accumulate rotation sampling metrics for one step; returns next state, empirical e2 increment, theory e2 increment."""
        rots_fm = self.interpolant.rots_fm
        dt_f = float(dt.item())
        sigma_one = torch.ones(
            (rotmats_t.shape[0],),
            device=self.interpolant.rots_fm.igso3.sigma_grid.device,
        )
        cal_rot_noise = (
            self.interpolant.rots_fm.igso3.sample(sigma_one, rotmats_t.shape[1])
            .to(t.device)
            .reshape(rotmats_t.shape[0], rotmats_t.shape[1], 3, 3)
        )
        cal_rot_angles = _angles_from_rotmats(cal_rot_noise)
        cal_rot_mean_angle = cal_rot_angles.mean().item()
        cal_rot_second_moment = cal_rot_angles.square().mean().item()
        rot_next_drift = rots_fm.euler_step(
            d_t=dt,
            t=t,
            rotmats_1=rotmats_1,
            rotmats_t=rotmats_t,
            stochasticity_scale=torch.zeros_like(t),
            potential=None,
        )
        delta_drift = torch.matmul(rot_next_drift, rotmats_t.transpose(-1, -2))
        drift_ang = _angles_from_rotmats(delta_drift)
        block.rots.add_drift(drift_ang.mean().item(), drift_ang.std().item())

        seed_all(52000 + step_i)
        rot_next_noise = rots_fm.euler_step(
            d_t=dt,
            t=t,
            rotmats_1=rotmats_1,
            rotmats_t=rotmats_t,
            stochasticity_scale=rots_fm.effective_stochastic_scale(
                t=t, stochastic_scale=torch.ones_like(t)
            ),
            potential=None,
        )
        delta_noise = torch.matmul(rot_next_noise, rot_next_drift.transpose(-1, -2))
        noise_ang = _angles_from_rotmats(delta_noise)
        block.rots.add_noise(noise_ang.mean().item(), noise_ang.std().item())
        add_emp = noise_ang.square().mean().item()

        sig_r = rots_fm._compute_sigma_t(
            t=rots_fm.time_sampling(t),
            scale=self.interpolant.cfg.rots.stochastic_noise_intensity,
        ).mean().item() * math.sqrt(dt_f)
        block.rots.add_sigma(sig_r * cal_rot_mean_angle)
        add_th = (sig_r**2) * cal_rot_second_moment

        block.rots.add_step_ratio(
            noise_ang.mean().item() / (drift_ang.mean().item() + 1e-8)
        )
        noise_vel = noise_ang.mean().item() / max(math.sqrt(dt_f), 1e-12)
        drift_vel = drift_ang.mean().item() / (dt_f + 1e-12)
        block.rots.add_vel_ratio(noise_vel / (drift_vel + 1e-12))

        rot_scale_vel = rots_fm._vf_scaling(t).mean().item()
        block.rots.add_theory_drift_scales(
            step_scale=rot_scale_vel * dt_f, vel_scale=rot_scale_vel, dt_value=dt_f
        )

        return rot_next_drift, add_emp, add_th

    def _step_torsions(
        self,
        step_i: int,
        dt: torch.Tensor,
        t: torch.Tensor,
        torsions_t: torch.Tensor,
        block: SamplingBlock,
        t_mean: float,
    ) -> Tuple[torch.Tensor, float, float]:
        """compute and accumulate torsion sampling metrics for one step; returns next state, empirical e2 increment, theory e2 increment."""
        torsions_fm = self.interpolant.torsions_fm
        dt_f = float(dt.item())
        tors_next_drift = torsions_fm.euler_step(
            d_t=dt,
            t=t,
            torsions_1=torch.zeros_like(torsions_t),
            torsions_t=torsions_t,
            stochasticity_scale=torch.zeros_like(t),
        )
        ang_t = torch.atan2(torsions_t[..., 0], torsions_t[..., 1])
        ang_next = torch.atan2(tors_next_drift[..., 0], tors_next_drift[..., 1])
        drift_da = torch.atan2(
            torch.sin(ang_next - ang_t), torch.cos(ang_next - ang_t)
        ).abs()
        block.tors.add_drift(drift_da.mean().item(), drift_da.std().item())

        seed_all(53000 + step_i)
        tors_next_noise = torsions_fm.euler_step(
            d_t=dt,
            t=t,
            torsions_1=torch.zeros_like(torsions_t),
            torsions_t=torsions_t,
            stochasticity_scale=torsions_fm.effective_stochastic_scale(
                t=t, stochastic_scale=torch.ones_like(t)
            ),
        )
        ang_next_noise = torch.atan2(tors_next_noise[..., 0], tors_next_noise[..., 1])
        noise_da = torch.atan2(
            torch.sin(ang_next_noise - ang_next), torch.cos(ang_next_noise - ang_next)
        ).abs()
        block.tors.add_noise(noise_da.mean().item(), noise_da.std().item())
        add_emp = noise_da.square().mean().item()

        sig_tor = torsions_fm._compute_sigma_t(
            t=torsions_fm.time_sampling(t),
            scale=self.interpolant.cfg.torsions.stochastic_noise_intensity,
        ).mean().item() * math.sqrt(dt_f)
        block.tors.add_sigma(sig_tor)
        add_th = sig_tor**2

        block.tors.add_step_ratio(
            noise_da.mean().item() / (drift_da.mean().item() + 1e-8)
        )
        scale_vel = 1.0 / max(1.0 - t_mean, 1e-12)
        block.tors.add_theory_drift_scales(
            step_scale=scale_vel * dt_f, vel_scale=scale_vel, dt_value=dt_f
        )

        return tors_next_drift, add_emp, add_th

    def _step_aatypes(
        self,
        step_i: int,
        dt: torch.Tensor,
        t: torch.Tensor,
        aatypes_t: torch.Tensor,
        block: SamplingBlock,
        num_batch: int,
        num_res: int,
    ) -> torch.Tensor:
        """compute and accumulate aatypes sampling metrics for one step."""
        aatypes_fm = self.interpolant.aatypes_fm
        device = t.device
        seed_all(54000 + step_i)
        aa_next_drift = aatypes_fm.euler_step(
            d_t=dt,
            t=t,
            logits_1=torch.zeros(
                num_batch, num_res, self.interpolant.num_tokens, device=device
            ),
            aatypes_t=aatypes_t,
            stochasticity_scale=torch.zeros_like(t),
            potential=None,
        )
        drift_frac = (aa_next_drift != aatypes_t).float().mean().item()
        block.aatypes.add_drift(drift_frac, 0.0)
        seed_all(54000 + step_i)
        aa_next_noise = aatypes_fm.euler_step(
            d_t=dt,
            t=t,
            logits_1=torch.zeros(
                num_batch, num_res, self.interpolant.num_tokens, device=device
            ),
            aatypes_t=aatypes_t,
            stochasticity_scale=aatypes_fm.effective_stochastic_scale(
                t=t, stochastic_scale=torch.ones_like(t)
            ),
            potential=None,
        )
        noise_frac = (aa_next_noise != aa_next_drift).float().mean().item()
        block.aatypes.add_noise(noise_frac, 0.0)
        tau = aatypes_fm._aatypes_schedule(t=t)
        nu_t = (
            aatypes_fm._poisson_noise_weight(
                t=tau,
                d_t=dt,
                scale=torch.full_like(
                    t, float(self.interpolant.cfg.aatypes.stochastic_noise_intensity)
                ),
            )
            .mean()
            .item()
        )
        block.aatypes.add_sigma(nu_t)
        block.aatypes.add_step_ratio(noise_frac / (drift_frac + 1e-8))
        dt_f = float(dt.item())
        block.aatypes.add_theory_drift_scales(
            step_scale=1.0 * dt_f, vel_scale=1.0, dt_value=dt_f
        )
        return aa_next_drift


class Plotter:
    """plot multi-panel summaries of training and sampling metrics."""

    def __init__(self):
        pass

    def maybe(self, axes, row, col):
        """return axes[row, col] if axes is provided, else none."""
        if axes is None:
            return None
        return axes[row, col]

    def _norm_t(self, t: list) -> np.ndarray:
        """normalize time array to [0, 1] based on first and last values."""
        if len(t) < 2:
            return np.array(t, dtype=float)
        t = np.array(t, dtype=float)
        return (t - t[0]) / (t[-1] - t[0] + 1e-12)

    def plot_sigma(
        self,
        ax,
        t_train,
        th_train,
        t_samp,
        th_samp,
        title,
        unit_label_left,
        unit_label_right=None,
        right_as_deg=False,
    ):
        """plot training and sampling scale schedules on shared time axes."""
        if ax is None:
            return
        ax.set_title(title)
        ax_r = ax.twinx() if unit_label_right else None
        y_th_train = np.array(th_train)
        if right_as_deg and unit_label_left == "deg":
            y_th_train = y_th_train * (180.0 / math.pi)
        ax.plot(t_train, y_th_train, label="train σ/ρ", color="#1f77b4")
        if ax_r is None:
            ax.plot(
                t_samp, th_samp, linestyle="--", color="#ff7f0e", label="samp σ√dt/νt"
            )
        else:
            y = np.array(th_samp)
            if right_as_deg:
                y = y * (180.0 / math.pi)
            ax_r.plot(t_samp, y, linestyle="--", color="#ff7f0e", label="samp σ√dt/νt")
            ax_r.set_ylabel(unit_label_right, color="#ff7f0e")
            ax_r.tick_params(axis="y", colors="#ff7f0e")
        ax.set_ylabel(unit_label_left, color="#1f77b4")
        ax.tick_params(axis="y", colors="#1f77b4")
        ax.grid(True, alpha=0.3)
        if ax_r is not None:
            l1, lab1 = ax.get_legend_handles_labels()
            l2, lab2 = ax_r.get_legend_handles_labels()
            ax.legend(l1 + l2, lab1 + lab2, loc="upper left")
        else:
            ax.legend(loc="upper left")

    def plot_corr(self, ax, t_train, mean, std, th, title, unit):
        """plot corruption noise mean and std with theory curve."""
        if ax is None:
            return
        ax.set_title(title)
        ax.plot(t_train, mean, label="noise mean")
        ax.fill_between(
            t_train,
            np.array(mean) - np.array(std),
            np.array(mean) + np.array(std),
            alpha=0.2,
        )
        ax.plot(t_train, th, linestyle="--", alpha=0.8, label="theory")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    def plot_samp(
        self,
        ax,
        t_samp,
        noise_m,
        noise_s,
        drift_m,
        drift_s,
        th_noise,
        th_drift,
        title,
        unit,
        right_as_deg=False,
    ):
        """plot empirical noise and drift with optional theory curves."""
        if ax is None:
            return
        ax.set_title(title)
        y_noise = np.array(noise_m)
        y_drift = np.array(drift_m)
        y_th_noise = np.array(th_noise) if th_noise is not None else None
        y_th_drift = np.array(th_drift) if th_drift is not None else None
        y_noise_s = np.array(noise_s)
        y_drift_s = np.array(drift_s)
        if right_as_deg:
            y_noise = y_noise * (180.0 / math.pi)
            y_drift = y_drift * (180.0 / math.pi)
            y_noise_s = y_noise_s * (180.0 / math.pi)
            y_drift_s = y_drift_s * (180.0 / math.pi)
            if y_th_noise is not None:
                y_th_noise = y_th_noise * (180.0 / math.pi)
            if y_th_drift is not None:
                y_th_drift = y_th_drift * (180.0 / math.pi)
        ax.plot(t_samp, y_noise, label="noise")
        ax.fill_between(t_samp, y_noise - y_noise_s, y_noise + y_noise_s, alpha=0.2)
        ax.plot(t_samp, y_drift, label="drift")
        if y_th_noise is not None:
            ax.plot(
                t_samp, y_th_noise, linestyle="--", alpha=0.8, label="theory noise step"
            )
        if y_th_drift is not None:
            ax.plot(
                t_samp, y_th_drift, linestyle=":", alpha=0.8, label="theory drift step"
            )
        # Clip y-limits based on empirical series to avoid exploding theory dominating scale
        try:
            emp_upper = np.nanmax(
                np.stack(
                    [
                        y_noise + y_noise_s,
                        y_drift + y_drift_s,
                    ]
                )
            )
            if not np.isfinite(emp_upper) or emp_upper <= 0:
                emp_upper = 1.0
            ax.set_ylim(0.0, float(emp_upper) * 1.2)
        except Exception:
            pass
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    def plot_ratio(
        self,
        ax,
        t_samp,
        step_ratio,
        vel_ratio,
        title,
        th_step_ratio=None,
        th_vel_ratio=None,
    ):
        """plot noise-to-drift ratios and optional theory ratios."""
        if ax is None:
            return
        ax.set_title(title)
        ax_r = ax.twinx() if (vel_ratio is not None) else None
        ax.plot(t_samp, step_ratio, label="step", color="#1f77b4")
        if th_step_ratio is not None:
            ax.plot(
                t_samp,
                th_step_ratio,
                linestyle=":",
                label="theory step",
                color="#1f77b4",
                alpha=0.9,
            )
        # Draw theory velocity on the left axis to keep the right axis focused on empirical ratios
        if th_vel_ratio is not None:
            ax.plot(
                t_samp,
                th_vel_ratio,
                linestyle="-.",
                label="theory velocity",
                color="#ff7f0e",
                alpha=0.9,
            )
        if ax_r is not None:
            if vel_ratio is not None:
                ax_r.plot(
                    t_samp, vel_ratio, linestyle="--", label="velocity", color="#ff7f0e"
                )
            ax_r.set_ylabel("vel ratio", color="#ff7f0e")
            ax_r.tick_params(axis="y", colors="#ff7f0e")
            # Keep the velocity ratio axis within a sensible range; cap only for translations
            try:
                y = np.array(vel_ratio, dtype=float)
                if np.isfinite(y).any():
                    ymax = float(np.nanmax(y))
                    ymax = (ymax * 1.1) if (np.isfinite(ymax) and ymax > 0) else 1.0
                else:
                    ymax = 1.0
                if "Trans" in title:
                    ymax = min(2.0, ymax)
                ax_r.set_ylim(0.0, ymax)
            except Exception:
                if "Trans" in title:
                    ax_r.set_ylim(0.0, 2.0)
        ax.set_ylabel("ratio", color="#1f77b4")
        ax.tick_params(axis="y", colors="#1f77b4")
        ax.grid(True, alpha=0.3)
        if ax_r is not None:
            l1, lab1 = ax.get_legend_handles_labels()
            l2, lab2 = ax_r.get_legend_handles_labels()
            ax.legend(l1 + l2, lab1 + lab2, loc="upper left")
        else:
            ax.legend(loc="upper left")

    def plot_cum(self, ax, t_samp, emp, th, title, unit, as_deg=False):
        """plot cumulative root-sum-of-squares noise versus theory."""
        if ax is None:
            return
        ax.set_title(title)
        y_emp = np.array(emp)
        y_th = np.array(th)
        if as_deg:
            y_emp = y_emp * (180.0 / math.pi)
            y_th = y_th * (180.0 / math.pi)
        ax.plot(t_samp, y_emp, label="empirical")
        ax.plot(t_samp, y_th, linestyle="--", label="theory")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    def plot_cum_abs(
        self, ax, t_samp, steps, noise_m, drift_m, title, unit, as_deg=False
    ):
        """plot cumulative absolute path length for noise and drift."""
        if ax is None:
            return
        ax.set_title(title)
        s = np.array(steps)
        # Cumulative absolute path length from per-step mean magnitudes
        noise = np.cumsum(np.array(noise_m) * s)
        drift = np.cumsum(np.array(drift_m) * s)
        if as_deg:
            noise = noise * (180.0 / math.pi)
            drift = drift * (180.0 / math.pi)
        ax.plot(t_samp, noise, label="noise")
        ax.plot(t_samp, drift, label="drift")
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left")

    def plot_all(self, train_out, samp_out, output_path: Optional[str]):
        """create the full figure with all panels and save or show it."""
        # 4 columns, 6 rows
        fig, axes = plt.subplots(6, 4, figsize=(24, 19), sharex="col", dpi=200)

        # normalized t
        t_train_n = self._norm_t(
            train_out.trans.t if train_out.trans.t else train_out.aatypes.t
        )
        t_samp_n = self._norm_t(
            samp_out.trans.t if samp_out.trans.t else samp_out.aatypes.t
        )

        # A row: training vs sampling scales (4 columns: AATypes, Trans, Rots, Tors)
        self.plot_sigma(
            self.maybe(axes, 0, 0),
            t_train_n,
            train_out.aatypes.theory_train,
            t_samp_n,
            samp_out.aatypes.theory_sample,
            "AATypes ρ(t) vs ν_t",
            unit_label_left="frac",
            unit_label_right="frac",
        )
        # Translations (Å)
        self.plot_sigma(
            self.maybe(axes, 0, 1),
            t_train_n,
            train_out.trans.theory_train,
            t_samp_n,
            samp_out.trans.theory_sample,
            "Translations σ(t) vs σ√dt",
            unit_label_left="Å",
            unit_label_right="Å",
        )
        # Rotations (rad on left, deg on right)
        self.plot_sigma(
            self.maybe(axes, 0, 2),
            t_train_n,
            train_out.rots.theory_train,
            t_samp_n,
            samp_out.rots.theory_sample,
            "Rotations σ(t) vs σ√dt",
            unit_label_left="deg",
            unit_label_right="deg",
            right_as_deg=True,
        )
        # Torsions (rad)
        self.plot_sigma(
            self.maybe(axes, 0, 3),
            t_train_n,
            train_out.tors.theory_train,
            t_samp_n,
            samp_out.tors.theory_sample,
            "Torsions σ(t) vs σ√dt",
            unit_label_left="deg",
            unit_label_right="deg",
            right_as_deg=True,
        )

        # B row: corruption
        self.plot_corr(
            self.maybe(axes, 1, 0),
            t_train_n,
            train_out.aatypes.corr_noise_mean,
            train_out.aatypes.corr_noise_std,
            train_out.aatypes.theory_train,
            "AATypes corruption noise",
            "frac",
        )
        self.plot_corr(
            self.maybe(axes, 1, 1),
            t_train_n,
            train_out.trans.corr_noise_mean,
            train_out.trans.corr_noise_std,
            train_out.trans.theory_train,
            "Translations corruption noise",
            "Å",
        )
        ax = self.maybe(axes, 1, 2)
        if ax is not None:
            ax.set_title("Rotations corruption noise")
            y = np.array(train_out.rots.corr_noise_mean) * (180.0 / math.pi)
            ystd = np.array(train_out.rots.corr_noise_std) * (180.0 / math.pi)
            ax.plot(t_train_n, y, label="mean")
            ax.fill_between(t_train_n, y - ystd, y + ystd, alpha=0.2)
            yth = np.array(train_out.rots.theory_train) * (180.0 / math.pi)
            ax.plot(t_train_n, yth, linestyle="--", alpha=0.8, label="theory")
            ax.set_ylabel("deg")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
        ax = self.maybe(axes, 1, 3)
        if ax is not None:
            ax.set_title("Torsions corruption noise")
            y = np.array(train_out.tors.corr_noise_mean) * (180.0 / math.pi)
            ystd = np.array(train_out.tors.corr_noise_std) * (180.0 / math.pi)
            ax.plot(t_train_n, y, label="mean")
            ax.fill_between(t_train_n, y - ystd, y + ystd, alpha=0.2)
            yth = np.array(train_out.tors.theory_train) * (180.0 / math.pi)
            ax.plot(t_train_n, yth, linestyle="--", alpha=0.8, label="theory")
            ax.set_ylabel("deg")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")

        # C row: sampling noise/drift
        self.plot_samp(
            ax=self.maybe(axes, 2, 0),
            t_samp=t_samp_n,
            noise_m=samp_out.aatypes.samp_noise_mean,
            noise_s=samp_out.aatypes.samp_noise_std,
            drift_m=samp_out.aatypes.samp_drift_mean,
            drift_s=samp_out.aatypes.samp_drift_std,
            th_noise=samp_out.aatypes.theory_sample,
            th_drift=samp_out.aatypes.theory_drift_step,
            title="AATypes sampling",
            unit="frac",
        )
        self.plot_samp(
            ax=self.maybe(axes, 2, 1),
            t_samp=t_samp_n,
            noise_m=samp_out.trans.samp_noise_mean,
            noise_s=samp_out.trans.samp_noise_std,
            drift_m=samp_out.trans.samp_drift_mean,
            drift_s=samp_out.trans.samp_drift_std,
            th_noise=samp_out.trans.theory_sample,
            th_drift=samp_out.trans.theory_drift_step,
            title="Translations sampling",
            unit="Å",
        )
        self.plot_samp(
            ax=self.maybe(axes, 2, 2),
            t_samp=t_samp_n,
            noise_m=samp_out.rots.samp_noise_mean,
            noise_s=samp_out.rots.samp_noise_std,
            drift_m=samp_out.rots.samp_drift_mean,
            drift_s=samp_out.rots.samp_drift_std,
            th_noise=samp_out.rots.theory_sample,
            th_drift=samp_out.rots.theory_drift_step,
            title="Rotations sampling",
            unit="deg",
            right_as_deg=True,
        )
        self.plot_samp(
            ax=self.maybe(axes, 2, 3),
            t_samp=t_samp_n,
            noise_m=samp_out.tors.samp_noise_mean,
            noise_s=samp_out.tors.samp_noise_std,
            drift_m=samp_out.tors.samp_drift_mean,
            drift_s=samp_out.tors.samp_drift_std,
            th_noise=samp_out.tors.theory_sample,
            th_drift=samp_out.tors.theory_drift_step,
            title="Torsions sampling",
            unit="deg",
            right_as_deg=True,
        )

        # D row: ratios
        self.plot_ratio(
            ax=self.maybe(axes, 3, 0),
            t_samp=t_samp_n,
            step_ratio=samp_out.aatypes.step_ratio,
            vel_ratio=None,
            title="AATypes noise/drift ratio",
            th_step_ratio=samp_out.aatypes.step_ratio_theory,
            th_vel_ratio=None,
        )
        self.plot_ratio(
            ax=self.maybe(axes, 3, 1),
            t_samp=t_samp_n,
            step_ratio=samp_out.trans.step_ratio,
            vel_ratio=samp_out.trans.vel_ratio,
            title="Trans noise/drift ratio",
            th_step_ratio=samp_out.trans.step_ratio_theory,
            th_vel_ratio=samp_out.trans.vel_ratio_theory,
        )
        self.plot_ratio(
            ax=self.maybe(axes, 3, 2),
            t_samp=t_samp_n,
            step_ratio=samp_out.rots.step_ratio,
            vel_ratio=samp_out.rots.vel_ratio,
            title="Rot noise/drift ratio",
            th_step_ratio=samp_out.rots.step_ratio_theory,
            th_vel_ratio=samp_out.rots.vel_ratio_theory,
        )
        self.plot_ratio(
            ax=self.maybe(axes, 3, 3),
            t_samp=t_samp_n,
            step_ratio=samp_out.tors.step_ratio,
            vel_ratio=samp_out.tors.vel_ratio,
            title="Tors noise/drift ratio",
            th_step_ratio=samp_out.tors.step_ratio_theory,
            th_vel_ratio=samp_out.tors.vel_ratio_theory,
        )

        # E row: cumulative noise (RSS) vs theory
        ax = self.maybe(axes, 4, 0)
        if ax is not None:
            ax.set_title("AATypes cumulative per-step noise (fraction)")
            steps = np.array(samp_out.aatypes.steps)
            noise = np.array(samp_out.aatypes.samp_noise_mean)
            th = np.array(samp_out.aatypes.theory_sample)
            cum_emp = np.cumsum(noise * steps)
            cum_th = np.cumsum(th * steps)
            ax.plot(t_samp_n, cum_emp, label="empirical")
            ax.plot(t_samp_n, cum_th, linestyle="--", label="theory")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="upper left")
        self.plot_cum(
            ax=self.maybe(axes, 4, 1),
            t_samp=t_samp_n,
            emp=samp_out.trans.samp_cum_rss,
            th=samp_out.trans.theory_cum_rss,
            title="Trans cumulative RSS",
            unit="Å",
        )
        self.plot_cum(
            ax=self.maybe(axes, 4, 2),
            t_samp=t_samp_n,
            emp=samp_out.rots.samp_cum_rss,
            th=samp_out.rots.theory_cum_rss,
            title="Rot cumulative RSS",
            unit="deg",
            as_deg=True,
        )
        self.plot_cum(
            ax=self.maybe(axes, 4, 3),
            t_samp=t_samp_n,
            emp=samp_out.tors.samp_cum_rss,
            th=samp_out.tors.theory_cum_rss,
            title="Tors cumulative RSS",
            unit="deg",
            as_deg=True,
        )

        # F row: cumulative absolute movement (empirical drift vs noise)
        self.plot_cum_abs(
            ax=self.maybe(axes, 5, 1),
            t_samp=t_samp_n,
            steps=samp_out.trans.steps,
            noise_m=samp_out.trans.samp_noise_mean,
            drift_m=samp_out.trans.samp_drift_mean,
            title="Trans cumulative |step|",
            unit="Å",
        )
        self.plot_cum_abs(
            ax=self.maybe(axes, 5, 2),
            t_samp=t_samp_n,
            steps=samp_out.rots.steps,
            noise_m=samp_out.rots.samp_noise_mean,
            drift_m=samp_out.rots.samp_drift_mean,
            title="Rot cumulative |step|",
            unit="deg",
            as_deg=True,
        )
        self.plot_cum_abs(
            ax=self.maybe(axes, 5, 3),
            t_samp=t_samp_n,
            steps=samp_out.tors.steps,
            noise_m=samp_out.tors.samp_noise_mean,
            drift_m=samp_out.tors.samp_drift_mean,
            title="Tors cumulative |step|",
            unit="deg",
            as_deg=True,
        )

        for r in range(6):
            for c in range(4):
                axes[r, c].set_xlim(0.0, 1.0)
        axes[5, 0].set_xlabel("time")
        axes[5, 1].set_xlabel("time")
        axes[5, 2].set_xlabel("time")
        axes[5, 3].set_xlabel("time")
        plt.tight_layout()
        if output_path:
            d = os.path.dirname(output_path)
            if d:
                os.makedirs(d, exist_ok=True)
            plt.savefig(output_path, dpi=200, bbox_inches="tight")
            print(f"Saved figure to {output_path}")
        else:
            plt.show()


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(cfg: Config) -> None:
    """generate metrics and plot results based on the hydra config."""
    cfg = OmegaConf.to_object(cfg)
    cfg = cfg.interpolate()

    seed_all(cfg.noise_plot.seed)

    interpolant = Interpolant(cfg=cfg.interpolant)
    device = get_device()
    interpolant.set_device(device)

    print(
        f"""
        Generating metrics: 
        num_batch={cfg.noise_plot.num_batch}
        num_res={cfg.noise_plot.num_res}
        num_timesteps={cfg.inference.interpolant.sampling.num_timesteps}
        stride={cfg.noise_plot.step_stride}
        """
    )

    train_out = TrainingMeasurer(cfg=cfg, interpolant=interpolant).measure()

    samp_out = SamplingMeasurer(cfg=cfg, interpolant=interpolant).measure()

    Plotter().plot_all(
        train_out=train_out, samp_out=samp_out, output_path=cfg.noise_plot.output_path
    )


if __name__ == "__main__":
    run()
