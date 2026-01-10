"""
Visualize noise distributions during corruption and sampling in Varco.

Uses Hydra config (inherits from VarcoConfig) so you can override any
interpolant parameters to see how they affect noise.

Usage:
    # Corruption analysis only (no model needed)
    python -m varco.visualize_noise

    # With model checkpoint for sampling analysis
    python -m varco.visualize_noise inference.ckpt_path=varco/ckpt/best.ckpt

    # Override interpolant config to test different noise settings
    python -m varco.visualize_noise \
        interpolant.trans_coupler.noise_scale=2.0 \
        interpolant.rotation_coupler.noise_scale=2.0

    # Change num_steps to see effect on per-step noise
    python -m varco.visualize_noise +num_steps=50
"""

import copy
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from cogeneration.data.noise_mask import uniform_so3
from cogeneration.data.so3_utils import (
    angle_from_rotmat,
    calc_rot_vf,
    geodesic_t,
    rotmat_to_rotvec,
)
from varco.config import (
    VarcoConfig,
    VarcoHazardConfig,
    VarcoHazardKind,
    VarcoInterpolantConfig,
)
from varco.data import DataBatch, DataCorrupted
from varco.interpolant import TreeInterpolant
from varco.module import BranchFlowModule
from varco.tree_plan import BatchedTreePlan, TreePlan

# ============================================================
# PART 1: THEORETICAL NOISE PROFILES (no model/data needed)
# ============================================================


@dataclass(frozen=True)
class MetricsCfg:
    pairwise_max_points: int = 256
    pairwise_seed: int = 0


def _expected_indel_step_metrics(
    split_mass: torch.Tensor,  # (B, P) time-independent insertion mass
    del_logits: torch.Tensor,  # (B, P)
    is_root: torch.Tensor,  # (B, P) bool
    valid_mask: torch.Tensor,  # (B, P) bool (scaffold-only in sampling)
    t_val: float,
    dt: float,
    split_hazard: VarcoHazardConfig,
    delete_hazard: VarcoHazardConfig,
    pred_split_pooled_log1p_mass: Optional[torch.Tensor] = None,  # (B,)
) -> Dict[str, float]:
    """
    Expected intensity diagnostics for per-token Bernoulli event sampling.

    Mirrors TreeInterpolant._sample_insert_delete_substitute(), but returns
    expected aggregate counts (sum of per-token probabilities) rather than samples.
    """
    eps = 1e-6
    t = max(eps, min(1.0 - eps, float(t_val)))
    t_next = min(1.0 - eps, t + float(dt))

    # Compute survival function S(t) for the split hazard
    p_split = float(max(1, split_hazard.power))
    if split_hazard.kind == VarcoHazardKind.uniform:
        S_t = 1.0 - t
    elif split_hazard.kind == VarcoHazardKind.early_power:
        S_t = (1.0 - t) ** p_split
    elif split_hazard.kind == VarcoHazardKind.late_power:
        S_t = 1.0 - (t**p_split)
    else:
        raise ValueError(f"Unknown split hazard kind: {split_hazard.kind!r}")

    # Convert mass to rate: R_t = M * S(t)
    split_mass = split_mass.clamp_min(0.0)
    split_rate = split_mass * S_t  # (B, P)

    # Calibrate using pooled prediction
    if pred_split_pooled_log1p_mass is not None:
        token_sum = (split_rate * valid_mask.float()).sum(dim=1)  # (B,)
        pooled_mass = torch.expm1(pred_split_pooled_log1p_mass).clamp_min(0.0)
        pooled_total = pooled_mass * S_t  # (B,)
        scale = pooled_total / token_sum.clamp_min(0.1)
        scale = scale.clamp(min=0.1, max=1.2)
        split_rate = split_rate * scale.unsqueeze(1)

    p_split = float(max(1, split_hazard.power))
    if split_hazard.kind == VarcoHazardKind.uniform:
        I_split = math.log((1.0 - t) / max(1e-12, 1.0 - t_next))
    elif split_hazard.kind == VarcoHazardKind.early_power:
        I_split = p_split * math.log((1.0 - t) / max(1e-12, 1.0 - t_next))
    elif split_hazard.kind == VarcoHazardKind.late_power:
        tp = t**p_split
        tnp = t_next**p_split
        I_split = math.log((1.0 - tp) / max(1e-12, 1.0 - tnp))
    else:
        raise ValueError(f"Unknown split hazard kind: {split_hazard.kind!r}")

    lam_ins = split_rate * float(I_split)
    p_ins = (1.0 - torch.exp(-lam_ins.clamp_max(10.0))).clamp(0.0, 0.95)
    p_ins = torch.where(valid_mask, p_ins, torch.zeros_like(p_ins))

    # --- Deletions ---
    p_del_final = torch.sigmoid(del_logits).clamp(eps, 1.0 - eps)
    p_del_pow = float(max(1, delete_hazard.power))
    if delete_hazard.kind == VarcoHazardKind.uniform:
        H_t = t
        H_tn = t_next
    elif delete_hazard.kind == VarcoHazardKind.early_power:
        H_t = 1.0 - (1.0 - t) ** p_del_pow
        H_tn = 1.0 - (1.0 - t_next) ** p_del_pow
    elif delete_hazard.kind == VarcoHazardKind.late_power:
        H_t = t**p_del_pow
        H_tn = t_next**p_del_pow
    else:
        raise ValueError(f"Unknown delete hazard kind: {delete_hazard.kind!r}")

    denom = (1.0 - p_del_final * float(H_t)).clamp_min(eps)
    numer = (1.0 - p_del_final * float(H_tn)).clamp_min(0.0)
    p_del = (1.0 - (numer / denom)).clamp(0.0, 0.95)
    p_del = torch.where(is_root, torch.zeros_like(p_del), p_del)
    p_del = torch.where(valid_mask, p_del, torch.zeros_like(p_del))

    deletable = valid_mask & ~is_root
    if deletable.any():
        p_del_final_mean = float(_masked_mean(p_del_final, deletable).item())
    else:
        p_del_final_mean = 0.0

    if valid_mask.any():
        split_rate_mean = float(_masked_mean(split_rate, valid_mask).item())
        p_ins_sum = float(p_ins[valid_mask].sum().item())
        p_del_sum = float(p_del[valid_mask].sum().item())
    else:
        split_rate_mean = 0.0
        p_ins_sum = 0.0
        p_del_sum = 0.0

    return {
        "I_split_mean": float(I_split),
        "split_rate_mean": split_rate_mean,
        "p_ins_sum": p_ins_sum,
        "p_del_final_mean": p_del_final_mean,
        "delta_H": float(H_tn - H_t),
        "p_del_sum": p_del_sum,
    }


def _masked_mean_per_batch(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Compute mean per batch element, then average across batch.

    x: (B, P, ...) and mask: (B, P)
    """
    if mask.ndim != 2:
        raise ValueError(f"mask must be (B, P); got {tuple(mask.shape)}")
    if x.shape[:2] != mask.shape:
        raise ValueError(
            f"x and mask must share (B, P); got {tuple(x.shape)} vs {tuple(mask.shape)}"
        )

    mask_f = mask.float()
    while mask_f.ndim < x.ndim:
        mask_f = mask_f.unsqueeze(-1)
    denom = mask_f.sum(dim=1).clamp_min(1.0)
    num = (x * mask_f).sum(dim=1)
    while denom.ndim < num.ndim:
        denom = denom.unsqueeze(-1)
    return (num / denom).mean()


def _masked_max_per_batch(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    x: (B, P) and mask: (B, P)
    """
    if x.shape != mask.shape:
        raise ValueError(
            f"x and mask must match; got {tuple(x.shape)} vs {tuple(mask.shape)}"
        )
    x_masked = torch.where(mask, x, torch.full_like(x, float("-inf")))
    per_b = x_masked.max(dim=1).values
    per_b = torch.where(torch.isfinite(per_b), per_b, torch.zeros_like(per_b))
    return per_b.mean()


def _translation_geometry_metrics(
    trans: torch.Tensor,  # (B, P, 3)
    mask: torch.Tensor,  # (B, P)
    metrics_cfg: MetricsCfg,
    seed_offset: int = 0,
) -> Dict[str, float]:
    if trans.ndim != 3 or trans.shape[-1] != 3:
        raise ValueError(f"trans must be (B, P, 3); got {tuple(trans.shape)}")
    if mask.shape != trans.shape[:2]:
        raise ValueError(
            f"mask must be (B, P); got {tuple(mask.shape)} vs {tuple(trans.shape)}"
        )

    B, P, _ = trans.shape
    device = trans.device

    # Center of mass (masked)
    mask_f = mask.float()
    denom = mask_f.sum(dim=1).clamp_min(1.0).view(B, 1, 1)
    com = (trans * mask_f.unsqueeze(-1)).sum(dim=1, keepdim=True) / denom  # (B, 1, 3)
    centered = trans - com
    r2 = centered.square().sum(dim=-1)  # (B, P)

    rg = torch.sqrt(_masked_mean_per_batch(r2, mask).clamp_min(0.0))
    max_r = torch.sqrt(_masked_max_per_batch(r2, mask).clamp_min(0.0))

    # Pairwise distances on a deterministic subsample per batch element
    pairwise_means: List[float] = []
    pairwise_medians: List[float] = []
    pairwise_p95s: List[float] = []
    rng = np.random.default_rng(int(metrics_cfg.pairwise_seed) + int(seed_offset))

    for b in range(B):
        idx = torch.nonzero(mask[b], as_tuple=False).view(-1)
        n = int(idx.numel())
        if n < 2:
            pairwise_means.append(0.0)
            pairwise_medians.append(0.0)
            pairwise_p95s.append(0.0)
            continue

        m = min(int(metrics_cfg.pairwise_max_points), n)
        if m < n:
            choice = rng.choice(n, size=m, replace=False)
            sel = idx[torch.from_numpy(choice).to(device=idx.device)]
        else:
            sel = idx

        x = trans[b, sel].to(device=device)  # (m, 3)
        d = torch.cdist(x, x)  # (m, m)
        triu = torch.triu_indices(m, m, offset=1, device=device)
        dvals = d[triu[0], triu[1]]
        if dvals.numel() == 0:
            pairwise_means.append(0.0)
            pairwise_medians.append(0.0)
            pairwise_p95s.append(0.0)
            continue

        pairwise_means.append(float(dvals.mean().item()))
        pairwise_medians.append(float(dvals.median().item()))
        pairwise_p95s.append(float(dvals.quantile(0.95).item()))

    return {
        "rg": float(rg.item()),
        "max_r": float(max_r.item()),
        "pairwise_mean": float(np.mean(pairwise_means)),
        "pairwise_median": float(np.mean(pairwise_medians)),
        "pairwise_p95": float(np.mean(pairwise_p95s)),
    }


def compute_bridge_variance(s: np.ndarray, t0: float) -> np.ndarray:
    """
    Compute Brownian bridge variance: var(s,t0) = (s-t0)(1-s)/(1-t0)
    for s in [t0, 1].
    """
    denom = max(1.0 - t0, 1e-6)
    s_clipped = np.clip(s, t0, 1.0)
    return (s_clipped - t0) * (1.0 - s_clipped) / denom


def compute_sigma_t(
    t: np.ndarray, noise_scale: float, noise_end_t: float
) -> np.ndarray:
    """
    Compute sigma(t) = sqrt(scale^2 * t_eff * (1 - t_eff))
    where t_eff = min(t / noise_end_t, 1.0)
    """
    t_eff = np.clip(t / noise_end_t, 0.0, 1.0)
    return np.sqrt(noise_scale**2 * t_eff * (1.0 - t_eff))


def plot_bridge_variance_profiles(cfg: VarcoInterpolantConfig) -> plt.Figure:
    """
    Plot var(s,t0) = (s-t0)(1-s)/(1-t0) for different birth times.
    Shows how noise varies across the corruption trajectory.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    s = np.linspace(0, 1, 200)
    birth_times = [0.0, 0.25, 0.5, 0.75]
    colors = plt.cm.viridis(np.linspace(0, 0.8, len(birth_times)))

    # Left: variance
    ax = axes[0]
    for t0, color in zip(birth_times, colors):
        var = compute_bridge_variance(s, t0)
        var[s < t0] = np.nan
        ax.plot(s, var, label=f"birth t={t0}", color=color, linewidth=2)

    ax.set_xlabel("Time s")
    ax.set_ylabel("Variance")
    ax.set_title("Brownian Bridge Variance: var(s,t0) = (s-t0)(1-s)/(1-t0)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)

    # Right: std dev scaled by noise_scale
    ax = axes[1]
    noise_scale = cfg.trans_coupler.noise_scale
    for t0, color in zip(birth_times, colors):
        var = compute_bridge_variance(s, t0)
        std = np.sqrt(var) * noise_scale
        std[s < t0] = np.nan
        ax.plot(s, std, label=f"birth t={t0}", color=color, linewidth=2)

    ax.set_xlabel("Time s")
    ax.set_ylabel(f"Std Dev (noise_scale={noise_scale})")
    ax.set_title("Translation Std Dev During Corruption (Angstroms)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Add annotation for peak
    peak_std = 0.5 * noise_scale  # At t0=0, s=0.5
    ax.axhline(y=peak_std, color="red", linestyle="--", alpha=0.5)
    ax.annotate(
        f"Peak: {peak_std:.2f} A",
        xy=(0.5, peak_std),
        xytext=(0.6, peak_std + 0.05),
        fontsize=10,
    )

    fig.tight_layout()
    return fig


def plot_sampling_sigma_schedule(
    cfg: VarcoInterpolantConfig, num_steps: int
) -> plt.Figure:
    """
    Plot sigma(t) schedule and per-step noise magnitude.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    t = np.linspace(0, 1, 200)
    dt = 1.0 / num_steps

    # Config values
    trans_cfg = cfg.trans_coupler
    rot_cfg = cfg.rotation_coupler
    aa_cfg = cfg.aatypes_coupler

    configs = [
        ("Translation", trans_cfg.noise_scale, trans_cfg.noise_end_t, "Angstroms"),
        ("Rotation", rot_cfg.noise_scale, rot_cfg.noise_end_t, "radians"),
        ("Sequence", aa_cfg.noise_scale, aa_cfg.noise_end_t, "unitless"),
    ]

    # Top left: sigma(t) for each domain
    ax = axes[0, 0]
    for name, scale, end_t, _ in configs:
        sigma = compute_sigma_t(t, scale, end_t)
        ax.plot(t, sigma, label=f"{name} (scale={scale}, end_t={end_t})", linewidth=2)

    ax.set_xlabel("Time t")
    ax.set_ylabel("sigma(t)")
    ax.set_title("Instantaneous Noise Schedule sigma(t)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Top right: Per-step noise = sigma(t) * sqrt(dt)
    ax = axes[0, 1]
    sqrt_dt = np.sqrt(dt)
    for name, scale, end_t, unit in configs:
        sigma = compute_sigma_t(t, scale, end_t)
        per_step = sigma * sqrt_dt
        ax.plot(
            t,
            per_step,
            label=f"{name} ({unit})",
            linewidth=2,
        )

    ax.set_xlabel("Time t")
    ax.set_ylabel(f"sigma(t) * sqrt(dt), dt={dt:.4f}")
    ax.set_title(f"Per-Step Noise Magnitude ({num_steps} steps)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom left: Cumulative noise (very rough approximation)
    ax = axes[1, 0]
    t_steps = np.linspace(0, 1, num_steps + 1)
    for name, scale, end_t, unit in configs:
        cumulative = np.zeros(num_steps + 1)
        for i in range(1, num_steps + 1):
            sigma_i = compute_sigma_t(np.array([t_steps[i - 1]]), scale, end_t)[0]
            # Cumulative RMS: sqrt(sum of variances)
            cumulative[i] = np.sqrt(cumulative[i - 1] ** 2 + (sigma_i * sqrt_dt) ** 2)
        ax.plot(t_steps, cumulative, label=f"{name}", linewidth=2)

    ax.set_xlabel("Time t")
    ax.set_ylabel("Cumulative RMS noise")
    ax.set_title("Cumulative Noise (RMS sum)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Bottom right: Text summary
    ax = axes[1, 1]
    ax.axis("off")

    summary_text = f"""
Configuration Summary
=====================

Translation:
  noise_scale = {trans_cfg.noise_scale}
  noise_end_t = {trans_cfg.noise_end_t}
  drift_step_cap = {trans_cfg.drift_step_cap_ang} A

Rotation:
  noise_scale = {rot_cfg.noise_scale}
  noise_end_t = {rot_cfg.noise_end_t}
  igso3_sigma_min = {rot_cfg.igso3_sigma_min}
  igso3_sigma_max = {rot_cfg.igso3_sigma_max}
  drift_step_cap = {rot_cfg.drift_step_cap_rad:.3f} rad ({np.degrees(rot_cfg.drift_step_cap_rad):.1f} deg)

Sequence:
  noise_scale = {aa_cfg.noise_scale}
  noise_end_t = {aa_cfg.noise_end_t}
  beta = {aa_cfg.beta}

Sampling:
  num_steps = {num_steps}
  dt = {dt:.5f}
  sqrt(dt) = {sqrt_dt:.5f}

Peak per-step noise (at t=0.5):
  Trans: {compute_sigma_t(np.array([0.5]), trans_cfg.noise_scale, trans_cfg.noise_end_t)[0] * sqrt_dt:.4f} A
  Rot: {compute_sigma_t(np.array([0.5]), rot_cfg.noise_scale, rot_cfg.noise_end_t)[0] * sqrt_dt:.4f} rad
"""
    ax.text(
        0.1,
        0.95,
        summary_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    fig.tight_layout()
    return fig


def plot_per_step_noise_comparison(cfg: VarcoInterpolantConfig) -> plt.Figure:
    """
    Compare per-step noise for different num_steps values.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    step_counts = [50, 100, 200, 500]
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, len(step_counts)))
    t = np.linspace(0.01, 0.95, 100)

    configs = [
        ("Translation", cfg.trans_coupler.noise_scale, cfg.trans_coupler.noise_end_t),
        (
            "Rotation",
            cfg.rotation_coupler.noise_scale,
            cfg.rotation_coupler.noise_end_t,
        ),
        ("Sequence", cfg.aatypes_coupler.noise_scale, cfg.aatypes_coupler.noise_end_t),
    ]

    for ax, (name, scale, end_t) in zip(axes, configs):
        sigma = compute_sigma_t(t, scale, end_t)

        for steps, color in zip(step_counts, colors):
            dt = 1.0 / steps
            per_step = sigma * np.sqrt(dt)
            ax.plot(
                t,
                per_step,
                label=f"{steps} steps (dt={dt:.4f})",
                color=color,
                linewidth=2,
            )

        ax.set_xlabel("Time t")
        ax.set_ylabel("Per-step noise magnitude")
        ax.set_title(f"{name}: sigma(t) * sqrt(dt)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


# ============================================================
# PART 2: EMPIRICAL CORRUPTION ANALYSIS (no model needed)
# ============================================================


def create_synthetic_databatch(
    N: int,
    B: int,
    device: torch.device,
    motif_fraction: float = 0.25,
    seed: Optional[int] = None,
) -> DataBatch:
    """
    Create synthetic DataBatch for corruption analysis.
    """
    if seed is not None:
        torch.manual_seed(seed)

    # Create motif mask: first motif_fraction of positions are motifs
    motif_count = max(1, int(N * motif_fraction))
    motif_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    motif_mask[:, :motif_count] = True

    # Random t=1 states
    trans_1 = torch.randn(B, N, 3, device=device) * 10.0  # ~10 angstrom spread
    rotmats_1 = uniform_so3(B, N, device=device)
    aatypes_1 = torch.randint(0, 20, (B, N), device=device)

    # res_mask and chain_idx
    res_mask = torch.ones(B, N, dtype=torch.long, device=device)
    chain_idx = torch.ones(B, N, dtype=torch.long, device=device)

    # Contact conditioning (zeros = no conditioning)
    contact_conditioning = torch.zeros(B, N, N, device=device)

    # B-factor and pLDDT
    res_bfactor = torch.zeros(B, N, device=device)
    res_plddt = torch.full((B, N), 90.0, device=device)

    # Generate tree plans
    tree_plans = []
    for b in range(B):
        plan_seed = None if seed is None else int(seed) + int(b)
        plan = TreePlan.generate(motif_mask=motif_mask[b].cpu(), seed=plan_seed)
        tree_plans.append(plan)

    tree = BatchedTreePlan.collate(tree_plans).to(device)

    return DataBatch(
        tree=tree,
        motif_mask=motif_mask,
        res_mask=res_mask,
        chain_idx=chain_idx,
        trans_1=trans_1,
        rotmats_1=rotmats_1,
        aatypes_1=aatypes_1,
        contact_conditioning=contact_conditioning,
        res_bfactor=res_bfactor,
        res_plddt=res_plddt,
    )


def analyze_corruption_noise(
    cfg: VarcoInterpolantConfig,
    num_samples: int,
    num_steps: int,
    device: torch.device,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Compare stochastic vs deterministic corruption.
    """
    N, B = 50, 4
    times = np.linspace(0.01, 0.99, num_steps)

    # Collect statistics
    trans_deviations_motif = []
    trans_deviations_scaffold = []
    rot_deviations_motif = []
    rot_deviations_scaffold = []
    seq_flips_motif = []
    seq_flips_scaffold = []
    # Geometry metrics for noisy corruption (absolute scale)
    geom_all_rg = []
    geom_all_pairwise_p95 = []
    geom_scaffold_rg = []
    geom_scaffold_pairwise_p95 = []

    metrics_cfg = MetricsCfg()

    print(f"Analyzing corruption noise over {num_samples} samples...")

    for sample_idx in tqdm(range(num_samples), desc="Corruption samples"):
        # Create fresh batch
        batch = create_synthetic_databatch(N, B, device, seed=sample_idx)

        # Stochastic interpolant (use config values)
        interpolant_noisy = TreeInterpolant(cfg=cfg, device=device)
        interpolant_noisy.set_device(device)

        # Deterministic interpolant (noise_scale=0)
        cfg_det = copy.deepcopy(cfg)
        cfg_det.trans_coupler.noise_scale = 0.0
        cfg_det.rotation_coupler.noise_scale = 0.0
        cfg_det.aatypes_coupler.noise_scale = 0.0
        interpolant_det = TreeInterpolant(cfg=cfg_det, device=device)
        interpolant_det.set_device(device)

        # Generate trajectories
        traj_noisy, _ = interpolant_noisy.corrupt_trajectory(
            batch=batch, times=list(times), seed=sample_idx
        )
        traj_det, _ = interpolant_det.corrupt_trajectory(
            batch=batch, times=list(times), seed=sample_idx
        )

        # Collect per-time deviations
        trans_dev_m, trans_dev_s = [], []
        rot_dev_m, rot_dev_s = [], []
        seq_flip_m, seq_flip_s = [], []

        for t_idx in range(len(times)):
            noisy = traj_noisy.samples[t_idx]
            det = traj_det.samples[t_idx]

            # Translation deviation
            trans_diff = (noisy.trans_t - det.trans_t).norm(dim=-1)  # (B, P)
            valid = noisy.valid_mask

            # Motif mask in packed space - approximate by checking first positions
            # In packed space, motifs are typically at the front
            P = noisy.trans_t.shape[1]
            pack_motif = noisy.motif_mask  # (B, P)

            motif_valid = valid & pack_motif
            scaffold_valid = valid & ~pack_motif

            if motif_valid.any():
                trans_dev_m.append(trans_diff[motif_valid].mean().item())
            else:
                trans_dev_m.append(0.0)

            if scaffold_valid.any():
                trans_dev_s.append(trans_diff[scaffold_valid].mean().item())
            else:
                trans_dev_s.append(0.0)

            # Rotation deviation (angle between rotation matrices)
            R_diff = torch.einsum(
                "...ij,...kj->...ik", noisy.rotmats_t, det.rotmats_t
            )  # R_noisy @ R_det.T
            angles = angle_from_rotmat(R_diff.reshape(-1, 3, 3))[0].reshape(
                noisy.rotmats_t.shape[:-2]
            )  # (B, P)

            if motif_valid.any():
                rot_dev_m.append(np.degrees(angles[motif_valid].mean().item()))
            else:
                rot_dev_m.append(0.0)

            if scaffold_valid.any():
                rot_dev_s.append(np.degrees(angles[scaffold_valid].mean().item()))
            else:
                rot_dev_s.append(0.0)

            # Sequence flip rate
            seq_diff = (noisy.aatypes_t != det.aatypes_t).float()  # (B, P)

            if motif_valid.any():
                seq_flip_m.append(seq_diff[motif_valid].mean().item())
            else:
                seq_flip_m.append(0.0)

            if scaffold_valid.any():
                seq_flip_s.append(seq_diff[scaffold_valid].mean().item())
            else:
                seq_flip_s.append(0.0)

            # Geometry metrics (noisy)
            geom_valid = noisy.valid_mask
            geom_all = _translation_geometry_metrics(
                trans=noisy.trans_t,
                mask=geom_valid,
                metrics_cfg=metrics_cfg,
                seed_offset=sample_idx * 1000 + t_idx,
            )
            geom_all_rg.append(geom_all["rg"])
            geom_all_pairwise_p95.append(geom_all["pairwise_p95"])

            geom_scaf = _translation_geometry_metrics(
                trans=noisy.trans_t,
                mask=geom_valid & ~noisy.motif_mask,
                metrics_cfg=metrics_cfg,
                seed_offset=sample_idx * 1000 + t_idx + 17,
            )
            geom_scaffold_rg.append(geom_scaf["rg"])
            geom_scaffold_pairwise_p95.append(geom_scaf["pairwise_p95"])

        trans_deviations_motif.append(trans_dev_m)
        trans_deviations_scaffold.append(trans_dev_s)
        rot_deviations_motif.append(rot_dev_m)
        rot_deviations_scaffold.append(rot_dev_s)
        seq_flips_motif.append(seq_flip_m)
        seq_flips_scaffold.append(seq_flip_s)

    # Convert to arrays and compute statistics
    trans_dev_motif = np.array(trans_deviations_motif)  # (num_samples, num_steps)
    trans_dev_scaffold = np.array(trans_deviations_scaffold)
    rot_dev_motif = np.array(rot_deviations_motif)
    rot_dev_scaffold = np.array(rot_deviations_scaffold)
    seq_flip_motif = np.array(seq_flips_motif)
    seq_flip_scaffold = np.array(seq_flips_scaffold)

    geom_all_rg = np.array(geom_all_rg).reshape(num_samples, num_steps)
    geom_all_pairwise_p95 = np.array(geom_all_pairwise_p95).reshape(
        num_samples, num_steps
    )
    geom_scaffold_rg = np.array(geom_scaffold_rg).reshape(num_samples, num_steps)
    geom_scaffold_pairwise_p95 = np.array(geom_scaffold_pairwise_p95).reshape(
        num_samples, num_steps
    )

    # Figure 1: Translation and Rotation
    fig1, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Translation - motif vs scaffold
    ax = axes[0, 0]
    mean_m = trans_dev_motif.mean(axis=0)
    std_m = trans_dev_motif.std(axis=0)
    mean_s = trans_dev_scaffold.mean(axis=0)
    std_s = trans_dev_scaffold.std(axis=0)

    ax.plot(times, mean_m, label="Motif", color="blue", linewidth=2)
    ax.fill_between(times, mean_m - std_m, mean_m + std_m, alpha=0.3, color="blue")
    ax.plot(times, mean_s, label="Scaffold", color="orange", linewidth=2)
    ax.fill_between(times, mean_s - std_s, mean_s + std_s, alpha=0.3, color="orange")

    ax.set_xlabel("Time t")
    ax.set_ylabel("Translation Deviation (A)")
    ax.set_title("Translation: Stochastic vs Deterministic Corruption")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Rotation - motif vs scaffold
    ax = axes[0, 1]
    mean_m = rot_dev_motif.mean(axis=0)
    std_m = rot_dev_motif.std(axis=0)
    mean_s = rot_dev_scaffold.mean(axis=0)
    std_s = rot_dev_scaffold.std(axis=0)

    ax.plot(times, mean_m, label="Motif", color="blue", linewidth=2)
    ax.fill_between(times, mean_m - std_m, mean_m + std_m, alpha=0.3, color="blue")
    ax.plot(times, mean_s, label="Scaffold", color="orange", linewidth=2)
    ax.fill_between(times, mean_s - std_s, mean_s + std_s, alpha=0.3, color="orange")

    ax.set_xlabel("Time t")
    ax.set_ylabel("Rotation Deviation (degrees)")
    ax.set_title("Rotation: Stochastic vs Deterministic Corruption")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Combined view
    ax = axes[1, 0]
    ax.plot(
        times, trans_dev_scaffold.mean(axis=0), label="Trans (scaffold)", linewidth=2
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Translation Deviation (A)", color="tab:blue")
    ax.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax.twinx()
    ax2.plot(
        times,
        rot_dev_scaffold.mean(axis=0),
        label="Rot (scaffold)",
        color="tab:orange",
        linewidth=2,
    )
    ax2.set_ylabel("Rotation Deviation (degrees)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax.set_title("Combined: Trans (blue) and Rot (orange) for Scaffold")
    ax.grid(True, alpha=0.3)

    # Theoretical comparison
    ax = axes[1, 1]
    # Theoretical std for birth_time=0
    theoretical_var = compute_bridge_variance(times, 0.0)
    theoretical_std_trans = np.sqrt(theoretical_var) * cfg.trans_coupler.noise_scale
    theoretical_std_rot = np.sqrt(theoretical_var) * cfg.rotation_coupler.noise_scale

    ax.plot(
        times,
        theoretical_std_trans,
        "--",
        label="Theory (trans)",
        color="blue",
        linewidth=2,
    )
    ax.plot(
        times,
        trans_dev_scaffold.mean(axis=0),
        label="Empirical (trans)",
        color="blue",
        alpha=0.7,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Standard Deviation")
    ax.set_title("Theoretical vs Empirical Noise (Scaffold, birth_time=0)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig1.tight_layout()

    # Figure 2: Sequence
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    mean_m = seq_flip_motif.mean(axis=0)
    std_m = seq_flip_motif.std(axis=0)
    mean_s = seq_flip_scaffold.mean(axis=0)
    std_s = seq_flip_scaffold.std(axis=0)

    ax.plot(times, mean_m, label="Motif", color="blue", linewidth=2)
    ax.fill_between(times, mean_m - std_m, mean_m + std_m, alpha=0.3, color="blue")
    ax.plot(times, mean_s, label="Scaffold", color="orange", linewidth=2)
    ax.fill_between(times, mean_s - std_s, mean_s + std_s, alpha=0.3, color="orange")

    ax.set_xlabel("Time t")
    ax.set_ylabel("Flip Rate")
    ax.set_title("Sequence: Fraction of Tokens Different from Deterministic")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Note about sequence behavior
    ax = axes[1]
    ax.axis("off")
    note_text = """
Sequence Noise Behavior
=======================

Unlike translation/rotation which use continuous
Brownian bridges, sequences use CTMC (Continuous
Time Markov Chain) with:

- Uniform substitution model (equal rates between all tokens)
- Rate parameter beta = {:.1f}
- noise_scale affects mixing with uniform distribution

The flip rate measures how often the stochastic
trajectory differs from the deterministic one.

Note: Motif aatypes are typically FIXED (not sampled
from prior), so motif flip rate should be ~0.

If motif flip rate > 0, it may indicate:
1. The test data doesn't properly mark motifs
2. Post-processing isn't fixing motif tokens
""".format(
        cfg.aatypes_coupler.beta
    )

    ax.text(
        0.1,
        0.95,
        note_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    fig2.tight_layout()
    # Figure 3: Geometry scale over corruption time
    fig3, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    mean_rg = geom_all_rg.mean(axis=0)
    std_rg = geom_all_rg.std(axis=0)
    ax.plot(times, mean_rg, label="All rg", linewidth=2)
    ax.fill_between(times, mean_rg - std_rg, mean_rg + std_rg, alpha=0.3)
    mean_rg_s = geom_scaffold_rg.mean(axis=0)
    std_rg_s = geom_scaffold_rg.std(axis=0)
    ax.plot(times, mean_rg_s, label="Scaffold rg", linewidth=2)
    ax.fill_between(times, mean_rg_s - std_rg_s, mean_rg_s + std_rg_s, alpha=0.3)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Radius of gyration (A)")
    ax.set_title("Corruption: geometry scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    mean_p95 = geom_all_pairwise_p95.mean(axis=0)
    std_p95 = geom_all_pairwise_p95.std(axis=0)
    ax.plot(times, mean_p95, label="All pairwise p95", linewidth=2)
    ax.fill_between(times, mean_p95 - std_p95, mean_p95 + std_p95, alpha=0.3)
    mean_p95_s = geom_scaffold_pairwise_p95.mean(axis=0)
    std_p95_s = geom_scaffold_pairwise_p95.std(axis=0)
    ax.plot(times, mean_p95_s, label="Scaffold pairwise p95", linewidth=2)
    ax.fill_between(times, mean_p95_s - std_p95_s, mean_p95_s + std_p95_s, alpha=0.3)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Pairwise distance p95 (A)")
    ax.set_title("Corruption: pairwise distance tail")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    return fig1, fig2, fig3


# ============================================================
# PART 3: SAMPLING ANALYSIS (requires model checkpoint)
# ============================================================


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if x.shape != mask.shape:
        raise ValueError(
            f"Expected x and mask to have same shape; got {tuple(x.shape)} vs {tuple(mask.shape)}"
        )
    mask_f = mask.to(dtype=x.dtype)
    denom = mask_f.sum().clamp_min(1.0)
    return (x * mask_f).sum() / denom


def _trans_rg(trans: torch.Tensor, mask: torch.Tensor) -> float:
    if trans.ndim != 3 or trans.shape[-1] != 3:
        raise ValueError(f"Expected trans shape (B, P, 3); got {tuple(trans.shape)}")
    if mask.shape != trans.shape[:2]:
        raise ValueError(
            f"Expected mask shape (B, P) to match trans prefix; got {tuple(mask.shape)} vs {tuple(trans.shape)}"
        )
    mask_f = mask.to(dtype=trans.dtype).unsqueeze(-1)
    denom = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (trans * mask_f).sum(dim=1, keepdim=True) / denom
    rg2 = ((trans - mean).square().sum(dim=-1) * mask_f.squeeze(-1)).sum(
        dim=1
    ) / denom.squeeze(-1)
    return rg2.sqrt().mean().item()


def _neighbor_distance_stats(
    trans: torch.Tensor,
    mask: torch.Tensor,
    thresholds: Tuple[float, float] = (1.5, 3.8),
) -> Dict[str, float]:
    if trans.ndim != 3 or trans.shape[-1] != 3:
        raise ValueError(f"Expected trans shape (B, P, 3); got {tuple(trans.shape)}")
    if mask.shape != trans.shape[:2]:
        raise ValueError(
            f"Expected mask shape (B, P) to match trans prefix; got {tuple(mask.shape)} vs {tuple(trans.shape)}"
        )
    valid_pairs = mask[:, 1:] & mask[:, :-1]
    d = (trans[:, 1:] - trans[:, :-1]).norm(dim=-1)  # (B, P-1)
    out: Dict[str, float] = {
        "neighbor_mean": float(_masked_mean(d, valid_pairs).item())
    }
    for thr in thresholds:
        out[f"neighbor_frac_gt_{thr:g}"] = float(
            _masked_mean((d > float(thr)).to(d.dtype), valid_pairs).item()
        )
    return out


def _rot_rg(rotmats: torch.Tensor, mask: torch.Tensor) -> float:
    if rotmats.ndim != 4 or rotmats.shape[-2:] != (3, 3):
        raise ValueError(
            f"Expected rotmats shape (B, P, 3, 3); got {tuple(rotmats.shape)}"
        )
    rotvec = rotmat_to_rotvec(rotmats)  # (B, P, 3)
    return _trans_rg(rotvec, mask)


def _mean_rot_residual_deg(
    rotmats_t: torch.Tensor, rotmats_1_pred: torch.Tensor, mask: torch.Tensor
) -> float:
    r_diff = torch.einsum("...ij,...kj->...ik", rotmats_1_pred, rotmats_t)
    angles = angle_from_rotmat(r_diff.reshape(-1, 3, 3))[0].reshape(
        r_diff.shape[:-2]
    )  # (B, P)
    return float(torch.rad2deg(_masked_mean(angles, mask)).item())


def analyze_sampling_trajectory(
    cfg: VarcoConfig,
    module: BranchFlowModule,
    data: DataBatch,
    num_steps: int,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run sampling and collect per-step statistics.
    """

    interpolant = TreeInterpolant(cfg=cfg.interpolant, device=device)
    interpolant.set_device(device)

    # Initialize batch
    batch = interpolant._init_sampling_batch(data=data)
    batch = batch.to(device)

    num_batch = batch.t.shape[0]
    metrics_cfg = MetricsCfg()

    # Statistics collectors
    stats = {
        "t": [],
        "trans_step_norm": [],
        # geometry (post-indel, after recenter)
        "geom_rg_all": [],
        "geom_max_r_all": [],
        "geom_pairwise_p95_all": [],
        "geom_rg_scaffold": [],
        "geom_pairwise_p95_scaffold": [],
        # drift/noise decomposition
        "trans_pred_gap_mean": [],
        "trans_drift_step_mean": [],
        "trans_drift_cap_frac": [],
        "trans_noise_sigma": [],
        "trans_noise_step_mean": [],
        "trans_guidance_norm_motif": [],
        "trans_guidance_over_drift_motif": [],
        "rot_scaling_mean": [],
        "rot_drift_cap_frac": [],
        "rot_noise_sigma": [],
        "rot_noise_angle_deg": [],
        "aa_entropy_mean": [],
        "aa_offdiag_mass_mean": [],
        "aa_offdiag_at_cap_frac": [],
        # expected event intensities (scaffold-only)
        "I_split_mean": [],
        "split_rate_mean": [],
        "p_ins_sum": [],
        "p_del_final_mean": [],
        "delta_H": [],
        "p_del_sum": [],
        "trans_rg_xt": [],
        "trans_rg_x1_pred": [],
        "trans_x1_pred_minus_xt_mean": [],
        "trans_neighbor_mean_xt": [],
        "trans_neighbor_frac_gt_1.5_xt": [],
        "trans_neighbor_frac_gt_3.8_xt": [],
        "trans_neighbor_mean_x1_pred": [],
        "trans_neighbor_frac_gt_1.5_x1_pred": [],
        "trans_neighbor_frac_gt_3.8_x1_pred": [],
        "rot_step_angle": [],
        "rot_rg_xt": [],
        "rot_rg_x1_pred": [],
        "rot_x1_pred_minus_xt_mean_deg": [],
        "aatype_flips": [],
        "insertions": [],
        "deletions": [],
        "length": [],
        "motif_trans_rmsd": [],
        "motif_rot_error": [],
    }

    model = module.model
    model.eval()

    with torch.no_grad():
        t_grid = torch.linspace(
            interpolant.min_t, 1.0, steps=num_steps + 1, device=device
        )

        prev_trans = batch.trans_t.clone()
        prev_rotmats = batch.rotmats_t.clone()
        prev_aatypes = batch.aatypes_t.clone()
        prev_length = batch.valid_mask.sum().item()

        pbar = tqdm(range(num_steps), desc="Sampling analysis")
        for step_num in pbar:
            t_val = float(t_grid[step_num].item())
            t_next = float(t_grid[step_num + 1].item())
            dt = float(max(1e-6, t_next - t_val))

            batch.t = torch.full(
                (num_batch,), t_val, dtype=torch.float32, device=device
            )
            pred = model.forward(batch)

            valid_now = batch.valid_mask

            trans_rg_xt = _trans_rg(batch.trans_t, valid_now)
            trans_rg_x1_pred = _trans_rg(pred.pred_trans_1, valid_now)
            trans_x1_pred_minus_xt_mean = float(
                _masked_mean(
                    (pred.pred_trans_1 - batch.trans_t).norm(dim=-1), valid_now
                ).item()
            )
            neigh_xt = _neighbor_distance_stats(batch.trans_t, valid_now)
            neigh_x1 = _neighbor_distance_stats(pred.pred_trans_1, valid_now)

            rot_rg_xt = _rot_rg(batch.rotmats_t, valid_now)
            rot_rg_x1_pred = _rot_rg(pred.pred_rotmats_1, valid_now)
            rot_x1_pred_minus_xt_mean_deg = _mean_rot_residual_deg(
                rotmats_t=batch.rotmats_t,
                rotmats_1_pred=pred.pred_rotmats_1,
                mask=valid_now,
            )

            # Compute guidance
            trans_guidance_vf, rotmats_guidance_vf = interpolant.compute_motif_pull_vf(
                t=batch.t,
                pred_trans_1=pred.pred_trans_1,
                trans_1_motifs=batch.trans_1_motifs,
                pred_rotmats_1=pred.pred_rotmats_1,
                rotmats_t=batch.rotmats_t,
                rotmats_1_motifs=batch.rotmats_1_motifs,
                motif_mask=batch.motif_mask,
            )

            # ------------------------------------------------------------
            # Diagnostics: drift/noise decomposition and cap saturation
            # ------------------------------------------------------------
            B, P = batch.trans_t.shape[:2]

            # Translation drift: v = (pred - x_t)/(1-t) (+ guidance), with drift-step cap.
            denom = (1.0 - batch.t).clamp_min(1e-4).view(B, 1, 1)
            base_v = (pred.pred_trans_1 - batch.trans_t) / denom  # (B, P, 3)
            v = base_v if trans_guidance_vf is None else (base_v + trans_guidance_vf)
            drift_step_unclipped = v * float(dt)  # (B, P, 3)

            drift_cap = float(cfg.interpolant.trans_coupler.drift_step_cap_ang)
            if drift_cap > 0.0:
                step_norm = drift_step_unclipped.norm(dim=-1, keepdim=True).clamp_min(
                    1e-6
                )
                shrink = (drift_cap / step_norm).clamp(max=1.0)
                drift_step = drift_step_unclipped * shrink
                cap_hit = (shrink.squeeze(-1) < 0.999) & valid_now
                trans_drift_cap_frac = (
                    float(cap_hit.float().mean().item()) if valid_now.any() else 0.0
                )
            else:
                drift_step = drift_step_unclipped
                trans_drift_cap_frac = 0.0

            trans_pred_gap_mean = float(
                _masked_mean(
                    (pred.pred_trans_1 - batch.trans_t).norm(dim=-1), valid_now
                ).item()
            )
            trans_drift_step_mean = float(
                _masked_mean(drift_step.norm(dim=-1), valid_now).item()
            )

            if float(cfg.interpolant.trans_coupler.noise_scale) > 0.0:
                sigma_t = interpolant.translation_coupler._compute_sigma_t(
                    t=batch.t,
                    scale=torch.full_like(
                        batch.t, float(cfg.interpolant.trans_coupler.noise_scale)
                    ),
                    min_sigma=0.0,
                    noise_end_t=float(cfg.interpolant.trans_coupler.noise_end_t),
                )
                trans_noise_sigma = float(
                    (sigma_t * math.sqrt(float(dt))).mean().item()
                )
            else:
                trans_noise_sigma = 0.0

            if trans_guidance_vf is not None and batch.motif_mask.any():
                g_norm = trans_guidance_vf.norm(dim=-1)  # (B, P)
                trans_guidance_norm_motif = float(
                    g_norm[batch.motif_mask].mean().item()
                )
                base_norm = base_v.norm(dim=-1).clamp_min(1e-8)
                ratio = (g_norm / base_norm).clamp(max=100.0)
                trans_guidance_over_drift_motif = float(
                    ratio[batch.motif_mask].mean().item()
                )
            else:
                trans_guidance_norm_motif = 0.0
                trans_guidance_over_drift_motif = 0.0

            # Deterministic translation step (no noise), for measuring realized noise.
            valid_fmask = (
                (batch.birth_time <= batch.t[:, None]).bool().float().unsqueeze(-1)
            )  # (B, P, 1)
            trans_det_next = (
                batch.trans_t + drift_step
            ) * valid_fmask + batch.trans_t * (1.0 - valid_fmask)

            # Rotation scaling + deterministic step (no IGSO3 noise), for measuring realized noise.
            if float(cfg.interpolant.rotation_coupler.exp_rate) > 0:
                r = float(cfg.interpolant.rotation_coupler.exp_rate)
                denom_exp = 1.0 - torch.exp(-r * (1.0 - batch.t))
                scaling = (r / denom_exp.clamp_min(1e-8)).clamp_min(1e-4)
            else:
                scaling = (1.0 / (1.0 - batch.t).clamp_min(1e-4)).clamp_min(1e-4)
            rot_scaling_mean = float(scaling.mean().item())

            rot_vf = calc_rot_vf(mat_t=batch.rotmats_t, mat_1=pred.pred_rotmats_1)
            if rotmats_guidance_vf is not None:
                rot_vf = rot_vf + rotmats_guidance_vf

            drift_step_cap_rad = float(
                cfg.interpolant.rotation_coupler.drift_step_cap_rad
            )
            rot_step_vec = rot_vf * (scaling.view(B, 1, 1) * float(dt))  # (B, P, 3)
            if drift_step_cap_rad > 0.0:
                step_norm = rot_step_vec.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                shrink = (drift_step_cap_rad / step_norm).clamp(max=1.0)
                rot_step_vec = rot_step_vec * shrink
                rot_cap_hit = (shrink.squeeze(-1) < 0.999) & valid_now
                rot_drift_cap_frac = (
                    float(rot_cap_hit.float().mean().item()) if valid_now.any() else 0.0
                )
                rot_vf = rot_step_vec / (scaling.view(B, 1, 1) * float(dt) + 1e-8)
            else:
                rot_drift_cap_frac = 0.0

            geodesic_time = (scaling * dt)[:, None, None]  # (B, 1, 1)
            rot_det_next = geodesic_t(
                t=geodesic_time,
                mat=pred.pred_rotmats_1,
                base_mat=batch.rotmats_t,
                rot_vf=rot_vf,
            )

            if float(cfg.interpolant.rotation_coupler.noise_scale) > 0.0:
                sigma_t = interpolant.rotation_coupler._compute_sigma_t(
                    t=batch.t,
                    scale=torch.full_like(
                        batch.t, float(cfg.interpolant.rotation_coupler.noise_scale)
                    ),
                    min_sigma=0.0,
                    noise_end_t=float(cfg.interpolant.rotation_coupler.noise_end_t),
                ).clamp_max(float(cfg.interpolant.rotation_coupler.igso3_sigma_max))
                rot_noise_sigma = float((sigma_t * math.sqrt(float(dt))).mean().item())
            else:
                rot_noise_sigma = 0.0

            # AATypes diagnostics
            probs = torch.softmax(pred.pred_aatype_logits, dim=-1)
            entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)  # (B, P)
            aa_entropy_mean = float(_masked_mean(entropy, valid_now).item())

            valid_birth = batch.birth_time <= batch.t[:, None]
            step_probs = interpolant.aatypes_coupler._compute_step_probs(
                logits=pred.pred_aatype_logits,
                x_t=batch.aatypes_t,
                t=batch.t,
                dt=dt,
                valid_mask=valid_birth,
            )
            cur = batch.aatypes_t.long().clamp(0, step_probs.shape[-1] - 1)
            p_stay = step_probs.gather(-1, cur.unsqueeze(-1)).squeeze(-1)  # (B, P)
            offdiag_mass = (1.0 - p_stay).clamp(0.0, 1.0)
            aa_offdiag_mass_mean = float(_masked_mean(offdiag_mass, valid_now).item())
            leave_cap = float(cfg.interpolant.aatypes_coupler.leave_mass_cap)
            if leave_cap > 0.0 and valid_now.any():
                aa_offdiag_at_cap_frac = float(
                    _masked_mean(
                        (offdiag_mass > 0.95 * leave_cap).float(), valid_now
                    ).item()
                )
            else:
                aa_offdiag_at_cap_frac = 0.0

            # Euler steps
            trans_next = interpolant.translation_coupler.euler_step(
                x_t=batch.trans_t,
                x1_pred=pred.pred_trans_1,
                t=batch.t,
                dt=dt,
                birth_time=batch.birth_time,
                motif_mask=batch.motif_mask,
                potential=trans_guidance_vf,
            )

            rotmats_next = interpolant.rotation_coupler.euler_step(
                x_t=batch.rotmats_t,
                x1_pred=pred.pred_rotmats_1,
                t=batch.t,
                dt=dt,
                birth_time=batch.birth_time,
                motif_mask=batch.motif_mask,
                potential=rotmats_guidance_vf,
            )

            aatypes_next = interpolant.aatypes_coupler.euler_step(
                x_t=batch.aatypes_t,
                x1_pred=pred.pred_aatype_logits,
                t=batch.t,
                dt=dt,
                birth_time=batch.birth_time,
                motif_mask=batch.motif_mask,
            )

            # Compute step statistics (before applying indels)
            valid = batch.valid_mask

            # Translation step norm
            trans_diff = (trans_next - batch.trans_t).norm(dim=-1)  # (B, P)
            trans_step = trans_diff[valid].mean().item() if valid.any() else 0.0
            trans_noise_step_mean = float(
                _masked_mean((trans_next - trans_det_next).norm(dim=-1), valid).item()
            )

            # Rotation step angle
            R_diff = torch.einsum("...ij,...kj->...ik", rotmats_next, batch.rotmats_t)
            angles = angle_from_rotmat(R_diff.reshape(-1, 3, 3))[0].reshape(
                R_diff.shape[:-2]
            )
            rot_step = np.degrees(angles[valid].mean().item()) if valid.any() else 0.0
            r_noise = torch.einsum("...ij,...kj->...ik", rotmats_next, rot_det_next)
            noise_angles = angle_from_rotmat(r_noise.reshape(-1, 3, 3))[0].reshape(
                r_noise.shape[:-2]
            )
            rot_noise_angle_deg = float(
                np.degrees(_masked_mean(noise_angles, valid).item())
            )

            # Sequence flips
            seq_flips = (aatypes_next != batch.aatypes_t).float()
            flip_count = seq_flips[valid].sum().item() if valid.any() else 0

            # Update batch
            batch.trans_t = trans_next
            batch.rotmats_t = rotmats_next
            batch.aatypes_t = aatypes_next

            # Sample indels
            scaffold_mask = batch.valid_mask & ~batch.motif_mask
            is_root = batch.birth_time <= 0.0

            expected_indel = _expected_indel_step_metrics(
                split_mass=pred.pred_split_mass,
                del_logits=pred.pred_del_logits,
                is_root=is_root,
                valid_mask=scaffold_mask,
                t_val=t_val,
                dt=dt,
                split_hazard=cfg.interpolant.sampling.split_hazard,
                delete_hazard=cfg.interpolant.sampling.delete_hazard,
                pred_split_pooled_log1p_mass=pred.pred_split_pooled_log1p_mass,
            )

            insertions, deletions, _ = interpolant._sample_insert_delete_substitute(
                split_mass=pred.pred_split_mass,
                del_logits=pred.pred_del_logits,
                is_root=is_root,
                valid_mask=scaffold_mask,
                t_val=t_val,
                dt=dt,
                split_hazard=cfg.interpolant.sampling.split_hazard,
                delete_hazard=cfg.interpolant.sampling.delete_hazard,
                pred_split_pooled_log1p_mass=pred.pred_split_pooled_log1p_mass,
            )

            # Enforce max length
            max_len = cfg.interpolant.sampling.max_length
            cur_lens = batch.valid_mask.sum(dim=1)
            at_limit = cur_lens >= max_len
            if at_limit.any():
                insertions = insertions & ~at_limit.unsqueeze(1)

            num_insertions = insertions.sum().item()
            num_deletions = deletions.sum().item()

            # Apply indels (this may change batch shape)
            batch, insert_mask, gather_idx = batch.apply_insertions_deletions(
                insertions=insertions,
                deletions=deletions,
                t_birth=t_next,
            )

            # Domain-specific initialization for newly inserted tokens (match TreeInterpolant.sample()).
            if insert_mask.any():
                # Add isotropic perturbation to inserted translations
                trans_noise = torch.randn_like(batch.trans_t) * 0.5
                batch.trans_t = (
                    batch.trans_t + insert_mask.unsqueeze(-1).float() * trans_noise
                )

                # Add small IGSO3 perturbation to inserted rotations
                B_new, P_new = batch.rotmats_t.shape[:2]
                interpolant.rotation_coupler._ensure_igso3_device(device)
                sigma_insert = torch.full(
                    (B_new,),
                    0.1,
                    device=interpolant.rotation_coupler.igso3.sigma_grid.device,
                )
                insert_noise = interpolant.rotation_coupler.igso3.sample(
                    sigma_insert, P_new
                ).to(device)
                rotmats_with_noise = torch.einsum(
                    "...ij,...jk->...ik", batch.rotmats_t, insert_noise
                )
                batch.rotmats_t = torch.where(
                    insert_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
                    rotmats_with_noise,
                    batch.rotmats_t,
                )

                # Sample amino acids for inserted positions from parent's insertion logits
                K = pred.pred_insertion_logits.shape[-1]
                P_old = pred.pred_insertion_logits.shape[1]
                insertion_logits_gathered = pred.pred_insertion_logits.gather(
                    1,
                    gather_idx.unsqueeze(-1).expand(-1, -1, K).clamp(0, P_old - 1),
                )  # (B, P_new, K)
                probs = torch.softmax(insertion_logits_gathered, dim=-1)
                uniform_dist = torch.ones_like(probs) / K
                probs = (
                    1.0 - float(cfg.interpolant.aatypes_coupler.noise_scale)
                ) * probs + float(
                    cfg.interpolant.aatypes_coupler.noise_scale
                ) * uniform_dist
                sampled_tokens = torch.multinomial(
                    probs.view(-1, K), num_samples=1
                ).view(B_new, P_new)
                batch.aatypes_t = torch.where(
                    insert_mask, sampled_tokens, batch.aatypes_t
                )

            # Recenter translations (matches TreeInterpolant.sample()).
            valid_mask_now = batch.valid_mask
            mask_f = valid_mask_now.float()
            denom = mask_f.sum(dim=1).clamp_min(1.0).view(batch.t.shape[0], 1, 1)
            com = (batch.trans_t * mask_f.unsqueeze(-1)).sum(
                dim=1, keepdim=True
            ) / denom
            batch.trans_t = batch.trans_t - com

            # Motif error (if motifs exist)
            motif_mask = batch.motif_mask
            if motif_mask.any():
                motif_trans_diff = (batch.trans_t - batch.trans_1_motifs).norm(dim=-1)
                motif_trans_rmsd = motif_trans_diff[motif_mask].mean().item()

                R_diff = torch.einsum(
                    "...ij,...kj->...ik", batch.rotmats_t, batch.rotmats_1_motifs
                )
                angles = angle_from_rotmat(R_diff.reshape(-1, 3, 3))[0].reshape(
                    R_diff.shape[:-2]
                )
                motif_rot_err = np.degrees(angles[motif_mask].mean().item())
            else:
                motif_trans_rmsd = 0.0
                motif_rot_err = 0.0

            new_length = batch.valid_mask.sum().item()

            geom_valid = batch.valid_mask
            geom_all = _translation_geometry_metrics(
                trans=batch.trans_t,
                mask=geom_valid,
                metrics_cfg=metrics_cfg,
                seed_offset=step_num,
            )
            geom_scaf = _translation_geometry_metrics(
                trans=batch.trans_t,
                mask=geom_valid & ~batch.motif_mask,
                metrics_cfg=metrics_cfg,
                seed_offset=step_num + 29,
            )

            # Store stats
            stats["t"].append(t_val)
            stats["trans_step_norm"].append(trans_step)
            stats["geom_rg_all"].append(geom_all["rg"])
            stats["geom_max_r_all"].append(geom_all["max_r"])
            stats["geom_pairwise_p95_all"].append(geom_all["pairwise_p95"])
            stats["geom_rg_scaffold"].append(geom_scaf["rg"])
            stats["geom_pairwise_p95_scaffold"].append(geom_scaf["pairwise_p95"])
            stats["trans_pred_gap_mean"].append(trans_pred_gap_mean)
            stats["trans_drift_step_mean"].append(trans_drift_step_mean)
            stats["trans_drift_cap_frac"].append(trans_drift_cap_frac)
            stats["trans_noise_sigma"].append(trans_noise_sigma)
            stats["trans_noise_step_mean"].append(trans_noise_step_mean)
            stats["trans_guidance_norm_motif"].append(trans_guidance_norm_motif)
            stats["trans_guidance_over_drift_motif"].append(
                trans_guidance_over_drift_motif
            )
            stats["rot_scaling_mean"].append(rot_scaling_mean)
            stats["rot_drift_cap_frac"].append(rot_drift_cap_frac)
            stats["rot_noise_sigma"].append(rot_noise_sigma)
            stats["rot_noise_angle_deg"].append(rot_noise_angle_deg)
            stats["aa_entropy_mean"].append(aa_entropy_mean)
            stats["aa_offdiag_mass_mean"].append(aa_offdiag_mass_mean)
            stats["aa_offdiag_at_cap_frac"].append(aa_offdiag_at_cap_frac)
            stats["I_split_mean"].append(expected_indel["I_split_mean"])
            stats["split_rate_mean"].append(expected_indel["split_rate_mean"])
            stats["p_ins_sum"].append(expected_indel["p_ins_sum"])
            stats["p_del_final_mean"].append(expected_indel["p_del_final_mean"])
            stats["delta_H"].append(expected_indel["delta_H"])
            stats["p_del_sum"].append(expected_indel["p_del_sum"])
            stats["trans_rg_xt"].append(trans_rg_xt)
            stats["trans_rg_x1_pred"].append(trans_rg_x1_pred)
            stats["trans_x1_pred_minus_xt_mean"].append(trans_x1_pred_minus_xt_mean)
            stats["trans_neighbor_mean_xt"].append(neigh_xt["neighbor_mean"])
            stats["trans_neighbor_frac_gt_1.5_xt"].append(
                neigh_xt["neighbor_frac_gt_1.5"]
            )
            stats["trans_neighbor_frac_gt_3.8_xt"].append(
                neigh_xt["neighbor_frac_gt_3.8"]
            )
            stats["trans_neighbor_mean_x1_pred"].append(neigh_x1["neighbor_mean"])
            stats["trans_neighbor_frac_gt_1.5_x1_pred"].append(
                neigh_x1["neighbor_frac_gt_1.5"]
            )
            stats["trans_neighbor_frac_gt_3.8_x1_pred"].append(
                neigh_x1["neighbor_frac_gt_3.8"]
            )
            stats["rot_step_angle"].append(rot_step)
            stats["rot_rg_xt"].append(rot_rg_xt)
            stats["rot_rg_x1_pred"].append(rot_rg_x1_pred)
            stats["rot_x1_pred_minus_xt_mean_deg"].append(rot_x1_pred_minus_xt_mean_deg)
            stats["aatype_flips"].append(flip_count)
            stats["insertions"].append(num_insertions)
            stats["deletions"].append(num_deletions)
            stats["length"].append(new_length)
            stats["motif_trans_rmsd"].append(motif_trans_rmsd)
            stats["motif_rot_error"].append(motif_rot_err)

            # Update prev for next step
            prev_trans = batch.trans_t.clone()
            prev_rotmats = batch.rotmats_t.clone()
            prev_aatypes = batch.aatypes_t.clone()
            prev_length = new_length

            pbar.set_postfix_str(
                f"L={new_length}, ins={num_insertions}, del={num_deletions}, "
                f"trans_rg_t={trans_rg_xt:.2f}A, "
                f"trans_|pred-t|={trans_x1_pred_minus_xt_mean:.2f}A, "
                f"rot_rg_t={rot_rg_xt:.2f}rad, "
                f"rot_err={rot_x1_pred_minus_xt_mean_deg:.1f}deg"
            )

    # Convert to numpy
    for k in stats:
        stats[k] = np.array(stats[k])

    return stats


def plot_sampling_step_distributions(stats: Dict[str, np.ndarray]) -> plt.Figure:
    """Plot per-step drift/noise distributions."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    t = stats["t"]

    # Translation step norm vs t
    ax = axes[0, 0]
    ax.plot(t, stats["trans_step_norm"], linewidth=1.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Translation Step (A)")
    ax.set_title("Per-Step Translation Displacement")
    ax.grid(True, alpha=0.3)

    # Rotation step angle vs t
    ax = axes[0, 1]
    ax.plot(t, stats["rot_step_angle"], linewidth=1.5, color="orange")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Rotation Step (degrees)")
    ax.set_title("Per-Step Rotation Angle")
    ax.grid(True, alpha=0.3)

    # Histogram of translation steps
    ax = axes[1, 0]
    ax.hist(stats["trans_step_norm"], bins=50, edgecolor="black", alpha=0.7)
    ax.set_xlabel("Translation Step (A)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of Per-Step Translation\nMean={stats['trans_step_norm'].mean():.3f} A"
    )
    ax.axvline(
        stats["trans_step_norm"].mean(), color="red", linestyle="--", label="Mean"
    )
    ax.legend()

    # Histogram of rotation steps
    ax = axes[1, 1]
    ax.hist(
        stats["rot_step_angle"], bins=50, edgecolor="black", alpha=0.7, color="orange"
    )
    ax.set_xlabel("Rotation Step (degrees)")
    ax.set_ylabel("Count")
    ax.set_title(
        f"Distribution of Per-Step Rotation\nMean={stats['rot_step_angle'].mean():.3f} deg"
    )
    ax.axvline(
        stats["rot_step_angle"].mean(), color="red", linestyle="--", label="Mean"
    )
    ax.legend()

    fig.tight_layout()
    return fig


def plot_sampling_drift_noise_diagnostics(stats: Dict[str, np.ndarray]) -> plt.Figure:
    """Plot drift/noise decomposition and cap saturation diagnostics."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    t = stats["t"]

    # Translation: drift vs noise (mean norms)
    ax = axes[0, 0]
    ax.plot(t, stats["trans_drift_step_mean"], label="drift_step", linewidth=2)
    ax.plot(
        t, stats["trans_noise_step_mean"], label="noise_step (measured)", linewidth=2
    )
    ax.plot(
        t,
        stats["trans_noise_sigma"] * np.sqrt(3.0),
        label="sigma*sqrt(3) (expected)",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Step norm (A)")
    ax.set_title("Translation: drift vs noise (pre-indel)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Translation: prediction gap and cap fraction
    ax = axes[0, 1]
    ax.plot(
        t, stats["trans_pred_gap_mean"], label="||pred_trans_1 - x_t||", linewidth=2
    )
    ax2 = ax.twinx()
    ax2.plot(
        t, stats["trans_drift_cap_frac"], label="cap frac", color="orange", linewidth=2
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Pred gap (A)", color="tab:blue")
    ax2.set_ylabel("Cap fraction", color="tab:orange")
    ax.set_title("Translation: stiffness indicators")
    ax.grid(True, alpha=0.3)

    # Rotation: scaling and cap fraction
    ax = axes[1, 0]
    ax.plot(t, stats["rot_scaling_mean"], label="scaling", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(
        t, stats["rot_drift_cap_frac"], label="cap frac", color="orange", linewidth=2
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Scaling", color="tab:blue")
    ax2.set_ylabel("Cap fraction", color="tab:orange")
    ax.set_title("Rotation: scaling and cap saturation")
    ax.grid(True, alpha=0.3)

    # Rotation: sigma vs measured noise angle
    ax = axes[1, 1]
    ax.plot(t, stats["rot_noise_angle_deg"], label="noise angle (deg)", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(
        t, stats["rot_noise_sigma"], label="sigma*sqrt(dt)", color="orange", linewidth=2
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Noise angle (deg)", color="tab:blue")
    ax2.set_ylabel("Noise sigma", color="tab:orange")
    ax.set_title("Rotation: stochasticity near end")
    ax.grid(True, alpha=0.3)

    # AATypes: entropy and expected leave mass
    ax = axes[2, 0]
    ax.plot(t, stats["aa_entropy_mean"], label="entropy", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(
        t,
        stats["aa_offdiag_mass_mean"],
        label="E[leave mass]",
        color="orange",
        linewidth=2,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Entropy (nats)", color="tab:blue")
    ax2.set_ylabel("Off-diagonal mass", color="tab:orange")
    ax.set_title("Sequence: uncertainty and step size")
    ax.grid(True, alpha=0.3)

    # Guidance: magnitude and ratio on motifs
    ax = axes[2, 1]
    ax.plot(
        t,
        stats["trans_guidance_norm_motif"],
        label="||guidance|| (motifs)",
        linewidth=2,
    )
    ax2 = ax.twinx()
    ax2.plot(
        t,
        stats["trans_guidance_over_drift_motif"],
        label="guidance/drift (motifs)",
        color="orange",
        linewidth=2,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Guidance norm", color="tab:blue")
    ax2.set_ylabel("Guidance/drift", color="tab:orange")
    ax.set_title("Motif guidance strength over time")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_sampling_geometry(stats: Dict[str, np.ndarray]) -> plt.Figure:
    """Plot sampling geometry scale over time (post-indel)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t = stats["t"]

    ax = axes[0]
    ax.plot(t, stats["geom_rg_all"], label="All rg", linewidth=2)
    ax.plot(t, stats["geom_rg_scaffold"], label="Scaffold rg", linewidth=2)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Radius of gyration (A)")
    ax.set_title("Sampling: geometry scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(t, stats["geom_pairwise_p95_all"], label="All pairwise p95", linewidth=2)
    ax.plot(
        t,
        stats["geom_pairwise_p95_scaffold"],
        label="Scaffold pairwise p95",
        linewidth=2,
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Pairwise distance p95 (A)")
    ax.set_title("Sampling: pairwise distance tail")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_sampling_event_intensity(stats: Dict[str, np.ndarray]) -> plt.Figure:
    """Plot expected per-step event counts vs realized event counts."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    t = stats["t"]

    ax = axes[0, 0]
    ax.plot(t, stats["I_split_mean"], linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(t, stats["split_rate_mean"], color="orange", linewidth=2)
    ax.set_xlabel("Time t")
    ax.set_ylabel("I_split")
    ax2.set_ylabel("mean(split_rate)", color="tab:orange")
    ax.set_title("Insertion intensity components")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(t, stats["p_ins_sum"], label="E[insertions] = sum(p_ins)", linewidth=2)
    ax.plot(t, stats["insertions"], label="sampled insertions", linewidth=2, alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Count / step")
    ax.set_title("Insertions: expected vs realized")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(t, stats["p_del_sum"], label="E[deletions] = sum(p_del)", linewidth=2)
    ax.plot(t, stats["deletions"], label="sampled deletions", linewidth=2, alpha=0.7)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Count / step")
    ax.set_title("Deletions: expected vs realized")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(t, stats["p_del_final_mean"], label="mean(p_del_final)", linewidth=2)
    ax2 = ax.twinx()
    ax2.plot(
        t, stats["delta_H"], label="ΔH = H(t+dt)-H(t)", color="orange", linewidth=2
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("p_del_final", color="tab:blue")
    ax2.set_ylabel("ΔH", color="tab:orange")
    ax.set_title("Deletion intensity components")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def analyze_corruption_trajectory_metrics(
    cfg: VarcoInterpolantConfig,
    data: DataBatch,
    times: List[float],
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """Collect corruption-time metrics (training distribution proxy) on the same data."""
    interpolant = TreeInterpolant(cfg=cfg, device=device)
    interpolant.set_device(device)
    metrics_cfg = MetricsCfg()

    if data.trans_1.device != device:
        data = DataBatch(
            tree=data.tree.to(device),
            motif_mask=data.motif_mask.to(device),
            res_mask=data.res_mask.to(device),
            chain_idx=data.chain_idx.to(device),
            trans_1=data.trans_1.to(device),
            rotmats_1=data.rotmats_1.to(device),
            aatypes_1=data.aatypes_1.to(device),
            contact_conditioning=data.contact_conditioning.to(device),
            res_bfactor=data.res_bfactor.to(device),
            res_plddt=data.res_plddt.to(device),
        )

    t_eff = [
        float(np.clip(float(t), interpolant.min_t, 1.0 - interpolant.min_t))
        for t in times
    ]
    traj, _ = interpolant.corrupt_trajectory(batch=data, times=t_eff, seed=0)

    out: Dict[str, List[float]] = {
        "t": [],
        "geom_rg_all": [],
        "geom_pairwise_p95_all": [],
        "geom_rg_scaffold": [],
        "geom_pairwise_p95_scaffold": [],
    }

    for i, sample in enumerate(traj.samples):
        valid = sample.valid_mask
        geom_all = _translation_geometry_metrics(
            trans=sample.trans_t,
            mask=valid,
            metrics_cfg=metrics_cfg,
            seed_offset=i,
        )
        geom_scaf = _translation_geometry_metrics(
            trans=sample.trans_t,
            mask=valid & ~sample.motif_mask,
            metrics_cfg=metrics_cfg,
            seed_offset=i + 31,
        )
        out["t"].append(float(t_eff[i]))
        out["geom_rg_all"].append(geom_all["rg"])
        out["geom_pairwise_p95_all"].append(geom_all["pairwise_p95"])
        out["geom_rg_scaffold"].append(geom_scaf["rg"])
        out["geom_pairwise_p95_scaffold"].append(geom_scaf["pairwise_p95"])

    return {k: np.array(v) for k, v in out.items()}


def plot_train_vs_sample_geometry_shift(
    corruption: Dict[str, np.ndarray], sampling: Dict[str, np.ndarray]
) -> plt.Figure:
    """Overlay geometry metrics between corruption (training proxy) and sampling rollout."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    t_c = corruption["t"]
    t_s = sampling["t"]

    ax = axes[0]
    ax.plot(t_c, corruption["geom_rg_all"], label="corrupt rg (all)", linewidth=2)
    ax.plot(t_s, sampling["geom_rg_all"], label="sample rg (all)", linewidth=2)
    ax.plot(
        t_c,
        corruption["geom_rg_scaffold"],
        label="corrupt rg (scaffold)",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        t_s,
        sampling["geom_rg_scaffold"],
        label="sample rg (scaffold)",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Radius of gyration (A)")
    ax.set_title("Train vs sample: geometry scale")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(
        t_c, corruption["geom_pairwise_p95_all"], label="corrupt p95 (all)", linewidth=2
    )
    ax.plot(
        t_s, sampling["geom_pairwise_p95_all"], label="sample p95 (all)", linewidth=2
    )
    ax.plot(
        t_c,
        corruption["geom_pairwise_p95_scaffold"],
        label="corrupt p95 (scaffold)",
        linewidth=2,
        linestyle="--",
    )
    ax.plot(
        t_s,
        sampling["geom_pairwise_p95_scaffold"],
        label="sample p95 (scaffold)",
        linewidth=2,
        linestyle="--",
    )
    ax.set_xlabel("Time t")
    ax.set_ylabel("Pairwise distance p95 (A)")
    ax.set_title("Train vs sample: distance tail")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_indel_statistics(stats: Dict[str, np.ndarray]) -> plt.Figure:
    """Plot insertion/deletion behavior."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    t = stats["t"]

    # Cumulative insertions/deletions
    ax = axes[0, 0]
    cum_ins = np.cumsum(stats["insertions"])
    cum_del = np.cumsum(stats["deletions"])
    ax.plot(t, cum_ins, label="Insertions", linewidth=2)
    ax.plot(t, cum_del, label="Deletions", linewidth=2)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Cumulative Count")
    ax.set_title(
        f"Cumulative Indels (Total: {cum_ins[-1]:.0f} ins, {cum_del[-1]:.0f} del)"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Length trajectory
    ax = axes[0, 1]
    ax.plot(t, stats["length"], linewidth=2, color="green")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Sequence Length")
    ax.set_title(
        f"Length Trajectory (Start={stats['length'][0]:.0f}, End={stats['length'][-1]:.0f})"
    )
    ax.grid(True, alpha=0.3)

    # Per-step indel rate
    ax = axes[1, 0]
    ax.plot(t, stats["insertions"], label="Insertions/step", alpha=0.7)
    ax.plot(t, stats["deletions"], label="Deletions/step", alpha=0.7)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Count per Step")
    ax.set_title("Indel Rate Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Net length change
    ax = axes[1, 1]
    net_change = np.cumsum(stats["insertions"] - stats["deletions"])
    ax.plot(t, net_change, linewidth=2, color="purple")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Net Length Change")
    ax.set_title("Cumulative Net Length Change (Insertions - Deletions)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_sequence_evolution(stats: Dict[str, np.ndarray]) -> plt.Figure:
    """Plot sequence changes over time."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    t = stats["t"]

    # Token flips per step
    ax = axes[0]
    ax.plot(t, stats["aatype_flips"], linewidth=1.5)
    ax.set_xlabel("Time t")
    ax.set_ylabel("Token Flips")
    ax.set_title("Sequence Token Changes Per Step")
    ax.grid(True, alpha=0.3)

    # Cumulative token flips
    ax = axes[1]
    cum_flips = np.cumsum(stats["aatype_flips"])
    ax.plot(t, cum_flips, linewidth=2, color="orange")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Cumulative Flips")
    ax.set_title(f"Cumulative Token Changes (Total: {cum_flips[-1]:.0f})")
    ax.grid(True, alpha=0.3)

    # Motif error (if available)
    ax = axes[2]
    if stats["motif_trans_rmsd"].max() > 0:
        ax.plot(t, stats["motif_trans_rmsd"], label="Trans RMSD (A)", linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(
            t,
            stats["motif_rot_error"],
            label="Rot Error (deg)",
            color="orange",
            linewidth=2,
        )
        ax.set_xlabel("Time t")
        ax.set_ylabel("Translation RMSD (A)", color="tab:blue")
        ax2.set_ylabel("Rotation Error (deg)", color="tab:orange")
        ax.set_title("Motif Reconstruction Error")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No motifs in sample",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_title("Motif Error (N/A)")

    fig.tight_layout()
    return fig


# ============================================================
# PART 4: MAIN ENTRY POINT
# ============================================================


@hydra.main(config_path=".", config_name="varco", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg_obj = VarcoConfig.from_dict_config(cfg).interpolate()

    # Get additional parameters (can be passed via +param=value)
    output_dir = OmegaConf.select(
        cfg, "output_dir", default="varco/outputs/noise_analysis"
    )
    num_samples = int(OmegaConf.select(cfg, "num_samples", default=10))
    num_steps = int(OmegaConf.select(cfg, "num_steps", default=200))

    output_path = Path(str(output_dir))
    if not output_path.is_absolute():
        output_path = Path(cfg_obj.shared.project_root) / output_path
    output_dir = str(output_path)

    # Determine device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, f"{output_dir}/config.yaml")

    print("=" * 60)
    print("NOISE ANALYSIS - Config Values")
    print("=" * 60)
    print(f"trans_coupler.noise_scale: {cfg_obj.interpolant.trans_coupler.noise_scale}")
    print(f"trans_coupler.noise_end_t: {cfg_obj.interpolant.trans_coupler.noise_end_t}")
    print(
        f"rotation_coupler.noise_scale: {cfg_obj.interpolant.rotation_coupler.noise_scale}"
    )
    print(
        f"rotation_coupler.noise_end_t: {cfg_obj.interpolant.rotation_coupler.noise_end_t}"
    )
    print(
        f"aatypes_coupler.noise_scale: {cfg_obj.interpolant.aatypes_coupler.noise_scale}"
    )
    print(
        f"aatypes_coupler.noise_end_t: {cfg_obj.interpolant.aatypes_coupler.noise_end_t}"
    )
    print(f"num_steps: {num_steps}")
    print(f"📁 output_dir: {output_dir}")
    print(f"💾 inference.ckpt_path: {cfg_obj.inference.ckpt_path}")
    print("=" * 60)

    # Theoretical plots
    print("Generating theoretical noise profiles...")
    fig1 = plot_bridge_variance_profiles(cfg_obj.interpolant)
    fig1.savefig(f"{output_dir}/01_bridge_variance.png", dpi=150, bbox_inches="tight")
    plt.close(fig1)

    print("Generating sigma schedule plots...")
    fig2 = plot_sampling_sigma_schedule(cfg_obj.interpolant, num_steps)
    fig2.savefig(f"{output_dir}/02_sigma_schedule.png", dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print("Generating per-step noise comparison...")
    fig3 = plot_per_step_noise_comparison(cfg_obj.interpolant)
    fig3.savefig(
        f"{output_dir}/03_per_step_vs_num_steps.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig3)

    # Corruption analysis
    print(f"Analyzing corruption noise ({num_samples} samples)...")
    fig4, fig5, fig6 = analyze_corruption_noise(
        cfg_obj.interpolant, num_samples, num_steps, device
    )
    fig4.savefig(
        f"{output_dir}/04_corruption_trans_rot.png", dpi=150, bbox_inches="tight"
    )
    fig5.savefig(
        f"{output_dir}/05_corruption_sequence.png", dpi=150, bbox_inches="tight"
    )
    fig6.savefig(
        f"{output_dir}/06_corruption_geometry.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig4)
    plt.close(fig5)
    plt.close(fig6)

    # Sampling analysis (if checkpoint provided)
    ckpt_path = cfg_obj.inference.ckpt_path
    if ckpt_path:
        print(f"\nRunning sampling analysis with checkpoint: {ckpt_path}")

        # Resolve path
        if not os.path.isabs(ckpt_path):
            ckpt_path = str(Path(cfg_obj.shared.project_root) / ckpt_path)

        if not os.path.exists(ckpt_path):
            print(f"WARNING: Checkpoint not found at {ckpt_path}")
            print("Skipping sampling analysis.")
        else:
            print(f"Loading model from {ckpt_path}...")
            module = BranchFlowModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                cfg=cfg_obj,
            )
            module = module.to(device)
            module.eval()

            # Create synthetic data for sampling
            data = create_synthetic_databatch(N=30, B=1, device=device, seed=42)

            print("Running sampling trajectory analysis...")
            stats = analyze_sampling_trajectory(
                cfg_obj, module, data, num_steps, device
            )

            fig7 = plot_sampling_step_distributions(stats)
            fig7.savefig(
                f"{output_dir}/07_sampling_steps.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig7)

            fig8 = plot_indel_statistics(stats)
            fig8.savefig(
                f"{output_dir}/08_indel_stats.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig8)

            fig9 = plot_sequence_evolution(stats)
            fig9.savefig(
                f"{output_dir}/09_sequence_evolution.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig9)

            fig10 = plot_sampling_drift_noise_diagnostics(stats)
            fig10.savefig(
                f"{output_dir}/10_sampling_drift_noise.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig10)

            fig11 = plot_sampling_geometry(stats)
            fig11.savefig(
                f"{output_dir}/11_sampling_geometry.png", dpi=150, bbox_inches="tight"
            )
            plt.close(fig11)

            times = [float(x) for x in stats["t"].tolist()]
            corrupt_metrics = analyze_corruption_trajectory_metrics(
                cfg=cfg_obj.interpolant,
                data=data,
                times=times,
                device=device,
            )
            fig12 = plot_train_vs_sample_geometry_shift(corrupt_metrics, stats)
            fig12.savefig(
                f"{output_dir}/12_train_vs_sample_geometry.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig12)

            fig13 = plot_sampling_event_intensity(stats)
            fig13.savefig(
                f"{output_dir}/13_sampling_event_intensity.png",
                dpi=150,
                bbox_inches="tight",
            )
            plt.close(fig13)

            # Save raw stats
            with open(f"{output_dir}/sampling_stats.json", "w") as f:
                json.dump({k: v.tolist() for k, v in stats.items()}, f, indent=2)

            print(f"Sampling stats saved to {output_dir}/sampling_stats.json")
    else:
        print("To run sampling analysis, use: inference.ckpt_path=path/to/ckpt")

    print(f"\nResults saved to {output_dir}/")
    print("Generated figures:")
    print("  01_bridge_variance.png - Brownian bridge variance profiles")
    print("  02_sigma_schedule.png - Sigma schedule and per-step noise")
    print("  03_per_step_vs_num_steps.png - Effect of num_steps on noise")
    print("  04_corruption_trans_rot.png - Corruption noise (trans/rot)")
    print("  05_corruption_sequence.png - Corruption noise (sequence)")
    print("  06_corruption_geometry.png - Corruption geometry scale")
    if ckpt_path and os.path.exists(ckpt_path):
        print("  07_sampling_steps.png - Per-step drift distributions")
        print("  08_indel_stats.png - Insertion/deletion statistics")
        print("  09_sequence_evolution.png - Sequence evolution")
        print("  10_sampling_drift_noise.png - Drift/noise decomposition + caps")
        print("  11_sampling_geometry.png - Sampling geometry scale")
        print("  12_train_vs_sample_geometry.png - Train vs sample geometry overlay")
        print("  13_sampling_event_intensity.png - Expected vs realized indel counts")


if __name__ == "__main__":
    main()
