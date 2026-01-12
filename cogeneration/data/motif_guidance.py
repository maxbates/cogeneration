"""
Centralized motif guidance for structure generation.

Implements twisted diffusion / FrameFlow-style motif guidance using autograd.
The potential biases the sampling trajectory toward satisfying motif constraints
by computing gradients of a log-likelihood proxy and adding them to the drift.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from cogeneration.config.base import MotifGuidanceConfig, MotifGuidanceVarScale
from cogeneration.data import so3_utils
from cogeneration.data.potentials import PotentialField
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass


@dataclass
class MotifGuidanceMetrics:
    """Metrics for motif guidance debugging and monitoring."""

    step: int
    t: float
    num_positions: int  # P (total positions)
    num_motif: int  # number of motif residues
    rmsd_raw: float  # RMSD in world frame (mean over batch)
    rmsd_aligned: float  # RMSD after alignment (mean over batch)
    sv_min: float  # minimum singular value of alignment cross-covariance
    condition_number: float  # condition number of alignment (sv_max / sv_min)
    delta_theta: float  # rotation change since last step (radians)
    trans_guidance_norm: float  # mean norm of translation guidance VF
    rot_guidance_norm: float  # mean norm of rotation guidance VF
    # Alignment rotation matrix (B, 3, 3) - pass to next step for delta_theta
    align_rot: Optional[torch.Tensor] = None

    def to_str(self) -> str:
        """Return a single-line string for progress bar output."""
        return (
            f"step={self.step:4d} | "
            f"t={self.t:6.3f} | "
            f"P={self.num_positions} | "
            f"M={self.num_motif} | "
            f"rmsd_r={self.rmsd_raw:6.3f} | "
            f"rmsd_a={self.rmsd_aligned:6.3f} | "
            f"sv_min={self.sv_min:7.3f} | "
            f"cond={self.condition_number:8.1f} | "
            f"dθ={self.delta_theta:6.3f} | "
            f"∇trans={self.trans_guidance_norm:6.3f} | "
            f"∇rot={self.rot_guidance_norm:6.3f}"
        )

    def to_postfix(self) -> str:
        """Return a compact string for tqdm postfix."""
        return (
            f"P={self.num_positions} "
            f"rmsd={self.rmsd_raw:.2f} "
            f"dθ={self.delta_theta:.2f} "
            f"cond={self.condition_number:.1e}"
        )


def motif_potential_window(
    t: torch.Tensor,
    motif_mask: torch.Tensor,
    cfg: MotifGuidanceConfig,
) -> torch.Tensor:
    """
    Compute the motif potential window scale based on time.

    Returns (B,) tensor with 0 where guidance is inactive,
    and a scale factor where active.

    Args:
        t: Current time values (B,)
        motif_mask: Boolean mask indicating motif residues (B, N)
        cfg: Motif guidance configuration

    Returns:
        Window scale tensor (B,)
    """
    if not cfg.enabled or not motif_mask.any():
        return torch.zeros_like(t)

    # Guidance window: only active on [guidance_start_t, guidance_end_t]
    start_t = cfg.guidance_start_t
    end_t = cfg.guidance_end_t
    window = (t >= start_t).float() * (t <= end_t).float()  # (B,)

    # Optional linear fade-out: decay to 0 as t -> end_t
    if cfg.var_decay:
        fade = (1.0 - t / max(end_t, 1e-6)).clamp(min=0.0)
        window = window * fade

    return window  # (B,)


def compute_motif_potential(
    t: torch.Tensor,
    trans_t: torch.Tensor,
    rotmats_t: torch.Tensor,
    pred_trans_1: torch.Tensor,
    pred_rotmats_1: torch.Tensor,
    trans_1_motifs: torch.Tensor,
    rotmats_1_motifs: torch.Tensor,
    motif_mask: torch.Tensor,
    valid_mask: torch.Tensor,  # or res_mask
    cfg: MotifGuidanceConfig,
    min_t: float = 0.01,
    align: bool = True,
    allow_none: bool = False,
    # Debug parameters - only used when cfg.debug=True
    step: int = 0,
    prev_metrics: Optional[MotifGuidanceMetrics] = None,
) -> Tuple[PotentialField, Optional[MotifGuidanceMetrics]]:
    """
    Compute twisted diffusion/FrameFlow-style motif guidance using autograd.
    Metrics are computed when cfg.debug=True

    This computes:
        potential = 0.5 * g(t)^2 / omega(t)^2 * nabla log p(motif | x_t)

    where the log-likelihood proxy is:
        log p ~ -0.5 * ||error||^2 / var(t)

    The gradient is computed via torch.autograd.grad with respect to trans_t
    and rotmats_t, then projected to tangent space for rotations.

    See FrameFlow paper for details about motif guidance (sec 3.2):
    https://arxiv.org/pdf/2401.04082

    Args:
        t: Current time values (B,)
        trans_t: Current translations with requires_grad=True (B, N, 3)
        rotmats_t: Current rotation matrices with requires_grad=True (B, N, 3, 3)
        pred_trans_1: Model predicted endpoint translations (B, N, 3)
        pred_rotmats_1: Model predicted endpoint rotations (B, N, 3, 3)
        trans_1_motifs: Target motif translations (B, N, 3)
        rotmats_1_motifs: Target motif rotations (B, N, 3, 3)
        motif_mask: Boolean mask for motif residues (B, N)
        cfg: Motif guidance configuration
        min_t: Minimum time for clamping (default 0.01)
        align: Whether to align motifs before computing loss (default True)
        allow_none: If True, return empty PotentialField instead of raising on no gradients
        valid_mask: Boolean mask for valid/present residues (B, N) - required if cfg.debug=True
        step: Current sampling step number (for debug metrics)
        prev_metrics: Previous step's metrics (for computing delta_theta)

    Returns:
        Tuple of (PotentialField, Optional[MotifGuidanceMetrics])
        Metrics are only computed and returned when cfg.debug=True
    """
    if not motif_mask.any() or not cfg.enabled:
        return PotentialField(), None

    window = motif_potential_window(t=t, motif_mask=motif_mask, cfg=cfg)
    if not (window > 0).any():
        return PotentialField(), None

    # Validate inputs
    assert (
        pred_trans_1.shape == trans_t.shape
    ), "Shape mismatch: pred_trans_1 vs trans_t"
    assert (
        pred_rotmats_1.shape == rotmats_t.shape
    ), "Shape mismatch: pred_rotmats_1 vs rotmats_t"
    assert (
        trans_t.requires_grad and rotmats_t.requires_grad
    ), "trans_t and rotmats_t must have requires_grad=True for gradient-based guidance"
    assert pred_trans_1.requires_grad and pred_rotmats_1.requires_grad, (
        "pred_trans_1 and pred_rotmats_1 must have requires_grad=True "
        "(model forward must be differentiable)"
    )

    B, N = motif_mask.shape
    eps = 1e-6
    motif_f = motif_mask.float()  # (B, N)
    t_clamped = t.clamp(min=min_t, max=1.0 - min_t)

    # Extract config values
    guidance_scale = cfg.scale_factor
    obs_noise_trans = cfg.obs_noise_trans_ang
    obs_noise_rot = cfg.obs_noise_rot_rad
    var_scale_type = cfg.var_scale_type
    var_scale_cap = cfg.var_scale_cap

    # Compute g(t) and omega^2(t) using FrameFlow formulation (Eq. 14-15)
    # g(t) = (1-t)/t, omega^2(t) = (1-t)^2 / (t^2 + (1-t)^2)
    one_minus_t = 1.0 - t_clamped
    g = (one_minus_t / t_clamped).clamp(min=0.0)
    g2 = g * g  # (B,)
    omega2 = (one_minus_t * one_minus_t) / (
        (t_clamped * t_clamped) + (one_minus_t * one_minus_t) + eps
    )
    omega2 = omega2.clamp_min(1e-3)  # avoid huge gradients from tiny omega^2

    # Determine base variance based on var_scale_type
    if var_scale_type == MotifGuidanceVarScale.ot:
        # OT-style: (1-t)/t is dimensionless time-dependent factor
        var_dim = ((1.0 - t_clamped) / t_clamped.clamp_min(1e-6)).clamp_min(0.0)
        var_trans = var_dim
        var_rot = var_dim
    elif var_scale_type == MotifGuidanceVarScale.linear:
        # Linear decay: simple (1-t) schedule
        var_dim = (1.0 - t_clamped).clamp_min(0.0)
        var_trans = var_dim
        var_rot = var_dim
    elif var_scale_type == MotifGuidanceVarScale.constant:
        # Constant: relies entirely on obs_noise floors for scale
        var_trans = torch.ones_like(t_clamped)
        var_rot = torch.ones_like(t_clamped)
    else:
        raise ValueError(f"Unknown var_scale_type: {var_scale_type}")

    # Combine with observation noise floors (prevents blow-up as var -> 0)
    var_trans = var_trans + (obs_noise_trans**2)
    var_rot = var_rot + (obs_noise_rot**2)

    # Per-domain scales (higher means stronger guidance)
    trans_scale_t = guidance_scale / var_trans.clamp_min(1e-8)  # (B,)
    rot_scale_t = guidance_scale / var_rot.clamp_min(1e-8)  # (B,)

    # Cap the scales to prevent extreme values
    trans_scale_t = trans_scale_t.clamp(max=var_scale_cap)
    rot_scale_t = rot_scale_t.clamp(max=var_scale_cap)

    # Optional alignment: center and rotate predictions into target frame
    if align:
        # Center of mass (detached - not part of gradient)
        com_t = batch_center_of_mass(trans_t, motif_mask).detach()  # (B, 3)
        com_1 = batch_center_of_mass(trans_1_motifs, motif_mask).detach()  # (B, 3)

        # Compute alignment rotation (detached)
        _, _, align_rots = batch_align_structures(
            trans_t.detach().clone(),
            trans_1_motifs.detach().clone(),
            mask=motif_mask,
            center=True,
        )
        align_rots = align_rots.detach().to(dtype=pred_trans_1.dtype)  # (B, 3, 3)

        # Center and align endpoint predictions
        pred_trans_for_loss = torch.einsum(
            "bni,bij->bnj", pred_trans_1 - com_t[:, None], align_rots
        )
        trans_target_for_loss = trans_1_motifs - com_1[:, None]

        # Rotate predicted endpoint frames by alignment rotation
        pred_rotmats_for_loss = so3_utils.rot_mult(
            pred_rotmats_1, align_rots[:, None, :, :]
        )
        rot_target_for_loss = rotmats_1_motifs
    else:
        pred_trans_for_loss = pred_trans_1
        trans_target_for_loss = trans_1_motifs
        pred_rotmats_for_loss = pred_rotmats_1
        rot_target_for_loss = rotmats_1_motifs

    # Translation log-likelihood proxy: log p ~ -0.5 * ||error||^2 / var(t)
    trans_sq = ((pred_trans_for_loss - trans_target_for_loss) ** 2).sum(
        dim=-1
    )  # (B, N)
    trans_ll_per_b = -0.5 * (trans_sq * motif_f).sum(dim=1) * trans_scale_t  # (B,)

    # Rotation log-likelihood: use squared norm of log-map vector as geodesic proxy
    rot_err = so3_utils.calc_rot_vf(
        mat_t=pred_rotmats_for_loss,
        mat_1=rot_target_for_loss,
    )  # (B, N, 3)
    rot_sq = (rot_err * rot_err).sum(dim=-1)  # (B, N)
    rot_ll_per_b = -0.5 * (rot_sq * motif_f).sum(dim=1) * rot_scale_t  # (B,)

    # Total log-likelihood to differentiate
    log_p = (trans_ll_per_b + rot_ll_per_b).sum()

    # Check gradient connectivity
    if not log_p.requires_grad:
        if allow_none:
            return PotentialField(), None
        raise ValueError(
            "log_p does not require grad. Ensure pred_trans_1/pred_rotmats_1 "
            "are differentiably connected to trans_t/rotmats_t through the model."
        )

    # Compute gradients w.r.t. current state
    grad_trans, grad_rot_mat = torch.autograd.grad(
        log_p,
        [trans_t, rotmats_t],
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    # Either both are None, or neither is None
    if grad_trans is None and grad_rot_mat is None:
        if allow_none:
            return PotentialField(), None
        raise ValueError("No gradients found for motif guidance")

    trans_potential = grad_trans

    # Convert rotation matrix gradient to tangent VF via Riemannian projection
    # A = skew(R^T dL/dR), then vee(A) -> (B, N, 3)
    RtG = so3_utils.rot_mult(so3_utils.rot_transpose(rotmats_t), grad_rot_mat)
    A = 0.5 * (RtG - so3_utils.rot_transpose(RtG))  # skew-symmetrize
    rot_potential_vf = so3_utils.skew_matrix_to_vector(A)  # (B, N, 3)

    # Scale by prefactor: 0.5 * g(t)^2 / omega(t)^2, plus window scaling
    # Clamp to prevent numerical issues
    pref = (0.5 * (g2 / omega2) * window).clamp(max=100.0).view(B, 1, 1)  # (B, 1, 1)
    trans_potential = pref * trans_potential
    rot_potential_vf = pref * rot_potential_vf

    # Mask to motifs only
    trans_potential = trans_potential * motif_f.unsqueeze(-1)
    rot_potential_vf = rot_potential_vf * motif_f.unsqueeze(-1)

    potential = PotentialField(
        trans=trans_potential.detach(),
        rotmats=rot_potential_vf.detach(),
    )

    # Compute debug metrics if enabled
    metrics = None
    if cfg.debug and valid_mask is not None:
        metrics = _compute_motif_guidance_metrics(
            step=step,
            t=t,
            trans_t=trans_t.detach() if trans_t.requires_grad else trans_t,
            trans_1_motifs=trans_1_motifs,
            motif_mask=motif_mask,
            valid_mask=valid_mask,
            potential=potential,
            prev_metrics=prev_metrics,
        )

    return potential, metrics


def _compute_motif_guidance_metrics(
    step: int,
    t: torch.Tensor,  # (B,)
    trans_t: torch.Tensor,  # (B, P, 3)
    trans_1_motifs: torch.Tensor,  # (B, P, 3)
    motif_mask: torch.Tensor,  # (B, P) bool
    valid_mask: torch.Tensor,  # (B, P) bool
    potential: PotentialField,
    prev_metrics: Optional[MotifGuidanceMetrics] = None,
) -> MotifGuidanceMetrics:
    """
    Compute motif guidance metrics for debugging and monitoring.

    Args:
        step: Current sampling step number
        t: Current time values (B,)
        trans_t: Current translations (B, P, 3)
        trans_1_motifs: Target motif translations (B, P, 3)
        motif_mask: Boolean mask for motif residues (B, P)
        valid_mask: Boolean mask for valid/present residues (B, P)
        potential: PotentialField with trans and rotmats guidance
        prev_metrics: Previous step's metrics (for extracting align_rot to compute delta_theta)

    Returns:
        MotifGuidanceMetrics with align_rot included for passing to next step
    """
    device = trans_t.device
    B, P = motif_mask.shape

    # Only consider motif residues that are currently present/valid
    mask = (motif_mask & valid_mask).bool()  # (B, P)
    motif_count = mask.sum(dim=1).clamp_min(1).float()  # (B,)

    X = trans_t
    Y = trans_1_motifs

    # Raw motif RMSD in world frame
    diff = (X - Y) * mask.unsqueeze(-1).float()
    rmsd_raw = torch.sqrt(
        (diff.square().sum(dim=-1).sum(dim=1) / motif_count).clamp_min(0.0)
    )

    # Center + align using existing utility
    com_x = batch_center_of_mass(X, mask=mask)
    com_y = batch_center_of_mass(Y, mask=mask)
    Xc = X - com_x[:, None, :]
    Yc = Y - com_y[:, None, :]

    _, _, Q = batch_align_structures(
        X.detach().clone(),
        Y.detach().clone(),
        mask=mask,
        center=True,
    )

    # Aligned motif RMSD
    X_aligned = torch.einsum("bpi,bij->bpj", Xc, Q)
    diff_a = (X_aligned - Yc) * mask.unsqueeze(-1).float()
    rmsd_aligned = torch.sqrt(
        (diff_a.square().sum(dim=-1).sum(dim=1) / motif_count).clamp_min(0.0)
    )

    # SVD singular values of weighted cross-covariance (alignment conditioning)
    w = mask.unsqueeze(-1).float()
    H = torch.einsum("bpi,bpj->bij", Xc * w, Yc)  # (B, 3, 3)
    _U, S, _Vh = torch.linalg.svd(H.float())  # no bf16
    cond = (S[..., 0] / S[..., 2].clamp_min(1e-8)).clamp_max(1e8)

    # Rotation change since last step
    prev_align_rot = prev_metrics.align_rot if prev_metrics is not None else None
    if prev_align_rot is None:
        dtheta = torch.zeros((Q.shape[0],), device=device, dtype=Q.dtype)
    else:
        dR = torch.einsum("bij,bjk->bik", prev_align_rot.transpose(-1, -2), Q)
        tr = dR[..., 0, 0] + dR[..., 1, 1] + dR[..., 2, 2]
        cos = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0)
        dtheta = torch.acos(cos)

    # Guidance norms
    if potential.trans is None:
        trans_g = torch.zeros_like(rmsd_raw)
    else:
        trans_g = (potential.trans.norm(dim=-1) * mask.float()).sum(dim=1) / motif_count

    if potential.rotmats is None:
        rot_g = torch.zeros_like(rmsd_raw)
    else:
        rot_g = (potential.rotmats.norm(dim=-1) * mask.float()).sum(dim=1) / motif_count

    metrics = MotifGuidanceMetrics(
        step=step,
        t=float(t.mean().item()),
        num_positions=P,
        num_motif=int(mask.sum().item()),
        rmsd_raw=float(rmsd_raw.mean().item()),
        rmsd_aligned=float(rmsd_aligned.mean().item()),
        sv_min=float(S[..., 2].mean().item()),
        condition_number=float(cond.mean().item()),
        delta_theta=float(dtheta.mean().item()),
        trans_guidance_norm=float(trans_g.mean().item()),
        rot_guidance_norm=float(rot_g.mean().item()),
        align_rot=Q.detach(),
    )

    return metrics
