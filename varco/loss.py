import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.nn import functional as F

from cogeneration.data import so3_utils
from cogeneration.data.const import ANG_TO_NM_SCALE, MASK_TOKEN_INDEX
from varco.config import VarcoLossConfig
from varco.data import DataBridged, DataCorrupted, ModelPrediction
from varco.interpolant import TreeCouplings, TreeInterpolant

# TODO - ideally we could share loss calculator with cogeneration, e.g. using static methods


@dataclass
class BranchFlowLosses:
    total_loss: torch.Tensor
    trans_loss: torch.Tensor  # MSE on translations
    pairwise_loss: torch.Tensor  # local pairwise distance loss
    rot_vf_loss: torch.Tensor  # MSE on rotation vector field
    base_seq_loss: torch.Tensor  # base sequence loss (token + anchor-prob)
    base_seq_prob_loss: torch.Tensor  # soft CE on amino acid logits vs anchor_probs
    base_seq_token_loss: (
        torch.Tensor
    )  # CE on amino acid logits vs sampled anchor tokens
    insertion_seq_loss: torch.Tensor  # soft CE on insertion logits vs anchor_probs
    split_token_loss: torch.Tensor  # Poisson loss on per-token remaining splits
    split_pooled_loss: torch.Tensor  # aux Poisson loss on total remaining splits
    del_loss: torch.Tensor  # BCE on per-token logits (terminal tokens only)
    bfactor_loss: torch.Tensor  # CE on binned b-factor predictions
    plddt_loss: torch.Tensor  # CE on binned pLDDT predictions


@dataclass
class BranchFlowMetrics:
    base_seq_ce: torch.Tensor  # token CE (nats), motif-weighted if configured
    base_seq_acc: torch.Tensor  # top-1 accuracy on anchor tokens, motif-weighted
    base_seq_ce_scaffold: torch.Tensor  # CE on scaffold tokens only (nats)
    base_seq_acc_scaffold: torch.Tensor  # top-1 accuracy on scaffold tokens only
    insertion_seq_ce: torch.Tensor  # unweighted soft CE on insertion logits (nats)
    insertion_target_entropy: (
        torch.Tensor
    )  # unweighted entropy of insertion targets (nats)
    insertion_ce_over_entropy: torch.Tensor  # mean over positions of CE/H(target)
    insertion_ce_minus_entropy: (
        torch.Tensor
    )  # mean over positions of CE - H(target) (nats)
    insertion_seq_kl: torch.Tensor  # alias for CE - H(target) (nats)
    trans_rmse_ang: torch.Tensor  # RMS translation error (angstroms)
    trans_mae_ang: torch.Tensor  # MAE translation error (angstroms)
    rot_mae_deg: torch.Tensor  # mean abs geodesic angle error (degrees)
    rot_rmse_deg: torch.Tensor  # RMS geodesic angle error (degrees)
    split_event_ce: torch.Tensor  # Bernoulli CE on split event (>0)
    split_event_precision: torch.Tensor
    split_event_recall: torch.Tensor
    split_event_f1: torch.Tensor
    split_event_auprc: torch.Tensor  # average precision (PR-AUC) for split event
    split_event_pos_rate: torch.Tensor  # fraction of positives for split event
    split_rate_mae: torch.Tensor  # MAE on pred split vs target count
    split_rate_mae_pos: torch.Tensor  # MAE conditioned on target>0
    split_rate_corr: torch.Tensor  # Pearson correlation of rate vs target count
    del_event_ce: (
        torch.Tensor
    )  # Bernoulli CE on delete event (terminal scaffold tokens)
    del_event_precision: torch.Tensor
    del_event_recall: torch.Tensor
    del_event_f1: torch.Tensor
    del_event_auprc: torch.Tensor  # average precision (PR-AUC) for delete event
    del_event_pos_rate: torch.Tensor  # fraction of positives for delete event
    del_prob_mean: torch.Tensor  # mean predicted P(delete) on supervised tokens
    del_true_rate: torch.Tensor  # empirical delete rate on supervised tokens
    del_brier: torch.Tensor  # mean (p - y)^2 on supervised tokens
    lddt_mean: torch.Tensor  # mean lDDT (0-1) computed from coords vs anchors
    plddt_ce: torch.Tensor  # unweighted, unclamped CE (nats)
    plddt_bin_acc: torch.Tensor  # top-1 accuracy on bins
    plddt_bin_acc_pm1: torch.Tensor  # accuracy within ±1 bin
    plddt_bin_mae: torch.Tensor  # mean abs bin error


@dataclass
class BranchFlowLossCalculator:
    cfg: VarcoLossConfig

    @staticmethod
    def _average_precision(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Average precision / PR-AUC for binary labels, computed over a flat vector."""
        if scores.ndim != 1 or labels.ndim != 1:
            raise ValueError("scores and labels must be 1D")
        if scores.numel() != labels.numel():
            raise ValueError("scores and labels must have the same length")
        if scores.numel() == 0:
            return torch.tensor(0.0, device=scores.device)

        labels_f = labels.to(dtype=torch.float32)
        pos = labels_f.sum()
        if float(pos.item()) <= 0.0:
            return torch.tensor(0.0, device=scores.device)

        order = torch.argsort(scores, descending=True)
        y = labels_f[order]
        cum_tp = torch.cumsum(y, dim=0)
        ranks = torch.arange(1, y.numel() + 1, device=scores.device, dtype=y.dtype)
        precision = cum_tp / ranks
        return (precision * y).sum() / pos

    @staticmethod
    def _pearson_corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Pearson correlation over flat vectors; returns 0 if undefined."""
        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be 1D")
        if x.numel() != y.numel():
            raise ValueError("x and y must have the same length")
        if x.numel() < 2:
            return torch.tensor(0.0, device=x.device)

        x = x.to(dtype=torch.float32)
        y = y.to(dtype=torch.float32)
        x = x - x.mean()
        y = y - y.mean()
        denom = (x.square().sum().sqrt() * y.square().sum().sqrt()).clamp_min(1e-8)
        return (x * y).sum() / denom

    def _time_norm_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-based normalization scale: 1 - min(t, clip).
        Higher weight (smaller divisor) as t -> 1.
        """
        t_clip = t.clamp(max=self.cfg.t_normalize_clip)
        t_norm = 1 - t_clip
        t_norm = t_norm**self.cfg.t_normalize_exponent
        return t_norm

    @staticmethod
    def log_clamp(x: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
        """
        Soft clamp using log compression above threshold.
        Preserves gradients above threshold but at diminishing scale: threshold + log(1 + excess)
        """
        return torch.where(x > threshold, threshold + torch.log1p(x - threshold), x)

    @staticmethod
    def _apply_motif_weight(
        mask_f: torch.Tensor,
        motif_mask: Optional[torch.Tensor],
        motif_weight: float,
    ) -> torch.Tensor:
        if motif_mask is None:
            return mask_f
        weights = torch.where(
            motif_mask,
            torch.full_like(mask_f, float(motif_weight)),
            torch.ones_like(mask_f),
        )
        return mask_f * weights

    def _soft_ce_from_probs(
        self,
        pred_logits: torch.Tensor,  # (B, P, K)
        target_probs: torch.Tensor,  # (B, P, K)
        mask: torch.Tensor,  # (B, P)
        motif_mask: Optional[torch.Tensor] = None,  # (B, P)
        motif_weight: float = 0.1,
        t: Optional[torch.Tensor] = None,  # (B,)
        apply_time_norm: bool = False,
        time_norm_divisor: float = 2.0,
        per_example: bool = True,
        mostly_mask_threshold: float = 0.75,
        require_mass: bool = True,
    ) -> torch.Tensor:
        """Soft cross-entropy on logits vs per-token target probabilities."""
        B, P, K = pred_logits.shape

        # Zero out mask token and renormalize target probs
        target_probs_masked = target_probs.clone()
        target_probs_masked[:, :, MASK_TOKEN_INDEX] = 0.0
        row_sums = target_probs_masked.sum(dim=-1, keepdim=True)
        has_mass = row_sums.squeeze(-1) > 1e-8  # (B, P)
        target_probs_masked = target_probs_masked / row_sums.clamp_min(1e-8)

        log_probs = F.log_softmax(pred_logits, dim=-1)  # (B, P, K)
        ce_per_token = -(target_probs_masked * log_probs).sum(dim=-1)  # (B, P)

        if apply_time_norm:
            if t is None:
                raise ValueError("t is required when apply_time_norm=True")
            t_norm = self._time_norm_scale(t=t).view(B, 1)  # (B, 1)
            ce_per_token = ce_per_token / (float(time_norm_divisor) * t_norm)  # (B, P)

        is_mostly_mask = target_probs[:, :, MASK_TOKEN_INDEX] >= float(
            mostly_mask_threshold
        )  # (B, P)
        mask_f = mask.float() * (~is_mostly_mask).float()
        if require_mass:
            mask_f = mask_f * has_mass.float()
        mask_f = self._apply_motif_weight(mask_f, motif_mask, motif_weight)

        if per_example:
            denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B,)
            loss_per_batch = (ce_per_token * mask_f).sum(dim=1) / denom  # (B,)
            return loss_per_batch.mean()

        denom = mask_f.sum().clamp_min(1.0)
        return (ce_per_token * mask_f).sum() / denom

    def _base_trans_loss(
        self,
        pred_trans: torch.Tensor,  # (B, P, 3) in angstroms
        target_trans: torch.Tensor,  # (B, P, 3) in angstroms
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """Translation loss for anchor/final positions."""
        B, P, D = pred_trans.shape

        # Time-based normalization (higher weight as t -> 1)
        t_norm = self._time_norm_scale(t=t).view(B, 1, 1)

        # Scale both to nm for loss computation
        pred_scaled = pred_trans * ANG_TO_NM_SCALE / t_norm
        target_scaled = target_trans * ANG_TO_NM_SCALE / t_norm
        mse = (pred_scaled - target_scaled).square()  # (B, P, 3)

        # Per-example masked mean (normalize by number of present positions * 3 for xyz)
        mask_f = mask.unsqueeze(-1).float()  # (B, P, 1)
        mse = mse * mask_f
        denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0) * 3  # (B,) * 3 for xyz coords
        loss_per_batch = mse.sum(dim=(1, 2)) / denom  # (B,)

        return (
            self.log_clamp(loss_per_batch.mean(), threshold=5.0)
            * self.cfg.trans_loss_weight
        )

    def _pairwise_distance_loss(
        self,
        pred_trans: torch.Tensor,  # (B, P, 3)
        target_trans: torch.Tensor,  # (B, P, 3)
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """Local pairwise distance loss for positions within proximity threshold."""
        B, P, D = pred_trans.shape

        # Time-based normalization (higher weight as t -> 1)
        t_norm = self._time_norm_scale(t=t).view(B, 1, 1)

        # Compute pairwise distances in angstroms (B, P, P)
        with torch.no_grad():
            target_dists = torch.cdist(target_trans, target_trans)
        pred_dists = torch.cdist(pred_trans, pred_trans)

        # Mask for valid pairs
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, P, P)

        # Limit to local neighborhood in ground truth
        proximity_mask = target_dists < self.cfg.proximity_threshold_ang
        pair_mask = pair_mask & proximity_mask

        # Exclude self-pairs
        eye = torch.eye(P, device=pred_trans.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        # Squared distance error with time normalization (scale to nm for loss)
        dist_error = ((pred_dists - target_dists) * ANG_TO_NM_SCALE / t_norm) ** 2
        dist_error = dist_error * pair_mask.float()

        # Normalize by number of valid pairs per batch
        denom = pair_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        loss_per_batch = dist_error.sum(dim=(1, 2)) / denom

        return (
            self.log_clamp(loss_per_batch.mean(), threshold=5.0)
            * self.cfg.pairwise_dist_loss_weight
        )

    @staticmethod
    def _trans_error_metrics_ang(
        pred_trans: torch.Tensor,  # (B, P, 3) angstroms
        target_trans: torch.Tensor,  # (B, P, 3) angstroms
        mask: torch.Tensor,  # (B, P)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Translation error metrics in angstroms: (RMSE, MAE) over per-token norms."""
        with torch.no_grad():
            diff = pred_trans - target_trans  # (B, P, 3)
            err = diff.norm(dim=-1)  # (B, P)
            mask_f = mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            mae = (err * mask_f).sum() / denom
            rmse = torch.sqrt(((err.square()) * mask_f).sum() / denom)
            return rmse, mae

    def _rot_vf_loss(
        self,
        pred_rotmats: torch.Tensor,  # (B, P, 3, 3)
        target_rotmats: torch.Tensor,  # (B, P, 3, 3) anchors at t=1
        rotmats_t: torch.Tensor,  # (B, P, 3, 3) current rotations
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """
        Rotation vector field loss.
        Computes MSE between predicted and target rotation vector fields.
        The VF is Log_{rotmats_t}(target) - the tangent vector pointing to target.
        """
        B, P = mask.shape

        # Time-based normalization (higher weight as t -> 1)
        t_norm = self._time_norm_scale(t=t).view(B, 1, 1)

        # Compute rotation vector fields: Log_{rotmats_t}(rotmats_1)
        # pred_rot_vf: direction from rotmats_t to pred_rotmats
        # target_rot_vf: direction from rotmats_t to target_rotmats (anchor)
        pred_rot_vf = so3_utils.calc_rot_vf(
            mat_t=rotmats_t.float(), mat_1=pred_rotmats.float()
        )  # (B, P, 3)
        target_rot_vf = so3_utils.calc_rot_vf(
            mat_t=rotmats_t.float(), mat_1=target_rotmats.float()
        )  # (B, P, 3)

        # MSE on vector field with time normalization
        rot_vf_error = (pred_rot_vf - target_rot_vf) / t_norm
        mse = rot_vf_error.square()  # (B, P, 3)

        # Per-example masked mean (normalize by number of present positions * 3 for xyz)
        mask_f = mask.unsqueeze(-1).float()  # (B, P, 1)
        mse = mse * mask_f
        denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0) * 3  # (B,) * 3 for xyz coords
        loss_per_batch = mse.sum(dim=(1, 2)) / denom  # (B,)

        return (
            self.log_clamp(loss_per_batch.mean(), threshold=5.0)
            * self.cfg.rot_vf_loss_weight
        )

    @staticmethod
    def _rot_geodesic_error_metrics_deg(
        pred_rotmats: torch.Tensor,  # (B, P, 3, 3)
        target_rotmats: torch.Tensor,  # (B, P, 3, 3)
        mask: torch.Tensor,  # (B, P)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Rotation error metrics: (MAE, RMSE) of geodesic angle error in degrees."""
        with torch.no_grad():
            rel = so3_utils.rot_mult(
                so3_utils.rot_transpose(pred_rotmats), target_rotmats
            )  # (B, P, 3, 3)
            angles_rad, _, _ = so3_utils.angle_from_rotmat(rel)  # (B, P)
            mask_f = mask.float()
            denom = mask_f.sum().clamp_min(1.0)
            mae_rad = (angles_rad.abs() * mask_f).sum() / denom
            rmse_rad = torch.sqrt((angles_rad.square() * mask_f).sum() / denom)
            rad2deg = 180.0 / math.pi
            return mae_rad * rad2deg, rmse_rad * rad2deg

    def _seq_token_loss(
        self,
        pred_aatype_logits: torch.Tensor,  # (B, P, K)
        target_anchor_tokens: torch.Tensor,  # (B, P) long
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
        motif_mask: Optional[torch.Tensor] = None,  # (B, P)
        motif_weight: float = 0.1,
    ) -> torch.Tensor:
        """Sequence loss: cross-entropy on amino acid logits vs sampled anchor tokens."""
        B, P, K = pred_aatype_logits.shape

        # Time-based normalization for likelihood weighting (higher weight as t -> 1)
        t_norm = self._time_norm_scale(t=t).view(B, 1)  # (B, 1)

        # Cross-entropy per token (reduction=none)
        ce = F.cross_entropy(
            pred_aatype_logits.view(-1, K),
            target_anchor_tokens.view(-1),
            reduction="none",
        ).view(
            B, P
        )  # (B, P)

        # Apply softened time normalization (likelihood weighting, half strength)
        ce = ce / (2.0 * t_norm)  # (B, P)

        # Mask out unknown residues
        mask_f = mask.float()  # (B, P)
        mask_f = mask_f * (target_anchor_tokens != MASK_TOKEN_INDEX).float()
        mask_f = self._apply_motif_weight(mask_f, motif_mask, motif_weight)
        # Masked mean per batch, then average over batch
        denom = mask_f.sum(dim=1).clamp_min(1.0)  # (B,)
        loss_per_batch = (ce * mask_f).sum(dim=1) / denom  # (B,)
        seq_loss = loss_per_batch.mean()

        return (
            self.log_clamp(seq_loss, threshold=5.0)
            * self.cfg.seq_loss_weight
            * self.cfg.seq_token_loss_weight
        )

    def _seq_prob_loss(
        self,
        pred_aatype_logits: torch.Tensor,  # (B, P, K)
        target_anchor_probs: torch.Tensor,  # (B, P, K)
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
        motif_mask: Optional[torch.Tensor] = None,  # (B, P)
        motif_weight: float = 0.1,
    ) -> torch.Tensor:
        """Sequence loss: soft cross-entropy on amino acid logits vs anchor probability targets."""
        seq_loss = self._soft_ce_from_probs(
            pred_logits=pred_aatype_logits,
            target_probs=target_anchor_probs,
            mask=mask,
            motif_mask=motif_mask,
            motif_weight=motif_weight,
            t=t,
            apply_time_norm=True,
            time_norm_divisor=2.0,
            per_example=True,
            mostly_mask_threshold=0.75,
            require_mass=True,
        )

        return (
            self.log_clamp(seq_loss, threshold=5.0)
            * self.cfg.seq_loss_weight
            * self.cfg.seq_prob_loss_weight
        )

    def _seq_ce_metric(
        self,
        pred_aatype_logits: torch.Tensor,  # (B, P, K)
        target_anchor_tokens: torch.Tensor,  # (B, P) long
        mask: torch.Tensor,  # (B, P)
        motif_mask: Optional[torch.Tensor] = None,  # (B, P)
        motif_weight: float = 0.1,
    ) -> torch.Tensor:
        """Aux metric: per-token CE on amino acids (nats), motif-weighted if provided."""
        with torch.no_grad():
            B, P, K = pred_aatype_logits.shape
            ce = F.cross_entropy(
                pred_aatype_logits.view(-1, K),
                target_anchor_tokens.view(-1),
                reduction="none",
            ).view(B, P)
            valid = mask & (target_anchor_tokens != MASK_TOKEN_INDEX)
            valid_f = self._apply_motif_weight(valid.float(), motif_mask, motif_weight)
            return (ce * valid_f).sum() / valid_f.sum().clamp_min(1.0)

    @staticmethod
    def _seq_acc_metric(
        pred_aatype_logits: torch.Tensor,  # (B, P, K)
        target_anchor_tokens: torch.Tensor,  # (B, P) long
        mask: torch.Tensor,  # (B, P)
        motif_mask: Optional[torch.Tensor] = None,  # (B, P)
        motif_weight: float = 0.1,
    ) -> torch.Tensor:
        """Aux metric: per-token top-1 accuracy on amino acids, motif-weighted."""
        with torch.no_grad():
            pred_tokens = pred_aatype_logits.argmax(dim=-1)  # (B, P)
            valid = mask & (target_anchor_tokens != MASK_TOKEN_INDEX)
            correct = (pred_tokens == target_anchor_tokens) & valid
            weight = BranchFlowLossCalculator._apply_motif_weight(
                valid.float(), motif_mask, motif_weight
            )
            denom = weight.sum().clamp_min(1.0)
            return (correct.float() * weight).sum() / denom

    def _insertion_seq_loss(
        self,
        pred_insertion_logits: torch.Tensor,  # (B, P, K)
        target_anchor_probs: torch.Tensor,  # (B, P, K)
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """
        Insertion sequence loss: soft cross-entropy on insertion logits vs anchor logits.
        For internal nodes, this is the distribution of children's amino acid types.
        For leaves, this is effectively cross-entropy on the final aatype.
        """
        if not mask.any():
            # Keep a graph connection to insertion-head params so DDP doesn't treat them
            # as unused on batches with no supervised insertion positions.
            return pred_insertion_logits.float().sum() * 0.0

        # Zero out mask token and renormalize target probs
        target_probs_masked = target_anchor_probs.clone()
        target_probs_masked[:, :, MASK_TOKEN_INDEX] = 0.0
        target_probs_masked = target_probs_masked / target_probs_masked.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)

        # Soft cross-entropy: -sum(target_probs * log_softmax(logits))
        log_probs = F.log_softmax(pred_insertion_logits, dim=-1)  # (B, P, K)
        ce_per_token = -(target_probs_masked * log_probs).sum(dim=-1)  # (B, P)

        # Mask out positions where target was mostly the mask token
        is_mostly_mask = target_anchor_probs[:, :, MASK_TOKEN_INDEX] >= 0.75
        mask_f = mask.float() * (~is_mostly_mask).float()
        denom = mask_f.sum().clamp_min(1.0)
        insertion_loss = (ce_per_token * mask_f).sum() / denom

        return (
            self.log_clamp(insertion_loss, threshold=5.0) * self.cfg.seq_ins_loss_weight
        )

    @staticmethod
    def _insertion_ce_entropy_metrics(
        pred_logits: torch.Tensor,  # (B, P, K)
        target_probs: torch.Tensor,  # (B, P, K)
        mask: torch.Tensor,  # (B, P)
        mostly_mask_threshold: float = 0.75,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute insertion CE vs target entropy metrics on the same mask convention as insertion_seq_ce:
        - mask out positions where target is mostly MASK_TOKEN_INDEX
        - remove MASK token mass and renormalize

        Returns:
            target_entropy_mean: mean H(target) over valid positions (nats)
            ce_over_entropy_mean: mean CE(pred,target)/H(target) over valid positions
            ce_minus_entropy_mean: mean CE(pred,target)-H(target) over valid positions (nats)
        """
        B, P, K = pred_logits.shape
        if target_probs.shape != (B, P, K):
            raise ValueError(
                f"target_probs must have shape (B, P, K); got {tuple(target_probs.shape)}"
            )
        if mask.shape != (B, P):
            raise ValueError(f"mask must have shape (B, P); got {tuple(mask.shape)}")

        # Drop mask token from the target distribution and renormalize.
        target_probs_masked = target_probs.clone()
        target_probs_masked[:, :, MASK_TOKEN_INDEX] = 0.0
        row_sums = target_probs_masked.sum(dim=-1, keepdim=True)
        has_mass = row_sums.squeeze(-1) > float(eps)  # (B, P)
        target_probs_masked = target_probs_masked / row_sums.clamp_min(float(eps))

        # Match insertion_seq_ce behavior: drop positions that were mostly mask-token supervision.
        is_mostly_mask = target_probs[:, :, MASK_TOKEN_INDEX] >= float(
            mostly_mask_threshold
        )
        valid = mask & (~is_mostly_mask) & has_mass  # (B, P)
        valid_f = valid.float()
        denom = valid_f.sum().clamp_min(1.0)

        log_q = F.log_softmax(pred_logits, dim=-1)  # (B, P, K)
        ce_per_token = -(target_probs_masked * log_q).sum(dim=-1)  # (B, P)

        # Entropy H(p) = -sum p log p
        log_p = torch.log(target_probs_masked.clamp_min(float(eps)))
        entropy_per_token = -(target_probs_masked * log_p).sum(dim=-1)  # (B, P)

        # CE - H = KL(p || q) >= 0 (when p has mass)
        ce_minus_entropy = ce_per_token - entropy_per_token

        # Ratio is only meaningful when entropy > 0; clamp to avoid infs for near-delta targets.
        ratio = ce_per_token / entropy_per_token.clamp_min(float(eps))

        target_entropy_mean = (entropy_per_token * valid_f).sum() / denom
        ce_minus_entropy_mean = (ce_minus_entropy * valid_f).sum() / denom
        ce_over_entropy_mean = (ratio * valid_f).sum() / denom

        return target_entropy_mean, ce_over_entropy_mean, ce_minus_entropy_mean

    def _split_token_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        motif_weight: float = 0.1,
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Per-token Poisson Bregman divergence on insertion mass M:
        D(M_target || M_pred) = M_pred - M_target + M_target * log(M_target/M_pred)

        The model predicts time-independent mass M; the target is:
            M_target = remaining_insertions / S(t)
        where S(t) = 1 - H(t) is the hazard survival function.

        Primary loss is on scaffolds (mask & ~motif_mask), with a smaller penalty
        (motif_weight) applied to motif positions.
        """
        if batch.remaining_insertions is None:
            raise ValueError("batch.remaining_insertions is required for split loss")

        # Compute survival function S(t) for each batch element
        # batch.t is (B,), we use mean t for the batch (they should be same in corrupt_batch)
        t_mean = float(batch.t.mean().item())
        S_t = TreeInterpolant.compute_hazard_survival(t_mean, self.cfg.split_hazard)
        S_t = max(eps, S_t)  # avoid division by zero near t=1

        # Target mass: M = R_t / S(t)
        remaining = batch.remaining_insertions.to(torch.float32)  # (B, P)
        target_mass = remaining / S_t  # (B, P)

        # Predicted mass
        mass = pred.pred_split_mass.clamp_min(eps)  # (B, P)

        # Poisson Bregman divergence on mass
        target_safe = target_mass.clamp_min(eps)
        token_loss = torch.where(
            target_mass > 0,
            mass - target_mass + target_mass * torch.log(target_safe / mass),
            mass,
        )

        # Scaffold loss (primary) and motif loss (small penalty)
        scaffold_mask = mask & ~motif_mask
        motif_loss_mask = mask & motif_mask

        scaffold_weight = scaffold_mask.float()
        scaffold_denom = scaffold_weight.sum(dim=1).clamp_min(1.0)  # (B,)
        scaffold_loss = (
            (token_loss * scaffold_weight).sum(dim=1) / scaffold_denom
        ).mean()

        motif_weight_tensor = motif_loss_mask.float()
        motif_denom = motif_weight_tensor.sum(dim=1).clamp_min(1.0)  # (B,)
        motif_loss = (
            (token_loss * motif_weight_tensor).sum(dim=1) / motif_denom
        ).mean()

        split_loss = scaffold_loss + motif_weight * motif_loss

        # Regularize diffuse per-token split mass to encourage sparsity.
        entropy_weight = self.cfg.split_mass_entropy_weight
        l2_weight = self.cfg.split_mass_l2_weight
        if (entropy_weight > 0.0) or (l2_weight > 0.0):
            mass_scaffold = mass * scaffold_mask.float()  # (B, P)
            sum_mass = mass_scaffold.sum(dim=1, keepdim=True).clamp_min(eps)  # (B, 1)
            w = mass_scaffold / sum_mass  # (B, P), sums to 1 over scaffold positions

            # Normalized entropy in [~0, 1] (0=one-hot; 1=uniform over N)
            n = scaffold_mask.float().sum(dim=1).clamp_min(2.0)  # (B,)
            entropy = -(w * torch.log(w.clamp_min(eps))).sum(dim=1)  # (B,)
            entropy_norm = entropy / torch.log(n)  # (B,)

            # Encourage concentration via L2 (max=1 for one-hot; min~1/N for uniform)
            l2 = w.square().sum(dim=1)  # (B,)
            l2_penalty = (1.0 - l2).clamp_min(0.0)  # (B,)

            regularize_loss = (
                entropy_weight * entropy_norm.mean() + l2_weight * l2_penalty.mean()
            )
            split_loss = split_loss + regularize_loss

        return self.log_clamp(split_loss, threshold=8.0) * self.cfg.split_loss_weight

    def _split_metrics(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Aux metrics for split prediction (scaffold only).

        Metrics are computed in rate space (R_t = M * S(t)) for interpretability,
        since remaining_insertions is the natural target at time t.
        """
        device = batch.trans_t.device
        eps = 1e-8
        if batch.remaining_insertions is None:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero, zero, zero, zero, zero, zero

        with torch.no_grad():
            # Compute S(t) to convert mass -> rate for metrics
            t_mean = float(batch.t.mean().item())
            S_t = TreeInterpolant.compute_hazard_survival(t_mean, self.cfg.split_hazard)
            S_t = max(eps, S_t)

            target = batch.remaining_insertions.to(torch.float32)  # (B, P)
            # Convert predicted mass to rate for comparison
            rate = pred.pred_split_mass.clamp_min(0.0) * S_t  # (B, P)

            split_mask = mask & ~motif_mask
            split_mask_f = split_mask.float()
            denom = split_mask_f.sum().clamp_min(1.0)

            # Event metrics: event is whether any insertions remain at this token.
            y = (target > 0).float()
            p = (1.0 - torch.exp(-rate)).clamp(min=1e-6, max=1.0 - 1e-6)
            bce = -(y * torch.log(p) + (1.0 - y) * torch.log1p(-p))
            split_event_ce = (bce * split_mask_f).sum() / denom

            pred_pos = p > 0.5
            true_pos = target > 0
            tp = (pred_pos & true_pos & split_mask).sum().to(torch.float32)
            fp = (pred_pos & ~true_pos & split_mask).sum().to(torch.float32)
            fn = ((~pred_pos) & true_pos & split_mask).sum().to(torch.float32)
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

            mae = ((rate - target).abs() * split_mask_f).sum() / denom
            pos_mask = split_mask & (target > 0)
            pos_mask_f = pos_mask.float()
            mae_pos = (
                (rate - target).abs() * pos_mask_f
            ).sum() / pos_mask_f.sum().clamp_min(1.0)

            split_rate_corr = self._pearson_corr(rate[split_mask], target[split_mask])
            split_event_labels = (target > 0)[split_mask].flatten()
            split_event_scores = p[split_mask].flatten()
            if split_event_labels.numel() == 0:
                split_event_pos_rate = torch.tensor(0.0, device=device)
                split_event_auprc = torch.tensor(0.0, device=device)
            else:
                split_event_pos_rate = split_event_labels.to(torch.float32).mean()
                split_event_auprc = self._average_precision(
                    split_event_scores, split_event_labels
                )

        return (
            split_event_ce,
            precision,
            recall,
            f1,
            mae,
            mae_pos,
            split_event_auprc,
            split_event_pos_rate,
            split_rate_corr,
        )

    def _split_pooled_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
    ) -> torch.Tensor:
        """Pooled MSE loss on total insertion mass, in log1p space.

        Target mass = remaining_total / S(t), model predicts log1p(total_mass).
        """
        eps = 1e-8
        t_mean = float(batch.t.mean().item())
        S_t = TreeInterpolant.compute_hazard_survival(t_mean, self.cfg.split_hazard)
        S_t = max(eps, S_t)

        # Target mass in log1p space
        target_mass = batch.remaining_total.to(torch.float32) / S_t  # (B,)
        target_log = torch.log1p(target_mass)  # (B,)

        pred_log1p = pred.pred_split_pooled_log1p_mass  # (B,)
        pooled_loss = F.mse_loss(pred_log1p, target_log)
        return (
            self.log_clamp(pooled_loss, threshold=10.0)
            * self.cfg.split_pooled_loss_weight
        )

    def _deletion_metrics(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Aux metrics for deletion prediction (terminal scaffold tokens only)."""
        device = batch.trans_t.device
        if batch.deleted is None or batch.remaining_insertions is None:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero, zero, zero, zero, zero, zero

        with torch.no_grad():
            terminal_mask = mask & (batch.remaining_insertions == 0) & ~motif_mask
            terminal_f = terminal_mask.float()
            denom = terminal_f.sum().clamp_min(1.0)

            logits = pred.pred_del_logits
            targets = batch.deleted.float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
            del_event_ce = (bce * terminal_f).sum() / denom

            probs = torch.sigmoid(logits)
            pred_pos = probs > 0.5
            true_pos = targets > 0.5
            tp = (pred_pos & true_pos & terminal_mask).sum().to(torch.float32)
            fp = (pred_pos & ~true_pos & terminal_mask).sum().to(torch.float32)
            fn = ((~pred_pos) & true_pos & terminal_mask).sum().to(torch.float32)
            precision = tp / (tp + fp).clamp_min(1.0)
            recall = tp / (tp + fn).clamp_min(1.0)
            f1 = 2.0 * precision * recall / (precision + recall).clamp_min(1e-8)

            del_scores = probs[terminal_mask].flatten()
            del_labels = (targets > 0.5)[terminal_mask].flatten()
            if del_labels.numel() == 0:
                del_event_pos_rate = torch.tensor(0.0, device=device)
                del_event_auprc = torch.tensor(0.0, device=device)
                del_prob_mean = torch.tensor(0.0, device=device)
                del_true_rate = torch.tensor(0.0, device=device)
                del_brier = torch.tensor(0.0, device=device)
            else:
                del_event_pos_rate = del_labels.to(torch.float32).mean()
                del_event_auprc = self._average_precision(del_scores, del_labels)
                del_prob_mean = del_scores.mean()
                del_true_rate = del_event_pos_rate
                del_brier = (
                    (del_scores - del_labels.to(del_scores.dtype)).square().mean()
                )

        return (
            del_event_ce,
            precision,
            recall,
            f1,
            del_event_auprc,
            del_event_pos_rate,
            del_prob_mean,
            del_true_rate,
            del_brier,
        )

    def _deletion_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,
        motif_mask: torch.Tensor,  # (B, P)
        pos_weight: float = 1.00,  # upweight deleted targets, note may bias toward deletion
    ) -> torch.Tensor:
        """Deletion loss, supervised only on terminal tokens.

        Primary loss is on scaffolds (mask & ~motif_mask), with a smaller penalty
        (motif_weight) applied to motif positions.
        """
        if batch.deleted is None:
            # Keep a graph connection to deletion-head params so DDP doesn't treat them
            # as unused on batches without deletion supervision.
            return pred.pred_del_logits.float().sum() * 0.0

        terminal_mask = mask & (batch.remaining_insertions == 0)
        if not bool(terminal_mask.any()):
            # Keep a graph connection to deletion-head params so DDP doesn't treat them
            # as unused on batches without any terminal tokens.
            return pred.pred_del_logits.float().sum() * 0.0

        del_logits = pred.pred_del_logits  # (B, P)
        del_targets = batch.deleted.float()  # (B, P)
        bce = F.binary_cross_entropy_with_logits(
            del_logits,
            del_targets,
            reduction="none",
        )

        # upweight deleted targets
        token_weight = torch.ones_like(del_targets)
        token_weight = torch.where(
            del_targets > 0.5,
            torch.full_like(token_weight, float(pos_weight)),
            token_weight,
        )

        # Scaffold loss (primary) and motif loss (small penalty)
        motif_weight = 0.1
        scaffold_mask = terminal_mask & ~motif_mask
        motif_loss_mask = terminal_mask & motif_mask

        scaffold_weight = token_weight * scaffold_mask.float()
        scaffold_denom = scaffold_weight.sum().clamp_min(1.0)
        scaffold_loss = (bce * scaffold_weight).sum() / scaffold_denom

        motif_weight_tensor = token_weight * motif_loss_mask.float()
        motif_denom = motif_weight_tensor.sum().clamp_min(1.0)
        motif_loss = (bce * motif_weight_tensor).sum() / motif_denom

        del_loss = scaffold_loss + motif_weight * motif_loss

        return self.log_clamp(del_loss, threshold=8.0) * self.cfg.del_loss_weight

    def _bfactor_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """
        Cross-entropy loss on b-factor histograms, adapted from cogeneration.
        B-factor prediction is optional, and invalid when all b-factors are zero
        (e.g. as in synthetic examples or predicted structures).
        """
        pred_logits = pred.pred_bfactor  # (B, P, num_bins) or None
        if pred_logits is None:
            return torch.tensor(0.0, device=batch.trans_t.device)
        if batch.res_bfactor is None:
            # Keep a graph connection to confidence-head params so DDP doesn't treat them
            # as unused on batches without bfactor supervision.
            return pred_logits.float().sum() * 0.0

        bins = pred_logits.shape[-1]
        gt_bfactor = batch.res_bfactor  # (B, P)
        boundaries = torch.linspace(0.0, 100.0, bins - 1, device=gt_bfactor.device)

        # Discretise ground-truth b-factors
        bin_idx = (gt_bfactor.unsqueeze(-1) > boundaries).sum(-1).long()  # (B, P)
        target_logits = F.one_hot(bin_idx, num_classes=bins).float()

        # Mask out synthetic examples (all-zero b-factors) + padding + motifs
        valid_mask = (gt_bfactor > 1e-5) & mask & ~batch.motif_mask  # (B, P)
        if not valid_mask.any():
            # Keep a graph connection to confidence-head params so DDP doesn't treat them
            # as unused on batches with no valid bfactor targets.
            return pred_logits.float().sum() * 0.0

        # Cross-entropy
        logp = F.log_softmax(pred_logits.float(), dim=-1)
        ce = -(target_logits * logp).sum(-1)  # (B, P)
        loss = (ce * valid_mask.float()).sum() / (valid_mask.sum().float() + 1e-5)

        return self.log_clamp(loss, threshold=5.0) * self.cfg.bfactor_loss_weight

    def _plddt_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        target_trans: torch.Tensor,  # (B, P, 3) anchor positions
        mask: torch.Tensor,  # (B, P)
        dist_cutoff: float = 15.0,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Cross-entropy on per-token lDDT bins (pLDDT).
        Uses current predicted coords vs. anchor coords to compute lDDT.
        """
        plddt_logits = pred.pred_plddt  # (B, P, num_bins) or None
        if plddt_logits is None:
            zero = torch.tensor(0.0, device=batch.trans_t.device)
            return zero, zero, zero, zero, zero, zero

        num_bins = plddt_logits.shape[-1]
        pred_trans = pred.pred_trans_1  # (B, P, 3)

        # Compute pairwise distances (B, P, P)
        pred_dists = torch.cdist(pred_trans, pred_trans)
        true_dists = torch.cdist(target_trans, target_trans)

        # Mask: only consider valid residues excluding motifs, exclude self-pairs
        loss_mask = mask & ~batch.motif_mask  # (B, P)
        pair_mask = loss_mask.unsqueeze(2) & loss_mask.unsqueeze(1)  # (B, P, P)
        eye = torch.eye(
            pred_dists.size(1), device=pred_dists.device, dtype=torch.bool
        ).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        # Pairs that are "local neighbours" in the reference structure
        neighbors = (true_dists < dist_cutoff) & pair_mask  # (B, P, P)

        # Diff in distance between pred and true for every pair
        diff = (pred_dists - true_dists).abs().unsqueeze(-1)  # (B, P, P, 1)

        # Four tolerance levels, as defined by lDDT
        cuts = torch.tensor(
            [0.5, 1.0, 2.0, 4.0], device=diff.device, dtype=diff.dtype
        ).view(1, 1, 1, 4)

        # Pass/fail at each tolerance
        in_proximity = (diff < cuts) & neighbors.unsqueeze(-1)  # (B, P, P, 4)

        # Per-residue counts
        in_proximity_counts = in_proximity.float().sum((2, 3))  # (B, P)
        pair_count = neighbors.float().sum(2)  # (B, P)

        target_lddt_score = in_proximity_counts / (
            pair_count.clamp_min(1.0) * 4.0
        )  # (B, P)
        lddt_mask = (pair_count > 0).float()  # (B, P)
        with torch.no_grad():
            valid_lddt = loss_mask.float() * lddt_mask
            denom_lddt = valid_lddt.sum().clamp_min(1.0)
            lddt_mean = (target_lddt_score * valid_lddt).sum() / denom_lddt

        # Discretise into bins
        target_lddt_bins = torch.clamp(
            (target_lddt_score * num_bins).long(), max=num_bins - 1
        )
        target_logits = F.one_hot(target_lddt_bins, num_classes=num_bins).float()

        # Cross-entropy of bin logits
        logp = F.log_softmax(plddt_logits.float(), dim=-1)  # (B, P, num_bins)
        ce = -(target_logits * logp).sum(-1)  # (B, P)
        denom = (loss_mask.float() * lddt_mask).sum() + 1e-5
        loss_ce = (ce * loss_mask.float() * lddt_mask).sum() / denom
        loss_ce = torch.nan_to_num(loss_ce, nan=0.0)

        # Accuracy metrics: top-1 accuracy, ±1 bin accuracy, mean absolute error
        with torch.no_grad():
            valid = loss_mask & (lddt_mask > 0.5)  # (B, P)
            valid_f = valid.float()
            denom_valid = valid_f.sum().clamp_min(1.0)
            pred_bins = plddt_logits.argmax(dim=-1)  # (B, P)
            err = (pred_bins - target_lddt_bins).abs()
            acc = ((err == 0).float() * valid_f).sum() / denom_valid
            acc_pm1 = ((err <= 1).float() * valid_f).sum() / denom_valid
            mae_bins = (err.float() * valid_f).sum() / denom_valid

        loss = self.log_clamp(loss_ce, threshold=10.0) * self.cfg.plddt_loss_weight
        return loss, loss_ce.detach(), acc, acc_pm1, mae_bins, lddt_mean.detach()

    def calculate(
        self,
        batch: DataCorrupted,
        pred: ModelPrediction,
        couplings: TreeCouplings,
        bridged: DataBridged,
    ) -> tuple[BranchFlowLosses, BranchFlowMetrics]:
        B, P, D = batch.trans_t.shape
        assert pred.pred_trans_1.shape == (B, P, D)

        trans_coupling = couplings.translation
        valid_mask = batch.valid_mask  # (B, P_max)

        # Use the SAME packing indices that were used to create the model input.
        # CRITICAL: must use bridged.planar_position for sorting to match pack_present()
        idx_pack, pack_mask, P_b, P_max = DataBridged.pack_present_indices(
            bridged.present_mask, bridged.planar_position
        )

        # Pack translation anchors (i.e. targets) into (B, P_max, D) in the same order as model inputs/predictions
        trans_anchors_pack = trans_coupling.anchors.gather(
            1, idx_pack.unsqueeze(-1).expand(-1, -1, D)
        )
        # zero pad
        trans_anchors_pack = trans_anchors_pack * pack_mask.unsqueeze(-1).float()

        # Pack aatype anchor tokens (targets for base sequence loss)
        aatypes_coupling = couplings.aatypes
        aatype_anchors_pack = aatypes_coupling.anchors.gather(1, idx_pack)  # (B, P_max)
        aatype_anchors_pack = aatype_anchors_pack * pack_mask.long()  # zero pad

        # Pack anchor_probs for insertion sequence loss
        K = aatypes_coupling.anchor_probs.shape[-1]  # 21
        aatype_anchor_probs_pack = aatypes_coupling.anchor_probs.gather(
            1, idx_pack.unsqueeze(-1).expand(-1, -1, K)
        )  # (B, P_max, K)
        aatype_anchor_probs_pack = (
            aatype_anchor_probs_pack * pack_mask.unsqueeze(-1).float()
        )

        # Pack rotation anchors for rotation VF loss
        rotation_coupling = couplings.rotation
        # rotation_coupling.anchors is (B, A, 3, 3), pack to (B, P_max, 3, 3)
        rot_anchors_pack = rotation_coupling.anchors.gather(
            1, idx_pack.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3)
        )
        # Set non-present to identity
        identity = torch.eye(
            3, device=rot_anchors_pack.device, dtype=rot_anchors_pack.dtype
        )
        rot_anchors_pack = torch.where(
            pack_mask.unsqueeze(-1).unsqueeze(-1),
            rot_anchors_pack,
            identity.unsqueeze(0).unsqueeze(0).expand(B, P_max, -1, -1),
        )

        # Translations base loss + pairwise aux loss
        trans_loss = self._base_trans_loss(
            pred_trans=pred.pred_trans_1,
            target_trans=trans_anchors_pack,
            t=batch.t,
            mask=valid_mask,
        )
        pairwise_loss = self._pairwise_distance_loss(
            pred_trans=pred.pred_trans_1,
            target_trans=trans_anchors_pack,
            t=batch.t,
            mask=valid_mask,
        )
        trans_rmse_ang, trans_mae_ang = self._trans_error_metrics_ang(
            pred_trans=pred.pred_trans_1,
            target_trans=trans_anchors_pack,
            mask=valid_mask,
        )

        # Rotation VF loss
        rot_vf_loss = self._rot_vf_loss(
            pred_rotmats=pred.pred_rotmats_1,
            target_rotmats=rot_anchors_pack,
            rotmats_t=batch.rotmats_t,
            t=batch.t,
            mask=valid_mask,
        )
        rot_mae_deg, rot_rmse_deg = self._rot_geodesic_error_metrics_deg(
            pred_rotmats=pred.pred_rotmats_1,
            target_rotmats=rot_anchors_pack,
            mask=valid_mask,
        )

        # Base sequence losses:
        # - token CE against sampled anchor tokens
        # - soft CE against anchor probability targets
        base_seq_token_loss = self._seq_token_loss(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            t=batch.t,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
            motif_weight=self.cfg.seq_motif_weight,
        )
        base_seq_prob_loss = self._seq_prob_loss(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_probs=aatype_anchor_probs_pack,
            t=batch.t,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
            motif_weight=self.cfg.seq_motif_weight,
        )
        base_seq_loss = base_seq_token_loss + base_seq_prob_loss
        base_seq_ce = self._seq_ce_metric(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
            motif_weight=self.cfg.seq_motif_weight,
        )
        base_seq_acc = self._seq_acc_metric(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
            motif_weight=self.cfg.seq_motif_weight,
        )
        scaffold_mask = valid_mask & ~batch.motif_mask
        base_seq_ce_scaffold = self._seq_ce_metric(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            mask=scaffold_mask,
        )
        base_seq_acc_scaffold = self._seq_acc_metric(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            mask=scaffold_mask,
        )

        # Insertion sequence loss
        # (soft CE against anchor_probs, only where future insertions exist)
        insertion_seq_loss = self._insertion_seq_loss(
            pred_insertion_logits=pred.pred_insertion_logits,
            target_anchor_probs=aatype_anchor_probs_pack,
            mask=valid_mask & (batch.remaining_insertions > 0),
        )
        insertion_seq_ce = self._soft_ce_from_probs(
            pred_logits=pred.pred_insertion_logits,
            target_probs=aatype_anchor_probs_pack,
            mask=valid_mask & (batch.remaining_insertions > 0),
            apply_time_norm=False,
            per_example=False,
            mostly_mask_threshold=0.75,
            require_mass=False,
        )
        (
            insertion_target_entropy,
            insertion_ce_over_entropy,
            insertion_ce_minus_entropy,
        ) = self._insertion_ce_entropy_metrics(
            pred_logits=pred.pred_insertion_logits,
            target_probs=aatype_anchor_probs_pack,
            mask=valid_mask & (batch.remaining_insertions > 0),
            mostly_mask_threshold=0.75,
        )
        insertion_seq_kl = insertion_ce_minus_entropy

        # Insertion / split losses
        split_token_loss = self._split_token_loss(
            pred=pred,
            batch=batch,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
        )
        split_pooled_loss = self._split_pooled_loss(pred=pred, batch=batch)
        (
            split_event_ce,
            split_event_precision,
            split_event_recall,
            split_event_f1,
            split_rate_mae,
            split_rate_mae_pos,
            split_event_auprc,
            split_event_pos_rate,
            split_rate_corr,
        ) = self._split_metrics(
            pred=pred,
            batch=batch,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
        )

        # Deletion loss
        del_loss = self._deletion_loss(
            pred=pred,
            batch=batch,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
        )
        (
            del_event_ce,
            del_event_precision,
            del_event_recall,
            del_event_f1,
            del_event_auprc,
            del_event_pos_rate,
            del_prob_mean,
            del_true_rate,
            del_brier,
        ) = self._deletion_metrics(
            pred=pred,
            batch=batch,
            mask=valid_mask,
            motif_mask=batch.motif_mask,
        )

        # Confidence prediction losses
        bfactor_loss = self._bfactor_loss(
            pred=pred,
            batch=batch,
            mask=valid_mask,
        )
        (
            plddt_loss,
            plddt_ce,
            plddt_bin_acc,
            plddt_bin_acc_pm1,
            plddt_bin_mae,
            lddt_mean,
        ) = self._plddt_loss(
            pred=pred,
            batch=batch,
            target_trans=trans_anchors_pack,
            mask=valid_mask,
        )

        total_loss = (
            trans_loss
            + rot_vf_loss
            + pairwise_loss
            + base_seq_loss
            + insertion_seq_loss
            + split_token_loss
            + split_pooled_loss
            + del_loss
            + bfactor_loss
            + plddt_loss
        )

        losses = BranchFlowLosses(
            total_loss=total_loss,
            trans_loss=trans_loss,
            pairwise_loss=pairwise_loss,
            rot_vf_loss=rot_vf_loss,
            base_seq_loss=base_seq_loss,
            base_seq_prob_loss=base_seq_prob_loss,
            base_seq_token_loss=base_seq_token_loss,
            insertion_seq_loss=insertion_seq_loss,
            split_token_loss=split_token_loss,
            split_pooled_loss=split_pooled_loss,
            del_loss=del_loss,
            bfactor_loss=bfactor_loss,
            plddt_loss=plddt_loss,
        )
        metrics = BranchFlowMetrics(
            base_seq_ce=base_seq_ce,
            base_seq_acc=base_seq_acc,
            base_seq_ce_scaffold=base_seq_ce_scaffold,
            base_seq_acc_scaffold=base_seq_acc_scaffold,
            insertion_seq_ce=insertion_seq_ce,
            insertion_target_entropy=insertion_target_entropy,
            insertion_ce_over_entropy=insertion_ce_over_entropy,
            insertion_ce_minus_entropy=insertion_ce_minus_entropy,
            insertion_seq_kl=insertion_seq_kl,
            trans_rmse_ang=trans_rmse_ang,
            trans_mae_ang=trans_mae_ang,
            rot_mae_deg=rot_mae_deg,
            rot_rmse_deg=rot_rmse_deg,
            split_event_ce=split_event_ce,
            split_event_precision=split_event_precision,
            split_event_recall=split_event_recall,
            split_event_f1=split_event_f1,
            split_event_auprc=split_event_auprc,
            split_event_pos_rate=split_event_pos_rate,
            split_rate_mae=split_rate_mae,
            split_rate_mae_pos=split_rate_mae_pos,
            split_rate_corr=split_rate_corr,
            del_event_ce=del_event_ce,
            del_event_precision=del_event_precision,
            del_event_recall=del_event_recall,
            del_event_f1=del_event_f1,
            del_event_auprc=del_event_auprc,
            del_event_pos_rate=del_event_pos_rate,
            del_prob_mean=del_prob_mean,
            del_true_rate=del_true_rate,
            del_brier=del_brier,
            lddt_mean=lddt_mean,
            plddt_ce=plddt_ce,
            plddt_bin_acc=plddt_bin_acc,
            plddt_bin_acc_pm1=plddt_bin_acc_pm1,
            plddt_bin_mae=plddt_bin_mae,
        )
        return losses, metrics
