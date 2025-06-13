from dataclasses import dataclass, fields
from functools import cached_property
from typing import Optional, Tuple

import torch

from cogeneration.config.base import Config
from cogeneration.data import all_atom, so3_utils
from cogeneration.data.data_transforms import common_torsion_mask
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import ModelPrediction
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp

"""
TODO(model) Additional losses to consider:

- side chain interactions / clashes / vdw

- translation vector field loss 
    instead of just translation coordinates

- explicit rotation loss 
    instead of just rotation VF

- FAPE 
    requires OpenFold implementation realistically
"""


@dataclass
class AuxiliaryMetrics:
    """Training auxiliary metrics"""

    batch_train_loss: torch.Tensor
    batch_rot_loss: torch.Tensor
    batch_trans_loss: torch.Tensor
    batch_atom_loss: torch.Tensor
    batch_dist_mat_loss: torch.Tensor
    batch_torsions_loss: torch.Tensor
    batch_multimer_interface_loss: torch.Tensor
    batch_multimer_clash_loss: torch.Tensor
    batch_bfactor_loss: torch.Tensor
    batch_plddt_loss: torch.Tensor
    train_loss: torch.Tensor
    rots_vf_loss: torch.Tensor
    trans_loss: torch.Tensor
    atom_loss: torch.Tensor
    dist_mat_loss: torch.Tensor
    torsions_loss: torch.Tensor
    multimer_interface_loss: torch.Tensor
    multimer_clash_loss: torch.Tensor
    bfactor_loss: torch.Tensor
    plddt_loss: torch.Tensor
    loss_denom_num_res: torch.Tensor
    examples_per_step: torch.Tensor
    res_length: torch.Tensor


@dataclass
class TrainingLosses:
    """Struct to collect losses from model training step."""

    trans_loss: torch.Tensor
    rots_vf_loss: torch.Tensor
    torsions_loss: torch.Tensor  # optional
    auxiliary_loss: torch.Tensor
    aatypes_loss: torch.Tensor
    bfactor_loss: torch.Tensor  # optional
    plddt_loss: torch.Tensor  # optional
    train_loss: torch.Tensor  # aggregated loss

    def items(self):
        # avoid using asdict() because deepcopy on Tensors creates issues
        # instead, return iterater over fields
        return ((f.name, getattr(self, f.name)) for f in fields(self))


@dataclass
class BatchGroundTruth:
    """Container for Ground Truth properties"""

    cfg: Config
    batch: NoisyFeatures

    def __post_init__(self):
        gt_bb_atoms, atom37_mask, _, _ = all_atom.rigid_to_atom37(
            rigid=all_atom.create_rigid(rots=self.rotmats_1, trans=self.trans_1),
            torsions=self.torsions_1,
            aatype=self.aatypes_1.int(),
        )
        self.bb_atoms = gt_bb_atoms  # (B, N, 37, 3)
        self.atom37_mask = atom37_mask  # (B, N, 37)

    @property
    def trans_1(self) -> torch.Tensor:
        return self.batch[bp.trans_1]  # (B, N, 3)

    @property
    def rotmats_1(self) -> torch.Tensor:
        return self.batch[bp.rotmats_1]  # (B, N, 3, 3)

    @property
    def aatypes_1(self) -> torch.Tensor:
        return self.batch[bp.aatypes_1]  # (B, N)

    @property
    def torsions_1(self) -> torch.Tensor:
        return self.batch[bp.torsions_1]  # (B, N, 7, 2)

    @property
    def rot_vf(self):
        return so3_utils.calc_rot_vf(
            mat_t=self.batch[nbp.rotmats_t],
            mat_1=self.rotmats_1.type(torch.float32),
        )


@dataclass
class BatchNormScales:
    """
    struct for loss scales for each domain
    scaled at `1 - min(t, clip)` (i.e. higher as t -> 1)
    """

    r3: torch.Tensor  # (B, 1, 1)
    so3: torch.Tensor  # (B, 1, 1)
    cat: torch.Tensor  # (B, 1)

    @classmethod
    def from_batch_cfg(cls, batch: NoisyFeatures, cfg: Config):
        train_cfg = cfg.experiment.training

        # Timestep used for normalization.
        r3_t = batch[nbp.r3_t]  # (B, 1)
        so3_t = batch[nbp.so3_t]  # (B, 1)
        cat_t = batch[nbp.cat_t]  # (B, 1)

        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(train_cfg.t_normalize_clip)
        )

        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(train_cfg.t_normalize_clip)
        )

        if train_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(train_cfg.t_normalize_clip)
            )
        else:
            cat_norm_scale = torch.ones_like(cat_t)

        return cls(
            r3=r3_norm_scale,
            so3=so3_norm_scale,
            cat=cat_norm_scale,
        )


@dataclass
class BatchLossCalculator:
    cfg: Config
    batch: NoisyFeatures
    pred: ModelPrediction

    def __post_init__(self):
        self.gt = BatchGroundTruth(cfg=self.cfg, batch=self.batch)
        self.norm_scales = BatchNormScales.from_batch_cfg(self.batch, self.cfg)

    @property
    def train_cfg(self):
        return self.cfg.experiment.training

    @cached_property
    def loss_mask(self) -> torch.Tensor:
        """Mask for residues to consider for loss calculation."""
        bb_mask = self.batch[bp.res_mask]
        diffuse_mask = self.batch[bp.diffuse_mask]
        loss_mask = bb_mask * diffuse_mask

        if self.train_cfg.mask_plddt:
            loss_mask *= self.batch[bp.plddt_mask]

        # for inpainting, ignore motifs in loss, esp sequence since fixed, but also structure
        if bp.motif_mask in self.batch and self.batch[bp.motif_mask] is not None:
            loss_mask *= 1 - self.batch[bp.motif_mask]

        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")

        return loss_mask.bool()

    @cached_property
    def loss_denom_num_res(self) -> torch.Tensor:
        return torch.sum(self.loss_mask, dim=-1).float()

    @cached_property
    def batch_loss_mask(self) -> torch.Tensor:
        return torch.any(self.batch[bp.res_mask], dim=-1)

    def normalize_loss(self, x: torch.Tensor):
        return x.sum() / (self.batch_loss_mask.sum() + 1e-10)

    @cached_property
    def num_batch(self) -> int:
        return self.batch[bp.res_mask].shape[0]

    @cached_property
    def num_res(self) -> int:
        return self.batch[bp.res_mask].shape[1]

    @cached_property
    def n_atoms_modeled(self) -> int:
        # Number of backbone atoms able consider for loss
        # 3 for C-alpha, N, C atoms, 5 if we also consider psi angles
        # We still only use 3 atoms for frames + flow matching, but consider 5 for backbone loss
        n_bb_atoms = 3
        if self.cfg.model.predict_psi_torsions:
            n_bb_atoms = 5
        if self.cfg.model.predict_all_torsions:
            n_bb_atoms = 14

        return n_bb_atoms

    @cached_property
    def pred_bb_atoms(self) -> torch.Tensor:
        """Predicted backbone atoms (B, N, 37, 3)"""
        pred_bb_atoms = all_atom.atom37_from_trans_rot(
            trans=self.pred[pbp.pred_trans],
            rots=self.pred[pbp.pred_rotmats],
            torsions=self.pred[pbp.pred_torsions],
        )
        return pred_bb_atoms

    @cached_property
    def pred_rot_vf(self) -> torch.Tensor:
        """Predicted rotation vector field (B, N, 3)"""
        pred_rot_vf = so3_utils.calc_rot_vf(
            mat_t=self.batch[nbp.rotmats_t].type(torch.float32),
            mat_1=self.pred[pbp.pred_rotmats],
        )

        if torch.any(torch.isnan(pred_rot_vf)):
            raise ValueError("NaN encountered in pred_rots_vf")

        return pred_rot_vf

    def loss_trans(self) -> torch.Tensor:
        """Calculate translation loss."""
        r3_norm_scale = self.norm_scales.r3

        trans_error = (
            (self.gt.trans_1 - self.pred[pbp.pred_trans])
            / r3_norm_scale
            * self.train_cfg.trans_scale
        )
        trans_loss = torch.sum(
            trans_error**2 * self.loss_mask[..., None], dim=(-1, -2)
        ) / (self.loss_denom_num_res * 3)
        return trans_loss

    def loss_rot_vf(self) -> torch.Tensor:
        """Calculate rotation vector field loss."""
        so3_norm_scale = self.norm_scales.so3

        rot_vf_error = (self.gt.rot_vf - self.pred_rot_vf) / so3_norm_scale
        rot_vf_loss = torch.sum(
            rot_vf_error**2 * self.loss_mask[..., None], dim=(-1, -2)
        ) / (self.loss_denom_num_res * 3)
        return rot_vf_loss

    def loss_torsions(self) -> torch.Tensor:
        """
        Torsion loss computes cosine distance using `1 - cos`, if torsions are predicted.
        (use 1-cos because smooth, no wrap around issues, simpler than atan2.)

        The model may predict all 7 torsions, or a subset (e.g. only 1 [psi] angle).
        The loss is calculated by taking the appropriate subset of the ground truth torsions.
        """
        pred_torsions = self.pred[pbp.pred_torsions]  # (B, N, K, 2)  K={1,7}

        if pred_torsions is None:
            return torch.zeros(
                self.num_batch,
                device=self.batch[bp.res_mask].device,
                dtype=torch.float32,
            )

        # Only consider torsions valid in both GT and pred sequences.
        # Note may have effect of rarely predicting x3 and x4 angles,
        # if model predicted sequence rarely == GT sequence for large residues.
        torsion_mask = common_torsion_mask(
            seq1=self.gt.aatypes_1,
            seq2=self.pred[pbp.pred_aatypes],
        )

        # handle predicting torsion subset
        # ground‑truth psi torsion: (B, N, 7, 2) -> (B, N, K, 2)
        K = pred_torsions.shape[2]
        assert K in (1, 5, 7)
        gt_torsions = self.batch[bp.torsions_1]
        if K < 7:
            gt_torsions = gt_torsions[..., 2 : 2 + K, :]
            torsion_mask = torsion_mask[..., 2 : 2 + K, :]

        angle_mask = torsion_mask[..., 0].float()  # (B, N, K)

        # first, norm penalty to keep magnitude ~1
        norm_diff = pred_torsions.norm(dim=-1) - 1.0  # (B, N, K)
        loss_norm = norm_diff.abs()
        # scaled smaller than angle loss
        loss_norm = loss_norm * 0.05 * angle_mask  # (B, N, K)

        # dot product of unit vectors = cosine of angle difference
        cos_delta = (gt_torsions * pred_torsions).sum(-1)  # (B, N, K)
        # 1 - delta -> 0 when they match, up to 2 if they are opposite
        loss_per_angle = (1.0 - cos_delta) * angle_mask  # (B, N, K)

        # average per residue
        angle_cnt = angle_mask.sum(-1)  # (B, N)
        loss_per_residue = (loss_per_angle + loss_norm).sum(dim=-1) / (
            angle_cnt + 1e-10
        )  # (B, N)

        loss = (loss_per_residue * self.loss_mask).sum(-1) / (
            self.loss_denom_num_res + 1e-10
        )
        return loss

    def loss_aatypes_ce(self) -> torch.Tensor:

        flat_logits = self.pred[pbp.pred_logits].reshape(
            -1, self.cfg.model.aa_pred.aatype_pred_num_tokens
        )
        flat_aatypes = self.gt.aatypes_1.flatten().long()

        ce_loss = (
            torch.nn.functional.cross_entropy(
                flat_logits,
                flat_aatypes,
                reduction="none",
            ).reshape(self.num_batch, self.num_res)
            / self.norm_scales.cat
        )
        aatypes_loss = (
            torch.sum(ce_loss * self.loss_mask, dim=-1) / self.loss_denom_num_res
        )
        return aatypes_loss

    def loss_atom_positions(self, n_bb_atoms: int) -> torch.Tensor:
        """Calculate backbone atom loss."""
        r3_norm_scale = self.norm_scales.r3

        gt_bb_atoms = self.gt.bb_atoms[..., :n_bb_atoms, :]
        gt_bb_atoms *= self.train_cfg.bb_atom_scale / r3_norm_scale[..., None]

        pred_bb_atoms = self.pred_bb_atoms[..., :n_bb_atoms, :]
        pred_bb_atoms *= self.train_cfg.bb_atom_scale / r3_norm_scale[..., None]

        bb_atom_loss_mask = self.gt.atom37_mask * self.loss_mask[..., None]
        bb_atom_loss_mask = bb_atom_loss_mask[..., :n_bb_atoms]  # (B, N, n_bb_atoms)

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)

        return bb_atom_loss

    def loss_bb_pairwise_dist(
        self, proximity_threshold_ang: float = 6.0
    ) -> torch.Tensor:
        # Pairwise distance loss
        num_batch = self.num_batch
        num_res = self.num_res
        # only consider backbone atoms for pairwise distances
        n_bb_atoms = 3

        gt_bb_atoms = self.gt.bb_atoms[..., :n_bb_atoms, :]
        pred_bb_atoms = self.pred_bb_atoms[..., :n_bb_atoms, :]

        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_bb_atoms, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * n_bb_atoms, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(self.loss_mask[:, :, None], (1, 1, n_bb_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_bb_atoms])
        flat_res_mask = torch.tile(self.loss_mask[:, :, None], (1, 1, n_bb_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_bb_atoms])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # limit distance loss to atoms in proximity
        proximity_mask = gt_pair_dists < proximity_threshold_ang
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) + 1

        return dist_mat_loss

    def loss_multimer_interface(
        self, interface_distance_ang: float = 8.0
    ) -> torch.Tensor:
        """
        Recapitulation of ground‐truth inter‐chain contacts for Ca < interface_distance_ang
        """
        chain_idx = self.batch[bp.chain_idx]  # (B, N)
        multi_chain_mask = chain_idx.max(-1).values != chain_idx.min(-1).values

        if not multi_chain_mask.any():
            return torch.zeros(self.num_batch, device=chain_idx.device)

        # ground truth interface
        gt_pos = self.gt.trans_1  # Ca only (B, N, 3)
        inter_chain = chain_idx.unsqueeze(1) != chain_idx.unsqueeze(2)  # (B, N, N)
        gt_pairwise_dist = torch.norm(
            gt_pos.unsqueeze(2) - gt_pos.unsqueeze(1), dim=-1
        )  # (B, N, N)
        interface_mask = inter_chain & (
            gt_pairwise_dist < interface_distance_ang
        )  # (B, N, N)

        # predicted interface
        pred_pos = self.pred[pbp.pred_trans]  # Ca-only (B, N, 3)
        dist_pred = torch.norm(pred_pos.unsqueeze(2) - pred_pos.unsqueeze(1), dim=-1)
        missing_contacts = torch.relu(dist_pred - interface_distance_ang)  # (B, N, N)
        missing_contacts = missing_contacts * interface_mask

        loss = missing_contacts.sum((1, 2)) / interface_mask.sum((1, 2)).clamp_min(1.0)

        return loss * multi_chain_mask.float()

    def loss_multimer_clash(self, clash_dist_ang: float = 2.0) -> torch.Tensor:
        """Penalize backbone atom (Ca, N, C) clashes across chains"""
        chain_idx = self.batch[bp.chain_idx]  # (B, N)
        multi_chain_mask = chain_idx.max(-1).values != chain_idx.min(-1).values  # (B,)

        if not multi_chain_mask.any():
            return torch.zeros(self.num_batch, device=chain_idx.device)

        n_bb_atoms = 3  # only consider backbone collisions
        pred_bb_atoms = self.pred_bb_atoms[..., :n_bb_atoms, :]  # (B, N, 3, 3)
        pred_flat_atoms = pred_bb_atoms.reshape(self.num_batch, -1, 3)  # (B, M, 3)

        # limit to valid residues
        atom_mask = self.gt.atom37_mask[..., :n_bb_atoms]  # (B, N, 3)
        atom_mask *= self.loss_mask.unsqueeze(-1)  # (B, N, 3)
        atom_mask_flat = atom_mask.reshape(self.num_batch, -1)  # (B, M, )
        valid_pair = atom_mask_flat.unsqueeze(-1) & atom_mask_flat.unsqueeze(
            -2
        )  # (B, M, M)

        # limit to inter‐chain
        chains = chain_idx.unsqueeze(-1).expand(-1, -1, 3).reshape(self.num_batch, -1)
        inter_chain = chains.unsqueeze(-1) != chains.unsqueeze(-2)  # (B, M, M)
        # limit self‐comparisons
        eye = torch.eye(valid_pair.size(-1), device=valid_pair.device, dtype=torch.bool)
        not_self = ~eye.unsqueeze(0)  # (1, M, M)

        # calculate clashes using pairwise squared distances
        diff = pred_flat_atoms.unsqueeze(2) - pred_flat_atoms.unsqueeze(
            1
        )  # (B, M, M, 3)
        pairwise_dists = torch.norm(diff, dim=-1)  # (B, M, M)
        clashes = (
            (pairwise_dists < clash_dist_ang) & inter_chain & valid_pair & not_self
        )

        clash_sum = clashes.sum((1, 2))  # (B, )
        clash_denom = valid_pair.float().sum((1, 2)).clamp_min(1.0)  # (B, )
        clash_rate = clash_sum / clash_denom

        return clash_rate * multi_chain_mask.float()

    def loss_bfactor(self) -> torch.Tensor:
        """
        Cross-entropy loss on b-factor histograms, adapted from Boltz.
        b-factor prediction is optional, and invalid when all b-factors are zero
        (e.g. as in synthetic examples).
        """
        pred_logits = self.pred.get(pbp.pred_bfactor, None)  # (B, N, num_bins)
        if pred_logits is None or bp.res_bfactor not in self.batch:
            return torch.zeros(self.num_batch, device=self.batch[bp.res_mask].device)

        bins = pred_logits.shape[-1]
        gt_bfactor = self.batch[bp.res_bfactor]  # (B, N)
        boundaries = torch.linspace(0.0, 100.0, bins - 1, device=gt_bfactor.device)

        # discretise ground-truth b-factors
        bin_idx = (gt_bfactor.unsqueeze(-1) > boundaries).sum(-1).long()  # (B, N)
        target_logits = torch.nn.functional.one_hot(bin_idx, num_classes=bins).float()

        # mask out synthetic examples (all-zero b-factors) + padding
        valid_mask = (gt_bfactor > 1e-5) & self.loss_mask  # (B, N)
        if not valid_mask.any():
            return torch.zeros(self.num_batch, device=self.batch[bp.res_mask].device)

        # cross-entropy
        logp = torch.nn.functional.log_softmax(pred_logits.float(), dim=-1)
        ce = -(target_logits * logp).sum(-1)  # (B, N)
        loss = (ce * valid_mask.float()).sum(-1) / (valid_mask.sum(-1).float() + 1e-5)

        return loss

    @staticmethod
    def _lddt_per_residue(
        pred_dists: torch.Tensor,  # (B, N, N)
        true_dists: torch.Tensor,  # (B, N, N)
        pair_mask: torch.Tensor,  # (B, N, N) pairs to consider
        dist_cutoff: float = 15.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-residue target lDDT, returning score and a validity mask (i.e. a valid neighbor exists)
        """
        # pairs that are "local neighbours" in the reference structure
        neighbors = (true_dists < dist_cutoff) & pair_mask  # (B, N, N)

        # diff in distance between pred and true for every pair
        diff = (pred_dists - true_dists).abs().unsqueeze(-1)  # (B, N, N, 1)

        # four tolerance levels, as defined by lDDT
        cuts = torch.tensor(
            [0.5, 1.0, 2.0, 4.0], device=diff.device, dtype=diff.dtype
        ).view(1, 1, 1, 4)

        # pass/fail at each tolerance
        in_proximity = (diff < cuts) & neighbors.unsqueeze(-1)  # (B, N, N, 4)

        # per-residue counts
        in_proximity_counts = in_proximity.float().sum((2, 3))  # (B, N)
        pair_count = neighbors.float().sum(2)  # (B, N)

        score = in_proximity_counts / (pair_count.clamp_min(1.0) * 4.0)  # (B, N)
        valid = (pair_count > 0).float()  # (B, N)

        return score, valid

    def loss_plddt(self, dist_cutoff: float = 15.0) -> torch.Tensor:
        """
        Cross-entropy on per-token lDDT bins (pLDDT).
        Uses current predicted coords vs. ground-truth coords to recompute lDDT.
        """
        plddt_logits = self.pred.get(pbp.pred_lddt, None)  # (B, N, num_bins)
        if plddt_logits is None:
            return torch.zeros(self.num_batch, device=self.batch[bp.res_mask].device)

        num_bins = plddt_logits.shape[-1]

        # gather Ca coordinates (B, N, 3) to compute pairwise distances (B, N, N)
        pred_trans = self.pred[pbp.pred_trans]
        true_trans = self.gt.trans_1
        pred_dists = torch.cdist(pred_trans, pred_trans)
        true_dists = torch.cdist(true_trans, true_trans)

        # mask: only consider resolved residues and exclude self-pairs
        pair_mask = self.loss_mask.unsqueeze(2) & self.loss_mask.unsqueeze(
            1
        )  # (B, N, N)
        # remove self
        eye = torch.eye(
            pred_dists.size(1), device=pred_dists.device, dtype=torch.bool
        ).unsqueeze(
            0
        )  # (1, N, N)
        pair_mask &= ~eye

        # lDDT score and mask (for which residues have lDDT neighbors)
        target_lddt_score, lddt_mask = self._lddt_per_residue(
            pred_dists, true_dists, pair_mask, dist_cutoff
        )  # (B,N), (B,N)
        # discretise into bins
        target_lddt_bins = torch.clamp(
            (target_lddt_score * num_bins).long(), max=num_bins - 1
        )
        target_logits = torch.nn.functional.one_hot(
            target_lddt_bins, num_classes=num_bins
        ).float()

        # cross-entropy of bin logits
        logp = torch.nn.functional.log_softmax(
            plddt_logits.float(), dim=-1
        )  # (B, N, num_bins)
        ce = -(target_logits * logp).sum(-1)  # (B, N)
        # mask & average
        loss = (ce * self.loss_mask.float() * lddt_mask).sum(-1) / (
            (self.loss_mask.float() * lddt_mask).sum(-1) + 1e-5
        )  # (B,)
        loss = torch.nan_to_num(loss, nan=0.0)

        return loss

    def calculate(self) -> Tuple[TrainingLosses, AuxiliaryMetrics]:
        """Calculate losses for the batch."""
        loss_trans = self.loss_trans()
        loss_rot_vf = self.loss_rot_vf()
        loss_torsions = self.loss_torsions()
        loss_aatypes_ce = self.loss_aatypes_ce()
        loss_atom_positions = self.loss_atom_positions(n_bb_atoms=self.n_atoms_modeled)
        loss_pairwise_dist = self.loss_bb_pairwise_dist()
        loss_multimer_interface = self.loss_multimer_interface()
        loss_multimer_clash = self.loss_multimer_clash()
        loss_bfactor = self.loss_bfactor()
        loss_plddt = self.loss_plddt()

        # scale losses by cfg weights
        loss_trans = loss_trans * self.train_cfg.translation_loss_weight
        loss_rot_vf = loss_rot_vf * self.train_cfg.rotation_loss_weight
        loss_torsions = loss_torsions * self.train_cfg.torsion_loss_weight
        loss_aatypes_ce = loss_aatypes_ce * self.train_cfg.aatypes_loss_weight
        loss_atom_positions = (
            loss_atom_positions * self.train_cfg.aux_bb_atom_loss_weight
        )
        loss_pairwise_dist = loss_pairwise_dist * self.train_cfg.aux_bb_pair_loss_weight
        loss_multimer_interface = (
            loss_multimer_interface * self.train_cfg.aux_multimer_interface_loss_weight
        )
        loss_multimer_clash = (
            loss_multimer_clash * self.train_cfg.aux_multimer_clash_loss_weight
        )
        loss_bfactor = loss_bfactor * self.train_cfg.aux_bfactor_loss_weight
        loss_plddt = loss_plddt * self.train_cfg.aux_plddt_loss_weight

        # auxiliary loss
        loss_auxiliary = (
            loss_atom_positions
            + loss_pairwise_dist
            + loss_multimer_clash
            + loss_multimer_interface
            + loss_bfactor
            + loss_plddt
        )
        loss_auxiliary *= self.train_cfg.aux_loss_weight

        # limit auxiliary loss to certain times
        loss_auxiliary *= self.batch[nbp.r3_t][:, 0] > self.train_cfg.aux_loss_t_pass
        loss_auxiliary *= self.batch[nbp.so3_t][:, 0] > self.train_cfg.aux_loss_t_pass

        # clamp certain losses
        loss_trans = torch.clamp(loss_trans, max=5)
        loss_auxiliary = torch.clamp(loss_auxiliary, max=5)

        # aggregate losses
        loss_total = loss_trans + loss_rot_vf + loss_aatypes_ce + loss_auxiliary

        if torch.any(torch.isnan(loss_total)):
            raise ValueError("NaN loss encountered")
        assert loss_total.shape == (self.num_batch,)

        aux = AuxiliaryMetrics(
            batch_train_loss=loss_total,
            batch_rot_loss=loss_rot_vf,
            batch_trans_loss=loss_trans,
            batch_atom_loss=loss_atom_positions,
            batch_dist_mat_loss=loss_pairwise_dist,
            batch_torsions_loss=loss_torsions,
            batch_multimer_interface_loss=loss_multimer_interface,
            batch_multimer_clash_loss=loss_multimer_clash,
            batch_bfactor_loss=loss_bfactor,
            batch_plddt_loss=loss_plddt,
            train_loss=self.normalize_loss(loss_total),
            rots_vf_loss=self.normalize_loss(loss_rot_vf),
            trans_loss=self.normalize_loss(loss_trans),
            atom_loss=self.normalize_loss(loss_atom_positions),
            dist_mat_loss=self.normalize_loss(loss_pairwise_dist),
            torsions_loss=self.normalize_loss(loss_torsions),
            multimer_interface_loss=self.normalize_loss(loss_multimer_interface),
            multimer_clash_loss=self.normalize_loss(loss_multimer_clash),
            bfactor_loss=self.normalize_loss(loss_bfactor),
            plddt_loss=self.normalize_loss(loss_plddt),
            loss_denom_num_res=self.loss_denom_num_res,
            examples_per_step=torch.tensor(self.num_batch),
            res_length=torch.mean(torch.sum(self.batch[bp.res_mask], dim=-1).float()),
        )

        losses = TrainingLosses(
            trans_loss=loss_trans,
            rots_vf_loss=loss_rot_vf,
            torsions_loss=loss_torsions,
            auxiliary_loss=loss_auxiliary,
            aatypes_loss=loss_aatypes_ce,
            bfactor_loss=loss_bfactor,
            plddt_loss=loss_plddt,
            train_loss=loss_total,
        )

        return losses, aux
