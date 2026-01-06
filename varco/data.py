from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch

from cogeneration.data import all_atom
from cogeneration.data.const import MASK_TOKEN_INDEX
from varco.tensor_utils import clone_detach, gather_and_pad, to_device
from varco.tree_plan import BatchedTreePlan, TreePlan

# B = batch size
# N = sampled positions in data (t=1)
# R = root positions in base (t=0)
# M = motif positions (constant t=0 -> t=1)
# A = aligned number of positions in constructed tree (with anchors + to-be-deleted)
# A_max = batch max length of A positions
# P = present positions at time t (M + R <= P <= A)
# P_max = batch max length of padded P positions

# Training structs:
# DataSample - t=1 data. per-sample (N,) domains at t=1, with predefined TreePlan
# DataBatch - t=1 collated DataSample. (B, N) domains with BatchedTreePlan [use LengthBatcher, t=1 all same length]
# DataBridged - time t, corrupted (B, A) domains + topology in tree-aligned space
# DataCorrupted - time t, packed (B, P_max) present positions for model input (i.e. tree node subset present at t)
# ModelPrediction - model outputs (B, P_max) for loss calculation
# ModelPrediction + DataBridged + Coupling (for domain anchors + P <-> A mapping) -> LossCalculator -> Losses
#
# Sampling structs:
# DataCorrupted - packed (B, P_max) state, mutated in-place with insert/delete
# SampleTrajectory - list of DataCorrupted snapshots and ModelPrediction per step


@dataclass
class DataSample:
    """Per-sample data (length N) at time t=1."""

    tree_plan: TreePlan
    motif_mask: torch.Tensor  # (N,) bool
    res_mask: torch.Tensor  # (N,) int
    chain_idx: torch.Tensor  # (N,) int
    trans_1: torch.Tensor  # (N, 3)
    rotmats_1: torch.Tensor  # (N, 3, 3)
    aatypes_1: torch.Tensor  # (N,)
    contact_conditioning: torch.Tensor  # (N, N) distance matrix for contact constraints
    res_bfactor: torch.Tensor  # (N,) Ca temp b-factors (exp) or 0.0 (predicted)
    res_plddt: torch.Tensor  # (N,) pLDDT scores (predicted) or 100.0 (exp)


@dataclass
class DataBatch:
    """
    Batched data for training.
    Use LengthBatcher, so all data samples have the same length N (no padding).
    """

    tree: BatchedTreePlan
    motif_mask: torch.Tensor  # (B, N) bool
    res_mask: torch.Tensor  # (B, N) int
    chain_idx: torch.Tensor  # (B, N) int
    trans_1: torch.Tensor  # (B, N, 3)
    rotmats_1: torch.Tensor  # (B, N, 3, 3)
    aatypes_1: torch.Tensor  # (B, N)
    contact_conditioning: (
        torch.Tensor
    )  # (B, N, N) distance matrix for contact constraints
    res_bfactor: torch.Tensor  # (B, N) Ca temp b-factors (exp) or 0.0 (predicted)
    res_plddt: torch.Tensor  # (B, N) pLDDT scores (predicted) or 100.0 (exp)


@dataclass
class DataCorrupted:
    """Model input: corrupted and packed (length P_max) points present at time t"""

    t: torch.Tensor  # (B,)
    motif_mask: torch.Tensor  # (B, P_max) bool; True for fixed motif positions
    birth_time: torch.Tensor  # (B, P_max) 0.0 for motifs & roots, +inf for padding
    res_mask: torch.Tensor  # (B, P_max) int
    chain_idx: torch.Tensor  # (B, P_max) int
    trans_t: torch.Tensor  # (B, P_max, 3)
    rotmats_t: torch.Tensor  # (B, P_max, 3, 3)
    aatypes_t: torch.Tensor  # (B, P_max)
    trans_1_motifs: (
        torch.Tensor
    )  # (B, P_max, 3) true t=1 positions for motifs (for guidance)
    rotmats_1_motifs: (
        torch.Tensor
    )  # (B, P_max, 3, 3) true t=1 rotations for motifs (for guidance)
    contact_conditioning: Optional[
        torch.Tensor
    ]  # (B, P_max, P_max) contact constraints for edge modulation
    res_bfactor: Optional[
        torch.Tensor
    ]  # (B, P_max) Ca temp b-factors (exp) or 0.0 (predicted)
    res_plddt: Optional[
        torch.Tensor
    ]  # (B, P_max) pLDDT scores (predicted) or 100.0 (exp)

    # supervision (corruption only)
    remaining_insertions: Optional[torch.Tensor] = (
        None  # (B, P_max) remaining splits per present token
    )
    deleted: Optional[torch.Tensor] = None  # (B, P_max) 1 if destined-to-delete

    @property
    def valid_mask(self) -> torch.Tensor:  # (B, P_max)
        """True for present and non-padding tokens"""
        return (self.birth_time <= self.t[:, None]).bool()

    @property
    def remaining_total(self) -> torch.Tensor:  # (B,)
        """Sum of remaining insertions for all present tokens"""
        if self.remaining_insertions is None:
            return torch.zeros(
                (self.t.shape[0],), device=self.t.device, dtype=torch.long
            )
        return (self.remaining_insertions * self.valid_mask.long()).sum(dim=1).long()

    def to(self, device: torch.device) -> "DataCorrupted":
        """Move all tensors to specified device"""
        return DataCorrupted(
            t=to_device(self.t, device),
            motif_mask=to_device(self.motif_mask, device),
            birth_time=to_device(self.birth_time, device),
            res_mask=to_device(self.res_mask, device),
            chain_idx=to_device(self.chain_idx, device),
            trans_t=to_device(self.trans_t, device),
            rotmats_t=to_device(self.rotmats_t, device),
            aatypes_t=to_device(self.aatypes_t, device),
            trans_1_motifs=to_device(self.trans_1_motifs, device),
            rotmats_1_motifs=to_device(self.rotmats_1_motifs, device),
            contact_conditioning=to_device(self.contact_conditioning, device),
            res_bfactor=to_device(self.res_bfactor, device),
            res_plddt=to_device(self.res_plddt, device),
            remaining_insertions=to_device(self.remaining_insertions, device),
            deleted=to_device(self.deleted, device),
        )

    def detach_clone(self, device: Optional[torch.device] = None) -> "DataCorrupted":
        """Detach and clone the data, e.g. to save in trajectory.

        Args:
            device: If provided, move tensors to this device (e.g. 'cpu' for trajectories)
        """
        result = DataCorrupted(
            t=clone_detach(self.t),
            motif_mask=clone_detach(self.motif_mask),
            birth_time=clone_detach(self.birth_time),
            res_mask=clone_detach(self.res_mask),
            chain_idx=clone_detach(self.chain_idx),
            trans_t=clone_detach(self.trans_t),
            rotmats_t=clone_detach(self.rotmats_t),
            aatypes_t=clone_detach(self.aatypes_t),
            trans_1_motifs=clone_detach(self.trans_1_motifs),
            rotmats_1_motifs=clone_detach(self.rotmats_1_motifs),
            contact_conditioning=clone_detach(self.contact_conditioning),
            res_bfactor=clone_detach(self.res_bfactor),
            res_plddt=clone_detach(self.res_plddt),
            remaining_insertions=clone_detach(self.remaining_insertions),
            deleted=clone_detach(self.deleted),
        )
        if device is not None:
            return result.to(device)
        return result

    def apply_insertions_deletions(
        self,
        insertions: torch.Tensor,  # (B, P) bool
        deletions: torch.Tensor,  # (B, P) bool
        t_birth: float,
    ) -> Tuple["DataCorrupted", torch.Tensor, torch.Tensor]:
        """
        Apply insertions and deletions to create a new batch.
        Returns new batch, positions where insertions occurred, and gather indices.

        Deletions remove positions from the sequence.
        Insertions duplicate positions, and child copies parent's features.

        Algorithm walkthrough:
        [0, 1, 2, 3] original valid positions
        [F, F, T, F] deletions - delete position 2
        [F, T, F, F] insertions - insert at position 1

        [1, 2, 0, 1] multiplicity in {0,1,2} (keep@0 - 1, ins@1 = 2, del@2 = 0)
        [1, 3, 3, 4] cumsum tracks total outputs 0 -> i
        [0, 1, 1, 3] gather_idx maps to source position
        [F, F, T, F] is_insertion tracks insertions in output
        """
        B, P = self.trans_t.shape[:2]
        device = self.trans_t.device

        valid = self.valid_mask  # (B, P)
        keep = valid & ~deletions  # (B, P)

        # Multiplicity = number of times each source position appears in output
        multiplicity = keep.long() + (keep & insertions).long()  # (B, P)

        # predetermine final batch length
        out_lens = multiplicity.sum(dim=1)  # (B,)
        P_new = int(out_lens.max().item())

        # Vectorized building of gather indices using cumsum + searchsorted
        cumsum = multiplicity.cumsum(dim=1).contiguous()  # (B, P)
        out_pos = (
            torch.arange(P_new, device=device).unsqueeze(0).expand(B, -1)
        ).contiguous()  # (B, P_new)
        # For each out_pos, find source index
        gather_idx = torch.searchsorted(cumsum, out_pos, right=True)  # (B, P_new)
        gather_idx = gather_idx.clamp(0, P - 1)

        # Insertion = second occurrence of a source = consecutive gather_idx match
        is_insertion = torch.zeros((B, P_new), dtype=torch.bool, device=device)
        is_insertion[:, 1:] = gather_idx[:, 1:] == gather_idx[:, :-1]

        # When an insertion happens, one source position becomes two child positions.
        # Track `is_split_child` and set birth_time for each.
        is_split_child = is_insertion.clone()
        is_split_child[:, :-1] = is_split_child[:, :-1] | is_insertion[:, 1:]

        # Valid mask for new batch
        new_valid = out_pos < out_lens.unsqueeze(1)
        is_insertion = is_insertion & new_valid
        is_split_child = is_split_child & new_valid

        # Update birth_time for split children to current time t, set padding to inf
        new_birth = gather_and_pad(
            self.birth_time, gather_idx, new_valid, fill_value=float("inf")
        )
        new_birth = torch.where(
            is_split_child, torch.full_like(new_birth, t_birth), new_birth
        )

        # Skip mapping optional supervised fields, make sure not defined (not used while sampling)
        assert self.remaining_insertions is None
        assert self.deleted is None

        device = self.trans_t.device
        identity = torch.eye(3, device=device, dtype=self.rotmats_t.dtype)

        new_batch = DataCorrupted(
            t=self.t.clone(),
            motif_mask=gather_and_pad(
                self.motif_mask, gather_idx, new_valid, fill_value=0
            ),
            birth_time=new_birth,
            res_mask=gather_and_pad(self.res_mask, gather_idx, new_valid, fill_value=0),
            chain_idx=gather_and_pad(
                self.chain_idx, gather_idx, new_valid, fill_value=0
            ),
            trans_t=gather_and_pad(self.trans_t, gather_idx, new_valid, fill_value=0.0),
            rotmats_t=gather_and_pad(
                self.rotmats_t, gather_idx, new_valid, fill_value=identity
            ),
            aatypes_t=gather_and_pad(
                self.aatypes_t, gather_idx, new_valid, fill_value=MASK_TOKEN_INDEX
            ),
            trans_1_motifs=gather_and_pad(
                self.trans_1_motifs, gather_idx, new_valid, fill_value=0.0
            ),
            rotmats_1_motifs=gather_and_pad(
                self.rotmats_1_motifs, gather_idx, new_valid, fill_value=identity
            ),
            # inserted positions inherit parent's contacts (always 0 for scaffolds)
            contact_conditioning=gather_and_pad(
                self.contact_conditioning,
                gather_idx,
                new_valid,
                fill_value=0.0,
                is_2d=True,
            ),
            res_bfactor=gather_and_pad(
                self.res_bfactor, gather_idx, new_valid, fill_value=0.0
            ),
            res_plddt=gather_and_pad(
                self.res_plddt, gather_idx, new_valid, fill_value=0.0
            ),
        )

        return new_batch, is_insertion, gather_idx

    def to_atom37(self) -> torch.Tensor:
        """
        Convert (trans_t, rotmats_t, aatypes_t) to atom37 representation.

        Returns:
            atom37: (B, P, 37, 3) atom positions in angstroms
        """
        return all_atom.atom37_from_trans_rot(
            trans=self.trans_t,
            rots=self.rotmats_t,
            torsions=None,
            aatype=self.aatypes_t,
            res_mask=self.valid_mask.float(),
            unknown_to_alanine=True,
        )


@dataclass
class DataBridged:
    """Corrupted and aligned (length A) points at time t"""

    t: torch.Tensor  # (B,)
    present_mask: torch.Tensor  # (B, A)
    motif_mask: torch.Tensor  # (B, A) bool
    birth_time: torch.Tensor  # (B, A) 0.0 for roots
    res_mask: torch.Tensor  # (B, A)
    chain_idx: torch.Tensor  # (B, A)
    trans_t: torch.Tensor  # (B, A, 3)
    rotmats_t: torch.Tensor  # (B, A, 3, 3)
    aatypes_t: torch.Tensor  # (B, A)
    # confidence metrics
    res_bfactor: torch.Tensor  # (B, A) Ca temp b-factors (exp) or 0.0 (predicted)
    res_plddt: torch.Tensor  # (B, A) pLDDT scores (predicted) or 100.0 (exp)
    contact_conditioning: (
        torch.Tensor
    )  # (B, A, A) contact constraints (motif-motif only)
    # guidance: t=1 values in motifs
    trans_1_motifs: torch.Tensor  # (B, A, 3)
    rotmats_1_motifs: torch.Tensor  # (B, A, 3, 3)
    # supervision
    remaining_insertions: torch.Tensor  # (B, A) target count per aligned node
    deleted: torch.Tensor  # (B, A) bool, aligned deletion label (leaf only)
    # planar ordering for packing (preserves sequence order)
    planar_position: torch.Tensor  # (B, A) float, position in sequence for sorting

    @staticmethod
    def pack_present_indices(
        present_mask: torch.Tensor,  # (B, A) bool
        planar_position: torch.Tensor,  # (B, A) float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Derive packed indices deterministically from a (B, A) present mask.

        planar_position is required to ensure consistent ordering between
        model inputs and loss targets. Omitting it previously caused a critical bug
        where targets were misaligned with predictions.

        Args:
            present_mask: (B, A) bool mask of present nodes
            planar_position: (B, A) float, for ordering

        Returns:
            idx_pack: (B, P_max) P -> A mapping; aligned indices for packed slots
            pack_mask: (B, P_max) True for real slots, False for padding
            P_b: (B,) number of present slots per example
            P_max: int max present slots in batch
        """
        if present_mask.ndim != 2:
            raise ValueError(
                f"present_mask must have shape (B, A); got {tuple(present_mask.shape)}"
            )
        if planar_position.shape != present_mask.shape:
            raise ValueError(
                f"planar_position shape {tuple(planar_position.shape)} must match "
                f"present_mask shape {tuple(present_mask.shape)}"
            )

        B, A = present_mask.shape
        device = present_mask.device

        # Sort by: (1) not present (inf) vs present, (2) planar position (sequence order)
        # Use a large value for non-present to push them to the end
        sort_key = torch.where(
            present_mask,
            planar_position,
            torch.full_like(planar_position, float("inf")),
        )

        idx_sorted = torch.argsort(sort_key, dim=1, stable=True)  # (B, A)

        P_b = present_mask.sum(dim=1)  # (B,)
        P_max = int(P_b.max().item()) if B > 0 else 0
        if P_max == 0:
            raise ValueError("No present nodes")

        idx_pack = idx_sorted[:, :P_max]  # (B, P_max)
        pack_mask = (
            torch.arange(P_max, device=device)[None, :] < P_b[:, None]
        )  # (B, P_max)
        return idx_pack, pack_mask, P_b, P_max

    def _pack_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.pack_present_indices(self.present_mask, self.planar_position)

    def pack_present(self) -> DataCorrupted:
        """Pack aligned (A) state into present (P_max) state for model input."""
        idx_pack, pack_mask, P_b, P_max = self._pack_indices()
        identity = torch.eye(
            3, device=self.rotmats_t.device, dtype=self.rotmats_t.dtype
        )

        return DataCorrupted(
            t=self.t,
            motif_mask=gather_and_pad(
                self.motif_mask, idx_pack, pack_mask, fill_value=0
            ),
            birth_time=gather_and_pad(
                self.birth_time, idx_pack, pack_mask, fill_value=float("inf")
            ),
            res_mask=gather_and_pad(self.res_mask, idx_pack, pack_mask, fill_value=0),
            chain_idx=gather_and_pad(self.chain_idx, idx_pack, pack_mask, fill_value=0),
            trans_t=gather_and_pad(self.trans_t, idx_pack, pack_mask, fill_value=0.0),
            rotmats_t=gather_and_pad(
                self.rotmats_t,
                idx_pack,
                pack_mask,
                fill_value=identity,
            ),
            aatypes_t=gather_and_pad(
                self.aatypes_t,
                idx_pack,
                pack_mask,
                fill_value=MASK_TOKEN_INDEX,
            ),
            res_bfactor=gather_and_pad(
                self.res_bfactor, idx_pack, pack_mask, fill_value=0.0
            ),
            res_plddt=gather_and_pad(
                self.res_plddt, idx_pack, pack_mask, fill_value=0.0
            ),
            # contact_conditioning: (B, A, A) -> (B, P, P) using 2D gather
            contact_conditioning=gather_and_pad(
                self.contact_conditioning,
                idx_pack,
                pack_mask,
                fill_value=0.0,
                is_2d=True,
            ),
            trans_1_motifs=gather_and_pad(
                self.trans_1_motifs, idx_pack, pack_mask, fill_value=0.0
            ),
            rotmats_1_motifs=gather_and_pad(
                self.rotmats_1_motifs,
                idx_pack,
                pack_mask,
                fill_value=identity,
            ),
            remaining_insertions=gather_and_pad(
                self.remaining_insertions, idx_pack, pack_mask, fill_value=0
            ),
            deleted=gather_and_pad(self.deleted, idx_pack, pack_mask, fill_value=0),
        )

    def validate(self) -> None:
        B, A, D = self.trans_t.shape
        if self.birth_time.shape != (B, A):
            raise ValueError("birth_time shape mismatch")
        if self.present_mask.shape != (B, A):
            raise ValueError("present_mask shape mismatch")
        if self.motif_mask.shape != (B, A):
            raise ValueError("motif_mask shape mismatch")
        if self.aatypes_t.shape != (B, A):
            raise ValueError("aatypes_t shape mismatch")
        if self.trans_1_motifs.shape != (B, A, D):
            raise ValueError("trans_1_motifs shape mismatch")
        if self.rotmats_t.shape != (B, A, 3, 3):
            raise ValueError("rotmats_t shape mismatch")
        if self.rotmats_1_motifs.shape != (B, A, 3, 3):
            raise ValueError("rotmats_1_motifs shape mismatch")
        if self.remaining_insertions.shape != (B, A):
            raise ValueError("remaining_insertions shape mismatch")
        if self.deleted.shape != (B, A):
            raise ValueError("deleted shape mismatch")
        if self.planar_position.shape != (B, A):
            raise ValueError("planar_position shape mismatch")
        if self.res_bfactor.shape != (B, A):
            raise ValueError("res_bfactor shape mismatch")
        if self.res_plddt.shape != (B, A):
            raise ValueError("res_plddt shape mismatch")
        if self.contact_conditioning.shape != (B, A, A):
            raise ValueError("contact_conditioning shape mismatch")
        if D != 3:
            raise ValueError("trans_t last dim must be 3")


@dataclass
class ModelPrediction:
    """t=1 prediction for present state (length P)"""

    pred_trans_1: torch.Tensor  # (B, P, 3) final/anchor positions
    pred_rotmats_1: torch.Tensor  # (B, P, 3, 3) final/anchor rotations
    pred_aatype_logits: torch.Tensor  # (B, P, 21) final/anchor aatype logits
    pred_insertion_logits: (
        torch.Tensor
    )  # (B, P, 21) amino acid logits for inserted children
    pred_split_rate: torch.Tensor  # (B, P) non-negative remaining splits per token
    pred_split_pooled_log1p_rate: (
        torch.Tensor
    )  # (B,) log1p-space total remaining splits
    pred_del_logits: torch.Tensor  # (B, P) deletion logit per token
    pred_bfactor: Optional[torch.Tensor] = None  # (B, P, num_bins) bfactor logits
    pred_plddt: Optional[torch.Tensor] = None  # (B, P, num_bins) pLDDT logits

    def to(self, device: torch.device) -> "ModelPrediction":
        """Move all tensors to specified device"""
        return ModelPrediction(
            pred_trans_1=to_device(self.pred_trans_1, device),
            pred_rotmats_1=to_device(self.pred_rotmats_1, device),
            pred_aatype_logits=to_device(self.pred_aatype_logits, device),
            pred_insertion_logits=to_device(self.pred_insertion_logits, device),
            pred_split_rate=to_device(self.pred_split_rate, device),
            pred_split_pooled_log1p_rate=to_device(
                self.pred_split_pooled_log1p_rate, device
            ),
            pred_del_logits=to_device(self.pred_del_logits, device),
            pred_bfactor=to_device(self.pred_bfactor, device),
            pred_plddt=to_device(self.pred_plddt, device),
        )

    def detach_clone(self, device: Optional[torch.device] = None) -> "ModelPrediction":
        """Detach and clone the prediction, e.g. to save in trajectory.

        Args:
            device: If provided, move tensors to this device (e.g. 'cpu' for trajectories)
        """
        result = ModelPrediction(
            pred_trans_1=clone_detach(self.pred_trans_1),
            pred_rotmats_1=clone_detach(self.pred_rotmats_1),
            pred_aatype_logits=clone_detach(self.pred_aatype_logits),
            pred_insertion_logits=clone_detach(self.pred_insertion_logits),
            pred_split_rate=clone_detach(self.pred_split_rate),
            pred_split_pooled_log1p_rate=clone_detach(
                self.pred_split_pooled_log1p_rate
            ),
            pred_del_logits=clone_detach(self.pred_del_logits),
            pred_bfactor=clone_detach(self.pred_bfactor),
            pred_plddt=clone_detach(self.pred_plddt),
        )
        if device is not None:
            return result.to(device)
        return result


@dataclass
class Trajectory:
    """
    Base trajectory class storing samples at each timestep.
    Samples should be detached and cloned before saving to trajectory.
    """

    samples: List[DataCorrupted] = field(default_factory=list)


@dataclass
class SampleTrajectory(Trajectory):
    """Trajectory from sampling, includes model predictions."""

    pred: List[ModelPrediction] = field(default_factory=list)
