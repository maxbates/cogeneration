"""
Simple implementation of branching flows for protein scaffolding.

Sample some proteins of length N (translations, rotations, and sequence) using LengthBatched protein dataset
Sample a motif_mask, with 1+ scaffolds
Sample some number of roots per scaffold, starting points at t=0 for the motifs + roots
Define a tree, with splits (anchors at intermediate time points) and deletions over trajectory

Sample some intermediate time t
Corrupt to X_t, using stochastic bridge from 0 to each anchor, continuing up to time t

A simple model predicts base (endpoint prediction), split (remaining children count), and deletion (destined to delete probability)

Then, a sampler (no tree) iterates to get base (endpoint prediction), sample split events, sample deletion events

TODOs / features:
- add a README
- break up this file
- ideally we could share loss calculator with cogeneration, e.g. using static methods
- support aatypes guidance potential, support particle-free ESM potential
"""

import copy
import datetime
import functools
import gc
import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.nn.functional as F
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import proj3d
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch import nn
from torch.utils.data import BatchSampler, DataLoader, Dataset
from tqdm import tqdm

from cogeneration.config.base import Config, DataConfig
from cogeneration.data import all_atom, so3_utils
from cogeneration.data.const import (
    ANG_TO_NM_SCALE,
    MASK_TOKEN_INDEX,
    NM_TO_ANG_SCALE,
    NUM_TOKENS,
    rigids_ang_to_nm,
    rigids_nm_to_ang,
)
from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.data.noise_mask import (
    centered_gaussian,
    uniform_categorical,
    uniform_so3,
)
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import (
    restype_order_with_x,
    restypes,
    restypes_with_x,
)
from cogeneration.data.rigid import batch_center_of_mass, create_rigid
from cogeneration.data.rigid_utils import Rigid
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.featurizer import BatchFeaturizer
from cogeneration.dataset.protein_dataloader import LengthBatcher
from cogeneration.models.aa_pred import AminoAcidPredictionNet
from cogeneration.models.attention.attention_trunk import AttentionTrunk
from cogeneration.models.attention.ipa_attention import AttentionIPATrunk
from cogeneration.models.attention.ipa_pytorch import Linear
from cogeneration.models.bfactors import BFactorModule
from cogeneration.models.confidence import PLDDTModule
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.embed import get_index_embedding, get_time_embedding
from cogeneration.models.esm_ckpt_loading import plan_esm_warm_start_state_dict_load
from cogeneration.models.esm_combiner import ESMCombinerNetwork
from cogeneration.models.utils import get_model_size_str
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.embed import PositionalEmbeddingMethod
from cogeneration.type.metrics import MetricName, OutputFileName
from cogeneration.type.task import DataTask, InferenceTask
from cogeneration.util.log import rank_zero_logger
from varco.config import (
    VarcoConfig,
    VarcoDatasetConfig,
    VarcoHazardConfig,
    VarcoHazardKind,
    VarcoInterpolantAATypesCouplerConfig,
    VarcoInterpolantConfig,
    VarcoInterpolantRotationCouplerConfig,
    VarcoInterpolantTransCouplerConfig,
    VarcoLossConfig,
    VarcoModelConfig,
    VarcoMotifGuidanceType,
)

logger = rank_zero_logger("BranchingFlow")

""" Data Flow """

# B = batch size
# N = sampled positions in data (t=1)
# R = root positions in base (t=0)
# M = motif positions (constant t=0 -> t=1)
# A = aligned number of positions in constructed tree (with anchors + to-be-deleted)
# A_max = batch max length of A positions
# P = present positions at time t (M + R <= P <= A)
# P_max = batch max length of padded P positions

# Data flow (training):
# TreePlan - t=0 -> t=1 per-sample tree topology (length A) with birth/split/delete times
# BatchedTreePlan - collated TreePlans with (B, A_max) tensors
# DataSample - t=1 data. per-sample (N,) domains at t=1, with predefined TreePlan
# DataBatch - t=1 collated DataSample. (B, N) domains with BatchedTreePlan [use LengthBatcher, t=1 all same length]
# DataBridged - time t, corrupted (B, A) domains + topology in tree-aligned space
# DataCorrupted - time t, packed (B, P_max) present positions for model input (i.e. tree node subset present at t)
# ModelPrediction - model outputs (B, P_max) for loss calculation
# ModelPrediction + DataBridged + Coupling (for domain anchors + P <-> A mapping) -> LossCalculator -> Losses
#
# Data flow (sampling):
# DataCorrupted - packed (B, P_max) state, mutated in-place with insert/delete
# SampleTrajectory - list of DataCorrupted snapshots and ModelPrediction per step


""" Tensor Utilities """


class SeededRNG:
    """Seeded random number generator wrapping torch.Generator for reproducibility."""

    def __init__(self, seed: Optional[int] = None, device: str = "cpu"):
        self.rng = torch.Generator(device=device)
        if seed is None:
            seed = int(torch.seed() % (2**31 - 1))
        self.rng.manual_seed(int(seed))

    def rand_int(self, high: int) -> int:
        """Sample uniform integer from [0, high)."""
        return int(torch.randint(0, high, (1,), generator=self.rng).item())

    def rand_float(self) -> float:
        """Sample uniform float from [0, 1)."""
        return float(torch.rand(1, generator=self.rng).squeeze().item())

    def rand(self, size: int, device: str = "cpu") -> torch.Tensor:
        """Sample uniform floats from [0, 1) as a tensor"""
        return torch.rand(size, generator=self.rng, device=device)

    def sample_exp1(self) -> float:
        """Sample from Exp(1) distribution via inverse CDF."""
        u = float(torch.rand((), generator=self.rng).clamp_min(1e-12).item())
        return -math.log(u)

    def sample_poisson(self, lam: float) -> int:
        """Sample from Poisson(lam) distribution using Knuth's algorithm."""
        if lam <= 0.0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rand_float()
            if p <= L:
                return k - 1

    def sample_beta(
        self, size: int, alpha: float = 1.0, beta: float = 1.0, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Sample from Beta(alpha, beta) distribution.

        Uses the ratio of gamma variates method: if X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1),
        then X / (X + Y) ~ Beta(alpha, beta).

        For alpha < beta, the distribution is biased toward 0 (earlier times).
        For alpha > beta, the distribution is biased toward 1 (later times).
        For alpha = beta = 1, this is uniform on [0, 1].

        Args:
            size: Number of samples to generate
            alpha: First shape parameter (controls left tail)
            beta: Second shape parameter (controls right tail)
            device: Device for output tensor

        Returns:
            Tensor of shape (size,) with samples in [0, 1]
        """
        if size == 0:
            return torch.empty(0, device=device)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")

        # Sample from Gamma distributions using the Marsaglia and Tsang method
        def sample_gamma(a: float, n: int) -> torch.Tensor:
            if a < 1:
                # For a < 1, use: Gamma(a) = Gamma(a+1) * U^(1/a)
                g = sample_gamma(a + 1, n)
                u = torch.rand(n, generator=self.rng, device="cpu")
                return g * (u ** (1.0 / a))

            d = a - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)

            samples = []
            while len(samples) < n:
                batch_size = max(n - len(samples), 64)
                z = torch.randn(batch_size, generator=self.rng, device="cpu")
                u = torch.rand(batch_size, generator=self.rng, device="cpu")

                v = (1.0 + c * z) ** 3
                valid = (z > -1.0 / c) & (
                    torch.log(u) < 0.5 * z**2 + d * (1.0 - v + torch.log(v))
                )
                samples.extend((d * v)[valid].tolist())

            return torch.tensor(samples[:n], device="cpu")

        x = sample_gamma(alpha, size)
        y = sample_gamma(beta, size)
        return (x / (x + y)).to(device=device)


def clone_detach(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Clone and detach an Optional tensor."""
    if x is None:
        return None
    return x.detach().clone()


def to_device(
    x: Optional[torch.Tensor], device: torch.device
) -> Optional[torch.Tensor]:
    """Move an Optional tensor to a device."""
    if x is None:
        return None
    return x.to(device=device)


def gather_and_pad(
    source: Optional[torch.Tensor],  # (B, N, ...) or (B, N, N) if is_2d
    index: torch.Tensor,  # (B, P)
    mask: torch.Tensor,  # (B, P)
    fill_value: Union[float, torch.Tensor] = 0.0,
    is_2d: bool = False,
) -> Optional[torch.Tensor]:  # (B, P, ...) or (B, P, P) if is_2d
    """
    Gather from source along dim=1 using index, then fill padding positions with fill_value.

    Handles arbitrary trailing dimensions by expanding index and mask.
    For positions where mask is False, the result is set to fill_value.

    Args:
        source: (B, N, ...) tensor to gather from, or (B, N, N) if is_2d=True
        index: (B, P) indices into dim=1 of source (must be in [0, N-1])
        mask: (B, P) boolean mask; True for valid positions, False for padding
        fill_value: value to fill where mask is False. Can be a scalar float or
                    a tensor with shape matching source trailing dimensions (...).
        is_2d: if True, source is (B, N, N) and we gather along both dim=1 and dim=2
               to produce (B, P, P). Used for contact_conditioning matrices.

    Returns:
        (B, P, ...) tensor with gathered values where mask is True, fill_value otherwise
        If is_2d=True, returns (B, P, P) tensor.
    """
    if source is None:
        return None

    B, P = index.shape

    if is_2d:
        # source is (B, N, N), gather along both dimensions to get (B, P, P)
        # First gather rows: (B, N, N) -> (B, P, N)
        idx_row = index.unsqueeze(-1).expand(-1, -1, source.shape[2])  # (B, P, N)
        gathered_rows = source.gather(1, idx_row)  # (B, P, N)
        # Then gather columns: (B, P, N) -> (B, P, P)
        idx_col = index.unsqueeze(1).expand(-1, P, -1)  # (B, P, P)
        gathered = gathered_rows.gather(2, idx_col)  # (B, P, P)

        # Fill value for 2D case
        if isinstance(fill_value, torch.Tensor):
            fill = fill_value.unsqueeze(0).unsqueeze(0).expand(B, P, P)
            fill = fill.to(device=gathered.device, dtype=gathered.dtype)
        else:
            fill = torch.full_like(gathered, fill_value)

        # 2D mask: both row and column must be valid
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, P, P)
        return torch.where(mask_2d, gathered, fill)

    # Standard 1D case
    trailing_shape = source.shape[2:]
    idx = index
    for _ in trailing_shape:
        idx = idx.unsqueeze(-1)
    idx = idx.expand(-1, -1, *trailing_shape)  # (B, P, ...)

    gathered = source.gather(1, idx)  # (B, P, ...)

    # Handle fill value - either scalar or tensor
    if isinstance(fill_value, torch.Tensor):
        # Tensor fill value: broadcast to (B, P, ...)
        fill = fill_value.unsqueeze(0).unsqueeze(0).expand(B, P, *trailing_shape)
        fill = fill.to(device=gathered.device, dtype=gathered.dtype)
    else:
        # Scalar fill value
        fill = torch.full_like(gathered, fill_value)

    if not trailing_shape:
        return torch.where(mask, gathered, fill)
    else:
        m = mask
        for _ in trailing_shape:
            m = m.unsqueeze(-1)
        m = m.expand_as(gathered)
        return torch.where(m, gathered, fill)


def pad_and_stack(
    tensors: List[torch.Tensor],  # (B, P, ...)
    max_len: int,
    fill_value: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:  # (B, P_max, ...)
    """
    Pad a list of 1D or 2D tensors to max_len and stack into a batch.

    Args:
        tensors: list of (P, ...) tensors with varying P
        max_len: target length to pad to
        fill_value: value to fill padding positions
        dtype: output dtype (defaults to first tensor's dtype)

    Returns:
        (B, P_max, ...) tensor
    """
    B = len(tensors)
    if B == 0:
        raise ValueError("Empty tensor list")

    first = tensors[0]
    dtype = dtype or first.dtype
    trailing_shape = first.shape[1:]  # () for 1D, (D,) for 2D

    out = torch.full((B, max_len, *trailing_shape), fill_value, dtype=dtype)
    for b, t in enumerate(tensors):
        L = t.shape[0]
        out[b, :L] = t.to(dtype)
    return out


""" Tree Plan """


@dataclass
class TreePlan:
    """Per-sample (non-batched) tree topology and sampled times (domain-agnostic)."""

    num_leaves: int  # N_data + N_deleted (leaf nodes only)
    num_deletions: int  # number of leaves destined to be deleted
    num_nodes: int  # A (leaves + internal)

    # topology
    topo_order: torch.Tensor  # (A,) long, structural topo (parent-before-child)
    motif_mask: torch.Tensor  # (A,) bool; True only for motif leaves (not anchors)
    roots: torch.Tensor  # (R,) long, root node ids
    parent_idx: torch.Tensor  # (A,) long, -1 for roots
    children_idx: torch.Tensor  # (A, 2) long, -1 for leaves
    total_leaves: torch.Tensor  # (A,) long, descendant leaf counts
    node_depth: torch.Tensor  # (A,) long, depth in tree (leaves=0, parents -> max)
    leaf_deleted: torch.Tensor  # (A,) bool; True for deleted leaves, False otherwise
    planar_position: torch.Tensor  # (A,) float, pre-computed node position/order

    # times
    birth_time: torch.Tensor  # (A,) float, segment start time; roots are 0
    split_time: torch.Tensor  # (A,) float, segment end time; leaves are +inf
    delete_time: torch.Tensor  # (A,) float, deletion time; non-deleted are +inf

    # tree -> data mapping
    # use BatchedTreePlan.broadcast_to_leaves() to broadcast t=1 data into aligned space
    leaf_map: torch.Tensor  # (A,) long; A -> N (tree leaf -> data) mapping

    @classmethod
    def generate(
        cls,
        motif_mask: torch.Tensor,  # (N,)
        seed: Optional[int] = None,
        min_t: float = 0.005,
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
        # Coalesce within each group by repeated adjacent merges.
        # For scaffold span groups (gid < span_gid), bias merges toward the boundaries so the
        # coalescent "collapses" inwards from both ends (both boundaries can in-fill).
        p_interior: float = 0.98,
        # Number of deletion leaves are sampled from a Poisson distribution with rate p_deletion * # scaffold positions.
        p_deletion: float = 0.20,
        # Beta distribution parameters for biasing split/insertion times.
        # With alpha < beta, splits are biased to occur earlier (closer to t=0).
        # Default (1.0, 2.0) gives mean ~0.33, moving mass toward earlier times.
        # Use (1.0, 1.0) for uniform sampling (no bias).
        split_time_beta: tuple[float, float] = (1.0, 2.0),
        # Beta distribution parameters for biasing deletion times.
        # With alpha > beta, deletions are biased to occur later (closer to t=1).
        # Default (2.0, 1.0) gives mean ~0.67, moving mass toward later times.
        delete_time_beta: tuple[float, float] = (2.0, 1.0),
    ) -> "TreePlan":
        """
        Generate a simple planar coalescent tree plan with sampled split/birth/delete times.

        Algorithm:
        - Provide t=1 positions of motif tokens (motif_mask == True) and scaffold positions
        - Augment the tree with K_del deletions, which are to-be-deleted leaves duplicating t=1 endpoints
        - Determine scaffold groups, assign group id
        - Sample K_scaffold per scaffold span; if K_scaffold > span_len, create extra deleted roots
        - Coalesce scaffold groups independently until K_scaffold roots remain within the group
        - Sample split times for internal nodes (aka "anchors")
        - Sample deletion times for deletion leaves (must come after split)
        - Define motif_mask etc. in aligned space (i.e. length A)
        - Compute planar_position, bookkeeping

        Note this scheme intentionally allows scaffold tokens to be present at t=0 (the roots), in
        addition to motif tokens, which avoids needing special boundary-sourcing logic.

        Nodes are either leaves or internal.
        Internal nodes are anchors, they always have children and a split time and are not themselves deleted.
        Leaf types:
        - Original leaves: leaf_map >= 0, leaf_deleted = False
        - Deleted duplicate leaves: leaf_map >= 0, leaf_deleted = True
        - Extra deleted roots: leaf_map = -1, leaf_deleted = True (no t=1 data reference)

        Beyond tracking topology, the phases of tree construction yields a node ordering:
        - Leaves (IDs 0..num_leaves-1)
          - original data
          - deletions (i.e. duplicates of original data)
          - extra deleted roots (scaffold roots to be deleted)
        - Internal nodes (IDs num_leaves..A-1)
          - anchors from coalescence

        However, when the tree (A) is packed (P), we use planar_position to order the nodes.
        """
        if motif_mask.ndim != 1:
            raise ValueError(
                f"Expected motif_mask to have shape (N,) got {tuple(motif_mask.shape)}"
            )
        if min_scaffold_nuclei < 1:
            raise ValueError("min_scaffold_nuclei must be >= 1")
        if max_scaffold_nuclei < 1:
            raise ValueError("max_scaffold_nuclei must be >= 1")
        if min_scaffold_nuclei > max_scaffold_nuclei:
            raise ValueError("min_scaffold_nuclei must be <= max_scaffold_nuclei")

        device = motif_mask.device
        N_data = motif_mask.shape[0]
        motif_mask_b = motif_mask.to(torch.bool)

        scaffold_idx = torch.where(~motif_mask_b)[0]
        n_scaffold = int(scaffold_idx.numel())

        # RNG helper
        rng = SeededRNG(seed=seed, device="cpu")

        # --- Deletions
        # Augment t=1 leaf sequence by inserting extra leaves that duplicate an existing scaffold leaf,
        # and mark the node to-be-deleted.

        lam = p_deletion * float(n_scaffold)
        k_del = rng.sample_poisson(lam)
        k_del = int(min(k_del, n_scaffold))  # cap to avoid blowups

        # dup_count[i] = how many deletion-duplicates to insert adjacent to original data index i
        if k_del > 0 and n_scaffold > 0:
            # Sample endpoints to dup WITH replacement on CPU (SeededRNG uses a CPU Generator)
            scaffold_idx_cpu = scaffold_idx.to("cpu")
            sample_j = torch.randint(
                low=0,
                high=n_scaffold,
                size=(k_del,),
                generator=rng.rng,
                device="cpu",
            )
            sampled_data_idx = scaffold_idx_cpu[sample_j]  # (k_del,)
            dup_count = torch.bincount(sampled_data_idx, minlength=N_data).to(
                torch.long
            )
        else:
            dup_count = torch.zeros((N_data,), dtype=torch.long, device="cpu")

        # Build augmented leaf order (includes deletion duplicates)
        # - build an augmented leaf_ref by repeating each original position 1 + dup_count times
        # - mark the inserted repeats (beyond the first) as deleted
        repeats = (1 + dup_count).to(torch.long)  # (N_data,)
        leaf_ref_t_cpu = torch.repeat_interleave(
            torch.arange(N_data, dtype=torch.long, device="cpu"), repeats
        )  # (num_leaves,)
        num_leaves = int(leaf_ref_t_cpu.numel())

        # Mark duplicates (all occurrences except the first in each block) as deleted.
        # For motif positions, repeats==1 so they never get duplicates.
        start = torch.cumsum(repeats, dim=0) - repeats  # (N_data,)
        dup_mask = torch.ones((num_leaves,), dtype=torch.bool, device="cpu")
        dup_mask[start] = False  # first occurrence is the "real" leaf
        leaf_del_t = dup_mask  # duplicates are the deletion-leaves
        num_deletions = int(leaf_del_t.sum().item())

        # Break planar_position ties for deletion-duplicate leaves so they don't deterministically
        # sort after the original when packed via planar_position (stable argsort).
        # Keep jitter small enough to avoid reordering across neighboring residues.
        leaf_pos_jitter_cpu = torch.zeros(
            (num_leaves,), dtype=torch.float32, device="cpu"
        )
        if num_deletions > 0:
            u = torch.rand(
                (num_deletions,), generator=rng.rng, device="cpu", dtype=torch.float32
            )
            leaf_pos_jitter_cpu[dup_mask] = (u - 0.5) * 0.4  # [-0.2, 0.2]

        # Move leaf-level bookkeeping to the target device
        leaf_map_leaves = leaf_ref_t_cpu.to(device=device)  # (num_leaves,)
        leaf_del_t = leaf_del_t.to(device=device)  # (num_leaves,)
        leaf_pos_jitter = leaf_pos_jitter_cpu.to(device=device)  # (num_leaves,)

        # --- Group ID assignment
        # Group ids computed in *leaf order* (includes deletion duplicates), fully vectorized.
        # Scaffold spans (motif_mask == False) are grouped into disjoint contiguous groups.
        # Each motif position gets its own singleton group id after scaffold-span ids.

        leaf_is_motif = motif_mask_b[leaf_map_leaves]  # (num_leaves,)
        is_scaffold = ~leaf_is_motif

        # scaffold_start marks the first position of each scaffold span
        scaffold_start = is_scaffold & torch.cat(
            [torch.tensor([True], device=device), ~is_scaffold[:-1]]
        )
        span_id = (
            torch.cumsum(scaffold_start.long(), dim=0) - 1
        )  # (-1 for non-scaffold)
        span_id = torch.where(
            is_scaffold,
            span_id,
            torch.full_like(span_id, -1),
        )
        span_gid = int(scaffold_start.sum().item())
        motif_index = torch.cumsum(leaf_is_motif.long(), dim=0) - 1
        group_ids_leaf = torch.where(
            is_scaffold,
            span_id,
            span_gid + motif_index,
        )
        if bool((group_ids_leaf < 0).any().item()):
            raise RuntimeError("Failed to assign group_ids_leaf for all leaf positions")

        # Active lists per group in sequence order
        groups: Dict[int, List[int]] = {}
        # Loop over groups is small (#scaffold spans + #motifs), so keep it simple.
        for gid in torch.unique(group_ids_leaf).tolist():
            idxs = torch.nonzero(group_ids_leaf == int(gid), as_tuple=False).view(-1)
            groups[int(gid)] = idxs.tolist()

        # --- Sample scaffold roots and create extra deleted roots
        # Sample K_scaffold per scaffold span. If K_scaffold > span_len, create extra deleted
        # root leaves now (before coalescence) to maintain contiguous leaf IDs.
        parent: List[int] = [-1] * num_leaves
        children: List[List[int]] = [[-1, -1] for _ in range(num_leaves)]
        weight: List[int] = [1 for _ in range(num_leaves)]

        extra_deleted_roots: List[int] = []
        extra_root_spans: List[Tuple[int, int]] = (
            []
        )  # (span_start, span_end) for planar positioning
        scaffold_K: Dict[int, int] = {}  # gid -> K_scaffold

        for gid, active in groups.items():
            is_scaffold_span = gid < span_gid

            if is_scaffold_span:
                span_len = len(active)
                span_start_leaf = min(active)
                span_end_leaf = max(active)
                k_hi = max_scaffold_nuclei
                k_lo = min(min_scaffold_nuclei, k_hi)
                K_scaffold = (
                    k_lo if k_lo == k_hi else (k_lo + rng.rand_int(k_hi - k_lo + 1))
                )
                scaffold_K[gid] = K_scaffold

                # Create extra deleted roots if K_scaffold > span_len
                if K_scaffold > span_len:
                    num_extra = K_scaffold - span_len
                    for _ in range(num_extra):
                        new_id = len(parent)
                        parent.append(-1)  # root
                        children.append([-1, -1])  # leaf
                        weight.append(1)
                        extra_deleted_roots.append(new_id)
                        extra_root_spans.append((span_start_leaf, span_end_leaf))
                        # Insert into active list at random position for planar ordering
                        insert_pos = rng.rand_int(len(active) + 1)
                        active.insert(insert_pos, new_id)
            else:
                scaffold_K[gid] = 1

        # Track original num_leaves (before extra roots) for leaf_map/leaf_deleted assignment
        num_leaves_original = num_leaves
        # Update num_leaves to include extra deleted roots
        num_leaves = len(parent)  # all nodes so far are leaves

        # --- Coalescence
        # Iteratively merge adjacent nodes, tracking parent-child relationships and weights (num children)
        # until we are left with K_scaffold roots for each scaffold span group.
        roots: List[int] = []
        for gid, active in groups.items():
            is_scaffold_span = gid < span_gid
            K_scaffold = scaffold_K[gid]

            while len(active) > K_scaffold:
                # Choose which adjacent pair to merge.
                if (
                    is_scaffold_span
                    and len(active) > 2
                    and (rng.rand_float() < (1.0 - p_interior))
                ):
                    # Boundary-biased: merge leftmost or rightmost adjacent pair (50/50)
                    if rng.rand_float() < 0.5:
                        i0 = 0
                    else:
                        i0 = len(active) - 2
                else:
                    # Uniform random adjacent pair
                    i0 = rng.rand_int(len(active) - 1)

                left = active[i0]
                right = active[i0 + 1]

                new_id = len(parent)
                parent.append(-1)
                children.append([left, right])

                weight_new = weight[left] + weight[right]
                weight.append(weight_new)

                parent[left] = new_id
                parent[right] = new_id

                # Replace pair with new internal node, preserving order
                active[i0 : i0 + 2] = [new_id]

            # Whatever remains are the roots/nuclei for this group.
            roots.extend(active)

        # Roots now track scaffold nuclei. Not extra deleted roots.
        roots = sorted(set(roots))

        # Tree now includes all nodes:
        # A = motifs, scaffold leaves, splits/anchors, and deletions
        A = len(parent)

        # Structural topo falls out of construction,
        # internal nodes are appended so parent ids > child ids.
        topo_order = torch.arange(A - 1, -1, -1, dtype=torch.long)

        # Compute node depth early for time sampling + traversals
        # Process children/leaves (depth 0+) before parents (highest depth)
        node_depth_list = [0] * A
        for node in range(A):
            # only internal nodes have depth > 0
            if weight[node] > 1:
                c0, c1 = children[node]
                if c0 >= 0 and c1 >= 0:
                    node_depth_list[node] = (
                        max(node_depth_list[c0], node_depth_list[c1]) + 1
                    )
                elif c0 >= 0:
                    node_depth_list[node] = node_depth_list[c0] + 1
                elif c1 >= 0:
                    node_depth_list[node] = node_depth_list[c1] + 1

        # leaf_map mapping: valid for original leaves, -1 for internal nodes and extra deleted roots
        leaf_map = torch.full((A,), -1, dtype=torch.long, device=device)
        leaf_map[:num_leaves_original] = leaf_map_leaves

        leaf_deleted = torch.zeros((A,), dtype=torch.bool, device=device)
        if num_leaves_original > 0:
            leaf_deleted[:num_leaves_original] = leaf_del_t

        # Mark extra deleted roots (have no t=1 data reference, leaf_map stays -1)
        if extra_deleted_roots:
            extra_t = torch.tensor(extra_deleted_roots, dtype=torch.long, device=device)
            leaf_deleted[extra_t] = True

        # --- Sample birth/split times

        # setup split/deletion time tensors
        birth_time = torch.full((A,), float("inf"), dtype=torch.float32)
        split_time = torch.full((A,), float("inf"), dtype=torch.float32)
        delete_time = torch.full((A,), float("inf"), dtype=torch.float32)

        # Roots start at time 0
        roots_t = torch.tensor(roots, dtype=torch.long, device=device)
        birth_time[roots_t] = 0.0

        # Depth-based time sampling (vectorized, across depths)
        # Process parents before children (depth max_depth down to 0)
        node_depth_t = torch.tensor(node_depth_list, dtype=torch.long, device=device)
        weight_t = torch.tensor(weight, dtype=torch.long, device=device)
        children_t = torch.tensor(children, dtype=torch.long, device=device)
        max_depth = max(node_depth_list)

        for depth in range(max_depth, -1, -1):
            # Find all nodes at this depth
            nodes_at_depth = torch.where(node_depth_t == depth)[0]
            if len(nodes_at_depth) == 0:
                continue

            # Get birth times and weights for nodes at this depth
            t0_vec = birth_time[nodes_at_depth]
            W_vec = weight_t[nodes_at_depth]

            # Filter to internal nodes with finite birth times
            is_internal = W_vec > 1
            has_finite_birth = torch.isfinite(t0_vec)
            valid = is_internal & has_finite_birth

            if valid.any():
                valid_nodes = nodes_at_depth[valid]
                t0_valid = t0_vec[valid]
                W_valid = W_vec[valid]

                # Exponential sampling for all valid nodes at this depth
                # st = 1 - (1 - t0) * exp(-E / (W - 1)), where E ~ Exp(1)
                # Use beta distribution to bias the base uniform toward earlier times
                alpha, beta = split_time_beta
                u = rng.sample_beta(len(valid_nodes), alpha, beta, device=device)
                E = -torch.log(u.clamp_min(1e-12))
                st = 1.0 - (1.0 - t0_valid) * torch.exp(
                    -E / (W_valid - 1).clamp_min(1.0)
                )
                st = torch.clamp(st, min=min_t, max=1.0 - 2 * min_t)

                # Write split times
                split_time[valid_nodes] = st

                # Propagate birth times to children
                children_pairs = children_t[valid_nodes]  # (N_valid, 2)
                for j in range(2):
                    c_indices = children_pairs[:, j]
                    valid_children = c_indices >= 0
                    if valid_children.any():
                        birth_time[c_indices[valid_children]] = st[valid_children]

            # Set split_time to inf for leaves at this depth
            leaf_mask = W_vec == 1
            if leaf_mask.any():
                split_time[nodes_at_depth[leaf_mask]] = float("inf")

        # --- Sample delete times
        # A deleted leaf has an unconditional deletion time distributed as Uniform(0, 1),
        # conditioned on being AFTER the leaf's birth time (i.e. dt | dt > birth).
        #
        # For Uniform(0, 1), the truncated-sampling is simply:
        #   dt = birth + (1 - birth) * u,  u ~ Uniform(0, 1)
        #
        # We enforce strict inequalities with tiny epsilons to avoid dt == birth or dt == 1
        # due to floating point edge cases.
        min_delete_eps = min_t
        max_delete_time = 1.0 - min_t

        # Set deletion times for nodes:
        # - Marked as deleted (leaf_deleted)
        # - Are leaf nodes (weight_t == 1)
        # - Are valid nodes in sample i.e. have finite birth time
        is_leaf = weight_t == 1
        valid_deleted = leaf_deleted & is_leaf & torch.isfinite(birth_time)

        num_deleted = valid_deleted.sum().item()
        if num_deleted > 0:
            b = birth_time[valid_deleted]

            # Sample beta-distributed random values for all deleted leaves at once
            # With alpha > beta, deletions are biased toward later times
            alpha, beta = delete_time_beta
            u = rng.sample_beta(int(num_deleted), alpha, beta, device=device)

            # Compute deletion times: dt = birth + (1 - birth) * u
            dt = b + (1.0 - b) * u

            # Clamp to valid range [birth + min_delete_eps, max_delete_time]
            # For births very close to 1, this ensures we have a valid deletion time
            dt = torch.clamp_min(dt, min=b + min_delete_eps)
            dt = torch.clamp_max(dt, max=max_delete_time)

            delete_time[valid_deleted] = dt

        # --- Compute planar_position
        # Original leaves (non-duplicates) get sequential positions 0, 1, 2, ..., N_data-1
        # Duplicate leaves get the same position as their original (same leaf_map value)
        # Extra deleted roots get random position within their scaffold span
        # Internal nodes get weighted average of children (computed bottom-up)
        planar_pos = torch.zeros((A,), dtype=torch.float32, device=device)

        # Build a mapping from data index to planar position (0, 1, ..., N_data-1)
        data_idx_to_pos = torch.arange(N_data, dtype=torch.float32, device=device)

        # Assign positions to original leaves (0..num_leaves_original-1) based on their leaf_map
        # This handles both originals and duplicates correctly - duplicates share position
        for i in range(num_leaves_original):
            data_idx = int(leaf_map_leaves[i].item())
            planar_pos[i] = data_idx_to_pos[data_idx] + leaf_pos_jitter[i]

        # Extra deleted roots: sample random position within their span's leaf range
        # span_start and span_end are node IDs of the first/last leaves in the span,
        # which for simple cases equal their planar positions
        for extra_id, (span_start, span_end) in zip(
            extra_deleted_roots, extra_root_spans
        ):
            # Use the planar positions of the span boundaries, not node IDs
            pos_start = planar_pos[span_start].item()
            pos_end = planar_pos[span_end].item()
            # Sample uniformly in [pos_start, pos_end + 1)
            planar_pos[extra_id] = pos_start + rng.rand_float() * (
                pos_end - pos_start + 1
            )

        # Internal nodes: bottom-up traversal (children before parents)
        # topo_order is parent-before-child, so reverse it
        # Use the Python lists (children, weight) since tensors aren't created yet
        for node in topo_order.flip(0).tolist():
            c0, c1 = children[node][0], children[node][1]
            if c0 >= 0 and c1 >= 0:  # internal node
                w0, w1 = float(weight[c0]), float(weight[c1])
                denom = max(1e-6, w0 + w1)
                planar_pos[node] = (planar_pos[c0] * w0 + planar_pos[c1] * w1) / denom

        # --- Final bookkeeping

        # Build motif mask in aligned tree space (A,)
        # Leaves 0..num_leaves-1 inherit motif/scaffold status from their referenced data index.
        # Internal nodes and extra deleted roots are always False (scaffold-derived).
        motif_mask_aligned = torch.zeros((A,), dtype=torch.bool, device=device)
        motif_mask_aligned[:num_leaves_original] = motif_mask_b[leaf_map_leaves]

        parent_idx = torch.tensor(parent, dtype=torch.long, device=device)
        children_idx = torch.tensor(children, dtype=torch.long, device=device)
        total_leaves = torch.tensor(weight, dtype=torch.long, device=device)
        roots_t = torch.tensor(roots, dtype=torch.long, device=device)

        leaf_map = leaf_map.to(device=device)
        birth_time = birth_time.to(device=device)
        split_time = split_time.to(device=device)
        delete_time = delete_time.to(device=device)
        topo_order = topo_order.to(device=device)
        node_depth = node_depth_t.to(device=device)

        # Count total leaf nodes (including extra deleted roots) and deletions
        total_leaf_nodes = int((total_leaves == 1).sum().item())
        total_deletions = int(leaf_deleted.sum().item())

        return cls(
            num_leaves=total_leaf_nodes,
            num_deletions=total_deletions,
            num_nodes=A,
            topo_order=topo_order,
            motif_mask=motif_mask_aligned,
            parent_idx=parent_idx,
            roots=roots_t,
            children_idx=children_idx,
            total_leaves=total_leaves,
            node_depth=node_depth,
            leaf_deleted=leaf_deleted,
            planar_position=planar_pos,
            birth_time=birth_time,
            split_time=split_time,
            delete_time=delete_time,
            leaf_map=leaf_map,
        )

    def validate(self) -> None:
        A = int(self.num_nodes)
        N_data = int(self.num_leaves - self.num_deletions)
        N_leaf = int(self.num_leaves)

        # shape checks
        if self.parent_idx.shape != (A,):
            raise ValueError("parent_idx shape mismatch")
        if self.children_idx.shape != (A, 2):
            raise ValueError("children_idx shape mismatch")
        if self.total_leaves.shape != (A,):
            raise ValueError("total_leaves shape mismatch")
        if self.birth_time.shape != (A,):
            raise ValueError("birth_time shape mismatch")
        if self.split_time.shape != (A,):
            raise ValueError("split_time shape mismatch")
        if self.delete_time.shape != (A,):
            raise ValueError("delete_time shape mismatch")
        if self.topo_order.shape != (A,):
            raise ValueError("topo_order shape mismatch")
        if self.leaf_map.shape != (A,):
            raise ValueError("leaf_map shape mismatch")
        if self.leaf_deleted.shape != (A,):
            raise ValueError("leaf_deleted shape mismatch")
        if self.motif_mask.shape != (A,):
            raise ValueError(f"motif_mask shape mismatch")

        # Internal nodes (total_leaves > 1) must never be marked as motif.
        if bool(((self.total_leaves > 1) & self.motif_mask).any().item()):
            bad = (
                torch.nonzero(
                    (self.total_leaves > 1) & self.motif_mask, as_tuple=False
                )[:16]
                .view(-1)
                .tolist()
            )
            raise ValueError(
                f"motif_mask must be False for internal nodes; bad node ids (first 16): {bad}"
            )

        roots = self.roots.to(torch.long)
        if roots.numel() == 0:
            raise ValueError("No roots")
        if int(roots.min()) < 0 or int(roots.max()) >= A:
            raise ValueError("Root id out of range")
        for r in roots.tolist():
            if abs(float(self.birth_time[r]) - 0.0) > 1e-6:
                raise ValueError(f"Root {r} birth_time must be 0")

        # leaves/internal split_time
        for i in range(A):
            W = int(self.total_leaves[i])
            if W <= 1:
                if torch.isfinite(self.split_time[i]):
                    raise ValueError(f"Leaf {i} must have split_time=inf")
            else:
                if not torch.isfinite(self.split_time[i]):
                    raise ValueError(f"Internal {i} must have finite split_time")

            # deletion invariants
            if int(self.total_leaves[i]) == 0:
                continue
            if bool(self.leaf_deleted[i].item()):
                if not torch.isfinite(self.delete_time[i]):
                    raise ValueError(f"Deleted node {i} must have finite delete_time")
                if float(self.delete_time[i]) <= float(self.birth_time[i]):
                    raise ValueError(f"Deleted node {i} must delete after birth")
                if float(self.delete_time[i]) >= 1.0:
                    raise ValueError(f"Deleted node {i} must delete before t=1")
            else:
                if torch.isfinite(self.delete_time[i]):
                    raise ValueError(f"Non-deleted node {i} must have delete_time=inf")

        # time consistency: birth(child) == split(parent)
        for c in range(A):
            p = int(self.parent_idx[c])
            if p < 0:
                continue
            if not torch.isfinite(self.split_time[p]):
                raise ValueError(f"Parent {p} of {c} has non-finite split_time")
            if abs(float(self.birth_time[c]) - float(self.split_time[p])) > 1e-5:
                raise ValueError(f"birth_time[{c}] must equal split_time[{p}]")

        # total_leaves consistency
        for i in range(A):
            W = int(self.total_leaves[i])
            if W <= 1:
                continue
            c0, c1 = int(self.children_idx[i, 0]), int(self.children_idx[i, 1])
            if c0 < 0 or c1 < 0:
                raise ValueError(f"Internal node {i} must have two children")
            if W != int(self.total_leaves[c0]) + int(self.total_leaves[c1]):
                raise ValueError(f"total_leaves[{i}] must equal children totals")

        # reachability from roots
        seen = set(roots.tolist())
        stack = list(roots.tolist())
        while stack:
            u = stack.pop()
            c0, c1 = int(self.children_idx[u, 0]), int(self.children_idx[u, 1])
            for v in (c0, c1):
                if v >= 0 and v not in seen:
                    seen.add(v)
                    stack.append(v)
        if len(seen) != A:
            missing = sorted(set(range(A)) - seen)
            raise ValueError(f"Unreachable nodes: {missing[:16]}")

        # leaf endpoint mapping validity
        for i in range(A):
            if int(self.total_leaves[i]) != 1:
                continue  # skip internal nodes
            idx = int(self.leaf_map[i].item())
            if idx == -1:
                # Deleted root with no data reference - must be marked as deleted
                if not bool(self.leaf_deleted[i]):
                    raise ValueError(
                        f"Node {i} with leaf_map=-1 must be marked as deleted"
                    )
                continue
            if idx < 0 or idx >= N_data:
                raise ValueError(f"leaf_map[{i}] out of range: {idx} (N_data={N_data})")

    def plot(self, path: Optional[str] = None, dpi: int = 200) -> str:
        """Plot the tree plan as time (y) vs planar x-position (leaf index space).

        Conventions:
        - y axis is time, with 0 at the top and 1 at the bottom.
        - each node is placed at a planar x-position based on descendant leaves, so internal anchors
          appear within their scaffold span.
        - a node's "life" is a vertical segment from birth_time to end_time, where
          end_time = split_time if finite else 1.0.
        - parent/child relationships are shown as horizontal connectors at the child's birth time (which equals the parent's split time).
        """
        # Resolve output path
        if path is None:
            fd, path = tempfile.mkstemp(prefix="treeplan_", suffix=".png")
            os.close(fd)
        else:
            out_dir = os.path.dirname(path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)

        # Move tensors to CPU for plotting
        parent = self.parent_idx.detach().to("cpu").numpy()
        children = self.children_idx.detach().to("cpu").numpy()
        birth = self.birth_time.detach().to("cpu").numpy()
        split = self.split_time.detach().to("cpu").numpy()
        delete = self.delete_time.detach().to("cpu").numpy()
        total = self.total_leaves.detach().to("cpu").numpy()
        roots = set(self.roots.detach().to("cpu").numpy().tolist())
        motif_mask = self.motif_mask.detach().to("cpu").numpy().astype(bool)
        leaf_map = self.leaf_map.detach().to("cpu").numpy()
        leaf_deleted = self.leaf_deleted.detach().to("cpu").numpy().astype(bool)
        planar_pos = self.planar_position.detach().to("cpu").numpy()

        A = int(self.num_nodes)
        N_leaf = int(self.num_leaves)

        motif_line_color = (
            "0.6"  # grey: straight lifelines for motif singleton residues
        )
        motif_marker_color = "black"
        scaffold_color = "blue"

        fig = plt.figure(figsize=(max(6.0, A / 8.0), 6.0), dpi=dpi)
        ax = fig.add_subplot(111)

        # Draw node lifelines (vertical segments)
        for i in range(A):
            y0 = float(birth[i])
            end = float(np.minimum(split[i], delete[i]))
            y1 = end if np.isfinite(end) else 1.0
            # Skip padded/uninitialized nodes (shouldn't exist in per-sample plan)
            if not np.isfinite(y0):
                continue
            y0 = max(0.0, min(1.0, y0))
            y1 = max(0.0, min(1.0, y1))
            xi = float(planar_pos[i])
            is_leaf = int(total[i]) == 1
            if is_leaf:
                is_motif_leaf = bool(motif_mask[i])
            else:
                is_motif_leaf = False
            # Motif residues are singleton groups: they appear as straight lifelines from t=0 to t=1.
            line_color = motif_line_color if is_motif_leaf else scaffold_color
            ax.plot([xi, xi], [y0, y1], linewidth=1.0, color=line_color)

        # Draw parent-child connectors at birth times
        for child_idx in range(A):
            p = int(parent[child_idx])
            if p < 0:
                continue
            y = float(birth[child_idx])
            if not np.isfinite(y):
                continue
            y = max(0.0, min(1.0, y))
            # All connectors here are within scaffold span trees (motif singletons never merge).
            ax.plot(
                [float(planar_pos[p]), float(planar_pos[child_idx])],
                [y, y],
                linewidth=1.0,
                color=scaffold_color,
            )

        # Scatter markers: roots at time 0, leaves at time 1, internal anchors at their split time
        motif_roots: List[int] = []
        scaffold_roots: List[int] = []
        for r in sorted(roots):
            if int(total[r]) == 1 and bool(motif_mask[r]):
                motif_roots.append(r)
            else:
                scaffold_roots.append(r)

        if len(motif_roots) > 0:
            ax.scatter(
                [float(planar_pos[r]) for r in motif_roots],
                [0.0] * len(motif_roots),
                marker="o",
                s=20,
                color=motif_marker_color,
                label="motif roots",
            )
        if len(scaffold_roots) > 0:
            ax.scatter(
                [float(planar_pos[r]) for r in scaffold_roots],
                [0.0] * len(scaffold_roots),
                marker="o",
                s=20,
                color=scaffold_color,
                label="scaffold roots",
            )

        deleted_leaf_ids = [i for i in range(N_leaf) if bool(leaf_deleted[i])]
        motif_leaf_ids = [
            i
            for i in range(N_leaf)
            if (bool(motif_mask[i]) and not bool(leaf_deleted[i]))
        ]
        scaffold_leaf_ids = [
            i
            for i in range(N_leaf)
            if ((not bool(motif_mask[i])) and not bool(leaf_deleted[i]))
        ]
        if len(motif_leaf_ids) > 0:
            ax.scatter(
                [float(planar_pos[i]) for i in motif_leaf_ids],
                [1.0] * len(motif_leaf_ids),
                marker="s",
                s=14,
                color=motif_marker_color,
                label="motif leaves",
            )
        if len(scaffold_leaf_ids) > 0:
            ax.scatter(
                [float(planar_pos[i]) for i in scaffold_leaf_ids],
                [1.0] * len(scaffold_leaf_ids),
                marker="s",
                s=14,
                color=scaffold_color,
                label="scaffold leaves",
            )
        if len(deleted_leaf_ids) > 0:
            ax.scatter(
                [float(planar_pos[i]) for i in deleted_leaf_ids],
                [float(delete[i]) for i in deleted_leaf_ids],
                marker="s",
                s=28,
                facecolors="none",
                edgecolors="red",
                linewidths=1.0,
                label="deaths (deleted leaves)",
            )

        # Mark anchors (finite split_time) for internal nodes
        internal_nodes = [
            i for i in range(A) if (int(total[i]) > 1 and np.isfinite(split[i]))
        ]
        if len(internal_nodes) > 0:
            internal_y = [float(split[i]) for i in internal_nodes]
            internal_x_plot = [float(planar_pos[i]) for i in internal_nodes]
            ax.scatter(
                internal_x_plot,
                internal_y,
                marker="^",
                s=14,
                color=scaffold_color,
                label="scaffold anchors",
            )

        ax.set_xlabel("Planar x-position (leaf index space)")
        ax.set_ylabel("Time")
        ax.set_title(
            f"TreePlan (N={self.num_leaves}, A={self.num_nodes}, R={int(self.roots.numel())})"
        )

        # Time axis: 0 at top, 1 at bottom
        ax.set_ylim(1.02, -0.02)
        # x-limit based on actual planar positions (extra deleted roots are within span, not at end)
        N_data = self.num_leaves - self.num_deletions
        ax.set_xlim(-1, max(float(planar_pos.max()) + 1.0, float(N_data)))
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), borderaxespad=0.0)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        logger.info(f"💾 Saved tree plan to {path}")
        return path


@dataclass
class BatchedTreePlan:
    """Batched/padded tree tensors derived from per-sample TreePlans.

    All tensors are padded to (B, A_max[,2]) on CPU in the collate function.
    Field names match TreePlan, but with an added batch dimension.
    """

    # topology
    topo_order: torch.Tensor  # (B, A_max) long
    motif_mask: torch.Tensor  # (B, A_max) bool (aligned tree-space motif mask)
    roots: torch.Tensor  # (B, R_max) long, -1 padded
    roots_mask: torch.Tensor  # (B, R_max) bool
    parent_idx: torch.Tensor  # (B, A_max) long
    children_idx: torch.Tensor  # (B, A_max, 2) long
    total_leaves: torch.Tensor  # (B, A_max) long
    node_depth: torch.Tensor  # (B, A_max) long
    leaf_deleted: torch.Tensor  # (B, A_max) bool
    planar_position: torch.Tensor  # (B, A_max) float

    # times
    birth_time: torch.Tensor  # (B, A_max) float
    split_time: torch.Tensor  # (B, A_max) float
    delete_time: torch.Tensor  # (B, A_max) float

    # tree -> data mapping
    leaf_map: torch.Tensor  # (B, A_max) long

    def to(self, device: torch.device) -> "BatchedTreePlan":
        """Move all tensors to the specified device."""
        return BatchedTreePlan(
            topo_order=to_device(self.topo_order, device),
            motif_mask=to_device(self.motif_mask, device),
            roots=to_device(self.roots, device),
            roots_mask=to_device(self.roots_mask, device),
            parent_idx=to_device(self.parent_idx, device),
            children_idx=to_device(self.children_idx, device),
            total_leaves=to_device(self.total_leaves, device),
            node_depth=to_device(self.node_depth, device),
            leaf_deleted=to_device(self.leaf_deleted, device),
            planar_position=to_device(self.planar_position, device),
            birth_time=to_device(self.birth_time, device),
            split_time=to_device(self.split_time, device),
            delete_time=to_device(self.delete_time, device),
            leaf_map=to_device(self.leaf_map, device),
        )

    def present_mask(
        self,
        t: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, A) bool
        """Mask of nodes present (alive) at time t."""
        end_time = torch.minimum(self.split_time, self.delete_time)
        return ((self.birth_time <= t[:, None]) & (t[:, None] < end_time)).bool()

    @property
    def leaf_mask(self) -> torch.Tensor:  # (B, A_max) bool
        """True for leaf nodes (total_leaves == 1), False for internal/padding."""
        return self.total_leaves == 1

    @property
    def is_internal(self) -> torch.Tensor:  # (B, A_max) bool
        """True for internal nodes (total_leaves > 1), False for leaves/padding."""
        return self.total_leaves > 1

    def remaining_insertions_t(
        self,
        t: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, A) long
        """Remaining insertions per node at time t, i.e. total_leaves - 1 for present nodes."""
        present = self.present_mask(t=t).to(torch.long)
        remaining = (self.total_leaves - 1).clamp_min(0) * present
        return remaining.to(torch.long)

    def broadcast_to_leaves(
        self,
        x: torch.Tensor,  # (B, N, ...) or (B, N, N) if is_2d
        fill_value: Union[float, torch.Tensor] = 0.0,
        is_2d: bool = False,
    ) -> torch.Tensor:  # (B, A_max, ...) or (B, A_max, A_max) if is_2d
        """Broadcast data from N-space to A-space, keeping only leaf positions.

        Args:
            x: (B, N, ...) tensor to broadcast, or (B, N, N) if is_2d=True
            fill_value: value for non-leaf positions
            is_2d: if True, x is (B, N, N) and we broadcast along both dims
                   to produce (B, A_max, A_max). Used for contact_conditioning.
        """
        if x.ndim < 2:
            raise ValueError(
                f"Expected x to have at least 2 dims (B, N, ...); got {x.shape}"
            )
        leaf_idx = self.leaf_map.to(device=x.device).clamp_min(0)  # (B, A)
        # Exclude leaves with no data reference (leaf_map == -1, e.g. extra deleted roots)
        has_data = self.leaf_map >= 0
        leaf_mask = self.leaf_mask.to(device=x.device) & has_data.to(device=x.device)
        return gather_and_pad(
            x, leaf_idx, mask=leaf_mask, fill_value=fill_value, is_2d=is_2d
        )

    def traverse_bottom_up(
        self,
        node_values: torch.Tensor,  # (B, A, ...)
        # combine_fn(batch_idx, node_idx, children, child_weights, node_values) -> parent_values
        combine_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            torch.Tensor,
        ],
    ) -> torch.Tensor:
        """Traverse tree bottom-up (children before parents) to compute node values."""
        max_depth = int(self.node_depth.max().item())
        for depth in range(1, max_depth + 1):
            at_depth = self.node_depth == depth
            valid = at_depth & self.is_internal & (self.total_leaves > 0)
            if not valid.any():
                continue
            batch_idx, node_idx = torch.where(valid)
            children = self.children_idx[batch_idx, node_idx, :]  # (N_valid, 2)
            # Gather child weights properly using batch indices
            child_weights = torch.stack(
                [
                    self.total_leaves[batch_idx, children[:, 0]],
                    self.total_leaves[batch_idx, children[:, 1]],
                ],
                dim=1,
            )  # (N_valid, 2)
            parent_values = combine_fn(
                batch_idx, node_idx, children, child_weights, node_values
            )
            node_values[batch_idx, node_idx] = parent_values.to(node_values.dtype)
        return node_values

    def traverse_top_down(
        self,
        creation_state: torch.Tensor,  # (B, A, ...)
        target_state: torch.Tensor,  # (B, A, ...)
        # split_fn(node_creation, node_target, node_t0, node_st) -> node_at_split
        split_fn: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ],
        max_split_time: Optional[torch.Tensor] = None,  # (B, A) optional time cutoff
    ) -> torch.Tensor:
        """Traverse tree top-down (parents before children) to propagate states.

        Args:
            creation_state: (B, A, ...) initial states at birth times
            target_state: (B, A, ...) target states at t=1
            split_fn: callable to sample bridge at split time
            max_split_time: (B, A) optional cutoff; skip splits happening after this time (e.g. corruption t)
        """
        max_depth = int(self.node_depth.max().item())
        for depth in range(max_depth, 0, -1):
            at_depth = self.node_depth == depth
            valid = (
                at_depth
                & self.is_internal
                & torch.isfinite(self.split_time)
                & (self.total_leaves > 0)
            )

            if max_split_time is not None:
                valid = valid & (self.split_time <= max_split_time)

            if not valid.any():
                continue
            batch_idx, node_idx = torch.where(valid)
            node_t0 = self.birth_time[batch_idx, node_idx]
            node_st = self.split_time[batch_idx, node_idx]
            node_creation = creation_state[batch_idx, node_idx]
            node_target = target_state[batch_idx, node_idx]
            node_at_split = split_fn(node_creation, node_target, node_t0, node_st)
            children = self.children_idx[batch_idx, node_idx, :]
            node_at_split = node_at_split.to(creation_state.dtype)
            for j in range(2):
                c_indices = children[:, j]
                valid_children = c_indices >= 0
                if valid_children.any():
                    creation_state[
                        batch_idx[valid_children], c_indices[valid_children]
                    ] = node_at_split[valid_children]
        return creation_state

    @classmethod
    def collate(cls, plans: List[TreePlan]) -> "BatchedTreePlan":
        B = len(plans)
        if B == 0:
            raise ValueError("Empty batch")

        A_max = max(int(p.num_nodes) for p in plans)
        R_max = max(int(p.roots.numel()) for p in plans)

        # roots_mask derived from per-sample root lengths
        roots_mask = (
            torch.arange(R_max)[None, :]
            < torch.tensor([p.roots.numel() for p in plans])[:, None]
        )

        return cls(
            # pad topo_order with -1 so accidental use raises an error
            topo_order=pad_and_stack(
                [p.topo_order for p in plans], A_max, fill_value=-1, dtype=torch.long
            ),
            motif_mask=pad_and_stack(
                [p.motif_mask for p in plans], A_max, fill_value=0, dtype=torch.bool
            ),
            roots=pad_and_stack(
                [p.roots for p in plans], R_max, fill_value=-1, dtype=torch.long
            ),
            roots_mask=roots_mask,
            parent_idx=pad_and_stack(
                [p.parent_idx for p in plans], A_max, fill_value=-1, dtype=torch.long
            ),
            children_idx=pad_and_stack(
                [p.children_idx for p in plans], A_max, fill_value=-1, dtype=torch.long
            ),
            total_leaves=pad_and_stack(
                [p.total_leaves for p in plans], A_max, fill_value=0, dtype=torch.long
            ),
            node_depth=pad_and_stack(
                [p.node_depth for p in plans], A_max, fill_value=0, dtype=torch.long
            ),
            leaf_deleted=pad_and_stack(
                [p.leaf_deleted for p in plans], A_max, fill_value=0, dtype=torch.bool
            ),
            planar_position=pad_and_stack(
                [p.planar_position for p in plans],
                A_max,
                fill_value=0.0,
                dtype=torch.float32,
            ),
            birth_time=pad_and_stack(
                [p.birth_time for p in plans],
                A_max,
                fill_value=float("inf"),
                dtype=torch.float32,
            ),
            split_time=pad_and_stack(
                [p.split_time for p in plans],
                A_max,
                fill_value=float("inf"),
                dtype=torch.float32,
            ),
            delete_time=pad_and_stack(
                [p.delete_time for p in plans],
                A_max,
                fill_value=float("inf"),
                dtype=torch.float32,
            ),
            leaf_map=pad_and_stack(
                [p.leaf_map for p in plans], A_max, fill_value=-1, dtype=torch.long
            ),
        )


""" Data / Batch structs """


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

        # Valid mask for new batch
        new_valid = out_pos < out_lens.unsqueeze(1)
        is_insertion = is_insertion & new_valid

        # Update birth_time for inserted positions to current time t, set padding to inf
        new_birth = gather_and_pad(
            self.birth_time, gather_idx, new_valid, fill_value=float("inf")
        )
        new_birth = torch.where(
            is_insertion, torch.full_like(new_birth, t_birth), new_birth
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


""" Protein Dataset + DataLoader """


class ProteinDataset(BaseDataset):
    """Wrapper to simplify BaseDataset and extract relevant features"""

    def __init__(
        self,
        cfg: VarcoDatasetConfig,
        eval: bool = False,
        use_test: bool = False,
    ):
        super().__init__(
            cfg=cfg,
            task=DataTask.inpainting,
            eval=eval,
            use_test=use_test,
        )

    def __getitem__(self, idx) -> DataSample:
        feats = super().__getitem__(idx)

        motif_mask = feats[bp.motif_mask].bool()
        res_mask = feats[bp.res_mask].int()
        chain_idx = feats[bp.chain_idx].int()

        # domains
        trans_1 = feats[bp.trans_1]
        rotmats_1 = feats[bp.rotmats_1]
        aatypes_1 = feats[bp.aatypes_1]

        # conditioning + confidence
        contact_conditioning = feats[bp.contact_conditioning]
        res_bfactor = feats[bp.res_bfactor]
        res_plddt = feats[bp.res_plddt]

        tree_plan = TreePlan.generate(motif_mask=motif_mask)
        tree_plan.validate()

        return DataSample(
            tree_plan=tree_plan,
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


class ProteinDataLoader(DataLoader):
    """DataLoader for ProteinDataset"""

    def __init__(
        self,
        dataset: ProteinDataset,
        batch_size: int = 1,
        num_workers: int = max(1, os.cpu_count() - 2),
        **kwargs,
    ):
        # force custom collate_fn
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=ProteinDataLoader.collate_fn,
            prefetch_factor=None if num_workers == 0 else 2,
            persistent_workers=False,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch: List[DataSample]) -> DataBatch:
        # Special handling for tree collation
        plans = [item.tree_plan for item in batch]
        tree = BatchedTreePlan.collate(plans)

        motif_mask = torch.stack([item.motif_mask for item in batch])  # (B, N)
        res_mask = torch.stack([item.res_mask for item in batch])  # (B, N)
        chain_idx = torch.stack([item.chain_idx for item in batch])  # (B, N)
        trans_1 = torch.stack([item.trans_1 for item in batch])  # (B, N, 3)
        rotmats_1 = torch.stack([item.rotmats_1 for item in batch])  # (B, N, 3, 3)
        aatypes_1 = torch.stack([item.aatypes_1 for item in batch])  # (B, N)
        contact_conditioning = torch.stack(
            [item.contact_conditioning for item in batch]
        )  # (B, N, N)
        res_bfactor = torch.stack([item.res_bfactor for item in batch])  # (B, N)
        res_plddt = torch.stack([item.res_plddt for item in batch])  # (B, N)

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


class ProteinDataModule(pl.LightningDataModule):
    """DataModule for ProteinDataset"""

    def __init__(
        self,
        cfg: DataConfig,
        dataset: ProteinDataset,
    ):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def train_dataloader(self, rank=None, num_replicas=None) -> DataLoader:
        batch_sampler = LengthBatcher(
            sampler_cfg=self.cfg.sampler,
            metadata_csv=self.dataset.csv,
            modeled_length_col=self.dataset.cfg.modeled_trim_method.to_dataset_column(),
            rank=rank or 0,
            num_replicas=num_replicas or 1,
        )

        return ProteinDataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.cfg.loader.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return ProteinDataLoader(
            dataset=self.dataset,
            batch_size=1,
            num_workers=0,
        )


""" Model """


class BranchFlowModel(nn.Module):
    def __init__(self, cfg: VarcoModelConfig):
        super().__init__()
        self.cfg = cfg

        self.num_aatype_tokens = NUM_TOKENS + 1  # 21: 20 amino acids + X
        self.pos_embed_dim = self.cfg.hyper_params.pos_embed_size
        self.time_embed_dim = self.cfg.hyper_params.timestep_embed_size
        self.node_dim = self.cfg.hyper_params.node_embed_size
        self.edge_dim = self.cfg.hyper_params.edge_embed_size

        self.input_dim = (
            1  # birth_time
            + 1  # motif_mask
            + 1  # chain_idx
            + self.time_embed_dim  # time_embed
            + self.pos_embed_dim  # pos_embed
            + self.num_aatype_tokens  # aatypes_onehot
        )

        # simpler MLP than NodeFeatureNet
        self.node_feature_net = nn.Sequential(
            nn.Linear(self.input_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim),
        )

        self.edge_feature_net = EdgeFeatureNet(cfg=self.cfg.edge_features)

        if self.cfg.esm_combiner.enabled:
            self.esm_combiner = ESMCombinerNetwork(cfg=self.cfg.esm_combiner)

        self.trunk = AttentionTrunk(
            cfg=self.cfg.trunk,
            attn_cfg=self.cfg.attention,
        )

        # IPA trunk for structure prediction (trans + rotmats using rigids in nm)
        self.ipa_trunk = AttentionIPATrunk(
            cfg=self.cfg.ipa,
            perform_final_edge_update=self.cfg.seq_trunk.enabled,
            perform_backbone_update=True,
            predict_psi_torsions=False,
            predict_all_torsions=False,
        )

        # Seq trunk
        if self.cfg.seq_trunk.enabled:
            self.seq_trunk = AttentionTrunk(
                cfg=self.cfg.seq_trunk,
                attn_cfg=self.cfg.attention,
            )

        # Base amino acid logits
        self.aatype_pred = AminoAcidPredictionNet(cfg=self.cfg.aa_pred)

        # Insertion amino acid logits
        self.insertion_logits_pred = nn.Sequential(
            nn.Linear(self.node_dim + self.num_aatype_tokens * 2, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.num_aatype_tokens),
        )

        # Insertions and deletions
        self.split_rate_pred = nn.Linear(self.node_dim, 1)
        self.split_pooled_log1p_rate_pred = nn.Linear(self.node_dim, 1)
        self.del_logits_pred = nn.Linear(self.node_dim, 1)

        # Confidence prediction modules (from cogeneration)
        if self.cfg.bfactor.enabled:
            self.bfactor_net = BFactorModule(cfg=self.cfg.bfactor)
        if self.cfg.plddt.enabled:
            self.plddt_net = PLDDTModule(cfg=self.cfg.plddt)

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        B, P, _ = batch.trans_t.shape

        valid = batch.valid_mask.float()  # (B, P)
        edge_valid = valid[:, None, :] * valid[:, :, None]  # (B, P, P)

        res_idx = BatchFeaturizer.infer_res_index(
            chain_idx=batch.chain_idx,
            valid_mask=batch.valid_mask,
        )  # (B, P)
        pos_embed = get_index_embedding(
            res_idx,
            embed_size=self.pos_embed_dim,
            max_len=1024,
            pos_embed_method=PositionalEmbeddingMethod.rotary,
        )  # (B, P, pos_embed_dim)
        pos_embed = pos_embed * batch.valid_mask.unsqueeze(-1).float()

        time_embed = get_time_embedding(
            timesteps=batch.t,
            embedding_dim=self.time_embed_dim,
            max_positions=1024,
        )[:, None, :].repeat(
            1, P, 1
        )  # (B, P, time_embed_dim)
        time_embed = time_embed * batch.valid_mask.unsqueeze(-1).float()

        # One-hot encode aatypes_t
        aatypes_onehot = F.one_hot(
            batch.aatypes_t.long(), num_classes=self.num_aatype_tokens
        ).float()  # (B, P, 21)

        # clamp birth_time +inf padding to 1.0
        birth_time = batch.birth_time[:, :, None].float().clamp(0.0, 1.0)

        input_feats = torch.cat(
            [
                birth_time,  # (B, P, 1)
                batch.motif_mask[:, :, None].float(),  # (B, P, 1)
                batch.chain_idx[:, :, None].float(),  # (B, P, 1)
                time_embed,  # (B, P, time_embed_dim)
                pos_embed,  # (B, P, pos_embed_dim)
                aatypes_onehot,  # (B, P, 21)
            ],
            dim=-1,
        )
        node_embed = self.node_feature_net(input_feats)  # (B, P, node_dim)
        node_embed = node_embed * valid.unsqueeze(-1)

        edge_embed = self.edge_feature_net(
            node_embed=node_embed,
            trans=batch.trans_t,
            trans_sc=None,
            edge_mask=edge_valid,
            diffuse_mask=~batch.motif_mask,
            chain_index=batch.chain_idx,
            contact_conditioning=batch.contact_conditioning,  # may be None
        )  # (B, P, P, edge_dim)
        edge_embed = edge_embed * edge_valid.unsqueeze(-1)

        init_node_embed = node_embed
        init_edge_embed = edge_embed
        if self.cfg.esm_combiner.enabled:
            node_embed, edge_embed = self.esm_combiner(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                aatypes_t=batch.aatypes_t,
                chain_index=batch.chain_idx,
                res_mask=torch.ones_like(valid),
                pad_mask=valid,
            )

        # Trunk
        node_embed, edge_embed = self.trunk(
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
            node_embed=node_embed,
            edge_embed=edge_embed,
            node_mask=valid,
            edge_mask=edge_valid,
            rigid=None,
            r3_t=batch.t,
        )

        # IPA trunk predicts structure updates using rigid, nm scale internally
        init_rigids = create_rigid(rots=batch.rotmats_t, trans=batch.trans_t)
        init_rigids_nm = rigids_ang_to_nm(init_rigids)
        node_embed, edge_embed, pred_rigids_nm, _ = self.ipa_trunk(
            node_embed=node_embed,
            edge_embed=edge_embed,
            node_mask=valid,
            edge_mask=edge_valid,
            diffuse_mask=valid,
            curr_rigids_nm=init_rigids_nm,
        )

        # Convert rigid back to angstroms for output
        pred_rigids_ang = rigids_nm_to_ang(pred_rigids_nm)
        pred_trans_1 = pred_rigids_ang.get_trans()  # (B, P, 3)
        pred_rotmats_1 = pred_rigids_ang.get_rots().get_rot_mats()  # (B, P, 3, 3)

        # Seq trunk
        if self.cfg.seq_trunk.enabled:
            node_embed, edge_embed = self.seq_trunk(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=valid,
                edge_mask=edge_valid,
                rigid=pred_rigids_nm,
                r3_t=batch.t,
            )

        # Predict amino acid logits
        pred_aatype_logits, _ = self.aatype_pred(
            node_embed=node_embed,
            aatypes_t=batch.aatypes_t,
            edge_embed=edge_embed,
            node_mask=valid,
            edge_mask=edge_valid,
            pred_rigids_nm=pred_rigids_nm,
            diffuse_mask=~batch.motif_mask,
            chain_index=batch.chain_idx,
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
        )  # (B, P, K)
        pred_aatype_logits = pred_aatype_logits * valid.unsqueeze(-1).float()

        # Predict insertion amino acid logits
        pred_insertion_logits = self.insertion_logits_pred(
            torch.cat(
                [
                    node_embed,  # (B, P, node_dim)
                    # aatypes_t one-hot (B, P, K)
                    F.one_hot(
                        batch.aatypes_t.long().clamp(0, self.num_aatype_tokens - 1),
                        num_classes=self.num_aatype_tokens,
                    ).float(),
                    # stopgrad pred logits (B, P, K)
                    pred_aatype_logits.detach(),
                ],
                dim=-1,
            )
        )  # (B, P, K)
        pred_insertion_logits = pred_insertion_logits * valid.unsqueeze(-1).float()

        # Predict nonnegative remaining-splits rates (Poisson-like regression)
        split_rate = F.softplus(self.split_rate_pred(node_embed)).squeeze(-1)  # (B, P)

        # Masked mean pool over alive tokens to predict total remaining insertions per example
        valid_count = valid.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        pooled = (node_embed * valid.unsqueeze(-1)).sum(
            dim=1
        ) / valid_count  # (B, model_dim)
        split_pooled_log1p_rate = self.split_pooled_log1p_rate_pred(pooled).squeeze(
            -1
        )  # (B,)

        # Predict deletion logits
        del_logits = self.del_logits_pred(node_embed).squeeze(-1)  # (B, P)

        # Confidence predictions
        pred_bfactor = None
        if self.cfg.bfactor.enabled:
            pred_bfactor = self.bfactor_net(node_embed=node_embed)  # (B, P, num_bins)

        pred_plddt = None
        if self.cfg.plddt.enabled:
            pred_plddt = self.plddt_net(node_embed=node_embed)  # (B, P, num_bins)

        return ModelPrediction(
            pred_trans_1=pred_trans_1,
            pred_rotmats_1=pred_rotmats_1,
            pred_aatype_logits=pred_aatype_logits,
            pred_insertion_logits=pred_insertion_logits,
            pred_split_rate=split_rate,
            pred_split_pooled_log1p_rate=split_pooled_log1p_rate,
            pred_del_logits=del_logits,
            pred_bfactor=pred_bfactor,
            pred_plddt=pred_plddt,
        )


""" Tree Coupling """


@dataclass
class Coupling(ABC):
    """
    Coupling struct captures domain-specific creation / anchor values for corruption tree plan.
    """

    tree: BatchedTreePlan
    anchors: torch.Tensor  # (B, A, ...)
    creation_state: torch.Tensor  # (B, A, ...)

    def validate(self) -> None:
        return


CouplingT = TypeVar("CouplingT")


class Coupler(ABC, Generic[CouplingT]):
    """Abstract interface for a domain coupler.

    A coupler owns:
      - how to sample the base distribution for its domain (`sample_base`)
      - how to corrupt a domain endpoint `x1` to time `t` using a shared tree plan (`corrupt`)

    The coupler returns the corrupted feature aligned to the tree capacity A, plus a domain-specific
    coupling/targets object used for supervision.
    """

    @abstractmethod
    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool - True for motif positions
        x1: torch.Tensor,  # (B, N, ...) endpoint values at t=1
        device: torch.device,
    ) -> torch.Tensor:
        """Sample base distribution values at t=0.

        Args:
            motif_mask: (B, N) bool mask where True indicates motif positions
            x1: (B, N, ...) endpoint values at t=1 (for motifs to preserve)
            device: device to create tensors on

        Returns:
            x0: (B, N, ...) base values at t=0
        """
        raise NotImplementedError

    @abstractmethod
    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, ...)
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Combine two child anchor values into parent anchor.

        Args:
            child_anchors: (N_valid, 2, ...) anchor values for the two children
            child_weights: (N_valid, 2) total_leaves weights for the two children

        Returns:
            parent_anchors: (N_valid, ...) combined anchor for parent
        """
        raise NotImplementedError

    @abstractmethod
    def sample_bridge(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        s: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample bridge from x_start at t0 to x_end at t=1, evaluated at time s.

        Args:
            x_start: Start values at time t0
            x_end: End values at time t=1
            s: Time to evaluate bridge
            t0: Birth time

        Returns:
            Sampled values at time s
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self,
        x_t: torch.Tensor,
        present_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Apply domain-specific masking for present_mask and motif_mask.

        Args:
            x_t: (B, A, ...) values at time t before masking
            present_mask: (B, A) True for nodes present at time t
            motif_mask: (B, A) True for motif nodes
            anchors: (B, A, ...) anchor values

        Returns:
            x_t: (B, A, ...) masked values
        """
        raise NotImplementedError

    def build_anchors(
        self,
        x1: torch.Tensor,
        tree: BatchedTreePlan,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Generic anchor building using depth-based traversal.

        Args:
            x1: (B, N, ...) endpoint values at t=1
            tree: Batched tree plan
            fill_value: Value to use for padding/internal nodes initially

        Returns:
            anchors: (B, A, ...) anchor values for all nodes
        """
        anchors = tree.broadcast_to_leaves(x1, fill_value=fill_value)

        def combine_fn(batch_idx, node_idx, children, child_weights, node_values):
            child_anchors = torch.stack(
                [
                    node_values[batch_idx, children[:, 0]],
                    node_values[batch_idx, children[:, 1]],
                ],
                dim=1,
            )
            return self.combine_anchors(
                child_anchors=child_anchors, child_weights=child_weights
            )

        return tree.traverse_bottom_up(anchors, combine_fn)

    @staticmethod
    def _compute_sigma_t(
        t: torch.Tensor,  # (B,)
        scale: torch.Tensor,  # (B,) per-domain scale
        min_sigma: float = 0.0,
        noise_end_t: float = 0.95,
    ) -> torch.Tensor:
        """
        Compute the instantaneous standard deviation of the noise at time t.

        Uses a sqrt-parabolic schedule with boundary conditions sigma(0)=sigma(1)=min_sigma:
            sigma(t) = sqrt(scale^2 * t * (1 - t) + min_sigma^2)

        Additionally, to make the final sampling steps noise-free, time is warped so that
        sigma(t) becomes 0 (up to min_sigma) for t >= noise_end_t.
        """
        if t.ndim != 1:
            raise ValueError(f"Expected t to have shape (B,); got {tuple(t.shape)}")
        if scale.shape != t.shape:
            raise ValueError(
                f"Expected scale to have shape (B,); got {tuple(scale.shape)} vs {tuple(t.shape)}"
            )
        if not (0.0 < float(noise_end_t) <= 1.0):
            raise ValueError(f"Expected noise_end_t in (0, 1]; got {noise_end_t}")

        t_eff = (t / float(noise_end_t)).clamp(0.0, 1.0)
        return torch.sqrt(
            scale.square() * t_eff * (1.0 - t_eff) + float(min_sigma) ** 2
        )

    def bridge_step(
        self,
        coupling: CouplingT,
        x_prev: torch.Tensor,  # (B, A, ...)
        t_prev: float,
        t_next: float,
    ) -> torch.Tensor:
        """Advance a single coupled bridge path from t_prev -> t_next.

        This is intended for time-coupled visualization / debugging: it reuses a fixed
        sampled coupling (anchors + creation_state) and produces a single stochastic
        trajectory instead of independent marginal samples at each timepoint.

        Notes:
        - Per-position start time is clamped as t0 = max(t_prev, birth_time).
        - For positions not yet born at t_prev, we start from coupling.creation_state.
        - post_process() is applied at t_next (masking, motif fixing, etc.).
        """
        if float(t_next) < float(t_prev):
            raise ValueError(f"Expected t_next >= t_prev; got {t_next} < {t_prev}")

        tree = coupling.tree
        birth_time = tree.birth_time  # (B, A)

        # validate x_prev is in aligned space, matches coupling states
        if (
            x_prev.shape != coupling.creation_state.shape
            or x_prev.shape != coupling.anchors.shape
        ):
            raise ValueError(
                "x_prev, coupling.creation_state, and coupling.anchors must have the same shape; "
                f"got x_prev={tuple(x_prev.shape)}, creation_state={tuple(coupling.creation_state.shape)}, "
                f"anchors={tuple(coupling.anchors.shape)}"
            )

        t_prev_full = torch.full_like(birth_time, float(t_prev))  # (B, A)
        t_next_full = torch.full_like(birth_time, float(t_next))  # (B, A)

        # start time is min_clamped by birth_time
        t0 = torch.maximum(t_prev_full, birth_time)

        # start state is creation state if t_prev < birth_time, otherwise provided x_prev
        use_prev = birth_time <= float(t_prev)  # (B, A)
        use_prev_expanded = use_prev
        while use_prev_expanded.ndim < x_prev.ndim:
            use_prev_expanded = use_prev_expanded.unsqueeze(-1)
        x_start = torch.where(use_prev_expanded, x_prev, coupling.creation_state)

        # sample bridge from x_prev (x_start) -> t_next (anchors)
        x_next = self.sample_bridge(
            x_start=x_start,  # (B, A, ...)
            x_end=coupling.anchors,  # (B, A, ...)
            s=t_next_full,
            t0=t0,
        )

        # Apply domain-specific masking / constraints at t_next
        present_mask = tree.present_mask(
            t=torch.full(
                (birth_time.shape[0],), float(t_next), device=birth_time.device
            )
        )
        x_next = self.post_process(
            x_t=x_next,
            present_mask=present_mask,
            motif_mask=tree.motif_mask,
            anchors=coupling.anchors,
        )

        return x_next

    def corrupt(
        self,
        tree: BatchedTreePlan,
        t: torch.Tensor,  # (B,)
        x1: torch.Tensor,  # (B, N, ...)
        x0: Optional[torch.Tensor] = None,  # (B, A, ...)
    ) -> Tuple[torch.Tensor, Coupling]:
        """Generic corruption using depth-based traversal.

        Args:
            tree: Batched tree plan
            t: (B,) corruption time for each example
            x1: (B, N, ...) endpoint values at t=1
            x0: (B, A, ...) optional base values at t=0 (aligned to tree)

        Returns:
            x_t: (B, A, ...) corrupted values at time t
            coupling: Domain-specific coupling object
        """
        device = x1.device
        B = x1.shape[0]
        A = tree.parent_idx.shape[1]
        t_expanded = t.unsqueeze(1).expand(B, A)

        if x0 is None:
            # Broadcast x1 to aligned space (copies leaves, fills internal nodes)
            x1_aligned = tree.broadcast_to_leaves(x1, fill_value=0)
            # Sample base distribution for all positions
            x0 = self.sample_base(
                motif_mask=tree.motif_mask,
                x1=x1_aligned,
                device=device,
            )

        # traverse bottom up to get anchors (which aggregate leaves).
        # for trans and rots, anchors are deterministic "centers" of leaves
        # for aatypes, anchor probs are deterministic but we *sample* an anchor token
        anchors = self.build_anchors(x1=x1, tree=tree)

        def split_fn(node_creation, node_target, node_t0, node_st):
            return self.sample_bridge(
                x_start=node_creation,
                x_end=node_target,
                s=node_st,
                t0=node_t0,
            )

        # traverse top-down from roots -> anchors -> leaves to get creation states
        # motifs + roots will use t=0 state. anchors states determined by bridges.
        creation_state = x0.clone()
        creation_state = tree.traverse_top_down(
            creation_state=creation_state,
            target_state=anchors,
            split_fn=split_fn,
            max_split_time=t_expanded,
        )

        # track domain's coupling: creation state + anchors (target states)
        coupling = self._make_coupling(
            tree=tree, anchors=anchors, creation_state=creation_state
        )

        # sample bridge from creation state -> anchor/terminal state to get state at t
        x_t = self.sample_bridge(
            x_start=creation_state,
            x_end=anchors,
            s=t_expanded,
            t0=tree.birth_time,
        )

        if x_t.dtype.is_floating_point and not torch.isfinite(x_t).all():
            raise RuntimeError(
                "sample_bridge produced non-finite values. "
                "Domain couplers must clamp time deltas for unborn nodes."
            )

        # domain specific post-processing, mostly null out non-present nodes
        x_t = self.post_process(
            x_t=x_t,
            present_mask=tree.present_mask(t=t),
            motif_mask=tree.motif_mask,
            anchors=anchors,
        )

        return x_t, coupling

    @abstractmethod
    def _make_coupling(
        self,
        tree: BatchedTreePlan,
        anchors: torch.Tensor,  # (B, A, ...)
        creation_state: torch.Tensor,  # (B, A, ...)
    ) -> Coupling:
        """Create domain-specific coupling object."""
        raise NotImplementedError

    @abstractmethod
    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, d)
        x1_pred: torch.Tensor,  # (B, P, d)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, ...)
    ) -> torch.Tensor:
        """Single Euler(-Maruyama) step for sampling.

        Noise is controlled by the coupler instance (e.g. sigma); if sigma is None/0, this is deterministic.
        Guidance potential can be provided to pull toward targets.
        """
        raise NotImplementedError


@dataclass
class TranslationCoupling(Coupling):
    """Coupling for translations using Brownian bridge."""

    pass


class TranslationCoupler(Coupler[TranslationCoupling]):
    def __init__(self, cfg: VarcoInterpolantTransCouplerConfig):
        self.cfg = cfg

    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool
        x1: torch.Tensor,  # (B, N, 3)
        device: torch.device,
    ) -> torch.Tensor:
        """Sample gaussian translations for all positions at t=0."""
        B, N = motif_mask.shape
        return centered_gaussian(B, N, n_bb_atoms=3, device=device) * NM_TO_ANG_SCALE

    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, 3)
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Weighted average of child translation anchors."""
        wsum = child_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = child_weights.float() / wsum.float()  # (N_valid, 2)
        return (child_anchors * weights.unsqueeze(-1)).sum(dim=1)  # (N_valid, 3)

    def sample_bridge(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        s: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample Brownian-bridge marginal at time s from (x_start at t0) to (x_end at 1)."""
        original_shape = x_start.shape
        flat_start = x_start.view(-1, 3)
        flat_end = x_end.view(-1, 3)
        flat_s = s.reshape(-1)
        flat_t0 = t0.reshape(-1)

        denom = (1.0 - flat_t0).clamp_min(1e-6)
        u = ((flat_s - flat_t0) / denom).clamp(0.0, 1.0)
        mean = flat_start + u.unsqueeze(-1) * (flat_end - flat_start)

        if self.cfg.noise_scale == 0.0:
            return mean.view(original_shape)

        var = (
            (flat_s - flat_t0).clamp_min(0.0) * (1.0 - flat_s).clamp_min(0.0) / denom
        ).clamp_min(0.0)
        std = (var.sqrt() * self.cfg.noise_scale).to(mean.dtype)
        eps = torch.randn_like(mean)
        return (mean + std.unsqueeze(-1) * eps).view(original_shape)

    def post_process(
        self,
        x_t: torch.Tensor,
        present_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Zero out non-present nodes."""
        return torch.where(
            present_mask.unsqueeze(-1),
            x_t,
            torch.zeros_like(x_t),
        )

    def _make_coupling(
        self,
        anchors: torch.Tensor,  # (B, A, 3)
        tree: BatchedTreePlan,
        creation_state: torch.Tensor,  # (B, A, 3)
    ) -> TranslationCoupling:
        return TranslationCoupling(
            tree=tree,
            anchors=anchors,
            creation_state=creation_state,
        )

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, 3)
        x1_pred: torch.Tensor,  # (B, P, 3)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, 3) guidance VF
    ) -> torch.Tensor:  # (B, P, 3)
        """Euler step for translation updates (all in angstroms)."""
        trans_pred = x1_pred
        if x_t.shape != trans_pred.shape:
            raise ValueError(
                f"x_t and x1_pred must match shape; got {tuple(x_t.shape)} vs {tuple(trans_pred.shape)}"
            )
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(
                f"Expected x_t to have shape (B, P, 3); got {tuple(x_t.shape)}"
            )
        if potential is not None:
            if potential.shape != x_t.shape:
                raise ValueError(
                    f"potential and x_t must match shape; got {tuple(potential.shape)} vs {tuple(x_t.shape)}"
                )

        B, P, _ = x_t.shape
        device = x_t.device

        valid_fmask = (
            (birth_time <= t[:, None]).bool().float().unsqueeze(-1)
        )  # (B, P, 1)

        denom = (1.0 - t).clamp_min(1e-4).view(B, 1, 1)
        v = (trans_pred - x_t) / denom

        if potential is not None:
            v = v + potential

        # cap drift jump
        drift_step = v * float(dt)
        if float(self.cfg.drift_step_cap_ang) > 0.0:
            cap = float(self.cfg.drift_step_cap_ang)
            step_norm = drift_step.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            drift_step = drift_step * (cap / step_norm).clamp(max=1.0)

        x_next = x_t + drift_step

        # add brownian noise
        if float(self.cfg.noise_scale) > 0.0 and float(dt) > 0.0:
            scale = torch.full_like(t, float(self.cfg.noise_scale))
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=scale,
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            ).to(dtype=x_next.dtype, device=x_next.device)
            x_next = x_next + torch.randn_like(x_next) * (
                sigma_t.view(B, 1, 1) * math.sqrt(float(dt))
            )

        return x_next * valid_fmask + x_t * (1.0 - valid_fmask)


@dataclass
class AATypesCoupling(Coupling):
    """Coupling for amino acid types using CTMC bridge."""

    # Anchor probability distributions (before sampling)
    anchor_probs: torch.Tensor  # (B, A, K)


class AATypesCoupler(Coupler[AATypesCoupling]):
    """
    Coupler for amino acid types using a continuous-time Markov chain (CTMC) bridge.

    Uses a uniform substitution CTMC (complete graph) with closed-form transition probabilities.
    The bridge maintains creation_token at birth_time and anchor token at t=1 for each node.
    """

    K: int = NUM_TOKENS + 1  # 21 tokens (20 amino acids + mask/X token at index 20)

    def __init__(
        self,
        cfg: VarcoInterpolantAATypesCouplerConfig,
    ):
        self.cfg = cfg

    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool
        x1: torch.Tensor,  # (B, N) amino acid indices
        device: torch.device,
    ) -> torch.Tensor:
        """Sample base distribution amino acid types at t=0.

        Preserves motif sequences and samples random sequences for scaffolds.
        """
        B, N = motif_mask.shape
        # Sample uniform random amino acids for all positions
        x0 = uniform_categorical(B, N, num_tokens=self.K, device=device)
        # Preserve motif sequences from x1
        x0 = torch.where(motif_mask, x1, x0)
        return x0

    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, K) probability distributions
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Weighted mixture of child probability distributions."""
        wsum = child_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = (child_weights.float() / wsum).unsqueeze(-1)  # (N_valid, 2, 1)
        return (child_anchors * weights).sum(dim=1)  # (N_valid, K)

    def _transition_prob(
        self,
        delta: torch.Tensor,  # (...) time delta
        same: bool = True,  # True if staying prob, False if transition prob
    ) -> torch.Tensor:
        """
        Compute CTMC transition probability for uniform substitution model.

        For a uniform substitution CTMC with leaving rate β:
        - P_ii(Δ) = 1/K + (1 - 1/K) * exp(-λΔ)
        - P_ij(Δ) = 1/K - (1/K) * exp(-λΔ)  for i ≠ j

        where λ = β * K / (K - 1)
        """
        delta = delta.clamp_min(0.0)
        lam = self.cfg.beta * self.K / (self.K - 1)
        exp_term = torch.exp(-lam * delta)
        inv_K = 1.0 / self.K

        if same:
            return inv_K + (1.0 - inv_K) * exp_term
        else:
            return inv_K - inv_K * exp_term

    def _bridge_marginal(
        self,
        token_start: torch.Tensor,  # (M,) long, token at t0
        token_end: torch.Tensor,  # (M,) long, token at t=1
        s: torch.Tensor,  # (M,) time to sample at
        t0: torch.Tensor,  # (M,) birth time
    ) -> torch.Tensor:
        """
        Sample from CTMC bridge marginal at time s conditioned on endpoints.

        For a CTMC bridge from X_{t0}=i to X_1=j, the marginal at time s is:
        P(X_s = k | X_{t0} = i, X_1 = j) ∝ P(s - t0)_{i,k} * P(1 - s)_{k,j}
        """
        M = token_start.shape[0]
        device = token_start.device

        # Time intervals
        delta_start = (s - t0).clamp_min(0.0)  # (M,)
        delta_end = (1.0 - s).clamp_min(0.0)  # (M,)

        # Build transition probability matrix rows
        # P(s - t0)_{i,k} for all k
        # P(1 - s)_{k,j} for all k
        K = self.K

        # Compute probabilities for each candidate token k
        # P(s - t0)_{i,k}: probability of going from i to k in time (s - t0)
        # P(1 - s)_{k,j}: probability of going from k to j in time (1 - s)

        # For efficiency, compute staying vs leaving probabilities
        p_stay_start = self._transition_prob(delta_start, same=True)  # (M,)
        p_jump_start = self._transition_prob(delta_start, same=False)  # (M,)
        p_stay_end = self._transition_prob(delta_end, same=True)  # (M,)
        p_jump_end = self._transition_prob(delta_end, same=False)  # (M,)

        # Build full probability vector for each token k
        # P_{i,k}(Δ1) = p_stay_start if k == i else p_jump_start
        # P_{k,j}(Δ2) = p_stay_end if k == j else p_jump_end

        # Create one-hot indicators
        k_range = torch.arange(K, device=device).unsqueeze(0)  # (1, K)
        is_start = (k_range == token_start.unsqueeze(1)).float()  # (M, K)
        is_end = (k_range == token_end.unsqueeze(1)).float()  # (M, K)

        # Transition probs from start to each k
        p_start_to_k = is_start * p_stay_start.unsqueeze(1) + (
            1.0 - is_start
        ) * p_jump_start.unsqueeze(
            1
        )  # (M, K)

        # Transition probs from each k to end
        p_k_to_end = is_end * p_stay_end.unsqueeze(1) + (
            1.0 - is_end
        ) * p_jump_end.unsqueeze(
            1
        )  # (M, K)

        # Bridge marginal: P(X_s = k) ∝ P_{i,k}(Δ1) * P_{k,j}(Δ2)
        probs = p_start_to_k * p_k_to_end  # (M, K)

        # Normalize, handling zero-sum rows (invalid/padding nodes)
        row_sums = probs.sum(dim=-1, keepdim=True)  # (M, 1)
        has_mass = row_sums > 1e-12  # (M, 1)

        # For rows with no mass (padding), use uniform distribution to allow multinomial
        uniform_fallback = torch.ones(M, K, device=device) / K
        probs = torch.where(
            has_mass, probs / row_sums.clamp_min(1e-12), uniform_fallback
        )

        # Sample
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (M,)

        # For invalid rows, return the start token as fallback
        sampled = torch.where(has_mass.squeeze(-1), sampled, token_start)

        return sampled

    def build_anchor_alignment(
        self,
        aatypes_1: torch.Tensor,
        tree: BatchedTreePlan,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build anchor tokens and probability distributions bottom-up.

        Args:
            aatypes_1: (B, N) amino acid types at t=1
            tree: Batched tree plan

        Returns:
            anchor_tokens: (B, A) long, sampled anchor tokens for all nodes
            anchor_probs: (B, A, K) probability distributions for all nodes
        """
        if aatypes_1.ndim != 2:
            raise ValueError(f"Expected aatypes_1 shape (B, N); got {aatypes_1.shape}")

        device = aatypes_1.device
        B, N = aatypes_1.shape
        A = tree.parent_idx.shape[1]
        K = self.K

        leaf_tokens = tree.broadcast_to_leaves(
            aatypes_1.to(torch.long),
            fill_value=MASK_TOKEN_INDEX,
        )

        anchor_probs = torch.zeros(B, A, K, device=device)

        leaf_mask = tree.leaf_mask
        leaf_idx = leaf_tokens.clamp(0, K - 1)
        anchor_probs.scatter_(
            -1,
            leaf_idx.unsqueeze(-1),
            leaf_mask.unsqueeze(-1).float(),
        )

        def combine_fn(batch_idx, node_idx, children, child_weights, node_values):
            child_probs = torch.stack(
                [
                    node_values[batch_idx, children[:, 0]],
                    node_values[batch_idx, children[:, 1]],
                ],
                dim=1,
            )
            return self.combine_anchors(
                child_anchors=child_probs,
                child_weights=child_weights,
            )

        anchor_probs = tree.traverse_bottom_up(anchor_probs, combine_fn)

        # uniform fallback where no mass for multinomial()
        row_sums = anchor_probs.sum(dim=-1, keepdim=True)
        has_mass = row_sums > 1e-12
        uniform_fallback = torch.ones(B, A, K, device=device) / K
        anchor_probs = torch.where(
            has_mass,
            anchor_probs / row_sums.clamp_min(1e-12),
            uniform_fallback,
        )

        # Mix noise into anchor_probs to sample anchor tokens
        noisy_anchor_probs = anchor_probs
        if self.cfg.noise_scale > 0:
            noise_weight = min(self.cfg.noise_scale * 0.15, 0.2)
            noisy_anchor_probs = (
                1.0 - noise_weight
            ) * noisy_anchor_probs + noise_weight * uniform_fallback
            # disallow mask token for anchor tokens
            noisy_anchor_probs[:, :, MASK_TOKEN_INDEX] = 0.0
            noisy_anchor_probs = noisy_anchor_probs / noisy_anchor_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        anchor_probs_flat = noisy_anchor_probs.view(-1, K)
        anchor_tokens = torch.multinomial(
            anchor_probs_flat,
            num_samples=1,
        ).squeeze(-1)
        anchor_tokens = anchor_tokens.view(B, A)

        anchor_tokens = torch.where(leaf_mask, leaf_tokens, anchor_tokens)

        anchor_tokens = torch.where(
            has_mass.squeeze(-1),
            anchor_tokens,
            torch.full_like(anchor_tokens, MASK_TOKEN_INDEX),
        )

        return anchor_tokens, anchor_probs

    def build_anchors(
        self,
        x1: torch.Tensor,
        tree: BatchedTreePlan,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Override to call build_anchor_alignment and return tokens only."""
        anchor_tokens, anchor_probs = self.build_anchor_alignment(
            aatypes_1=x1,
            tree=tree,
        )
        self._last_anchor_probs = anchor_probs
        return anchor_tokens

    def sample_bridge(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        s: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample CTMC bridge marginal at time s."""
        original_shape = x_start.shape
        return self._bridge_marginal(
            token_start=x_start.reshape(-1),
            token_end=x_end.reshape(-1),
            s=s.reshape(-1),
            t0=t0.reshape(-1),
        ).view(original_shape)

    def post_process(
        self,
        x_t: torch.Tensor,
        present_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Mask non-present nodes and fix motif sequence."""
        x_t = torch.where(
            present_mask,
            x_t,
            torch.full_like(x_t, MASK_TOKEN_INDEX),
        )
        x_t = torch.where(motif_mask, anchors, x_t)
        return x_t

    def _make_coupling(
        self,
        tree: BatchedTreePlan,
        anchors: torch.Tensor,  # (B, A)
        creation_state: torch.Tensor,  # (B, A)
    ) -> AATypesCoupling:
        return AATypesCoupling(
            tree=tree,
            anchors=anchors,
            creation_state=creation_state,
            anchor_probs=self._last_anchor_probs,  # (B, A, K)
        )

    def _uncertainty_gate(
        self,
        x_t: torch.Tensor,  # (B, P) current tokens
        probs: torch.Tensor,  # (B, P, K) predicted probabilities
    ) -> torch.Tensor:
        """
        Compute uncertainty gate: 1 - p(current token), raised to sharpness.
        Reduces jump probability when the model is confident about the current token.
        """
        K = self.K
        # Get probability of current token
        p_current = probs.gather(-1, x_t.long().clamp(0, K - 1).unsqueeze(-1)).squeeze(
            -1
        )
        p_current = p_current.clamp(0.0, 1.0)  # (B, P)

        # Uncertainty = (1 - p_current) ^ sharpness
        uncertainty = (1.0 - p_current) ** self.cfg.uncertainty_sharpness
        return uncertainty  # (B, P)

    def _compute_step_probs(
        self,
        logits: torch.Tensor,  # (B, P, K) predicted logits for t=1
        x_t: torch.Tensor,  # (B, P) current tokens
        t: torch.Tensor,  # (B,) current time
        dt: float,
        valid_mask: torch.Tensor,  # (B, P) positions that are valid (born)
    ) -> torch.Tensor:
        """
        Compute regularized step probabilities for discrete Euler sampling.
        Applies temperature, uncertainty gate, noise, leave mass cap, and regularization.
        """
        B, P = x_t.shape
        K = self.K
        device = x_t.device

        # Softmax with temperature
        probs = F.softmax(logits / self.cfg.drift_temp, dim=-1)  # (B, P, K)

        # Uncertainty gating
        uncertainty = self._uncertainty_gate(x_t, probs)  # (B, P)

        # Drift gain: 1 / (1 - t), clamped
        t_clamped = t.clamp(0.0, 0.99)
        drift_gain = 1.0 / (1.0 - t_clamped)
        drift_gain = drift_gain.clamp(max=float(self.cfg.drift_gain_cap)).view(
            B, 1, 1
        )  # (B, 1, 1)

        # Compute off-diagonal drift mass
        step_probs = dt * drift_gain * probs * uncertainty.unsqueeze(-1)  # (B, P, K)

        # Zero out current token (off-diagonal only)
        current_onehot = F.one_hot(x_t.long().clamp(0, K - 1), num_classes=K).float()
        step_probs = step_probs * (1.0 - current_onehot)

        # Noise injection: add uniform mass to off-diagonal, scaled by sigma_t
        if self.cfg.noise_scale > 0:
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=torch.ones_like(t),
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            ).view(B, 1, 1)
            noise_weight = float(self.cfg.noise_scale) * dt * sigma_t.square()

            # Uniform over non-current tokens
            uniform_noise = (1.0 - current_onehot) / max(K - 1, 1)
            step_probs = step_probs + noise_weight * uniform_noise

        # Cap leave mass
        if self.cfg.leave_mass_cap is not None and self.cfg.leave_mass_cap > 0:
            row_sum = step_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            shrink = (self.cfg.leave_mass_cap / row_sum).clamp_max(1.0)
            step_probs = step_probs * shrink

        # Regularize: set diagonal to 1 - sum(off-diagonal)
        row_sum = step_probs.sum(dim=-1, keepdim=True)
        step_probs = step_probs + current_onehot * (1.0 - row_sum)

        # Clamp to valid range
        step_probs = step_probs.clamp(min=0.0, max=1.0)

        # For invalid positions: set to "stay" distribution (100% on current token)
        # This avoids wasted compute and makes debugging clearer
        stay_dist = current_onehot  # (B, P, K)
        step_probs = torch.where(valid_mask.unsqueeze(-1), step_probs, stay_dist)

        return step_probs

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P) current tokens
        x1_pred: torch.Tensor,  # (B, P, K) predicted logits for t=1
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, K) guidance logits
    ) -> torch.Tensor:
        """Single Euler step for discrete amino acid sampling.

        Uses discrete jump process with uncertainty gating and noise injection.
        See _compute_step_probs for details on the probability computation.
        """
        B, P = x_t.shape
        K = self.K

        if x1_pred.shape != (B, P, K):
            raise ValueError(f"Expected x1_pred shape (B, P, K); got {x1_pred.shape}")

        assert potential is None, "potential not yet supported"

        # Valid positions are those born before current time
        valid_mask = birth_time <= t[:, None]  # (B, P)

        # Compute step probabilities with uncertainty gating, noise, and regularization
        step_probs = self._compute_step_probs(
            logits=x1_pred,
            x_t=x_t,
            t=t,
            dt=dt,
            valid_mask=valid_mask,
        )

        # Sample new tokens
        x_next = torch.multinomial(step_probs.view(-1, K), num_samples=1).squeeze(-1)
        x_next = x_next.view(B, P)

        # Keep invalid positions unchanged
        x_next = torch.where(valid_mask, x_next, x_t)

        # Keep motif positions fixed
        x_next = torch.where(motif_mask, x_t, x_next)

        return x_next


@dataclass
class RotationCoupling(Coupling):
    """Coupling for SO(3) rotations using geodesic bridge with IGSO3 noise."""

    pass


class RotationCoupler(Coupler[RotationCoupling]):
    """
    Coupler for SO(3) rotations using geodesic interpolation with IGSO3 noise.

    Implements stochastic bridges on SO(3) using the IGSO3 distribution for
    intermediate noise. Supports geodesic interpolation between rotation matrices.
    """

    def __init__(self, cfg: VarcoInterpolantRotationCouplerConfig):
        self.cfg = cfg
        self._igso3: Optional[so3_utils.SampleIGSO3] = None
        self._device: torch.device = torch.device("cpu")

    def set_device(self, device: torch.device):
        """Set device for IGSO3 sampler. Must be called before using."""
        self._device = device
        # Move IGSO3 to GPU for CUDA, but keep on CPU for MPS (VonMises issues)
        if self._igso3 is not None and device.type != "mps":
            self._igso3.to(device)

    @property
    def igso3(self) -> so3_utils.SampleIGSO3:
        """Lazy initialization of IGSO3 sampler."""
        if self._igso3 is None:
            steps = 1000
            sigma_grid = torch.logspace(
                math.log10(self.cfg.igso3_sigma_min),
                math.log10(self.cfg.igso3_sigma_max),
                steps=steps,
                dtype=torch.float64,
            )
            self._igso3 = so3_utils.SampleIGSO3(steps, sigma_grid, cache_dir=".cache")
            if self._device is not None and self._device.type != "mps":
                self._igso3.to(self._device)
            self._igso3 = self._igso3.float()
        return self._igso3

    def _ensure_igso3_device(self, target_device: torch.device):
        """Ensure IGSO3 is on correct device before sampling."""
        igso3_device = self.igso3.sigma_grid.device
        if target_device.type == "mps":
            # IGSO3 stays on CPU for MPS, results moved after
            return
        if igso3_device != target_device:
            self._igso3.to(target_device)

    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool
        x1: torch.Tensor,  # (B, N, 3, 3) rotation matrices
        device: torch.device,
    ) -> torch.Tensor:
        """Sample uniform SO(3) rotations for scaffolds, keep motif rotations."""
        B, N = motif_mask.shape
        # Sample uniform SO(3) for all positions
        return uniform_so3(B, N, device=device)

    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, 3, 3)
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Weighted geodesic average of two child rotation anchors."""
        # Normalize weights
        wsum = child_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = child_weights.float() / wsum.float()  # (N_valid, 2)

        # For two rotation matrices R0 and R1 with weights w0 and w1:
        # Use geodesic interpolation: geodesic_t(w1, R1, R0)
        # where w1 is the weight of the second child (first child has weight w0 = 1 - w1)
        R0 = child_anchors[:, 0]  # (N_valid, 3, 3)
        R1 = child_anchors[:, 1]  # (N_valid, 3, 3)
        # w1 needs shape (N_valid, 1) to broadcast with rot_vf (N_valid, 3) -> (N_valid, 3)
        w1 = weights[:, 1].unsqueeze(-1)  # (N_valid, 1)

        # Geodesic from R0 toward R1, parameterized by w1
        return so3_utils.geodesic_t(t=w1, mat=R1, base_mat=R0)  # (N_valid, 3, 3)

    def sample_bridge(
        self,
        x_start: torch.Tensor,  # (..., 3, 3)
        x_end: torch.Tensor,  # (..., 3, 3)
        s: torch.Tensor,  # (...) current time
        t0: torch.Tensor,  # (...) birth time
    ) -> torch.Tensor:
        """Sample SO(3) bridge marginal at time s from (x_start at t0) to (x_end at 1)."""
        original_shape = x_start.shape  # (..., 3, 3)

        # Flatten to (N, 3, 3) for processing
        flat_start = x_start.reshape(-1, 3, 3)  # (N, 3, 3)
        flat_end = x_end.reshape(-1, 3, 3)  # (N, 3, 3)
        flat_s = s.reshape(-1)  # (N,)
        flat_t0 = t0.reshape(-1)  # (N,)
        N = flat_start.shape[0]

        # Compute interpolation parameter u in [0, 1]
        denom = (1.0 - flat_t0).clamp_min(1e-6)
        u = ((flat_s - flat_t0) / denom).clamp(0.0, 1.0)  # (N,)

        # Deterministic geodesic interpolation
        # u needs shape (N, 1) to broadcast with rot_vf (N, 3) -> (N, 3)
        mean = so3_utils.geodesic_t(
            t=u.unsqueeze(-1),  # (N, 1)
            mat=flat_end,
            base_mat=flat_start,
        )  # (N, 3, 3)

        if self.cfg.noise_scale == 0.0:
            return mean.view(original_shape)

        # Stochastic bridge: add IGSO3 noise scaled by bridge variance
        # Variance for Brownian bridge: (s - t0)(1 - s) / (1 - t0)
        var = (
            (flat_s - flat_t0).clamp_min(0.0) * (1.0 - flat_s).clamp_min(0.0) / denom
        ).clamp_min(
            0.0
        )  # (N,)

        # Scale variance by cfg.sigma and compute std for IGSO3
        std = (var.sqrt() * self.cfg.noise_scale).clamp_min(1e-6)  # (N,)

        # Sample IGSO3 noise and apply
        self._ensure_igso3_device(mean.device)
        sigma_for_igso3 = std.clamp(
            self.cfg.igso3_sigma_min, self.cfg.igso3_sigma_max
        ).to(self.igso3.sigma_grid.device)

        # Only sample noise where std > min threshold
        apply_mask = std > self.cfg.igso3_sigma_min
        if apply_mask.any():
            identity_noise = (
                torch.eye(3, device=mean.device).unsqueeze(0).expand(N, -1, -1)
            )
            sigma_sel = sigma_for_igso3[apply_mask]
            noise_sel = self.igso3.sample(sigma_sel, 1).to(mean.device)
            noise_sel = noise_sel.squeeze(1)  # (N_apply, 3, 3)
            intermediate_noise = identity_noise.clone()
            intermediate_noise[apply_mask] = noise_sel
            mean = torch.einsum("...ij,...jk->...ik", mean, intermediate_noise)

        return mean.view(original_shape)

    def post_process(
        self,
        x_t: torch.Tensor,  # (B, P, 3, 3)
        present_mask: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        anchors: torch.Tensor,  # (B, P, 3, 3)
    ) -> torch.Tensor:
        """Set non-present nodes to identity rotation."""
        identity = torch.eye(3, device=x_t.device, dtype=x_t.dtype)
        return torch.where(
            present_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
            x_t,
            identity.unsqueeze(0).unsqueeze(0).expand_as(x_t),
        )

    def _make_coupling(
        self,
        tree: BatchedTreePlan,
        anchors: torch.Tensor,  # (B, A, 3, 3)
        creation_state: torch.Tensor,  # (B, A, 3, 3)
    ) -> RotationCoupling:
        return RotationCoupling(
            tree=tree,
            anchors=anchors,
            creation_state=creation_state,
        )

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, 3, 3)
        x1_pred: torch.Tensor,  # (B, P, 3, 3)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, 3) rotation vector field
    ) -> torch.Tensor:
        """
        Single Euler step for rotation sampling using geodesic flow.

        Uses calc_rot_vf to compute vector field and geodesic_t to step.
        Optionally adds IGSO3 noise for stochastic sampling.
        """
        if x_t.shape != x1_pred.shape:
            raise ValueError(
                f"x_t and x1_pred must match shape; got {tuple(x_t.shape)} vs {tuple(x1_pred.shape)}"
            )
        if x_t.ndim != 4 or x_t.shape[-2:] != (3, 3):
            raise ValueError(
                f"Expected x_t to have shape (B, P, 3, 3); got {tuple(x_t.shape)}"
            )
        if potential is not None and potential.shape != (*x_t.shape[:2], 3):
            raise ValueError(
                f"potential must have shape (B, P, 3); got {tuple(potential.shape)}"
            )

        B, P = x_t.shape[:2]
        device = x_t.device

        # Mask for valid (born) positions
        valid_mask = (
            (birth_time <= t[:, None]).unsqueeze(-1).unsqueeze(-1)
        )  # (B, P, 1, 1)

        # VF scaling: 1 / (1 - t), clamped
        # Using exponential schedule if configured
        if self.cfg.exp_rate > 0:
            r = self.cfg.exp_rate
            denom = 1.0 - torch.exp(-r * (1.0 - t))
            scaling = (r / denom.clamp_min(1e-8)).clamp_min(1e-4)
        else:
            scaling = (1.0 / (1.0 - t).clamp_min(1e-4)).clamp_min(1e-4)

        # Compute rotation vector field: Log_{x_t}(x1_pred)
        rot_vf = so3_utils.calc_rot_vf(mat_t=x_t, mat_1=x1_pred)  # (B, P, 3)

        # Add guidance potential if provided
        if potential is not None:
            rot_vf = rot_vf + potential

        # Clamp per-residue rotation step.
        drift_step_cap = float(self.cfg.drift_step_cap_rad)
        if drift_step_cap > 0.0:
            rot_step = rot_vf * (scaling.view(B, 1, 1) * float(dt))
            step_norm = rot_step.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rot_step = rot_step * (drift_step_cap / step_norm).clamp(max=1.0)
            rot_vf = rot_step / (scaling.view(B, 1, 1) * float(dt) + 1e-8)

        # Geodesic step: geodesic_t(scaling * dt, x1_pred, x_t, rot_vf)
        geodesic_time = (scaling * dt)[:, None, None]  # (B, 1, 1)
        x_next = so3_utils.geodesic_t(
            t=geodesic_time,
            mat=x1_pred,
            base_mat=x_t,
            rot_vf=rot_vf,
        )  # (B, P, 3, 3)

        # Optionally add IGSO3 noise for stochastic sampling
        if float(self.cfg.noise_scale) > 0.0 and float(dt) > 0.0:
            self._ensure_igso3_device(device)

            # Compute sigma_t scaled by sqrt(dt)
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=torch.full_like(t, float(self.cfg.noise_scale)),
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            ).clamp_max(float(self.cfg.igso3_sigma_max))
            sqrt_dt = math.sqrt(float(dt))
            sigma_t = (sigma_t * sqrt_dt).to(self.igso3.sigma_grid.device)

            apply_mask = sigma_t > self.cfg.igso3_sigma_min
            if apply_mask.any():
                sigma_sel = sigma_t[apply_mask]
                identity_noise = (
                    torch.eye(3, device=device)
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .expand(B, P, -1, -1)
                )
                noise_sel = self.igso3.sample(sigma_sel, P).to(device)
                noise_sel = noise_sel.reshape(apply_mask.sum(), P, 3, 3)
                intermediate_noise = identity_noise.clone()
                intermediate_noise[apply_mask] = noise_sel
                x_next = torch.einsum("...ij,...jk->...ik", x_next, intermediate_noise)

        # Keep unborn positions unchanged
        x_next = torch.where(valid_mask.expand(-1, -1, 3, 3), x_next, x_t)

        return x_next


""" Interpolant supporting tree coupling """


@dataclass
class TreeCouplings:
    """Container for all domain couplings from a single corruption call."""

    translation: TranslationCoupling
    aatypes: AATypesCoupling
    rotation: RotationCoupling


@dataclass
class TreeInterpolant:
    cfg: VarcoInterpolantConfig
    device: torch.device = torch.device("cpu")
    min_t: float = 0.005
    translation_coupler: Coupler[TranslationCoupling] = field(init=False)
    aatypes_coupler: Coupler[AATypesCoupling] = field(init=False)
    rotation_coupler: RotationCoupler = field(init=False)

    def __post_init__(self):
        self.translation_coupler = TranslationCoupler(cfg=self.cfg.trans_coupler)
        self.aatypes_coupler = AATypesCoupler(cfg=self.cfg.aatypes_coupler)
        self.rotation_coupler = RotationCoupler(cfg=self.cfg.rotation_coupler)

    def set_device(self, device: torch.device):
        self.device = device
        self.rotation_coupler.set_device(device)  # for IGSO3 device

    def seed_all(self, seed: int):
        if seed is None:
            return
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    @staticmethod
    def _hazard_multiplier(t_val: float, hazard: VarcoHazardConfig) -> float:
        """
        Compute g(t) = h(t) / (1 - H(t)) for a simple closed-form hazard family.

        This is the continuous-time multiplier that replaces the hard-coded 1/(1-t) when H(t)=t.
        """
        t = float(min(max(t_val, 0.0), 1.0 - 1e-8))
        power = int(max(1, hazard.power))

        if hazard.kind == VarcoHazardKind.uniform:
            survival = max(1e-8, 1.0 - t)
            return 1.0 / survival

        if hazard.kind == VarcoHazardKind.early_power:
            # H(t) = 1 - (1 - t)^p  =>  h(t) = p (1 - t)^(p-1),  S(t) = (1 - t)^p
            survival = max(1e-8, 1.0 - t)
            return float(power) / survival

        if hazard.kind == VarcoHazardKind.late_power:
            # H(t) = t^p  =>  h(t) = p t^(p-1),  S(t) = 1 - t^p
            t_pow = t**power
            survival = max(1e-8, 1.0 - t_pow)
            h = float(power) * (t ** (power - 1))
            return h / survival

        raise ValueError(f"Unknown hazard kind: {hazard.kind!r}")

    def compute_motif_guidance_vf(
        self,
        t: torch.Tensor,  # (B,)
        pred_trans_1: torch.Tensor,  # (B, P, 3)
        trans_1_motifs: torch.Tensor,  # (B, P, 3) true motif positions
        pred_rotmats_1: torch.Tensor,  # (B, P, 3, 3)
        rotmats_t: torch.Tensor,  # (B, P, 3, 3)
        rotmats_1_motifs: torch.Tensor,  # (B, P, 3, 3) true motif rotations
        motif_mask: torch.Tensor,  # (B, P)
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute guidance velocity fields pulling motif positions toward their targets.

        Returns (trans_guidance_vf, rotmats_guidance_vf), each (B, P, 3) or None.
        Returns (None, None) if guidance is disabled or no motifs present.
        """
        guidance_cfg = self.cfg.motif_guidance
        if not guidance_cfg.enabled or not motif_mask.any():
            return None, None

        B, P = motif_mask.shape
        t_clamped = t.clamp(min=1e-3, max=1.0 - 1e-3)

        # Compute scale based on config
        if guidance_cfg.scale_type == VarcoMotifGuidanceType.posterior_variance:
            # scale = 0.5 * g² / ω² where g = κ/t, ω² = κ²/(t² + κ²), κ = 1-t
            # see cogeneration interpolant for details
            kappa = 1.0 - t_clamped
            g = kappa / t_clamped
            omega2 = kappa**2 / (t_clamped**2 + kappa**2)
            scale = 0.5 * g * g / omega2
            scale = scale.clamp(min=0.0, max=guidance_cfg.var_scale_cap)
        elif guidance_cfg.scale_type == VarcoMotifGuidanceType.linear_decay:
            scale = guidance_cfg.linear_decay_strength * (1.0 - t_clamped)
        else:
            raise ValueError(f"Unknown scale_type: {guidance_cfg.scale_type}")

        # --- Translation guidance ---
        trans_guidance_vf = (trans_1_motifs - pred_trans_1) * scale.view(B, 1, 1)
        trans_guidance_vf = trans_guidance_vf * motif_mask.unsqueeze(-1).float()

        # Cap per-residue magnitude
        if guidance_cfg.max_step_force_ang > 0:
            norm = trans_guidance_vf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            trans_guidance_vf = trans_guidance_vf * (
                guidance_cfg.max_step_force_ang / norm
            ).clamp(max=1.0)

        # --- Rotation guidance ---
        # Compute rotation vector fields in tangent space
        rot_vf_to_target = so3_utils.calc_rot_vf(
            mat_t=rotmats_t, mat_1=rotmats_1_motifs
        )
        rot_vf_to_pred = so3_utils.calc_rot_vf(mat_t=rotmats_t, mat_1=pred_rotmats_1)

        # Guidance = scale * (target_vf - pred_vf)
        rotmats_guidance_vf = (rot_vf_to_target - rot_vf_to_pred) * scale.view(B, 1, 1)
        rotmats_guidance_vf = rotmats_guidance_vf * motif_mask.unsqueeze(-1).float()

        # Cap per-residue rotation magnitude (in radians)
        if guidance_cfg.max_rot_step_force_rad > 0:
            norm = rotmats_guidance_vf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rotmats_guidance_vf = rotmats_guidance_vf * (
                guidance_cfg.max_rot_step_force_rad / norm
            ).clamp(max=1.0)

        return trans_guidance_vf, rotmats_guidance_vf

    def pack_bridged_states(
        self,
        batch: DataBatch,
        t: torch.Tensor,  # (B,)
        trans_t: torch.Tensor,  # (B, A, 3)
        rotmats_t: torch.Tensor,  # (B, A, 3, 3)
        aatypes_t: torch.Tensor,  # (B, A)
    ) -> DataBridged:
        """
        Pack static batch fields + per-domain tree-aligned states into a DataBridged.
        """
        tree = batch.tree.to(self.device)
        trans_1 = batch.trans_1.to(self.device)
        rotmats_1 = batch.rotmats_1.to(self.device)
        present_mask = tree.present_mask(t=t)

        res_mask = tree.broadcast_to_leaves(
            x=batch.res_mask.to(self.device), fill_value=0
        )
        chain_idx = tree.broadcast_to_leaves(
            x=batch.chain_idx.to(self.device), fill_value=0
        )
        res_bfactor = tree.broadcast_to_leaves(
            x=batch.res_bfactor.to(self.device), fill_value=0.0
        )
        res_plddt = tree.broadcast_to_leaves(
            x=batch.res_plddt.to(self.device), fill_value=0.0
        )
        contact_conditioning = tree.broadcast_to_leaves(
            x=batch.contact_conditioning.to(self.device), fill_value=0.0, is_2d=True
        )

        trans_1_motifs = tree.broadcast_to_leaves(x=trans_1, fill_value=0.0)
        trans_1_motifs = trans_1_motifs * tree.motif_mask.unsqueeze(-1).float()

        identity = torch.eye(3, device=self.device, dtype=rotmats_1.dtype)
        rotmats_1_motifs = tree.broadcast_to_leaves(x=rotmats_1, fill_value=identity)
        rotmats_1_motifs = torch.where(
            tree.motif_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
            rotmats_1_motifs,
            identity.unsqueeze(0).unsqueeze(0).expand_as(rotmats_1_motifs),
        )

        # Recenter translations to maintain translation invariance
        trans_t = trans_t - batch_center_of_mass(trans_t, mask=present_mask)[:, None]

        bridged = DataBridged(
            t=t,
            present_mask=present_mask,
            motif_mask=tree.motif_mask,
            birth_time=tree.birth_time,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_t=trans_t,
            rotmats_t=rotmats_t,
            aatypes_t=aatypes_t,
            trans_1_motifs=trans_1_motifs,
            rotmats_1_motifs=rotmats_1_motifs,
            remaining_insertions=tree.remaining_insertions_t(t=t),
            deleted=tree.leaf_deleted,
            planar_position=tree.planar_position,
            res_bfactor=res_bfactor,
            res_plddt=res_plddt,
            contact_conditioning=contact_conditioning,
        )
        bridged.validate()
        return bridged

    def corrupt_to(
        self,
        batch: DataBatch,
        t: torch.Tensor,  # (B,)
        trans_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
        rotmats_0: Optional[torch.Tensor] = None,
    ) -> Tuple[DataBridged, TreeCouplings]:
        """Corrupt a batch to time t."""
        tree = batch.tree.to(self.device)
        t = t.to(self.device)

        # Corrupt domains to time t

        trans_t, trans_coupling = self.translation_coupler.corrupt(
            tree=tree,
            t=t,
            x1=batch.trans_1.to(self.device),
            x0=to_device(trans_0, self.device),
        )
        trans_coupling.validate()

        rotmats_t, rotation_coupling = self.rotation_coupler.corrupt(
            tree=tree,
            t=t,
            x1=batch.rotmats_1.to(self.device),
            x0=to_device(rotmats_0, self.device),
        )
        rotation_coupling.validate()

        aatypes_t, aatypes_coupling = self.aatypes_coupler.corrupt(
            tree=tree,
            t=t,
            x1=batch.aatypes_1.to(self.device),
            x0=to_device(aatypes_0, self.device),
        )
        aatypes_coupling.validate()

        couplings = TreeCouplings(
            translation=trans_coupling,
            aatypes=aatypes_coupling,
            rotation=rotation_coupling,
        )

        bridged = self.pack_bridged_states(
            batch=batch,
            t=t,
            trans_t=trans_t,
            rotmats_t=rotmats_t,
            aatypes_t=aatypes_t,
        )

        return bridged, couplings

    def corrupt_batch(self, batch: DataBatch) -> Tuple[DataBridged, TreeCouplings]:
        """
        Corrupt a batch to a shared time.
        Pick a single time to share across the batch, biased slightly toward later times,
        simply so they have a similar number of insertion/deletions to simulate
        since corruption is run across the batch
        """
        shared_t = torch.rand(1, device=self.device) ** 0.8
        shared_t = shared_t.clamp(min=self.min_t, max=1.0 - self.min_t)
        t = torch.ones(batch.trans_1.shape[0], device=self.device) * shared_t  # (B,)
        return self.corrupt_to(batch=batch, t=t)

    def corrupt_trajectory(
        self,
        batch: DataBatch,
        times: Optional[List[float]] = None,
        seed: Optional[int] = None,
        trans_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
        rotmats_0: Optional[torch.Tensor] = None,
    ) -> Tuple[Trajectory, TreeCouplings]:
        """Generate a time-coupled corruption trajectory"""
        self.set_device(batch.trans_1.device)
        self.seed_all(seed)

        B = batch.trans_1.shape[0]
        tree = batch.tree.to(self.device)
        trans_1 = batch.trans_1.to(self.device)
        rotmats_1 = batch.rotmats_1.to(self.device)
        aatypes_1 = batch.aatypes_1.to(self.device)

        if times is None:
            times = list(np.linspace(0.0, 1.0, 50))
        if len(times) == 0:
            raise ValueError("times must be non-empty")
        times = [float(np.clip(t, self.min_t, 1.0 - self.min_t)) for t in times]

        # corrupt to t_build to get couplings for trajectory
        t_build = float(times[-1])
        t_build_tensor = torch.ones(B, device=self.device) * t_build

        # Define consistent base samples for the whole trajectory (in aligned space)
        if trans_0 is None:
            trans_0 = self.translation_coupler.sample_base(
                motif_mask=tree.motif_mask,
                x1=tree.broadcast_to_leaves(trans_1, fill_value=0),
                device=self.device,
            )
        if rotmats_0 is None:
            rotmats_0 = self.rotation_coupler.sample_base(
                motif_mask=tree.motif_mask,
                x1=tree.broadcast_to_leaves(
                    rotmats_1, fill_value=torch.eye(3, device=self.device)
                ),
                device=self.device,
            )
        if aatypes_0 is None:
            aatypes_0 = self.aatypes_coupler.sample_base(
                motif_mask=tree.motif_mask,
                x1=tree.broadcast_to_leaves(aatypes_1, fill_value=MASK_TOKEN_INDEX),
                device=self.device,
            )

        # Build couplings once (anchors + creation states)
        _, trans_coupling = self.translation_coupler.corrupt(
            tree=tree,
            t=t_build_tensor,
            x1=trans_1,
            x0=to_device(trans_0, self.device),
        )
        _, rotmats_coupling = self.rotation_coupler.corrupt(
            tree=tree,
            t=t_build_tensor,
            x1=rotmats_1,
            x0=to_device(rotmats_0, self.device),
        )
        _, aatypes_coupling = self.aatypes_coupler.corrupt(
            tree=tree,
            t=t_build_tensor,
            x1=aatypes_1,
            x0=to_device(aatypes_0, self.device),
        )
        couplings = TreeCouplings(
            translation=trans_coupling,
            aatypes=aatypes_coupling,
            rotation=rotmats_coupling,
        )

        # Start from creation states (defined at birth_time for each node), and step forward.
        trans_cur = trans_coupling.creation_state
        rotmats_cur = rotmats_coupling.creation_state
        aatypes_cur = aatypes_coupling.creation_state
        t_prev = 0.0

        # Iterate through time, bridging current state to next timepoint
        samples: List[DataCorrupted] = []
        for t_val in tqdm(times, desc="corrupt_trajectory()", leave=False):
            trans_cur = self.translation_coupler.bridge_step(
                coupling=trans_coupling,
                x_prev=trans_cur,
                t_prev=t_prev,
                t_next=t_val,
            )
            rotmats_cur = self.rotation_coupler.bridge_step(
                coupling=rotmats_coupling,
                x_prev=rotmats_cur,
                t_prev=t_prev,
                t_next=t_val,
            )
            aatypes_cur = self.aatypes_coupler.bridge_step(
                coupling=aatypes_coupling,
                x_prev=aatypes_cur,
                t_prev=t_prev,
                t_next=t_val,
            )

            t_tensor = torch.ones(B, device=self.device) * float(t_val)
            bridged = self.pack_bridged_states(
                batch=batch,
                t=t_tensor,
                trans_t=trans_cur,
                rotmats_t=rotmats_cur,
                aatypes_t=aatypes_cur,
            )
            samples.append(bridged.pack_present())
            t_prev = float(t_val)

        return Trajectory(samples=samples), couplings

    @staticmethod
    def _sample_initial_positions(
        motif_mask: torch.Tensor,  # (B, N)
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample initial positions for branching flow sampling.

        For each batch element:
        - All motif positions (motif_mask == True) are included
        - Each scaffold span (contiguous motif_mask == False) contributes K sampled roots,
          where K is sampled uniformly from [min_scaffold_nuclei, min(max_scaffold_nuclei, span_len)]

        Returns:
            init_length: (B,) long - number of initial positions per batch element
            motif_idx: (B, P_max) long - source index in original (N,) data for motif positions,
                       -1 for scaffold root positions. Padding is implicit: positions >= init_length.
        """
        device = motif_mask.device
        B, N = motif_mask.shape
        motif_mask = motif_mask.bool()

        # Use SeededRNG for determinism
        rng = SeededRNG(seed=seed, device="cpu")

        # For each batch element, compute indices and whether they map to original data
        # motif_idx_val >= 0 means motif (value is source index), -1 means scaffold root
        init_indices_list: List[List[int]] = []

        for b in range(B):
            mask_b = motif_mask[b]  # (N,)
            # indices stores: source index for motifs, -1 for scaffold roots
            indices: List[int] = []

            i = 0
            while i < N:
                if mask_b[i].item():
                    # Motif position - include with source index
                    indices.append(i)
                    i += 1
                else:
                    # Scaffold span - find extent
                    span_start = i
                    while i < N and not mask_b[i].item():
                        i += 1
                    span_len = i - span_start

                    # Sample K roots for this span
                    k_hi = min(max_scaffold_nuclei, span_len)
                    k_lo = min(min_scaffold_nuclei, k_hi)
                    target_k = (
                        k_lo if k_lo == k_hi else (k_lo + rng.rand_int(k_hi - k_lo + 1))
                    )

                    # Add -1 for each scaffold root (they don't map to original data)
                    for _ in range(target_k):
                        indices.append(-1)

            init_indices_list.append(indices)

        # Compute init_length and P_max
        init_lengths = [len(indices) for indices in init_indices_list]
        P_max = max(init_lengths) if init_lengths else 0

        # Build output tensors
        init_length = torch.tensor(
            init_lengths, dtype=torch.long, device=device
        )  # (B,)
        # Padding value is -1, same as scaffold roots, but distinguished by position >= init_length
        motif_idx = torch.full((B, P_max), -1, dtype=torch.long, device=device)

        for b in range(B):
            L = init_lengths[b]
            motif_idx[b, :L] = torch.tensor(
                init_indices_list[b], dtype=torch.long, device=device
            )

        return init_length, motif_idx

    def _init_sampling_batch(
        self,
        data: DataBatch,
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
        seed: Optional[int] = None,
    ) -> DataCorrupted:
        """
        Initialize a batch of samples for sampling.

        Uses _sample_initial_positions to determine which positions to include at t=0:
        - Motif positions are gathered from data (res_mask, chain_idx)
        - Scaffold roots get fresh samples from base prior
        """
        device = self.device
        B, N = data.motif_mask.shape

        # Get initial position layout
        init_length, motif_idx = self._sample_initial_positions(
            motif_mask=data.motif_mask.to(device),
            min_scaffold_nuclei=min_scaffold_nuclei,
            max_scaffold_nuclei=max_scaffold_nuclei,
            seed=seed,
        )

        # Set up masks / indices
        P_max = motif_idx.shape[1]
        pos_idx = torch.arange(P_max, device=device).unsqueeze(0)  # (1, P_max)
        valid_mask = pos_idx < init_length.unsqueeze(1)  # (B, P_max)
        is_motif = (motif_idx >= 0) & valid_mask  # (B, P_max)

        motif_mask = is_motif

        birth_time = torch.full((B, P_max), float("inf"), device=device)
        birth_time[valid_mask] = 0.0

        # Gather some features from data using motif_idx (clamp to 0 for valid gather idx)
        # For scaffold roots (motif_idx=-1) and padding, fill with 0
        gather_idx = motif_idx.clamp(min=0)  # (B, P_max)
        res_mask = gather_and_pad(
            data.res_mask.to(device), gather_idx, is_motif, fill_value=0
        )
        chain_idx = gather_and_pad(
            data.chain_idx.to(device), gather_idx, is_motif, fill_value=0
        )

        # contact_conditioning: (B, N, N) -> (B, P_max, P_max), zeros for scaffolds
        contact_conditioning = gather_and_pad(
            data.contact_conditioning.to(device),
            gather_idx,
            is_motif,
            fill_value=0.0,
            is_2d=True,
        )
        # confident confidence!
        res_plddt = gather_and_pad(
            data.res_plddt.to(device), gather_idx, is_motif, fill_value=90.0
        )
        res_bfactor = None

        # Gather x1 values from data for motif positions (fill scaffolds with placeholder)
        trans_1_gathered = gather_and_pad(
            data.trans_1.to(device), gather_idx, is_motif, fill_value=0.0
        )
        rotmats_1_gathered = gather_and_pad(
            data.rotmats_1.to(device),
            gather_idx,
            is_motif,
            fill_value=torch.eye(3, device=device, dtype=data.rotmats_1.dtype),
        )
        aatypes_1_gathered = gather_and_pad(
            data.aatypes_1.to(device), gather_idx, is_motif, fill_value=MASK_TOKEN_INDEX
        )

        # Sample base distributions using coupler interfaces
        trans_0 = self.translation_coupler.sample_base(
            motif_mask=is_motif, x1=trans_1_gathered, device=device
        )
        rotmats_0 = self.rotation_coupler.sample_base(
            motif_mask=is_motif, x1=rotmats_1_gathered, device=device
        )
        aatypes_0 = self.aatypes_coupler.sample_base(
            motif_mask=is_motif, x1=aatypes_1_gathered, device=device
        )

        # init batch at min_t
        t = torch.full((B,), self.min_t, dtype=torch.float32, device=device)

        return DataCorrupted(
            t=t,
            motif_mask=motif_mask,
            birth_time=birth_time,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_t=trans_0,
            rotmats_t=rotmats_0,
            aatypes_t=aatypes_0,
            trans_1_motifs=trans_1_gathered,
            rotmats_1_motifs=rotmats_1_gathered,
            contact_conditioning=contact_conditioning,
            res_bfactor=res_bfactor,
            res_plddt=res_plddt,
        )

    @staticmethod
    def _sample_insert_delete_substitute(
        split_rate: torch.Tensor,  # (B, P)
        del_logits: torch.Tensor,  # (B, P)
        is_root: torch.Tensor,  # (B, P) bool
        valid_mask: torch.Tensor,  # (B, P) bool
        t_val: float,  # current time
        dt: float,
        split_hazard_mult: float,
        delete_hazard_mult: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample insert/delete/substitute events for present positions in a batch.
        """
        # Convert rates/logits to per-step probabilities using the configured hazard multipliers.
        # - split_rate predicts remaining-event statistics (counting-process view)
        # - del_logits predicts "destined-to-delete" probability (counting process with K in {0,1})
        dt_split = float(dt) * float(split_hazard_mult)
        dt_del = float(dt) * float(delete_hazard_mult)

        # Insert probability from split rate.
        lam_ins = (split_rate.clamp_min(0.0) * dt_split).clamp_max(20.0)
        p_ins = (1.0 - torch.exp(-lam_ins)).clamp(0.0, 0.95)

        # Delete probability from logits
        # del_logits predicts "destined-to-delete", convert to instantaneous probability
        lam_del = (torch.sigmoid(del_logits) * dt_del).clamp_max(20.0)
        p_del = (1.0 - torch.exp(-lam_del)).clamp(0.0, 0.95)
        p_del = torch.where(is_root, torch.zeros_like(p_del), p_del)

        insertions = torch.rand_like(p_ins) < p_ins
        deletions = torch.rand_like(p_del) < p_del
        insertions = insertions & valid_mask
        deletions = deletions & valid_mask

        # Resolve conflicts: if both sampled, mark substitution instead
        substitutions = insertions & deletions
        insertions = insertions & ~substitutions
        deletions = deletions & ~substitutions

        return insertions, deletions, substitutions

    def sample(
        self,
        model: BranchFlowModel,
        data: DataBatch,
        num_steps: int = 200,
        traj_frames: Optional[int] = None,
    ) -> SampleTrajectory:
        device = self.device

        # Create initial batch, which we edit in-place through the trajectory
        num_batch, _ = data.motif_mask.shape
        batch = self._init_sampling_batch(data=data)

        traj = SampleTrajectory()
        traj.samples.append(batch.detach_clone(device=torch.device("cpu")))

        model.eval()
        with torch.no_grad():
            t_grid = torch.linspace(self.min_t, 1.0, steps=num_steps + 1, device=device)
            pbar = tqdm(range(num_steps), total=num_steps, desc="Sampling", leave=False)
            for step_num in pbar:
                t_val = float(t_grid[step_num].item())
                t_next = float(t_grid[step_num + 1].item())
                dt = float(max(1e-6, t_next - t_val))

                # Set current time and predict
                batch.t = torch.full(
                    (num_batch,), t_val, dtype=torch.float32, device=device
                )
                pred = model.forward(batch)

                if traj_frames is None or step_num % traj_frames == 0:
                    traj.pred.append(pred.detach_clone(device=torch.device("cpu")))

                # Compute motif guidance VFs
                trans_guidance_vf, rotmats_guidance_vf = self.compute_motif_guidance_vf(
                    t=batch.t,
                    pred_trans_1=pred.pred_trans_1,
                    trans_1_motifs=batch.trans_1_motifs,
                    pred_rotmats_1=pred.pred_rotmats_1,
                    rotmats_t=batch.rotmats_t,
                    rotmats_1_motifs=batch.rotmats_1_motifs,
                    motif_mask=batch.motif_mask,
                )

                # Euler steps for domains

                trans_next = self.translation_coupler.euler_step(
                    x_t=batch.trans_t,
                    x1_pred=pred.pred_trans_1,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                    potential=trans_guidance_vf,
                )
                batch.trans_t = trans_next

                rotmats_next = self.rotation_coupler.euler_step(
                    x_t=batch.rotmats_t,
                    x1_pred=pred.pred_rotmats_1,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                    potential=rotmats_guidance_vf,
                )
                batch.rotmats_t = rotmats_next

                aatypes_next = self.aatypes_coupler.euler_step(
                    x_t=batch.aatypes_t,
                    x1_pred=pred.pred_aatype_logits,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                )
                batch.aatypes_t = aatypes_next

                # Sample and apply insertion/deletion events, disallowed in motifs
                is_root = batch.birth_time <= 0.0  # (B, P)
                split_hazard_mult = self._hazard_multiplier(
                    t_val=t_val, hazard=self.cfg.sampling.split_hazard
                )
                delete_hazard_mult = self._hazard_multiplier(
                    t_val=t_val, hazard=self.cfg.sampling.delete_hazard
                )
                insertions, deletions, _ = self._sample_insert_delete_substitute(
                    split_rate=pred.pred_split_rate,
                    del_logits=pred.pred_del_logits,
                    is_root=is_root,
                    valid_mask=batch.valid_mask & ~batch.motif_mask,
                    t_val=t_val,
                    dt=dt,
                    split_hazard_mult=split_hazard_mult,
                    delete_hazard_mult=delete_hazard_mult,
                )

                # Enforce max_length: block insertions once we're at the limit
                max_len = self.cfg.sampling.max_length
                cur_lens = batch.valid_mask.sum(dim=1)  # (B,)
                at_limit = cur_lens >= max_len  # (B,)
                if at_limit.any():
                    insertions = insertions & ~at_limit.unsqueeze(1)

                batch, insert_mask, gather_idx = batch.apply_insertions_deletions(
                    insertions=insertions,
                    deletions=deletions,
                    t_birth=t_next,  # born at t_next since after euler step
                )

                # Domain-specific initialization for newly inserted tokens.
                # TODO - use couplings for domain-specific corruptions
                if insert_mask.any():
                    # Add isotropic perturbation to inserted translations
                    trans_noise = torch.randn_like(batch.trans_t) * 0.5
                    batch.trans_t = (
                        batch.trans_t + insert_mask.unsqueeze(-1).float() * trans_noise
                    )

                    # Add small IGSO3 perturbation to inserted rotations
                    # Inserted positions inherit parent's rotation from apply_insertions_deletions
                    # Add noise to break symmetry
                    B_new, P_new = batch.rotmats_t.shape[:2]
                    self.rotation_coupler._ensure_igso3_device(device)
                    # Use small sigma for perturbation
                    sigma_insert = torch.full(
                        (B_new,),
                        self.cfg.rotation_coupler.igso3_sigma_min * 10,
                        device=self.rotation_coupler.igso3.sigma_grid.device,
                    )
                    insert_noise = self.rotation_coupler.igso3.sample(
                        sigma_insert, P_new
                    ).to(
                        device
                    )  # (B, P, 3, 3)
                    # Apply noise to inserted positions only
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

                    # Gather parent's insertion logits for new positions
                    insertion_logits_gathered = pred.pred_insertion_logits.gather(
                        1,
                        gather_idx.unsqueeze(-1).expand(-1, -1, K).clamp(0, P_old - 1),
                    )  # (B, P_new, K)

                    # Convert to probs and mix in uniform noise (like _compute_step_probs)
                    probs = F.softmax(
                        insertion_logits_gathered, dim=-1
                    )  # (B, P_new, K)
                    uniform_dist = torch.ones_like(probs) / K
                    probs = (
                        1.0 - self.cfg.aatypes_coupler.noise_scale
                    ) * probs + self.cfg.aatypes_coupler.noise_scale * uniform_dist

                    # Sample from noisy distribution
                    sampled_tokens = torch.multinomial(
                        probs.view(-1, K), num_samples=1
                    ).view(B_new, P_new)
                    batch.aatypes_t = torch.where(
                        insert_mask, sampled_tokens, batch.aatypes_t
                    )

                # Recenter translations to maintain translation invariance
                # Everything is "present" in sampling,so use valid_mask
                com = batch_center_of_mass(batch.trans_t, mask=batch.valid_mask)
                batch.trans_t = batch.trans_t - com[:, None, :]

                # Save
                if traj_frames is None or step_num % traj_frames == 0:
                    traj.samples.append(batch.detach_clone(device=torch.device("cpu")))

                # Update progress bar with batch dimensions
                B, P = batch.trans_t.shape[:2]
                pbar.set_postfix_str(f"B={B} P={P}")

                # Cleanup
                if step_num % 10 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            pbar.close()

        return traj


""" Loss """


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
    base_seq_ce: torch.Tensor  # unweighted token CE (nats)
    insertion_seq_ce: torch.Tensor  # unweighted soft CE on insertion logits (nats)
    insertion_target_entropy: (
        torch.Tensor
    )  # unweighted entropy of insertion targets (nats)
    insertion_ce_over_entropy: torch.Tensor  # mean over positions of CE/H(target)
    insertion_ce_minus_entropy: (
        torch.Tensor
    )  # mean over positions of CE - H(target) (nats)
    split_event_ce: torch.Tensor  # Bernoulli CE on split event (>0)
    split_event_precision: torch.Tensor
    split_event_recall: torch.Tensor
    split_event_f1: torch.Tensor
    split_rate_mae: torch.Tensor  # MAE on pred split vs target count
    split_rate_mae_pos: torch.Tensor  # MAE conditioned on target>0
    del_event_ce: (
        torch.Tensor
    )  # Bernoulli CE on delete event (terminal scaffold tokens)
    del_event_precision: torch.Tensor
    del_event_recall: torch.Tensor
    del_event_f1: torch.Tensor
    plddt_ce: torch.Tensor  # unweighted, unclamped CE (nats)
    plddt_bin_acc: torch.Tensor  # top-1 accuracy on bins
    plddt_bin_acc_pm1: torch.Tensor  # accuracy within ±1 bin
    plddt_bin_mae: torch.Tensor  # mean abs bin error


@dataclass
class BranchFlowLossCalculator:
    cfg: VarcoLossConfig

    def _time_norm_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-based normalization scale: 1 - min(t, clip).
        Higher weight (smaller divisor) as t -> 1.
        """
        return 1 - torch.min(
            t, torch.tensor(self.cfg.t_normalize_clip, device=t.device)
        )

    @staticmethod
    def log_clamp(x: torch.Tensor, threshold: float = 5.0) -> torch.Tensor:
        """
        Soft clamp using log compression above threshold.
        Preserves gradients above threshold but at diminishing scale: threshold + log(1 + excess)
        """
        return torch.where(x > threshold, threshold + torch.log1p(x - threshold), x)

    def _soft_ce_from_probs(
        self,
        pred_logits: torch.Tensor,  # (B, P, K)
        target_probs: torch.Tensor,  # (B, P, K)
        mask: torch.Tensor,  # (B, P)
        t: Optional[torch.Tensor] = None,  # (B,)
        apply_time_norm: bool = False,
        time_norm_divisor: float = 2.0,
        per_example: bool = True,
        mostly_mask_threshold: float = 0.75,
        require_mass: bool = True,
    ) -> torch.Tensor:
        """Soft cross-entropy on logits vs per-token target probabilities."""
        B, P, K = pred_logits.shape
        if apply_time_norm:
            if t is None:
                raise ValueError("t is required when apply_time_norm=True")
            t_norm = self._time_norm_scale(t=t).view(B, 1)  # (B, 1)

        # Zero out mask token and renormalize target probs
        target_probs_masked = target_probs.clone()
        target_probs_masked[:, :, MASK_TOKEN_INDEX] = 0.0
        row_sums = target_probs_masked.sum(dim=-1, keepdim=True)
        has_mass = row_sums.squeeze(-1) > 1e-8  # (B, P)
        target_probs_masked = target_probs_masked / row_sums.clamp_min(1e-8)

        log_probs = F.log_softmax(pred_logits, dim=-1)  # (B, P, K)
        ce_per_token = -(target_probs_masked * log_probs).sum(dim=-1)  # (B, P)

        if apply_time_norm:
            ce_per_token = ce_per_token / (float(time_norm_divisor) * t_norm)  # (B, P)

        is_mostly_mask = target_probs[:, :, MASK_TOKEN_INDEX] >= float(
            mostly_mask_threshold
        )  # (B, P)
        mask_f = mask.float() * (~is_mostly_mask).float()
        if require_mass:
            mask_f = mask_f * has_mass.float()

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

    def _seq_token_loss(
        self,
        pred_aatype_logits: torch.Tensor,  # (B, P, K)
        target_anchor_tokens: torch.Tensor,  # (B, P) long
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
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
    ) -> torch.Tensor:
        """Sequence loss: soft cross-entropy on amino acid logits vs anchor probability targets."""
        seq_loss = self._soft_ce_from_probs(
            pred_logits=pred_aatype_logits,
            target_probs=target_anchor_probs,
            mask=mask,
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
    ) -> torch.Tensor:
        """Aux metric: unweighted per-token CE on amino acids (nats)."""
        with torch.no_grad():
            B, P, K = pred_aatype_logits.shape
            ce = F.cross_entropy(
                pred_aatype_logits.view(-1, K),
                target_anchor_tokens.view(-1),
                reduction="none",
            ).view(B, P)
            valid = mask & (target_anchor_tokens != MASK_TOKEN_INDEX)
            valid_f = valid.float()
            return (ce * valid_f).sum() / valid_f.sum().clamp_min(1.0)

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
        Per-token Poisson Bregman divergence:
        D(k || r) = r - k + k * log(k/r)  for k > 0
        D(0 || r) = r                     for k = 0

        Primary loss is on scaffolds (mask & ~motif_mask), with a smaller penalty
        (motif_weight) applied to motif positions.
        """
        if batch.remaining_insertions is None:
            raise ValueError("batch.remaining_insertions is required for split loss")

        target = batch.remaining_insertions.to(torch.float32)  # (B, P)
        rate = pred.pred_split_rate.clamp_min(eps)  # (B, P)

        target_safe = target.clamp_min(eps)
        token_loss = torch.where(
            target > 0,
            rate - target + target * torch.log(target_safe / rate),
            rate,
        )

        # Upweight rare internal nodes (target > 0) and larger remaining counts
        # to discourage the "predict no insertions" degenerate.
        pos_weight = 5.0  # upweight insertion targets
        count_weight_power = 0.5  # upweight exp for remaining counts
        max_token_weight = 20.0  # cap weight

        token_weight = torch.ones_like(target)
        token_weight = torch.where(
            target > 0, torch.full_like(token_weight, pos_weight), token_weight
        )
        token_weight = token_weight * (1.0 + target).pow(float(count_weight_power))
        token_weight = token_weight.clamp_max(float(max_token_weight))

        # Scaffold loss (primary) and motif loss (small penalty)
        scaffold_mask = mask & ~motif_mask
        motif_loss_mask = mask & motif_mask

        scaffold_weight = token_weight * scaffold_mask.float()
        scaffold_denom = scaffold_weight.sum(dim=1).clamp_min(1.0)  # (B,)
        scaffold_loss = (
            (token_loss * scaffold_weight).sum(dim=1) / scaffold_denom
        ).mean()

        motif_weight_tensor = token_weight * motif_loss_mask.float()
        motif_denom = motif_weight_tensor.sum(dim=1).clamp_min(1.0)  # (B,)
        motif_loss = (
            (token_loss * motif_weight_tensor).sum(dim=1) / motif_denom
        ).mean()

        split_loss = scaffold_loss + motif_weight * motif_loss
        return self.log_clamp(split_loss, threshold=5.0) * self.cfg.split_loss_weight

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
    ]:
        """Aux metrics for split prediction (scaffold only)."""
        device = batch.trans_t.device
        if batch.remaining_insertions is None:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero, zero, zero

        with torch.no_grad():
            target = batch.remaining_insertions.to(torch.float32)  # (B, P)
            rate = pred.pred_split_rate.clamp_min(0.0)  # (B, P)

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

        return split_event_ce, precision, recall, f1, mae, mae_pos

    def _split_pooled_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
    ) -> torch.Tensor:
        """Pooled MSE loss, in log1p space to reduce dynamic range"""
        target_log = torch.log1p(batch.remaining_total.to(torch.float32))  # (B,)
        pred_log1p = (
            pred.pred_split_pooled_log1p_rate
        )  # (B,) model predicts in log1p space
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aux metrics for deletion prediction (terminal scaffold tokens only)."""
        device = batch.trans_t.device
        if batch.deleted is None:
            zero = torch.tensor(0.0, device=device)
            return zero, zero, zero, zero

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

        return del_event_ce, precision, recall, f1

    def _deletion_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,
        motif_mask: torch.Tensor,  # (B, P)
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
        del_pos_weight = 5.0
        token_weight = torch.ones_like(del_targets)
        token_weight = torch.where(
            del_targets > 0.5,
            torch.full_like(token_weight, float(del_pos_weight)),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cross-entropy on per-token lDDT bins (pLDDT).
        Uses current predicted coords vs. anchor coords to compute lDDT.
        """
        plddt_logits = pred.pred_plddt  # (B, P, num_bins) or None
        if plddt_logits is None:
            zero = torch.tensor(0.0, device=batch.trans_t.device)
            return zero, zero, zero, zero, zero

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
        return loss, loss_ce.detach(), acc, acc_pm1, mae_bins

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

        # Rotation VF loss
        rot_vf_loss = self._rot_vf_loss(
            pred_rotmats=pred.pred_rotmats_1,
            target_rotmats=rot_anchors_pack,
            rotmats_t=batch.rotmats_t,
            t=batch.t,
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
        )
        base_seq_prob_loss = self._seq_prob_loss(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_probs=aatype_anchor_probs_pack,
            t=batch.t,
            mask=valid_mask,
        )
        base_seq_loss = base_seq_token_loss + base_seq_prob_loss
        base_seq_ce = self._seq_ce_metric(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            mask=valid_mask,
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
        del_event_ce, del_event_precision, del_event_recall, del_event_f1 = (
            self._deletion_metrics(
                pred=pred,
                batch=batch,
                mask=valid_mask,
                motif_mask=batch.motif_mask,
            )
        )

        # Confidence prediction losses
        bfactor_loss = self._bfactor_loss(
            pred=pred,
            batch=batch,
            mask=valid_mask,
        )
        plddt_loss, plddt_ce, plddt_bin_acc, plddt_bin_acc_pm1, plddt_bin_mae = (
            self._plddt_loss(
                pred=pred,
                batch=batch,
                target_trans=trans_anchors_pack,
                mask=valid_mask,
            )
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
            insertion_seq_ce=insertion_seq_ce,
            insertion_target_entropy=insertion_target_entropy,
            insertion_ce_over_entropy=insertion_ce_over_entropy,
            insertion_ce_minus_entropy=insertion_ce_minus_entropy,
            split_event_ce=split_event_ce,
            split_event_precision=split_event_precision,
            split_event_recall=split_event_recall,
            split_event_f1=split_event_f1,
            split_rate_mae=split_rate_mae,
            split_rate_mae_pos=split_rate_mae_pos,
            del_event_ce=del_event_ce,
            del_event_precision=del_event_precision,
            del_event_recall=del_event_recall,
            del_event_f1=del_event_f1,
            plddt_ce=plddt_ce,
            plddt_bin_acc=plddt_bin_acc,
            plddt_bin_acc_pm1=plddt_bin_acc_pm1,
            plddt_bin_mae=plddt_bin_mae,
        )
        return losses, metrics


""" Visualization """


@dataclass
class TrajectoryFrame:
    """Visualization-ready frame data, converted to numpy.

    This is a single-sample (no batch dimension) representation suitable for
    plotting. All tensors are numpy arrays on CPU.
    """

    trans: np.ndarray  # (P, 3) CA positions
    rotmats: np.ndarray  # (P, 3, 3) rotation matrices
    aatypes: np.ndarray  # (P,) amino acid indices 0-20
    motif_mask: np.ndarray  # (P,) bool
    valid_mask: np.ndarray  # (P,) bool
    t: float  # timestep value
    remaining_insertions: Optional[np.ndarray] = None  # (P,) or None
    atom37: Optional[np.ndarray] = None  # (P, 37, 3) pre-computed if needed

    def get_backbone_positions(self, only_alpha_carbons: bool) -> np.ndarray:
        """Return backbone positions for valid residues.

        Args:
            only_alpha_carbons: If True, return (n_valid, 3) CA positions.
                If False, return (n_valid, 3, 3) N/CA/C positions.
                Requires atom37 to have been computed at construction time.
        """
        valid = self.valid_mask
        if only_alpha_carbons:
            return self.trans[valid]  # (n_valid, 3)
        else:
            if self.atom37 is None:
                raise ValueError(
                    "atom37 not available. Set include_atom37=True when creating frame."
                )
            # N=0, CA=1, C=2 in atom37 ordering
            return self.atom37[valid][:, [0, 1, 2], :]  # (n_valid, 3, 3)

    @classmethod
    def from_data_corrupted(
        cls,
        data: DataCorrupted,
        batch_idx: int,
        include_atom37: bool = False,
    ) -> "TrajectoryFrame":
        """Extract single batch item from DataCorrupted, convert to numpy.

        Args:
            data: The DataCorrupted containing batched samples.
            batch_idx: Which batch item to extract.
            include_atom37: If True, compute and store atom37 representation.
                This is faster than computing it later since the data is still
                on the original device (potentially GPU).
        """
        atom37 = None
        if include_atom37:
            # Compute atom37 on device before moving to CPU
            atom37_batch = data.to_atom37()  # (B, P, 37, 3)
            atom37 = atom37_batch[batch_idx].cpu().numpy()

        return cls(
            trans=data.trans_t[batch_idx].cpu().numpy(),
            rotmats=data.rotmats_t[batch_idx].cpu().numpy(),
            aatypes=data.aatypes_t[batch_idx].cpu().numpy(),
            motif_mask=data.motif_mask[batch_idx].cpu().numpy(),
            valid_mask=data.valid_mask[batch_idx].cpu().numpy(),
            t=data.t[batch_idx].item(),
            remaining_insertions=(
                data.remaining_insertions[batch_idx].cpu().numpy()
                if data.remaining_insertions is not None
                else None
            ),
            atom37=atom37,
        )

    @classmethod
    def from_model_prediction(
        cls,
        pred: ModelPrediction,
        sample: DataCorrupted,
        batch_idx: int,
        include_atom37: bool = False,
    ) -> "TrajectoryFrame":
        """Extract single batch item from ModelPrediction, convert to numpy.

        Uses the sample for motif_mask, valid_mask, and t since predictions
        are made for a given sample state.

        Args:
            pred: The ModelPrediction containing batched predictions.
            sample: The corresponding DataCorrupted sample (for metadata).
            batch_idx: Which batch item to extract.
            include_atom37: If True, compute and store atom37 representation.
        """
        atom37 = None
        if include_atom37:
            # Compute atom37 from prediction on device before moving to CPU
            atom37_batch = all_atom.atom37_from_trans_rot(
                trans=pred.pred_trans_1,
                rots=pred.pred_rotmats_1,
                torsions=None,
                aatype=pred.pred_aatype_logits.argmax(dim=-1),
                res_mask=sample.valid_mask.float(),
                unknown_to_alanine=True,
            )
            atom37 = atom37_batch[batch_idx].cpu().numpy()

        return cls(
            trans=pred.pred_trans_1[batch_idx].cpu().numpy(),
            rotmats=pred.pred_rotmats_1[batch_idx].cpu().numpy(),
            aatypes=pred.pred_aatype_logits[batch_idx].argmax(dim=-1).cpu().numpy(),
            motif_mask=sample.motif_mask[batch_idx].cpu().numpy(),
            valid_mask=sample.valid_mask[batch_idx].cpu().numpy(),
            t=sample.t[batch_idx].item(),
            remaining_insertions=pred.pred_split_rate[batch_idx].cpu().numpy(),
            atom37=atom37,
        )


@dataclass
class PlotPanel:
    """Manages matplotlib artists for one sequence+structure visualization panel.

    A panel consists of a sequence bar (2D axes) and a 3D structure view.
    This class encapsulates the artists needed to render a TrajectoryFrame.
    """

    ax_seq: plt.Axes
    ax_3d: Any  # Axes3D
    seq_artists: (
        tuple  # (rectangles, texts, motif_rects, letters, colors, positions_per_row)
    )
    scatter_artist: Any  # PathCollection3D
    letter_artists: Optional[List]
    title_prefix: str
    max_atoms: int
    only_alpha_carbons: bool
    color_by: str

    @classmethod
    def create(
        cls,
        ax_seq: plt.Axes,
        ax_3d: Any,
        max_seq_len: int,
        max_atoms: int,
        trans_min: np.ndarray,
        trans_max: np.ndarray,
        only_alpha_carbons: bool,
        color_by: str,
        show_residue_letters: bool,
        title_prefix: str = "",
    ) -> "PlotPanel":
        """Create all artists for this panel."""
        seq_artists = cls._create_sequence_artists(ax_seq, max_seq_len)
        scatter_artist = cls._create_3d_scatter_artist(
            ax_3d, max_atoms, trans_min, trans_max, only_alpha_carbons, color_by
        )
        letter_artists = (
            cls._create_3d_residue_letter_artists(ax_3d, max_seq_len)
            if show_residue_letters
            else None
        )
        return cls(
            ax_seq=ax_seq,
            ax_3d=ax_3d,
            seq_artists=seq_artists,
            scatter_artist=scatter_artist,
            letter_artists=letter_artists,
            title_prefix=title_prefix,
            max_atoms=max_atoms,
            only_alpha_carbons=only_alpha_carbons,
            color_by=color_by,
        )

    def update(self, frame: TrajectoryFrame) -> None:
        """Update all artists with new frame data."""
        rectangles, texts, motif_rects, letters, colors, _ = self.seq_artists
        valid = frame.valid_mask

        # Update sequence bar
        self._update_sequence_bar(
            rectangles,
            texts,
            motif_rects,
            letters,
            colors,
            frame.aatypes[valid],
            frame.motif_mask[valid],
        )

        # Update 3D scatter
        backbone_pos = frame.get_backbone_positions(self.only_alpha_carbons)
        remaining_ins = (
            frame.remaining_insertions[valid]
            if frame.remaining_insertions is not None
            else None
        )
        self._update_3d_scatter(
            self.scatter_artist,
            self.ax_3d,
            backbone_pos,
            frame.motif_mask[valid],
            frame.aatypes[valid],
            self.max_atoms,
            frame.t,
            self.only_alpha_carbons,
            remaining_ins,
            self.color_by,
        )

        # Update title with prefix
        n_res = valid.sum()
        title = f"{self.title_prefix}t = {frame.t:.2f} (N={n_res})"
        self.ax_3d.set_title(title)

        # Update residue letters if enabled
        if self.letter_artists is not None:
            ca_pos = frame.trans[valid]
            self._update_3d_residue_letter_artists(
                self.letter_artists,
                self.ax_3d,
                ca_pos,
                frame.aatypes[valid],
            )

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _aa_letters_and_colors() -> Tuple[Tuple[str, ...], np.ndarray]:
        """Returns (letters, colors) for 21 amino acid types + X. Cached."""
        letters = list(restypes_with_x)
        letters[20] = "-"  # UNK

        def tint(rgb, f):  # mix with white
            return tuple((1 - f) * c + f for c in rgb)

        NEG = (0.22, 0.47, 0.75)  # blue
        POS = (0.84, 0.15, 0.16)  # red
        POL = (0.17, 0.63, 0.17)  # green
        NON = (0.75, 0.25, 0.75)  # purple
        aa_map = {
            # negative
            "D": tint(NEG, 0.2),
            "E": tint(NEG, 0.4),
            # positive
            "K": tint(POS, 0.2),
            "R": tint(POS, 0.4),
            "H": tint(POS, 0.8),
            # polar
            "N": tint(POL, 0.1),
            "Q": tint(POL, 0.4),
            "S": tint(POL, 0.5),
            "T": tint(POL, 0.6),
            "C": tint(POL, 0.3),
            "Y": tint(POL, 0.7),
            "W": tint(POL, 0.2),
            # non-polar
            "A": tint(NON, 0.8),
            "V": tint(NON, 0.2),
            "L": tint(NON, 0.1),
            "I": tint(NON, 0.3),
            "M": tint(NON, 0.4),
            "F": tint(NON, 0.5),
            "P": tint(NON, 0.6),
            "G": tint(NON, 0.9),
            "-": (0.9, 0.9, 0.9),
        }
        colors = np.array([aa_map[ltr] for ltr in letters])
        return tuple(letters), colors

    @staticmethod
    @functools.lru_cache(maxsize=1)
    def _aa_listed_colormap() -> ListedColormap:
        """A categorical colormap for amino acid indices (0..20)."""
        _, colors = PlotPanel._aa_letters_and_colors()
        return ListedColormap(colors, name="aa")

    @staticmethod
    def _create_sequence_artists(
        ax: plt.Axes, max_len: int, positions_per_row: int = 175
    ):
        """Pre-create all artists needed for sequence visualization."""
        letters, colors = PlotPanel._aa_letters_and_colors()
        num_rows = math.ceil(max_len / positions_per_row)

        box_width = 1.0
        box_height = 1.0
        row_spacing = 0.3
        row_height = box_height + row_spacing

        ax.set_xlim(-0.1, positions_per_row + 0.1)
        total_height = num_rows * row_height
        ax.set_ylim(-total_height - 0.1, 0.5)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        rectangles = []
        texts = []
        motif_rects = []

        for i in range(max_len):
            row = i // positions_per_row
            col = i % positions_per_row
            y_base = -row * row_height
            x_pos = col * box_width

            rect = plt.Rectangle(
                (x_pos, y_base - box_height),
                box_width,
                box_height,
                facecolor=colors[0],
                edgecolor="white",
                lw=0.5,
            )
            ax.add_patch(rect)
            rectangles.append(rect)

            text = ax.text(
                x_pos + box_width / 2,
                y_base - box_height / 2,
                letters[0],
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
            texts.append(text)

            motif_rect = plt.Rectangle(
                (x_pos, y_base - box_height - 0.25),
                box_width,
                0.15,
                facecolor="black",
                lw=0,
            )
            ax.add_patch(motif_rect)
            motif_rects.append(motif_rect)

        return rectangles, texts, motif_rects, letters, colors, positions_per_row

    @staticmethod
    def _update_sequence_bar(
        rectangles,
        texts,
        motif_rects,
        letters,
        colors,
        aatypes: np.ndarray,
        motif_mask: np.ndarray,
    ):
        """Update pre-created artists with new sequence data."""
        n = len(aatypes)
        for i in range(len(rectangles)):
            if i < n:
                aa_idx = int(aatypes[i]) if aatypes[i] < len(letters) else 20
                rectangles[i].set_facecolor(colors[aa_idx])
                rectangles[i].set_visible(True)
                texts[i].set_text(letters[aa_idx])
                texts[i].set_visible(True)
                motif_rects[i].set_visible(bool(motif_mask[i]))
            else:
                rectangles[i].set_visible(False)
                texts[i].set_visible(False)
                motif_rects[i].set_visible(False)

    @staticmethod
    def _create_3d_scatter_artist(
        ax: plt.Axes,
        max_atoms: int,
        trans_min: np.ndarray,
        trans_max: np.ndarray,
        only_alpha_carbons: bool = False,
        color_by: Literal["position", "sequence"] = "position",
    ):
        """Pre-create a 3D scatter artist with max_atoms capacity."""
        dummy_pos = np.zeros((max_atoms, 3))
        dummy_colors = np.zeros(max_atoms)
        dummy_sizes = np.ones(max_atoms) * 40.0

        scat = ax.scatter(
            dummy_pos[:, 0],
            dummy_pos[:, 1],
            dummy_pos[:, 2],
            c=dummy_colors,
            cmap=(
                PlotPanel._aa_listed_colormap()
                if color_by == "sequence"
                else "Spectral"
            ),
            vmin=0,
            vmax=1,
            s=dummy_sizes,
            depthshade=True,
            alpha=0.75,
        )

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=25, azim=45)
        ax.set_xlim(trans_min[0], trans_max[0])
        ax.set_ylim(trans_min[1], trans_max[1])
        ax.set_zlim(trans_min[2], trans_max[2])

        return scat

    @staticmethod
    def _create_3d_residue_letter_artists(
        ax: plt.Axes,
        max_len: int,
        fontsize: float = 8.0,
    ):
        """Pre-create per-residue text artists."""
        texts = []
        for _ in range(max_len):
            text = ax.text2D(
                0.0,
                0.0,
                "",
                transform=ax.transData,
                ha="center",
                va="center",
                fontsize=fontsize,
                color="k",
                alpha=0.9,
            )
            text.set_visible(False)
            texts.append(text)
        return texts

    @staticmethod
    def _update_3d_residue_letter_artists(
        texts,
        ax: plt.Axes,
        ca_pos: np.ndarray,
        aatypes: np.ndarray,
    ) -> None:
        """Update pre-created artists with new residue letters/positions."""
        letters, _ = PlotPanel._aa_letters_and_colors()
        n = len(aatypes)

        if n > 0:
            x2, y2, _ = proj3d.proj_transform(
                ca_pos[:, 0],
                ca_pos[:, 1],
                ca_pos[:, 2],
                ax.get_proj(),
            )

        for i in range(len(texts)):
            if i < n:
                aa_idx = int(aatypes[i]) if aatypes[i] < len(letters) else 20
                texts[i].set_text(letters[aa_idx])
                texts[i].set_position((float(x2[i]), float(y2[i])))
                texts[i].set_visible(True)
            else:
                texts[i].set_visible(False)

    @staticmethod
    def _update_3d_scatter(
        scat,
        ax: plt.Axes,
        backbone_pos: np.ndarray,
        motif_alive: np.ndarray,
        aatypes_alive: Optional[np.ndarray],
        max_atoms: int,
        t_val: float,
        only_alpha_carbons: bool = False,
        remaining_insertions_alive: Optional[np.ndarray] = None,
        color_by: Literal["position", "sequence"] = "position",
    ):
        """Update pre-created 3D scatter artist with backbone atoms."""
        n_res = backbone_pos.shape[0] if backbone_pos.size > 0 else 0

        if n_res > 0:
            if only_alpha_carbons:
                n_atoms = n_res
                flat_pos = backbone_pos

                padded_pos = np.zeros((max_atoms, 3))
                padded_pos[:n_atoms] = flat_pos

                color_idx = np.zeros(max_atoms)
                if color_by == "sequence":
                    if aatypes_alive is None:
                        raise ValueError(
                            "aatypes_alive is required for color_by='sequence'"
                        )
                    color_idx[:n_atoms] = aatypes_alive
                else:
                    color_idx[:n_atoms] = np.arange(n_res)

                if remaining_insertions_alive is not None:
                    is_anchor = remaining_insertions_alive > 0
                    base_sizes = np.where(
                        is_anchor,
                        30.0 + 10.0 * remaining_insertions_alive,
                        20.0,
                    )
                else:
                    base_sizes = np.full(n_res, 20.0)
                motif_factor = np.where(motif_alive, 0.6, 1.0)
                sizes = np.zeros(max_atoms)
                sizes[:n_atoms] = base_sizes * motif_factor
            else:
                n_atoms = n_res * 3
                flat_pos = backbone_pos.reshape(-1, 3)

                padded_pos = np.zeros((max_atoms, 3))
                padded_pos[:n_atoms] = flat_pos

                color_idx = np.zeros(max_atoms)
                if color_by == "sequence":
                    if aatypes_alive is None:
                        raise ValueError(
                            "aatypes_alive is required for color_by='sequence'"
                        )
                    color_idx[:n_atoms] = np.repeat(aatypes_alive, 3)
                else:
                    res_colors = np.repeat(np.arange(n_res), 3)
                    color_idx[:n_atoms] = res_colors

                if remaining_insertions_alive is not None:
                    is_anchor = remaining_insertions_alive > 0
                    ca_sizes = np.where(
                        is_anchor,
                        30.0 + 10.0 * remaining_insertions_alive,
                        20.0,
                    )
                else:
                    ca_sizes = np.full(n_res, 20.0)
                base_sizes = np.zeros(n_atoms)
                base_sizes[0::3] = 10.0
                base_sizes[1::3] = ca_sizes
                base_sizes[2::3] = 10.0

                motif_expanded = np.repeat(motif_alive, 3)
                motif_factor = np.where(motif_expanded, 0.6, 1.0)
                sizes = np.zeros(max_atoms)
                sizes[:n_atoms] = base_sizes * motif_factor

            scat._offsets3d = (padded_pos[:, 0], padded_pos[:, 1], padded_pos[:, 2])
            scat.set_array(color_idx)
            if color_by == "sequence":
                scat.set_cmap(PlotPanel._aa_listed_colormap())
                _, colors = PlotPanel._aa_letters_and_colors()
                scat.set_clim(-0.5, float(colors.shape[0] - 1) + 0.5)
            else:
                scat.set_cmap("Spectral")
                scat.set_clim(0, max(n_res - 1, 1))
            scat.set_sizes(sizes)
        else:
            scat.set_sizes(np.zeros(max_atoms))

        ax.set_title(f"t = {t_val:.2f} (N={n_res})")


class BranchingFlowVisualizer:
    def __init__(
        self,
        sigma: Optional[float] = 0.0,
    ):
        self.interpolant = TreeInterpolant(
            cfg=VarcoInterpolantConfig(
                trans_coupler=VarcoInterpolantTransCouplerConfig(noise_scale=sigma),
                aatypes_coupler=VarcoInterpolantAATypesCouplerConfig(drift_temp=sigma),
            ),
        )

    @staticmethod
    def _get_anim_writer() -> Tuple[str, animation.AbstractMovieWriter]:
        if animation.writers.is_available("ffmpeg"):
            return "mp4", animation.FFMpegWriter(
                fps=10,
                codec="libx264",
                extra_args=[
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                ],
            )
        if animation.writers.is_available("imagemagick"):
            return "gif", animation.ImageMagickWriter(fps=10)
        return "gif", animation.PillowWriter(fps=10)

    def _plot_trajectory_frames(
        self,
        frame_lists: List[List[TrajectoryFrame]],
        panel_titles: List[str],
        out_dir: str,
        filename: str,
        only_alpha_carbons: bool,
        show_residue_letters: bool,
        color_by: str,
        max_cols: int = 2,
    ) -> str:
        """Shared implementation for plotting trajectory frames.

        Args:
            frame_lists: List of frame lists, one per panel column.
                Each inner list contains TrajectoryFrames for that column.
                Caller is responsible for downsampling before calling.
            panel_titles: Title prefix for each panel column (e.g., "Sample: ", "Prediction: ").
            out_dir: Output directory for the animation.
            filename: Base filename (without extension).
            only_alpha_carbons: If True, show only CA atoms.
            show_residue_letters: If True, overlay AA letters on 3D view.
            color_by: "position" or "sequence".
            max_cols: Maximum columns in the grid.

        Returns:
            Path to the saved animation file.
        """
        num_panels = len(frame_lists)
        num_frames = len(frame_lists[0])

        # Validate frame lists have same length
        for i, frames in enumerate(frame_lists):
            if len(frames) != num_frames:
                raise ValueError(
                    f"Frame list {i} has {len(frames)} frames, expected {num_frames}"
                )

        # Compute global limits and max sequence length across all frames in all lists
        trans_min = np.full(3, np.inf)
        trans_max = np.full(3, -np.inf)
        max_seq_len = 0
        for frames in frame_lists:
            for frame in frames:
                valid_trans = frame.trans[frame.valid_mask]
                if valid_trans.shape[0] > 0:
                    trans_min = np.minimum(trans_min, valid_trans.min(axis=0))
                    trans_max = np.maximum(trans_max, valid_trans.max(axis=0))
                max_seq_len = max(max_seq_len, frame.valid_mask.sum())

        # Setup figure layout
        num_cols = min(num_panels, max_cols)
        num_rows = math.ceil(num_panels / num_cols)

        fig = plt.figure(figsize=(10 * num_cols, 12 * num_rows))
        gs = fig.add_gridspec(
            num_rows * 2,
            num_cols,
            height_ratios=[2, 10] * num_rows,
            hspace=0.02,
            wspace=0.05,
        )
        fig.subplots_adjust(
            left=0.03, right=0.97, bottom=0.03, top=0.95, wspace=0.05, hspace=0.05
        )

        # Create panels
        max_atoms = max_seq_len if only_alpha_carbons else max_seq_len * 3
        panels: List[PlotPanel] = []
        for i in range(num_panels):
            row, col = divmod(i, num_cols)
            ax_seq = fig.add_subplot(gs[row * 2, col])
            ax_3d = fig.add_subplot(gs[row * 2 + 1, col], projection="3d")

            panel = PlotPanel.create(
                ax_seq=ax_seq,
                ax_3d=ax_3d,
                max_seq_len=max_seq_len,
                max_atoms=max_atoms,
                trans_min=trans_min,
                trans_max=trans_max,
                only_alpha_carbons=only_alpha_carbons,
                color_by=color_by,
                show_residue_letters=show_residue_letters,
                title_prefix=panel_titles[i] if i < len(panel_titles) else "",
            )
            panels.append(panel)

        # Save animation
        ext, writer = self._get_anim_writer()
        anim_path = os.path.join(out_dir, f"{filename}.{ext}")
        logger.info(f"💾 Saving trajectory animation to {anim_path}")

        with writer.saving(fig, anim_path, dpi=100):
            for frame_idx in tqdm(
                range(num_frames), desc="trajectory frames", leave=False
            ):
                for panel_idx, panel in enumerate(panels):
                    frame = frame_lists[panel_idx][frame_idx]
                    panel.update(frame)
                writer.grab_frame()

        plt.close(fig)
        return anim_path

    def plot_trajectory(
        self,
        traj: Trajectory,
        out_dir: Optional[str] = None,
        filename: str = "trajectory",
        max_frames: Optional[int] = 50,
        max_samples: int = 2,
        max_cols: int = 2,
        only_alpha_carbons: bool = True,
        show_residue_letters: bool = True,
        color_by: Literal["auto", "position", "sequence"] = "auto",
    ) -> str:
        """Plot a trajectory animation showing multiple batch samples.

        Args:
            traj: Trajectory containing samples to plot.
            out_dir: Output directory (defaults to temp directory).
            filename: Base filename for the animation.
            max_frames: Maximum frames to render (downsamples if exceeded).
            max_samples: Maximum batch samples to show.
            max_cols: Maximum columns in the grid.
            only_alpha_carbons: If True, show only CA atoms (faster).
            show_residue_letters: If True, overlay AA letters on 3D view.
            color_by: 'auto' (infer), 'position' (chain index), or 'sequence' (aatype).

        Returns:
            Path to the saved animation file.
        """
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        os.makedirs(out_dir, exist_ok=True)

        if not traj.samples:
            raise ValueError("Trajectory has no samples to plot")

        if color_by not in {"auto", "position", "sequence"}:
            raise ValueError(
                f"Invalid color_by={color_by!r}; expected 'auto', 'position', or 'sequence'"
            )
        if color_by == "auto":
            color_by = "sequence" if show_residue_letters else "position"

        num_batch = traj.samples[0].trans_t.shape[0]
        num_plots = min(num_batch, max_samples)
        num_total_frames = len(traj.samples)

        # Compute frame indices BEFORE converting to TrajectoryFrame
        if max_frames is not None and num_total_frames > max_frames:
            sample_indices = np.linspace(0, num_total_frames - 1, max_frames, dtype=int)
        else:
            sample_indices = np.arange(num_total_frames)

        # Convert only the needed frames to TrajectoryFrames
        frame_lists: List[List[TrajectoryFrame]] = []
        for batch_idx in range(num_plots):
            frames = [
                TrajectoryFrame.from_data_corrupted(
                    data=traj.samples[idx],
                    batch_idx=batch_idx,
                    include_atom37=not only_alpha_carbons,
                )
                for idx in sample_indices
            ]
            frame_lists.append(frames)

        return self._plot_trajectory_frames(
            frame_lists=frame_lists,
            panel_titles=[""] * num_plots,
            out_dir=out_dir,
            filename=filename,
            only_alpha_carbons=only_alpha_carbons,
            show_residue_letters=show_residue_letters,
            color_by=color_by,
            max_cols=max_cols,
        )

    def plot_sampling_trajectory(
        self,
        traj: SampleTrajectory,
        batch_idx: int = 0,
        out_dir: Optional[str] = None,
        filename: str = "sampling_trajectory",
        max_frames: Optional[int] = 50,
        only_alpha_carbons: bool = True,
        show_residue_letters: bool = True,
        color_by: Literal["auto", "position", "sequence"] = "auto",
    ) -> str:
        """Plot sample and model prediction side-by-side for one batch item.

        Args:
            traj: SampleTrajectory containing samples and predictions.
            batch_idx: Which batch item to visualize.
            out_dir: Output directory (defaults to temp directory).
            filename: Base filename for the animation.
            max_frames: Maximum frames to render (downsamples if exceeded).
            only_alpha_carbons: If True, show only CA atoms (faster).
            show_residue_letters: If True, overlay AA letters on 3D view.
            color_by: 'auto' (infer), 'position' (chain index), or 'sequence' (aatype).

        Returns:
            Path to the saved animation file.
        """
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        os.makedirs(out_dir, exist_ok=True)

        if not traj.samples:
            raise ValueError("Trajectory has no samples to plot")
        if not traj.pred:
            raise ValueError("SampleTrajectory has no predictions to plot")

        if color_by not in {"auto", "position", "sequence"}:
            raise ValueError(
                f"Invalid color_by={color_by!r}; expected 'auto', 'position', or 'sequence'"
            )
        if color_by == "auto":
            color_by = "sequence" if show_residue_letters else "position"

        num_total_frames = len(traj.samples)

        # Compute frame indices BEFORE converting to TrajectoryFrame
        if max_frames is not None and num_total_frames > max_frames:
            sample_indices = np.linspace(0, num_total_frames - 1, max_frames, dtype=int)
        else:
            sample_indices = np.arange(num_total_frames)

        # Convert only the needed samples to TrajectoryFrames
        sample_frames = [
            TrajectoryFrame.from_data_corrupted(
                data=traj.samples[idx],
                batch_idx=batch_idx,
                include_atom37=not only_alpha_carbons,
            )
            for idx in sample_indices
        ]

        # Convert predictions (clamping index to available predictions)
        num_preds = len(traj.pred)
        pred_frames = [
            TrajectoryFrame.from_model_prediction(
                pred=traj.pred[min(idx, num_preds - 1)],
                sample=traj.samples[idx],
                batch_idx=batch_idx,
                include_atom37=not only_alpha_carbons,
            )
            for idx in sample_indices
        ]

        return self._plot_trajectory_frames(
            frame_lists=[sample_frames, pred_frames],
            panel_titles=["Sample: ", "Prediction: "],
            out_dir=out_dir,
            filename=filename,
            only_alpha_carbons=only_alpha_carbons,
            show_residue_letters=show_residue_letters,
            color_by=color_by,
            max_cols=2,
        )

    def visualize_corruption(
        self,
        batch: DataBatch,
        out_dir: Optional[str] = None,
        times: Optional[List[float]] = None,
        only_alpha_carbons: bool = False,  # faster; skips to_atom37
        filename: str = "corruption",
        coupled: bool = True,
        seed: Optional[int] = None,
    ) -> str:
        """Create a corruption trajectory and plot it.

        If coupled=True, generates a time-coupled trajectory from a single sampled set of
        domain couplings (anchors + creation states), rather than sampling each timepoint
        marginal independently.
        """
        self.interpolant.set_device(batch.trans_1.device)
        self.interpolant.seed_all(seed)
        if times is None:
            times = list(np.linspace(0.0, 1.0, 50))
        times = sorted(times)

        num_batch = batch.trans_1.shape[0]
        device = batch.trans_1.device
        tree = batch.tree.to(device)
        min_t = float(self.interpolant.min_t)
        times = [float(np.clip(t, min_t, 1.0 - min_t)) for t in times]

        # Define consistent base samples for the whole trajectory (in aligned space)
        trans_0 = self.interpolant.translation_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=tree.broadcast_to_leaves(batch.trans_1.to(device), fill_value=0),
            device=device,
        )
        rotmats_0 = self.interpolant.rotation_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=tree.broadcast_to_leaves(
                batch.rotmats_1.to(device), fill_value=torch.eye(3, device=device)
            ),
            device=device,
        )
        aatypes_0 = self.interpolant.aatypes_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=tree.broadcast_to_leaves(
                batch.aatypes_1.to(device), fill_value=MASK_TOKEN_INDEX
            ),
            device=device,
        )

        traj, _ = self.interpolant.corrupt_trajectory(
            batch=batch,
            times=times,
            trans_0=trans_0,
            rotmats_0=rotmats_0,
            aatypes_0=aatypes_0,
        )
        return self.plot_trajectory(
            traj=traj,
            out_dir=out_dir,
            filename=filename,
            only_alpha_carbons=only_alpha_carbons,
        )


""" Module """


class BranchFlowModule(pl.LightningModule):
    def __init__(self, cfg: VarcoConfig):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters("cfg")
        self._log = rank_zero_logger(__name__)

        self.model = BranchFlowModel(cfg=self.cfg.model)
        self.loss_calculator = BranchFlowLossCalculator(cfg=self.cfg.loss)
        self.interpolant = TreeInterpolant(cfg=self.cfg.interpolant)

        # track EMA of training loss, hacky init at 10 to avoid nan handling
        self.register_buffer("_train_loss_ema", torch.tensor(10.0), persistent=False)

        self._folding_validator: Optional[FoldingValidator] = None
        self._val_top_samples: List[Dict[str, Any]] = []
        self._predict_top_samples: List[Dict[str, Any]] = []

    def _get_folding_validator(self) -> FoldingValidator:
        if self._folding_validator is None:
            self._folding_validator = FoldingValidator(
                cfg=self.cfg.folding, device="cpu"
            )
        return self._folding_validator

    def _gather_top_samples(
        self, local_top_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not dist.is_available() or not dist.is_initialized():
            return local_top_samples

        gathered: List[Optional[List[Dict[str, Any]]]] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, local_top_samples)

        merged: List[Dict[str, Any]] = []
        for part in gathered:
            if part:
                merged.extend(part)
        return merged

    def on_validation_start(self) -> None:
        if self.cfg.inference.folding_validation.enabled:
            self._get_folding_validator().set_device_id(str(self.device))

    def on_predict_start(self) -> None:
        if self.cfg.inference.folding_validation.enabled:
            self._get_folding_validator().set_device_id(str(self.device))

    def on_validation_epoch_start(self) -> None:
        self._val_top_samples = []

    def on_predict_epoch_start(self) -> None:
        self._predict_top_samples = []

    def load_state_dict(self, state_dict, strict: bool = True):
        if strict:
            plan = plan_esm_warm_start_state_dict_load(
                checkpoint_state_dict=state_dict,
                current_state_dict=self.state_dict(),
                strict=True,
                allow_missing_esm_combiner_pair=True,
            )
            for note in plan.notes:
                self._log.info(note)
            strict = plan.strict

        return super().load_state_dict(state_dict, strict=strict)

    def load_cogeneration_weights(self, ckpt_path: str):
        """
        Load compatible weights from a cogeneration checkpoint.
        Copies specified modules if shapes match. Fails on shape mismatch.
        """
        logger.info(f"⚡️ Loading cogeneration weights from: {ckpt_path}")

        cogen_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        varco_state = self.state_dict()

        # Modules to copy: cogen prefix -> varco prefix(es)
        module_map = {
            "model.esm_combiner.": "model.esm_combiner.",
            "model.edge_feature_net.": "model.edge_feature_net.",
            "model.trunk.": "model.trunk.",
            "model.ipa_trunk.": "model.ipa_trunk.",
            "model.seq_trunk.": "model.seq_trunk.",
            "model.aa_pred_net.": "model.aatype_pred.",
            "model.bfactor_net.": "model.bfactor_net.",
            "model.plddt_net.": "model.plddt_net.",
        }

        mapped = {}
        for cogen_key, value in cogen_state.items():
            # Skip frozen ESM (loaded separately)
            if ".esm_combiner.esm." in cogen_key:
                continue
            for cogen_prefix, varco_prefix in module_map.items():
                if cogen_key.startswith(cogen_prefix):
                    suffix = cogen_key[len(cogen_prefix) :]
                    targets = (
                        [varco_prefix]
                        if isinstance(varco_prefix, str)
                        else varco_prefix
                    )
                    for t in targets:
                        varco_key = t + suffix
                        if varco_key in varco_state:
                            if varco_state[varco_key].shape != value.shape:
                                raise ValueError(
                                    f"Shape mismatch: {cogen_key} {value.shape} vs {varco_key} {varco_state[varco_key].shape}"
                                )
                            mapped[varco_key] = value
                    break

        self.load_state_dict(mapped, strict=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self.cfg.experiment.optimizer.asdict(),
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Override to time"""
        backward_start = time.perf_counter()
        optimizer.step(closure=optimizer_closure)
        backward_time = time.perf_counter() - backward_start
        self.log("t/backward_ms", backward_time * 1000)

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        return self.model(batch)

    def training_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        self.interpolant.set_device(self.device)

        # Time corruption (CPU time, no synchronization needed)
        corrupt_start = time.perf_counter()
        bridged, couplings = self.interpolant.corrupt_batch(batch=batch)
        corrupt_time = time.perf_counter() - corrupt_start

        corrupted = bridged.pack_present()

        # Time forward pass
        forward_start = time.perf_counter()
        pred = self.forward(corrupted)
        forward_time = time.perf_counter() - forward_start

        # Time loss calculation (part of forward)
        loss_start = time.perf_counter()
        losses, metrics = self.loss_calculator.calculate(
            batch=corrupted, pred=pred, couplings=couplings, bridged=bridged
        )
        loss_time = time.perf_counter() - loss_start

        # bin t value (bucketed into 0.0-0.2, 0.2-0.4, ..., 0.8-1.0)
        mean_t = bridged.t.mean().detach().item()
        t_bin_idx = min(
            int(mean_t * 5), 4
        )  # 0-4 for bins [0.0-0.2), [0.2-0.4), ..., [0.8-1.0]
        t_bin_start = t_bin_idx * 0.2
        t_bin_end = t_bin_start + 0.2
        t_bin_key = f"{t_bin_start:.1f}-{t_bin_end:.1f}"

        # primary losses
        self.log("L/train", losses.total_loss, prog_bar=True)
        self.log("L/trans", losses.trans_loss)
        self.log("L/rot", losses.rot_vf_loss)
        self.log("L/cdist", losses.pairwise_loss)
        self.log("L/seq", losses.base_seq_loss)
        self.log("L/seq_prob", losses.base_seq_prob_loss)
        self.log("L/seq_tok", losses.base_seq_token_loss)
        self.log("L/seq_ins", losses.insertion_seq_loss)
        self.log("L/split", losses.split_token_loss)
        self.log("L/split_pooled", losses.split_pooled_loss)
        self.log("L/del", losses.del_loss)
        self.log("L/bfactor", losses.bfactor_loss)
        self.log("L/plddt", losses.plddt_loss)

        # EMA of training loss
        train_loss = losses.total_loss.detach().item()
        beta = float(self.cfg.experiment.train_loss_ema_beta)
        self._train_loss_ema = beta * self._train_loss_ema + (1.0 - beta) * train_loss
        self.log("L/train_ema", self._train_loss_ema, prog_bar=True)

        # aux metrics
        self.log("A/base_seq_ce", metrics.base_seq_ce)
        self.log("A/insertion_seq_ce", metrics.insertion_seq_ce)
        self.log("A/insertion_target_entropy", metrics.insertion_target_entropy)
        self.log("A/insertion_ce_over_entropy", metrics.insertion_ce_over_entropy)
        self.log("A/insertion_ce_minus_entropy", metrics.insertion_ce_minus_entropy)
        self.log("A/split_event_ce", metrics.split_event_ce)
        self.log("A/split_event_precision", metrics.split_event_precision)
        self.log("A/split_event_recall", metrics.split_event_recall)
        self.log("A/split_event_f1", metrics.split_event_f1)
        self.log("A/split_rate_mae", metrics.split_rate_mae)
        self.log("A/split_rate_mae_pos", metrics.split_rate_mae_pos)
        self.log("A/del_event_ce", metrics.del_event_ce)
        self.log("A/del_event_precision", metrics.del_event_precision)
        self.log("A/del_event_recall", metrics.del_event_recall)
        self.log("A/del_event_f1", metrics.del_event_f1)
        self.log("A/plddt_ce", metrics.plddt_ce)
        self.log("A/plddt_bin_acc", metrics.plddt_bin_acc)
        self.log("A/plddt_bin_acc_pm1", metrics.plddt_bin_acc_pm1)
        self.log("A/plddt_bin_mae", metrics.plddt_bin_mae)

        # t-stratified losses for primary losses
        self.log(f"L_t/trans_t{t_bin_key}", losses.trans_loss)
        self.log(f"L_t/rot_t{t_bin_key}", losses.rot_vf_loss)
        self.log(f"L_t/seq_t{t_bin_key}", losses.base_seq_loss)

        # Timing statistics
        batch_size = corrupted.trans_t.shape[0]
        self.log("t/t", bridged.t.mean())
        self.log("t/batch_size", float(batch_size))
        # skip startup noise
        if batch_idx > 3:
            self.log("t/forward", forward_time * 1000)
            self.log("t/loss", loss_time * 1000)
        # Corruption time as function of batch size / t
        self.log("t/corrupt_ms_per_batch", corrupt_time * 1000 / batch_size)
        self.log(f"t/corrupt_ms_t{t_bin_key}", corrupt_time * 1000)

        # MPS clean up
        if batch_idx % 100 == 0 and torch.backends.mps.is_available():
            alloc = torch.mps.current_allocated_memory() / 1e9
            drv = torch.mps.driver_allocated_memory() / 1e9
            logger.debug(f"step {batch_idx} mps alloc={alloc:.2f}GB driver={drv:.2f}GB")
            gc.collect()
            torch.mps.empty_cache()

        return losses.total_loss

    def validation_step(self, batch: DataBatch, batch_idx: int) -> None:
        self.interpolant.set_device(self.device)
        sample_name = f"val_sample_{batch_idx}"

        sample_traj = self.interpolant.sample(
            model=self.model,
            data=batch,
        )

        viz = BranchingFlowVisualizer(sigma=1.0)
        val_dir = os.path.join(
            self.cfg.inference.predict_dir, "val", f"epoch{self.current_epoch:03d}"
        )
        viz.plot_sampling_trajectory(
            traj=sample_traj,
            out_dir=val_dir,
            filename=sample_name,
        )

        # run folding validation (refold/designability)for max_batches samples
        fold_val_cfg = self.cfg.inference.folding_validation
        if (
            fold_val_cfg.enabled
            and batch_idx < fold_val_cfg.max_batches
            and DDPInfo.from_env().local_rank == 0
        ):
            sample_dir = os.path.join(val_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            final_sample = sample_traj.samples[-1]
            pred_atom37 = final_sample.to_atom37()[0].detach().cpu().numpy()
            pred_aa = final_sample.aatypes_t[0].detach().cpu().numpy()
            chain_idx = final_sample.chain_idx[0].detach().cpu().numpy()
            res_idx = np.arange(pred_aa.shape[0], dtype=np.int32)

            pred_pdb_path = os.path.join(sample_dir, OutputFileName.sample_pdb)
            write_prot_to_pdb(
                prot_pos=pred_atom37,
                file_path=pred_pdb_path,
                aatype=pred_aa,
                chain_idx=chain_idx,
                res_idx=res_idx,
                no_indexing=True,
                overwrite=True,
            )

            top_sample_metrics, _ = self._get_folding_validator().assess_sample(
                task=InferenceTask.unconditional,
                sample_name=sample_name,
                sample_dir=sample_dir,
                pred_pdb_path=pred_pdb_path,
                pred_bb_positions=pred_atom37,
                pred_aa=pred_aa,
                sample_aa_traj=np.expand_dims(pred_aa, axis=0),
                diffuse_mask=np.ones_like(pred_aa, dtype=np.int8),
                motif_mask=None,
                chain_idx=chain_idx,
                res_idx=res_idx,
                true_bb_positions=None,
                true_aa=None,
                inverse_fold=fold_val_cfg.assess_designability,
                also_fold_pmpnn_seq=fold_val_cfg.assess_designability,
                n_inverse_folds=self.cfg.folding.protein_mpnn.seq_per_sample,
            )

            self._val_top_samples.append(top_sample_metrics)
            if MetricName.plddt_mean in top_sample_metrics:
                self.log("val/plddt_mean", top_sample_metrics[MetricName.plddt_mean])
            if MetricName.bb_rmsd_folded in top_sample_metrics:
                self.log("val/bb_rmsd", top_sample_metrics[MetricName.bb_rmsd_folded])

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0):
        self.interpolant.set_device(self.device)

        sample_traj = self.interpolant.sample(
            model=self.model,
            data=batch,
        )

        rank = DDPInfo.from_env().rank
        sample_name = f"predict_rank{rank:03d}_idx{batch_idx:05d}"

        sample_dir = os.path.join(
            self.cfg.inference.predict_dir,
            self.cfg.inference.inference_subdir,
            sample_name,
        )
        os.makedirs(sample_dir, exist_ok=True)

        # write PDB
        final_sample = sample_traj.samples[-1]
        pred_atom37 = final_sample.to_atom37()[0].detach().cpu().numpy()
        pred_aa = final_sample.aatypes_t[0].detach().cpu().numpy()
        chain_idx = final_sample.chain_idx[0].detach().cpu().numpy()
        res_idx = np.arange(pred_aa.shape[0], dtype=np.int32)

        # write predicted PDB file
        pred_pdb_path = os.path.join(sample_dir, OutputFileName.sample_pdb)
        write_prot_to_pdb(
            prot_pos=pred_atom37,
            file_path=pred_pdb_path,
            aatype=pred_aa,
            chain_idx=chain_idx,
            res_idx=res_idx,
            no_indexing=True,
            overwrite=True,
        )

        # plot trajectory
        if self.cfg.inference.plot.enabled:
            viz = BranchingFlowVisualizer()
            viz.plot_sampling_trajectory(
                traj=sample_traj,
                out_dir=sample_dir,
                filename="sampling_trajectory",
                max_frames=self.cfg.inference.plot.max_frames,
                max_samples=self.cfg.inference.plot.max_samples,
                max_cols=self.cfg.inference.plot.max_cols,
                only_alpha_carbons=self.cfg.inference.plot.only_alpha_carbons,
                show_residue_letters=self.cfg.inference.plot.show_residue_letters,
                color_by=self.cfg.inference.plot.color_by,
            )

        # folding validation - either refolding, or designability
        fold_val_cfg = self.cfg.inference.folding_validation
        if fold_val_cfg.enabled and batch_idx < fold_val_cfg.max_batches:
            top_sample_metrics, _ = self._get_folding_validator().assess_sample(
                task=InferenceTask.unconditional,
                sample_name=sample_name,
                sample_dir=sample_dir,
                pred_pdb_path=pred_pdb_path,
                pred_bb_positions=pred_atom37,
                pred_aa=pred_aa,
                sample_aa_traj=np.expand_dims(pred_aa, axis=0),
                diffuse_mask=np.ones_like(pred_aa, dtype=np.int8),
                motif_mask=None,
                chain_idx=chain_idx,
                res_idx=res_idx,
                true_bb_positions=None,
                true_aa=None,
                inverse_fold=fold_val_cfg.assess_designability,
                also_fold_pmpnn_seq=fold_val_cfg.assess_designability,
                n_inverse_folds=self.cfg.folding.protein_mpnn.seq_per_sample,
            )
            self._predict_top_samples.append(top_sample_metrics)

    def on_validation_epoch_end(self) -> None:
        if not self.cfg.inference.folding_validation.enabled:
            return
        if DDPInfo.from_env().local_rank != 0:
            return

        all_top_samples = self._gather_top_samples(self._val_top_samples)
        if len(all_top_samples) == 0:
            return

        val_dir = os.path.join(
            self.cfg.inference.predict_dir, "val", f"epoch{self.current_epoch:03d}"
        )
        os.makedirs(val_dir, exist_ok=True)
        pd.DataFrame(all_top_samples).to_csv(
            os.path.join(val_dir, OutputFileName.all_top_samples_df), index=False
        )

    def on_predict_epoch_end(self) -> None:
        if not self.cfg.inference.folding_validation.enabled:
            return
        if DDPInfo.from_env().local_rank != 0:
            return

        all_top_samples = self._gather_top_samples(self._predict_top_samples)
        if len(all_top_samples) == 0:
            return

        predict_dir = os.path.join(self.cfg.inference.predict_dir, "predict")
        os.makedirs(predict_dir, exist_ok=True)
        pd.DataFrame(all_top_samples).to_csv(
            os.path.join(predict_dir, OutputFileName.all_top_samples_df), index=False
        )


""" Training """


torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

# Enable memory-efficient attention backends in PyTorch when available
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


class Experiment:
    def __init__(self, cfg: VarcoConfig):
        self.cfg = cfg
        self.logger = rank_zero_logger(__name__)
        self._warm_start_ckpt_path: Optional[str] = None

    def setup(self):
        pl.seed_everything(self.cfg.shared.seed, workers=True)

        # Support warm starts / resume from a varco Lightning checkpoint.
        if self.cfg.experiment.warm_start_ckpt is not None:
            ckpt_path = str(self.cfg.experiment.warm_start_ckpt)
            if not os.path.isabs(ckpt_path):
                ckpt_path = str(Path(self.cfg.shared.project_root) / ckpt_path)
            if not ckpt_path.endswith(".ckpt"):
                raise ValueError(
                    f"Invalid warm start checkpoint path {ckpt_path!r}; expected .ckpt"
                )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Warm start checkpoint {ckpt_path!r} does not exist."
                )
            self._warm_start_ckpt_path = ckpt_path
            self.logger.info(f"🚩 Warm starting from {ckpt_path}")

        # Handle DDP set up in case pytorch lightning doesn't handle it
        # (e.g. on mac laptop)
        setup_ddp(
            trainer_strategy=self.cfg.experiment.trainer.strategy,
            accelerator=self.cfg.experiment.trainer.accelerator,
            rank=str(DDPInfo.from_env().rank),
            world_size=str(self.cfg.experiment.num_devices),
        )

        # Create output directory for this run
        predict_dir = self.cfg.inference.predict_dir
        os.makedirs(predict_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {predict_dir}")

        self.data_module = ProteinDataModule(
            dataset=ProteinDataset(cfg=self.cfg.dataset),
            cfg=self.cfg.data,
        )

        self.module = BranchFlowModule(cfg=self.cfg)

        # Load cogeneration weights if specified
        if self.cfg.experiment.cogen_ckpt_path:
            if self._warm_start_ckpt_path is not None:
                raise ValueError(
                    "`cfg.experiment.cogen_ckpt_path` cannot be used together with "
                    "`cfg.experiment.warm_start_ckpt`."
                )
            self.module.load_cogeneration_weights(self.cfg.experiment.cogen_ckpt_path)

        self.logger.info("\n" + str(ModelSummary(self.module, max_depth=2)))

    def debug(self, n: int = 10):
        if n <= 0:
            return

        debug_dir = os.path.join(self.cfg.inference.predict_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Plot corruption planning trees
        for i in range(n):
            datum = self.data_module.dataset[i]
            datum.tree_plan.plot(
                path=os.path.join(debug_dir, f"init_tree_plan_{i}.png")
            )

        # Visualize corruption processes
        # 0 for deterministic bridges, >0 for stochasticity
        viz = BranchingFlowVisualizer(sigma=0.0)
        data_loader = self.data_module.train_dataloader(rank=0, num_replicas=1)
        for i, debug_batch in enumerate(data_loader):
            if i >= n:
                break
            viz.visualize_corruption(
                batch=debug_batch, out_dir=debug_dir, filename=f"debug_corruption_{i}"
            )

    def save_config(self, wandb_logger: WandbLogger):
        # Save config if main process
        local_rank = DDPInfo.from_env().local_rank
        if local_rank == 0:
            # write locally
            ckpt_dir = self.cfg.experiment.checkpointer.dirpath
            self.logger.info(
                f"💾 Checkpoints, config, validations etc. will be saved to: {ckpt_dir}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

            # write to w&b
            if wandb_logger is not None and isinstance(wandb_logger, WandbLogger):
                wandb_logger.experiment.config.update(self.cfg.flatdict())

    def train(self):
        # Set up w&b logging
        wandb_logger = WandbLogger(**self.cfg.experiment.wandb.asdict())

        callbacks = []

        # Simple progress bar
        # callbacks.append(TQDMProgressBar(refresh_rate=1))

        # Model checkpoints
        monitor = "L/train_ema"
        checkpoint_cfg = self.cfg.experiment.checkpointer.asdict()
        checkpoint_cfg["monitor"] = monitor
        callbacks.append(ModelCheckpoint(**checkpoint_cfg))

        # Save every n training steps
        # TODO - clean up, use cfg explicitly
        n_step_cfg = copy.deepcopy(checkpoint_cfg)
        del n_step_cfg["every_n_epochs"]
        n_step_cfg["every_n_train_steps"] = 2000
        n_step_cfg["monitor"] = monitor
        callbacks.append(ModelCheckpoint(**n_step_cfg))

        self.save_config(wandb_logger=wandb_logger)

        self.logger.info("Setting up Trainer...")
        trainer = Trainer(
            **self.cfg.experiment.trainer.asdict(),
            callbacks=callbacks,
            logger=wandb_logger,
            use_distributed_sampler=False,  # TODO - ddp
            devices=self.cfg.experiment.num_devices,
            enable_model_summary=False,  # manual model summary
            enable_progress_bar=True,
        )

        trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=self._warm_start_ckpt_path,
        )

        self.logger.info(f"Training complete")
        self.logger.info(f"💾 ckpt saved to {self.cfg.experiment.checkpointer.dirpath}")
        self.logger.info(f"🏆 Best checkpoint monitor: {monitor}")

        return self.module


@hydra.main(config_path=".", config_name="varco", version_base=None)
def main(cfg: VarcoConfig):
    cfg = OmegaConf.to_object(cfg)
    cfg = cfg.interpolate()

    experiment = Experiment(cfg=cfg)
    experiment.setup()
    experiment.debug(n=1)
    experiment.train()


if __name__ == "__main__":
    main()
