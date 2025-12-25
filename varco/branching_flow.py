"""
Simple implementation of branching flows for protein scaffolding.

Sample some proteins of length N (translations, later sequence and rotations) using LengthBatched protein dataset
Sample a motif_mask, with 1+ scaffolds
Sample some number of roots per scaffold, starting points at t=0 for the motifs + roots
Define a tree, with splits (anchors at intermediate time points) and deletions over trajectory

Sample some intermediate time t
Corrupt to X_t, using stochastic bridge from 0 to each anchor, continuing up to time t

A simple model predicts base (endpoint prediction), split (remaining children count), and deletion (destined to delete probability)

Then, a sampler (no tree) iterates to get base (endpoint prediction), sample split events, sample deletion events

Fix backlog:

TODOs / features:
- improve sampling
  - motif guidance for positions
  - add validation loss (e.g. folding validation?)
  - cap sampling length to avoid overloading gpu memory
- support rotations with an IGSO(3) bridge
- sequence: also predict insertion logits, which are sampled on insertions, and replace the parent's residue
- improve visualizations
  - sequence needs to be easier to read
  - speed
- tree
  - bias splits earlier, deletions later
  - allow more roots than leaves in a scaffold, learn to delete them
"""

import datetime
import functools
import gc
import math
import os
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, List, Optional, Tuple, TypeVar

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from sklearn import datasets
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cogeneration.config.base import Config, DataConfig
from cogeneration.data.const import (
    ANG_TO_NM_SCALE,
    MASK_TOKEN_INDEX,
    NM_TO_ANG_SCALE,
    NUM_TOKENS,
)
from cogeneration.data.noise_mask import centered_gaussian, uniform_categorical
from cogeneration.data.residue_constants import (
    restype_order_with_x,
    restypes,
    restypes_with_x,
)
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.featurizer import BatchFeaturizer
from cogeneration.dataset.protein_dataloader import LengthBatcher
from cogeneration.models.attention.attention_trunk import AttentionTrunk
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.embed import get_index_embedding, get_time_embedding
from cogeneration.models.esm_combiner import ESMCombinerNetwork
from cogeneration.models.utils import get_model_size_str
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.embed import PositionalEmbeddingMethod
from cogeneration.type.task import DataTask
from cogeneration.util.log import rank_zero_logger
from varco.config import VarcoConfig, VarcoDatasetConfig, VarcoModelConfig

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


def gather_and_pad(
    source: torch.Tensor,  # (B, N, ...)
    index: torch.Tensor,  # (B, P)
    mask: torch.Tensor,  # (B, P)
    fill_value: float = 0.0,
) -> torch.Tensor:  # (B, P, ...)
    """
    Gather from source along dim=1 using index, then fill padding positions with fill_value.

    Handles arbitrary trailing dimensions by expanding index and mask.
    For positions where mask is False, the result is set to fill_value.

    Args:
        source: (B, N, ...) tensor to gather from
        index: (B, P) indices into dim=1 of source (must be in [0, N-1])
        mask: (B, P) boolean mask; True for valid positions, False for padding
        fill_value: value to fill where mask is False

    Returns:
        (B, P, ...) tensor with gathered values where mask is True, fill_value otherwise
    """
    trailing_shape = source.shape[2:]
    idx = index
    for _ in trailing_shape:
        idx = idx.unsqueeze(-1)
    idx = idx.expand(-1, -1, *trailing_shape)  # (B, P, ...)

    gathered = source.gather(1, idx)  # (B, P, ...)

    if not trailing_shape:
        fill = torch.full_like(gathered, fill_value)
        return torch.where(mask, gathered, fill)
    else:
        m = mask
        for _ in trailing_shape:
            m = m.unsqueeze(-1)
        m = m.expand_as(gathered)
        fill = torch.full_like(gathered, fill_value)
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
    node_depth: (
        torch.Tensor
    )  # (A,) long, depth in tree (leaves=0, parent=max(child_depths)+1)
    leaf_deleted: torch.Tensor  # (A,) bool; True for deleted leaves, False otherwise

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
    ) -> "TreePlan":
        f"""
        Generate a simple planar coalescent tree plan with sampled split/birth/delete times.

        Algorithm:
        - Provide t=1 positions of motif tokens (motif_mask == True) and scaffold positions
        - Augment the tree with K_del deletions, which are to-be-deleted leaves duplicating t=1 endpoints
        - Determine scaffold groups, assign group id
        - Coalesce scaffold groups independently until K_scaffold roots remain within the group
        - Sample split times for internal nodes (aka "anchors")
        - Sample deletion times for deletion leaves (must come after split)
        - Define motif_mask etc. in aligned space (i.e. length A)

        Note this intentionally allows scaffold tokens to be present at t=0 (the nuclei), in
        addition to motif tokens, which avoids needing special boundary-sourcing logic.
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

        # Move leaf-level bookkeeping to the target device
        leaf_map_leaves = leaf_ref_t_cpu.to(device=device)  # (num_leaves,)
        leaf_del_t = leaf_del_t.to(device=device)  # (num_leaves,)

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

        # --- Coalescence
        # Iteratively merge adjacent nodes, tracking parent-child relationships and weights (num children)
        # until we are left with K_scaffold roots for each scaffold span group.

        # Initialize leaf nodes (leaf ids are 0..num_leaves-1)
        parent: List[int] = [-1] * num_leaves
        children: List[List[int]] = [[-1, -1] for _ in range(num_leaves)]
        weight: List[int] = [1 for _ in range(num_leaves)]

        roots: List[int] = []
        for gid, active in groups.items():
            is_scaffold_span = gid < span_gid

            # For scaffold spans: keep K nuclei (active nodes) instead of collapsing to 1.
            if is_scaffold_span:
                span_len = len(active)
                k_hi = min(max_scaffold_nuclei, span_len)
                k_lo = min(min_scaffold_nuclei, k_hi)
                K_scaffold = (
                    k_lo if k_lo == k_hi else (k_lo + rng.rand_int(k_hi - k_lo + 1))
                )
            else:
                # Motif singleton groups are already len==1; enforce target 1.
                K_scaffold = 1

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

        # Roots now track scaffold nuclei
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

        # leaf_map mapping is valid for leaves (< num_leaves) and 0 for internal nodes
        leaf_map = torch.zeros((A,), dtype=torch.long, device=device)
        leaf_map[:num_leaves] = leaf_map_leaves

        leaf_deleted = torch.zeros((A,), dtype=torch.bool, device=device)
        if num_leaves > 0:
            leaf_deleted[:num_leaves] = leaf_del_t

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
                E = -torch.log(
                    rng.rand(len(valid_nodes), device=device).clamp_min(1e-12)
                )
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

            # Sample uniform random values for all deleted leaves at once
            u = rng.rand(num_deleted, device=device)

            # Compute deletion times: dt = birth + (1 - birth) * u
            dt = b + (1.0 - b) * u

            # Clamp to valid range [birth + min_delete_eps, max_delete_time]
            # For births very close to 1, this ensures we have a valid deletion time
            dt = torch.clamp_min(dt, min=b + min_delete_eps)
            dt = torch.clamp_max(dt, max=max_delete_time)

            delete_time[valid_deleted] = dt

        # --- Final bookkeeping

        # Build motif mask in aligned tree space (A,)
        # Leaves 0..num_leaves-1 inherit motif/scaffold status from their referenced data index.
        # Internal nodes are always False.
        motif_mask_aligned = torch.zeros((A,), dtype=torch.bool, device=device)
        motif_mask_aligned[:num_leaves] = motif_mask_b[leaf_map_leaves]

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

        return cls(
            num_leaves=num_leaves,
            num_deletions=num_deletions,
            num_nodes=A,
            topo_order=topo_order,
            motif_mask=motif_mask_aligned,
            parent_idx=parent_idx,
            roots=roots_t,
            children_idx=children_idx,
            total_leaves=total_leaves,
            node_depth=node_depth,
            leaf_deleted=leaf_deleted,
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
        for i in range(min(A, N_leaf)):
            if int(self.total_leaves[i]) != 1:
                continue
            idx = int(self.leaf_map[i].item())
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

        A = int(self.num_nodes)
        N_leaf = int(self.num_leaves)

        motif_line_color = (
            "0.6"  # grey: straight lifelines for motif singleton residues
        )
        motif_marker_color = "black"
        scaffold_color = "blue"

        # Compute a planar x-position per node based on descendant leaf positions.
        # This keeps internal anchors visually located within their scaffold span, even though
        # internal node ids are appended and would otherwise plot far to the right.
        x_pos = np.full((A,), np.nan, dtype=np.float32)

        # Leaf nodes are 0..N_leaf-1 (includes deletion duplicates).
        for i in range(N_leaf):
            x_pos[i] = float(i)

        # Child-before-parent order
        child_first = self.topo_order.detach().to("cpu").numpy()[::-1].tolist()
        for node in child_first:
            if not np.isfinite(birth[node]):
                continue
            if int(total[node]) <= 1:
                continue
            c0, c1 = int(children[node][0]), int(children[node][1])
            if c0 < 0 or c1 < 0:
                continue
            if not np.isfinite(x_pos[c0]) or not np.isfinite(x_pos[c1]):
                continue
            w0 = float(total[c0])
            w1 = float(total[c1])
            denom = max(1e-6, w0 + w1)
            x_pos[node] = (x_pos[c0] * w0 + x_pos[c1] * w1) / denom

        # Fallback: any remaining unset node (shouldn't happen) uses its raw id
        for i in range(A):
            if not np.isfinite(x_pos[i]):
                x_pos[i] = float(i)

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
            xi = float(x_pos[i])
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
                [float(x_pos[p]), float(x_pos[child_idx])],
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
                [float(x_pos[r]) for r in motif_roots],
                [0.0] * len(motif_roots),
                marker="o",
                s=20,
                color=motif_marker_color,
                label="motif roots",
            )
        if len(scaffold_roots) > 0:
            ax.scatter(
                [float(x_pos[r]) for r in scaffold_roots],
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
                [float(x_pos[i]) for i in motif_leaf_ids],
                [1.0] * len(motif_leaf_ids),
                marker="s",
                s=14,
                color=motif_marker_color,
                label="motif leaves",
            )
        if len(scaffold_leaf_ids) > 0:
            ax.scatter(
                [float(x_pos[i]) for i in scaffold_leaf_ids],
                [1.0] * len(scaffold_leaf_ids),
                marker="s",
                s=14,
                color=scaffold_color,
                label="scaffold leaves",
            )
        if len(deleted_leaf_ids) > 0:
            ax.scatter(
                [float(x_pos[i]) for i in deleted_leaf_ids],
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
            internal_x_plot = [float(x_pos[i]) for i in internal_nodes]
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
        ax.set_xlim(-1, max(float(x_pos.max()) + 1.0, float(self.num_leaves)))
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), borderaxespad=0.0)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        print(f"💾 Saved tree plan to {path}")
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

    # times
    birth_time: torch.Tensor  # (B, A_max) float
    split_time: torch.Tensor  # (B, A_max) float
    delete_time: torch.Tensor  # (B, A_max) float

    # tree -> data mapping
    leaf_map: torch.Tensor  # (B, A_max) long

    def to(self, device: torch.device) -> "BatchedTreePlan":
        """Move all tensors to the specified device."""
        return BatchedTreePlan(
            topo_order=self.topo_order.to(device),
            motif_mask=self.motif_mask.to(device),
            roots=self.roots.to(device),
            roots_mask=self.roots_mask.to(device),
            parent_idx=self.parent_idx.to(device),
            children_idx=self.children_idx.to(device),
            total_leaves=self.total_leaves.to(device),
            node_depth=self.node_depth.to(device),
            leaf_deleted=self.leaf_deleted.to(device),
            birth_time=self.birth_time.to(device),
            split_time=self.split_time.to(device),
            delete_time=self.delete_time.to(device),
            leaf_map=self.leaf_map.to(device),
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
        x: torch.Tensor,  # (B, N, ...)
        fill_value: float = 0.0,
    ) -> torch.Tensor:  # (B, A_max, ...)
        """Broadcast data from N-space to A-space, keeping only leaf positions."""
        if x.ndim < 2:
            raise ValueError(
                f"Expected x to have at least 2 dims (B, N, ...); got {x.shape}"
            )
        leaf_idx = self.leaf_map.to(device=x.device).clamp_min(0)  # (B, A)
        leaf_mask = self.leaf_mask.to(device=x.device)  # (B, A)
        return gather_and_pad(x, leaf_idx, mask=leaf_mask, fill_value=fill_value)

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
            node_values[batch_idx, node_idx] = parent_values
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
                [p.leaf_map for p in plans], A_max, fill_value=0, dtype=torch.long
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
    aatypes_1: torch.Tensor  # (N,)


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
    aatypes_1: torch.Tensor  # (B, N)


@dataclass
class DataCorrupted:
    """Model input: corrupted and packed (length P_max) points present at time t"""

    t: torch.Tensor  # (B,)
    motif_mask: torch.Tensor  # (B, P_max) bool; True for fixed motif positions
    birth_time: torch.Tensor  # (B, P_max) 0.0 for motifs & roots, +inf for padding
    res_mask: torch.Tensor  # (B, P_max) int
    chain_idx: torch.Tensor  # (B, P_max) int
    trans_t: torch.Tensor  # (B, P_max, 3)
    aatypes_t: torch.Tensor  # (B, P_max)
    remaining_insertions: Optional[torch.Tensor] = (
        None  # (B, P_max) supervised target, remaining splits per present token
    )
    deleted: Optional[torch.Tensor] = (
        None  # (B, P_max) supervised target, 1 if destined-to-delete
    )

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

    def detach_clone(self) -> "DataCorrupted":
        """Detach and clone the data, e.g. to save in trajectory"""
        return DataCorrupted(
            t=self.t.detach().clone(),
            motif_mask=self.motif_mask.detach().clone(),
            birth_time=self.birth_time.detach().clone(),
            res_mask=self.res_mask.detach().clone(),
            chain_idx=self.chain_idx.detach().clone(),
            trans_t=self.trans_t.detach().clone(),
            aatypes_t=self.aatypes_t.detach().clone(),
            remaining_insertions=(
                self.remaining_insertions.detach().clone()
                if self.remaining_insertions is not None
                else None
            ),
            deleted=(
                self.deleted.detach().clone() if self.deleted is not None else None
            ),
        )

    def apply_insertions_deletions(
        self,
        insertions: torch.Tensor,  # (B, P) bool
        deletions: torch.Tensor,  # (B, P) bool
        t_birth: float,
    ) -> Tuple["DataCorrupted", torch.Tensor]:
        """
        Apply insertions and deletions to create a new batch.
        Returns new batch and positions where insertions occurred.

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
            aatypes_t=gather_and_pad(
                self.aatypes_t, gather_idx, new_valid, fill_value=MASK_TOKEN_INDEX
            ),
        )

        return new_batch, is_insertion


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
    aatypes_t: torch.Tensor  # (B, A)
    # supervision
    remaining_insertions: torch.Tensor  # (B, A) target count per aligned node
    deleted: torch.Tensor  # (B, A) bool, aligned deletion label (leaf only)

    @staticmethod
    def pack_present_indices(
        present_mask: torch.Tensor,  # (B, A) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Derive packed indices deterministically from a (B, A) present mask.

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

        B, A = present_mask.shape
        device = present_mask.device

        sort_key = (~present_mask).to(torch.int32)
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
        return self.pack_present_indices(self.present_mask)

    def pack_present(self) -> DataCorrupted:
        """Pack aligned (A) state into present (P_max) state for model input."""
        idx_pack, pack_mask, P_b, P_max = self._pack_indices()

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
            aatypes_t=gather_and_pad(
                self.aatypes_t,
                idx_pack,
                pack_mask,
                fill_value=MASK_TOKEN_INDEX,
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
        if self.remaining_insertions.shape != (B, A):
            raise ValueError("remaining_insertions shape mismatch")
        if self.deleted.shape != (B, A):
            raise ValueError("deleted shape mismatch")
        if D != 3:
            raise ValueError("trans_t last dim must be 3")


@dataclass
class ModelPrediction:
    """t=1 prediction for present state (length P)"""

    pred_trans_1: torch.Tensor  # (B, P, 3) base; predicted final/anchor positions
    pred_aatype_logits: torch.Tensor  # (B, P, 21) base; amino acid logits
    pred_split_rate: torch.Tensor  # (B, P) non-negative remaining splits per token
    pred_split_pooled_log1p_rate: (
        torch.Tensor
    )  # (B,) log1p-space total remaining splits
    pred_del_logits: torch.Tensor  # (B, P) deletion logit per token

    def detach_clone(self) -> "ModelPrediction":
        """Detach and clone the prediction, e.g. to save in trajectory"""
        return ModelPrediction(
            pred_trans_1=self.pred_trans_1.detach().clone(),
            pred_aatype_logits=self.pred_aatype_logits.detach().clone(),
            pred_split_rate=self.pred_split_rate.detach().clone(),
            pred_split_pooled_log1p_rate=self.pred_split_pooled_log1p_rate.detach().clone(),
            pred_del_logits=self.pred_del_logits.detach().clone(),
        )


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
    ):
        super().__init__(
            cfg=cfg,
            task=DataTask.inpainting,
            eval=False,
            use_test=False,
        )

    def __getitem__(self, idx) -> DataSample:
        feats = super().__getitem__(idx)

        motif_mask = feats[bp.motif_mask].bool()
        res_mask = feats[bp.res_mask].int()
        chain_idx = feats[bp.chain_idx].int()

        # domains
        trans_1 = feats[bp.trans_1]
        aatypes_1 = feats[bp.aatypes_1]

        tree_plan = TreePlan.generate(motif_mask=motif_mask)
        tree_plan.validate()

        return DataSample(
            tree_plan=tree_plan,
            motif_mask=motif_mask,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_1=trans_1,
            aatypes_1=aatypes_1,
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
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=ProteinDataLoader.collate_fn,
            prefetch_factor=None if num_workers == 0 else 1,
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
        aatypes_1 = torch.stack([item.aatypes_1 for item in batch])  # (B, N)

        return DataBatch(
            tree=tree,
            motif_mask=motif_mask,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_1=trans_1,
            aatypes_1=aatypes_1,
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
            3  # trans_t
            + 1  # birth_time
            + 1  # motif_mask
            + 1  # chain_idx
            + self.time_embed_dim  # time_embed
            + self.pos_embed_dim  # pos_embed
            + self.num_aatype_tokens  # aatypes_onehot
        )

        self.node_feature_net = nn.Linear(self.input_dim, self.node_dim)

        self.edge_feature_net = EdgeFeatureNet(cfg=self.cfg.edge_features)

        if self.cfg.esm_combiner.enabled:
            self.esm_combiner = ESMCombinerNetwork(cfg=self.cfg.esm_combiner)

        self.trunk = AttentionTrunk(
            cfg=self.cfg.trunk,
            attn_cfg=self.cfg.attention,
        )

        self.trans_pred = nn.Linear(self.node_dim, 3)

        self.aatype_pred = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.num_aatype_tokens),
        )

        # Insertions and deletions
        self.split_rate_pred = nn.Linear(self.node_dim, 1)
        self.split_pooled_log1p_rate_pred = nn.Linear(self.node_dim, 1)
        self.del_logits_pred = nn.Linear(self.node_dim, 1)

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
                batch.trans_t,  # (B, P, 3)
                birth_time,  # (B, P, 1)
                batch.motif_mask[:, :, None].float(),  # (B, P, 1)
                batch.chain_idx[:, :, None].int(),  # (B, P, 1)
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
            diffuse_mask=batch.motif_mask,
            chain_index=batch.chain_idx,
            contact_conditioning=None,
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

        # Predict translations
        pred_trans_1 = self.trans_pred(node_embed)

        # Predict amino acid logits
        pred_aatype_logits = self.aatype_pred(node_embed)  # (B, P, 21)
        # Ensure mask logits disabled
        if self.num_aatype_tokens == NUM_TOKENS + 1:
            pred_aatype_logits[:, :, MASK_TOKEN_INDEX] = -1e9

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

        return ModelPrediction(
            pred_trans_1=pred_trans_1,
            pred_aatype_logits=pred_aatype_logits,
            pred_split_rate=split_rate,
            pred_split_pooled_log1p_rate=split_pooled_log1p_rate,
            pred_del_logits=del_logits,
        )


""" Tree Coupling """


class Coupling(ABC):
    """Coupling struct tracks domain-specific anchors, and the corruption tree plan."""

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

        if x0 is None:
            # Broadcast x1 to aligned space (copies leaves, fills internal nodes)
            x1_aligned = tree.broadcast_to_leaves(x1, fill_value=0)
            # Sample base distribution for all positions
            x0 = self.sample_base(
                motif_mask=tree.motif_mask,
                x1=x1_aligned,
                device=device,
            )

        anchors = self.build_anchors(x1=x1, tree=tree)

        creation_state = x0.clone()
        t_expanded = t.unsqueeze(1).expand(B, A)

        def split_fn(node_creation, node_target, node_t0, node_st):
            return self.sample_bridge(
                x_start=node_creation,
                x_end=node_target,
                s=node_st,
                t0=node_t0,
            )

        creation_state = tree.traverse_top_down(
            creation_state=creation_state,
            target_state=anchors,
            split_fn=split_fn,
            max_split_time=t_expanded,
        )

        present_mask = tree.present_mask(t=t)

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

        x_t = self.post_process(
            x_t=x_t,
            present_mask=present_mask,
            motif_mask=tree.motif_mask,
            anchors=anchors,
        )

        return x_t, self._make_coupling(anchors=anchors, tree=tree)

    @abstractmethod
    def _make_coupling(
        self,
        anchors: torch.Tensor,
        tree: BatchedTreePlan,
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
    ) -> torch.Tensor:
        """Single Euler(-Maruyama) step for sampling.

        Noise is controlled by the coupler instance (e.g. sigma); if sigma is None/0, this is deterministic.
        """
        raise NotImplementedError


@dataclass
class TranslationCoupling(Coupling):
    tree: BatchedTreePlan

    # domain-specific anchors (used for supervision)
    anchors: torch.Tensor  # (B, A, 3)


class TranslationCoupler(Coupler[TranslationCoupling]):
    def __init__(self, sigma: Optional[float] = 1.0):
        self.sigma = sigma

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

        if self.sigma is None:
            return mean.view(original_shape)

        var = (
            (flat_s - flat_t0).clamp_min(0.0) * (1.0 - flat_s).clamp_min(0.0) / denom
        ).clamp_min(0.0)
        std = (var.sqrt() * self.sigma).to(mean.dtype)
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
        anchors: torch.Tensor,
        tree: BatchedTreePlan,
    ) -> TranslationCoupling:
        return TranslationCoupling(anchors=anchors, tree=tree)

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, 3)
        x1_pred: torch.Tensor,  # (B, P, 3)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        trans_pred = x1_pred
        if x_t.shape != trans_pred.shape:
            raise ValueError(
                f"x_t and x1_pred must match shape; got {tuple(x_t.shape)} vs {tuple(trans_pred.shape)}"
            )
        if x_t.ndim != 3 or x_t.shape[-1] != 3:
            raise ValueError(
                f"Expected x_t to have shape (B, P, 3); got {tuple(x_t.shape)}"
            )

        B, P, _ = x_t.shape
        device = x_t.device

        valid_fmask = (
            (birth_time <= t[:, None]).bool().float().unsqueeze(-1)
        )  # (B, P, 1)

        denom = (1.0 - t).clamp_min(1e-4).view(B, 1, 1)
        v = (trans_pred - x_t) / denom
        x_next = x_t + v * float(dt)

        if (self.sigma is not None) and float(self.sigma) > 0.0 and float(dt) > 0.0:
            x_next = x_next + torch.randn_like(x_next) * (
                float(self.sigma) * math.sqrt(float(dt))
            )

        return x_next * valid_fmask + x_t * (1.0 - valid_fmask)


@dataclass
class AATypesCoupling(Coupling):
    """Coupling for amino acid types using CTMC bridge."""

    tree: BatchedTreePlan

    # Anchor tokens sampled from descendant distributions
    anchors: torch.Tensor  # (B, A) long

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
        beta: float = 3.0,
        drift_temp: float = 1.0,
        noise_scale: float = 0.5,
        uncertainty_sharpness: float = 1.0,
        leave_mass_cap: float = 0.25,
    ):
        """
        beta: Total leaving rate for the CTMC (rate of jumping away from current state)
        drift_temp: Temperature for softmax in euler_step (lower = sharper)
        noise_scale: Scale for uniform noise added to step probabilities (0 = no noise)
        uncertainty_sharpness: Exponent for uncertainty gating (higher = sharper gate)
        leave_mass_cap: Maximum total off-diagonal (jump) probability per step
        """
        self.beta = beta
        self.drift_temp = drift_temp
        self.noise_scale = noise_scale
        self.uncertainty_sharpness = uncertainty_sharpness
        self.leave_mass_cap = leave_mass_cap

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
        lam = self.beta * self.K / (self.K - 1)
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

        if self.noise_scale > 0:
            uniform_dist = torch.ones(B, A, K, device=device) / K
            noise_weight = min(self.noise_scale * 0.2, 0.2)
            anchor_probs = (
                1.0 - noise_weight
            ) * anchor_probs + noise_weight * uniform_dist

        row_sums = anchor_probs.sum(dim=-1, keepdim=True)
        has_mass = row_sums > 1e-12

        uniform_fallback = torch.ones(B, A, K, device=device) / K
        anchor_probs = torch.where(
            has_mass,
            anchor_probs / row_sums.clamp_min(1e-12),
            uniform_fallback,
        )

        anchor_probs_flat = anchor_probs.view(-1, K)
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
        anchors: torch.Tensor,
        tree: BatchedTreePlan,
    ) -> AATypesCoupling:
        return AATypesCoupling(
            anchors=anchors,
            anchor_probs=self._last_anchor_probs,
            tree=tree,
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
        uncertainty = (1.0 - p_current) ** self.uncertainty_sharpness
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
        probs = F.softmax(logits / self.drift_temp, dim=-1)  # (B, P, K)

        # Uncertainty gating
        uncertainty = self._uncertainty_gate(x_t, probs)  # (B, P)

        # Drift gain: 1 / (1 - t), clamped
        t_clamped = t.clamp(0.0, 0.99)
        drift_gain = 1.0 / (1.0 - t_clamped + 1e-4)
        drift_gain = drift_gain.clamp(max=100.0).view(B, 1, 1)  # (B, 1, 1)

        # Compute off-diagonal drift mass
        step_probs = dt * drift_gain * probs * uncertainty.unsqueeze(-1)  # (B, P, K)

        # Zero out current token (off-diagonal only)
        current_onehot = F.one_hot(x_t.long().clamp(0, K - 1), num_classes=K).float()
        step_probs = step_probs * (1.0 - current_onehot)

        # Noise injection: add uniform mass to off-diagonal, scaled by sigma_t
        if self.noise_scale > 0:
            # sigma_t^2 ~ t(1-t), peaks at t=0.5
            sigma_t_sq = (t * (1.0 - t)).view(B, 1, 1)  # (B, 1, 1)
            noise_weight = self.noise_scale * dt * sigma_t_sq  # (B, 1, 1)

            # Uniform over non-current tokens
            uniform_noise = (1.0 - current_onehot) / max(K - 1, 1)
            step_probs = step_probs + noise_weight * uniform_noise

        # Cap leave mass
        if self.leave_mass_cap is not None and self.leave_mass_cap > 0:
            row_sum = step_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            shrink = (self.leave_mass_cap / row_sum).clamp_max(1.0)
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
    ) -> torch.Tensor:
        """Single Euler step for discrete amino acid sampling.

        Uses discrete jump process with uncertainty gating and noise injection.
        See _compute_step_probs for details on the probability computation.
        """
        B, P = x_t.shape
        K = self.K

        if x1_pred.shape != (B, P, K):
            raise ValueError(f"Expected x1_pred shape (B, P, K); got {x1_pred.shape}")

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


""" Interpolant supporting tree coupling """


@dataclass
class TreeCouplings:
    """Container for all domain couplings from a single corruption call."""

    translation: TranslationCoupling
    aatypes: AATypesCoupling


@dataclass
class TreeInterpolant:
    device: torch.device = torch.device("cpu")
    min_t: float = 0.005
    translation_coupler: Coupler[TranslationCoupling] = field(
        default_factory=lambda: TranslationCoupler(sigma=1.0)
    )
    aatypes_coupler: Coupler[AATypesCoupling] = field(
        default_factory=lambda: AATypesCoupler(
            beta=3.0, drift_temp=1.0, noise_scale=0.5, uncertainty_sharpness=1.0
        )
    )

    def set_device(self, device: torch.device):
        self.device = device

    def corrupt_to(
        self,
        batch: DataBatch,
        t: torch.Tensor,  # (B,)
        trans_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
    ) -> Tuple[DataBridged, TreeCouplings]:
        tree = batch.tree.to(self.device)
        trans_1 = batch.trans_1.to(self.device)
        aatypes_1 = batch.aatypes_1.to(self.device)
        t = t.to(self.device)
        if trans_0 is not None:
            trans_0 = trans_0.to(self.device)
        if aatypes_0 is not None:
            aatypes_0 = aatypes_0.to(self.device)

        res_mask = batch.tree.broadcast_to_leaves(
            x=batch.res_mask.to(self.device), fill_value=0
        )
        chain_idx = batch.tree.broadcast_to_leaves(
            x=batch.chain_idx.to(self.device), fill_value=0
        )

        # Corrupt translations
        trans_t, trans_coupling = self.translation_coupler.corrupt(
            tree=tree,
            t=t,
            x1=trans_1,
            x0=trans_0,
        )
        trans_coupling.validate()

        # Corrupt aatypes
        aatypes_t, aatypes_coupling = self.aatypes_coupler.corrupt(
            tree=tree,
            t=t,
            x1=aatypes_1,
            x0=aatypes_0,
        )
        aatypes_coupling.validate()

        bridged = DataBridged(
            t=t,
            present_mask=tree.present_mask(t=t),
            motif_mask=tree.motif_mask,
            birth_time=tree.birth_time,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_t=trans_t,
            aatypes_t=aatypes_t,
            remaining_insertions=tree.remaining_insertions_t(t=t),
            deleted=tree.leaf_deleted,
        )
        bridged.validate()

        couplings = TreeCouplings(translation=trans_coupling, aatypes=aatypes_coupling)
        return bridged, couplings

    def corrupt_batch(self, batch: DataBatch) -> Tuple[DataBridged, TreeCouplings]:
        # pick a single time to share across the batch,
        # simply so they have a similar number of insertion/deletions to simulate
        # since corruption is run across the batch
        shared_t = (
            torch.rand(1, device=self.device) * (1.0 - 2.0 * self.min_t) + self.min_t
        )
        t = torch.ones(batch.trans_1.shape[0], device=self.device) * shared_t  # (B,)

        return self.corrupt_to(batch=batch, t=t)

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

        # Gather x1 values from data for motif positions (fill scaffolds with placeholder)
        trans_1_gathered = gather_and_pad(
            data.trans_1.to(device), gather_idx, is_motif, fill_value=0.0
        )
        aatypes_1_gathered = gather_and_pad(
            data.aatypes_1.to(device), gather_idx, is_motif, fill_value=MASK_TOKEN_INDEX
        )

        # Sample base distributions using coupler interfaces
        trans_0 = self.translation_coupler.sample_base(
            motif_mask=is_motif, x1=trans_1_gathered, device=device
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
            aatypes_t=aatypes_0,
        )

    @staticmethod
    def _sample_insert_delete_substitute(
        split_rate: torch.Tensor,  # (B, P)
        del_logits: torch.Tensor,  # (B, P)
        is_root: torch.Tensor,  # (B, P) bool
        valid_mask: torch.Tensor,  # (B, P) bool
        t_val: float,  # current time
        dt: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample insert/delete/substitute events for present positions in a batch.
        """
        # Convert rates/logits to per-step probabilities via hazard-rate formulation
        # Both split_rate and del_logits predict "eventual" quantities, so we scale by 1/(1-t)
        denom = max(1e-4, 1.0 - t_val)

        # Insert probability from split rate
        lam_ins = (split_rate.clamp_min(0.0) * (dt / denom)).clamp_max(20.0)
        p_ins = (1.0 - torch.exp(-lam_ins)).clamp(0.0, 0.95)

        # Delete probability from logits
        # del_logits predicts "destined-to-delete", convert to instantaneous probability
        lam_del = (torch.sigmoid(del_logits) * (dt / denom)).clamp_max(20.0)
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
        traj.samples.append(batch.detach_clone())

        model.eval()
        with torch.no_grad():
            t_grid = torch.linspace(self.min_t, 1.0, steps=num_steps, device=device)
            pbar = tqdm(
                enumerate(range(num_steps)),
                total=num_steps,
                desc="Sampling",
                leave=False,
            )
            for step_num, step in pbar:
                t_val = float(t_grid[step].item())
                t_next = (
                    float(t_grid[step + 1].item())
                    if step + 1 < num_steps
                    else (1.0 - self.min_t)
                )
                dt = float(max(1e-6, t_next - t_val))

                # Set current time and predict
                batch.t = torch.full(
                    (num_batch,), t_val, dtype=torch.float32, device=device
                )
                pred = model.forward(batch)

                if traj_frames is None or step_num % traj_frames == 0:
                    traj.pred.append(pred.detach_clone())

                # Euler step for alive tokens' translations
                trans_next = self.translation_coupler.euler_step(
                    x_t=batch.trans_t,
                    x1_pred=pred.pred_trans_1,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                )
                batch.trans_t = trans_next

                # Euler step for alive tokens' aatypes
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
                insertions, deletions, _ = self._sample_insert_delete_substitute(
                    split_rate=pred.pred_split_rate,
                    del_logits=pred.pred_del_logits,
                    is_root=is_root,
                    valid_mask=batch.valid_mask & ~batch.motif_mask,
                    t_val=t_val,
                    dt=dt,
                )
                batch, insert_mask = batch.apply_insertions_deletions(
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

                    # TODO - apply insertion logits, once we predict them
                    # TODO - apply substitutions for sequence

                # TODO - recenter

                # Save
                if traj_frames is None or step_num % traj_frames == 0:
                    traj.samples.append(batch.detach_clone())

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
    base_trans_loss: torch.Tensor  # MSE on translations
    pairwise_loss: torch.Tensor  # local pairwise distance loss
    base_seq_loss: torch.Tensor  # CE on amino acid logits vs anchor tokens
    split_token_loss: torch.Tensor  # Poisson loss on per-token remaining splits
    split_pooled_loss: torch.Tensor  # aux Poisson loss on total remaining splits
    del_loss: torch.Tensor  # BCE on per-token logits (terminal tokens only)


@dataclass
class BranchFlowLossCalculator:
    # Time normalization clip (higher weight as t -> 1)
    t_normalize_clip: float = 0.9
    # Local pairwise distance threshold (angstroms)
    proximity_threshold_ang: float = 7.0
    # Loss weights
    trans_loss_weight: float = 2.0
    pairwise_dist_loss_weight: float = 0.2
    seq_loss_weight: float = 1.0
    split_loss_weight: float = 0.2
    split_pooled_loss_weight: float = 0.025
    del_loss_weight: float = 0.2

    def _time_norm_scale(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute time-based normalization scale: 1 - min(t, clip).
        Higher weight (smaller divisor) as t -> 1.
        """
        return 1 - torch.min(t, torch.tensor(self.t_normalize_clip, device=t.device))

    def _base_trans_loss(
        self,
        pred_trans: torch.Tensor,  # (B, P, 3)
        target_trans: torch.Tensor,  # (B, P, 3)
        t: torch.Tensor,  # (B,)
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """Translation loss for anchor/final positions"""
        B, P, D = pred_trans.shape

        # Time-based normalization (higher weight as t -> 1)
        t_norm = self._time_norm_scale(t=t).view(B, 1, 1)

        pred_scaled = pred_trans * ANG_TO_NM_SCALE / t_norm
        target_scaled = target_trans * ANG_TO_NM_SCALE / t_norm
        mse = (pred_scaled - target_scaled).square()  # (B, P, 3)

        # Per-example masked mean (normalize by number of present positions)
        mask_f = mask.unsqueeze(-1).float()  # (B, P, 1)
        mse = mse * mask_f
        denom = mask_f.sum(dim=(1, 2)).clamp_min(1.0)  # (B,)
        loss_per_batch = mse.sum(dim=(1, 2)) / denom  # (B,)

        return (loss_per_batch.mean() * self.trans_loss_weight).clamp(max=10.0)

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

        # Compute pairwise distances (B, P, P)
        with torch.no_grad():
            target_dists = torch.cdist(target_trans, target_trans)
        pred_dists = torch.cdist(pred_trans, pred_trans)

        # Mask for valid pairs
        pair_mask = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, P, P)

        # Limit to local neighborhood in ground truth
        proximity_mask = target_dists < self.proximity_threshold_ang
        pair_mask = pair_mask & proximity_mask

        # Exclude self-pairs
        eye = torch.eye(P, device=pred_trans.device, dtype=torch.bool).unsqueeze(0)
        pair_mask = pair_mask & ~eye

        # Squared distance error with time normalization
        dist_error = ((pred_dists - target_dists) * ANG_TO_NM_SCALE / t_norm) ** 2
        dist_error = dist_error * pair_mask.float()

        # Normalize by number of valid pairs per batch
        denom = pair_mask.float().sum(dim=(1, 2)).clamp_min(1.0)
        loss_per_batch = dist_error.sum(dim=(1, 2)) / denom

        return (loss_per_batch.mean() * self.pairwise_dist_loss_weight).clamp(max=5.0)

    def _seq_loss(
        self,
        pred_aatype_logits: torch.Tensor,  # (B, P, K)
        target_anchor_tokens: torch.Tensor,  # (B, P) long
        mask: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        """Sequence loss: cross-entropy on amino acid logits vs anchor tokens."""
        B, P, K = pred_aatype_logits.shape

        # Flatten for cross-entropy: (B*P, K) and (B*P,)
        logits_flat = pred_aatype_logits.view(-1, K)
        targets_flat = target_anchor_tokens.view(-1)
        mask_flat = mask.view(-1).float()

        # Cross-entropy per token (reduction=none)
        ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # (B*P,)

        # Masked mean
        denom = mask_flat.sum().clamp_min(1.0)
        seq_loss = (ce * mask_flat).sum() / denom

        return (seq_loss * self.seq_loss_weight).clamp(max=10.0)

    def _split_token_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,  # (B, P)
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Per-token Poisson Bregman divergence:
        D(k || r) = r - k + k * log(k/r)  for k > 0
        D(0 || r) = r                     for k = 0
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

        pad_f = mask.float()
        denom = pad_f.sum(dim=1).clamp_min(1.0)  # (B,)
        return ((token_loss * pad_f).sum(dim=1) / denom).mean() * self.split_loss_weight

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
        return F.mse_loss(pred_log1p, target_log) * self.split_pooled_loss_weight

    def _deletion_loss(
        self,
        pred: ModelPrediction,
        batch: DataCorrupted,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Deletion loss, supervised only on terminal tokens"""
        if batch.deleted is None:
            return torch.tensor(0.0, device=batch.trans_t.device)

        del_mask = mask & (batch.remaining_insertions == 0)
        if not bool(del_mask.any()):
            return torch.tensor(0.0, device=batch.trans_t.device)

        del_logits = pred.pred_del_logits  # (B, P)
        del_targets = batch.deleted.float()  # (B, P)
        bce = F.binary_cross_entropy_with_logits(
            del_logits, del_targets, reduction="none"
        )
        denom = del_mask.float().sum().clamp_min(1.0)
        del_loss = (bce * del_mask.float()).sum() / denom

        return del_loss * self.del_loss_weight

    def calculate(
        self,
        batch: DataCorrupted,
        pred: ModelPrediction,
        couplings: TreeCouplings,
    ) -> BranchFlowLosses:
        B, P, D = batch.trans_t.shape
        assert pred.pred_trans_1.shape == (B, P, D)

        trans_coupling = couplings.translation
        present_mask = trans_coupling.tree.present_mask(t=batch.t)  # (B, A)
        is_root = (trans_coupling.tree.parent_idx < 0).to(torch.long)
        valid_mask = batch.valid_mask  # (B, P_max)

        # Reconstruct the same packing indices used by TwoMoonsBridged.pack_present()
        idx_pack, pack_mask, P_b, P_max = DataBridged.pack_present_indices(present_mask)

        # Pack translation anchors (i.e. targets) into (B, P_max, D) in the same order as model inputs/predictions
        trans_anchors_pack = trans_coupling.anchors.gather(
            1, idx_pack.unsqueeze(-1).expand(-1, -1, D)
        )
        # zero pad
        trans_anchors_pack = trans_anchors_pack * pack_mask.unsqueeze(-1).float()

        # Pack aatype anchor tokens (targets for sequence loss)
        aatypes_coupling = couplings.aatypes
        aatype_anchors_pack = aatypes_coupling.anchors.gather(1, idx_pack)  # (B, P_max)
        # zero pad, will be masked
        aatype_anchors_pack = aatype_anchors_pack * pack_mask.long()

        # Base loss on predicting existing residues' final positions
        base_loss = self._base_trans_loss(
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

        # Base sequence loss
        base_seq_loss = self._seq_loss(
            pred_aatype_logits=pred.pred_aatype_logits,
            target_anchor_tokens=aatype_anchors_pack,
            mask=valid_mask,
        )

        # Insertion / split losses
        split_token_loss = self._split_token_loss(
            pred=pred,
            batch=batch,
            mask=valid_mask,
        )
        split_pooled_loss = self._split_pooled_loss(pred=pred, batch=batch)

        # Deletion loss
        del_loss = self._deletion_loss(pred=pred, batch=batch, mask=valid_mask)

        total_loss = (
            base_loss
            + pairwise_loss
            + base_seq_loss
            + split_token_loss
            + split_pooled_loss
            + del_loss
        )

        return BranchFlowLosses(
            total_loss=total_loss,
            base_trans_loss=base_loss,
            pairwise_loss=pairwise_loss,
            base_seq_loss=base_seq_loss,
            split_token_loss=split_token_loss,
            split_pooled_loss=split_pooled_loss,
            del_loss=del_loss,
        )


""" Visualization """


class BranchingFlowVisualizer:
    def __init__(
        self,
        sigma: Optional[float] = 1.0,
    ):
        # Use couplers with sigma set explicitly
        self.translation_coupler = TranslationCoupler(sigma=sigma)

        self.interpolant = TreeInterpolant(
            translation_coupler=self.translation_coupler,
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
    def _create_sequence_artists(ax: plt.Axes, max_len: int):
        """Pre-create all artists needed for sequence visualization."""
        letters, colors = BranchingFlowVisualizer._aa_letters_and_colors()

        ax.set_xlim(0, max_len)
        ax.set_ylim(-0.3, 1)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")

        rectangles = []
        texts = []
        motif_rects = []

        for i in range(max_len):
            # AA background rectangle
            rect = plt.Rectangle(
                (i, 0), 1, 1, facecolor=colors[0], edgecolor="white", lw=0.5
            )
            ax.add_patch(rect)
            rectangles.append(rect)

            # AA letter text
            text = ax.text(
                i + 0.5,
                0.5,
                letters[0],
                ha="center",
                va="center",
                fontsize=8,
                color="k",
            )
            texts.append(text)

            # Motif underline rectangle
            motif_rect = plt.Rectangle((i, -0.25), 1, 0.15, facecolor="black", lw=0)
            ax.add_patch(motif_rect)
            motif_rects.append(motif_rect)

        return rectangles, texts, motif_rects, letters, colors

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
                # Update visible artists
                aa_idx = int(aatypes[i]) if aatypes[i] < len(letters) else 20
                rectangles[i].set_facecolor(colors[aa_idx])
                rectangles[i].set_visible(True)
                texts[i].set_text(letters[aa_idx])
                texts[i].set_visible(True)
                motif_rects[i].set_visible(bool(motif_mask[i]))
            else:
                # Hide unused artists
                rectangles[i].set_visible(False)
                texts[i].set_visible(False)
                motif_rects[i].set_visible(False)

    @staticmethod
    def _create_3d_scatter_artist(
        ax: plt.Axes,
        max_points: int,
        trans_min: np.ndarray,
        trans_max: np.ndarray,
    ):
        """Pre-create a 3D scatter artist with max_points capacity.

        Returns a scatter object that can be updated via _update_3d_scatter.
        """
        # Initialize with zeros - we'll update positions/colors each frame
        dummy_pos = np.zeros((max_points, 3))
        dummy_colors = np.zeros(max_points)
        dummy_sizes = np.ones(max_points) * 40.0

        # Create the scatter plot (initially all hidden via alpha=0 for unused)
        scat = ax.scatter(
            dummy_pos[:, 0],
            dummy_pos[:, 1],
            dummy_pos[:, 2],
            c=dummy_colors,
            cmap="Spectral",
            vmin=0,
            vmax=1,
            s=dummy_sizes,
            depthshade=True,
            alpha=0.75,
        )

        # Set axis properties once
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.view_init(elev=25, azim=45)
        ax.set_xlim(trans_min[0], trans_max[0])
        ax.set_ylim(trans_min[1], trans_max[1])
        ax.set_zlim(trans_min[2], trans_max[2])

        return scat

    @staticmethod
    def _update_3d_scatter(
        scat,
        ax: plt.Axes,
        trans_alive: np.ndarray,  # (n_alive, 3)
        motif_alive: np.ndarray,  # (n_alive,)
        max_points: int,
        t_val: float,
    ):
        """Update pre-created 3D scatter artist with new data.

        Pads positions to max_points and hides unused points by moving them off-screen.
        """
        n_alive = trans_alive.shape[0] if trans_alive.size > 0 else 0

        # Pad positions to max_points (hide extras by making alpha 0 via color array)
        if n_alive > 0:
            # Prepare padded arrays
            padded_pos = np.zeros((max_points, 3))
            padded_pos[:n_alive] = trans_alive

            # Color by position index
            color_idx = np.zeros(max_points)
            color_idx[:n_alive] = np.arange(n_alive)

            # Sizes: motifs smaller
            sizes = np.zeros(max_points)
            sizes[:n_alive] = np.where(motif_alive, 15.0, 40.0)

            # Alpha: visible for alive, 0 for hidden
            alphas = np.zeros(max_points)
            alphas[:n_alive] = 0.75

            # Update scatter data
            scat._offsets3d = (padded_pos[:, 0], padded_pos[:, 1], padded_pos[:, 2])
            scat.set_array(color_idx)
            scat.set_clim(0, max(n_alive - 1, 1))
            scat.set_sizes(sizes)
            # Note: per-point alpha not directly supported, but we use size=0 for hidden
            # Set sizes to 0 for hidden points
            sizes[n_alive:] = 0
            scat.set_sizes(sizes)
        else:
            # No points - hide all
            scat.set_sizes(np.zeros(max_points))

        ax.set_title(f"t = {t_val:.2f} (N={n_alive})")

    def plot_trajectory(
        self,
        traj: Trajectory,
        out_dir: Optional[str] = None,
        filename: str = "trajectory",
        max_frames: Optional[int] = 50,
    ) -> str:
        """Plot a trajectory animation from any Trajectory (corruption or sampling)."""
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        os.makedirs(out_dir, exist_ok=True)

        if not traj.samples:
            raise ValueError("Trajectory has no samples to plot")

        ext, writer = self._get_anim_writer()
        anim_path = os.path.join(out_dir, f"{filename}.{ext}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"💾 Saving trajectory animation to {anim_path}")

        num_batch = traj.samples[0].trans_t.shape[0]
        num_plots = min(num_batch, 4)

        # Compute camera limits and max sequence length from all trajectory samples
        device = traj.samples[0].trans_t.device
        trans_min = torch.zeros(3, device=device)
        trans_max = torch.zeros(3, device=device)
        max_seq_len = 0
        for sample in traj.samples:
            valid = sample.valid_mask
            valid = valid & (
                torch.arange(num_batch, device=device).unsqueeze(1) < num_plots
            )
            trans_t = sample.trans_t[valid]  # (num_plots*P, 3)
            trans_min = torch.min(trans_min, trans_t.min(dim=0).values)
            trans_max = torch.max(trans_max, trans_t.max(dim=0).values)
            # Track max sequence length across all plots
            for i in range(num_plots):
                max_seq_len = max(max_seq_len, int(sample.valid_mask[i].sum().item()))
        trans_min = np.tile(trans_min.cpu().numpy(), (num_batch, 1))
        trans_max = np.tile(trans_max.cpu().numpy(), (num_batch, 1))

        num_cols = min(num_plots, 2)
        num_rows = math.ceil(num_plots / num_cols)
        fig = plt.figure(figsize=(10 * num_cols, 12 * num_rows))
        # 2 rows per plot: sequence bar (height 2) + 3D structure (height 10)
        gs = fig.add_gridspec(
            num_rows * 2,
            num_cols,
            height_ratios=[2, 10] * num_rows,
            hspace=0.02,
            wspace=0.05,
        )

        fig.subplots_adjust(
            left=0.03,
            right=0.97,
            bottom=0.03,
            top=0.95,
            wspace=0.05,
            hspace=0.05,
        )

        # Create all axes and sequence artists once before the animation loop
        axes_seq = []
        axes_3d = []
        seq_artists = (
            []
        )  # Store (rectangles, texts, motif_rects, letters, colors) for each plot
        scatter_artists = []  # Pre-created 3D scatter artists for each plot
        for i in range(num_plots):
            row, col = divmod(i, num_cols)
            ax_seq = fig.add_subplot(gs[row * 2, col])
            ax_3d = fig.add_subplot(gs[row * 2 + 1, col], projection="3d")
            axes_seq.append(ax_seq)
            axes_3d.append(ax_3d)

            # Pre-create sequence artists for this plot
            artists = BranchingFlowVisualizer._create_sequence_artists(
                ax_seq, max_seq_len
            )
            seq_artists.append(artists)

            # Pre-create 3D scatter artist (avoids ax.clear() each frame)
            scat = BranchingFlowVisualizer._create_3d_scatter_artist(
                ax_3d, max_seq_len, trans_min[i], trans_max[i]
            )
            scatter_artists.append(scat)

        # Downsample to max_frames
        if max_frames is not None and len(traj.samples) > max_frames:
            sample_indices = np.linspace(
                0, len(traj.samples) - 1, max_frames, dtype=int
            )
        else:
            sample_indices = np.arange(len(traj.samples))

        with writer.saving(fig, anim_path, dpi=100):
            for idx in tqdm(sample_indices, desc="plot_trajectory()", leave=False):
                sample = traj.samples[idx]
                valid_mask = sample.valid_mask.cpu().numpy().astype(bool)  # (B, P_max)
                trans_t = sample.trans_t.cpu().numpy()  # (B, P_max, 3)
                aatypes_t = sample.aatypes_t.cpu().numpy()  # (B, P_max)
                motif_mask = sample.motif_mask.cpu().numpy().astype(bool)  # (B, P_max)
                t_val = sample.t[0].item() if sample.t.numel() > 0 else 0.0

                for i in range(num_plots):
                    valid_i = valid_mask[i]
                    trans_alive = trans_t[i][valid_i]  # (P, 3)
                    aatypes_alive = aatypes_t[i][valid_i]  # (P,)
                    motif_alive = motif_mask[i][valid_i]  # (P,)
                    n_alive = trans_alive.shape[0]

                    # Update sequence bar using pre-created artists
                    rectangles, texts, motif_rects, letters, colors = seq_artists[i]
                    if n_alive > 0:
                        BranchingFlowVisualizer._update_sequence_bar(
                            rectangles,
                            texts,
                            motif_rects,
                            letters,
                            colors,
                            aatypes_alive,
                            motif_alive,
                        )
                    else:
                        # Hide all artists when no sequence
                        for rect, text, motif_rect in zip(
                            rectangles, texts, motif_rects
                        ):
                            rect.set_visible(False)
                            text.set_visible(False)
                            motif_rect.set_visible(False)

                    # Update 3D structure using pre-created scatter artist
                    BranchingFlowVisualizer._update_3d_scatter(
                        scat=scatter_artists[i],
                        ax=axes_3d[i],
                        trans_alive=trans_alive,
                        motif_alive=motif_alive,
                        max_points=max_seq_len,
                        t_val=t_val,
                    )

                writer.grab_frame()

        plt.close(fig)
        return anim_path

    def visualize_corruption(
        self,
        batch: DataBatch,
        out_dir: Optional[str] = None,
        times: Optional[List[float]] = None,
        filename: str = "corruption",
    ) -> str:
        """Create a corruption trajectory and plot it."""
        self.interpolant.set_device(batch.trans_1.device)
        if times is None:
            times = list(np.linspace(0.0, 1.0, 50))
        times = sorted(times)

        num_batch = batch.trans_1.shape[0]
        device = batch.trans_1.device
        tree = batch.tree.to(device)

        # Define consistent base samples for the whole trajectory (in aligned space)
        # Broadcast x1 to get aligned space for motif preservation
        trans_1_aligned = tree.broadcast_to_leaves(
            batch.trans_1.to(device), fill_value=0
        )
        aatypes_1_aligned = tree.broadcast_to_leaves(
            batch.aatypes_1.to(device), fill_value=MASK_TOKEN_INDEX
        )
        trans_0 = self.translation_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=trans_1_aligned,
            device=device,
        )
        aatypes_0 = self.interpolant.aatypes_coupler.sample_base(
            motif_mask=tree.motif_mask,
            x1=aatypes_1_aligned,
            device=device,
        )

        # Build the trajectory
        samples: List[DataCorrupted] = []
        for time in tqdm(times, desc="visualize_corruption() corrupt", leave=False):
            bridged, _ = self.interpolant.corrupt_to(
                batch=batch,
                t=torch.ones(num_batch, device=device) * time,
                trans_0=trans_0,
                aatypes_0=aatypes_0,
            )
            samples.append(bridged.pack_present())

        traj = Trajectory(samples=samples)
        return self.plot_trajectory(traj=traj, out_dir=out_dir, filename=filename)


""" Module """


class BranchFlowModule(pl.LightningModule):
    def __init__(self, cfg: VarcoConfig):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters("cfg")

        self.model = BranchFlowModel(cfg=self.cfg.model)
        self.loss_calculator = BranchFlowLossCalculator()
        self.interpolant = TreeInterpolant()

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
        self.log("time/backward_ms", backward_time * 1000)

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
        loss = self.loss_calculator.calculate(
            batch=corrupted, pred=pred, couplings=couplings
        )
        loss_time = time.perf_counter() - loss_start

        self.log("L/train", loss.total_loss, prog_bar=True)
        self.log("L/trans", loss.base_trans_loss, prog_bar=True)
        self.log("L/cdist", loss.pairwise_loss)
        self.log("L/seq", loss.base_seq_loss, prog_bar=True)
        self.log("L/split", loss.split_token_loss, prog_bar=True)
        self.log("L/split_pooled", loss.split_pooled_loss)
        self.log("L/del", loss.del_loss, prog_bar=True)
        self.log("aux/t", bridged.t.mean())

        # Timing statistics (in milliseconds)
        self.log("t/corrupt", corrupt_time * 1000)
        self.log("t/forward", forward_time * 1000)
        self.log("t/loss", loss_time * 1000)

        # Corruption time as function of batch size
        batch_size = corrupted.trans_t.shape[0]
        self.log("aux/batch_size", float(batch_size))
        self.log(
            "t/corrupt_per_batch", corrupt_time * 1000 / batch_size, prog_bar=False
        )

        # Corruption time as function of t value (bucketed)
        mean_t = bridged.t.mean().detach().item()
        t_bucket = round(mean_t * 10) / 5.0  # Round to nearest 0.2
        self.log(f"t/corrupt_t{t_bucket:.1f}_ms", corrupt_time * 1000, prog_bar=False)

        # MPS clean up
        if batch_idx % 100 == 0 and torch.backends.mps.is_available():
            alloc = torch.mps.current_allocated_memory() / 1e9
            drv = torch.mps.driver_allocated_memory() / 1e9
            print(f"step {batch_idx} mps alloc={alloc:.2f}GB driver={drv:.2f}GB")
            gc.collect()
            torch.mps.empty_cache()

        return loss.total_loss

    def validation_step(self, batch: DataBatch, batch_idx: int) -> None:
        self.interpolant.set_device(self.device)

        sample_traj = self.interpolant.sample(
            model=self.model,
            data=batch,
        )

        viz = BranchingFlowVisualizer(sigma=1.0)
        val_dir = os.path.join(
            self.cfg.inference.predict_dir, "val", f"epoch{self.current_epoch:03d}"
        )
        viz.plot_trajectory(
            sample_traj,
            out_dir=val_dir,
            filename=f"val_sample_{batch_idx}",
        )

        # TODO compute a loss (e.g. folding validation) use for model checkpointer


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

    def setup(self):
        pl.seed_everything(self.cfg.shared.seed, workers=True)

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
        print(f"📁 Output directory: {predict_dir}")

        self.data_module = ProteinDataModule(
            dataset=ProteinDataset(cfg=self.cfg.dataset),
            cfg=self.cfg.data,
        )

        self.module = BranchFlowModule(cfg=self.cfg)
        self.logger.info("\n" + str(ModelSummary(self.module, max_depth=2)))

    def debug(self):
        debug_dir = os.path.join(self.cfg.inference.predict_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Plot corruption planning trees
        # for i in range(10):
        #     datum = self.data_module.dataset[i]
        #     datum.tree_plan.plot(
        #         path=os.path.join(debug_dir, f"init_tree_plan_{i}.png")
        #     )

        # Visualize corruption processes
        # 0 for deterministic bridges, >0 for stochasticity
        viz = BranchingFlowVisualizer(sigma=0.0)
        data_loader = self.data_module.train_dataloader(rank=0, num_replicas=1)
        for i, debug_batch in enumerate(data_loader):
            if i > 10:
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
                f"Checkpoints, config, validations etc. will be saved to: {ckpt_dir}"
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

        callbacks.append(TQDMProgressBar(refresh_rate=1))

        # Model checkpoints
        checkpoint_cfg = self.cfg.experiment.checkpointer.asdict()
        checkpoint_cfg["monitor"] = "L/train"
        callbacks.append(ModelCheckpoint(**checkpoint_cfg))

        self.save_config(wandb_logger=wandb_logger)

        self.logger.info("Setting up Trainer...")
        trainer = Trainer(
            **self.cfg.experiment.trainer.asdict(),
            callbacks=callbacks,
            logger=wandb_logger,
            use_distributed_sampler=False,  # TODO - ddp
            devices=self.cfg.experiment.num_devices,
            enable_model_summary=False,  # manual model summary
        )

        trainer.fit(self.module, datamodule=self.data_module)

        self.logger.info(f"Training complete")
        self.logger.info(f"💾 ckpt saved to {self.cfg.experiment.checkpointer.dirpath}")
        self.logger.info(
            f"🏆 Best validation loss: {self.cfg.experiment.checkpointer.monitor}"
        )

        return self.module


@hydra.main(config_path=".", config_name="varco", version_base=None)
def main(cfg: VarcoConfig):
    cfg = OmegaConf.to_object(cfg)
    cfg = cfg.interpolate()

    experiment = Experiment(cfg=cfg)
    experiment.setup()
    experiment.debug()
    experiment.train()


if __name__ == "__main__":
    main()
