"""
Simple implementation of branching flows.

Sample some proteins of length N (translations, later sequence and rotations) using LengthBatched protein dataset
Sample a motif_mask (only support internal motif scaffolding for now so always have "real" parent)
Sample some starting points X0 for the motifs
Define a tree, which inserts scaffold residues over trajectory with anchors at intermediate time points, and alignment of length A
Sample some intermediate time t
Corrupt to X_t, using brownian bridge from 0 to each anchor, continuing up to time t
A simple model predicts base (endpoint prediction), split (remaining children count), and deletion (destined to delete probability)
Losses are base (MSE), split (binned cross entropy), and deletion (BCE)

Then, a sampler (no tree) iterates to get base (endpoint prediction), sample split events, sample deletion events
Evaluation: MMD between X1 and the sampled points, and compare size distribution

Features:
- add deletions (death times, model predictions, update is_alive mask, etc.)
- support sampling, add validation loss and validation_step (and data loader that includes scaffold nuclei)
- support rotations with an IGSO(3) bridge
- handle no motifs, just scaffold nuclei
- support sequence with an protein insertion logits (and sequence logits)
- support a mini frozen ESM model up front (once we have sequence)

"""

import gc
import math
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Generic, List, Optional, Tuple, TypeVar

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from sklearn import datasets
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from cogeneration.config.base import (
    Config,
    DatasetConfig,
    DatasetFilterConfig,
    DatasetInpaintingConfig,
    DatasetInpaintingMotifStrategy,
)
from cogeneration.data.const import ANG_TO_NM_SCALE, NM_TO_ANG_SCALE
from cogeneration.data.noise_mask import centered_gaussian
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.protein_dataloader import LengthBatcher
from cogeneration.models.embed import get_index_embedding
from cogeneration.models.utils import get_model_size_str
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.embed import PositionalEmbeddingMethod
from cogeneration.type.task import DataTask

""" Constants """

# number of possible remaining insertions per root
n_insertion_logits = 128 + 1  # +1 for 0 insertion


""" Tree Plan """

# B = batch size
# N = sampled points in data (t=1)
# R = root points in base (t=0)
# A = aligned number of points in constructed tree with coupling + deletions
# P = present points at time t (R <= P <= A)
# P_max = batch max length of P packed points


@dataclass
class TreePlan:
    """Per-sample (non-batched) tree topology and sampled times (domain-agnostic)."""

    num_leaves: int  # N
    num_nodes: int  # A
    parent_idx: torch.Tensor  # (A,) long, -1 for roots
    children_idx: torch.Tensor  # (A, 2) long, -1 for leaves
    total_leaves: torch.Tensor  # (A,) long, descendant leaf counts
    birth_time: torch.Tensor  # (A,) float, segment start time; roots are 0
    split_time: torch.Tensor  # (A,) float, segment end time; leaves are +inf
    topo_order: torch.Tensor  # (A,) long, structural topo (parent-before-child)
    roots: torch.Tensor  # (R,) long, root node ids
    motif_mask: torch.Tensor  # (N,) bool
    group_ids: torch.Tensor  # (N,) long

    @classmethod
    def generate(
        cls,
        x1: torch.Tensor,
        motif_mask: torch.Tensor,
        seed: Optional[int] = None,
        min_t: float = 0.001,
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
    ) -> "TreePlan":
        """Generate a simple planar coalescent tree plan with sampled split/birth times.

        - Scaffold spans (motif_mask == False) are grouped into disjoint contiguous groups.
        - Each motif position is its own singleton group (never merges).
        - For each scaffold span group, we sample a number of "nuclei" K in
            [min_scaffold_nuclei, min(max_scaffold_nuclei, span_len)]
        and we coalesce the span only until K active nodes remain.
        Those K nodes become roots for that scaffold span.
        - Merges are biased toward scaffold boundaries (left/right) with small probability of
        random interior merges for variability.

        Note: this intentionally allows scaffold tokens to be present at t=0 (the nuclei), in
        addition to motif tokens. This avoids needing special boundary-sourcing logic.
        """
        if x1.ndim != 2:
            raise ValueError(f"Expected x1 to have shape (N, d); got {tuple(x1.shape)}")
        if motif_mask.ndim != 1 or motif_mask.shape[0] != x1.shape[0]:
            raise ValueError(
                f"Expected motif_mask to have shape (N,) matching x1; got {tuple(motif_mask.shape)}"
            )

        device = x1.device
        N = x1.shape[0]
        motif_mask_b = motif_mask.to(torch.bool)

        if min_scaffold_nuclei < 1:
            raise ValueError("min_scaffold_nuclei must be >= 1")
        if max_scaffold_nuclei < 1:
            raise ValueError("max_scaffold_nuclei must be >= 1")
        if min_scaffold_nuclei > max_scaffold_nuclei:
            raise ValueError("min_scaffold_nuclei must be <= max_scaffold_nuclei")

        # Sample split times top-down from uniform base time distribution + exponential waiting time
        # Use a local torch.Generator for determinism and to avoid global RNG interaction.
        rng = torch.Generator(device="cpu")
        if seed is None:
            # fall back to a nondeterministic seed
            seed = int(torch.seed() % (2**31 - 1))
        rng.manual_seed(int(seed))

        # Group ids:
        # - each contiguous scaffold span (motif_mask == False) is its own group (0..S-1)
        # - each motif position (motif_mask == True) is its own singleton group after scaffold groups
        group_ids = torch.full((N,), -1, dtype=torch.long, device=device)

        # Scaffold span groups
        span_gid = 0
        i = 0
        while i < N:
            if bool(motif_mask_b[i].item()):
                i += 1
                continue
            j = i
            while j < N and (not bool(motif_mask_b[j].item())):
                j += 1
            # [i, j) is a contiguous scaffold span
            group_ids[i:j] = span_gid
            span_gid += 1
            i = j

        # Motif singleton groups
        motif_positions = (
            torch.nonzero(motif_mask_b, as_tuple=False).squeeze(-1).tolist()
        )
        for k, pos in enumerate(motif_positions):
            group_ids[pos] = span_gid + k

        if bool((group_ids < 0).any().item()):
            raise RuntimeError("Failed to assign group_ids for all positions")

        # Initialize leaf nodes
        parent: List[int] = [-1] * N
        children: List[List[int]] = [[-1, -1] for _ in range(N)]
        weight: List[int] = [1 for _ in range(N)]

        # Active lists per group in sequence order
        groups: Dict[int, List[int]] = {}
        for i in range(N):
            gid = int(group_ids[i].item())
            groups.setdefault(gid, []).append(i)

        # Coalesce within each group by repeated adjacent merges.
        # For scaffold span groups (gid < span_gid), bias merges toward the boundaries so the
        # coalescent "collapses" inwards from both ends (both boundaries can in-fill).
        # For motif singleton groups, len(active) == 1 so nothing happens.
        p_interior = 0.5  # probability to merge a random interior adjacent pair

        def rand_int(high: int) -> int:
            return int(torch.randint(0, high, (1,), generator=rng).item())

        def rand_float() -> float:
            return float(torch.rand(1, generator=rng).squeeze().item())

        roots: List[int] = []
        for gid, active in groups.items():
            is_scaffold_span = gid < span_gid

            # For scaffold spans: keep K nuclei (active nodes) instead of collapsing to 1.
            if is_scaffold_span:
                span_len = len(active)
                k_hi = min(max_scaffold_nuclei, span_len)
                k_lo = min(min_scaffold_nuclei, k_hi)
                target_k = k_lo if k_lo == k_hi else (k_lo + rand_int(k_hi - k_lo + 1))
            else:
                # Motif singleton groups are already len==1; enforce target 1.
                target_k = 1

            while len(active) > target_k:
                # Choose which adjacent pair to merge.
                if (
                    is_scaffold_span
                    and len(active) > 2
                    and (rand_float() < (1.0 - p_interior))
                ):
                    # Boundary-biased: merge leftmost or rightmost adjacent pair (50/50)
                    if rand_float() < 0.5:
                        i0 = 0
                    else:
                        i0 = len(active) - 2
                else:
                    # Uniform random adjacent pair
                    i0 = rand_int(len(active) - 1)

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

        roots = sorted(set(roots))

        A = len(parent)

        birth = torch.full((A,), float("inf"), dtype=torch.float32)
        split = torch.full((A,), float("inf"), dtype=torch.float32)

        # Roots start at time 0
        for r in roots:
            birth[r] = 0.0

        def sample_exp1() -> float:
            # Exp(1) via inverse CDF from Uniform(0,1)
            u = float(torch.rand((), generator=rng).clamp_min(1e-12).item())
            return -math.log(u)

        def sample_split_time_uniform(W: int, t0: float) -> float:
            # next_split_time for Uniform[0,1]: 1 - (1 - t0) * exp(-E / (W-1))
            if W <= 1:
                return float("inf")
            m = W - 1
            E = sample_exp1()
            s = 1.0 - (1.0 - t0) * math.exp(-E / float(m))
            s = max(min_t, min(1.0 - min_t, s))
            return s

        # Structural topo: in this construction, internal nodes are appended so parent ids > child ids.
        topo_order = torch.arange(A - 1, -1, -1, dtype=torch.long)

        # Traverse parents before children and propagate birth times
        for node in topo_order.tolist():
            t0 = float(birth[node].item())
            if not math.isfinite(t0):
                continue

            W = int(weight[node])
            if W <= 1:
                split[node] = float("inf")
                continue

            st = sample_split_time_uniform(W=W, t0=t0)
            split[node] = st

            c0, c1 = children[node]
            for c in (c0, c1):
                if c >= 0:
                    birth[c] = st

        parent_idx = torch.tensor(parent, dtype=torch.long, device=device)
        children_idx = torch.tensor(children, dtype=torch.long, device=device)
        total_leaves = torch.tensor(weight, dtype=torch.long, device=device)
        roots_t = torch.tensor(roots, dtype=torch.long, device=device)

        birth_time = birth.to(device=device)
        split_time = split.to(device=device)
        topo_order_t = topo_order.to(device=device)

        return cls(
            num_leaves=N,
            num_nodes=A,
            parent_idx=parent_idx,
            children_idx=children_idx,
            total_leaves=total_leaves,
            birth_time=birth_time,
            split_time=split_time,
            topo_order=topo_order_t,
            roots=roots_t,
            motif_mask=motif_mask_b.to(device=device),
            group_ids=group_ids,
        )

    def validate(self) -> None:
        A = int(self.num_nodes)
        N = int(self.num_leaves)

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
        if self.topo_order.shape != (A,):
            raise ValueError("topo_order shape mismatch")
        if self.motif_mask.shape != (N,):
            raise ValueError("motif_mask shape mismatch")

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
        total = self.total_leaves.detach().to("cpu").numpy()
        roots = set(self.roots.detach().to("cpu").numpy().tolist())
        motif_mask = self.motif_mask.detach().to("cpu").numpy().astype(bool)

        A = int(self.num_nodes)
        N = int(self.num_leaves)

        motif_line_color = (
            "0.6"  # grey: straight lifelines for motif singleton residues
        )
        motif_marker_color = "black"
        scaffold_color = "red"

        # Compute a planar x-position per node based on descendant leaf positions.
        # This keeps internal anchors visually located within their scaffold span, even though
        # internal node ids are appended and would otherwise plot far to the right.
        x_pos = np.full((A,), np.nan, dtype=np.float32)

        # Leaf nodes (in this toy) are the original indices 0..N-1
        for i in range(int(self.num_leaves)):
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
            y1 = float(split[i]) if np.isfinite(split[i]) else 1.0
            # Skip padded/uninitialized nodes (shouldn't exist in per-sample plan)
            if not np.isfinite(y0):
                continue
            y0 = max(0.0, min(1.0, y0))
            y1 = max(0.0, min(1.0, y1))
            xi = float(x_pos[i])
            is_leaf = int(total[i]) == 1
            is_motif_leaf = bool(motif_mask[i]) if (is_leaf and i < N) else False
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
        motif_roots = [r for r in sorted(roots) if (r < N and bool(motif_mask[r]))]
        scaffold_roots = [r for r in sorted(roots) if r not in set(motif_roots)]

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

        motif_leaf_ids = [i for i in range(N) if bool(motif_mask[i])]
        scaffold_leaf_ids = [i for i in range(N) if not bool(motif_mask[i])]
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

    parent_idx: torch.Tensor  # (B, A) long
    children_idx: torch.Tensor  # (B, A, 2) long
    birth_time: torch.Tensor  # (B, A) float
    split_time: torch.Tensor  # (B, A) float
    total_leaves: torch.Tensor  # (B, A) long
    topo_order: torch.Tensor  # (B, A) long

    roots: torch.Tensor  # (B, R_max) long, -1 padded
    roots_mask: torch.Tensor  # (B, R_max) bool

    def to(self, device: torch.device) -> "BatchedTreePlan":
        """Move all tensors to the specified device."""
        return BatchedTreePlan(
            parent_idx=self.parent_idx.to(device),
            children_idx=self.children_idx.to(device),
            birth_time=self.birth_time.to(device),
            split_time=self.split_time.to(device),
            total_leaves=self.total_leaves.to(device),
            topo_order=self.topo_order.to(device),
            roots=self.roots.to(device),
            roots_mask=self.roots_mask.to(device),
        )

    def present_mask(
        self,
        t: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, A) bool
        """Mask of nodes present (alive) at time t."""
        return ((self.birth_time <= t[:, None]) & (t[:, None] < self.split_time)).bool()

    def remaining_insertions_t(
        self,
        t: torch.Tensor,  # (B,)
    ) -> torch.Tensor:  # (B, A) long
        """Remaining insertions per node at time t.

        Definition: for each present node, remaining insertions is (total_leaves - 1).
        Non-present (and padded) nodes are 0.
        """
        present = self.present_mask(t=t).to(torch.long)
        remaining = (self.total_leaves - 1).clamp_min(0) * present
        return remaining.clamp_max(n_insertion_logits - 1).to(torch.long)

    @classmethod
    def collate(cls, plans: List["TreePlan"]) -> "BatchedTreePlan":
        B = len(plans)
        if B == 0:
            raise ValueError("Empty batch")

        A_max = max(int(p.num_nodes) for p in plans)
        R_max = max(int(p.roots.numel()) for p in plans)

        parent_idx = torch.full((B, A_max), -1, dtype=torch.long)
        children_idx = torch.full((B, A_max, 2), -1, dtype=torch.long)
        birth_time = torch.full((B, A_max), float("inf"), dtype=torch.float32)
        split_time = torch.full((B, A_max), float("inf"), dtype=torch.float32)
        total_leaves = torch.zeros((B, A_max), dtype=torch.long)
        topo_order = torch.zeros((B, A_max), dtype=torch.long)
        roots = torch.full((B, R_max), -1, dtype=torch.long)
        roots_mask = torch.zeros((B, R_max), dtype=torch.bool)

        for b, p in enumerate(plans):
            A_i = int(p.num_nodes)
            R_i = int(p.roots.numel())

            parent_idx[b, :A_i] = p.parent_idx.to(torch.long)
            children_idx[b, :A_i, :] = p.children_idx.to(torch.long)
            birth_time[b, :A_i] = p.birth_time.to(torch.float32)
            split_time[b, :A_i] = p.split_time.to(torch.float32)
            total_leaves[b, :A_i] = p.total_leaves.to(torch.long)

            topo_order[b, :A_i] = p.topo_order.to(torch.long)
            if A_i < A_max:
                topo_order[b, A_i:] = torch.arange(A_i, A_max, dtype=torch.long)

            roots[b, :R_i] = p.roots.to(torch.long)
            roots_mask[b, :R_i] = True

        return cls(
            parent_idx=parent_idx,
            children_idx=children_idx,
            birth_time=birth_time,
            split_time=split_time,
            total_leaves=total_leaves,
            topo_order=topo_order,
            roots=roots,
            roots_mask=roots_mask,
        )


@dataclass
class DataSample:
    """Per-sample data (length N) at time t=1."""

    trans_1: torch.Tensor  # (N, 3)
    motif_mask: torch.Tensor  # (N,) bool
    tree_plan: TreePlan


@dataclass
class DataBatch:
    """Batched data for training."""

    trans_1: torch.Tensor  # (B, N, 3)
    motif_mask: torch.Tensor  # (B, N) bool
    tree: BatchedTreePlan


@dataclass
class DataCorrupted:
    """Model input: corrupted and packed (length P_max) points present at time t"""

    t: torch.Tensor  # (B,)
    trans_t: torch.Tensor  # (B, P_max, 3)
    birth_time: torch.Tensor  # (B, P_max) 0.0 for roots
    remaining_insertions: Optional[torch.Tensor] = (
        None  # (B, P_max) supervised target, remaining splits per present token
    )

    @property
    def alive_mask(self) -> torch.Tensor:  # (B, P_max)
        return (self.birth_time <= self.t[:, None]).bool()

    @property
    def remaining_total(self) -> torch.Tensor:
        # batch.remaining_insertions: (B, P_max) padded with 0
        # batch.alive_mask: (B, P_max) True for real packed tokens, False for padding
        remaining_total = (self.remaining_insertions * self.alive_mask.long()).sum(
            dim=1
        )  # (B,)
        remaining_total = remaining_total.clamp_max(n_insertion_logits - 1).long()
        return remaining_total


@dataclass
class DataBridged:
    """Corrupted and aligned (length A) points at time t"""

    t: torch.Tensor  # (B,)
    trans_t: torch.Tensor  # (B, A, 3)
    birth_time: torch.Tensor  # (B, A) 0.0 for roots
    present_mask: torch.Tensor  # (B, A)
    remaining_insertions: torch.Tensor  # (B, A) target count per aligned node

    @staticmethod
    def pack_present_indices(
        present_mask: torch.Tensor,  # (B, A) bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        """
        Derive packed indices deterministically from a (B, A) present mask.

        Returns:
            idx_pack: (B, P_max) aligned indices for packed slots
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
        B, A, D = self.trans_t.shape
        idx_pack, pack_mask, P_b, P_max = self._pack_indices()

        trans_t_pack = self.trans_t.gather(
            1, idx_pack.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, P_max, D)
        birth_time_pack = self.birth_time.gather(1, idx_pack)  # (B, P_max)
        remaining_insertions_pack = self.remaining_insertions.gather(
            1, idx_pack
        )  # (B, P_max)

        trans_t_pack = trans_t_pack * pack_mask.unsqueeze(-1).float()  # zero pad
        birth_time_pack = torch.where(
            pack_mask,
            birth_time_pack,
            torch.full_like(
                birth_time_pack, float("inf")
            ),  # infinite birth time for pad tokens
        )
        remaining_insertions_pack = torch.where(
            pack_mask,
            remaining_insertions_pack,
            torch.zeros_like(remaining_insertions_pack),  # 0 pad
        )

        return DataCorrupted(
            t=self.t,
            trans_t=trans_t_pack,
            birth_time=birth_time_pack,
            remaining_insertions=remaining_insertions_pack,
        )

    def validate(self) -> None:
        B, A, D = self.trans_t.shape
        if self.birth_time.shape != (B, A):
            raise ValueError("birth_time shape mismatch")
        if self.present_mask.shape != (B, A):
            raise ValueError("present_mask shape mismatch")
        if self.remaining_insertions.shape != (B, A):
            raise ValueError("remaining_insertions shape mismatch")
        if D != 3:
            raise ValueError("trans_t last dim must be 3")


@dataclass
class ModelPrediction:
    """t=1 prediction for present state (length P)"""

    pred_trans_1: torch.Tensor  # (B, P, 3) base; predicted final/anchor positions
    pred_split_logits: torch.Tensor  # (B, P, n_insertion_logits) split logits per token
    pred_split_pooled_logits: (
        torch.Tensor
    )  # (B, n_insertion_logits) pooled split logits


@dataclass
class SampleTrajectory:
    samples: List[DataCorrupted]  # samples at each time step
    pred: List[ModelPrediction]  # predictions at each time step


""" Protein Dataset + DataLoader """


class ProteinDataset(BaseDataset):
    """Wrapper to simplify BaseDataset and extract relevant features"""

    def __init__(
        self,
    ):
        # Define DatasetConfig for inpainting
        dataset_cfg = DatasetConfig(
            # Use PDB and AFDB
            enable_cogeneration_pdb=True,
            enable_cogeneration_afdb=True,
            enable_cogeneration_redesigns=False,
            enable_multiflow_redesigned=False,
            enable_multiflow_synthetic=False,
            debug_head_samples=1000,
            filter=DatasetFilterConfig(
                max_num_res=256,
                num_chains=[1, 2],
            ),
            # Always bridge a scaffold inside a single chain
            inpainting=DatasetInpaintingConfig(
                strategy=DatasetInpaintingMotifStrategy.single_scaffold,
            ),
            # override interpolated props
            max_eval_length=256,
            seed=0,
        )
        super().__init__(
            cfg=dataset_cfg,
            task=DataTask.inpainting,
            eval=False,
            use_test=False,
        )

    def __getitem__(self, idx) -> DataSample:
        feats = super().__getitem__(idx)

        trans_1 = feats[bp.trans_1]
        motif_mask = feats[bp.motif_mask]

        tree_plan = TreePlan.generate(x1=trans_1, motif_mask=motif_mask)
        tree_plan.validate()

        return DataSample(
            trans_1=trans_1,
            motif_mask=motif_mask,
            tree_plan=tree_plan,
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
            prefetch_factor=None if num_workers == 0 else 2,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch: List[DataSample]) -> DataBatch:
        trans_1 = torch.stack([item.trans_1 for item in batch])  # (B, N, 3)
        motif_mask = torch.stack([item.motif_mask for item in batch])  # (B, N)

        plans = [item.tree_plan for item in batch]
        tree = BatchedTreePlan.collate(plans)

        return DataBatch(
            trans_1=trans_1,
            motif_mask=motif_mask,
            tree=tree,
        )


class ProteinDataModule(pl.LightningDataModule):
    """DataModule for ProteinDataset"""

    def __init__(
        self,
        dataset: ProteinDataset,
        num_workers: int = max(1, os.cpu_count() - 2),
    ):
        super().__init__()
        self._full_cfg = Config().interpolate()
        self.dataset = dataset
        self.num_workers = num_workers

    def train_dataloader(self, rank=None, num_replicas=None) -> DataLoader:
        batch_sampler = LengthBatcher(
            sampler_cfg=self._full_cfg.data.sampler,
            metadata_csv=self.dataset.csv,
            modeled_length_col=self.dataset.cfg.modeled_trim_method.to_dataset_column(),
            rank=rank or 0,
            num_replicas=num_replicas or 1,
        )

        return ProteinDataLoader(
            dataset=self.dataset,
            batch_sampler=batch_sampler,
            num_workers=self.num_workers,
        )

    def val_dataloader(self) -> DataLoader:
        return ProteinDataLoader(self.dataset)


""" Model """


class BranchFlowModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.point_dim = 3
        self.pos_embed_dim = 32

        # x_t (point_dim), t (1), birth_time (1), pos_embed
        self.input_dim = self.point_dim + 1 + 1 + self.pos_embed_dim
        self.model_dim = 64

        self.input_proj = nn.Linear(self.input_dim, self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=8,
            dim_feedforward=4 * self.model_dim,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.x1_pred = nn.Linear(self.model_dim, self.point_dim)
        self.split_logits_pred = nn.Linear(self.model_dim, n_insertion_logits)
        self.split_pooled_logits_pred = nn.Linear(self.model_dim, n_insertion_logits)

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        B, P, _ = batch.trans_t.shape

        # attention mask to mask out positions not alive yet / padded
        key_padding_mask = ~batch.alive_mask.bool()

        pos_embed = get_index_embedding(
            torch.arange(P, device=batch.trans_t.device).unsqueeze(0).expand(B, -1),
            embed_size=self.pos_embed_dim,
            max_len=1024,
            pos_embed_method=PositionalEmbeddingMethod.rotary,
        )
        pos_embed = pos_embed * batch.alive_mask.unsqueeze(-1).float()

        x_t = torch.cat(
            [
                batch.trans_t,  # (B, P, 3)
                batch.t[:, None, None].expand(B, P, 1),  # (B, P, 1)
                batch.birth_time[:, :, None]
                .float()
                .clamp(0.0, 1.0),  # (B, P, 1) clamp +inf padding
                pos_embed,  # (B, P, pos_embed_dim)
            ],
            dim=-1,
        )

        x_t = self.input_proj(x_t)
        x_t = self.transformer(x_t, src_key_padding_mask=key_padding_mask)

        x1_pred = self.x1_pred(x_t)

        split_logits = self.split_logits_pred(x_t)  # (B, P, n_insertion_logits)

        # Masked sum pool over alive tokens to predict total remaining insertions per example
        alive = batch.alive_mask.bool()  # (B, P)
        pooled = (x_t * alive.unsqueeze(-1).float()).sum(dim=1)  # (B, model_dim)
        split_pooled_logits = self.split_pooled_logits_pred(
            pooled
        )  # (B, n_insertion_logits)

        return ModelPrediction(
            pred_trans_1=x1_pred,
            pred_split_logits=split_logits,
            pred_split_pooled_logits=split_pooled_logits,
        )


""" Tree Coupling """


class Coupling(ABC):
    """Coupling tracks domain-specific anchors, and the corruption tree plan."""

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
        self, num_batch: int, num_roots: int, device: torch.device
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def corrupt(
        self,
        x1: torch.Tensor,
        tree: BatchedTreePlan,
        t: torch.Tensor,
        x0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, CouplingT]:
        raise NotImplementedError


@dataclass
class TranslationCoupling(Coupling):
    # domain-specific anchors (used for supervision)
    anchors: torch.Tensor  # (B, A, 3)

    # shared tree plan (domain-agnostic)
    tree: BatchedTreePlan


class TranslationCoupler(Coupler[TranslationCoupling]):
    def __init__(self, sigma: Optional[float] = 1.0):
        self.sigma = sigma

    def sample_base(
        self, num_batch: int, num_roots: int, device: torch.device
    ) -> torch.Tensor:
        return (
            centered_gaussian(num_batch, num_roots, n_bb_atoms=3, device=device)
            * NM_TO_ANG_SCALE
        )

    def build_anchor_alignment(
        self, x1: torch.Tensor, tree: BatchedTreePlan
    ) -> torch.Tensor:
        """Build translation anchors for all nodes (leaves + internal) from leaf endpoints x1.

        Assumptions for this toy:
        - Leaves correspond to x1 in positions [0..N-1].
        - Internal nodes ("anchors") are derived purely from topology + descendant weights.
        """
        if x1.ndim != 3 or x1.shape[-1] != 3:
            raise ValueError(
                f"Expected x1 to have shape (B, N, 3); got {tuple(x1.shape)}"
            )

        device = x1.device
        B, N, _ = x1.shape
        A = tree.parent_idx.shape[1]

        anchor = torch.zeros((B, A, 3), dtype=torch.float32, device=device)
        anchor[:, :N, :] = x1.to(torch.float32)

        # Build internal anchors bottom-up (children -> parent).
        # In this construction, parent ids > child ids, so iterating i from 0..A-1 ensures children exist first.
        for i in range(A):
            w_i = tree.total_leaves[:, i]
            exists = w_i > 0
            is_internal = w_i > 1
            valid = exists & is_internal
            if not bool(valid.any()):
                continue

            kids = tree.children_idx[:, i, :]  # (B, 2)
            k = kids.clamp_min(0)
            child_anchor = anchor.gather(
                1, k.unsqueeze(-1).expand(-1, 2, 3)
            )  # (B, 2, 3)

            child_w = tree.total_leaves.gather(1, k)  # (B, 2)
            wsum = (child_w.sum(dim=1).clamp_min(1)).to(torch.float32)  # (B,)

            merged = child_anchor[:, 0, :] * (
                child_w[:, 0].to(torch.float32) / wsum
            ).unsqueeze(-1) + child_anchor[:, 1, :] * (
                child_w[:, 1].to(torch.float32) / wsum
            ).unsqueeze(
                -1
            )  # (B, 3)

            bidx = torch.arange(B, device=device)
            anchor[bidx[valid], i, :] = merged[valid]

        return anchor

    def _bridge_brownian(
        self,
        x_start: torch.Tensor,  # (M, 3)
        x_end: torch.Tensor,  # (M, 3)
        s: torch.Tensor,  # (M,)
        t0: torch.Tensor,  # (M,)
    ) -> torch.Tensor:
        """Sample Brownian-bridge marginal at time s from (x_start at t0) to (x_end at 1)."""
        denom = (1.0 - t0).clamp_min(1e-6)
        u = ((s - t0) / denom).clamp(0.0, 1.0)
        mean = x_start + u.unsqueeze(-1) * (x_end - x_start)

        if self.sigma is None:
            return mean

        var = ((s - t0).clamp_min(0.0) * (1.0 - s).clamp_min(0.0) / denom).clamp_min(
            0.0
        )
        std = (var.sqrt() * self.sigma).to(mean.dtype)
        eps = torch.randn_like(mean)
        return mean + std.unsqueeze(-1) * eps

    def corrupt(
        self,
        x1: torch.Tensor,  # (B, N, 3)
        tree: BatchedTreePlan,
        t: torch.Tensor,  # (B,)
        x0: Optional[torch.Tensor] = None,  # (B, R_max, 3)
    ) -> Tuple[torch.Tensor, TranslationCoupling]:
        """Corrupt translations to time t using the shared tree plan.

        Returns:
            trans_t_aligned: (B, A, 3)
            coupling: minimal coupling info for losses (anchors + tree)
        """
        if x1.ndim != 3 or x1.shape[-1] != 3:
            raise ValueError(
                f"Expected x1 to have shape (B, N, 3); got {tuple(x1.shape)}"
            )

        device = x1.device
        B, N, _ = x1.shape
        A = int(tree.parent_idx.shape[1])

        # Determine how many roots are used per example in this batch
        R_max = int(tree.roots_mask.sum(dim=1).max().item())
        if R_max <= 0:
            raise ValueError("No roots in tree plan")

        # Sample / validate base roots
        if x0 is None:
            x0 = self.sample_base(num_batch=B, num_roots=R_max, device=device)
        if x0.ndim != 3 or x0.shape[0] != B or x0.shape[-1] != 3:
            raise ValueError(
                f"Expected x0 to have shape (B, R, 3); got {tuple(x0.shape)}"
            )

        # Build x0_aligned by broadcasting first root and then writing per-example roots into tree root slots
        x0_aligned = x0[:, :1, :].expand(B, A, 3).contiguous().clone()
        for b in range(B):
            roots_i = tree.roots[b][tree.roots_mask[b]].to(torch.long)
            for j, r in enumerate(roots_i.tolist()):
                if j >= x0.shape[1]:
                    break
                x0_aligned[b, r, :] = x0[b, j, :]

        # Build domain anchors (leaf endpoints + internal merges)
        anchor_aligned = self.build_anchor_alignment(x1=x1, tree=tree)

        # Save the coupling for losses
        coupling = TranslationCoupling(anchors=anchor_aligned, tree=tree)

        # Track creation states at each node's birth time (shared across siblings)
        creation_state = (
            x0_aligned.to(dtype=torch.float32).clone().contiguous()
        )  # (B, A, 3)

        # Topological pass: sample each parent's state at split time once, then assign to both children
        for k in range(A):
            node_idx = tree.topo_order[:, k]  # (B,)

            t0 = tree.birth_time.gather(1, node_idx.unsqueeze(1)).squeeze(1)  # (B,)
            st = tree.split_time.gather(1, node_idx.unsqueeze(1)).squeeze(1)  # (B,)

            is_leaf = ~torch.isfinite(st)
            if bool(is_leaf.all()):
                continue

            node_creation = creation_state.gather(
                1, node_idx.view(B, 1, 1).expand(-1, 1, 3)
            ).squeeze(
                1
            )  # (B, 3)
            node_anchor = anchor_aligned.gather(
                1, node_idx.view(B, 1, 1).expand(-1, 1, 3)
            ).squeeze(
                1
            )  # (B, 3)

            node_at_split = self._bridge_brownian(
                x_start=node_creation,
                x_end=node_anchor,
                s=st,
                t0=t0,
            )  # (B, 3)

            kids = (
                tree.children_idx.gather(1, node_idx.view(B, 1, 1).expand(-1, 1, 2))
                .to(torch.long)
                .squeeze(1)
            )  # (B, 2)

            bidx = torch.arange(B, device=device)
            for j in range(2):
                c = kids[:, j]
                valid = (c >= 0) & (~is_leaf)
                if bool(valid.any()):
                    creation_state[bidx[valid], c[valid]] = node_at_split[valid]

        # Evaluate node states at time t (bridge from creation_state at birth_time to anchor at 1)
        t_mat = t.unsqueeze(1).expand(B, A)  # (B, A)
        b_mat = tree.birth_time  # (B, A)
        e_mat = tree.split_time  # (B, A)
        present_mask = (t_mat >= b_mat) & (t_mat < e_mat)

        trans_t = self._bridge_brownian(
            x_start=creation_state.view(-1, 3),
            x_end=anchor_aligned.view(-1, 3),
            s=t_mat.reshape(-1),
            t0=b_mat.reshape(-1),
        ).view(B, A, 3)

        trans_t = torch.where(
            present_mask.unsqueeze(-1), trans_t, torch.zeros_like(trans_t)
        )

        return trans_t, coupling


""" Interpolant supporting tree coupling """


@dataclass
class TreeInterpolant:
    device: torch.device = torch.device("cpu")
    min_t: float = 0.005
    translation_coupler: Coupler[TranslationCoupling] = TranslationCoupler(sigma=1.0)

    def set_device(self, device: torch.device):
        self.device = device

    def corrupt_to(
        self,
        batch: DataBatch,
        t: torch.Tensor,  # (B,)
        x0: Optional[torch.Tensor] = None,
    ) -> Tuple[DataBridged, TranslationCoupling]:
        tree = batch.tree.to(self.device)
        trans_1 = batch.trans_1.to(self.device)
        t = t.to(self.device)
        if x0 is not None:
            x0 = x0.to(self.device)

        trans_t, trans_coupling = self.translation_coupler.corrupt(
            x1=trans_1,
            tree=tree,
            t=t,
            x0=x0,
        )
        trans_coupling.validate()

        bridged = DataBridged(
            t=t,
            trans_t=trans_t,
            birth_time=tree.birth_time,
            present_mask=tree.present_mask(t=t),
            remaining_insertions=tree.remaining_insertions_t(t=t),
        )
        bridged.validate()

        return bridged, trans_coupling

    def corrupt_batch(
        self, batch: DataBatch
    ) -> Tuple[DataBridged, TranslationCoupling]:
        # pick a single time to share across the batch
        shared_t = (
            torch.rand(1, device=self.device) * (1.0 - 2.0 * self.min_t) + self.min_t
        )
        t = torch.ones(batch.trans_1.shape[0], device=self.device) * shared_t  # (B,)

        return self.corrupt_to(batch=batch, t=t)

    def sample(
        self,
        model: BranchFlowModel,
        num_batch: int = 1,
        num_roots: int = 50,
        num_steps: int = 100,
    ) -> SampleTrajectory:
        samples = []
        preds = []

        trans_0 = self.translation_coupler.sample_base(
            num_batch=num_batch, num_roots=num_roots, device=self.device
        )  # (B, R, 3)

        # TODO - we should probably store the motif mask, and embed it
        motif_mask = torch.ones(
            num_batch, num_roots, dtype=torch.bool, device=self.device
        )

        batch = DataCorrupted(
            trans_t=trans_0,
            t=torch.zeros(num_batch, device=self.device),
            birth_time=torch.zeros_like(trans_0[:, :, 0]),
            remaining_insertions=None,
        )
        samples.append(batch)

        for i in range(num_steps):
            t = (i / num_steps) * (1.0 - 2.0 * self.min_t) + self.min_t
            batch.t = torch.ones(num_batch, device=self.device) * t
            pred = model.forward(batch)
            preds.append(pred)

            # Batch updates: euler step for existing points, sample insertion events, sample deletion events
            # TODO

        # after sampling, clean up final sample, e.g. removing deleted points

        return SampleTrajectory(samples=samples, pred=preds)


""" Loss """


@dataclass
class BranchFlowLosses:
    total_loss: torch.Tensor
    base_trans_loss: torch.Tensor  # MSE on translations
    pairwise_loss: torch.Tensor  # local pairwise distance loss
    split_loss: torch.Tensor  # cross-entropy on split logits


@dataclass
class BranchFlowLossCalculator:
    # Time normalization clip (higher weight as t -> 1)
    t_normalize_clip: float = 0.9
    # Local pairwise distance threshold (angstroms)
    proximity_threshold_ang: float = 7.0
    # Loss weights
    pairwise_loss_weight: float = 0.5
    split_loss_weight: float = (
        0.05  # scale down CE loss (129 classes → high raw values)
    )

    def _time_norm_scale(self, t: torch.Tensor) -> torch.Tensor:
        """Compute time-based normalization scale: 1 - min(t, clip).
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
        return loss_per_batch.mean()

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

        return loss_per_batch.mean()

    def calculate(
        self,
        batch: DataCorrupted,
        pred: ModelPrediction,
        coupling: TranslationCoupling,
    ) -> BranchFlowLosses:
        B, P, D = batch.trans_t.shape
        assert pred.pred_trans_1.shape == (B, P, D)

        present_mask = coupling.tree.present_mask(t=batch.t)  # (B, A)
        is_root = (coupling.tree.parent_idx < 0).to(torch.long)
        pad_mask = batch.alive_mask  # (B, P_max)

        # Reconstruct the same packing indices used by TwoMoonsBridged.pack_present()
        idx_pack, pack_mask, P_b, P_max = DataBridged.pack_present_indices(present_mask)

        # Pack anchor targets into (B, P_max, D) in the same order as model inputs/predictions
        anchors_pack = coupling.anchors.gather(
            1, idx_pack.unsqueeze(-1).expand(-1, -1, D)
        )
        anchors_pack = (
            anchors_pack * pack_mask.unsqueeze(-1).float()
        )  # zero pad (optional)

        # Base loss on predicting existing residues' final positions
        base_loss = self._base_trans_loss(
            pred_trans=pred.pred_trans_1,
            target_trans=anchors_pack,
            t=batch.t,
            mask=pad_mask,
        )
        pairwise_loss = self._pairwise_distance_loss(
            pred_trans=pred.pred_trans_1,
            target_trans=anchors_pack,
            t=batch.t,
            mask=pad_mask,
        )
        pairwise_loss = pairwise_loss * self.pairwise_loss_weight

        # split loss: cross-entropy over remaining insertions (tokenwise) + pooled auxiliary head
        # Compute per-example masked mean CE (so variable present token counts don't change magnitude).
        logits = pred.pred_split_logits  # (B, P, C)
        targets = batch.remaining_insertions.to(torch.long)  # (B, P)
        ce_flat = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        ).reshape(B, P)
        pad_mask_f = pad_mask.float()
        denom_tokens = pad_mask_f.sum(dim=1).clamp_min(1.0)  # (B,)
        split_loss_token = ((ce_flat * pad_mask_f).sum(dim=1) / denom_tokens).mean()

        # pooled target is total remaining insertions in this example
        pooled_targets = (coupling.tree.total_leaves * is_root).sum(dim=1)
        pooled_targets = (
            (pooled_targets - present_mask.sum(dim=1))
            .clamp_min(0)
            .clamp_max(n_insertion_logits - 1)
            .to(torch.long)
        )
        split_loss_pooled = nn.CrossEntropyLoss()(
            pred.pred_split_pooled_logits, pooled_targets
        )

        split_loss = split_loss_token + split_loss_pooled
        split_loss = split_loss * self.split_loss_weight

        total_loss = base_loss + pairwise_loss + split_loss

        return BranchFlowLosses(
            total_loss=total_loss,
            base_trans_loss=base_loss,
            pairwise_loss=pairwise_loss,
            split_loss=split_loss,
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

    def visualize_corruption(
        self,
        batch: DataBatch,
        out_dir: Optional[str] = None,
        times: Optional[List[float]] = None,
    ):
        self.interpolant.set_device(batch.trans_1.device)
        if out_dir is None:
            out_dir = tempfile.mkdtemp()
        if times is None:
            times = list(np.linspace(0.0, 1.0, 50))
        times = sorted(times)

        num_batch = batch.trans_1.shape[0]
        num_plots = min(num_batch, 4)  # only plot first 4 structures

        ext, writer = self._get_anim_writer()
        anim_path = os.path.join(out_dir, f"corruption.{ext}")
        os.makedirs(out_dir, exist_ok=True)
        print(f"💾 Saving corruption animation to {anim_path}")

        # define a consistent x0
        x0 = self.translation_coupler.sample_base(
            num_batch=num_batch,
            num_roots=batch.motif_mask.sum(dim=1)[:num_plots].max().item(),
            device=batch.trans_1.device,
        )

        # use trans_1 for camera limits
        trans_1 = batch.trans_1.cpu().numpy()  # (B, N, 3)
        trans_1_min = trans_1.min(axis=1)  # (B, 3)
        trans_1_max = trans_1.max(axis=1)  # (B, 3)

        num_cols = min(num_plots, 2)
        num_rows = math.ceil(num_plots / num_cols)
        fig = plt.figure(figsize=(10 * num_cols, 10 * num_rows))
        plt.subplots_adjust(
            left=0.01, right=0.99, bottom=0.01, top=0.95, wspace=0.25, hspace=0.30
        )

        with writer.saving(fig, anim_path, dpi=100):
            for time in tqdm(
                times, desc="visualize_corruption() timesteps", leave=False
            ):
                bridged, trans_coupling = self.interpolant.corrupt_to(
                    batch=batch,
                    t=torch.ones(num_batch, device=batch.trans_1.device) * time,
                    x0=x0,
                )
                corrupted = bridged.pack_present()

                alive_mask = (
                    corrupted.alive_mask.cpu().numpy().astype(bool)
                )  # (B, P_max)
                trans_t = corrupted.trans_t.cpu().numpy()  # (B, P_max, 3)
                motif_mask = batch.motif_mask.cpu().numpy().astype(bool)  # (B, N)

                idx_pack, _, _, _ = DataBridged.pack_present_indices(
                    bridged.present_mask
                )
                idx_pack = idx_pack.cpu().numpy()  # (B, P_max)

                fig.clf()
                for i in range(num_plots):
                    ax = fig.add_subplot(num_rows, num_cols, i + 1, projection="3d")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_zticks([])
                    plt.subplots_adjust(
                        left=0.03,
                        right=0.97,
                        bottom=0.03,
                        top=0.95,
                        wspace=0.05,
                        hspace=0.05,
                    )

                    trans_t_alive = trans_t[i][alive_mask[i]]  # (P, 3)
                    node_ids_alive = idx_pack[i][alive_mask[i]]  # (P,)

                    # Book-keeping of intermediate states
                    N = trans_1.shape[1]
                    is_leaf = node_ids_alive < N
                    is_motif = np.zeros_like(is_leaf, dtype=bool)
                    if bool(is_leaf.any()):
                        is_motif[is_leaf] = motif_mask[i][node_ids_alive[is_leaf]]
                    is_anchor = ~is_leaf
                    is_scaffold_leaf = is_leaf & (~is_motif)

                    ax.set_title(
                        f"t = {time:.2f} (N={trans_t_alive.shape[0]}/{trans_1.shape[1]})"
                    )
                    if trans_t_alive.shape[0] > 0:
                        vmax = int(trans_1[i].shape[0]) - 1
                        # motifs as small points, anchors + scaffolds as large points
                        sizes = np.full(
                            (trans_t_alive.shape[0],), 40.0, dtype=np.float32
                        )
                        sizes[is_leaf & is_motif] = 15.0

                        # anchors get a black outline
                        edgecolors = np.zeros(
                            (trans_t_alive.shape[0], 4), dtype=np.float32
                        )
                        edgecolors[is_anchor] = np.array(
                            [0.0, 0.0, 0.0, 1.0], dtype=np.float32
                        )
                        linewidths = np.where(is_anchor, 1.0, 0.0).astype(np.float32)

                        ax.scatter(
                            trans_t_alive[:, 0],
                            trans_t_alive[:, 1],
                            trans_t_alive[:, 2],
                            c=node_ids_alive,
                            cmap="Spectral",
                            vmin=0,
                            vmax=vmax,
                            s=sizes,
                            edgecolors=edgecolors,
                            linewidths=linewidths,
                            depthshade=True,
                            alpha=0.75,
                        )
                    ax.view_init(elev=25, azim=45)
                    # set camera limits
                    ax.set_xlim(trans_1_min[i][0], trans_1_max[i][0])
                    ax.set_ylim(trans_1_min[i][1], trans_1_max[i][1])
                    ax.set_zlim(trans_1_min[i][2], trans_1_max[i][2])

                writer.grab_frame()

        plt.close(fig)
        return anim_path


""" Module """


class BranchFlowModule(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = BranchFlowModel()
        self.loss_calculator = BranchFlowLossCalculator()
        self.interpolant = TreeInterpolant()

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def on_train_start(self) -> None:
        model_size = get_model_size_str(self.model)
        print(f"Model size: {model_size}")

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        return self.model(batch)

    def training_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        self.interpolant.set_device(self.device)
        bridged, coupling = self.interpolant.corrupt_batch(batch=batch)

        corrupted = bridged.pack_present()

        pred = self.forward(corrupted)

        loss = self.loss_calculator.calculate(
            batch=corrupted, pred=pred, coupling=coupling
        )

        self.log("loss/train", loss.total_loss.item(), prog_bar=True)
        self.log("loss/trans", loss.base_trans_loss.item(), prog_bar=True)
        self.log("loss/pairwise", loss.pairwise_loss.item())
        self.log("loss/split", loss.split_loss.item(), prog_bar=True)
        self.log(
            "aux/B*N",
            bridged.trans_t.shape[0] * bridged.trans_t.shape[1],
            prog_bar=True,
        )
        self.log("aux/t", bridged.t.mean().item())

        if batch_idx % 100 == 0 and torch.backends.mps.is_available():
            alloc = torch.mps.current_allocated_memory() / 1e9
            drv = torch.mps.driver_allocated_memory() / 1e9
            print(f"step {batch_idx} mps alloc={alloc:.2f}GB driver={drv:.2f}GB")
            gc.collect()
            torch.mps.empty_cache()

        return loss.total_loss


""" Training """


def train(
    max_epochs: int = 20,
    devices: int = 1,
    seed: int = 0,
):
    pl.seed_everything(seed, workers=True)

    data_module = ProteinDataModule(
        dataset=ProteinDataset(),
        num_workers=0,  # debugging
    )

    # Debugging: plot a corrruption planning tree
    datum = data_module.dataset[0]
    datum.tree_plan.plot()

    # Debugging: visualize the corruption process once using a real training batch.
    viz = BranchingFlowVisualizer(sigma=1.0)
    debug_batch = next(iter(data_module.train_dataloader(rank=0, num_replicas=1)))
    viz.visualize_corruption(batch=debug_batch)

    module = BranchFlowModule()

    checkpoint_callback = ModelCheckpoint(
        dirpath="varco/ckpt",
        filename="varco-{epoch:02d}",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="auto",
        devices=devices,
        max_epochs=max_epochs,
        check_val_every_n_epoch=min(max_epochs, 5),
        limit_val_batches=20,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(module, datamodule=data_module)

    print(f"Training complete")
    print(f"💾 ckpt saved to {checkpoint_callback.best_model_path}")
    print(f"🏆 Best validation loss: {checkpoint_callback.best_model_score}")

    return module


if __name__ == "__main__":
    train()
