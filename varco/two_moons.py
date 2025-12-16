"""
Toy implementation of branching flows.

Sample N, number of points, and X1, the moons points
Sample a single starting point X0
Define a tree, and the fixed-capacity representation (length A), of the birth trajectory
Aample some intermediate time t
Corrupt to X_t, using a bridge from 0 to each birth up to t
Define a simple model, which predicts base (endpoint prediction), split (remaining children count), and deletion (destined to delete probability)
Losses are base (MSE), split (supervised, MSE for now), and deletion (BCE)

Then, a sampler (no tree) iterates to get base (endpoint prediction), sample split events, sample deletion events
Evaluation: MMD between X1 and the sampled points, and compare size distribution

Features:
- handle variable lengths, batch size > 1, variable lengths within a batch
- start with more than one point (motifs)
- add deletions (death times, model predictions, update is_alive mask, etc.)
- add stochasticity to process.
- use a protei translations instead of 2D point
- support sequence with an protein insertion logits (and sequence logits)
- support rotations with an IGSO(3) bridge
- handle no point at index 0 (i.e. need dummy to insert at index 0, or support left-like insertions/deletions)
"""

import os
import math
import tempfile
from types import NoneType
from typing import Optional, List, Dict, Tuple, TypeVar, Generic
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cogeneration.models.utils import get_model_size_str
from cogeneration.type.embed import PositionalEmbeddingMethod
from cogeneration.models.embed import get_index_embedding

""" Constants """

# number of possible remaining insertions per root
n_insertion_logits = 128 + 1  # +1 for 0 insertion


""" Batch + Pred """

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
    ) -> "TreePlan":
        """Generate a simple planar coalescent tree plan with sampled split/birth times.

        Policy (POC):
          - Scaffold spans (motif_mask == False) are grouped into disjoint contiguous groups.
          - Each motif position is its own singleton group (never merges).
          - Within a group, repeatedly merge the leftmost adjacent pair.
        """
        if x1.ndim != 2:
            raise ValueError(f"Expected x1 to have shape (N, d); got {tuple(x1.shape)}")
        if motif_mask.ndim != 1 or motif_mask.shape[0] != x1.shape[0]:
            raise ValueError(f"Expected motif_mask to have shape (N,) matching x1; got {tuple(motif_mask.shape)}")

        device = x1.device
        N = x1.shape[0]
        motif_mask_b = motif_mask.to(torch.bool)

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
        motif_positions = torch.nonzero(motif_mask_b, as_tuple=False).squeeze(-1).tolist()
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
            g = int(group_ids[i].item())
            groups.setdefault(g, []).append(i)

        # Coalesce within each group by repeated adjacent merges
        for g, active in groups.items():
            while len(active) > 1:
                i0 = 0
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

        roots = [active[0] for active in groups.values()]
        roots.sort()

        A = len(parent)

        # Sample split times top-down from uniform base time distribution + exponential waiting time
        # Use a local torch.Generator for determinism and to avoid global RNG interaction.
        g = torch.Generator(device="cpu")
        if seed is None:
            # fall back to a nondeterministic seed
            seed = int(torch.seed() % (2**31 - 1))
        g.manual_seed(int(seed))

        birth = torch.full((A,), float("inf"), dtype=torch.float32)
        split = torch.full((A,), float("inf"), dtype=torch.float32)

        # Roots start at time 0
        for r in roots:
            birth[r] = 0.0

        def sample_exp1() -> float:
            # Exp(1) via inverse CDF from Uniform(0,1)
            u = float(torch.rand((), generator=g).clamp_min(1e-12).item())
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
        if self.parent_idx.shape != (A,): raise ValueError("parent_idx shape mismatch")
        if self.children_idx.shape != (A, 2): raise ValueError("children_idx shape mismatch")
        if self.total_leaves.shape != (A,): raise ValueError("total_leaves shape mismatch")
        if self.birth_time.shape != (A,): raise ValueError("birth_time shape mismatch")
        if self.split_time.shape != (A,): raise ValueError("split_time shape mismatch")
        if self.topo_order.shape != (A,): raise ValueError("topo_order shape mismatch")
        if self.motif_mask.shape != (N,): raise ValueError("motif_mask shape mismatch")

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

        A = int(self.num_nodes)

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
            ax.plot([xi, xi], [y0, y1], linewidth=1.0)

        # Draw parent-child connectors at birth times
        for child_idx in range(A):
            p = int(parent[child_idx])
            if p < 0:
                continue
            y = float(birth[child_idx])
            if not np.isfinite(y):
                continue
            y = max(0.0, min(1.0, y))
            ax.plot([float(x_pos[p]), float(x_pos[child_idx])], [y, y], linewidth=1.0)

        # Scatter markers: roots at time 0, leaves at time 1, internal anchors at their split time
        root_x = [float(x_pos[r]) for r in sorted(roots)]
        if len(root_x) > 0:
            ax.scatter(root_x, [0.0] * len(root_x), marker="o", s=20, label="roots")

        leaf_x = [float(x_pos[i]) for i in range(A) if int(total[i]) == 1]
        if len(leaf_x) > 0:
            ax.scatter(leaf_x, [1.0] * len(leaf_x), marker="s", s=14, label="leaves")

        # Mark anchors (finite split_time) for internal nodes
        internal_nodes = [i for i in range(A) if (int(total[i]) > 1 and np.isfinite(split[i]))]
        if len(internal_nodes) > 0:
            internal_y = [float(split[i]) for i in internal_nodes]
            internal_x_plot = [float(x_pos[i]) for i in internal_nodes]
            ax.scatter(internal_x_plot, internal_y, marker="^", s=14, label="anchors")

        ax.set_xlabel("Planar x-position (leaf index space)")
        ax.set_ylabel("Time")
        ax.set_title(f"TreePlan (N={self.num_leaves}, A={self.num_nodes}, R={int(self.roots.numel())})")

        # Time axis: 0 at top, 1 at bottom
        ax.set_ylim(1.02, -0.02)
        ax.set_xlim(-1, max(float(x_pos.max()) + 1.0, float(self.num_leaves)))
        ax.grid(True, linewidth=0.5, alpha=0.3)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), borderaxespad=0.)

        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)

        print(path)
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
class TwoMoonsSample:
    """Per-sample data (length N) at time t=1."""

    moons_1: torch.Tensor  # (N, 2)
    moons_labels: torch.Tensor  # (N,)
    motif_mask: torch.Tensor  # (N,) bool
    tree_plan: TreePlan



@dataclass
class TwoMoonsBatch:
    """Batched data for training."""

    moons_1: torch.Tensor  # (B, N, 2)
    moons_labels: torch.Tensor  # (B, N)
    motif_mask: torch.Tensor  # (B, N) bool
    tree: BatchedTreePlan


@dataclass
class TwoMoonsCorrupted:
    """Model input: corrupted and packed (length P_max) points present at time t"""
    t: torch.Tensor  # (B,)
    moons_t: torch.Tensor  # (B, P_max, 2)
    birth_time: torch.Tensor  # (B, P_max) 0.0 for roots
    remaining_insertions: Optional[torch.Tensor] = None  # (B, P_max) supervised target, remaining splits per present token

    @property
    def alive_mask(self) -> torch.Tensor:  # (B, P_max)
        return (self.birth_time <= self.t[:, None]).bool()

    @property
    def remaining_total(self) -> torch.Tensor:
        # batch.remaining_insertions: (B, P_max) padded with 0
        # batch.alive_mask: (B, P_max) True for real packed tokens, False for padding
        remaining_total = (self.remaining_insertions * self.alive_mask.long()).sum(dim=1)  # (B,)
        remaining_total = remaining_total.clamp_max(n_insertion_logits - 1).long()
        return remaining_total


@dataclass
class TwoMoonsBridged:
    """Corrupted and aligned (length A) points at time t"""
    t: torch.Tensor  # (B,)
    moons_t: torch.Tensor  # (B, A, 2)
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
            raise ValueError(f"present_mask must have shape (B, A); got {tuple(present_mask.shape)}")

        B, A = present_mask.shape
        device = present_mask.device

        sort_key = (~present_mask).to(torch.int32)
        idx_sorted = torch.argsort(sort_key, dim=1, stable=True)  # (B, A)

        P_b = present_mask.sum(dim=1)  # (B,)
        P_max = int(P_b.max().item()) if B > 0 else 0
        if P_max == 0:
            raise ValueError("No present nodes")

        idx_pack = idx_sorted[:, :P_max]  # (B, P_max)
        pack_mask = (torch.arange(P_max, device=device)[None, :] < P_b[:, None])  # (B, P_max)
        return idx_pack, pack_mask, P_b, P_max
    
    def _pack_indices(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return self.pack_present_indices(self.present_mask)

    def pack_present(self) -> TwoMoonsCorrupted:
        """Pack aligned (A) state into present (P_max) state for model input."""
        B, A, D = self.moons_t.shape
        idx_pack, pack_mask, P_b, P_max = self._pack_indices()

        moons_t_pack = self.moons_t.gather(1, idx_pack.unsqueeze(-1).expand(-1, -1, D))  # (B, P_max, D)
        birth_time_pack = self.birth_time.gather(1, idx_pack)  # (B, P_max)
        remaining_insertions_pack = self.remaining_insertions.gather(1, idx_pack)  # (B, P_max)

        moons_t_pack = moons_t_pack * pack_mask.unsqueeze(-1).float()  # zero pad
        birth_time_pack = torch.where(
            pack_mask,
            birth_time_pack,
            torch.full_like(birth_time_pack, float("inf")),  # infinite birth time for pad tokens
        )
        remaining_insertions_pack = torch.where(
            pack_mask,
            remaining_insertions_pack,
            torch.zeros_like(remaining_insertions_pack),  # 0 pad
        )

        return TwoMoonsCorrupted(
            t=self.t,
            moons_t=moons_t_pack,
            birth_time=birth_time_pack,
            remaining_insertions=remaining_insertions_pack,
        )

    def validate(self) -> None:
        B, A, D = self.moons_t.shape
        if self.birth_time.shape != (B, A): raise ValueError("birth_time shape mismatch")
        if self.present_mask.shape != (B, A): raise ValueError("present_mask shape mismatch")
        if self.remaining_insertions.shape != (B, A): raise ValueError("remaining_insertions shape mismatch")
        if D != 2: raise ValueError("moons_t last dim must be 2")


@dataclass
class ModelPrediction:
    """t=1 prediction for present state (length P)"""
    moons_pred_1: torch.Tensor  # (B, P, 2) base; predicted final/anchor positions
    moons_pred_split: torch.Tensor  # (B, P, n_insertion_logits) split logits per token
    moons_pred_split_pooled: torch.Tensor  # (B, n_insertion_logits) pooled split logits


@dataclass
class TwoMoonsTrajectory:
    samples: List[TwoMoonsCorrupted]  # samples at each time step
    pred: List[ModelPrediction]  # predictions at each time step


""" Datasets + DataLoaders """


class TwoMoonsDataset(Dataset):
    def __init__(self, num_samples: int, mean_points: int, mean_noise: float, random_state: Optional[int] = None):
        self.num_samples = num_samples
        self.mean_points = mean_points
        self.mean_noise = mean_noise
        self.random_state = random_state

    def __len__(self):
        return self.num_samples

    @staticmethod
    def _sample_motif_mask(N: int, rng: Optional[np.random.RandomState] = None) -> torch.Tensor:
        if rng is None:
            rng = np.random.RandomState()

        # choose 1-2 non-overlapping contiguous scaffold spans, away from endpoints
        num_spans = int(rng.randint(1, 3))  # {1,2}
        min_len = 4
        max_len = min(10, max(min_len, N // 2))

        motif_mask = np.ones(N, dtype=np.bool_)
        for _ in range(num_spans):
            L = int(rng.randint(min_len, max_len + 1))
            # enforce "intermediate": keep at least 1 index margin at both ends
            lo = 1
            hi = (N - 1) - L
            if hi <= lo:
                break

            placed = False
            for _attempt in range(32):
                start = int(rng.randint(lo, hi + 1))
                end = start + L
                if motif_mask[start:end].all():
                    motif_mask[start:end] = False
                    placed = True
                    break
            if not placed:
                break

        motif_mask = torch.from_numpy(motif_mask)  # (N,) bool
        # guarantee at least one motif/root
        if not bool(motif_mask.any()):
            motif_mask[0:min_len] = True
        return motif_mask

    def __getitem__(self, idx: int) -> TwoMoonsSample:
        noisy_moons = datasets.make_moons(
            n_samples=self.mean_points,  # We'll use the length-batching proteins in the future anyways
            noise=float(np.random.rand() * self.mean_noise + self.mean_noise),
        )

        moons_xy, moons_labels = noisy_moons

        moons_1 = torch.tensor(moons_xy, dtype=torch.float32)
        moons_labels_t = torch.tensor(moons_labels, dtype=torch.long)

        motif_mask = self._sample_motif_mask(N=moons_1.shape[0])

        tree_plan = TreePlan.generate(x1=moons_1, motif_mask=motif_mask)
        tree_plan.validate()  # TODO - disable to actually train

        return TwoMoonsSample(
            moons_1=moons_1,
            moons_labels=moons_labels_t,
            motif_mask=motif_mask,
            tree_plan=tree_plan,
        )


class TwoMoonsDataLoader(DataLoader):
    def __init__(
        self,
        num_samples: int = 1000,
        mean_points: int = 50,
        mean_noise: float = 0.05,
        batch_size: int = 1,
        num_workers: int = max(1, os.cpu_count() - 2),
        random_state: Optional[int] = None,
    ):
        dataset = TwoMoonsDataset(num_samples=num_samples, mean_points=mean_points, mean_noise=mean_noise, random_state=random_state)
        super().__init__(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=TwoMoonsDataLoader.collate_fn,
            persistent_workers=False
        )

    @staticmethod
    def collate_fn(batch: List[TwoMoonsSample]) -> TwoMoonsBatch:
        moons_1 = torch.stack([item.moons_1 for item in batch])  # (B, N, 2)
        moons_labels = torch.stack([item.moons_labels for item in batch])  # (B, N)
        motif_mask = torch.stack([item.motif_mask for item in batch])  # (B, N)

        plans = [item.tree_plan for item in batch]
        tree = BatchedTreePlan.collate(plans)
        
        return TwoMoonsBatch(
            moons_1=moons_1,
            moons_labels=moons_labels,
            motif_mask=motif_mask,
            tree=tree,
        )


""" Model """


class BranchFlowModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.point_dim = 2
        self.pos_embed_dim = 32

        # x_t (point_dim), t (1), birth_time (1), pos_embed
        self.input_dim = self.point_dim + 1 + 1 + self.pos_embed_dim
        self.model_dim = 64

        self.input_proj = nn.Linear(self.input_dim, self.model_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim, 
            nhead=8, 
            dim_feedforward=4 * self.model_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.x1_pred = nn.Linear(self.model_dim, self.point_dim)
        self.split_logits_pred = nn.Linear(self.model_dim, n_insertion_logits)
        self.split_pooled_logits_pred = nn.Linear(self.model_dim, n_insertion_logits)


    def forward(self, batch: TwoMoonsCorrupted) -> ModelPrediction:
        B, P, _ = batch.moons_t.shape

        # attention mask to mask out positions not alive yet / padded
        key_padding_mask = ~batch.alive_mask.bool()

        pos_embed = get_index_embedding(
            torch.arange(P, device=batch.moons_t.device).unsqueeze(0).expand(B, -1),
            embed_size=self.pos_embed_dim,
            max_len=1024,
            pos_embed_method=PositionalEmbeddingMethod.rotary,
        ) 
        pos_embed = pos_embed * batch.alive_mask.unsqueeze(-1).float()

        x_t = torch.cat([
            batch.moons_t,  # (B, P, 2)
            batch.t[:, None, None].expand(B, P, 1),  # (B, P, 1)
            batch.birth_time[:, :, None].float().clamp(0.0, 1.0),  # (B, P, 1) clamp +inf padding
            pos_embed,  # (B, P, pos_embed_dim)
        ], dim=-1)
        
        x_t = self.input_proj(x_t)
        x_t = self.transformer(x_t, src_key_padding_mask=key_padding_mask)

        x1_pred = self.x1_pred(x_t)

        split_logits = self.split_logits_pred(x_t)  # (B, P, n_insertion_logits)

        # Masked sum pool over alive tokens to predict total remaining insertions per example
        alive = batch.alive_mask.bool()  # (B, P)
        pooled = (x_t * alive.unsqueeze(-1).float()).sum(dim=1)  # (B, model_dim)
        split_pooled_logits = self.split_pooled_logits_pred(pooled)  # (B, n_insertion_logits)

        return ModelPrediction(
            moons_pred_1=x1_pred,
            moons_pred_split=split_logits,
            moons_pred_split_pooled=split_pooled_logits,
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
    def sample_base(self, num_batch: int, num_roots: int, device: torch.device) -> torch.Tensor:
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
    anchors: torch.Tensor  # (B, A, 2)

    # shared tree plan (domain-agnostic)
    tree: BatchedTreePlan


class TranslationCoupler(Coupler[TranslationCoupling]):
    def __init__(self, sigma: float = 1.0):
        self.sigma = float(sigma)

    def sample_base(self, num_batch: int, num_roots: int, device: torch.device) -> torch.Tensor:
        return torch.randn(num_batch, num_roots, 2, device=device)

    def build_anchor_alignment(self, x1: torch.Tensor, tree: BatchedTreePlan) -> torch.Tensor:
        """Build translation anchors for all nodes (leaves + internal) from leaf endpoints x1.

        Assumptions for this toy:
        - Leaves correspond to x1 in positions [0..N-1].
        - Internal nodes are derived purely from topology + descendant weights.

        For proteins later, this stays the same idea: leaf anchors are true endpoints, internal anchors are
        domain-specific merges of child anchors.
        """
        if x1.ndim != 3 or x1.shape[-1] != 2:
            raise ValueError(f"Expected x1 to have shape (B, N, 2); got {tuple(x1.shape)}")

        device = x1.device
        B, N, _ = x1.shape
        A = tree.parent_idx.shape[1]

        anchor = torch.zeros((B, A, 2), dtype=torch.float32, device=device)
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
            child_anchor = anchor.gather(1, k.unsqueeze(-1).expand(-1, 2, 2))  # (B, 2, 2)

            child_w = tree.total_leaves.gather(1, k)  # (B, 2)
            wsum = (child_w.sum(dim=1).clamp_min(1)).to(torch.float32)  # (B,)

            merged = (
                child_anchor[:, 0, :] * (child_w[:, 0].to(torch.float32) / wsum).unsqueeze(-1)
                + child_anchor[:, 1, :] * (child_w[:, 1].to(torch.float32) / wsum).unsqueeze(-1)
            )  # (B, 2)

            bidx = torch.arange(B, device=device)
            anchor[bidx[valid], i, :] = merged[valid]

        return anchor

    def _bridge_brownian(
        self,
        x_start: torch.Tensor,  # (M, 2)
        x_end: torch.Tensor,  # (M, 2)
        s: torch.Tensor,  # (M,)
        t0: torch.Tensor,  # (M,)
    ) -> torch.Tensor:
        """Sample Brownian-bridge marginal at time s from (x_start at t0) to (x_end at 1)."""
        denom = (1.0 - t0).clamp_min(1e-6)
        u = ((s - t0) / denom).clamp(0.0, 1.0)
        mean = x_start + u.unsqueeze(-1) * (x_end - x_start)

        var = ((s - t0).clamp_min(0.0) * (1.0 - s).clamp_min(0.0) / denom).clamp_min(0.0)
        std = (var.sqrt() * self.sigma).to(mean.dtype)
        eps = torch.randn_like(mean)
        return mean + std.unsqueeze(-1) * eps

    def corrupt(
        self,
        x1: torch.Tensor,  # (B, N, 2)
        tree: BatchedTreePlan,
        t: torch.Tensor,  # (B,)
        x0: Optional[torch.Tensor] = None,  # (B, R_max, 2)
    ) -> Tuple[torch.Tensor, TranslationCoupling]:
        """Corrupt translations to time t using the shared tree plan.

        Returns:
            moons_t_aligned: (B, A, 2)
            coupling: minimal coupling info for losses (anchors + tree)
        """
        if x1.ndim != 3 or x1.shape[-1] != 2:
            raise ValueError(f"Expected x1 to have shape (B, N, 2); got {tuple(x1.shape)}")

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
        if x0.ndim != 3 or x0.shape[0] != B or x0.shape[-1] != 2:
            raise ValueError(f"Expected x0 to have shape (B, R, 2); got {tuple(x0.shape)}")

        # Build x0_aligned by broadcasting first root and then writing per-example roots into tree root slots
        x0_aligned = x0[:, :1, :].expand(B, A, 2).contiguous().clone()
        for b in range(B):
            roots_i = tree.roots[b][tree.roots_mask[b]].to(torch.long)
            for j, r in enumerate(roots_i.tolist()):
                if j >= x0.shape[1]:
                    break
                x0_aligned[b, r, :] = x0[b, j, :]

        # Build domain anchors (leaf endpoints + internal merges)
        anchor_aligned = self.build_anchor_alignment(x1=x1, tree=tree)

        # Track creation states at each node's birth time (shared across siblings)
        creation_state = x0_aligned.to(dtype=torch.float32).clone()  # (B, A, 2)

        # Topological pass: sample each parent's state at split time once, then assign to both children
        for k in range(A):
            node_idx = tree.topo_order[:, k]  # (B,)

            t0 = tree.birth_time.gather(1, node_idx.unsqueeze(1)).squeeze(1)  # (B,)
            st = tree.split_time.gather(1, node_idx.unsqueeze(1)).squeeze(1)  # (B,)

            is_leaf = ~torch.isfinite(st)
            if bool(is_leaf.all()):
                continue

            node_creation = creation_state.gather(1, node_idx.view(B, 1, 1).expand(-1, 1, 2)).squeeze(1)  # (B, 2)
            node_anchor = anchor_aligned.gather(1, node_idx.view(B, 1, 1).expand(-1, 1, 2)).squeeze(1)  # (B, 2)

            node_at_split = self._bridge_brownian(
                x_start=node_creation,
                x_end=node_anchor,
                s=st,
                t0=t0,
            )  # (B, 2)

            kids = tree.children_idx.gather(1, node_idx.view(B, 1, 1).expand(-1, 1, 2)).to(torch.long).squeeze(1)  # (B, 2)

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

        moons_t = self._bridge_brownian(
            x_start=creation_state.view(-1, 2),
            x_end=anchor_aligned.view(-1, 2),
            s=t_mat.reshape(-1),
            t0=b_mat.reshape(-1),
        ).view(B, A, 2)

        moons_t = torch.where(present_mask.unsqueeze(-1), moons_t, torch.zeros_like(moons_t))

        coupling = TranslationCoupling(anchors=anchor_aligned, tree=tree)
        return moons_t, coupling

    # Removed stale validate method that referenced nonexistent self.anchors/self.tree


""" Interpolant supporting tree coupling """


@dataclass
class TreeInterpolant:
    device: torch.device = torch.device("cpu")
    min_t: float = 0.005
    translation_coupler: Coupler[TranslationCoupling] = TranslationCoupler(sigma=1.0)

    def set_device(self, device: torch.device):
        self.device = device

    def corrupt_batch(self, batch: TwoMoonsBatch) -> Tuple[TwoMoonsBridged, TranslationCoupling]:
        # pick a single time to share across the batch
        shared_t = torch.rand(1, device=self.device) * (1.0 - 2.0 * self.min_t) + self.min_t
        t = torch.ones(batch.moons_1.shape[0], device=self.device) * shared_t  # (B,)

        tree = batch.tree.to(self.device)
        moons_1 = batch.moons_1.to(self.device)

        moons_t, coupling = self.translation_coupler.corrupt(
            x1=moons_1,
            tree=tree,
            t=t,
            x0=None,
        )
        coupling.validate()

        bridged = TwoMoonsBridged(
            t=t,
            moons_t=moons_t,
            birth_time=tree.birth_time,
            present_mask=tree.present_mask(t=t),  # (B, A),
            remaining_insertions=tree.remaining_insertions_t(t=t),
        )
        bridged.validate()
        
        return bridged, coupling

    def sample(self, model: BranchFlowModel, num_batch: int = 1, num_roots: int = 50, num_steps: int = 100) -> TwoMoonsTrajectory:
        samples = []
        preds = []

        moons_0 = self.translation_coupler.sample_base(num_batch=num_batch, num_roots=num_roots, device=self.device)  # (B, R, 2)
        batch = TwoMoonsCorrupted(
            moons_t=moons_0,
            t=torch.zeros(num_batch, device=self.device),
            birth_time=torch.zeros_like(moons_0[:, :, 0]),
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
            
        return TwoMoonsTrajectory(samples=samples, pred=preds)


""" Loss """

@dataclass
class BranchFlowLosses:
    total_loss: torch.Tensor
    base_loss: torch.Tensor  # MSE
    split_loss: torch.Tensor  # MSE


@dataclass
class BranchFlowLossCalculator:
    def calculate(self, batch: TwoMoonsCorrupted, pred: ModelPrediction, coupling: TranslationCoupling) -> BranchFlowLosses:
        B, P, D = batch.moons_t.shape
        assert pred.moons_pred_1.shape == (B, P, D)

        present_mask = coupling.tree.present_mask(t=batch.t)  # (B, A)
        is_root = (coupling.tree.parent_idx < 0).to(torch.long)
        pad_mask = batch.alive_mask  # (B, P_max)

        # Reconstruct the same packing indices used by TwoMoonsBridged.pack_present()
        idx_pack, pack_mask, P_b, P_max = TwoMoonsBridged.pack_present_indices(present_mask)

        # Pack anchor targets into (B, P_max, D) in the same order as model inputs/predictions
        anchors_pack = coupling.anchors.gather(1, idx_pack.unsqueeze(-1).expand(-1, -1, D))
        anchors_pack = anchors_pack * pack_mask.unsqueeze(-1).float()  # zero pad (optional)

        # base loss final position MSE (paired in packed order)
        base_loss = nn.MSELoss()(
            pred.moons_pred_1[pad_mask],  # (K, D)
            anchors_pack[pad_mask],       # (K, D)
        )

        # split loss: cross-entropy over remaining insertions (tokenwise) + pooled auxiliary head
        logits_flat = pred.moons_pred_split[pad_mask]  # (K, n_insertion_logits)
        targets_flat = batch.remaining_insertions[pad_mask].to(torch.long)  # (K,)
        split_loss_token = nn.CrossEntropyLoss()(logits_flat, targets_flat)

        # pooled target is total remaining insertions in this example
        pooled_targets = (coupling.tree.total_leaves * is_root).sum(dim=1)
        pooled_targets = (pooled_targets - present_mask.sum(dim=1)).clamp_min(0).clamp_max(n_insertion_logits - 1).to(torch.long)
        split_loss_pooled = nn.CrossEntropyLoss()(pred.moons_pred_split_pooled, pooled_targets)

        split_loss = split_loss_token + 1.0 * split_loss_pooled

        total_loss = base_loss + split_loss

        return BranchFlowLosses(
            total_loss=total_loss,
            base_loss=base_loss,
            split_loss=split_loss,
        )


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

    def forward(self, batch: TwoMoonsCorrupted) -> ModelPrediction:
        return self.model(batch)

    def training_step(self, batch: TwoMoonsBatch, batch_idx: int) -> torch.Tensor:
        self.interpolant.set_device(self.device)
        bridged, coupling = self.interpolant.corrupt_batch(batch=batch)

        corrupted = bridged.pack_present()

        pred = self.forward(corrupted)

        loss = self.loss_calculator.calculate(batch=corrupted, pred=pred, coupling=coupling)
        self.log("train/loss", loss.total_loss, prog_bar=True)
        self.log("train/base_loss", loss.base_loss, prog_bar=True)
        self.log("train/split_loss", loss.split_loss, prog_bar=True)
        self.log("aux/t", bridged.t.mean(), prog_bar=True)

        return loss.total_loss


""" Training """

def train(
    num_samples: int = 16_000,
    mean_points: int = 50,
    mean_noise: float = 0.05,
    batch_size: int = 16,
    max_epochs: int = 5,
    devices: int = 1,
    seed: int = 0,
):
    pl.seed_everything(seed, workers=True)

    loader = TwoMoonsDataLoader(
        num_samples=num_samples,
        mean_points=mean_points,
        mean_noise=mean_noise,
        batch_size=batch_size,
        random_state=seed,
    )

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
        log_every_n_steps=50,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
    )

    trainer.fit(module, train_dataloaders=loader)
    return module


if __name__ == "__main__":
    train()