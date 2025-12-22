"""
Simple implementation of branching flows.

Sample some proteins of length N (translations, later sequence and rotations) using LengthBatched protein dataset
Sample a motif_mask (only support internal motif scaffolding for now so always have "real" parent)
Define a tree, which inserts scaffold residues over trajectory with anchors at intermediate time points, and alignment of length A
Sample some number of roots, starting points X0 for the motifs + roots

Sample some intermediate time t
Corrupt to X_t, using brownian bridge from 0 to each anchor, continuing up to time t

A simple model predicts base (endpoint prediction), split (remaining children count), and deletion (destined to delete probability)
Losses are base (MSE), split (poisson-like), and deletion (BCE)

Then, a sampler (no tree) iterates to get base (endpoint prediction), sample split events, sample deletion events
Evaluation: MMD between X1 and the sampled points, and compare size distribution

TODOs / features:
- Cleaner motif defining
  - improve base distribution, i.e. a protein with motifs
  - pass through res_mask and chain_idx
  - use multiple motif scaffolding strategies
  - move to unconditional dataset, use MotifFactory here to get motifs/scaffolds so easier to seed scaffolds
- support sequence with an protein insertion logits (and sequence logits)
  - also predict insertion logits, which are sampled on insertions, rather than cloning the existing residue
- improve sampling
  - motif guidance for positions
  - add validation loss (e.g. folding validation?) and validation_step (and data loader that includes scaffold nuclei)
- support a mini frozen ESM model up front (once we have sequence)
- make sure we handle positional embedding correctly (i.e. ~ arange on current sequence, but handle chain_idx)
- support rotations with an IGSO(3) bridge
- use a Config
- simplify tree plan and coupling
  - maintain ~rough order in alignment space of final ordering (i.e. anchors mixed in with leaves)
    - or at least, easier to get planar index (e.g. use for plotting)
  - speed up construction
  - speed up corruption
"""

import gc
import math
import os
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
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

""" Tree Plan """

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
        min_t: float = 0.001,
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
        # Coalesce within each group by repeated adjacent merges.
        # For scaffold span groups (gid < span_gid), bias merges toward the boundaries so the
        # coalescent "collapses" inwards from both ends (both boundaries can in-fill).
        p_interior: float = 0.95,
        # Number of deletion leaves are sampled from a Poisson distribution with rate p_deletion * # scaffold positions.
        p_deletion: float = 0.20,
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

        # Sample split times top-down from uniform base time distribution + exponential waiting time
        # Use a local torch.Generator for determinism and to avoid global RNG interaction.
        rng = torch.Generator(device="cpu")
        if seed is None:
            # fall back to a nondeterministic seed
            seed = int(torch.seed() % (2**31 - 1))
        rng.manual_seed(int(seed))

        def rand_int(high: int) -> int:
            return int(torch.randint(0, high, (1,), generator=rng).item())

        def rand_float() -> float:
            return float(torch.rand(1, generator=rng).squeeze().item())

        # --- Deletions: duplicate existing scaffold leaves and mark one of each pair as destined-to-delete.
        # t=1 is augmented by inserting deletion leaves that duplicate an endpoint.
        # Only scaffold positions are eligible.

        leaf_ref: List[int] = list(range(N_data))  # leaf-order -> original data idx
        leaf_del: List[bool] = [False for _ in range(N_data)]

        scaffold_positions = [
            i for i in range(N_data) if not bool(motif_mask_b[i].item())
        ]
        n_scaffold = len(scaffold_positions)

        # Poisson sampler (Knuth), deterministic under our local RNG
        def sample_poisson(lam: float) -> int:
            if lam <= 0.0:
                return 0
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while True:
                k += 1
                p *= rand_float()
                if p <= L:
                    return k - 1

        lam = p_deletion * float(n_scaffold)
        k_del = sample_poisson(lam)
        k_del = int(min(k_del, n_scaffold))

        for _ in range(k_del):
            cur_scaffold_pos = [
                j
                for j, ref in enumerate(leaf_ref)
                if not bool(motif_mask_b[ref].item())
            ]
            if len(cur_scaffold_pos) == 0:
                break
            pos = cur_scaffold_pos[rand_int(len(cur_scaffold_pos))]
            ref_idx = leaf_ref[pos]

            insert_after = rand_float() < 0.5
            ins_pos = pos + 1 if insert_after else pos
            leaf_ref.insert(ins_pos, ref_idx)
            leaf_del.insert(ins_pos, False)

            # mark exactly one of the pair as deleted
            if not insert_after:
                orig_pos = pos + 1
                dup_pos = pos
            else:
                orig_pos = pos
                dup_pos = pos + 1
            if rand_float() < 0.5:
                leaf_del[orig_pos] = True
            else:
                leaf_del[dup_pos] = True

        num_leaves = len(leaf_ref)
        num_deletions = int(sum(leaf_del))

        # Group ids computed in *leaf order* (includes deletion duplicates).
        group_ids_leaf = torch.full((num_leaves,), -1, dtype=torch.long, device=device)

        def leaf_is_motif(j: int) -> bool:
            return bool(motif_mask_b[leaf_ref[j]].item())

        span_gid = 0
        j = 0
        while j < num_leaves:
            if leaf_is_motif(j):
                j += 1
                continue
            k = j
            while k < num_leaves and (not leaf_is_motif(k)):
                k += 1
            group_ids_leaf[j:k] = span_gid
            span_gid += 1
            j = k

        motif_leaf_positions = [j for j in range(num_leaves) if leaf_is_motif(j)]
        for kk, pos in enumerate(motif_leaf_positions):
            group_ids_leaf[pos] = span_gid + kk

        if bool((group_ids_leaf < 0).any().item()):
            raise RuntimeError("Failed to assign group_ids_leaf for all leaf positions")

        leaf_map_leaves = torch.tensor(
            leaf_ref, dtype=torch.long, device=device
        )  # (num_leaves,)

        # Initialize leaf nodes (leaf ids are 0..num_leaves-1)
        parent: List[int] = [-1] * num_leaves
        children: List[List[int]] = [[-1, -1] for _ in range(num_leaves)]
        weight: List[int] = [1 for _ in range(num_leaves)]

        # Active lists per group in sequence order
        groups: Dict[int, List[int]] = {}
        for j in range(num_leaves):
            gid = int(group_ids_leaf[j].item())
            groups.setdefault(gid, []).append(j)

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

        birth_time = torch.full((A,), float("inf"), dtype=torch.float32)
        split_time = torch.full((A,), float("inf"), dtype=torch.float32)

        # leaf_map mapping is valid for leaves (< num_leaves) and 0 for internal nodes
        leaf_map = torch.zeros((A,), dtype=torch.long, device=device)
        leaf_map[:num_leaves] = leaf_map_leaves

        leaf_deleted = torch.zeros((A,), dtype=torch.bool, device=device)
        if num_leaves > 0:
            leaf_deleted[:num_leaves] = torch.tensor(
                leaf_del, dtype=torch.bool, device=device
            )

        delete_time = torch.full((A,), float("inf"), dtype=torch.float32)

        # Roots start at time 0
        for r in roots:
            birth_time[r] = 0.0

        def sample_exp1() -> float:
            # Exp(1) via inverse CDF from Uniform(0,1)
            u = float(torch.rand((), generator=rng).clamp_min(1e-12).item())
            return -math.log(u)

        def sample_split_time_uniform(W: int, t0: float) -> float:
            # next_split_time for Uniform[min_t,1-2*min_t]: 1 - (1 - t0) * exp(-E / (W-1))
            if W <= 1:
                return float("inf")
            m = W - 1
            E = sample_exp1()
            s = 1.0 - (1.0 - t0) * math.exp(-E / float(m))
            s = max(min_t, min(1.0 - 2 * min_t, s))
            return s

        # Structural topo: in this construction, internal nodes are appended so parent ids > child ids.
        topo_order = torch.arange(A - 1, -1, -1, dtype=torch.long)

        # Traverse parents before children and propagate birth times
        for node in topo_order.tolist():
            t0 = float(birth_time[node].item())
            if not math.isfinite(t0):
                continue

            W = int(weight[node])
            if W <= 1:
                split_time[node] = float("inf")
                continue

            st = sample_split_time_uniform(W=W, t0=t0)
            split_time[node] = st

            c0, c1 = children[node]
            for c in (c0, c1):
                if c >= 0:
                    birth_time[c] = st

        # Sample delete times for deleted leaves using the paper/JL semantics:
        # A deleted leaf has an unconditional deletion time distributed as Uniform(0, 1),
        # conditioned on being AFTER the leaf's birth time (i.e. dt | dt > birth).
        #
        # For Uniform(0, 1), the truncated-sampling is simply:
        #   dt = birth + (1 - birth) * u,  u ~ Uniform(0, 1)
        #
        # We enforce strict inequalities with tiny epsilons to avoid dt == birth or dt == 1
        # due to floating point edge cases.
        min_delete_eps = 1e-6
        max_delete_time = 1.0 - 1e-6
        for i in range(num_leaves):
            if not leaf_del[i]:
                continue
            b = float(birth_time[i].item())
            if not math.isfinite(b):
                continue

            u = rand_float()
            dt = b + (1.0 - b) * u

            # Numerical safety: ensure dt is strictly after birth.
            if dt <= b + min_delete_eps:
                dt = b + min_delete_eps

            # Numerical safety: ensure dt is strictly before 1.
            if dt >= max_delete_time:
                dt = max_delete_time

            # If birth is extremely close to 1, fall back to the latest valid time.
            # (This should be rare because split times are already clamped away from 1.)
            if dt <= b:
                dt = max_delete_time

            delete_time[i] = dt

        # Build motif mask in aligned tree space (A,)
        # Leaves 0..num_leaves-1 inherit motif/scaffold status from their referenced data index.
        # Internal nodes are always False.
        motif_mask_aligned = torch.zeros((A,), dtype=torch.bool, device=device)
        for i in range(num_leaves):
            motif_mask_aligned[i] = bool(motif_mask_b[leaf_ref[i]].item())

        parent_idx = torch.tensor(parent, dtype=torch.long, device=device)
        children_idx = torch.tensor(children, dtype=torch.long, device=device)
        total_leaves = torch.tensor(weight, dtype=torch.long, device=device)
        roots_t = torch.tensor(roots, dtype=torch.long, device=device)

        leaf_map = leaf_map.to(device=device)
        birth_time = birth_time.to(device=device)
        split_time = split_time.to(device=device)
        delete_time = delete_time.to(device=device)
        topo_order = topo_order.to(device=device)

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

        device = x.device
        B, N = x.shape[:2]
        trailing_shape = x.shape[2:]
        A = self.leaf_map.shape[1]

        # Expand leaf_map indices for arbitrary trailing dims
        idx = self.leaf_map.to(device=device).clamp_min(0)  # (B, A)
        for _ in trailing_shape:
            idx = idx.unsqueeze(-1)
        idx = idx.expand(B, A, *trailing_shape)  # (B, A, ...)

        # Gather from data space to tree space
        x_broadcast = x.gather(1, idx)  # (B, A, ...)

        # Mask: only keep leaf values, fill non-leaves with fill_value
        mask = self.leaf_mask.to(device=device)  # (B, A)
        for _ in trailing_shape:
            mask = mask.unsqueeze(-1)
        mask = mask.expand_as(x_broadcast)  # (B, A, ...)

        fill = torch.full_like(x_broadcast, fill_value)
        return torch.where(mask, x_broadcast, fill)

    @classmethod
    def collate(cls, plans: List["TreePlan"]) -> "BatchedTreePlan":
        B = len(plans)
        if B == 0:
            raise ValueError("Empty batch")

        A_max = max(int(p.num_nodes) for p in plans)
        R_max = max(int(p.roots.numel()) for p in plans)

        # topology
        topo_order = torch.zeros((B, A_max), dtype=torch.long)
        motif_mask = torch.zeros((B, A_max), dtype=torch.bool)
        roots = torch.full((B, R_max), -1, dtype=torch.long)
        roots_mask = torch.zeros((B, R_max), dtype=torch.bool)
        parent_idx = torch.full((B, A_max), -1, dtype=torch.long)
        children_idx = torch.full((B, A_max, 2), -1, dtype=torch.long)
        total_leaves = torch.zeros((B, A_max), dtype=torch.long)
        leaf_deleted = torch.zeros((B, A_max), dtype=torch.bool)

        # times
        birth_time = torch.full((B, A_max), float("inf"), dtype=torch.float32)
        split_time = torch.full((B, A_max), float("inf"), dtype=torch.float32)
        delete_time = torch.full((B, A_max), float("inf"), dtype=torch.float32)

        # tree -> data mapping
        leaf_map = torch.zeros((B, A_max), dtype=torch.long)

        for b, p in enumerate(plans):
            A_i = int(p.num_nodes)
            R_i = int(p.roots.numel())

            # topology
            topo_order[b, :A_i] = p.topo_order.to(torch.long)
            if A_i < A_max:
                topo_order[b, A_i:] = torch.arange(A_i, A_max, dtype=torch.long)
            motif_mask[b, :A_i] = p.motif_mask.to(torch.bool)
            roots[b, :R_i] = p.roots.to(torch.long)
            roots_mask[b, :R_i] = True
            parent_idx[b, :A_i] = p.parent_idx.to(torch.long)
            children_idx[b, :A_i, :] = p.children_idx.to(torch.long)
            total_leaves[b, :A_i] = p.total_leaves.to(torch.long)
            leaf_deleted[b, :A_i] = p.leaf_deleted.to(torch.bool)

            # times
            birth_time[b, :A_i] = p.birth_time.to(torch.float32)
            split_time[b, :A_i] = p.split_time.to(torch.float32)
            delete_time[b, :A_i] = p.delete_time.to(torch.float32)

            # tree -> data mapping
            leaf_map[b, :A_i] = p.leaf_map.to(torch.long)

        return cls(
            topo_order=topo_order,
            motif_mask=motif_mask,
            roots=roots,
            roots_mask=roots_mask,
            parent_idx=parent_idx,
            children_idx=children_idx,
            total_leaves=total_leaves,
            leaf_deleted=leaf_deleted,
            birth_time=birth_time,
            split_time=split_time,
            delete_time=delete_time,
            leaf_map=leaf_map,
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
    birth_time: torch.Tensor  # (B, P_max) 0.0 for motifs & roots, +inf for padding
    motif_mask: torch.Tensor  # (B, P_max) bool; True for fixed motif positions
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
        return (self.remaining_insertions * self.valid_mask.long()).sum(dim=1).long()

    def detach_clone(self) -> "DataCorrupted":
        """Detach and clone the data, e.g. to save in trajectory"""
        return DataCorrupted(
            t=self.t.detach().clone(),
            trans_t=self.trans_t.detach().clone(),
            birth_time=self.birth_time.detach().clone(),
            motif_mask=self.motif_mask.detach().clone(),
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
        cumsum = multiplicity.cumsum(dim=1)  # (B, P)
        out_pos = (
            torch.arange(P_new, device=device).unsqueeze(0).expand(B, -1)
        )  # (B, P_new)
        # For each out_pos, find source index
        gather_idx = torch.searchsorted(cumsum, out_pos, right=True)  # (B, P_new)
        gather_idx = gather_idx.clamp(0, P - 1)

        # Insertion = second occurrence of a source = consecutive gather_idx match
        is_insertion = torch.zeros((B, P_new), dtype=torch.bool, device=device)
        is_insertion[:, 1:] = gather_idx[:, 1:] == gather_idx[:, :-1]

        # Valid mask for new batch
        new_valid = out_pos < out_lens.unsqueeze(1)
        is_insertion = is_insertion & new_valid

        # TODO - consider padding helper as add domains

        # Update birth_time for inserted positions to current time t, set padding to inf
        new_birth = torch.gather(self.birth_time, 1, gather_idx)
        new_birth = torch.where(
            is_insertion, torch.full_like(new_birth, t_birth), new_birth
        )
        new_birth = torch.where(
            new_valid, new_birth, torch.full_like(new_birth, float("inf"))
        )

        # inserted children inherit parent's motif status (don't expect indels in motifs)
        new_motif = torch.gather(self.motif_mask, 1, gather_idx)
        new_motif = torch.where(new_valid, new_motif, torch.zeros_like(new_motif))

        # translations, zero-padded
        new_trans = torch.gather(
            self.trans_t, 1, gather_idx.unsqueeze(-1).expand(-1, -1, 3)
        )
        new_trans = torch.where(
            new_valid.unsqueeze(-1), new_trans, torch.zeros_like(new_trans)
        )

        # Skip mapping optional supervised fields, make sure not defined (not used while sampling)
        assert self.remaining_insertions is None
        assert self.deleted is None

        new_batch = DataCorrupted(
            t=self.t.clone(),
            trans_t=new_trans,
            birth_time=new_birth,
            motif_mask=new_motif,
        )

        insertion_mask = is_insertion & new_valid
        return new_batch, insertion_mask


@dataclass
class DataBridged:
    """Corrupted and aligned (length A) points at time t"""

    t: torch.Tensor  # (B,)
    trans_t: torch.Tensor  # (B, A, 3)
    birth_time: torch.Tensor  # (B, A) 0.0 for roots
    present_mask: torch.Tensor  # (B, A)
    motif_mask: torch.Tensor  # (B, A) bool
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
        B, A, D = self.trans_t.shape
        idx_pack, pack_mask, P_b, P_max = self._pack_indices()

        trans_t_pack = self.trans_t.gather(
            1, idx_pack.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, P_max, D)
        birth_time_pack = self.birth_time.gather(1, idx_pack)  # (B, P_max)
        motif_mask_pack = self.motif_mask.gather(1, idx_pack)  # (B, P_max)
        remaining_insertions_pack = self.remaining_insertions.gather(
            1, idx_pack
        )  # (B, P_max)
        deleted_pack = self.deleted.gather(1, idx_pack)  # (B, P_max)

        trans_t_pack = trans_t_pack * pack_mask.unsqueeze(-1).float()  # zero pad
        birth_time_pack = torch.where(
            pack_mask,
            birth_time_pack,
            torch.full_like(
                birth_time_pack, float("inf")
            ),  # infinite birth time for pad tokens
        )
        motif_mask_pack = torch.where(
            pack_mask, motif_mask_pack, torch.zeros_like(motif_mask_pack)
        )
        remaining_insertions_pack = torch.where(
            pack_mask,
            remaining_insertions_pack,
            torch.zeros_like(remaining_insertions_pack),  # 0 pad
        )
        deleted_pack = torch.where(
            pack_mask,
            deleted_pack,
            torch.zeros_like(deleted_pack),
        )

        return DataCorrupted(
            t=self.t,
            trans_t=trans_t_pack,
            birth_time=birth_time_pack,
            motif_mask=motif_mask_pack,
            remaining_insertions=remaining_insertions_pack,
            deleted=deleted_pack,
        )

    def validate(self) -> None:
        B, A, D = self.trans_t.shape
        if self.birth_time.shape != (B, A):
            raise ValueError("birth_time shape mismatch")
        if self.present_mask.shape != (B, A):
            raise ValueError("present_mask shape mismatch")
        if self.motif_mask.shape != (B, A):
            raise ValueError("motif_mask shape mismatch")
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
    pred_split_rate: torch.Tensor  # (B, P) non-negative remaining splits per token
    pred_split_pooled_log1p_rate: (
        torch.Tensor
    )  # (B,) log1p-space total remaining splits
    pred_del_logits: torch.Tensor  # (B, P) deletion logit per token

    def detach_clone(self) -> "ModelPrediction":
        """Detach and clone the prediction, e.g. to save in trajectory"""
        return ModelPrediction(
            pred_trans_1=self.pred_trans_1.detach().clone(),
            pred_split_rate=self.pred_split_rate.detach().clone(),
            pred_split_pooled_log1p_rate=self.pred_split_pooled_log1p_rate.detach().clone(),
            pred_del_logits=self.pred_del_logits.detach().clone(),
        )


@dataclass
class SampleTrajectory:
    samples: List[DataCorrupted] = field(
        default_factory=list
    )  # samples at each time step
    pred: List[ModelPrediction] = field(
        default_factory=list
    )  # predictions at each time step


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

        tree_plan = TreePlan.generate(motif_mask=motif_mask)
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

        # x_t (point_dim), t (1), birth_time (1), motif_mask (1), pos_embed
        self.input_dim = self.point_dim + 1 + 1 + 1 + self.pos_embed_dim
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
        self.split_rate_pred = nn.Linear(self.model_dim, 1)
        self.split_pooled_log1p_rate_pred = nn.Linear(self.model_dim, 1)
        self.del_logits_pred = nn.Linear(self.model_dim, 1)

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        B, P, _ = batch.trans_t.shape

        valid = batch.valid_mask.bool()  # (B, P)
        # attention mask to mask out positions not alive yet / padded
        key_padding_mask = ~valid

        pos_embed = get_index_embedding(
            torch.arange(P, device=batch.trans_t.device).unsqueeze(0).expand(B, -1),
            embed_size=self.pos_embed_dim,
            max_len=1024,
            pos_embed_method=PositionalEmbeddingMethod.rotary,
        )
        pos_embed = pos_embed * batch.valid_mask.unsqueeze(-1).float()

        x_t = torch.cat(
            [
                batch.trans_t,  # (B, P, 3)
                batch.t[:, None, None].expand(B, P, 1),  # (B, P, 1)
                batch.birth_time[:, :, None]
                .float()
                .clamp(0.0, 1.0),  # (B, P, 1) clamp +inf padding to 1.0
                batch.motif_mask[:, :, None].float(),  # (B, P, 1)
                pos_embed,  # (B, P, pos_embed_dim)
            ],
            dim=-1,
        )

        x_t = self.input_proj(x_t)
        x_t = self.transformer(x_t, src_key_padding_mask=key_padding_mask)

        x1_pred = self.x1_pred(x_t)

        # Predict nonnegative remaining-splits rates (Poisson-like regression)
        split_rate = F.softplus(self.split_rate_pred(x_t)).squeeze(-1)  # (B, P)
        del_logits = self.del_logits_pred(x_t).squeeze(-1)  # (B, P)

        # Masked mean pool over alive tokens to predict total remaining insertions per example
        valid_count = valid.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        pooled = (x_t * valid.unsqueeze(-1).float()).sum(
            dim=1
        ) / valid_count  # (B, model_dim)
        split_pooled_log1p_rate = self.split_pooled_log1p_rate_pred(pooled).squeeze(
            -1
        )  # (B,)

        return ModelPrediction(
            pred_trans_1=x1_pred,
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

    @abstractmethod
    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, d)
        x1_pred: torch.Tensor,  # (B, P, d)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: Optional[torch.Tensor] = None,  # (B, P)
    ) -> torch.Tensor:
        """Single Euler(-Maruyama) step for sampling.

        Noise is controlled by the coupler instance (e.g. sigma); if sigma is None/0, this is deterministic.
        """
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
        self, x1: torch.Tensor, tree: BatchedTreePlan  # (B, N, 3)
    ) -> torch.Tensor:
        """Build translation anchors for all nodes (leaves + internal) from leaf endpoints x1.

        Assumptions for this toy:
        - Leaf endpoints are broadcast into leaf slots using tree.leaf_map.
        - Internal nodes ("anchors") are derived purely from topology + descendant weights.
        """
        if x1.ndim != 3 or x1.shape[-1] != 3:
            raise ValueError(
                f"Expected x1 to have shape (B, N, 3); got {tuple(x1.shape)}"
            )

        device = x1.device
        B, N, _ = x1.shape
        A = tree.parent_idx.shape[1]

        # Broadcast x1 endpoints into leaf slots, zeros for internal nodes
        anchor = tree.broadcast_to_leaves(
            x1.to(torch.float32), fill_value=0.0
        )  # (B, A, 3)

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
        # TODO - consider zero array init instead of clone
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
        present_mask = tree.present_mask(t=t)  # (B, A)

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

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, 3)
        x1_pred: torch.Tensor,  # (B, P, 3)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
    ) -> torch.Tensor:
        if x_t.shape != x1_pred.shape:
            raise ValueError(
                f"x_t and x1_pred must match shape; got {tuple(x_t.shape)} vs {tuple(x1_pred.shape)}"
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
        v = (x1_pred - x_t) / denom
        x_next = x_t + v * float(dt)

        if (self.sigma is not None) and float(self.sigma) > 0.0 and float(dt) > 0.0:
            x_next = x_next + torch.randn_like(x_next) * (
                float(self.sigma) * math.sqrt(float(dt))
            )

        return x_next * valid_fmask + x_t * (1.0 - valid_fmask)


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
            motif_mask=tree.motif_mask,
            remaining_insertions=tree.remaining_insertions_t(t=t),
            deleted=tree.leaf_deleted,
        )
        bridged.validate()

        return bridged, trans_coupling

    def corrupt_batch(
        self, batch: DataBatch
    ) -> Tuple[DataBridged, TranslationCoupling]:
        # pick a single time to share across the batch,
        # simply so they have a similar number of insertion/deletions to simulate
        # since corruption is run across the batch
        shared_t = (
            torch.rand(1, device=self.device) * (1.0 - 2.0 * self.min_t) + self.min_t
        )
        t = torch.ones(batch.trans_1.shape[0], device=self.device) * shared_t  # (B,)

        return self.corrupt_to(batch=batch, t=t)

    def _init_sampling_batch(
        self,
        num_batch: int,
        num_roots: int,
        motif_mask: Optional[torch.Tensor] = None,
    ) -> DataCorrupted:
        """
        Initialize a batch of samples for sampling.
        """

        if motif_mask is None:
            motif_mask = torch.zeros(
                (num_batch, num_roots), dtype=torch.bool, device=self.device
            )

        t = (
            torch.ones((num_batch,), dtype=torch.float32, device=self.device)
            * self.min_t
        )
        birth_time = torch.zeros(
            (num_batch, num_roots), dtype=torch.float32, device=self.device
        )
        trans_0 = self.translation_coupler.sample_base(
            num_batch=num_batch, num_roots=num_roots, device=self.device
        )

        return DataCorrupted(
            t=t,
            birth_time=birth_time,
            motif_mask=motif_mask,
            trans_t=trans_0,
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
        num_batch: int = 1,
        num_roots: int = 50,
        num_steps: int = 200,
    ) -> SampleTrajectory:
        model.eval()
        device = self.device

        # Create initial batch, which we edit in-place through the trajectory
        # TODO - support taking input structure with motifs, use motif guidance
        batch = self._init_sampling_batch(
            num_batch=num_batch,
            num_roots=num_roots,
            motif_mask=None,
        )

        traj = SampleTrajectory()
        traj.samples.append(batch.detach_clone())

        with torch.no_grad():
            t_grid = torch.linspace(self.min_t, 1.0, steps=num_steps, device=device)
            for step in range(num_steps):
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
                traj.pred.append(pred.detach_clone())

                # Euler step for alive tokens' translations
                trans_next = self.translation_coupler.euler_step(
                    x_t=batch.trans_t,
                    x1_pred=pred.pred_trans_1,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                )
                batch.trans_t = trans_next

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

                # Save
                traj.samples.append(batch.detach_clone())

        return traj


""" Loss """


@dataclass
class BranchFlowLosses:
    total_loss: torch.Tensor
    base_trans_loss: torch.Tensor  # MSE on translations
    pairwise_loss: torch.Tensor  # local pairwise distance loss
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
    trans_loss_weight: float = 1.0
    pairwise_dist_loss_weight: float = 0.2
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

        return loss_per_batch.mean().clamp(max=10.0) * self.trans_loss_weight

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

        return loss_per_batch.mean() * self.pairwise_dist_loss_weight

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
        trans_coupling: TranslationCoupling,
    ) -> BranchFlowLosses:
        B, P, D = batch.trans_t.shape
        assert pred.pred_trans_1.shape == (B, P, D)

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
            base_loss + pairwise_loss + split_token_loss + split_pooled_loss + del_loss
        )

        return BranchFlowLosses(
            total_loss=total_loss,
            base_trans_loss=base_loss,
            pairwise_loss=pairwise_loss,
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
        num_roots = int(batch.tree.roots_mask.sum(dim=1)[:num_plots].max().item())
        x0 = self.translation_coupler.sample_base(
            num_batch=num_batch,
            num_roots=num_roots,
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

                valid_mask = (
                    corrupted.valid_mask.cpu().numpy().astype(bool)
                )  # (B, P_max)
                trans_t = corrupted.trans_t.cpu().numpy()  # (B, P_max, 3)
                motif_mask_aligned = (
                    batch.tree.motif_mask.cpu().numpy().astype(bool)
                )  # (B, A)

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

                    trans_t_alive = trans_t[i][valid_mask[i]]  # (P, 3)
                    node_ids_alive = idx_pack[i][valid_mask[i]]  # (P,)

                    # Leaf/anchor status from tree.total_leaves (supports deletion-duplicate leaves).
                    node_ids_t = torch.tensor(node_ids_alive, dtype=torch.long)
                    node_total = (
                        batch.tree.total_leaves[i].gather(0, node_ids_t).cpu().numpy()
                    )
                    is_leaf = node_total == 1
                    is_anchor = ~is_leaf

                    # Motif status is stored directly in aligned tree space
                    is_motif = motif_mask_aligned[i][node_ids_alive]
                    is_scaffold_leaf = is_leaf & (~is_motif)

                    # Underlying data index broadcast is still useful for color-coding
                    # TODO - planar color coding for anchors (since dont map to leaf)
                    ref_idx = batch.tree.leaf_map[i].gather(0, node_ids_t).cpu().numpy()

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
                            c=ref_idx,
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
        bridged, trans_coupling = self.interpolant.corrupt_batch(batch=batch)

        corrupted = bridged.pack_present()

        pred = self.forward(corrupted)

        loss = self.loss_calculator.calculate(
            batch=corrupted, pred=pred, trans_coupling=trans_coupling
        )

        self.log("loss/train", loss.total_loss.item(), prog_bar=True)
        self.log("loss/trans", loss.base_trans_loss.item(), prog_bar=True)
        self.log("loss/pairwise", loss.pairwise_loss.item())
        self.log("loss/split_token", loss.split_token_loss.item(), prog_bar=True)
        self.log("loss/split_pooled", loss.split_pooled_loss.item(), prog_bar=True)
        self.log("loss/del", loss.del_loss.item(), prog_bar=True)
        self.log("aux/t", bridged.t.mean().item())

        # MPS clean up
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
