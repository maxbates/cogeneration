import os
import tempfile
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from matplotlib import pyplot as plt

from cogeneration.util.log import rank_zero_logger
from varco.tensor_utils import SeededRNG, gather_and_pad, pad_and_stack, to_device

# Trees define the corruption process, which in a nutshell looks like:
# Sample some proteins of length N (translations, rotations, and sequence) using LengthBatched protein dataset
# Sample a motif_mask, with 1+ scaffolds
# Sample some number of roots per scaffold, starting points at t=0 for the motifs + roots
# Define a tree, with splits (anchors at intermediate time points) and deletions over trajectory

# Data Flow:
# TreePlan - t=0 -> t=1 per-sample tree topology (length A) with birth/split/delete times
# BatchedTreePlan - collated TreePlans with (B, A_max) tensors

logger = rank_zero_logger(__name__)


@dataclass
class TreePlan:
    """Per-sample (non-batched) tree topology and sampled times (domain-agnostic)."""

    num_leaves: int  # N_data + N_deleted (leaf nodes only)
    num_deletions: int  # number of leaves destined to be deleted
    num_nodes: int  # A (leaves + internal)

    # topology
    topo_order: torch.Tensor  # (A,) long, structural topo (parent-before-child)
    motif_mask: torch.Tensor  # (A,) bool; True only for motif leaves (not anchors)
    chain_idx: torch.Tensor  # (A,) long; chain id per node
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
        chain_idx: Optional[torch.Tensor] = None,  # (N,) assume monomer if not provided
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
        # For early_power(p) hazard, the PDF is Beta(1, p), so (1.0, 2.5) matches
        # the default sampling hazard early_power(2.5).
        # Use (1.0, 1.0) for uniform sampling (no bias).
        split_time_beta: tuple[float, float] = (1.0, 2.5),
        # Beta distribution parameters for biasing deletion times.
        # Default (1.0, 1.0) is uniform, matching uniform delete_hazard in sampling.
        delete_time_beta: tuple[float, float] = (1.0, 1.0),
        # Maximum time for indel events. Events are clamped to this value.
        # Set to < 1.0 to match sampling behavior where late indels are suppressed
        # via indel sharpening. Default 0.85 matches effective sampling cutoff.
        max_indel_time: float = 0.85,
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
        if chain_idx is None:
            chain_idx_b = torch.ones(N_data, dtype=torch.long, device=device)
        else:
            chain_idx_b = chain_idx.to(device=device).long()
        if chain_idx_b.shape != motif_mask_b.shape:
            raise ValueError(
                "chain_idx must have the same shape as motif_mask; "
                f"got {tuple(chain_idx_b.shape)} vs {tuple(motif_mask_b.shape)}"
            )

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
        leaf_chain_idx = chain_idx_b[leaf_map_leaves]  # (num_leaves,)
        is_scaffold = ~leaf_is_motif

        # scaffold_start marks the first position of each scaffold span,
        # splitting spans on chain breaks even when motif_mask is all False.
        prev_is_scaffold = torch.cat(
            [torch.tensor([False], device=device), is_scaffold[:-1]]
        )
        chain_break = torch.cat(
            [
                torch.tensor([False], device=device),
                leaf_chain_idx[1:] != leaf_chain_idx[:-1],
            ]
        )
        scaffold_start = is_scaffold & (~prev_is_scaffold | chain_break)
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
        extra_root_chain_idx: List[int] = []
        scaffold_K: Dict[int, int] = {}  # gid -> K_scaffold

        for gid, active in groups.items():
            is_scaffold_span = gid < span_gid

            if is_scaffold_span:
                span_len = len(active)
                span_start_leaf = min(active)
                span_end_leaf = max(active)
                span_chain_idx = int(leaf_chain_idx[span_start_leaf].item())
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
                        extra_root_chain_idx.append(span_chain_idx)
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

        # --- Chain index propagation
        chain_idx_aligned = torch.zeros((A,), dtype=torch.long, device=device)
        if num_leaves_original > 0:
            chain_idx_aligned[:num_leaves_original] = leaf_chain_idx[
                :num_leaves_original
            ]
        if extra_deleted_roots:
            extra_chain = torch.tensor(
                extra_root_chain_idx, dtype=torch.long, device=device
            )
            chain_idx_aligned[extra_t] = extra_chain

        # Internal nodes inherit chain id from children.
        for node in topo_order.flip(0).tolist():
            c0, c1 = children[node][0], children[node][1]
            if c0 < 0 and c1 < 0:
                continue
            c0_val = int(chain_idx_aligned[c0].item()) if c0 >= 0 else 0
            c1_val = int(chain_idx_aligned[c1].item()) if c1 >= 0 else 0
            if c0_val == c1_val:
                chain_idx_aligned[node] = c0_val
            elif c0_val == 0:
                chain_idx_aligned[node] = c1_val
            elif c1_val == 0:
                chain_idx_aligned[node] = c0_val
            else:
                logger.warning(
                    "Conflicting chain_idx in TreePlan.generate: node=%d child0=%d child1=%d",
                    int(node),
                    c0_val,
                    c1_val,
                )
                chain_idx_aligned[node] = c0_val

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

                # Sample split times using Beta distribution scaled to [t0, T].
                # u ~ Beta(alpha, beta) gives values in [0, 1] biased toward 0 when alpha < beta.
                # Then st = t0 + (T - t0) * u maps to [t0, T].
                # With Beta(1, p), this matches the early_power(p) hazard CDF.
                alpha, beta = split_time_beta
                u = rng.sample_beta(len(valid_nodes), alpha, beta, device=device)
                T = max_indel_time - 2 * min_t  # buffer for deletions
                st = t0_valid + (T - t0_valid).clamp_min(min_t) * u
                st = torch.clamp(st, min=min_t, max=T)

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
        #
        # We also clamp to max_indel_time to match sampling behavior where late deletions
        # are suppressed via indel sharpening.
        min_delete_eps = min_t
        max_delete_time = min(max_indel_time, 1.0 - min_t)

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
            alpha, beta = delete_time_beta
            u = rng.sample_beta(int(num_deleted), alpha, beta, device=device)

            # Compute deletion times: dt = birth + (max_delete_time - birth) * u
            # This samples in [birth, max_delete_time] instead of [birth, 1]
            dt = b + (max_delete_time - b).clamp_min(min_delete_eps) * u

            # Clamp to valid range [birth + min_delete_eps, max_delete_time]
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
            chain_idx=chain_idx_aligned,
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
        if self.chain_idx.shape != (A,):
            raise ValueError("chain_idx shape mismatch")

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
    chain_idx: torch.Tensor  # (B, A_max) long
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
            chain_idx=to_device(self.chain_idx, device),
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
            chain_idx=pad_and_stack(
                [p.chain_idx for p in plans], A_max, fill_value=0, dtype=torch.long
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
