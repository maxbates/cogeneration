import random
import re
from dataclasses import dataclass
from math import ceil, floor
from typing import List, Optional, Tuple

import numpy as np
import torch

from cogeneration.config.base import (
    DatasetInpaintingConfig,
    DatasetInpaintingMotifStrategy,
)


@dataclass
class Segment:
    start: int  # 0-indexed start in input residues
    end: int  # 0-indexed end in input residues, *inclusive*

    @property
    def length(self):
        return self.end - self.start + 1


@dataclass
class Motif(Segment):
    chain: Optional[str] = None
    chain_idx: Optional[int] = None

    def __post_init__(self):
        assert (
            self.chain is not None or self.chain_idx is not None
        ), "Either chain or chain_idx must be provided"


@dataclass
class Scaffold(Segment):
    new_length: int

    @property
    def length(self):
        return self.new_length


@dataclass
class ChainBreak(Segment):

    @property
    def length(self):
        return 0


@dataclass
class MotifFactory:
    cfg: DatasetInpaintingConfig
    rng: np.random.Generator

    def segments_from_contigmap(self, contigmap: str) -> List[Segment]:
        # Parse a contigmap like "5-15/A1-20/20/0 A39-50" into a list of segment objects.

        # TODO(inpainting) - require input `feats` and ground positions in feats
        #   motif should not require a chain. It's start / end should refer to feat indices.
        #   Just use flattened chains in `feats` as positions. Map the contigmap to those positions.
        #   Note, requires `chain_idx` and chains specified in `contigmap` to be aligned...
        segments = []
        length_so_far = 0

        # split contigmap on either "/" or ","
        tokens = re.split(r"[/,]", contigmap)
        for token in tokens:
            # special case for chain-break token: "0 ", since won't split on " "
            if token.startswith("0 "):
                cb = ChainBreak(start=length_so_far, end=length_so_far)
                segments.append(cb)
                # remove "0 " prefix and continue
                token = token[2:]

            if re.match(r"^\d+$", token):  # scaffold token e.g. "20"
                scaffold_length = int(token)
                scaffold = Scaffold(
                    start=length_so_far, end=length_so_far, new_length=scaffold_length
                )
                segments.append(scaffold)
                length_so_far += scaffold.new_length
            elif re.match(r"^\d+-\d+$", token):  # variable length scaffold e.g. "5-15"
                scaffold_length = random.randint(
                    int(token.split("-")[0]),
                    int(token.split("-")[1] + 1),  # end inclusive
                )
                scaffold = Scaffold(
                    start=length_so_far, end=length_so_far, new_length=scaffold_length
                )
                segments.append(scaffold)
                length_so_far += scaffold.new_length
            elif re.match(r"^[A-Za-z]\d+-\d+$", token):  # motif token e.g. "A1-20"
                chain = token[0]
                start_str, end_str = token[1:].split("-")  # end index not inclusive
                motif = Motif(start=int(start_str), end=int(end_str), chain=chain)
                segments.append(motif)
                length_so_far += motif.length
            else:
                raise ValueError(f"Invalid token '{token}' in contigmap {contigmap}")

        return segments

    @staticmethod
    def _is_multimeric(
        chain_idx: torch.Tensor,
    ) -> bool:
        """Check if the input is a multimeric structure"""
        return len(chain_idx.unique()) > 1

    @staticmethod
    def get_chain_breaks(chain_idx: torch.Tensor) -> List[int]:
        # return indices where chains change
        breaks = (chain_idx[1:] != chain_idx[:-1]).nonzero(as_tuple=False).flatten() + 1
        return breaks.tolist()

    @staticmethod
    def _plddt_start_stop(
        res_mask: torch.Tensor, plddt_mask: torch.Tensor
    ) -> Tuple[int, int, int]:
        """
        Trim low pLDDT residues to get start / stop bounds and length
        Returns positions of first and last `1` in `res_mask * plddt_mask`
        """
        N = plddt_mask.shape[-1]
        idx = torch.nonzero(plddt_mask * res_mask, as_tuple=False).squeeze()

        # if no residues pass the mask, return empty bounds
        if idx.numel() == 0:
            return 0, N, N

        # start = first index, stop_exclusive = last index + 1 for slicing
        start = int(idx.min().item())
        stop_exclusive = int(idx.max().item()) + 1
        length = stop_exclusive - start
        return start, stop_exclusive, length

    def generate_segments_from_motif_mask(
        self,
        motif_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        random_scale_range=(0.5, 2),
    ):
        """
        Generate segments from a diffuse mask.
        Motifs are defined by diffuse_mask == 0, scaffolds are defined by diffuse_mask == 1
        Scaffold regions are randomly scaled by a factor in random_scale_range.
        Chain breaks are inserted when `chain_idx` changes.
        """
        N = motif_mask.shape[0]
        segments: List[Segment] = []
        seg_start = 0
        current_is_motif = motif_mask[0]
        current_chain = int(chain_idx[0].item())

        for i in range(1, N):
            is_motif = bool(motif_mask[i].item())
            chain = int(chain_idx[i].item())

            # boundary if mask flips or chain changes
            if is_motif != current_is_motif or chain != current_chain:
                seg_end = i - 1
                # scaffold: randomly scale length of original segment span
                orig_len = seg_end - seg_start + 1
                new_len = (
                    orig_len
                    if orig_len <= 5
                    else int(round(orig_len * self.rng.uniform(*random_scale_range)))
                )

                # append motif or scaffold segment
                if current_is_motif:
                    segments.append(
                        Motif(
                            start=seg_start,
                            end=seg_end,
                            chain_idx=current_chain,
                        )
                    )
                else:
                    segments.append(
                        Scaffold(
                            start=seg_start,
                            end=seg_end,
                            new_length=new_len,
                        )
                    )

                # if chain has changed, insert a chain break
                if chain != current_chain:
                    cb = ChainBreak(start=seg_end, end=seg_end)
                    segments.append(cb)

                # reset for next segment
                seg_start = i
                current_is_motif = is_motif
                current_chain = chain

        # final tail segment
        seg_end_inclusive = N - 1
        # scaffold: randomly scale length
        orig_len = seg_end_inclusive - seg_start + 1
        new_length = (
            orig_len
            if orig_len <= 5
            else int(round(orig_len * self.rng.uniform(*random_scale_range)))
        )
        if current_is_motif:
            segments.append(
                Motif(start=seg_start, end=seg_end_inclusive, chain_idx=current_chain)
            )
        else:
            segments.append(
                Scaffold(start=seg_start, end=seg_end_inclusive, new_length=new_length)
            )

        return segments

    def _determine_bounds(
        self, res_mask: torch.Tensor, plddt_mask: torch.Tensor
    ) -> Tuple[int, int, int, int]:
        """
        Determine the acceptable bounds of the motif.
        Returns: (bound_start, bound_stop_exclusive, bound_length, num_res)

        If `cfg.trim_low_plddt_ends` is set, it will trim the ends of the motif based on pLDDT scores.
        Otherwise, it will use the full range of the res_mask.
        """
        N = len(res_mask)

        if self.cfg.trim_low_plddt_ends:
            plddt_start, plddt_stop_exclusive, plddt_length = self._plddt_start_stop(
                res_mask=res_mask, plddt_mask=plddt_mask
            )
            return plddt_start, plddt_stop_exclusive, plddt_length, N

        # If not trimming, use the full range of the res_mask
        return 0, N, N, N

    def _pick_chain_bounds(
        self,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
        chain_idx: torch.Tensor,
    ) -> Tuple[int, int, int, int]:
        """
        Pick a chain at random that exceeds minumum motif length, return its bounds.
        If no chain meets minimum length, ignore `chain_idx` and treat like monomer.
        """
        chain_to_length = {
            chain: (chain_idx == chain).sum().item() for chain in chain_idx.unique()
        }

        # limit to minimum length
        num_res = res_mask.shape[0]
        min_length = max(
            self.cfg.min_motif_len, floor(num_res * self.cfg.min_percent_motifs)
        )
        chain_to_length = {
            chain: length
            for chain, length in chain_to_length.items()
            if length >= min_length
        }
        if len(chain_to_length) == 0:
            return self._determine_bounds(res_mask=res_mask, plddt_mask=plddt_mask)

        # pick a random chain to use as the bounds
        chain_idx = int(self.rng.choice(list(chain_to_length.keys())))
        bound_start = (
            res_mask[chain_idx == chain_idx].nonzero(as_tuple=False).min().item()
        )
        bound_stop_exclusive = (
            res_mask[chain_idx == chain_idx].nonzero(as_tuple=False).max().item() + 1
        )
        bound_length = bound_stop_exclusive - bound_start
        return bound_start, bound_stop_exclusive, bound_length, num_res

    def generate_single_motif_diffuse_mask(
        self,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        break_across_chains: bool = True,
    ) -> torch.Tensor:
        """
        Generate a `diffuse_mask` for a single motif in a sequence.
        """
        # To break across chains, just pick one chain and set bounds.
        if break_across_chains and self._is_multimeric(chain_idx=res_mask):
            bound_start, bound_stop_exclusive, bound_length, num_res = (
                self._pick_chain_bounds(
                    res_mask=res_mask, plddt_mask=plddt_mask, chain_idx=chain_idx
                )
            )
        else:
            bound_start, bound_stop_exclusive, bound_length, num_res = (
                self._determine_bounds(res_mask=res_mask, plddt_mask=plddt_mask)
            )

        motif_length = self.rng.integers(
            # lower bound exceeding `min_motif_length`, `min_percent_motifs`
            max(self.cfg.min_motif_len, floor(num_res * self.cfg.min_percent_motifs)),
            # upper bound capped by `max_motif_length`, `max_percent_motifs`, max width of diffusable residues
            min(
                self.cfg.max_motif_len + 1,
                ceil(num_res * self.cfg.max_percent_motifs),
                bound_length,
            ),
        )

        motif_start = self.rng.integers(
            low=max(bound_start, self.cfg.min_padding),
            high=min(
                bound_stop_exclusive,
                bound_length - motif_length - self.cfg.min_padding + 1,
            ),
        )

        motif_end = motif_start + motif_length

        # diffuse everything but the motif
        diffuse_mask = torch.ones(num_res, dtype=torch.float32)
        diffuse_mask[motif_start:motif_end] = 0.0

        return diffuse_mask

    def generate_multiple_motif_diffuse_mask(
        self,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate a `diffuse_mask` for 1 or more motifs.

        This is a relatively simple scaffolding strategy, similar to FrameFlow.
        The primary difference from FrameFlow is the skew toward a smaller number of motifs.
        We pick some number of residues to be motifs, pick some number of motifs, then define the windows.
        The selection of window size + positions is also less random, and so shouldn't fail.
        We also allow for bounds on the motif length.
        However, note that `diffuse_mask` may select residues not in `res_mask`,
        which would subsequently be masked out.
        """
        bound_start, bound_stop_exclusive, bound_length, num_res = (
            self._determine_bounds(res_mask=res_mask, plddt_mask=plddt_mask)
        )

        # Choose how many motifs to create (skewed toward small numbers)
        possible_num_motifs = list(
            range(self.cfg.min_num_motifs, self.cfg.max_num_motifs + 1)
        )
        weights = [1.0 / ((m + 1) ** 1.5) for m in possible_num_motifs]
        probs = [w / sum(weights) for w in weights]
        num_motifs = self.rng.choice(possible_num_motifs, p=probs)

        # Compute how many residues in total will become motifs
        min_motif_total_length = max(
            # lower bound target percentage
            floor(self.cfg.min_percent_motifs * num_res),
            # require minimum motif length for each motif
            num_motifs * self.cfg.min_motif_len,
        )
        # ensure min <= bound
        min_motif_total_length = min(min_motif_total_length, bound_length)
        max_motif_total_length = min(
            # upper bound target percentage
            ceil(self.cfg.max_percent_motifs * num_res),
            # cap at maximum motif length
            num_motifs * self.cfg.max_motif_len,
            # cap at remaining sequence for motifs accounting for padding
            bound_length - (num_motifs - 1) * self.cfg.min_padding,
        )
        # ensure max >= min
        max_motif_total_length = max(max_motif_total_length, min_motif_total_length)
        total_motif_length = self.rng.integers(
            low=min_motif_total_length, high=max_motif_total_length + 1
        )

        # Break total_motif_length into lengths for each motif
        lengths = []
        leftover = total_motif_length
        for i in range(num_motifs):
            if i < num_motifs - 1:
                # Ensure there's enough room left for the remaining motifs (min lengths)
                max_len_for_current = (
                    leftover - (num_motifs - i - 1) * self.cfg.min_motif_len
                )
                length = self.rng.integers(
                    low=min(max_len_for_current, self.cfg.min_motif_len),
                    high=min(self.cfg.max_motif_len, max_len_for_current) + 1,
                )
                leftover -= length
            else:
                # Last motif takes whatever is leftover
                length = leftover
                leftover = 0
            lengths.append(length)

        # Pick start positions
        # Count remaining residues after motifs + minimal padding
        total_padding = (num_motifs - 1) * self.cfg.min_padding
        leftover_space = max(num_res - (total_motif_length + total_padding), 0)
        # Pick num_motifs random positions within [0, leftover_space] to use as starts
        if leftover_space > 0:
            offsets = self.rng.choice(
                range(leftover_space + 1), size=num_motifs, replace=False
            )
        else:
            offsets = [0] * num_motifs
        offsets.sort()
        # Get actual motif starts by incorporating motif lengths + padding
        motif_starts = [offsets[0]]
        for i in range(1, num_motifs):
            previous_start = motif_starts[i - 1]
            previous_motif_length = lengths[i - 1]
            position_gap = offsets[i] - offsets[i - 1]
            next_start = (
                previous_start
                + previous_motif_length
                + self.cfg.min_padding
                + position_gap
            )
            motif_starts.append(next_start)

        # Build motif_mask
        motif_mask = torch.zeros(num_res, dtype=torch.float32)
        for start, length in zip(motif_starts, lengths):
            end = min(start + length, num_res)
            motif_mask[start:end] = 1.0

        # diffuse_mask is the scaffold, i.e. not motif regions
        diffuse_mask = 1 - motif_mask

        return diffuse_mask

    def generate_masked_neighbors_diffuse_mask(
        self,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
        trans_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Pick a residue, mask N closest residues as scaffold, remaining structure as motif
        This method is similar to the published FoldFold2 method
        """
        bound_start, bound_stop_exclusive, bound_length, num_res = (
            self._determine_bounds(res_mask=res_mask, plddt_mask=plddt_mask)
        )

        dist2d = torch.linalg.norm(trans_1[:, None, :] - trans_1[None, :, :], dim=-1)

        # pick a random residue within bounds
        seed_idx = self.rng.integers(low=bound_start, high=bound_stop_exclusive)
        # sample a number of residues that will be scaffolded. ignore `motif_length` criteria, go off percentage.
        min_scaffold_length = min(
            bound_length, num_res - ceil(num_res * self.cfg.max_percent_motifs)
        )
        max_scaffold_length = max(
            bound_length, num_res - floor(num_res * self.cfg.min_percent_motifs)
        )
        scaffold_length = self.rng.integers(min_scaffold_length, max_scaffold_length)

        # get a distance cut off
        seed_dists = dist2d[seed_idx]
        dist_cutoff = torch.sort(seed_dists)[0][scaffold_length]

        # set the mask to all residues within the cutoff
        diffuse_mask = (seed_dists <= dist_cutoff).float()
        return diffuse_mask

    def generate_densest_neighbors_diffuse_mask(
        self,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
        trans_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample a residue with many neighbors, mask all neighbors within a distance threshold
        """
        bound_start, bound_stop_exclusive, bound_length, num_res = (
            self._determine_bounds(res_mask=res_mask, plddt_mask=plddt_mask)
        )

        # use 8.0 instead of 6.0 to allow picking up residues just outside "interaction" range
        interaction_distance_threshold = 8.0
        # larger distance for mask
        mask_distance_threshold = 12.0

        dist2d = torch.linalg.norm(trans_1[:, None, :] - trans_1[None, :, :], dim=-1)

        # weight each residue by the number of neighbors within the interaction distance
        # count neighbors within interaction threshold (exclude self)
        neighbor_counts = (dist2d <= interaction_distance_threshold).sum(dim=1) - 1
        # only keep residues with at least 3 neighbors (at least one more than immediate neighbors)
        valid = neighbor_counts >= 3
        if valid.sum() == 0:
            probs = torch.ones_like(neighbor_counts, dtype=torch.float)
        else:
            weights = neighbor_counts.clone().float() ** 1.5
            weights[~valid] = 0.0
            # only pick residues within bounds
            weights[:bound_start] = 0.0
            weights[bound_stop_exclusive:] = 0.0
            probs = weights / weights.sum()

        # sample a seed residue index
        seed_idx = int(torch.multinomial(probs, num_samples=1).item())

        # cap `mask_distance_threshold` using maximum scaffold length
        seed_dists = dist2d[seed_idx]
        max_scaffold_length = num_res - ceil(self.cfg.min_percent_motifs * num_res)
        max_scaffold_length_dist_cutoff = torch.sort(seed_dists)[0][max_scaffold_length]
        mask_distance_threshold = min(
            max_scaffold_length_dist_cutoff, mask_distance_threshold
        )

        # mask all residues within the mask threshold of that seed
        diffuse_mask = (dist2d[seed_idx] <= mask_distance_threshold).float()
        return diffuse_mask

    def generate_binding_interface_diffuse_mask(
        self,
        chain_idx: torch.Tensor,
        trans_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Mask residues in binding interface(s) as scaffold.

        Residues within `binding_threshold` across chains are marked as interacting.
        Then, residues between those residues and within `proximity_threshold` are added to the scaffold.
        Does not consider which chains are interacting, just all interactions across all chains.
        """
        N = chain_idx.shape[0]
        binding_threshold = self.cfg.interaction_dist_threshold_ang
        proximity_threshold = self.cfg.proximity_dist_threshold_ang

        # compute min distance to other residues
        dist_2d = torch.linalg.norm(trans_1[:, None, :] - trans_1[None, :, :], dim=-1)
        # mask out intra-chain distances
        dist_2d[chain_idx[:, None] == chain_idx[None, :]] = float("inf")

        # flag which residues have any partner within each threshold
        prox_flag = (dist_2d <= proximity_threshold).any(dim=1)
        bind_flag = (dist_2d <= binding_threshold).any(dim=1)

        # iterate through, inspect each proximity window, set ranges where some residues
        # are under `binding_threshold` to scaffold (1), rest is motifs (0)
        diffuse_mask = torch.zeros(N, dtype=torch.float32)
        i = 0
        while i < N:
            if not prox_flag[i]:
                i += 1
                continue
            start = i
            # consume proximity window
            while i < N and prox_flag[i]:
                i += 1
            end = i
            # if any residue here is in binding range, scaffold the whole window
            if bind_flag[start:end].any():
                diffuse_mask[start:end] = 1.0

        return diffuse_mask

    def generate_binding_interface_single_chain_diffuse_mask(
        self,
        chain_idx: torch.Tensor,
        trans_1: torch.Tensor,
        chain_to_mask: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Mask residues in binding interface, and in one chain (`chain_to_mask`) as scaffold.
        """
        # ablate only one chain at the interface
        N = chain_idx.shape[0]
        proximity_threshold = self.cfg.proximity_dist_threshold_ang
        if chain_to_mask is None:
            chain_to_mask = int(self.rng.choice(chain_idx.unique().tolist()))

        # compute pairwise distances
        dist_2d = torch.linalg.norm(trans_1[:, None, :] - trans_1[None, :, :], dim=-1)
        # mask out intra-chain distances
        dist_2d[chain_idx[:, None] == chain_idx[None, :]] = float("inf")

        # flag residues in proximity
        prox_flag = (dist_2d <= proximity_threshold).any(dim=1)

        # set residues in proximity to scaffolds (1) and rest to motifs (0)
        diffuse_mask = torch.zeros(N, dtype=torch.float32)
        diffuse_mask[(chain_idx == chain_to_mask) & prox_flag] = 1.0
        return diffuse_mask

    def generate_binder_diffuse_mask(
        self,
        chain_idx: torch.Tensor,
        chain_to_mask: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Mask out one chain entirely. Assumes any chain is reasonable target, i.e. interacting.
        """
        if chain_to_mask is None:
            chain_to_mask = int(self.rng.choice(chain_idx.unique().tolist()))

        diffuse_mask = torch.ones(chain_idx.shape[0], dtype=torch.float32)
        diffuse_mask[chain_idx == chain_to_mask] = 0.0
        return diffuse_mask

    def strategy_diffuse_mask(
        self,
        strategy: DatasetInpaintingMotifStrategy,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        res_idx: torch.Tensor,
        trans_1: torch.Tensor,
        rotmats_1: torch.Tensor,
        aatypes_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Get a `diffuse_mask` for a specified `strategy`.
        """
        if strategy == DatasetInpaintingMotifStrategy.single_motif:
            return self.generate_single_motif_diffuse_mask(
                res_mask=res_mask,
                plddt_mask=plddt_mask,
                chain_idx=chain_idx,
            )
        elif strategy == DatasetInpaintingMotifStrategy.variable_motifs:
            return self.generate_multiple_motif_diffuse_mask(
                res_mask=res_mask,
                plddt_mask=plddt_mask,
            )
        elif strategy == DatasetInpaintingMotifStrategy.random_neighbors:
            return self.generate_masked_neighbors_diffuse_mask(
                res_mask=res_mask, plddt_mask=plddt_mask, trans_1=trans_1
            )
        elif strategy == DatasetInpaintingMotifStrategy.densest_neighbors:
            return self.generate_densest_neighbors_diffuse_mask(
                res_mask=res_mask, plddt_mask=plddt_mask, trans_1=trans_1
            )
        elif strategy == DatasetInpaintingMotifStrategy.binding_interface_single_chain:
            assert self._is_multimeric(chain_idx=chain_idx)
            return self.generate_binding_interface_single_chain_diffuse_mask(
                chain_idx=chain_idx,
                trans_1=trans_1,
            )
        elif strategy == DatasetInpaintingMotifStrategy.binding_interface:
            assert self._is_multimeric(chain_idx=chain_idx)
            return self.generate_binding_interface_diffuse_mask(
                chain_idx=chain_idx,
                trans_1=trans_1,
            )
        elif strategy == DatasetInpaintingMotifStrategy.binder:
            assert self._is_multimeric(chain_idx=chain_idx)
            return self.generate_binder_diffuse_mask(
                chain_idx=chain_idx,
                chain_to_mask=None,
            )
        else:
            raise ValueError(f"Unknown motif selection strategy: {self.cfg.strategy}")

    def generate_motif_mask(
        self,
        res_mask: torch.Tensor,
        plddt_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        res_idx: torch.Tensor,
        trans_1: torch.Tensor,
        rotmats_1: torch.Tensor,
        aatypes_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Entrypoint to get a `motif_mask` [int tensor] specified by `DatasetInpaintingConfig`
        """
        strategy = self.cfg.strategy

        # if `strategy == ALL`, pick a random one, handling multimer/monomer strategies
        if strategy == DatasetInpaintingMotifStrategy.ALL:
            is_multimer = self._is_multimeric(chain_idx=chain_idx)
            while strategy == DatasetInpaintingMotifStrategy.ALL:
                new_strategy = self.rng.choice(list(DatasetInpaintingMotifStrategy))
                if not is_multimer and DatasetInpaintingMotifStrategy.is_multimeric(
                    new_strategy
                ):
                    continue
                strategy = new_strategy

        diffuse_mask = self.strategy_diffuse_mask(
            strategy=strategy,
            res_mask=res_mask,
            plddt_mask=plddt_mask,
            chain_idx=chain_idx,
            res_idx=res_idx,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
        )
        diffuse_mask = diffuse_mask.int()
        motif_mask = 1 - diffuse_mask
        return motif_mask
