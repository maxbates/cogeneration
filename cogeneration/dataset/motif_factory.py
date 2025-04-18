import random
import re
from dataclasses import dataclass
from math import ceil, floor
from typing import List, Optional

import numpy as np
import torch

from cogeneration.config.base import DatasetInpaintingConfig


@dataclass
class Segment:
    start: int  # 0-indexed start in input residues
    end: int  # 0-indexed end in input residues, not inclusive

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
class MotifFactory:
    cfg: DatasetInpaintingConfig
    rng: np.random.Generator

    def segments_from_contigmap(self, contigmap) -> List[Segment]:
        # Parse a contigmap like "A1-20/20/A39-50" into a list of segment objects.
        segments = []
        length_so_far = 0

        # split contigmap on either "/" or ","
        tokens = re.split(r"[/,]", contigmap)
        for token in tokens:
            if re.match(r"^\d+$", token):  # scaffold token e.g. "20"
                scaffold_length = int(token)
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

    def generate_segments_from_diffuse_mask(
        self,
        diffuse_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        random_scale_range=(0.5, 2),
    ):
        """
        Generate segments from a diffuse mask.
        Motifs are defined by diffuse_mask == 0, scaffolds are defined by diffuse_mask == 1
        Scaffold regions are randomly scaled by a factor in random_scale_range.
        """
        rng = self.rng

        # TODO(multimer) handle multiple chains i.e. diffuse mask over multiple chains

        segments = []
        seg_start = 0
        current_val = diffuse_mask[0]
        for i in range(1, len(diffuse_mask)):
            if diffuse_mask[i] != current_val:
                seg_end = i - 1
                # 1 = scaffold region
                if current_val:
                    length = seg_end - seg_start + 1
                    factor = rng.uniform(*random_scale_range)  # random scale
                    target_length = int(round(length * factor))
                    segments.append(
                        Scaffold(start=seg_start, end=seg_end, new_length=target_length)
                    )
                # 0 = motif region
                else:
                    segments.append(
                        Motif(
                            start=seg_start,
                            end=seg_end,
                            chain_idx=chain_idx[seg_start].item(),
                        )
                    )
                current_val = diffuse_mask[i]
                seg_start = i

        # Process final segment.
        seg_end = len(diffuse_mask) - 1
        length = seg_end - seg_start + 1
        if current_val:
            factor = rng.uniform(*random_scale_range)
            target_length = int(round(length * factor))
            segments.append(
                Scaffold(start=seg_start, end=seg_end, new_length=target_length)
            )
        else:
            segments.append(Motif(start=seg_start, end=seg_end))

        return segments

    def generate_single_motif_diffuse_mask(
        self, res_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a `diffuse_mask` for a single motif in a sequence.
        """
        num_res = len(res_mask)
        cfg = self.cfg
        rng = self.rng

        motif_length = rng.integers(
            max(cfg.min_motif_len, floor(num_res * cfg.min_percent_motifs)),
            min(cfg.max_motif_len + 1, ceil(num_res * cfg.max_percent_motifs)),
        )

        motif_start = rng.integers(
            low=cfg.min_padding, high=num_res - motif_length - cfg.min_padding + 1
        )
        motif_end = motif_start + motif_length

        # diffuse everything but the motif
        diffuse_mask = torch.ones(num_res, dtype=torch.float32)
        diffuse_mask[motif_start:motif_end] = 0.0

        return diffuse_mask

    def generate_multiple_motif_diffuse_mask(
        self, res_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Generate a `diffuse_mask` for 1 or more motifs.

        This is a relatively simple scaffolding strategy which is similar to FrameFlow.
        We pick some number of residues to be motifs, pick some number of motifs, then define the windows.
        The primary difference is the skew toward a smaller number of motifs.
        The selection of window size + positions is also less random, and so shouldn't fail.
        We also allow for bounds on the motif length.
        However, note that `diffuse_mask` may select residues not in `res_mask`,
        which would subsequently be masked out.
        """
        num_res = len(res_mask)
        cfg = self.cfg
        rng = self.rng

        # Choose how many motifs to create (skewed toward small numbers)
        possible_motifs = list(range(cfg.min_num_motifs, cfg.max_num_motifs + 1))
        weights = [1.0 / ((m + 1) ** 1.5) for m in possible_motifs]
        probs = [w / sum(weights) for w in weights]
        num_motifs = rng.choice(possible_motifs, p=probs)

        # Compute how many residues in total will become motifs
        min_motif_total = max(
            # lower bound target percentage
            floor(cfg.min_percent_motifs * num_res),
            # require minimum motif length for each motif
            num_motifs * cfg.min_motif_len,
        )
        max_motif_total = min(
            # upper bound target percentage
            ceil(cfg.max_percent_motifs * num_res),
            # cap at maximum motif length
            num_motifs * cfg.max_motif_len,
            # cap at remaining sequence for motifs accounting for padding
            num_res - (num_motifs - 1) * cfg.min_padding,
        )
        max_motif_total = max(max_motif_total, min_motif_total)  # ensure max >= min
        total_motif_length = rng.integers(low=min_motif_total, high=max_motif_total + 1)

        # Break total_motif_length into lengths for each motif
        lengths = []
        leftover = total_motif_length
        for i in range(num_motifs):
            if i < num_motifs - 1:
                # Ensure there's enough room left for the remaining motifs (min lengths)
                max_len_for_current = (
                    leftover - (num_motifs - i - 1) * cfg.min_motif_len
                )
                length = rng.integers(
                    low=cfg.min_motif_len,
                    high=min(cfg.max_motif_len, max_len_for_current) + 1,
                )
                leftover -= length
            else:
                # Last motif takes whatever is leftover
                length = leftover
                leftover = 0
            lengths.append(length)

        # Pick start positions
        # Count remaining residues after motifs + minimal padding
        total_padding = (num_motifs - 1) * cfg.min_padding
        leftover_space = max(num_res - (total_motif_length + total_padding), 0)
        # Pick num_motifs random positions within [0, leftover_space] to use as starts
        if leftover_space > 0:
            offsets = rng.choice(
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
                previous_start + previous_motif_length + cfg.min_padding + position_gap
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

    def generate_diffuse_mask(self, res_mask: torch.Tensor) -> torch.Tensor:
        """
        Main entrypoint to get a diffuse_mask for a given set of residues.
        """
        # Some percentage of the time, diffuse everything to effectively do unconditional generation
        if random.random() < self.cfg.unconditional_percent:
            diffuse_mask = torch.ones_like(res_mask)
        else:
            diffuse_mask = self.generate_multiple_motif_diffuse_mask(res_mask)

        return diffuse_mask
