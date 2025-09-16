from pathlib import Path

import numpy as np
import pytest
import torch
from numpy.random import default_rng

from cogeneration.config.base import (
    DatasetConfig,
    DatasetInpaintingConfig,
    DatasetInpaintingMotifStrategy,
)
from cogeneration.dataset.featurizer import BatchFeaturizer
from cogeneration.dataset.motif_factory import ChainBreak, Motif, MotifFactory, Scaffold
from cogeneration.dataset.process_pdb import process_pdb_file
from cogeneration.type.batch import BatchProp as bp


class TestMotifFactory:
    @pytest.mark.parametrize("strategy", list(DatasetInpaintingMotifStrategy))
    def test_generate_diffuse_mask(
        self, strategy: DatasetInpaintingMotifStrategy, pdb_2qlw_path
    ):
        if strategy == DatasetInpaintingMotifStrategy.ALL:
            return

        cfg = DatasetInpaintingConfig(strategy=strategy)
        rng = default_rng(0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        processed_pdb = process_pdb_file(
            pdb_file_path=str(pdb_2qlw_path),
            pdb_name="2qlw",
        )
        pdb_batch_features = BatchFeaturizer.batch_features_from_processed_file(
            processed_file=processed_pdb,
            csv_row={},
            cfg=DatasetConfig(),
            processed_file_path=str(pdb_2qlw_path),
        )

        motif_mask = factory.generate_motif_mask(
            res_mask=pdb_batch_features[bp.res_mask],
            plddt_mask=pdb_batch_features[bp.plddt_mask],
            chain_idx=pdb_batch_features[bp.chain_idx],
            res_idx=pdb_batch_features[bp.res_idx],
            trans_1=pdb_batch_features[bp.trans_1],
            rotmats_1=pdb_batch_features[bp.rotmats_1],
            aatypes_1=pdb_batch_features[bp.aatypes_1],
        )
        assert motif_mask.shape == pdb_batch_features[bp.res_mask].shape
        assert isinstance(motif_mask, torch.Tensor)

        diffuse_mask = 1 - motif_mask
        assert not (
            diffuse_mask == 0
        ).all(), "At least one residue should be scaffolded"
        assert not (
            diffuse_mask == 1
        ).all(), "At least one residue should be part of the motif"

    def test_generate_single_motif_diffuse_mask(self):
        # There should be exactly one contiguous motif of length 5
        cfg = DatasetInpaintingConfig(
            min_motif_len=5,
            max_motif_len=5,
            min_percent_motifs=0.0,
            max_percent_motifs=1.0,
            min_padding=0,
        )

        rng = default_rng(0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        N = 20
        res_mask = torch.ones(N, dtype=torch.float32)
        plddt_mask = torch.ones(N, dtype=torch.float32)
        chain_idx = torch.ones(N, dtype=torch.float32)
        mask = factory.generate_single_motif_diffuse_mask(
            res_mask=res_mask,
            plddt_mask=plddt_mask,
            chain_idx=chain_idx,
        )
        assert isinstance(mask, torch.Tensor)
        assert mask.shape == (N,)
        assert mask.dtype == torch.float32

        # Identify motif region (zeros)
        mask_arr = mask.numpy()
        zeros = np.where(mask_arr == 0)[0]
        assert zeros.size == 5
        assert zeros[-1] - zeros[0] + 1 == zeros.size

        # Ensure all other positions are scaffold (ones)
        ones = np.where(mask_arr == 1)[0]
        expected_ones = set(range(N)) - set(zeros.tolist())
        assert set(ones.tolist()) == expected_ones

    def test_generate_segments_from_diffuse_mask_fixed_scale(self):
        """
        Test that generate_segments_from_diffuse_mask produces correct segments
        when using a fixed scale factor of 1.0 (no length change).
        """
        rng = default_rng(0)
        factory = MotifFactory(cfg=DatasetInpaintingConfig(), rng=rng)

        # Expect three segments: scaffold 0-1 (len 2), motif 2-3 (len 2), scaffold 4-4 (len 1)
        diffuse_mask = torch.tensor([1, 1, 0, 0, 1]).int()
        motif_mask = 1 - diffuse_mask

        segments = factory.generate_segments_from_motif_mask(
            motif_mask,
            chain_idx=torch.tensor([1, 1, 1, 1, 1]),
            random_scale_range=(1.0, 1.0),
        )

        assert len(segments) == 3
        # First segment: Scaffold
        seg0 = segments[0]
        assert isinstance(seg0, Scaffold)
        assert seg0.start == 0 and seg0.end == 1 and seg0.new_length == 2
        # Second segment: Motif
        seg1 = segments[1]
        assert isinstance(seg1, Motif)
        assert seg1.start == 2 and seg1.end == 3 and seg1.length == 2
        # Third segment: Scaffold
        seg2 = segments[2]
        assert isinstance(seg2, Scaffold)
        assert seg2.start == 4 and seg2.end == 4 and seg2.new_length == 1
        # Total new length equals original mask length
        total_length = sum(seg.length for seg in segments)
        assert total_length == diffuse_mask.size(0)

    def test_generate_masked_neighbors_diffuse_mask(self):
        """
        Test that generate_segments_from_diffuse_mask produces correct segments
        when using a fixed scale factor of 1.0 (no length change).
        """
        rng = default_rng(0)
        cfg = DatasetInpaintingConfig()
        factory = MotifFactory(cfg=cfg, rng=rng)

        N = 20
        res_mask = torch.ones(N, dtype=torch.float32)
        plddt_mask = torch.ones(N, dtype=torch.float32)
        trans_1 = torch.randn((N, 3), dtype=torch.float32) * 10

        diffuse_mask = factory.generate_masked_neighbors_diffuse_mask(
            res_mask=res_mask,
            plddt_mask=plddt_mask,
            trans_1=trans_1,
        )

        assert (diffuse_mask == 0.0).sum() <= N * cfg.max_percent_motifs
        assert (diffuse_mask == 0.0).sum() >= N * cfg.min_percent_motifs
        assert (
            diffuse_mask.shape == res_mask.shape
        ), "Output mask shape should match input mask shape"
        assert (
            diffuse_mask == 1.0
        ).any(), "At least one residue should remain scaffolded"
        assert (
            diffuse_mask == 0.0
        ).any(), "At least one residue should be part of the motif"

    def test_generate_segments_respects_max_length_with_minimum_scaffolds(self):
        """
        Define a simple motif/scaffold pattern and enforce a max length that is
        only slightly above motifs + minimum scaffold lengths. Ensure segments are
        shortened accordingly and never below the minimum allowed per-scaffold length.
        """
        # Motif (1) and scaffold (0) pattern: M M S S S M M S S
        motif_mask = torch.tensor([1, 1, 0, 0, 0, 1, 1, 0, 0]).int()
        chain_idx = torch.ones_like(motif_mask)

        cfg = (
            DatasetInpaintingConfig()
        )  # default scaffold_length_scale=2.0 -> min_scale=0.5
        rng = default_rng(0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        # Compute motif and scaffold properties for expected bounds
        # Motifs: [0-1] len 2, [5-6] len 2 -> total 4
        motif_total = 4
        scaffold_orig_lengths = [3, 2]  # [2-4], [7-8]
        min_scale = 1.0 / float(cfg.scaffold_length_scale)  # 0.5 by default
        min_scaffold_lengths = [
            int(np.ceil(l * min_scale)) for l in scaffold_orig_lengths
        ]
        # Choose max length just 1 above the minimum feasible total
        max_total_length = motif_total + sum(min_scaffold_lengths) + 1

        segments = factory.generate_segments_from_motif_mask(
            motif_mask=motif_mask,
            chain_idx=chain_idx,
            max_total_length=max_total_length,
        )

        total_len = sum(seg.length for seg in segments)
        assert total_len == max_total_length

        # Extract scaffold new lengths and validate against min lengths
        sc_new_lengths = [
            seg.new_length for seg in segments if isinstance(seg, Scaffold)
        ]
        assert len(sc_new_lengths) == 2
        assert sc_new_lengths[0] >= min_scaffold_lengths[0]
        assert sc_new_lengths[1] >= min_scaffold_lengths[1]

        # With the chosen weights, the shorter scaffold should shrink by 1
        # Original scaffolds: [3, 2]; min allowed: [2, 1]; delta=1 -> expect [3, 1]
        assert sc_new_lengths == [3, 1]

    def test_generate_densest_neighbors_diffuse_mask(self):
        cfg = DatasetInpaintingConfig()
        rng = default_rng(seed=0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        N = 20
        res_mask = torch.ones(N, dtype=torch.float32)
        plddt_mask = torch.ones(N, dtype=torch.float32)
        trans_1 = torch.randn((N, 3), dtype=torch.float32) * 10

        diffuse_mask = factory.generate_densest_neighbors_diffuse_mask(
            res_mask=res_mask,
            plddt_mask=plddt_mask,
            trans_1=trans_1,
        )

        assert (
            diffuse_mask.shape == res_mask.shape
        ), "Output mask shape should match input mask shape"
        assert (
            diffuse_mask == 1.0
        ).any(), "At least one residue should remain scaffolded"
        assert (
            diffuse_mask == 0.0
        ).any(), "At least one residue should be part of the motif"

    def test_segments_from_contigmap(self):
        cfg = DatasetInpaintingConfig()
        rng = default_rng(0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        contigmap = "2/B4-6/0 C7-8"
        segments = factory.segments_from_contigmap(contigmap)
        assert len(segments) == 4

        seg0 = segments[0]
        assert isinstance(seg0, Scaffold)
        assert seg0.start == 0 and seg0.end == 0
        assert seg0.new_length == 2

        seg1 = segments[1]
        assert isinstance(seg1, Motif)
        assert seg1.chain == "B"
        assert seg1.start == 4 and seg1.end == 6
        assert seg1.length == 3

        seg2 = segments[2]
        assert isinstance(seg2, ChainBreak)

        seg3 = segments[3]
        assert isinstance(seg3, Motif)
        assert seg3.chain == "C"
        assert seg3.start == 7 and seg3.end == 8
        assert seg3.length == 2

    def test_segments_from_contigmap_invalid_token_raises(self):
        cfg = DatasetInpaintingConfig()
        rng = default_rng(0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        invalid = "foo"
        try:
            factory.segments_from_contigmap(invalid)
            assert False, "Expected ValueError for invalid token"
        except ValueError as e:
            assert "Invalid token" in str(e)
