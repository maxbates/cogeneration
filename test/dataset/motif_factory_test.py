import numpy as np
import torch
from numpy.random import default_rng

from cogeneration.config.base import DatasetInpaintingConfig
from cogeneration.dataset.motif_factory import Motif, MotifFactory, Scaffold


class TestMotifFactory:
    def test_generate_single_motif_diffuse_mask(self):
        # There should be exactly one contiguous motif of length 5
        cfg = DatasetInpaintingConfig(
            min_motif_len=5,
            max_motif_len=5,
            min_num_motifs=1,
            max_num_motifs=1,
            min_percent_motifs=0.0,
            max_percent_motifs=1.0,
            min_padding=0,
        )

        # Use a fixed random seed for reproducibility
        rng = default_rng(0)
        factory = MotifFactory(cfg=cfg, rng=rng)

        N = 20
        res_mask = torch.ones(N, dtype=torch.float32)
        mask = factory.generate_single_motif_diffuse_mask(res_mask)
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

        segments = factory.generate_segments_from_diffuse_mask(
            diffuse_mask,
            chain_idx=torch.tensor(
                [1, 1, 1, 1, 1]
            ),  # TODO(multimer) test multiple chains with mask over chain break
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
