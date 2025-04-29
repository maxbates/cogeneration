import torch

from cogeneration.config.base import InferenceSamplesConfig
from cogeneration.dataset.datasets import BaseDataset, LengthSamplingDataset
from cogeneration.dataset.motif_factory import Motif, Scaffold
from cogeneration.type.batch import BatchProps as bp
from cogeneration.type.batch import empty_feats


class TestBaseDataset:
    def test_segment_features_preserves_motif_and_masks_scaffold(self):
        """
        Test that BaseDataset.segment_features correctly preserves features in motif regions
        and applies default (masked) features in scaffold regions, returning the correct length.
        """
        # Create dummy features for a sequence of length 5
        N = 7
        feats = empty_feats(N)

        segments = [
            Scaffold(start=0, end=2, new_length=5),
            Motif(start=2, end=5, chain="A"),
            Scaffold(start=5, end=7, new_length=1),
        ]

        new_feats = BaseDataset.segment_features(feats=feats, segments=segments)

        # Confirm trans + rots are preserved in motif in new positions
        assert torch.equal(new_feats[bp.trans_1][5:8], feats[bp.trans_1][2:5])
        assert torch.equal(new_feats[bp.rotmats_1][5:8], feats[bp.rotmats_1][2:5])
        assert torch.equal(
            new_feats[bp.torsion_angles_sin_cos_1][5:8],
            feats[bp.torsion_angles_sin_cos_1][2:5],
        )

        # Check output length matches sum of segment lengths
        expected_length = sum(seg.length for seg in segments)
        assert new_feats[bp.res_mask].shape[0] == expected_length

        # Check metadata fields are carried over
        assert new_feats[bp.pdb_name] == feats[bp.pdb_name]

class TestLengthSamplingDataset:
    def test_multimer(self):
        cfg = InferenceSamplesConfig(
            samples_per_length=1,
            length_subset=[50, 99, 100, 500, 20000],
            multimer_fraction=1.0,
            multimer_min_length=50,
            chain_gap_dist=200,
        )
        dataset = LengthSamplingDataset(cfg=cfg)

        # short samples should be monomers
        len50 = dataset[0]
        assert (len50[bp.chain_idx] == 1.0).all()
        len99 = dataset[1]
        assert (len99[bp.chain_idx] == 1.0).all()
        # 100 should have 2 chains both 50 long
        len100 = dataset[2]
        assert (len100[bp.chain_idx]).float().mean() == 1.5  # half 1, half 2
        assert len100[bp.res_idx][50] == 250
        len500 = dataset[3]
        # 500 should have several chains
        assert len(len500[bp.chain_idx].unique()) > 1
        assert len(len500[bp.chain_idx].unique()) <= 10
        len20000 = dataset[4]
        # should basically never pick min length for all
        assert len(len500[bp.chain_idx].unique()) < (20000 / 50)



