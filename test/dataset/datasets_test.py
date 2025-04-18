import torch

from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.motif_factory import Motif, Scaffold
from cogeneration.dataset.util import empty_feats


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
