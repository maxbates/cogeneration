import torch

from cogeneration.config.base import InferenceSamplesConfig
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.dataset.datasets import BaseDataset, LengthSamplingDataset
from cogeneration.dataset.motif_factory import ChainBreak, Motif, Scaffold
from cogeneration.type.batch import BatchProp as bp
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

        # segment start/end are *inclusive*
        segments = [
            Scaffold(start=0, end=2, new_length=5),  # 0,1,2 -> 0,1,2,3,4
            Motif(start=2, end=5, chain="A"),  # 2,3,4,5 -> 5,6,7,8
            ChainBreak(start=5, end=5),
            Scaffold(start=5, end=7, new_length=1),  # 6,7 -> 9
        ]

        new_feats = BaseDataset.segment_features(feats=feats, segments=segments)

        # check diffuse_mask and motif_mask
        # inpainting with guidance -> whole structure diffused
        assert (feats[bp.diffuse_mask] == 1).all()
        # segment 0 = scaffold
        assert (new_feats[bp.motif_mask][0:5] == 0).all()
        assert (new_feats[bp.diffuse_mask][0:5] == 1).all()
        # segment 1 = motif
        assert (new_feats[bp.motif_mask][5:9] == 1).all()
        assert (new_feats[bp.diffuse_mask][5:9] == 1).all()
        # segment 3 = scaffold
        assert (new_feats[bp.motif_mask][9:10] == 0).all()
        assert (new_feats[bp.diffuse_mask][9:10] == 1).all()

        # Confirm trans + rots + aatypes are preserved in motif in new positions
        assert torch.equal(new_feats[bp.trans_1][5:9], feats[bp.trans_1][2:6])
        assert torch.equal(new_feats[bp.rotmats_1][5:9], feats[bp.rotmats_1][2:6])
        assert torch.equal(new_feats[bp.aatypes_1][5:9], feats[bp.aatypes_1][2:6])
        assert torch.equal(
            new_feats[bp.torsion_angles_sin_cos_1][5:9],
            feats[bp.torsion_angles_sin_cos_1][2:6],
        )

        # trans + rots + aatypes masked in scaffold regions
        assert torch.equal(new_feats[bp.trans_1][0:5], torch.zeros(5, 3))
        assert torch.equal(
            new_feats[bp.rotmats_1][0:5], torch.eye(3)[None].repeat(5, 1, 1)
        )
        assert torch.equal(
            new_feats[bp.aatypes_1][0:5], torch.ones(5) * MASK_TOKEN_INDEX
        )
        assert torch.equal(
            new_feats[bp.torsion_angles_sin_cos_1][0:5], torch.zeros(5, 7, 2)
        )

        # chain break observed for final scaffold
        assert (new_feats[bp.chain_idx][0:9] == 1.0).all()
        assert (new_feats[bp.chain_idx][9:10] == 2.0).all()

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
        assert len100[bp.res_idx][50] == 1  # res_idx resets
        len500 = dataset[3]
        # 500 should have several chains
        assert len(len500[bp.chain_idx].unique()) > 1
        assert len(len500[bp.chain_idx].unique()) <= 10
        len20000 = dataset[4]
        # should basically never pick min length for all
        assert len(len500[bp.chain_idx].unique()) < (20000 / 50)
