import pytest
import torch

from cogeneration.config.base import DatasetFilterConfig, InferenceSamplesConfig
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.noise_mask import torsions_empty
from cogeneration.dataset.datasets import (
    BaseDataset,
    BatchFeaturizer,
    LengthSamplingDataset,
)
from cogeneration.dataset.motif_factory import ChainBreak, Motif, Scaffold
from cogeneration.dataset.test_utils import create_pdb_batch
from cogeneration.type.batch import METADATA_BATCH_PROPS
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import empty_feats
from cogeneration.type.task import DataTask


class TestBaseDataset:
    def test_pdb_batch(self, mock_cfg):
        # use inpainting so motif_mask is present
        mock_cfg.data.task = DataTask.inpainting
        batch = create_pdb_batch(cfg=mock_cfg)

        assert batch is not None
        for batch_prop in bp:
            if batch_prop in METADATA_BATCH_PROPS:
                continue

            # all batch properties should be present
            assert (
                batch_prop in batch
            ), f"Batch property {batch_prop} not found in batch"

            # non-metadata properties should be tensors
            assert isinstance(
                batch[batch_prop], torch.Tensor
            ), f"Batch property {batch_prop} should be a tensor"

    @pytest.mark.parametrize("task", [DataTask.inpainting, DataTask.hallucination])
    def test_features_defined(self, task, mock_cfg_uninterpolated):
        """
        Ensure that certain features are present in the batch esp. when their generation probabilities are set to 0 (i.e. always on).
        """

        # always generate motifs
        mock_cfg_uninterpolated.data.task = task
        mock_cfg_uninterpolated.interpolant.inpainting_unconditional_prop = 0.0
        mock_cfg_uninterpolated.interpolant.codesign_forward_fold_prop = 0.0
        mock_cfg_uninterpolated.interpolant.codesign_inverse_fold_prop = 0.0

        # always generate hotspots + contact conditioning
        # hotspots requires multimers
        mock_cfg_uninterpolated.dataset.filter = DatasetFilterConfig.multimeric()
        mock_cfg_uninterpolated.dataset.hotspots.hotspots_prob_disabled = 0.0
        mock_cfg_uninterpolated.dataset.contact_conditioning.conditioning_prob_disabled = (
            0.0
        )
        mock_cfg_uninterpolated.dataset.contact_conditioning.conditioning_prob_motif_only = (
            0.0
        )
        # smaller gap, since short proteins in test
        mock_cfg_uninterpolated.dataset.contact_conditioning.min_res_gap = 3

        cfg = mock_cfg_uninterpolated.interpolate()

        batch = create_pdb_batch(cfg=cfg, training=False)

        assert bp.contact_conditioning in batch
        assert (batch[bp.contact_conditioning] > 0).any()

        assert bp.hot_spots in batch
        assert batch[bp.hot_spots].sum() > 0

        if task == DataTask.inpainting:
            assert bp.motif_mask in batch
            assert (batch[bp.motif_mask] > 0).any()

    def test_segment_features_preserves_motif_and_masks_scaffold(self):
        """
        Test that BaseDataset.segment_features correctly preserves features in motif regions
        and applies default (masked) features in scaffold regions, returning the correct length.
        """
        # Create dummy features for a sequence of length 5
        N = 7
        feats = empty_feats(N)

        # Don't require input feats contain motif_mask
        assert bp.motif_mask not in feats

        # segment start/end are *inclusive*
        segments = [
            Scaffold(start=0, end=2, new_length=5),  # 0,1,2 -> 0,1,2,3,4
            Motif(start=2, end=5, chain="A"),  # 2,3,4,5 -> 5,6,7,8
            ChainBreak(start=5, end=5),
            Scaffold(start=5, end=7, new_length=1),  # 6,7 -> 9
        ]

        new_feats = BatchFeaturizer.segment_features(feats=feats, segments=segments)

        # check diffuse_mask and motif_mask
        # adds motif_mask prop
        assert bp.motif_mask in new_feats
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
            new_feats[bp.torsions_1][5:9],
            feats[bp.torsions_1][2:6],
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
            new_feats[bp.torsions_1][0:5],
            torsions_empty(num_batch=1, num_res=5, num_angles=7, device="cpu").squeeze(
                0
            ),
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
