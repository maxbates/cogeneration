import copy as _copy
import math

import numpy as np
import pytest
import torch

from cogeneration.config.base import (
    DatasetFilterConfig,
    InterpolantAATypesInterpolantTypeEnum,
)
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.data_transforms import make_one_hot
from cogeneration.data.fm.aatypes import (
    FlowMatcherAATypesMasking,
    FlowMatcherAATypesUniform,
)
from cogeneration.data.fm.rotations import FlowMatcherRotations
from cogeneration.data.fm.torsions import FlowMatcherTorsions
from cogeneration.data.fm.translations import FlowMatcherTrans
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import (
    centered_gaussian,
    centered_harmonic,
    torsions_noise,
    uniform_so3,
)
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass
from cogeneration.data.so3_utils import angle_from_rotmat
from cogeneration.dataset.test_utils import MockDataloader, create_pdb_batch
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.task import DataTask


class TestInterpolant:
    """Test suite for the Interpolant class."""

    def test_batch_ot(self, mock_cfg_uninterpolated):
        """
        Test the _batch_ot method of the Interpolant class.
        This method should:
        1. Compute optimal transport between two batches of translations
        2. Re-center both batches
        3. Return the OT mapping of trans_0 to trans_1
        """
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        device = torch.device("cpu")
        interpolant = Interpolant(mock_cfg.interpolant)
        interpolant.set_device(device)

        num_batch, num_res = 2, 10

        res_mask = torch.ones(num_batch, num_res, device=device)

        # Set some residues to be fixed
        diffuse_mask = torch.ones(num_batch, num_res, device=device)
        diffuse_mask[:, 2:5] = 0
        # ensure diffuse_mask is actually a mask
        assert diffuse_mask.sum() > 0 and (diffuse_mask == 0).sum() > 0

        trans_1 = centered_gaussian(*res_mask.shape, device=device)
        assert torch.allclose(
            batch_center_of_mass(trans_1), torch.zeros(num_batch, 3), atol=0.1
        )

        # Add a translation to trans_0 to ensure it is off-center
        trans_0 = centered_gaussian(*res_mask.shape, device=device)
        trans_0_offset = torch.tensor([3.0, 5.0, 0.0], device=device)
        trans_0 += trans_0_offset
        assert torch.allclose(batch_center_of_mass(trans_0), trans_0_offset, atol=0.1)

        # Pre-compute the alignment, ensure centered
        trans_0_aligned_centered, _, _ = batch_align_structures(
            trans_0, trans_1, mask=res_mask, center=True
        )
        trans_0_aligned_centered.reshape(num_batch, num_res, 3)
        assert torch.allclose(
            batch_center_of_mass(trans_0_aligned_centered),
            torch.zeros(num_batch, 3),
            atol=0.1,
        )

        result = interpolant.trans_fm.batch_ot(
            trans_0, trans_1, res_mask=res_mask, center=True
        )

        # ensure centered
        assert result.shape == trans_1.shape
        assert torch.allclose(
            batch_center_of_mass(result), torch.zeros(num_batch, 3), atol=0.1
        )

        # Check that the result has lower total distance than the original noisy batch
        original_distances = torch.linalg.norm(trans_0 - trans_1, dim=-1)
        result_distances = torch.linalg.norm(result - trans_1, dim=-1)
        assert torch.sum(result_distances) <= torch.sum(original_distances)

    def test_set_corruption_times_inpainting(self, mock_cfg_uninterpolated):
        """
        Test some portion of examples are converted to non-inpainting tasks
        """
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        mock_cfg_uninterpolated.interpolant.codesign_separate_t = True
        mock_cfg_uninterpolated.interpolant.codesign_inverse_fold_prop = 0.25
        mock_cfg_uninterpolated.interpolant.codesign_forward_fold_prop = 0.25
        mock_cfg_uninterpolated.interpolant.inpainting_unconditional_prop = 0.5
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        # create a very large batch so we'll observe each case
        B, N = 100, 30

        dataloader = MockDataloader(
            sample_lengths=[N] * B,
            task=DataTask.inpainting,
            corrupt=False,
            multimer=False,
            batch_size=B,
        )
        batch = next(iter(dataloader))

        # check batch masks
        assert batch[bp.diffuse_mask].sum() == B * N
        percent_motifs = batch[bp.motif_mask].float().mean()
        assert 0.2 < percent_motifs < 0.8

        # sets in place
        interpolant._set_corruption_times(batch, task=DataTask.inpainting)

        # some rows will have a domain set to t=1 for forward / inverse folding
        fwd_mask = batch[nbp.cat_t] == 1
        inv_mask = batch[nbp.r3_t] == 1
        # some rows will reset the motif mask for non-motif training
        no_motif_mask = torch.sum(batch[bp.motif_mask], dim=-1) == 0
        with_motif_mask = ~no_motif_mask

        # all cases should be present
        assert 0.1 < fwd_mask.float().mean() < 0.4  # ~0.25
        assert 0.1 < inv_mask.float().mean() < 0.4  # ~0.25
        assert 0.3 < no_motif_mask.float().mean() < 0.7  # ~0.5
        # check some folding is with motifs and some without
        assert torch.any(fwd_mask & with_motif_mask)
        assert torch.any(inv_mask & with_motif_mask)
        assert torch.any(fwd_mask & no_motif_mask)
        assert torch.any(inv_mask & no_motif_mask)

        # examples with motif_mask reset should be re-centered
        no_motif_batch_com = batch_center_of_mass(
            batch[bp.trans_1][no_motif_mask],
            mask=batch[bp.res_mask][no_motif_mask],
        )
        assert torch.allclose(
            no_motif_batch_com,
            torch.zeros_like(no_motif_batch_com),
            atol=0.1,
        )

        # for training, full t=1 structure + sequence should be present.
        # ensure scaffold positions exist in t=1 values if motifs dropped. tolerate 1 pos per sample.
        assert (
            torch.isclose(
                batch[bp.trans_1][no_motif_mask],
                torch.zeros_like(batch[bp.trans_1][no_motif_mask]),
                atol=0.01,
            ).sum()
            < B
        )
        # another way to test the same thing: subsequent positions differ.
        # in a real protein no subsequent positions should be the same,
        # but since we have random values, tolerate 1 pos per sample
        assert (
            torch.isclose(
                batch[bp.trans_1][:, :-1, :],
                batch[bp.trans_1][:, 1:, :],
                atol=0.01,
            ).sum()
            < B
        )

    @pytest.mark.parametrize(
        "task",
        [DataTask.hallucination, DataTask.inpainting],
    )
    @pytest.mark.parametrize(
        "stochastic",
        [True, False],
    )
    def test_corrupt_batch_works(
        self,
        task,
        stochastic,
        mock_cfg_uninterpolated,
    ):
        mock_cfg_uninterpolated.data.task = task
        mock_cfg_uninterpolated.shared.stochastic = stochastic
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = create_pdb_batch(cfg=cfg, training=True)

        interpolant = Interpolant(cfg.interpolant)
        interpolant.corrupt_batch(batch, task)

    @pytest.mark.parametrize("task", [DataTask.hallucination, DataTask.inpainting])
    def test_corrupt_batch_basic_behavior(self, task, mock_cfg_uninterpolated):
        """
        Test that corrupt_batch adds expected keys with correct shapes
        """
        mock_cfg_uninterpolated.data.task = task
        cfg = mock_cfg_uninterpolated.interpolate()

        # Build interpolant
        interpolant = Interpolant(cfg.interpolant)
        batch = create_pdb_batch(cfg)
        noisy_batch = interpolant.corrupt_batch(batch, task)

        # Batch dimensions
        B, N = batch[bp.res_mask].shape

        # Expected noisy keys
        expected_keys = {
            nbp.cat_t,
            nbp.so3_t,
            nbp.r3_t,
            nbp.trans_t,
            nbp.rotmats_t,
            nbp.torsions_t,
            nbp.aatypes_t,
            nbp.trans_sc,
            nbp.aatypes_sc,
        }
        for key in expected_keys:
            assert key.value in noisy_batch, f"Missing noisy batch key {key}"

        # Check shapes of noisy tensors
        # t values shape [B]
        assert noisy_batch[nbp.so3_t].shape == (B,)
        assert noisy_batch[nbp.r3_t].shape == (B,)
        assert noisy_batch[nbp.cat_t].shape == (B,)
        # translations and self-cond
        assert noisy_batch[nbp.trans_t].shape == (B, N, 3)
        assert noisy_batch[nbp.trans_sc].shape == (B, N, 3)
        # rotations
        assert noisy_batch[nbp.rotmats_t].shape == (B, N, 3, 3)
        # torsions
        assert noisy_batch[nbp.torsions_t].shape == (B, N, 7, 2)
        # amino acids and self-cond logits
        assert noisy_batch[nbp.aatypes_t].shape == (B, N)
        assert noisy_batch[nbp.aatypes_sc].shape == (B, N, interpolant.num_tokens)

        # Self-conditioned outputs should be zero
        assert torch.all(noisy_batch[nbp.trans_sc] == 0), "trans_sc should be zeroed"
        assert torch.all(
            noisy_batch[nbp.aatypes_sc] == 0
        ), "aatypes_sc should be zeroed"

        # Check that t values are within expected range [min_t, 1]
        min_t = cfg.interpolant.min_t
        t_vals = noisy_batch[nbp.r3_t]
        assert torch.all(t_vals >= min_t - 1e-6)
        assert torch.all(t_vals <= 1 + 1e-6)

        # Corrupted structure should differ from originals
        trans_diff = noisy_batch[nbp.trans_t] - batch[bp.trans_1]
        assert torch.any(trans_diff.abs() > 1e-6), "trans_t should be corrupted"
        rot_diff = noisy_batch[nbp.rotmats_t] - batch[bp.rotmats_1]
        assert torch.any(rot_diff.abs() > 1e-6), "rotmats_t should be corrupted"
        torsions_diff = noisy_batch[nbp.torsions_t] - batch[bp.torsions_1]
        assert torch.any(torsions_diff.abs() > 1e-6), "torsions_t should be corrupted"

    @pytest.mark.parametrize("stochastic", [False, True])
    @pytest.mark.parametrize("noise_type", ["gaussian", "harmonic"])
    def test_corrupt_translations(
        self, stochastic, noise_type, mock_cfg_uninterpolated
    ):
        mock_cfg_uninterpolated.shared.stochastic = stochastic
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = 5, 30
        chain_idx = torch.ones(B, N).long()
        res_mask = torch.ones(B, N).float()
        diffuse_mask = torch.ones(B, N).float()
        t = torch.rand(B)

        if noise_type == "gaussian":
            trans_1 = centered_gaussian(B, N, device=interpolant._device)
        elif noise_type == "harmonic":
            trans_1 = centered_harmonic(chain_idx=chain_idx, device=interpolant._device)
        else:
            raise ValueError(f"Unknown noise type {noise_type}")

        stochasticity_scale = torch.ones_like(t) if stochastic else torch.zeros_like(t)

        trans_fm = FlowMatcherTrans(cfg=interpolant.cfg.trans)
        trans_fm.set_device(interpolant._device)
        trans_t = trans_fm.corrupt(
            trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=stochasticity_scale,
        )

        assert not torch.allclose(trans_t, trans_1)

    @pytest.mark.parametrize("stochastic", [False, True])
    def test_corrupt_rotations(self, stochastic, mock_cfg_uninterpolated):
        mock_cfg_uninterpolated.shared.stochastic = stochastic
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = 4, 16
        chain_idx = torch.ones(B, N).long()
        res_mask = torch.ones(B, N).float()
        diffuse_mask = torch.ones(B, N).float()
        t = torch.rand(B)

        rotmats_1 = uniform_so3(B, N, device=interpolant._device)

        stochasticity_scale = torch.ones_like(t) if stochastic else torch.zeros_like(t)

        rots_fm = FlowMatcherRotations(cfg=interpolant.cfg.rots)
        rots_fm.set_device(interpolant._device)
        rotmats_t = rots_fm.corrupt(
            rotmats_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=stochasticity_scale,
        )

        assert not torch.allclose(rotmats_t, rotmats_1)

        # must remain valid rotations
        RRT = torch.matmul(rotmats_t, rotmats_t.transpose(-1, -2))
        I3 = torch.eye(3)
        assert torch.allclose(RRT, I3, atol=1e-3)
        assert torch.allclose(
            torch.det(rotmats_t), torch.ones_like(torch.det(rotmats_t)), atol=1e-3
        )

    @pytest.mark.parametrize("stochastic", [False, True])
    def test_corrupt_torsions(self, stochastic, mock_cfg_uninterpolated):
        mock_cfg_uninterpolated.shared.stochastic = stochastic
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = 4, 16
        res_mask = torch.ones(B, N).float()
        diffuse_mask = torch.ones(B, N).float()
        t = torch.rand(B)

        torsions_1 = torsions_noise(
            sigma=torch.ones(B, device=interpolant._device),
            num_samples=N,
            num_angles=7,
        )

        stochasticity_scale = torch.ones_like(t) if stochastic else torch.zeros_like(t)

        torsions_fm = FlowMatcherTorsions(cfg=interpolant.cfg.torsions)
        torsions_fm.set_device(interpolant._device)
        torsions_t = torsions_fm.corrupt(
            torsions_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=stochasticity_scale,
        )

        assert torsions_t.shape == (B, N, 7, 2)
        assert not torch.allclose(torsions_t, torsions_1)

    @pytest.mark.parametrize("stochastic", [False, True])
    @pytest.mark.parametrize(
        "interpolant_type",
        [
            InterpolantAATypesInterpolantTypeEnum.masking,
            InterpolantAATypesInterpolantTypeEnum.uniform,
        ],
    )
    @pytest.mark.parametrize("purity", [False, True])
    def test_corrupt_aatypes(
        self, stochastic, interpolant_type, purity, mock_cfg_uninterpolated
    ):
        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform and purity:
            pytest.skip()

        cfg = mock_cfg_uninterpolated
        cfg.shared.stochastic = stochastic
        cfg.interpolant.aatypes.interpolant_type = interpolant_type
        cfg.interpolant.aatypes.do_purity = purity
        cfg = cfg.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            assert interpolant.num_tokens == 20
        else:
            assert interpolant.num_tokens == 21

        B, N = 5, 30
        chain_idx = torch.ones(B, N).long()
        res_mask = torch.ones(B, N).float()
        diffuse_mask = torch.ones(B, N).float()
        t = torch.rand(B)

        # ground-truth sequence (cyclic so every aa appears, but no mask)
        aatype_1 = torch.arange(N).repeat(B, 1) % 20  # (B, N)

        stochasticity_scale = torch.ones_like(t) if stochastic else torch.zeros_like(t)

        out = interpolant.aatypes_fm.corrupt(
            aatype_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=stochasticity_scale,
        )

        assert not torch.equal(out, aatype_1)  # something corrupted
        assert out.le(interpolant.num_tokens).all()  # all valid aatypes

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            if stochastic:
                # for masking interpolant, when stochastic=True we expect new identities
                # outside the deterministic noise mechanism; i.e. out ⊈ {original aa, MASK}
                new_ids = out[out != aatype_1]
                assert new_ids.numel() > 0
            else:
                # for masking interpolant, when stochastic=False there should be **no new identities**
                # outside the deterministic noise mechanism; i.e. out ⊆ {original aa, MASK}
                new_ids = out[(out != aatype_1) & (out != MASK_TOKEN_INDEX)]
                assert new_ids.numel() == 0

    @pytest.mark.parametrize("task", [DataTask.hallucination, DataTask.inpainting])
    def test_corrupt_batch_multimer(self, task, mock_cfg_uninterpolated):
        """
        Test that corrupt_batch adds expected keys with correct shapes
        """
        mock_cfg_uninterpolated.data.task = task
        mock_cfg_uninterpolated.dataset.filter = DatasetFilterConfig.multimeric()
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        batch = create_pdb_batch(cfg)
        noisy_batch = interpolant.corrupt_batch(batch, task)

        # confirm multimers, 1-indexed
        assert noisy_batch[bp.chain_idx].min() == 1
        assert noisy_batch[bp.chain_idx].float().mean() > 1.1
        assert noisy_batch[bp.res_idx].min() == 1

        # chain breaks -> res_idx resets to 1
        chain_break_start_mask = torch.zeros_like(
            noisy_batch[bp.chain_idx], dtype=torch.bool
        )
        chain_break_start_mask[:, 1:] = (
            noisy_batch[bp.chain_idx][:, :-1] != noisy_batch[bp.chain_idx][:, 1:]
        )
        assert (noisy_batch[bp.res_idx][chain_break_start_mask] == 1).all()

        # diffuse masks as expected
        if task == DataTask.inpainting:
            assert noisy_batch[bp.motif_mask].float().mean() < 1.0
            assert noisy_batch[bp.diffuse_mask].float().mean() == 1.0
        elif task == DataTask.hallucination:
            assert noisy_batch[bp.diffuse_mask].float().mean() == 1.0

    def test_corrupt_batch_preserves_motif_sequences_in_inpainting(
        self, mock_cfg_uninterpolated
    ):
        """
        For inpainting, motif positions (diffuse_mask == 0) should preserve original amino acids.
        """
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        cfg = mock_cfg_uninterpolated.interpolate()

        # Create a mock batch, instead of PDB, because PDBs contain UNK and less predictable
        B = 5
        N = 10
        batch = next(
            iter(
                MockDataloader(
                    corrupt=False,
                    sample_lengths=[N] * B,
                    batch_size=B,
                )
            )
        )

        interpolant = Interpolant(cfg.interpolant)
        # Create a custom diffuse mask: first half scaffold (1), second half motif (0)
        diffuse_mask = torch.zeros((B, N), dtype=torch.int)
        diffuse_mask[:, : N // 2] = 1
        batch[bp.diffuse_mask] = diffuse_mask
        motif_mask = 1 - diffuse_mask
        batch[bp.motif_mask] = motif_mask

        noisy_batch = interpolant.corrupt_batch(batch, task=DataTask.inpainting)

        # define selection masks
        motif_sel = motif_mask.bool()
        scaffold_sel = ~motif_sel

        assert torch.equal(
            noisy_batch[nbp.aatypes_t][motif_sel], batch[bp.aatypes_1][motif_sel]
        ), "Motif amino acids should be preserved in inpainting"

        # aatypes fixed in motifs, some percent corrupted in scaffolds dep on `t`
        # We might expect ~0.24 to be corrupted on average:
        #    diffuse_mask.mean() == 0.5
        #    t.mean() == 0.5
        #    (aatypes_1 == UNK).mean() == 0.04
        # But we'll pick a conservative lower bound.
        assert (
            noisy_batch[nbp.aatypes_t][scaffold_sel]
            != batch[bp.aatypes_1][scaffold_sel]
        ).float().mean() > 0.05, (
            "Scaffold amino acids should be corrupted in inpainting"
        )

        # structure interpolates in scaffolds and motifs
        assert torch.all(
            noisy_batch[nbp.trans_t][scaffold_sel] != batch[bp.trans_1][scaffold_sel]
        ), "scaffold structure should change"
        assert torch.all(
            noisy_batch[nbp.trans_t][motif_sel] != batch[bp.trans_1][motif_sel]
        ), "motif structure should change"

    @pytest.mark.parametrize("time", [0, 1])
    def test_corruption_t_0_1(self, time, mock_cfg_uninterpolated):
        # no noise should be added at t=0 or t=1 for stochastic paths (boundary condition)
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg_uninterpolated.interpolant.trans.batch_ot = False
        mock_cfg_uninterpolated.interpolant.trans.batch_align = False
        mock_cfg_uninterpolated.dataset.noise_atom_positions_angstroms = 0.0
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = 5, 20
        chain_idx = torch.ones(B, N).long()
        res_mask = torch.ones(B, N).float()
        diffuse_mask = torch.ones(B, N).float()
        t = torch.ones((B,)) * time

        # at both t=0 and t=1, sigma_t should be 0
        assert torch.allclose(
            interpolant.trans_fm._compute_sigma_t(t, scale=1, min_sigma=0.0),
            torch.zeros((B,)).float(),
        )

        # pass scale 1 in all cases - sigma_t should still be 0
        stochasticity_scale = torch.ones_like(t)

        # Inspect corruptions

        aatypes_1 = torch.randint(0, 20, (B, N)).long()
        aatypes_t = interpolant.aatypes_fm.corrupt(
            aatypes_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=stochasticity_scale,
        )
        if time == 1:
            assert torch.equal(aatypes_1, aatypes_t)

        trans_1 = centered_gaussian(B, N, device=interpolant._device)
        trans_1 -= batch_center_of_mass(trans_1, mask=res_mask)[:, None]
        trans_fm = FlowMatcherTrans(cfg=interpolant.cfg.trans)
        trans_fm.set_device(interpolant._device)
        trans_t = trans_fm.corrupt(
            trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=stochasticity_scale,
        )
        if time == 1:
            assert torch.allclose(trans_1, trans_t, atol=1e-3)

        rotmats_1 = uniform_so3(B, N, device=interpolant._device)
        rotmats_t = interpolant.rots_fm.corrupt(
            rotmats_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=stochasticity_scale,
        )
        if time == 1:
            # geodesic may introduce some numerical gaps in log-exp map for geodesic,
            # so compare rotations rather than values directly.
            delta = torch.matmul(rotmats_t, rotmats_1.transpose(-1, -2))  # (...,3,3)
            ang, _, _ = angle_from_rotmat(delta)  # (...,)
            assert torch.quantile(ang, 0.99) <= 1e-2  # 99 % within 0.01 rad ~ 0.57°

    @pytest.mark.parametrize(
        "interpolant_type",
        [
            InterpolantAATypesInterpolantTypeEnum.masking,
            InterpolantAATypesInterpolantTypeEnum.uniform,
        ],
    )
    def test_corrupt_consistency_stochastic_off_vs_scale_zero(
        self, interpolant_type, mock_cfg_uninterpolated
    ):
        """
        When noise is disabled across batch, corruptions should be identical whether
        disabling via cfg.shared.stochastic=False or by passing stochasticity_scale=0.0.

        Compare translations, rotations, torsions, and aatypes given the same
        base distributions, masks, and time `t`.
        """
        device = torch.device("cpu")
        B, N = 4, 16

        # Build two configs: one with stochastic disabled, one enabled (and set scale to 0)
        cfg_off = mock_cfg_uninterpolated
        cfg_off.shared.stochastic = False
        cfg_off.interpolant.aatypes.interpolant_type = interpolant_type
        cfg_off = cfg_off.interpolate()

        cfg_on = mock_cfg_uninterpolated
        cfg_on.shared.stochastic = True
        cfg_on.interpolant.aatypes.interpolant_type = interpolant_type
        cfg_on = cfg_on.interpolate()

        # Instantiate interpolants
        interp_off = Interpolant(cfg_off.interpolant)
        interp_on = Interpolant(cfg_on.interpolant)

        # Build common inputs
        torch.manual_seed(0)
        chain_idx = torch.ones(B, N, device=device).long()
        res_mask = torch.ones(B, N, device=device).float()
        diffuse_mask = torch.ones(B, N, device=device).float()
        t = torch.rand(B, device=device)

        # Build a base batch once (use cfg_off); reuse it for both calls
        torch.manual_seed(1)
        batch = create_pdb_batch(cfg_off, training=True)
        # Ensure masks are fully diffused to avoid task-dependent fixing
        batch[bp.diffuse_mask] = torch.ones_like(batch[bp.diffuse_mask])
        # second copy for the scale-zero path
        batch_scale0 = _copy.deepcopy(batch)
        # provide stochastic scale zeros (B,)
        B = batch[bp.res_mask].shape[0]
        batch_scale0[bp.stochastic_scale] = torch.zeros(B, device=device)

        # Call interpolant.corrupt_batch directly and compare outputs
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()
        noisy_scale0 = interp_on.corrupt_batch(batch_scale0, task=cfg_on.data.task)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        noisy_off = interp_off.corrupt_batch(batch, task=cfg_off.data.task)

        assert torch.allclose(
            noisy_scale0[nbp.trans_t], noisy_off[nbp.trans_t], atol=1e-6
        )
        assert torch.allclose(
            noisy_scale0[nbp.rotmats_t], noisy_off[nbp.rotmats_t], atol=1e-6
        )
        assert torch.allclose(
            noisy_scale0[nbp.torsions_t], noisy_off[nbp.torsions_t], atol=1e-6
        )
        assert torch.equal(noisy_scale0[nbp.aatypes_t], noisy_off[nbp.aatypes_t])

    def test_corrupt_per_sample_stochastic_scale(self, mock_cfg_uninterpolated):
        """
        With per-sample stochasticity scales (B,), only samples with scale>0
        should receive stochastic noise.
        """
        device = torch.device("cpu")

        # Enable stochastic globally and per-domain with positive intensity
        cfg = mock_cfg_uninterpolated
        cfg.shared.stochastic = True
        cfg = cfg.interpolate()

        interpolant = Interpolant(cfg.interpolant)

        # Build a small batch (B>=2) and ensure fully diffused
        batch = create_pdb_batch(cfg, training=True)
        B, N = batch[bp.res_mask].shape
        assert B >= 2, "Test expects batch size >= 2"
        batch[bp.diffuse_mask] = torch.ones_like(batch[bp.diffuse_mask])

        # Two runs: (1) all zero scales; (2) mixed scales [0, 1]
        base_allzero = _copy.deepcopy(batch)
        base_mixed = _copy.deepcopy(batch)
        base_allzero[bp.stochastic_scale] = torch.zeros(B, device=device)
        mixed_scale = torch.zeros(B, device=device)
        mixed_scale[1] = 1.0
        base_mixed[bp.stochastic_scale] = mixed_scale

        # Compare per-domain with other domains disabled to avoid RNG stream differences
        def compare_domain(domain: str):
            cfg_d = _copy.deepcopy(cfg)
            cfg_d.interpolant.trans.corrupt = domain == "trans"
            cfg_d.interpolant.rots.corrupt = domain == "rots"
            cfg_d.interpolant.torsions.corrupt = domain == "torsions"
            cfg_d.interpolant.aatypes.corrupt = domain == "aatypes"
            interp_d = Interpolant(cfg_d.interpolant)

            torch_state = torch.random.get_rng_state()
            np_state = np.random.get_state()
            noisy_zero = interp_d.corrupt_batch(
                _copy.deepcopy(base_allzero), task=cfg_d.data.task
            )
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
            noisy_mix = interp_d.corrupt_batch(
                _copy.deepcopy(base_mixed), task=cfg_d.data.task
            )

            idx0, idx1 = 0, 1
            if domain == "trans":
                # sample 0 equal, sample 1 likely different
                assert torch.allclose(
                    noisy_zero[nbp.trans_t][idx0],
                    noisy_mix[nbp.trans_t][idx0],
                    atol=1e-6,
                )
                assert not torch.allclose(
                    noisy_zero[nbp.trans_t][idx1], noisy_mix[nbp.trans_t][idx1]
                )
            elif domain == "rots":
                assert torch.allclose(
                    noisy_zero[nbp.rotmats_t][idx0],
                    noisy_mix[nbp.rotmats_t][idx0],
                    atol=1e-6,
                )
                assert not torch.allclose(
                    noisy_zero[nbp.rotmats_t][idx1], noisy_mix[nbp.rotmats_t][idx1]
                )
            elif domain == "torsions":
                assert torch.allclose(
                    noisy_zero[nbp.torsions_t][idx0],
                    noisy_mix[nbp.torsions_t][idx0],
                    atol=1e-6,
                )
                assert not torch.allclose(
                    noisy_zero[nbp.torsions_t][idx1], noisy_mix[nbp.torsions_t][idx1]
                )
            elif domain == "aatypes":
                assert torch.equal(
                    noisy_zero[nbp.aatypes_t][idx0], noisy_mix[nbp.aatypes_t][idx0]
                )
                assert not torch.equal(
                    noisy_zero[nbp.aatypes_t][idx1], noisy_mix[nbp.aatypes_t][idx1]
                )

        for d in ["trans", "rots", "torsions", "aatypes"]:
            compare_domain(d)

    def test_stochasticity_scale_monotonic_noise(self, mock_cfg_uninterpolated):
        """Noise magnitude should be nondecreasing as stochasticity_scale increases across domains

        Metrics:
        - translations: mean L2 delta vs scale=0 output
        - rotations: mean geodesic angle vs scale=0 output
        - torsions: mean wrapped angle delta vs scale=0 output
        - aatypes: mean noise rates row sum from CTMC builder
        """
        cfg = mock_cfg_uninterpolated
        cfg.shared.stochastic = True
        cfg = cfg.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        device = interpolant._device

        B, N = 6, 48
        chain_idx = torch.ones(B, N, device=device).long()
        res_mask = torch.ones(B, N, device=device).float()
        diffuse_mask = torch.ones(B, N, device=device).float()
        t = torch.full((B,), 0.5, device=device)

        # base truths
        trans_1 = centered_gaussian(B, N, device=device)
        rotmats_1 = uniform_so3(B, N, device=device)
        torsions_1 = torsions_noise(
            sigma=torch.ones(B, device=device), num_samples=N, num_angles=7
        )

        # aatypes current state
        if interpolant.num_tokens == 21:
            aatypes_1 = torch.randint(low=0, high=21, size=(B, N), device=device)
        else:
            aatypes_1 = torch.randint(low=0, high=20, size=(B, N), device=device)

        # Save RNG to reproduce same base/intermediate draws across scales
        torch.manual_seed(0)
        np.random.seed(0)
        torch_state = torch.random.get_rng_state()
        np_state = np.random.get_state()

        # translations
        trans_fm = FlowMatcherTrans(cfg=interpolant.cfg.trans)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        trans_base = trans_fm.corrupt(
            trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            stochasticity_scale=torch.zeros_like(t),
        )
        trans_deltas = []
        for s in np.linspace(0.0, 2.0, 21)[1:]:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
            stochasticity_scale = trans_fm.effective_stochastic_scale(
                t=t,
                stochastic_scale=torch.full_like(t, float(s)),
            )
            trans_s = trans_fm.corrupt(
                trans_1,
                t=t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                stochasticity_scale=stochasticity_scale,
            )
            d = (trans_s - trans_base).norm(dim=-1).mean().item()
            trans_deltas.append(d)
        assert np.all(np.diff(trans_deltas) >= -1e-6)

        # rotations
        rots_fm = FlowMatcherRotations(cfg=interpolant.cfg.rots)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        rot_base = rots_fm.corrupt(
            rotmats_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.zeros_like(t),
        )
        rot_deltas = []
        for s in np.linspace(0.0, 2.0, 21)[1:]:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
            stochasticity_scale = rots_fm.effective_stochastic_scale(
                t=t,
                stochastic_scale=torch.full_like(t, float(s)),
            )
            rot_s = rots_fm.corrupt(
                rotmats_1,
                t=t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                stochasticity_scale=stochasticity_scale,
            )
            delta = torch.matmul(rot_s, rot_base.transpose(-1, -2))
            ang, _, _ = angle_from_rotmat(delta)
            rot_deltas.append(float(ang.mean().item()))
        assert np.all(np.diff(rot_deltas) >= -1e-6)

        # torsions
        tors_fm = FlowMatcherTorsions(cfg=interpolant.cfg.torsions)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        tors_base = tors_fm.corrupt(
            torsions_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.zeros_like(t),
        )
        angles_base = torch.atan2(tors_base[..., 0], tors_base[..., 1])
        tors_deltas = []
        for s in np.linspace(0.0, 2.0, 21)[1:]:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(np_state)
            stochasticity_scale = tors_fm.effective_stochastic_scale(
                t=t,
                stochastic_scale=torch.full_like(t, float(s)),
            )
            tors_s = tors_fm.corrupt(
                torsions_1,
                t=t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                stochasticity_scale=stochasticity_scale,
            )
            angles_s = torch.atan2(tors_s[..., 0], tors_s[..., 1])
            d_ang = (angles_s - angles_base + math.pi) % (2.0 * math.pi) - math.pi
            tors_deltas.append(float(d_ang.abs().mean().item()))
        assert np.all(np.diff(tors_deltas) >= -1e-6)

        # aatypes
        if interpolant.num_tokens == 21:
            aatypes_fm = FlowMatcherAATypesMasking(cfg=interpolant.cfg.aatypes)
        else:
            aatypes_fm = FlowMatcherAATypesUniform(cfg=interpolant.cfg.aatypes)
        aatypes_fm.set_device(device)
        torch.random.set_rng_state(torch_state)
        np.random.set_state(np_state)
        aatypes_base = aatypes_fm.corrupt(
            aatypes_1=aatypes_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            stochasticity_scale=torch.zeros_like(t),
        )
        change_scales = np.linspace(0.0, 2.0, 21)
        change_means = []
        for s in change_scales:
            stochasticity_scale = aatypes_fm.effective_stochastic_scale(
                t=t,
                stochastic_scale=torch.full_like(t, float(s)),
            )
            aatypes_s = aatypes_fm.corrupt(
                aatypes_1=aatypes_1,
                t=t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                stochasticity_scale=stochasticity_scale,
            )
            change_means.append(float(aatypes_s.ne(aatypes_base).sum().item()))

        # For aatypes (categorical jumps), single-sample paths can show small non-monotone fluctuations
        # as probabilities shift with scale. We therefore check a positive linear trend with sufficient R^2
        # instead of strict stepwise monotonicity.
        x = np.asarray(change_scales, dtype=float)
        y = np.asarray(change_means, dtype=float)
        A = np.vstack([x, np.ones_like(x)]).T
        slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
        y_hat = slope * x + intercept
        ss_res = np.sum((y - y_hat) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2) + 1e-12
        r2 = 1.0 - ss_res / ss_tot
        assert slope > 0.0 and r2 >= 0.8
