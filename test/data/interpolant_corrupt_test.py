import pytest
import torch

from cogeneration.config.base import (
    DatasetFilterConfig,
    InterpolantAATypesInterpolantTypeEnum,
)
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.data_transforms import make_one_hot
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import (
    centered_gaussian,
    centered_harmonic,
    uniform_so3,
)
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass
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

        result = interpolant._batch_ot(trans_0, trans_1, res_mask=res_mask, center=True)

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
            nbp.aatypes_t,
            nbp.trans_sc,
            nbp.aatypes_sc,
        }
        for key in expected_keys:
            assert key.value in noisy_batch, f"Missing noisy batch key {key}"

        # Check shapes of noisy tensors
        # t values shape [B, 1]
        assert noisy_batch[nbp.so3_t].shape == (B, 1)
        assert noisy_batch[nbp.r3_t].shape == (B, 1)
        assert noisy_batch[nbp.cat_t].shape == (B, 1)
        # translations and self-cond
        assert noisy_batch[nbp.trans_t].shape == (B, N, 3)
        assert noisy_batch[nbp.trans_sc].shape == (B, N, 3)
        # rotations
        assert noisy_batch[nbp.rotmats_t].shape == (B, N, 3, 3)
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

        # Corrupted translations and rotations should differ from originals
        trans_diff = noisy_batch[nbp.trans_t] - batch[bp.trans_1]
        assert torch.any(trans_diff.abs() > 1e-6), "trans_t should be corrupted"
        rot_diff = noisy_batch[nbp.rotmats_t] - batch[bp.rotmats_1]
        assert torch.any(rot_diff.abs() > 1e-6), "rotmats_t should be corrupted"

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
        t = torch.rand(B, 1)

        if noise_type == "gaussian":
            trans_1 = centered_gaussian(B, N, device=interpolant._device)
        elif noise_type == "harmonic":
            trans_1 = centered_harmonic(chain_idx=chain_idx, device=interpolant._device)
        else:
            raise ValueError(f"Unknown noise type {noise_type}")

        out = interpolant._corrupt_trans(
            trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
        )

        assert not torch.allclose(out, trans_1)
        # TODO - check values better

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
        t = torch.rand(B, 1)

        rotmats_1 = uniform_so3(B, N, device=interpolant._device)

        out = interpolant._corrupt_rotmats(
            rotmats_1, t=t, res_mask=res_mask, diffuse_mask=diffuse_mask
        )

        assert not torch.allclose(out, rotmats_1)

        # must remain valid rotations
        RRT = torch.matmul(out, out.transpose(-1, -2))
        I3 = torch.eye(3)
        assert torch.allclose(RRT, I3, atol=1e-3)
        assert torch.allclose(
            torch.det(out), torch.ones_like(torch.det(out)), atol=1e-3
        )

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
        t = torch.rand(B, 1)

        # ground-truth sequence (cyclic so every aa appears, but no mask)
        aatype_1 = torch.arange(N).repeat(B, 1) % 20  # (B,N)

        out = interpolant._corrupt_aatypes(
            aatype_1, t=t, res_mask=res_mask, diffuse_mask=diffuse_mask
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

    def test_aatype_jump_step(self, mock_cfg_uninterpolated):
        mock_cfg_uninterpolated.interpolant.aatypes.stochastic = True
        mock_cfg_uninterpolated.interpolant.aatypes.stochastic_noise_intensity = 1.0
        cfg = mock_cfg_uninterpolated.interpolate()

        B, N = 10, 50
        t = 0.5  # max sigma_t
        d_t = 0.1  # take a large timestep, more likely to jump
        aatypes_t = torch.randint(0, 20, (B, N)).long()
        logits_1 = make_one_hot(aatypes_t, num_classes=20)
        logits_1 *= 3  # confident for current aatypes
        logits_1 += torch.randn_like(logits_1) * 0.5  # noisy logits

        interpolant = Interpolant(cfg=cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        noisy_aatypes_t = interpolant._aatype_jump_step(
            d_t=d_t,
            t=t,
            logits_1=logits_1,
            aatypes_t=aatypes_t,
        )

        assert not torch.equal(noisy_aatypes_t, aatypes_t)
        assert (noisy_aatypes_t != aatypes_t).float().mean() >= 0.01

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
            noisy_batch[nbp.trans_t][scaffold_sel] != batch[bp.trans_1][scaffold_sel]
        ), "motif structure should change"

    @pytest.mark.parametrize("time", [0, 1])
    def test_corruption_t_0_1(self, time, mock_cfg_uninterpolated):
        # no noise should be added at t=0 or t=1 for stochastic paths (boundary condition)
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg_uninterpolated.interpolant.trans.batch_ot = False
        mock_cfg_uninterpolated.interpolant.trans.batch_align = False
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = 5, 20
        chain_idx = torch.ones(B, N).long()
        res_mask = torch.ones(B, N).float()
        diffuse_mask = torch.ones(B, N).float()
        t = torch.ones((B, 1)) * time

        assert torch.allclose(
            interpolant._compute_sigma_t(t, scale=1, min_sigma=0.0),
            torch.zeros((B, 1)).float(),
        )

        trans_1 = centered_gaussian(B, N, device=interpolant._device)
        trans_1 -= batch_center_of_mass(trans_1, mask=res_mask)[:, None]
        trans_t = interpolant._corrupt_trans(
            trans_1,
            t=t,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
        )
        if time == 1:
            assert torch.allclose(trans_1, trans_t, atol=1e-4)

        rotmats_1 = uniform_so3(B, N, device=interpolant._device)
        rotmats_t = interpolant._corrupt_rotmats(
            rotmats_1, t=t, res_mask=res_mask, diffuse_mask=diffuse_mask
        )
        if time == 1:
            assert torch.allclose(rotmats_1, rotmats_t, atol=1e-4)

        aatypes_1 = torch.randint(0, 20, (B, N)).long()
        aatypes_t = interpolant._corrupt_aatypes(
            aatypes_1, t=t, res_mask=res_mask, diffuse_mask=diffuse_mask
        )
        if time == 1:
            assert torch.equal(aatypes_1, aatypes_t)
