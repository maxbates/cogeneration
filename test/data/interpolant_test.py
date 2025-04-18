import numpy as np
import pytest
import torch

from cogeneration.config.base import Config, DataTaskEnum, InferenceTaskEnum
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.batch_props import PredBatchProps as pbp
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import centered_gaussian
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass
from cogeneration.dataset.test_utils import create_pdb_batch


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
        mock_cfg_uninterpolated.data.task = DataTaskEnum.inpainting
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

    @pytest.mark.parametrize(
        "task", [DataTaskEnum.hallucination, DataTaskEnum.inpainting]
    )
    def test_corrupt_batch_basic_behavior(self, task, mock_cfg_uninterpolated):
        """
        Test that corrupt_batch adds expected keys with correct shapes
        """
        mock_cfg_uninterpolated.data.task = task
        cfg = mock_cfg_uninterpolated.interpolate()

        torch.manual_seed(0)

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

        # Check that t values are within expected range [min_t, 1-min_t]
        min_t = cfg.interpolant.min_t
        t_vals = noisy_batch[nbp.r3_t]
        assert torch.all(t_vals >= min_t - 1e-6)
        assert torch.all(t_vals <= (1 - min_t) + 1e-6)

        # Corrupted translations and rotations should differ from originals
        trans_diff = noisy_batch[nbp.trans_t] - batch[bp.trans_1]
        assert torch.any(trans_diff.abs() > 1e-6), "trans_t should be corrupted"
        rot_diff = noisy_batch[nbp.rotmats_t] - batch[bp.rotmats_1]
        assert torch.any(rot_diff.abs() > 1e-6), "rotmats_t should be corrupted"

    def test_corrupt_batch_preserves_motif_sequences_in_inpainting(
        self, mock_cfg_uninterpolated
    ):
        """
        For inpainting, motif positions (diffuse_mask == 0) should preserve original amino acids.
        """
        mock_cfg_uninterpolated.data.task = DataTaskEnum.inpainting
        cfg = mock_cfg_uninterpolated.interpolate()

        torch.manual_seed(0)

        interpolant = Interpolant(cfg.interpolant)
        batch = create_pdb_batch(cfg)
        # Create a custom diffuse mask: first half scaffold (1), second half motif (0)
        B, N = batch[bp.res_mask].shape
        diffuse_mask = torch.zeros((B, N), dtype=torch.int)
        diffuse_mask[:, : N // 2] = 1
        batch[bp.diffuse_mask] = diffuse_mask

        noisy_batch = interpolant.corrupt_batch(batch, DataTaskEnum.inpainting)

        motif_mask = diffuse_mask == 0
        scaffold_mask = diffuse_mask == 1
        assert torch.equal(
            noisy_batch[nbp.aatypes_t][motif_mask], batch[bp.aatypes_1][motif_mask]
        ), "Motif amino acids should be preserved in inpainting"
        assert torch.any(
            noisy_batch[nbp.aatypes_t][scaffold_mask]
            != batch[bp.aatypes_1][scaffold_mask]
        ), "Scaffold amino acids should be corrupted in inpainting"


class TestInterpolantSample:
    """Test suite for Interpolant.sample()."""

    def _run_sample(self, cfg: Config, batch, task: InferenceTaskEnum):
        cfg.interpolant.sampling.num_timesteps = 2  # run quickly with few timesteps

        interpolant = Interpolant(cfg.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = batch[bp.res_mask].shape
        num_tokens = interpolant.num_tokens
        T = cfg.interpolant.sampling.num_timesteps

        # Dummy model
        class ModelStub:
            def __call__(self, batch):
                t = batch[nbp.trans_t]
                r = batch[nbp.rotmats_t]
                psi = torch.zeros((B, N, 2), dtype=torch.float32)
                aa = batch[nbp.aatypes_t].long()
                logits = torch.nn.functional.one_hot(aa, num_classes=num_tokens).float()
                return {
                    pbp.pred_trans: t,
                    pbp.pred_rotmats: r,
                    pbp.pred_psi: psi,
                    pbp.pred_aatypes: aa,
                    pbp.pred_logits: logits,
                }

        model = ModelStub()

        # set up kwargs for sample()
        kwargs = {}
        kwargs.update(
            diffuse_mask=batch[bp.diffuse_mask],
            chain_idx=batch[bp.chain_idx],
            res_idx=batch[bp.res_idx],
        )
        if task == InferenceTaskEnum.unconditional:
            pass
        elif task == InferenceTaskEnum.inpainting:
            kwargs.update(
                trans_1=batch[bp.trans_1],
                rotmats_1=batch[bp.rotmats_1],
                psis_1=batch[bp.torsion_angles_sin_cos_1][..., 2, :],
                aatypes_1=batch[bp.aatypes_1],
            )
        elif task == InferenceTaskEnum.forward_folding:
            kwargs.update(
                aatypes_1=batch[bp.aatypes_1],
            )
        elif task == InferenceTaskEnum.inverse_folding:
            kwargs.update(
                trans_1=batch[bp.trans_1],
                rotmats_1=batch[bp.rotmats_1],
                psis_1=batch[bp.torsion_angles_sin_cos_1][..., 2, :],
            )

        prot_traj, model_traj = interpolant.sample(
            num_batch=B, num_res=N, model=model, task=task, **kwargs
        )

        assert prot_traj.structure.shape == (B, T + 1, N, 37, 3)

        assert prot_traj.amino_acids.shape == (B, T + 1, N)
        assert prot_traj.amino_acids.dtype == torch.long
        assert torch.all(prot_traj.amino_acids >= 0) and torch.all(
            prot_traj.amino_acids < num_tokens
        )

        assert model_traj.logits.shape == (B, T, N, num_tokens)

        return prot_traj, model_traj

    def test_sample_unconditional(
        self, mock_cfg_uninterpolated, mock_pred_unconditional_dataloader
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.unconditional
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = next(iter(mock_pred_unconditional_dataloader))
        self._run_sample(cfg=cfg, batch=batch, task=InferenceTaskEnum.unconditional)

    @pytest.mark.parametrize(
        "task",
        [
            InferenceTaskEnum.inpainting,
            InferenceTaskEnum.forward_folding,
            InferenceTaskEnum.inverse_folding,
        ],
    )
    def test_sample_conditional(
        self, task, mock_cfg_uninterpolated, mock_pred_conditional_dataloader
    ):
        mock_cfg_uninterpolated.inference.task = task
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = next(iter(mock_pred_conditional_dataloader))
        self._run_sample(cfg=cfg, batch=batch, task=task)

    def test_inpainting_preserves_motif_sequence_in_sampling(
        self, mock_cfg_uninterpolated, mock_pred_conditional_dataloader
    ):
        """
        Inpainting sampling should preserve amino acid sequence at motif positions
        """
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.inpainting
        mock_cfg_uninterpolated.interpolant.sampling.num_timesteps = 3
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = next(iter(mock_pred_conditional_dataloader))
        _, model_traj = self._run_sample(
            cfg=cfg, batch=batch, task=InferenceTaskEnum.inpainting
        )

        final_aa = model_traj.amino_acids[:, -1]
        orig_aa = batch[bp.aatypes_1]
        motif_mask = batch[bp.diffuse_mask] == 0
        assert torch.equal(
            final_aa[motif_mask], orig_aa[motif_mask]
        ), "Motif amino acids should be preserved in inpainting sampling"
