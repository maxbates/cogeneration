import pytest
import torch

from cogeneration.config.base import Config
from cogeneration.data.interpolant import Interpolant
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import PredBatchProp as pbp
from cogeneration.type.task import InferenceTask


class TestInterpolantSample:
    """Test suite for Interpolant.sample()."""

    def _run_sample(self, cfg: Config, batch, task: InferenceTask):
        cfg.inference.interpolant.sampling.num_timesteps = (
            2  # run quickly with few timesteps
        )
        cfg.interpolant.sampling.num_timesteps = (
            4  # ensure don't use training interpolant
        )

        interpolant = Interpolant(cfg.inference.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = batch[bp.res_mask].shape
        num_tokens = interpolant.num_tokens
        T = cfg.inference.interpolant.sampling.num_timesteps

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
            motif_mask=batch.get(bp.motif_mask, None),
            chain_idx=batch[bp.chain_idx],
            res_idx=batch[bp.res_idx],
        )
        if task == InferenceTask.unconditional:
            pass
        elif task == InferenceTask.inpainting:
            kwargs.update(
                trans_1=batch[bp.trans_1],
                rotmats_1=batch[bp.rotmats_1],
                psis_1=batch[bp.torsion_angles_sin_cos_1][..., 2, :],
                aatypes_1=batch[bp.aatypes_1],
            )
        elif task == InferenceTask.forward_folding:
            kwargs.update(
                aatypes_1=batch[bp.aatypes_1],
            )
        elif task == InferenceTask.inverse_folding:
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

    @pytest.mark.parametrize(
        "task",
        [
            InferenceTask.unconditional,
            InferenceTask.inpainting,
            InferenceTask.forward_folding,
            InferenceTask.inverse_folding,
        ],
    )
    @pytest.mark.parametrize("stochastic", [True, False])
    def test_sample(
        self,
        task,
        stochastic,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_pred_inpainting_dataloader,
        mock_pred_conditional_dataloader,
    ):
        mock_cfg_uninterpolated.inference.task = task
        mock_cfg_uninterpolated.shared.stochastic = stochastic
        cfg = mock_cfg_uninterpolated.interpolate()

        if task == InferenceTask.unconditional:
            dataloader = mock_pred_unconditional_dataloader
        elif task == InferenceTask.inpainting:
            dataloader = mock_pred_inpainting_dataloader
        elif task == InferenceTask.forward_folding:
            dataloader = mock_pred_conditional_dataloader
        elif task == InferenceTask.inverse_folding:
            dataloader = mock_pred_conditional_dataloader
        else:
            raise ValueError(f"Unknown task {task}")

        batch = next(iter(dataloader))
        self._run_sample(cfg=cfg, batch=batch, task=task)

    def test_inpainting_preserves_motif_sequence_in_sampling(
        self, mock_cfg_uninterpolated, mock_pred_inpainting_dataloader
    ):
        """
        Inpainting sampling should preserve amino acid sequence at motif positions
        """
        mock_cfg_uninterpolated.inference.task = InferenceTask.inpainting
        mock_cfg_uninterpolated.interpolant.sampling.num_timesteps = 3
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = next(iter(mock_pred_inpainting_dataloader))
        _, model_traj = self._run_sample(
            cfg=cfg, batch=batch, task=InferenceTask.inpainting
        )

        final_aa = model_traj.amino_acids[:, -1]
        orig_aa = batch[bp.aatypes_1]
        motif_sel = batch[bp.motif_mask].bool()
        assert torch.equal(
            final_aa[motif_sel], orig_aa[motif_sel]
        ), "Motif amino acids should be preserved in inpainting sampling"
