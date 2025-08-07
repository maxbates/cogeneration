import pytest
import torch

from cogeneration.config.base import Config, DatasetFilterConfig
from cogeneration.data.interpolant import Interpolant
from cogeneration.dataset.test_utils import create_pdb_batch, mock_feats
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp
from cogeneration.type.task import DataTask, InferenceTask


class TestInterpolantSample:
    """Test suite for Interpolant.sample()."""

    def _run_sample(
        self,
        cfg: Config,
        batch: NoisyFeatures,
        task: InferenceTask,
        model_predicts_true: bool = False,
        model_corruption: float = 0.0,
    ):
        # ensure don't use training interpolant, will mess up shapes
        cfg.interpolant.sampling.num_timesteps = 7

        interpolant = Interpolant(cfg.inference.interpolant)
        interpolant.set_device(torch.device("cpu"))

        B, N = batch[bp.res_mask].shape
        num_tokens = interpolant.num_tokens
        T = cfg.inference.interpolant.sampling.num_timesteps

        # Dummy model
        class ModelStub:
            def __call__(self, noisy_batch: NoisyFeatures):
                trans = (
                    batch[bp.trans_1]
                    if model_predicts_true
                    else noisy_batch[nbp.trans_t]
                )
                rotmats = (
                    batch[bp.rotmats_1]
                    if model_predicts_true
                    else noisy_batch[nbp.rotmats_t]
                )
                torsions = (
                    batch[bp.torsions_1]
                    if model_predicts_true
                    else noisy_batch[nbp.torsions_t]
                )
                aatypes = (
                    batch[bp.aatypes_1]
                    if model_predicts_true
                    else noisy_batch[nbp.aatypes_t]
                ).long()

                # apply low fidelity corruptions
                if model_corruption > 0.0:
                    trans = (
                        trans
                        + torch.randn_like(noisy_batch[nbp.trans_t])
                        * model_corruption
                        * 2
                    )
                    rotmats = (
                        rotmats
                        + torch.randn_like(noisy_batch[nbp.rotmats_t])
                        * model_corruption
                    )
                    torsions = (
                        torsions
                        + torch.randn_like(noisy_batch[nbp.torsions_t])
                        * model_corruption
                    )
                    aatypes = aatypes + torch.randint(
                        0, num_tokens, aatypes.shape, dtype=torch.long
                    )
                    aatypes = aatypes % num_tokens

                logits = torch.nn.functional.one_hot(
                    aatypes, num_classes=num_tokens
                ).float()

                return {
                    pbp.pred_trans: trans,
                    pbp.pred_rotmats: rotmats,
                    pbp.pred_torsions: torsions,
                    pbp.pred_aatypes: aatypes,
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
            hot_spots=batch.get(bp.hot_spots, None),
        )
        if task == InferenceTask.unconditional:
            pass
        elif task == InferenceTask.inpainting:
            kwargs.update(
                trans_1=batch[bp.trans_1],
                rotmats_1=batch[bp.rotmats_1],
                torsions_1=batch[bp.torsions_1],
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
                torsions_1=batch[bp.torsions_1],
            )

        sample_traj, model_traj, fk_traj = interpolant.sample(
            num_batch=B, num_res=N, model=model, task=task, **kwargs
        )

        assert sample_traj.structure.shape == (B, T + 1, N, 37, 3)

        assert sample_traj.amino_acids.shape == (B, T + 1, N)
        assert sample_traj.amino_acids.dtype == torch.long
        assert torch.all(sample_traj.amino_acids >= 0) and torch.all(
            sample_traj.amino_acids < num_tokens
        )

        assert model_traj.logits.shape == (B, T, N, num_tokens)

        return sample_traj, model_traj, fk_traj

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

    def test_sampling_conditioning(
        self,
        mock_cfg_uninterpolated,
    ):
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        mock_cfg_uninterpolated.inference.task = InferenceTask.inpainting
        mock_cfg_uninterpolated.dataset.filter = DatasetFilterConfig.multimeric()
        mock_cfg_uninterpolated.dataset.hotspots.hotspots_prob_disabled = 0.0
        mock_cfg_uninterpolated.dataset.contact_conditioning.conditioning_prob_disabled = (
            0.0
        )
        mock_cfg_uninterpolated.dataset.contact_conditioning.include_inter_chain_prob = (
            1.0
        )
        mock_cfg_uninterpolated.dataset.contact_conditioning.downsample_inter_chain_prob = (
            0.0
        )
        cfg = mock_cfg_uninterpolated.interpolate()

        interpolant = Interpolant(cfg.interpolant)
        batch = create_pdb_batch(cfg)
        batch = interpolant.corrupt_batch(batch, DataTask.inpainting)

        # confirm multimers, 1-indexed
        assert batch[bp.chain_idx].min() == 1
        assert batch[bp.chain_idx].float().mean() > 1.1

        # Note - if the input structure doesn't have any contacts, no conditioning will be defined.
        # We assume that the input structure has contacts.
        assert (
            batch[bp.contact_conditioning] is not None
        ), "Contact conditioning should be present"
        assert (
            batch[bp.contact_conditioning].sum() > 0
        ), "Contact conditioning should be non-zero"

        self._run_sample(cfg=cfg, batch=batch, task=InferenceTask.inpainting)

    @pytest.mark.parametrize("corruption", [0.0, 0.025])
    def test_inpainting_preserves_motif_in_sampling(
        self, corruption, mock_cfg_uninterpolated, mock_pred_inpainting_dataloader
    ):
        """
        Inpainting sampling should preserve motif positions
        """
        mock_cfg_uninterpolated.shared.stochastic = False
        mock_cfg_uninterpolated.inference.task = InferenceTask.inpainting
        mock_cfg_uninterpolated.interpolant.sampling.num_timesteps = 3
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = next(iter(mock_pred_inpainting_dataloader))
        sample_traj, model_traj, fk_traj = self._run_sample(
            cfg=cfg,
            batch=batch,
            task=InferenceTask.inpainting,
            model_predicts_true=True,  # model should predict true structure
            model_corruption=corruption,  # minimal drift from model, so mostly motif drift
        )

        motif_sel = batch[bp.motif_mask].bool()
        assert motif_sel.sum() > 0, "Test batch should have some motif residues"

        if corruption > 0.0:
            # sanity check - if model not predicting exactly:
            # don't expect model and sample trajectories to be equal in motifs.
            # if they are, likely accidentally overwriting the model states (or fixed motifs!)
            assert not torch.equal(
                model_traj[-1].aatypes[motif_sel],
                sample_traj[-1].aatypes[motif_sel],
            ), "Model and sample trajectories should not be equal in motifs"
            assert not torch.allclose(
                model_traj[-1].trans[motif_sel],
                sample_traj[-1].trans[motif_sel],
            ), "Model and sample trajectories should not be equal in motifs"

        # compare aatypes - should be fixed
        final_aa = sample_traj[-1].aatypes
        orig_aa = batch[bp.aatypes_1]
        assert torch.equal(
            final_aa[motif_sel], orig_aa[motif_sel]
        ), "Motif amino acids should be preserved in inpainting sampling"

        # compare structure - should match due to "guidance"
        # But account for COM changing.
        final_trans = sample_traj[-1].trans
        orig_trans = batch[bp.trans_1]
        assert torch.allclose(
            final_trans[motif_sel],
            orig_trans[motif_sel],
            atol=1e-6 if corruption == 0.0 else 0.25,
        ), "Motif translations should be preserved in inpainting sampling"

    @pytest.mark.parametrize(
        "task",
        [
            InferenceTask.unconditional,
            InferenceTask.inpainting,
        ],
    )
    def test_sample_fk_steering(
        self,
        task,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_pred_inpainting_dataloader,
    ):
        mock_cfg_uninterpolated.inference.interpolant.steering.num_particles = 3
        mock_cfg_uninterpolated.inference.interpolant.steering.resampling_interval = 1
        cfg = mock_cfg_uninterpolated.interpolate()

        if task == InferenceTask.unconditional:
            dataloader = mock_pred_unconditional_dataloader
        elif task == InferenceTask.inpainting:
            dataloader = mock_pred_inpainting_dataloader

        batch = next(iter(dataloader))
        num_batch, num_res = batch[bp.res_mask].shape

        sample_traj, model_traj, fk_traj = self._run_sample(
            cfg=cfg, batch=batch, task=task
        )

        # particles downselected to one per sample
        final_state = sample_traj.steps[-1]
        assert final_state.structure.shape == (num_batch, num_res, 37, 3)
        final_pred = model_traj.steps[-1]
        assert final_pred.aatypes.shape == (num_batch, num_res)

        # inspect fk_traj for FK steering properties
        assert (
            fk_traj is not None
        ), "FK trajectory should be returned when FK steering is enabled"

        # FK trajectory should contain intermediate particle states during steering
        T = cfg.inference.interpolant.sampling.num_timesteps
        num_particles = cfg.inference.interpolant.steering.num_particles
        resampling_interval = cfg.inference.interpolant.steering.resampling_interval

        assert resampling_interval == 1, "Resampling interval should be 1 for this test"
        assert (
            len(fk_traj.metrics) == T
        ), "FK trajectory should have one step per sampling timestep"

        step0 = fk_traj.metrics[0]
        assert step0.energy is not None, "Energy should be computed for step 0"
        assert step0.weights is not None, "Weights should be computed for step 0"
        assert (
            step0.effective_sample_size is not None
        ), "Effective sample size should be computed for step 0"
        assert step0.log_G is not None, "Log G should be computed for step 0"
        assert (
            step0.log_G_delta is not None
        ), "Log G delta should be computed for step 0"
