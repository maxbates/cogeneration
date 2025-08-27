import pytest
import torch

from cogeneration.config.base import (
    Config,
    DatasetFilterConfig,
    InterpolantAATypesInterpolantTypeEnum,
)
from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.interpolant import BatchTrueFeatures, Interpolant
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

    def test_motif_potentials_points_toward_motifs(
        self, mock_cfg_uninterpolated, mock_pred_inpainting_dataloader
    ):
        """The motif guidance vector field should point toward the true motif positions/orientations."""
        mock_cfg_uninterpolated.inference.task = InferenceTask.inpainting
        cfg = mock_cfg_uninterpolated.interpolate()

        # get motif-containing batch and corrupt (trans_t, rotmats_t, etc.)
        batch = next(iter(mock_pred_inpainting_dataloader))
        interp_corrupt = Interpolant(cfg.interpolant)
        interp_corrupt.set_device(torch.device("cpu"))
        noisy = interp_corrupt.corrupt_batch(batch, DataTask.inpainting)

        # true features at t=1 (used by guidance); logits auto-onehot from aatypes
        true_feats = BatchTrueFeatures.from_optional(
            res_mask=batch[bp.res_mask],
            trans=batch[bp.trans_1],
            rotmats=batch[bp.rotmats_1],
            torsions=batch[bp.torsions_1],
            aatypes=batch[bp.aatypes_1],
            num_tokens=interp_corrupt.num_tokens,
        )

        # use current noisy state as the "prediction", so the guidance direction = (true - current)
        pred = {
            pbp.pred_trans: noisy[nbp.trans_t],
            pbp.pred_rotmats: noisy[nbp.rotmats_t],
        }

        # compute motif potentials at the current scalar t (any scalar in (0,1) is fine)
        t_scalar = noisy[nbp.r3_t].mean()  # scalar tensor
        motif_sel = batch[bp.motif_mask].to(dtype=torch.bool)
        assert motif_sel.any(), "Test batch should include a non-empty motif mask"

        interp_infer = Interpolant(cfg.inference.interpolant)
        interp_infer.set_device(torch.device("cpu"))
        pf = interp_infer.motif_potentials(
            t=t_scalar,
            noisy_batch=noisy,
            pred=pred,
            true_feats=true_feats,
            motif_mask=motif_sel,
        )
        trans_vf, rot_vf = pf.trans, pf.rotmats

        # --- translations: check <VF, (true - current)> > 0 on motif residues ---
        dir_trans = batch[bp.trans_1] - noisy[nbp.trans_t]  # (B, N, 3)
        dots_trans = (trans_vf * dir_trans).sum(dim=-1)  # (B, N)
        assert torch.all(
            dots_trans[motif_sel] > 0
        ), "Translation VF should point toward motif positions"

        # --- rotations: check <VF, log_{R_t}(R_true)> > 0 in T_{R_t}SO(3) on motif residues ---
        dir_rot = so3_utils.calc_rot_vf(
            mat_t=noisy[nbp.rotmats_t], mat_1=batch[bp.rotmats_1]
        )  # (B, N, 3)
        dots_rot = (rot_vf * dir_rot).sum(dim=-1)  # (B, N)
        assert torch.all(
            dots_rot[motif_sel] > 0
        ), "Rotation VF should point toward motif orientations"

    def test_motif_guidance_single_step_moves_toward_motifs(
        self, mock_cfg_uninterpolated, mock_pred_inpainting_dataloader
    ):
        """1 deterministic step should reduce motif translation RMSD and SO(3) geodesic error."""
        mock_cfg_uninterpolated.shared.stochastic = False
        mock_cfg_uninterpolated.inference.task = InferenceTask.inpainting
        # don't take too big of steps of may overshoot and confuse test results
        mock_cfg_uninterpolated.interpolant.sampling.num_timesteps = 10
        cfg = mock_cfg_uninterpolated.interpolate()

        batch = next(iter(mock_pred_inpainting_dataloader))
        # have the mock model predict the true structure to isolate guidance/euler behavior
        sample_traj, model_traj, _ = self._run_sample(
            cfg=cfg,
            batch=batch,
            task=InferenceTask.inpainting,
            model_predicts_true=True,
            model_corruption=0.0,
        )

        motif_sel = batch[bp.motif_mask].to(dtype=torch.bool)
        assert motif_sel.any(), "Test batch should include a non-empty motif mask"

        # step 0 = initial state, step 1 = after first Euler step
        trans_true = batch[bp.trans_1]
        rot_true = batch[bp.rotmats_1]

        trans_0 = sample_traj[0].trans
        trans_1 = sample_traj[1].trans

        # translation RMSD on motif residues should decrease
        d0 = ((trans_true - trans_0) ** 2).sum(dim=-1).sqrt()[motif_sel].mean()
        d1 = ((trans_true - trans_1) ** 2).sum(dim=-1).sqrt()[motif_sel].mean()
        assert (
            d1 < d0
        ), f"Motif translation error should decrease after 1 step (before: {float(d0):.4f}, after: {float(d1):.4f})"

        # SO(3) geodesic error on motif residues should decrease (use tangent norm as the geodesic)
        rot_0 = sample_traj[0].rotmats
        rot_1 = sample_traj[1].rotmats
        ang0 = (
            so3_utils.calc_rot_vf(mat_t=rot_0, mat_1=rot_true)
            .norm(dim=-1)[motif_sel]
            .mean()
        )
        ang1 = (
            so3_utils.calc_rot_vf(mat_t=rot_1, mat_1=rot_true)
            .norm(dim=-1)[motif_sel]
            .mean()
        )
        assert (
            ang1 < ang0
        ), f"Motif rotational error should decrease after 1 step (before: {float(ang0):.4f}, after: {float(ang1):.4f})"

    @pytest.mark.parametrize("method", ["uniform", "masking", "purity"])
    def test_aatypes_euler_step_guidance_increases_target_fraction(
        self, method, mock_cfg_uninterpolated: Config
    ):
        torch.manual_seed(0)

        # Configure interpolant for deterministic aatype updates and desired method
        mock_cfg_uninterpolated.shared.stochastic = False
        cfg = mock_cfg_uninterpolated.interpolate()
        cfg.interpolant.aatypes.stochastic = False
        cfg.interpolant.aatypes.temp = 1.0
        cfg.interpolant.aatypes.noise = (
            0.5  # moderate to allow movement without overwhelming effect
        )

        if method == "uniform":
            cfg.interpolant.aatypes.interpolant_type = (
                InterpolantAATypesInterpolantTypeEnum.uniform
            )
            cfg.interpolant.aatypes.purity_selection = False
            num_states = 20
        elif method == "masking":
            cfg.interpolant.aatypes.interpolant_type = (
                InterpolantAATypesInterpolantTypeEnum.masking
            )
            cfg.interpolant.aatypes.purity_selection = False
            num_states = 21
        elif method == "purity":
            cfg.interpolant.aatypes.interpolant_type = (
                InterpolantAATypesInterpolantTypeEnum.masking
            )
            cfg.interpolant.aatypes.purity_selection = True
            num_states = 21
        else:
            raise AssertionError("Unknown method")

        interp = Interpolant(cfg.interpolant)
        interp.set_device(torch.device("cpu"))

        B, N = 3, 64
        steps = 5
        d_t = 1.0 / (steps + 1)

        # Fixed model logits (zeros); guidance potential will be added to these
        logits_1 = torch.zeros(B, N, num_states)

        # Choose target token (non-mask)
        target_token = 7

        # Initialize aatypes depending on method
        if method == "uniform":
            aatypes_t0 = torch.randint(low=0, high=20, size=(B, N))
        else:
            aatypes_t0 = torch.full((B, N), fill_value=MASK_TOKEN_INDEX)

        def run_with_potential(potential_scale: float):
            aatypes_t = aatypes_t0.clone()
            potential = torch.zeros_like(logits_1)
            potential[:, :, target_token] = potential_scale

            for i in range(steps):
                t = float(i + 1) / float(steps + 1)
                t_tensor = torch.tensor(t)
                d_t_tensor = torch.tensor(d_t)
                aatypes_t = interp._aatypes_euler_step(
                    d_t=d_t_tensor,
                    t=t_tensor,
                    logits_1=logits_1,
                    aatypes_t=aatypes_t,
                    stochasticity_scale=0.0,
                    potential=potential,
                )

            return aatypes_t

        # Baseline (no potential)
        aatypes_baseline = run_with_potential(potential_scale=0.0)
        # Low- and high-strength guidance
        aatypes_low = run_with_potential(potential_scale=1.0)
        aatypes_high = run_with_potential(potential_scale=4.0)

        # Compute fraction of residues equal to target token (ignoring masks for masking/purity)
        def target_fraction(aatypes: torch.Tensor) -> float:
            if num_states == 21:
                non_mask = aatypes != MASK_TOKEN_INDEX
                if non_mask.sum() == 0:
                    return 0.0
                return float(
                    ((aatypes == target_token) & non_mask).sum().item()
                    / non_mask.sum().item()
                )
            else:
                return float((aatypes == target_token).float().mean().item())

        frac_baseline = target_fraction(aatypes_baseline)
        frac_low = target_fraction(aatypes_low)
        frac_high = target_fraction(aatypes_high)

        # Guidance should steer toward the target
        assert (
            frac_low > frac_baseline + 1e-3
        ), f"Expected guidance to increase target fraction ({frac_low} > {frac_baseline})"
        # Stronger guidance should have stronger effect
        assert (
            frac_high > frac_low + 1e-3
        ), f"Expected stronger guidance to have larger effect ({frac_high} > {frac_low})"
