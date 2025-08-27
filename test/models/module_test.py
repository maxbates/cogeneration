import numpy as np
import pandas as pd
import pytest
import torch
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.utils.data import DataLoader

from cogeneration.config.base import (
    Config,
    DatasetFilterConfig,
    InferenceSamplesConfig,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.dataset.datasets import EvalDatasetConstructor
from cogeneration.dataset.test_utils import create_pdb_noisy_batch
from cogeneration.models.loss_calculator import TrainingLosses
from cogeneration.models.module import FlowModule
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import DataTask, InferenceTask


class TestFlowModule:
    def test_init_cfg(self):
        cfg = Config().interpolate()
        module = FlowModule(cfg)
        assert module is not None
        assert module.model is not None
        print(ModelSummary(module, max_depth=3))

    def test_init_mock_cfg(self, mock_cfg):
        module = FlowModule(mock_cfg)
        assert module is not None
        assert module.model is not None
        print(ModelSummary(module, max_depth=3))

    def test_model_step(self, mock_cfg, pdb_noisy_batch):
        module = FlowModule(mock_cfg)
        losses = module.model_step(pdb_noisy_batch)
        assert isinstance(losses, TrainingLosses)

    def test_training_step(self, mock_cfg, pdb_noisy_batch):
        module = FlowModule(mock_cfg)
        loss = module.training_step(pdb_noisy_batch)
        assert loss is not None
        assert hasattr(loss, "backward"), "Loss object does not have backward() method"

    def test_training_step_stochastic(self, mock_cfg_uninterpolated):
        # test batch corruption with stochastic paths
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        assert mock_cfg.interpolant.trans.stochastic
        assert mock_cfg.interpolant.rots.stochastic

        # create batch using stochastic paths
        pdb_noisy_batch = create_pdb_noisy_batch(mock_cfg)

        module = FlowModule(mock_cfg)
        module.training_step(pdb_noisy_batch)

    def test_training_step_inpainting(self, mock_cfg_uninterpolated):
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        pdb_noisy_batch = create_pdb_noisy_batch(mock_cfg)
        assert (pdb_noisy_batch[bp.diffuse_mask] == 1).all()
        assert pdb_noisy_batch[bp.motif_mask].mean() > 0.1
        assert pdb_noisy_batch[bp.motif_mask].mean() < 0.9

        module = FlowModule(mock_cfg)
        module.training_step(pdb_noisy_batch)

    def test_training_step_inpainting_but_actually_unconditional(
        self, mock_cfg_uninterpolated
    ):
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        mock_cfg_uninterpolated.interpolant.inpainting_unconditional_prop = 1.0
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        pdb_noisy_batch = create_pdb_noisy_batch(mock_cfg)
        assert (pdb_noisy_batch[bp.diffuse_mask] == 1.0).all()

        module = FlowModule(mock_cfg)
        module.training_step(pdb_noisy_batch)

    @pytest.mark.slow
    def test_training_step_inpainting_multimers_stochastic(
        self, mock_cfg_uninterpolated
    ):
        mock_cfg_uninterpolated.data.task = DataTask.inpainting
        mock_cfg_uninterpolated.dataset.filter = DatasetFilterConfig.multimeric()
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        assert mock_cfg.interpolant.trans.stochastic
        assert mock_cfg.interpolant.rots.stochastic

        # create multimer batch using stochastic paths
        pdb_noisy_batch = create_pdb_noisy_batch(mock_cfg)
        assert (pdb_noisy_batch[bp.diffuse_mask] == 1).all()
        assert pdb_noisy_batch[bp.motif_mask].mean() > 0.1
        assert pdb_noisy_batch[bp.motif_mask].mean() < 0.9
        assert pdb_noisy_batch[bp.chain_idx].unique().shape[0] > 1

        module = FlowModule(mock_cfg)
        module.training_step(pdb_noisy_batch)

    @pytest.mark.parametrize(
        "task", [InferenceTask.unconditional, InferenceTask.inpainting]
    )
    def test_validation_step(
        self, mock_cfg_uninterpolated, pdb_noisy_batch, mock_folding_validation, task
    ):
        mock_cfg_uninterpolated.inference.task = task
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)

        validation_mock_values = mock_folding_validation(
            batch=pdb_noisy_batch,
            cfg=mock_cfg,
            n_inverse_folds=1,  # only one for validation
        )

        # Run validation step
        batch_metrics_df = module.validation_step(batch=pdb_noisy_batch, batch_idx=42)

        # check for subset of data tracked in metrics
        expected_columns = [
            MetricName.sample_pdb_path,  # input
            MetricName.sequence,  # inverse folded
            MetricName.folded_pdb_path,  # folded structure
            MetricName.bb_rmsd_folded,  # structure comparison
            MetricName.inverse_folding_bb_rmsd_min,  # summary metric
            MetricName.plddt_mean,  # af2 metric
            MetricName.coil_percent,  # mdtraj
            MetricName.ca_ca_deviation,  # ca_ca metric
            MetricName.aatype_histogram_dist,  # aatype batch metric
        ]
        assert len(set(expected_columns).intersection(batch_metrics_df.columns)) == len(
            expected_columns
        ), f"Expected columns {expected_columns} not found in {batch_metrics_df.columns}"

        sample = batch_metrics_df.iloc[0]
        input_seq = "".join(
            [
                restypes_with_x[x]
                for x in pdb_noisy_batch[bp.aatypes_1][0].cpu().detach().numpy()
            ]
        )

        print(validation_mock_values[0].seqs)
        print(validation_mock_values[0].true_aa)
        with open(validation_mock_values[0].mpnn_fasta_path, "r") as f:
            print(f.read())
        print(sample.to_csv(sep="\t"))

        assert sample[MetricName.sequence] == input_seq
        # should not sample the original structure
        assert sample[MetricName.bb_rmsd_folded] > 0.1
        # random seq above should have low recovery
        # i.e. for inpainting, should not include motifs
        assert (
            sample[MetricName.inverse_folding_sequence_recovery_mean] < 0.2
        ), f"Expected low sequence recovery for {task}"

    @pytest.mark.parametrize(
        "task", [InferenceTask.unconditional, InferenceTask.inpainting]
    )
    def test_predict_step_default_outputs(
        self,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_pred_inpainting_dataloader,
        mock_folding_validation,
        task,
    ):
        mock_cfg_uninterpolated.inference.task = task
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        if task == InferenceTask.unconditional:
            batch = next(iter(mock_pred_unconditional_dataloader))
        elif task == InferenceTask.inpainting:
            batch = next(iter(mock_pred_inpainting_dataloader))
        else:
            raise ValueError(f"Unknown task {task}")

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        predictions = module.predict_step(batch, 0)

        assert predictions is not None and isinstance(predictions, pd.DataFrame)

        # check for subset of data tracked in metrics
        expected_columns = [
            MetricName.sample_pdb_path,  # input
            MetricName.sequence,  # inverse folded
            MetricName.folded_pdb_path,  # folded structure
            MetricName.bb_rmsd_folded,  # structure comparison
            MetricName.inverse_folding_bb_rmsd_min,  # summary metric
            MetricName.plddt_mean,  # af2 metric
            MetricName.coil_percent,  # mdtraj
            MetricName.ca_ca_deviation,  # ca_ca metric
        ]
        assert len(set(expected_columns).intersection(predictions.columns)) == len(
            expected_columns
        ), f"Expected columns {expected_columns} not found in {predictions.columns}"
        # Don't expect some validation columns
        assert (
            MetricName.aatype_histogram_dist not in predictions.columns
        ), f"Unexpected metric {MetricName.aatype_histogram_dist}"

    def test_predict_step_unconditional_torsions_option(
        self,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.unconditional
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        # flip the default, whether we predict torsion angles or not
        mock_cfg.model.predict_psi_torsions = not mock_cfg.model.predict_psi_torsions
        mock_cfg.model.predict_all_torsions = mock_cfg.model.predict_psi_torsions

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just check that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_unconditional_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.unconditional
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_inpainting_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_inpainting_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.inpainting
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_inpainting_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_forward_folding_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_conditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.forward_folding
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_conditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_inverse_folding_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_conditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.inverse_folding
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_conditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_unconditional_stochastic_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.unconditional
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        assert mock_cfg.inference.interpolant.rots.stochastic
        assert mock_cfg.inference.interpolant.trans.stochastic

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_unconditional_multimer_stochastic_works(
        self,
        mock_cfg_uninterpolated,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTask.unconditional
        mock_cfg_uninterpolated.dataset.filter = DatasetFilterConfig.multimeric()
        mock_cfg_uninterpolated.inference.samples = InferenceSamplesConfig(
            multimer_fraction=1.0,
            samples_per_length=1,
            num_batch=1,
            length_subset=[200],
        )
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        assert mock_cfg.inference.interpolant.rots.stochastic
        assert mock_cfg.inference.interpolant.trans.stochastic

        module = FlowModule(mock_cfg)

        dataloader = EvalDatasetConstructor(
            cfg=mock_cfg.inference.samples,
            task=mock_cfg_uninterpolated.inference.task,
            dataset_cfg=mock_cfg_uninterpolated.dataset,
            use_test=False,
        ).create_dataloader()
        batch = next(iter(dataloader))
        assert batch[bp.chain_idx].unique().shape[0] > 1

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.protein_mpnn.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_state_dict_excludes_frozen_ESM_model(self, mock_cfg):
        assert mock_cfg.model.esm_combiner.enabled, "ESM must be enabled"
        module = FlowModule(mock_cfg)
        sd = module.state_dict()
        # expect combiner module
        assert any(
            "esm_combiner" in k for k in sd.keys()
        ), "ESM combiner not in state_dict"
        # but not ESM model
        esm_keys = [k for k in sd if "esm." in k]
        assert len(esm_keys) == 0, f"Found ESM keys in state_dict: {esm_keys}"

    def test_checkpoint_excludes_frozen_ESM_model(self, tmp_path, mock_checkpoint):
        cfg = Config.test_uninterpolated(tmp_path=tmp_path / "test").interpolate()
        assert cfg.model.esm_combiner.enabled, "ESM must be enabled"
        cfg, ckpt_path = mock_checkpoint(cfg=cfg)
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert any(
            "esm_combiner" in k for k in ckpt["state_dict"].keys()
        ), "ESM combiner not in checkpoint"
        assert not any(
            "esm." in k for k in ckpt["state_dict"].keys()
        ), "Found ESM keys in checkpoint"
