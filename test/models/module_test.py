import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from cogeneration.config.base import InterpolantTranslationsScheduleEnum
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.dataset.datasets import DatasetConstructor
from cogeneration.dataset.test_utils import (
    create_pdb_dataloader,
    create_pdb_noisy_batch,
)
from cogeneration.models.module import FlowModule, TrainingLosses
from cogeneration.type.batch import BatchProps as bp
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import DataTaskEnum, InferenceTaskEnum


class TestFlowModule:
    def test_init(self, mock_cfg):
        module = FlowModule(mock_cfg)
        assert module is not None
        assert module.model is not None

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

    def test_validation_step(self, mock_cfg, pdb_noisy_batch, mock_folding_validation):
        module = FlowModule(mock_cfg)

        mock_folding_validation(
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
        assert sample[MetricName.sequence] == input_seq
        # should not sample the original structure
        assert sample[MetricName.bb_rmsd_folded] > 0.1
        # random seq above should have low recovery
        assert sample[MetricName.inverse_folding_sequence_recovery_mean] < 0.2

    def test_predict_step_default_outputs(
        self, mock_cfg, mock_pred_unconditional_dataloader, mock_folding_validation
    ):
        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
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

    def test_predict_step_torsions_option(
        self, mock_cfg, mock_pred_unconditional_dataloader, mock_folding_validation
    ):
        # flip the default, whether we predict torsion angles or not
        mock_cfg.model.predict_psi_torsions = not mock_cfg.model.predict_psi_torsions

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just check that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_unconditional_works(
        self, mock_cfg, mock_pred_unconditional_dataloader, mock_folding_validation
    ):
        assert mock_cfg.inference.task == InferenceTaskEnum.unconditional

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_inpainting_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_inpainting_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.inpainting
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_inpainting_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_inpainting_but_actually_unconditional_works(
        self,
        mock_cfg_uninterpolated,
        mock_folding_validation,
    ):
        # modify config for inpainting, but actually unconditional
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.inpainting
        mock_cfg_uninterpolated.dataset.inpainting.unconditional_percent = 1.0
        mock_cfg = mock_cfg_uninterpolated.interpolate()
        assert mock_cfg.dataset.inpainting.unconditional_percent == 1.0

        # Explicitly create dataset, to pass mock_cfg
        dataloader = create_pdb_dataloader(
            cfg=mock_cfg,
            task=DataTaskEnum.inpainting,
            training=False,
            eval_batch_size=1,
        )
        batch = next(iter(dataloader))

        # ensure get an unconditional-like diffuse mask
        assert (batch[bp.diffuse_mask] == 1).all()

        module = FlowModule(mock_cfg)

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_forward_folding_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_conditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.forward_folding
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        assert mock_cfg.inference.interpolant.aatypes.noise == 0.0
        assert mock_cfg.inference.interpolant.aatypes.do_purity is False

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_conditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_inverse_folding_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_conditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.inverse_folding
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_conditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)

    def test_predict_step_unconditional_stochastic_works(
        self,
        mock_cfg_uninterpolated,
        mock_pred_unconditional_dataloader,
        mock_folding_validation,
    ):
        mock_cfg_uninterpolated.inference.task = InferenceTaskEnum.unconditional
        mock_cfg_uninterpolated.shared.stochastic = True
        mock_cfg = mock_cfg_uninterpolated.interpolate()

        assert mock_cfg.inference.interpolant.rots.stochastic
        assert mock_cfg.inference.interpolant.trans.stochastic
        assert (
            mock_cfg.inference.interpolant.trans.sample_schedule
            == InterpolantTranslationsScheduleEnum.vpsde
        )

        module = FlowModule(mock_cfg)
        batch = next(iter(mock_pred_unconditional_dataloader))

        mock_folding_validation(
            batch=batch,
            cfg=mock_cfg,
            n_inverse_folds=mock_cfg.folding.seq_per_sample,
        )

        # just test that it works
        _ = module.predict_step(batch, 0)
