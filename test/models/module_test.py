import pandas as pd

from cogeneration.config.base import InferenceTaskEnum
from cogeneration.models.module import FlowModule, TrainingLosses


class TestFlowModule:
    def test_init(self, mock_cfg):
        module = FlowModule(mock_cfg)
        assert module is not None
        assert module.model is not None

    def test_model_step(self, pdb_noisy_batch, mock_cfg):
        module = FlowModule(mock_cfg)
        losses = module.model_step(pdb_noisy_batch)
        assert isinstance(losses, TrainingLosses)

    def test_training_step(self, pdb_noisy_batch, mock_cfg):
        module = FlowModule(mock_cfg)
        loss = module.training_step(pdb_noisy_batch)
        assert loss is not None
        assert hasattr(loss, "backward"), "Loss object does not have backward() method"

    def test_validation_step(self, pdb_noisy_batch, mock_cfg):
        # TODO - mock folding / inverse folding
        module = FlowModule(mock_cfg)

        batch_metrics_df = module.validation_step(batch=pdb_noisy_batch, batch_idx=42)
        assert batch_metrics_df is not None
        # check for some specific data tracked in metrics
        expected_columns = ["bb_rmsd"]
        assert len(set(expected_columns).intersection(batch_metrics_df.columns)) == len(
            expected_columns
        ), f"Expected columns {expected_columns} not found in {batch_metrics_df.columns}"

    def test_predict_step_default_outputs(
        self, mock_cfg, mock_pred_unconditional_dataloader
    ):
        # TODO - mock folding / inverse folding
        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_unconditional_dataloader))
        predictions = module.predict_step(batch, 0)

        # ensure runs (not None), correct output format
        assert predictions is not None and isinstance(predictions, dict)

        # check output DF
        first_sample_id = list(predictions.keys())[0]
        df = pd.read_csv(predictions[first_sample_id])
        assert (
            "sample_id" in df.columns
            and "length" in df.columns
            and "seq_codesign" in df.columns
        )

    def test_predict_step_torsions_option(
        self, mock_cfg, mock_pred_unconditional_dataloader
    ):
        # flip the default, whether we predict torsion angles or not
        mock_cfg.model.predict_psi_torsions = not mock_cfg.model.predict_psi_torsions

        # TODO - mock folding / inverse folding
        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_unconditional_dataloader))
        predictions = module.predict_step(batch, 0)

        # ensure runs (not None), correct output format
        assert predictions is not None and isinstance(predictions, dict)

    def test_predict_step_unconditional_works(
        self, mock_cfg, mock_pred_unconditional_dataloader
    ):
        mock_cfg.inference.task = InferenceTaskEnum.unconditional

        # TODO - mock folding / inverse folding
        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_unconditional_dataloader))
        predictions = module.predict_step(batch, 0)
        assert predictions is not None and isinstance(predictions, dict)

    def test_predict_step_forward_folding_works(
        self, mock_cfg, mock_pred_conditional_dataloader
    ):
        # modify config for forward folding
        mock_cfg.inference.task = InferenceTaskEnum.forward_folding
        mock_cfg.inference.interpolant.aatypes.noise = 0.0
        mock_cfg.inference.interpolant.aatypes.do_purity = False

        # TODO - mock folding / inverse folding
        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_conditional_dataloader))
        predictions = module.predict_step(batch, 0)
        assert predictions is not None and isinstance(predictions, dict)

    def test_predict_step_inverse_folding_works(
        self, mock_cfg, mock_pred_conditional_dataloader
    ):
        mock_cfg.inference.task = InferenceTaskEnum.inverse_folding

        # TODO - mock folding / inverse folding
        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_conditional_dataloader))
        predictions = module.predict_step(batch, 0)
        assert predictions is not None and isinstance(predictions, dict)
