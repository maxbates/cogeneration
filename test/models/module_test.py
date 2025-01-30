import pandas as pd

from cogeneration.config.base import InferenceTaskEnum
from cogeneration.models.module import FlowModule


class TestFlowModule:
    def test_init(self, mock_cfg):
        module = FlowModule(mock_cfg)
        assert module is not None
        assert module.model is not None

    def test_training_step(self, pdb_noisy_batch, mock_cfg):
        module = FlowModule(mock_cfg)
        loss = module.training_step(pdb_noisy_batch)
        assert loss is not None and hasattr(loss, "backward")

    def test_predict_step_default_outputs(
        self, mock_cfg, mock_pred_unconditional_dataloader
    ):
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

    def test_predict_step_unconditional_works(
        self, mock_cfg, mock_pred_unconditional_dataloader
    ):
        mock_cfg.inference.task = InferenceTaskEnum.unconditional
        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_unconditional_dataloader))
        predictions = module.predict_step(batch, 0)
        assert predictions is not None and isinstance(predictions, dict)

    def test_predict_step_forward_folding_works(
        self, mock_cfg, mock_pred_conditional_dataloader
    ):
        mock_cfg.inference.task = InferenceTaskEnum.forward_folding

        # modify config for forward folding
        mock_cfg.interpolant.aatypes.noise = 0.0
        mock_cfg.interpolant.aatypes.do_purity = False

        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_conditional_dataloader))
        predictions = module.predict_step(batch, 0)
        assert predictions is not None and isinstance(predictions, dict)

    def test_predict_step_inverse_folding_works(
        self, mock_cfg, mock_pred_conditional_dataloader
    ):
        mock_cfg.inference.task = InferenceTaskEnum.inverse_folding

        module = FlowModule(mock_cfg)

        batch = next(iter(mock_pred_conditional_dataloader))
        predictions = module.predict_step(batch, 0)
        assert predictions is not None and isinstance(predictions, dict)
