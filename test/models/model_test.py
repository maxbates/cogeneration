import pytest
import torch

from cogeneration.config.base import Config, ModelConfig, ModelSequencePredictionEnum
from cogeneration.models.model import FlowModel
from cogeneration.type.batch import BatchProp as bp


class TestFlowModel:
    def test_init(self, mock_cfg):
        model = FlowModel(mock_cfg.model)
        assert model is not None

    def test_forward_pdb_dataloader(self, mock_cfg, pdb_noisy_batch):
        model = FlowModel(mock_cfg.model)
        output = model(pdb_noisy_batch)
        assert output is not None

    @pytest.mark.parametrize(
        "mock_corrupted_dataloader",
        [
            {"batch_size": 1, "sample_lengths": [10]},
            {"batch_size": 1, "sample_lengths": [10, 12]},
            {
                "batch_size": 2,
                "sample_lengths": [10, 10],
            },  # batches must be same length
            {
                "batch_size": 2,
                "sample_lengths": [12, 12, 8, 8],
            },  # batches must be same length
        ],
        indirect=True,
    )
    def test_forward_mock_dataloader(self, mock_cfg, mock_corrupted_dataloader):
        model = FlowModel(mock_cfg.model)

        for batch in mock_corrupted_dataloader:
            assert batch is not None
            output = model(batch)
            assert output is not None

    def test_model_sequence_ipa_net(self, mock_cfg, pdb_noisy_batch):
        mock_cfg.model.sequence_pred_type = ModelSequencePredictionEnum.sequence_ipa_net
        model = FlowModel(mock_cfg.model)
        output = model(pdb_noisy_batch)
        assert output is not None

    # require 10s completion time
    @pytest.mark.timeout(10)
    def test_model_torch_compiles(self, mock_cfg, pdb_noisy_batch):
        model = FlowModel(mock_cfg.model)
        compiled_model = torch.compile(model)
        output = compiled_model(pdb_noisy_batch)
        assert output is not None
