import torch

from cogeneration.config.base import Config, ModelConfig, ModelSequencePredictionEnum
from cogeneration.models.model import FlowModel


class TestFlowModel:
    def test_init(self, mock_cfg):
        model = FlowModel(mock_cfg.model)
        assert model is not None

    def test_forward_pdb_dataloader(self, mock_cfg, pdb_noisy_batch):
        model = FlowModel(mock_cfg.model)
        output = model(pdb_noisy_batch)
        assert output is not None

    def test_forward_mock_dataloader(self, mock_cfg, pdb_noisy_batch, mock_dataloader):
        model = FlowModel(mock_cfg.model)
        input_feats = next(iter(mock_dataloader))
        output = model(input_feats)
        assert output is not None

    def test_model_torch_compiles(self, mock_cfg, pdb_noisy_batch):
        model = FlowModel(mock_cfg.model)
        compiled_model = torch.compile(model)
        output = compiled_model(pdb_noisy_batch)
        assert output is not None

    def test_model_sequence_ipa_net(self, mock_cfg, pdb_noisy_batch):
        mock_cfg.model.sequence_pred_type = ModelSequencePredictionEnum.sequence_ipa_net
        model = FlowModel(mock_cfg.model)
        output = model(pdb_noisy_batch)
        assert output is not None
