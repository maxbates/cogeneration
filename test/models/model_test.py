import torch
from config.base import Config, ModelConfig
from hydra.utils import instantiate
from omegaconf import OmegaConf

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
