import pytest
import torch

from cogeneration.config.base import AttentionType, ModelSequencePredictionEnum
from cogeneration.dataset.test_utils import MockDataloader
from cogeneration.models.model import FlowModel


class TestFlowModel:
    def test_init(self, mock_cfg):
        model = FlowModel(mock_cfg.model)
        assert model is not None

    def test_forward_pdb_dataloader(self, mock_cfg, pdb_noisy_batch):
        model = FlowModel(mock_cfg.model)
        output = model(pdb_noisy_batch)
        assert output is not None

    @pytest.mark.parametrize(
        "batch_size, sample_lengths",
        [
            (1, [10]),  # single sample
            (1, [10, 12]),  # mismatched lengths allowed for batch_size=1
            (2, [10, 10]),  # batches must be same length
            (2, [12, 12, 8, 8]),  # two batches of 2 samples each
        ],
    )
    def test_forward_mock_dataloader(self, mock_cfg, batch_size, sample_lengths):
        model = FlowModel(mock_cfg.model)

        mock_dataloader = MockDataloader(
            corrupt=True,
            batch_size=batch_size,
            sample_lengths=sample_lengths,
        )

        for batch in mock_dataloader:
            assert batch is not None
            output = model(batch)
            assert output is not None

    @pytest.mark.parametrize("attn_type", list(AttentionType))
    def test_forward_attention_types(
        self, mock_cfg_uninterpolated, attn_type, pdb_noisy_batch
    ):
        mock_cfg_uninterpolated.model.trunk.attn_type = attn_type
        cfg = mock_cfg_uninterpolated.interpolate()

        model = FlowModel(cfg.model)
        output = model(pdb_noisy_batch)
        assert output is not None

    # require 10s completion time
    @pytest.mark.timeout(10)
    @pytest.mark.skip  # TODO(perf) get model compilation working
    def test_model_torch_compiles(self, mock_cfg, pdb_noisy_batch):
        model = FlowModel(mock_cfg.model)
        compiled_model = torch.compile(model)
        output = compiled_model(pdb_noisy_batch)
        assert output is not None
