import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cogeneration.config.base import Config, ModelConfig, ModelNodeFeaturesConfig


class TestConfig:
    def test_init(self):
        cfg = Config()
        assert cfg is not None

        # can access nested properties
        assert cfg.model is not None
        assert cfg.interpolant.aatypes is not None

    def test_instantiate(self):
        cfg = Config()
        assert not isinstance(
            cfg.model.node_features.c_s, int
        )  # str before interpolation
        static_cfg = instantiate(Config, cfg)
        assert isinstance(
            static_cfg.model.node_features.c_s, int
        )  # int after interpolation

    def test_to_object(self):
        # interpolate the config, as we do for mock_cfg fixture
        cfg: Config = OmegaConf.to_object(OmegaConf.create(Config()))
        assert cfg is not None
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.model.node_features, ModelNodeFeaturesConfig)
        assert isinstance(cfg.model.node_features.c_s, int)


class TestMockConfig:
    def test_init(self, mock_cfg):
        assert mock_cfg is not None
        assert mock_cfg.model is not None

        # test interpolations
        # Note, assumes `OmegaConf.to_object()` and `OmegaConf.create()` called in fixture
        # Note that when printed, shows string template, but value correctly interpolated
        assert isinstance(mock_cfg.model.hyper_params.node_embed_size, int)
        assert isinstance(mock_cfg.model.edge_features.c_s, int)
        assert (
            mock_cfg.model.hyper_params.node_embed_size
            == mock_cfg.model.edge_features.c_s
        )
