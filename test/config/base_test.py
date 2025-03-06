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

    def test_interpolate(self):
        # our own helper function to yield an interpolated config

        raw_cfg = Config()
        # str before interpolation
        assert not isinstance(raw_cfg.model.node_features.c_s, int)

        cfg = raw_cfg.interpolate()
        # nested fields are dataclasses
        assert isinstance(cfg.model, ModelConfig)
        # int after interpolation
        assert isinstance(cfg.model.node_features.c_s, int)

    def test_hydra_instantiate(self):
        # hydra uses `instantiate` when main() is wrapped with hydra

        cfg = Config()
        # str before interpolation
        assert not isinstance(cfg.model.node_features.c_s, int)

        static_cfg = instantiate(Config, cfg)
        # nested fields are dataclasses
        assert isinstance(cfg.model, ModelConfig)
        # int after interpolation
        assert isinstance(static_cfg.model.node_features.c_s, int)

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
