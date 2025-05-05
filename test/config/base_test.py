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
        # Note that when printed, uninterpolated shows string template,
        # but value should be correctly interpolated by accessor.
        assert isinstance(mock_cfg.model.hyper_params.node_embed_size, int)
        assert isinstance(mock_cfg.model.edge_features.c_s, int)
        assert (
            mock_cfg.model.hyper_params.node_embed_size
            == mock_cfg.model.edge_features.c_s
        )


class TestBaseClassConfig:
    def test_asdict_and_flatdict(self):
        cfg = Config()
        # asdict returns nested dict
        d = cfg.asdict()
        assert isinstance(d, dict)
        # top-level keys present
        assert "shared" in d and "data" in d

        # flatdict returns flattened keys
        f = cfg.flatdict()
        assert isinstance(f, dict)
        # nested field key present, un-interpolated value is string
        raw_cs = f.get("model:node_features:c_s")
        assert isinstance(raw_cs, str)

        # interpolated flatdict yields int value for same key
        f_i = cfg.flatdict(interpolate=True)
        interp_cs = f_i.get("model:node_features:c_s")
        assert isinstance(interp_cs, int)

    def test_merge_dict_and_merge(self):
        cfg = Config()
        # default SharedConfig.local is True
        assert cfg.shared.local is True

        # merge_dict overrides via dict
        merged = cfg.merge_dict({"shared": {"local": False}}, interpolate=False)
        assert isinstance(merged, Config)
        assert merged.shared.local is False
        # original unchanged
        assert cfg.shared.local is True

        # merge overrides via another Config instance
        other = Config()
        other.shared.local = False
        merged2 = cfg.merge(other, interpolate=False)
        assert isinstance(merged2, Config)
        assert merged2.shared.local is False
        assert cfg.shared.local is True

    def test_load_from_file(self, tmp_path):
        # create a yaml config with overrides for nested fields
        cfg_file = tmp_path / "cfg.yaml"
        cfg_yaml = """
shared:
  local: False
data:
  task: inpainting
"""
        cfg_file.write_text(cfg_yaml)
        loaded, is_multiflow = Config.load_dict_from_file(str(cfg_file))
        assert isinstance(loaded, dict)
        # convert to Config
        loaded = Config().merge_dict(loaded, interpolate=True)
        # check overrides applied
        assert loaded.shared.local is False
        from cogeneration.type.task import DataTask

        assert loaded.data.task == DataTask.inpainting

    def test_merge_checkpoint_cfg(self, tmp_path):
        # set up an original config with a changed inference flag
        orig = Config()
        orig.inference.use_gpu = False
        # create a fake checkpoint directory and files
        ckpt_dir = tmp_path / "ckpt"
        ckpt_dir.mkdir()
        # write a minimal config.yaml overriding model.hyper_params.node_embed_size
        cfg_file = ckpt_dir / "config.yaml"
        cfg_yaml = """
model:
  hyper_params:
    node_embed_size: 123
shared:
  local: False
"""
        cfg_file.write_text(cfg_yaml)
        # create dummy ckpt file
        ckpt_path = ckpt_dir / "last.ckpt"
        ckpt_path.write_text("dummy")
        # merge checkpoint config
        merged, returned_ckpt = orig.merge_checkpoint_cfg(str(ckpt_path))
        assert isinstance(merged, Config)
        # model override applied
        assert merged.model.hyper_params.node_embed_size == 123
        # inference flag preserved
        assert merged.inference.use_gpu is False
        # returned path should be same as input
        assert returned_ckpt == str(ckpt_path)

    def test_merge_checkpoint_cfg_invalid_path(self):
        orig = Config()
        # path must end with .ckpt
        with pytest.raises(ValueError):
            orig.merge_checkpoint_cfg("not_a_ckpt.txt")
