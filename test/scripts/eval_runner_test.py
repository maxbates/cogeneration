import os.path
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cogeneration.config.base import Config
from cogeneration.models.module import FlowModule
from cogeneration.scripts.predict import EvalRunner


class TestEvalRunner:
    def test_init(self):
        # use actual config, so parameters match public multiflow we are loading from checkpoint
        cfg = Config().interpolate()

        _ = EvalRunner(cfg=cfg)

    def test_can_load_public_weights_with_default_config(self, public_weights_path):
        # use actual config, so parameters match public multiflow we are loading from checkpoint
        cfg = Config().interpolate()

        # create EvalRunner, merge configs, which creates merged checkpoint
        merged_cfg, merged_ckpt_path = EvalRunner.merge_checkpoint_cfg(
            cfg=cfg,
            ckpt_path=str(public_weights_path / "last.ckpt"),
        )
        print(f"Merged checkpoint path: {merged_ckpt_path}")
        assert merged_ckpt_path != str(public_weights_path), f"Expected new ckpt path"
        assert os.path.exists(
            merged_ckpt_path
        ), f"Merged ckpt not found at {merged_ckpt_path}"

        # inspect merged_cfg
        assert not isinstance(merged_cfg, Config), f"Should not get Config instance"

        # ensure can load new checkpoint
        FlowModule.load_from_checkpoint(
            checkpoint_path=merged_ckpt_path,
            cfg=merged_cfg,
        )
