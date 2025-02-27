import os.path
from pathlib import Path

import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from cogeneration.config.base import Config
from cogeneration.scripts.predict import EvalRunner
from models.module import FlowModule


class TestEvalRunner:
    def test_init(self, mock_cfg):
        _ = EvalRunner(cfg=mock_cfg)

    def test_can_load_public_weights_with_default_config(self, mock_cfg):
        # check paths
        public_weights_path = (Path(__file__).parent / "../../multiflow_weights").resolve()
        print(f"Public weights path: {public_weights_path} (must be downloaded!)")
        assert os.path.exists(public_weights_path), f"Public weights not found at {public_weights_path}"
        assert os.path.exists(public_weights_path / "config.yaml"), f"Public config not found at {public_weights_path}"
        assert os.path.exists(public_weights_path / "last.ckpt"), f"Public ckpt not found at {public_weights_path}"

        # create EvalRunner, merge configs, which creates merged checkpoint
        merged_cfg, merged_ckpt_path = EvalRunner.merge_checkpoint_cfg(
            cfg=mock_cfg,
            ckpt_path=str(public_weights_path / "last.ckpt"),
        )
        print(f"Merged checkpoint path: {merged_ckpt_path}")
        assert merged_ckpt_path != str(public_weights_path), f"Expected new ckpt path"
        assert os.path.exists(merged_ckpt_path), f"Merged ckpt not found at {merged_ckpt_path}"

        # inspect merged_cfg
        assert not isinstance(merged_cfg, Config), f"Should not get Config instance"

        # ensure can load new checkpoint
        FlowModule.load_from_checkpoint(
            checkpoint_path=merged_ckpt_path,
            cfg=merged_cfg,
        )

