import os.path

import pytest
import torch

from cogeneration.config.base import Config, InferenceSamplesConfig
from cogeneration.dataset.datasets import LengthSamplingDataset
from cogeneration.dataset.test_utils import create_pdb_dataloader
from cogeneration.models.module import FlowModule
from cogeneration.scripts.predict import EvalRunner
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import InferenceTask


class TestEvalRunner:
    def test_public_multiflow_init(self):
        # use public multiflow config
        cfg = Config.public_multiflow().interpolate()
        _ = EvalRunner(cfg=cfg)

    def test_can_load_public_weights_with_default_config(self, public_weights_path):
        # use public multiflow config
        cfg = Config.public_multiflow().interpolate()

        # merge configs, which creates merged checkpoint
        merged_cfg, merged_ckpt_path = cfg.merge_checkpoint_cfg(
            ckpt_path=str(public_weights_path / "last.ckpt"),
        )
        assert merged_ckpt_path != str(public_weights_path), f"Expected new ckpt path"
        assert os.path.exists(
            merged_ckpt_path
        ), f"Merged ckpt not found at {merged_ckpt_path}"

        # inspect merged_cfg
        assert isinstance(merged_cfg, Config), f"Should get Config instance"

        # ensure can load new checkpoint
        FlowModule.load_from_checkpoint(
            checkpoint_path=merged_ckpt_path,
            cfg=merged_cfg,
        )

    # This is a slow test, because it actually samples with real model + many timesteps; can run manually.
    @pytest.mark.skip
    def test_public_weights_sampling(self, public_weights_path, tmp_path):
        cfg_uninterpolated = Config.public_multiflow()

        # specify task (note public multiflow not trained to support inpainting)
        # cfg_uninterpolated.inference.task = InferenceTask.unconditional
        cfg_uninterpolated.inference.task = InferenceTask.inpainting
        # stochastic paths (NOTE public multiflow not trained to support, but can force)
        cfg_uninterpolated.shared.stochastic = False
        # set up predict_dir to tmp_path
        cfg_uninterpolated.inference.predict_dir = str(tmp_path / "inference")
        # control number of timesteps. e.g. use 1 to debug folding validation / plotting
        cfg_uninterpolated.inference.interpolant.sampling.num_timesteps = 200
        # limit eval length
        cfg_uninterpolated.dataset.max_eval_length = 120
        # skip designability? requires folding each ProteinMPNN sequence
        cfg_uninterpolated.inference.also_fold_pmpnn_seq = False
        # write trajectories to inspect
        cfg_uninterpolated.inference.write_sample_trajectories = True

        cfg = cfg_uninterpolated.interpolate()

        # merge configs, which creates merged checkpoint
        merged_cfg, merged_ckpt_path = cfg.merge_checkpoint_cfg(
            ckpt_path=str(public_weights_path / "last.ckpt"),
        )
        assert merged_ckpt_path != str(public_weights_path), f"Expected new ckpt path"

        module = FlowModule.load_from_checkpoint(
            checkpoint_path=merged_ckpt_path,
            cfg=merged_cfg,
        )
        # usually handled by eval runner
        module.folding_validator.set_device_id(0)
        module.eval()

        # create inference batch
        if cfg.inference.task == InferenceTask.unconditional:
            dataset = LengthSamplingDataset(
                InferenceSamplesConfig(
                    samples_per_length=1,
                    num_batch=1,
                    length_subset=[cfg.dataset.max_eval_length],
                    multimer_fraction=0.0,
                )
            )
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        else:
            dataloader = create_pdb_dataloader(
                cfg=cfg,
                task=InferenceTask.to_data_task(cfg.inference.task),
                training=False,
                eval_batch_size=1,
            )
        batch = next(iter(dataloader))

        # sample
        top_sample_metrics = module.predict_step(batch, batch_idx=0)

        print(cfg.inference.predict_dir)
        print(top_sample_metrics.to_csv(index=False))
