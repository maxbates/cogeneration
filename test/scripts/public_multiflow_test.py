import os.path

import pytest

from cogeneration.config.base import Config, InferenceSamplesConfig
from cogeneration.dataset.datasets import EvalDatasetConstructor
from cogeneration.models.module import FlowModule
from cogeneration.scripts.predict import EvalRunner
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

    # This is a slow test, because it actually samples with real model, many timesteps, and animates.
    # Can run manually.
    @pytest.mark.skip
    def test_public_weights_sampling(self, public_weights_path, tmp_path):
        cfg = Config.public_multiflow()

        # specify task (note public multiflow not trained to support inpainting)
        # cfg.inference.task = InferenceTask.unconditional
        cfg.inference.task = InferenceTask.inpainting
        # stochastic paths (NOTE public multiflow not trained to support, but can force)
        cfg.shared.stochastic = True
        cfg.inference.interpolant.trans.stochastic_noise_intensity = 0.75
        cfg.inference.interpolant.rots.stochastic_noise_intensity = 0.5
        cfg.inference.interpolant.aatypes.stochastic_noise_intensity = 0.25
        # FK Steering
        cfg.inference.interpolant.steering.num_particles = 4
        # set up predict_dir to tmp_path
        cfg.inference.predict_dir = str(tmp_path / "inference")
        # control number of timesteps. e.g. use 1 to debug folding validation / plotting
        cfg.inference.interpolant.sampling.num_timesteps = 200
        # number of samples + eval lengths etc.
        cfg.inference.samples.samples_per_length = 1
        cfg.inference.samples.num_batch = 1
        cfg.inference.samples.multimer_fraction = 0.0
        cfg.inference.samples.length_subset = [120]
        # skip designability? requires folding each ProteinMPNN sequence
        cfg.inference.also_fold_pmpnn_seq = False
        # write trajectories to inspect
        cfg.inference.write_sample_trajectories = True
        cfg.inference.write_animations: bool = True

        cfg = cfg.interpolate()

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

        eval_constructor = EvalDatasetConstructor(
            cfg=cfg.inference.samples,
            # Multiflow not trained to support inpainting but sort of can
            task=cfg.inference.task,
            dataset_cfg=cfg.dataset,
            use_test=False,
        )
        dataloader = eval_constructor.create_dataloader(
            batch_size=1,
            shuffle=True,
        )
        batch = next(iter(dataloader))

        # sample
        top_sample_metrics = module.predict_step(batch, batch_idx=0)

        print("results:")
        print(cfg.inference.predict_dir)
        print(top_sample_metrics.to_csv(index=False))
