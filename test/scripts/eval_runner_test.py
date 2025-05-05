import os.path

import pytest

from cogeneration.config.base import Config
from cogeneration.dataset.datasets import LengthSamplingDataset
from cogeneration.dataset.test_utils import create_pdb_dataloader
from cogeneration.models.module import FlowModule
from cogeneration.scripts.predict import EvalRunner
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import InferenceTask


class TestEvalRunner:
    def test_mock_init(
        self, mock_cfg, mock_checkpoint, mock_folding_validation, tmp_path
    ):
        cfg, ckpt_path = mock_checkpoint(cfg=mock_cfg)
        _ = EvalRunner(cfg=cfg)

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
    def test_public_weights_inpainting(self, public_weights_path, tmp_path):
        cfg_uninterpolated = Config.public_multiflow()

        # specify task
        # cfg_uninterpolated.inference.task = InferenceTask.unconditional
        cfg_uninterpolated.inference.task = InferenceTask.inpainting
        # stochastic paths (NOTE public multiflow not trained to support)
        cfg_uninterpolated.shared.stochastic = False
        # set up predict_dir to tmp_path
        cfg_uninterpolated.inference.predict_dir = str(tmp_path / "inference")
        # control number of timesteps. use 1 to debug folding validation
        cfg_uninterpolated.inference.interpolant.sampling.num_timesteps = 500
        # limit eval length
        cfg_uninterpolated.dataset.max_eval_length = 70
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
            dataloader = LengthSamplingDataset(cfg.inference.samples)
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

    def test_sampling_and_compute_metrics(
        self, mock_cfg, mock_checkpoint, mock_folding_validation, tmp_path
    ):
        # This is a long-running end-to-end test that performs sampling and computes metrics.

        # create a dummy checkpoint
        cfg, ckpt_path = mock_checkpoint(cfg=mock_cfg)

        # only sample one sample
        # TODO(test) - support multiple samples
        #   We need to mock folding validation for all samples in pred dataloader.
        n_samples_expected = 1
        cfg.inference.samples.samples_per_length = 1
        cfg.inference.samples.length_subset = [23]

        # Run sampling
        sampler = EvalRunner(cfg=cfg)

        # we implicitly test that inference config takes priority over checkpoint
        assert (
            len(sampler.dataloader) == n_samples_expected
        ), f"Expected only one sample in dataloader, got {len(sampler.dataloader)}"
        pred_batch = next(iter(sampler.dataloader))
        mock_folding_validation(
            batch=pred_batch,
            cfg=cfg,
            n_inverse_folds=cfg.folding.seq_per_sample,  # prediction
        )

        # run sampling
        sampler.run_sampling()

        # ensure we get top samples
        assert os.path.exists(cfg.inference.predict_dir), f"Predict dir not found"

        # compute metrics (using patched methods)
        top_samples_df, top_metrics_df = sampler.compute_metrics()

        # write to inspect
        print(f"samples + metrics written to: {tmp_path}")
        top_samples_df.to_csv(tmp_path / "top_samples_df.csv")
        top_metrics_df.to_csv(tmp_path / "top_metrics_df.csv")

        # check top samples
        assert (
            len(top_samples_df) == n_samples_expected
        ), f"Expected {n_samples_expected} samples"
        for col in [
            # pdb structure paths
            MetricName.sample_pdb_path,
            MetricName.folded_pdb_path,
            # subset of metrics
            MetricName.bb_rmsd_folded,
            MetricName.inverse_folding_sequence_recovery_mean,
            MetricName.helix_percent,
        ]:
            assert col in top_samples_df.columns, f"Expected {col} in top_samples_df"

        # check summary metrics
        assert len(top_metrics_df) == 1, f"Expected summary metrics to be single row"
        assert (
            "Total Samples" in top_metrics_df.columns
        ), f"Expected 'Total Samples' in top_metrics_df"
        assert (
            top_metrics_df["Total Samples"].iloc[0] == n_samples_expected
        ), f"Expected {n_samples_expected} samples"
