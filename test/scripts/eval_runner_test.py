import os.path

from cogeneration.config.base import Config, InferenceTaskEnum
from cogeneration.data.enum import MetricName
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

    def test_sampling_and_compute_metrics(
        self, mock_cfg, mock_checkpoint, mock_folding_validation, tmp_path
    ):
        # create a dummy checkpoint
        ckpt_cfg_path, ckpt_path = mock_checkpoint(cfg=mock_cfg, path=tmp_path)

        # update config with the checkpoint
        assert mock_cfg.inference.task == InferenceTaskEnum.unconditional
        mock_cfg.inference.unconditional_ckpt_path = str(ckpt_path)

        # Run sampling

        # only sample one sample
        # TODO - support multiple samples
        #   We need to mock folding validation for all samples in pred dataloader.
        mock_cfg.inference.samples.samples_per_length = 1
        mock_cfg.inference.samples.length_subset = [23]

        n_samples_expected = 1

        sampler = EvalRunner(cfg=mock_cfg)

        # we implicitly test that inference config takes priority over checkpoint
        assert (
            len(sampler.dataloader) == n_samples_expected
        ), f"Expected only one sample in dataloader, got {len(sampler.dataloader)}"
        pred_batch = next(iter(sampler.dataloader))
        mock_folding_validation(
            batch=pred_batch,
            cfg=mock_cfg,
            n_inverse_folds=8,  # prediction
        )

        # run sampling
        sampler.run_sampling()

        # ensure we get top samples
        assert os.path.exists(mock_cfg.inference.predict_dir), f"Predict dir not found"

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
            MetricName.bb_rmsd,
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
