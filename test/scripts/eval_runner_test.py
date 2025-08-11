import os.path

import pytest

from cogeneration.scripts.predict import EvalRunner
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import InferenceTask


class TestEvalRunner:
    def test_mock_init(
        self, mock_cfg, mock_checkpoint, mock_folding_validation, tmp_path
    ):
        cfg, ckpt_path = mock_checkpoint(cfg=mock_cfg)
        _ = EvalRunner(cfg=cfg)

    @pytest.mark.parametrize(
        "task", [InferenceTask.unconditional, InferenceTask.inpainting]
    )
    def test_sampling_and_compute_metrics(
        self,
        mock_cfg_uninterpolated,
        mock_checkpoint,
        mock_folding_validation,
        tmp_path,
        task,
    ):
        # This is a end-to-end test that performs sampling and computes metrics
        mock_cfg_uninterpolated.dataset.task = task.to_data_task(task)
        mock_cfg_uninterpolated.inference.task = task

        # only sample one sample
        # TODO(test) - support multiple samples
        #   We need to mock folding validation for all samples in pred dataloader.
        mock_cfg_uninterpolated.inference.samples.samples_per_length = 1
        mock_cfg_uninterpolated.inference.samples.length_subset = [36]

        cfg = mock_cfg_uninterpolated.interpolate()

        # create a dummy checkpoint
        cfg, ckpt_path = mock_checkpoint(cfg=cfg)

        # set up eval runner
        sampler = EvalRunner(cfg=cfg)

        # TODO(inpainting) - support better inpainting dataset / dataloader cfg
        #   currently not respecting cfg.inference, just using dataset eval == validation
        # TODO(inpainting) - handle stochasticity when sampling from dataloader
        #   e.g. scaffold lengths change, motif positions change, etc.
        # HACK - truncate dataset to single item for inpainting
        if task == InferenceTask.inpainting:
            sampler.dataloader.dataset.csv = sampler.dataloader.dataset.csv.head(1)

        # for unconditional, we implicitly test that inference config takes priority over checkpoint for LengthSamplingDataset
        assert (
            len(sampler.dataloader) == 1
        ), f"Expected only one sample in dataloader, got {len(sampler.dataloader)}"
        pred_batch = next(iter(sampler.dataloader))
        mock_folding_validation(
            batch=pred_batch,
            cfg=cfg,
            n_inverse_folds=cfg.folding.protein_mpnn.seq_per_sample,  # prediction
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
        assert len(top_samples_df) == 1, f"Expected 1 top sample"
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
        assert top_metrics_df["Total Samples"].iloc[0] == 1, f"Expected 1 sample"
