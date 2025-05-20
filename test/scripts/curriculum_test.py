import os

import pytest

from cogeneration.config.base import Config
from cogeneration.config.curriculum import Curriculum, TrainingStep
from cogeneration.scripts.train import Experiment


class TestCurriculum:
    def test_check_model_equivalence(self, tmp_path):
        cfg1 = Config.test_uninterpolated(tmp_path=tmp_path / "step1")
        cfg1.shared.id = "step1"
        cfg2 = Config.test_uninterpolated(tmp_path=tmp_path / "step2")
        cfg2.model.sequence_ipa_net.use_init_embed = False
        cfg2.shared.id = "step2"

        # Ensure init raises an error if models are not equivalent
        assert cfg1.model != cfg2.model
        with pytest.raises(ValueError, match="must share the same model schema"):
            Curriculum(
                steps=[
                    TrainingStep(name="step1", cfg=cfg1.interpolate()),
                    TrainingStep(name="step2", cfg=cfg2.interpolate()),
                ]
            )

    def test_run(self, tmp_path, monkeypatch):
        # Patch Experiment to avoid heavy initialization and training
        monkeypatch.setattr(Experiment, "__init__", lambda self, cfg: None)
        monkeypatch.setattr(Experiment, "train", lambda self: None)

        # Create two dummy configs with unique IDs and checkpoint directories
        cfg1 = Config.test_uninterpolated(tmp_path=tmp_path / "step1")
        cfg1.shared.id = "step1"
        cfg1 = cfg1.interpolate()

        cfg2 = Config.test_uninterpolated(tmp_path=tmp_path / "step2")
        cfg2.shared.id = "step2"
        cfg2 = cfg2.interpolate()

        # Instantiate Curriculum, which should coordinate checkpoints
        curriculum = Curriculum(
            steps=[
                TrainingStep(name="step1", cfg=cfg1),
                TrainingStep(name="step2", cfg=cfg2),
            ]
        )

        # Verify that the second step warm_start_ckpt is set to the first step's last.ckpt
        expected_ckpt = os.path.join(cfg1.experiment.checkpointer.dirpath, "last.ckpt")
        assert curriculum.steps[1].cfg.experiment.warm_start_ckpt == expected_ckpt
        assert curriculum.steps[1].cfg.experiment.warm_start_cfg_override is True

        # Running the curriculum should invoke our patched Experiment without errors
        curriculum.run()

    def test_resume(self, tmp_path, monkeypatch, mock_checkpoint):
        cfg1 = Config.test_uninterpolated(tmp_path=tmp_path / "step1")
        cfg1.shared.id = "step1"
        cfg1 = cfg1.interpolate()

        cfg2 = Config.test_uninterpolated(tmp_path=tmp_path / "step2")
        cfg2.shared.id = "step2"
        cfg2 = cfg2.interpolate()

        # Create a dummy checkpoint for step1
        cfg1, ckpt1 = mock_checkpoint(cfg=cfg1)
        final_ckpt = os.path.join(cfg1.experiment.checkpointer.dirpath, "final.ckpt")
        assert ckpt1.replace("last.ckpt", "final.ckpt") == final_ckpt
        assert os.path.exists(final_ckpt)

        # Monkeypatch Experiment to avoid actually training
        called = []

        def fake_init(self, cfg):
            called.append(cfg)

        monkeypatch.setattr(Experiment, "__init__", fake_init)
        monkeypatch.setattr(Experiment, "train", lambda self: None)

        curriculum = Curriculum(
            steps=[
                TrainingStep(name="step1", cfg=cfg1),
                TrainingStep(name="step2", cfg=cfg2),
            ],
            # skip interpolation so maintain reference and can confirm ckpt configuration
            interpolate=False,
        )

        # Execute curriculum; step1 should be skipped, step2 runs
        curriculum.run()

        # Only step2 should have been initialized
        # can't compare directly because some paths will be modified
        assert len(called) == 1
        assert called[0].shared.id == cfg2.shared.id
        # Step2 warm_start should equal the checkpoint from step1
        assert cfg2.experiment.warm_start_ckpt == final_ckpt
