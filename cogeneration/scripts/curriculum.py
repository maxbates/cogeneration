import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from cogeneration.config.base import Config, ModelConfig
from cogeneration.scripts.train import Experiment
from cogeneration.util.log import rank_zero_logger

log = rank_zero_logger(__name__)


@dataclass
class TrainingStep:
    """
    A single step in the training curriculum, holding a unique name
    and a fully defined Config for that step.
    """

    name: str
    cfg: Config


@dataclass
class Curriculum:
    """
    training curriculum for sequential training of models.

    Note that if `interpolate` is set to `True`, the configs are interpolated copies
    and the original configs are not modified.
    """

    steps: List[TrainingStep] = field(default_factory=list)
    interpolate: bool = True

    def __post_init__(self):
        """
        Interpolate and validate and coordinate
        """
        if self.interpolate:
            self.interpolate_configs()

        self.validate_steps()
        self.coordinate_steps()

    def interpolate_configs(self):
        """
        Interpolates each step config to replace template strings etc.
        """
        for step in self.steps:
            try:
                step.cfg = step.cfg.interpolate()
            except Exception as e:
                raise ValueError(
                    f"Interpolation failed for step '{step.name}': {e}"
                ) from e

    def validate_steps(self):
        """
        Ensure the steps are unique where appropriate and compatible with each other
        Coordinates `warm_start` across steps.
        """
        # Check for duplicate step names
        names = [step.name for step in self.steps]
        dup_names = {n for n in names if names.count(n) > 1}
        if dup_names:
            raise ValueError(f"Duplicate step names in curriculum: {dup_names}")

        # Check for duplicate run identifiers (`cfg.shared.id`)
        run_ids = [step.cfg.shared.id for step in self.steps]
        dup_runs = {r for r in run_ids if run_ids.count(r) > 1}
        if dup_runs:
            raise ValueError(
                f"Duplicate run identifiers (`cfg.shared.id`) in curriculum: {dup_runs}. These must be explicitly defined and unique."
            )

        # Check for duplicate checkpoint directories
        ckpt_dirs = [step.cfg.experiment.checkpointer.dirpath for step in self.steps]
        dup_ckpt_dirs = {d for d in ckpt_dirs if ckpt_dirs.count(d) > 1}
        if dup_ckpt_dirs:
            raise ValueError(
                f"Duplicate checkpoint directories in curriculum: {dup_ckpt_dirs}. Either use default with `shared.id` or define explicitly."
            )

        # Ensure models are compatible
        # Assumes interpolated, otherwise template values may equal each other
        seen_model_cfg: Optional[ModelConfig] = None
        for step in self.steps:
            model_cfg = step.cfg.model
            if seen_model_cfg is None:
                seen_model_cfg = model_cfg
            elif model_cfg != seen_model_cfg:
                raise ValueError(
                    f"Model config mismatch in step '{step.name}'; all steps must share the same model schema"
                )

        # Force some assumptions
        for step in self.steps:
            assert (
                step.cfg.experiment.checkpointer.save_last is True
            ), "Curriculum training requires saving the last checkpoint"
            assert (
                step.cfg.experiment.save_final_ckpt is True
            ), "Curriculum training requires saving the final checkpoint for resuming"

    def coordinate_steps(self):
        """
        Tie model checkpoints between step cfgs
        """
        # `cfg.experiment.checkpointer.dirpath` is used to save the checkpoint and is name spaced by `cfg.shared.id`
        # `cfg.experiment.warm_start_ckpt` is used to load a checkpoint from a previous step
        last_ckpt = None
        for step in self.steps:
            # Set the warm start checkpoint for the next step, if one isn't defined
            if last_ckpt and step.cfg.experiment.warm_start_ckpt is None:
                step.cfg.experiment.warm_start_ckpt = last_ckpt
                step.cfg.experiment.warm_start_cfg_override = True

            # Save trained model checkpoint for next step warm start
            last_ckpt = os.path.join(
                step.cfg.experiment.checkpointer.dirpath, "last.ckpt"
            )

    def _find_resume(self) -> Tuple[int, Optional[str]]:
        """
        Scan each step's checkpoint directory for 'last.ckpt'.
        Returns the index of the last completed step and its checkpoint path.
        """
        # TODO - track how many steps / epochs already completed rather than restarting?

        last_idx = -1
        last_ckpt = None
        for idx, step in enumerate(self.steps):
            ckpt_dir = step.cfg.experiment.checkpointer.dirpath
            last_ckpt_path = os.path.join(ckpt_dir, "last.ckpt")
            final_ckpt_path = os.path.join(ckpt_dir, "final.ckpt")

            # check if this step completed, if so go to the next one
            if os.path.isfile(final_ckpt_path):
                last_idx = idx + 1
                last_ckpt = final_ckpt_path

            # check if this step started, if so resume
            elif os.path.isfile(last_ckpt_path):
                last_idx = idx
                last_ckpt = last_ckpt_path

        return last_idx, last_ckpt

    def run(self):
        """
        Execute each TrainingStep sequentially.
        """
        # Resume, if possible
        start_idx, last_ckpt = self._find_resume()
        if last_ckpt:
            log.info(
                f"Resuming curriculum at step {start_idx}/{len(self.steps)} from {last_ckpt}"
            )
        else:
            log.info("Starting curriculum from first step.")

        for idx, step in enumerate(self.steps):
            if idx < start_idx:
                log.info(f"Skipping already completed step {idx} ('{step.name}')")
                continue

            log.info(f"=== Running step: {step.name} ===")
            if step.cfg.experiment.warm_start_ckpt is not None:
                log.info(f"Loading ckpt: {step.cfg.experiment.warm_start_ckpt}")
            log.info(f"Saving -> {step.cfg.experiment.checkpointer.dirpath}")

            exp = Experiment(cfg=step.cfg)
            exp.train()

            log.info(f"=== Finished step: {step.name} ===")

            # TODO - cleanup?
