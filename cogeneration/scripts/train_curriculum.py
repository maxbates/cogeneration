import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from cogeneration.config.base import Config, DatasetFilterConfig
from cogeneration.config.curriculum import Curriculum, TrainingStep


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(cfg: DictConfig) -> None:

    # shared timestamp id prefix
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    steps: list[TrainingStep] = []

    # Phase 1: 50 epochs, all data incl. synthetic, len <= 256
    phase1 = Config.from_dict_config(cfg)
    phase1.shared.id = f"{timestamp}_p1_len25"
    phase1.experiment.trainer.max_epochs = 50
    phase1.dataset.filter.max_num_res = 256
    phase1.dataset.enable_multiflow_synthetic = True
    phase1 = phase1.interpolate()
    steps.append(TrainingStep(name="phase1_len256", cfg=phase1))

    # Phase 2: 50 epochs, all data, len <= 512
    phase2 = Config.from_dict_config(cfg)
    phase2.shared.id = f"{timestamp}_p2_len512"
    phase2.experiment.trainer.max_epochs = 50
    phase2.dataset.filter.max_num_res = 512
    phase2.dataset.enable_multiflow_synthetic = True
    phase2 = phase2.interpolate()
    steps.append(TrainingStep(name="phase2_len512", cfg=phase2))

    # Phase 3: 50 epochs, drop synthetic, len <= 1024
    phase3 = Config.from_dict_config(cfg)
    phase3.shared.id = f"{timestamp}_p3_len1024_no_synth"
    phase3.experiment.trainer.max_epochs = 50
    phase3.dataset.filter.max_num_res = 1024
    phase3.dataset.enable_multiflow_synthetic = False
    phase3 = phase3.interpolate()
    steps.append(TrainingStep(name="phase3_len1024_no_synth", cfg=phase3))

    # Phase 4: 20 epochs, multimers, len <= 2048
    # (requires gpu size sufficient to fit under max res squared per batch constraint)
    # TODO - ideally 50/50 multimers vs monomers. requires new dataset filter option.
    phase4 = Config.from_dict_config(cfg)
    phase4.shared.id = f"{timestamp}_p4_len2048_multimers"
    phase4.experiment.trainer.max_epochs = 20
    phase4.dataset.filter = DatasetFilterConfig.multimeric()
    phase4.dataset.filter.max_num_res = 2048
    phase4 = phase4.interpolate()
    steps.append(TrainingStep(name="phase4_len2048_multimers", cfg=phase4))

    Curriculum(steps=steps).run()


if __name__ == "__main__":
    run()
