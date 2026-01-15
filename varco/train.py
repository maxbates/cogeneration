import hydra
import torch
from omegaconf import OmegaConf

from varco.config import VarcoConfig
from varco.experiment import Experiment

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

# Enable memory-efficient attention backends in PyTorch when available
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


@hydra.main(config_path=".", config_name="varco", version_base=None)
def main(cfg: VarcoConfig):
    cfg = OmegaConf.to_object(cfg)
    cfg = cfg.interpolate()

    experiment = Experiment(cfg=cfg)
    experiment.setup()
    experiment.debug(n=1)
    experiment.train()


if __name__ == "__main__":
    main()
