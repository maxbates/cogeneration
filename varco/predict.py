import copy
import os
from pathlib import Path
from typing import Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.util.log import rank_zero_logger
from varco.config import VarcoConfig
from varco.dataset import ProteinDataLoader, ProteinDataset
from varco.module import BranchFlowModule

logger = rank_zero_logger(__name__)

torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")

# Enable memory-efficient attention backends in PyTorch when available
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


class VarcoEvaluator:
    def __init__(self, cfg: VarcoConfig):
        self._input_cfg: VarcoConfig = copy.deepcopy(cfg)
        self.cfg = self._input_cfg
        self.cfg = self.cfg.interpolate()

        pl.seed_everything(self.cfg.shared.seed, workers=True)

        self._trainer: Optional[Trainer] = None
        self._dataloader: Optional[torch.utils.data.DataLoader] = None

        if not self.cfg.inference.ckpt_path:
            raise ValueError("`cfg.inference.ckpt_path` is required for prediction.")
        ckpt_path = str(self.cfg.inference.ckpt_path)
        if not os.path.isabs(ckpt_path):
            ckpt_path = str(Path(self.cfg.shared.project_root) / ckpt_path)
        if not ckpt_path.endswith(".ckpt"):
            raise ValueError(f"Invalid checkpoint path {ckpt_path!r}; expected .ckpt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint {ckpt_path} does not exist.")

        # Ensure DDP is set up for scenarios where Lightning doesn't handle it
        # (e.g. debugging on a Mac laptop with MPS/CPU).
        setup_ddp(
            trainer_strategy=self.cfg.experiment.trainer.strategy,
            accelerator=self.cfg.experiment.trainer.accelerator,
            rank=str(DDPInfo.from_env().rank),
            world_size=str(self.cfg.experiment.num_devices),
        )

        # Compute output directory deterministically on all ranks; write config only on rank 0.
        inference_dir = self._setup_inference_dir(ckpt_path=ckpt_path)
        self.cfg.inference.predict_dir = inference_dir
        if DDPInfo.from_env().local_rank == 0:
            os.makedirs(inference_dir, exist_ok=True)
            config_path = os.path.join(inference_dir, "config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f)
            logger.info(f"💾 Saving inference config to {config_path}")

        # Read checkpoint and initialize module
        try:
            self._module = BranchFlowModule.load_from_checkpoint(
                checkpoint_path=ckpt_path,
                cfg=self.cfg,
            )
        except Exception as e:
            logger.error(f"Failed to load checkpoint {ckpt_path}: {e}")
            raise

        self._module.eval()
        logger.info("\n" + str(ModelSummary(self._module, max_depth=2)))

    def _setup_inference_dir(self, ckpt_path: str) -> str:
        ckpt_name = "/".join(Path(ckpt_path).with_suffix("").parts[-3:])
        predict_root = Path(self.cfg.inference.predict_dir)
        if not predict_root.is_absolute():
            predict_root = Path(self.cfg.shared.project_root) / predict_root
        return str(predict_root / ckpt_name / self.cfg.inference.inference_subdir)

    @property
    def dataloader(self) -> torch.utils.data.DataLoader:
        if self._dataloader is not None:
            return self._dataloader

        dataset = ProteinDataset(
            cfg=self.cfg.dataset,
            eval=True,
            use_test=True,
        )

        self._dataloader = ProteinDataLoader(
            dataset=dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )
        return self._dataloader

    @property
    def trainer(self) -> Trainer:
        if self._trainer is not None:
            return self._trainer

        trainer_kwargs = dict(self.cfg.experiment.trainer.asdict())
        trainer_kwargs.update(
            {
                "logger": False,
                "enable_checkpointing": False,
                "enable_model_summary": False,
                "enable_progress_bar": True,
                "use_distributed_sampler": True,
                "devices": self.cfg.experiment.num_devices,
            }
        )
        self._trainer = Trainer(**trainer_kwargs)
        return self._trainer

    def run(self) -> None:
        self.trainer.predict(
            model=self._module,
            dataloaders=self.dataloader,
        )
        logger.info("Prediction complete")


@hydra.main(config_path=".", config_name="varco", version_base=None)
def run(cfg: VarcoConfig) -> None:
    cfg = OmegaConf.to_object(cfg)
    cfg = cfg.interpolate()

    evaluator = VarcoEvaluator(cfg=cfg)
    evaluator.run()


if __name__ == "__main__":
    run()
