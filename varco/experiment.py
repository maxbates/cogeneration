import copy
import os
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.util.log import rank_zero_logger
from varco.config import VarcoConfig
from varco.dataset import ProteinDataModule, ProteinDataset
from varco.module import BranchFlowModule
from varco.viz import BranchingFlowVisualizer

logger = rank_zero_logger("BranchingFlow")


class Experiment:
    def __init__(self, cfg: VarcoConfig):
        self.cfg = cfg
        self.logger = rank_zero_logger(__name__)
        self._warm_start_ckpt_path: Optional[str] = None

    def setup(self):
        pl.seed_everything(self.cfg.shared.seed, workers=True)

        # Support warm starts / resume from a varco Lightning checkpoint.
        if self.cfg.experiment.warm_start_ckpt is not None:
            ckpt_path = str(self.cfg.experiment.warm_start_ckpt)
            if not os.path.isabs(ckpt_path):
                ckpt_path = str(Path(self.cfg.shared.project_root) / ckpt_path)
            if not ckpt_path.endswith(".ckpt"):
                raise ValueError(
                    f"Invalid warm start checkpoint path {ckpt_path!r}; expected .ckpt"
                )
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Warm start checkpoint {ckpt_path!r} does not exist."
                )
            self._warm_start_ckpt_path = ckpt_path
            self.logger.info(f"🚩 Warm starting from {ckpt_path}")

        # Handle DDP set up in case pytorch lightning doesn't handle it
        # (e.g. on mac laptop)
        setup_ddp(
            trainer_strategy=self.cfg.experiment.trainer.strategy,
            accelerator=self.cfg.experiment.trainer.accelerator,
            rank=str(DDPInfo.from_env().rank),
            world_size=str(self.cfg.experiment.num_devices),
        )

        # Create output directory for this run
        predict_dir = self.cfg.inference.predict_dir
        os.makedirs(predict_dir, exist_ok=True)
        logger.info(f"📁 Output directory: {predict_dir}")

        self.data_module = ProteinDataModule(
            dataset=ProteinDataset(cfg=self.cfg.dataset),
            cfg=self.cfg.data,
        )

        self.module = BranchFlowModule(cfg=self.cfg)

        # Load cogeneration weights if specified
        if self.cfg.experiment.cogen_ckpt_path:
            if self._warm_start_ckpt_path is not None:
                raise ValueError(
                    "`cfg.experiment.cogen_ckpt_path` cannot be used together with "
                    "`cfg.experiment.warm_start_ckpt`."
                )
            self.module.load_cogeneration_weights(self.cfg.experiment.cogen_ckpt_path)

        self.logger.info("\n" + str(ModelSummary(self.module, max_depth=2)))

    def debug(self, n: int = 10):
        if n <= 0:
            return

        debug_dir = os.path.join(self.cfg.inference.predict_dir, "debug")
        os.makedirs(debug_dir, exist_ok=True)

        # Plot corruption planning trees
        for i in range(n):
            datum = self.data_module.dataset[i]
            datum.tree_plan.plot(
                path=os.path.join(debug_dir, f"init_tree_plan_{i}.png")
            )

        # Visualize corruption processes
        # 0 for deterministic bridges, >0 for stochasticity
        viz = BranchingFlowVisualizer(sigma=0.0)
        data_loader = self.data_module.train_dataloader(rank=0, num_replicas=1)
        for i, debug_batch in enumerate(data_loader):
            if i >= n:
                break
            viz.visualize_corruption(
                batch=debug_batch, out_dir=debug_dir, filename=f"debug_corruption_{i}"
            )

    def save_config(self, wandb_logger: WandbLogger):
        # Save config if main process
        local_rank = DDPInfo.from_env().local_rank
        if local_rank == 0:
            # write locally
            ckpt_dir = self.cfg.experiment.checkpointer.dirpath
            self.logger.info(
                f"💾 Checkpoints, config, validations etc. will be saved to: {ckpt_dir}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

            # write to w&b
            if wandb_logger is not None and isinstance(wandb_logger, WandbLogger):
                wandb_logger.experiment.config.update(self.cfg.flatdict())

    def train(self):
        # Set up w&b logging
        wandb_logger = WandbLogger(**self.cfg.experiment.wandb.asdict())

        callbacks = []

        # Simple progress bar
        # callbacks.append(TQDMProgressBar(refresh_rate=1))

        # Model checkpoints
        monitor = "L/train_ema"
        checkpoint_cfg = self.cfg.experiment.checkpointer.asdict()
        checkpoint_cfg["monitor"] = monitor
        callbacks.append(ModelCheckpoint(**checkpoint_cfg))

        # Save every n training steps
        # TODO - clean up, use cfg explicitly
        n_step_cfg = copy.deepcopy(checkpoint_cfg)
        del n_step_cfg["every_n_epochs"]
        n_step_cfg["every_n_train_steps"] = 2000
        n_step_cfg["monitor"] = monitor
        callbacks.append(ModelCheckpoint(**n_step_cfg))

        self.save_config(wandb_logger=wandb_logger)

        self.logger.info("Setting up Trainer...")
        trainer = Trainer(
            **self.cfg.experiment.trainer.asdict(),
            callbacks=callbacks,
            logger=wandb_logger,
            use_distributed_sampler=False,  # TODO - ddp
            devices=self.cfg.experiment.num_devices,
            enable_model_summary=False,  # manual model summary
            enable_progress_bar=True,
        )

        trainer.fit(
            self.module,
            datamodule=self.data_module,
            ckpt_path=self._warm_start_ckpt_path,
        )

        self.logger.info(f"Training complete")
        self.logger.info(f"💾 ckpt saved to {self.cfg.experiment.checkpointer.dirpath}")
        self.logger.info(f"🏆 Best checkpoint monitor: {monitor}")

        return self.module
