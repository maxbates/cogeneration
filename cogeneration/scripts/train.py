import os
from dataclasses import asdict

import hydra
import torch
import torch._dynamo as dynamo
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.trainer import Trainer

import wandb
from cogeneration.config.base import Config
from cogeneration.dataset.datasets import BaseDataset, DatasetConstructor, PdbDataset
from cogeneration.dataset.protein_dataloader import ProteinData
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils import flatten_dict, get_available_device, print_timing
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.util.log import rank_zero_logger

log = rank_zero_logger(__name__)
torch.set_float32_matmul_precision("high")


class Experiment:
    def __init__(self, *, cfg: Config):
        self.cfg = cfg

        # Warm start
        if cfg.experiment.warm_start is not None:
            ckpt_dir = os.path.dirname(cfg.experiment.warm_start)
            log.info(f"Warm starting from {ckpt_dir}")
            assert os.path.exists(
                ckpt_dir
            ), f"Warm start directory {ckpt_dir} does not exist."

            # If specified, load model and merge in model config (to ensure all fields present)
            if cfg.experiment.warm_start_cfg_override:
                ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
                ckpt_cfg = OmegaConf.load(ckpt_cfg_path)
                OmegaConf.set_struct(cfg.model, False)
                OmegaConf.set_struct(ckpt_cfg.model, False)
                cfg.model = OmegaConf.merge(cfg.model, ckpt_cfg.model)
                OmegaConf.set_struct(cfg.model, True)
                log.info(f"Loaded warm start config from {ckpt_cfg_path}")

        # Ensure DDP is set up for scenarios were pytorch lightning doesn't handle it
        # (e.g. debugging on mac laptop)
        setup_ddp(
            trainer_strategy=cfg.experiment.trainer.strategy,
            accelerator=cfg.experiment.trainer.accelerator,
            rank=str(DDPInfo.from_env().rank),
            world_size=str(cfg.experiment.num_devices),
        )

        # Setup datasets
        dataset_constructor = DatasetConstructor.from_cfg(self.cfg)
        train_dataset, valid_dataset = dataset_constructor.create_datasets()
        assert len(train_dataset) > 0, "Training dataset is empty"
        assert len(valid_dataset) > 0, "Validation dataset is empty"
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self.cfg.data,
            dataset_cfg=self.cfg.dataset,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )

        # Determine devices, accounting that folding may be assigned its own device.
        if self.cfg.experiment.trainer.accelerator == "cpu":
            folding_device_id = 0
            # set to number of processors, rather than list of devices, this is what Trainer() wants.
            self._train_device_ids = torch.multiprocessing.cpu_count() // 2
        else:
            total_devices = self.cfg.experiment.num_devices
            if self.cfg.folding.own_device:
                total_devices += 1

            device_ids = get_available_device(device_limit=total_devices)

            if self.cfg.folding.own_device:
                assert (
                    len(device_ids) > 1
                ), "Dedicated folding device requires at least 2 devices."
                folding_device_id = device_ids[0]
                self._train_device_ids = device_ids[1:]
                log.info(f"Folding device id: {folding_device_id}")
            else:
                folding_device_id = 0
                self._train_device_ids = device_ids

        log.info(f"Training with devices: {self._train_device_ids}")

        # Initialize module
        self._module = FlowModule(
            self.cfg,
            folding_device_id=folding_device_id,
        )

        # Load model state dict if provided
        if self.cfg.experiment.raw_state_dict_reload is not None:
            self._module.load_state_dict(
                torch.load(self.cfg.experiment.raw_state_dict_reload)["state_dict"]
            )

    @print_timing
    def train(self):
        callbacks = []

        if self.cfg.experiment.debug:
            # Debug mode uses only one device and no workers
            self._train_device_ids = [self._train_device_ids[0]]
            self.cfg.data.loader.num_workers = 0

        if self.cfg.shared.local:
            # If local, use tensorboard logger
            log.info(
                f"Local mode. Using Tensorboard logger @ {self.cfg.experiment.trainer.local_tensorboard_logdir}"
            )
            logger = TensorBoardLogger(
                save_dir=self.cfg.experiment.trainer.local_tensorboard_logdir,
                name=self.cfg.experiment.wandb.name,
            )
        else:
            # Set up w&b logging
            logger = WandbLogger(**asdict(self.cfg.experiment.wandb))
            # Model checkpoints
            callbacks.append(
                ModelCheckpoint(**asdict(self.cfg.experiment.checkpointer))
            )

        # Save config, only for main process.
        local_rank = DDPInfo.from_env().local_rank
        if local_rank == 0:
            # write locally
            ckpt_dir = self.cfg.experiment.checkpointer.dirpath
            log.info(f"Checkpoints saved to {ckpt_dir}")
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

            # write to w&b
            if logger is not None and isinstance(logger, WandbLogger):
                flat_cfg = dict(flatten_dict(asdict(self.cfg)))
                logger.experiment.config.update(flat_cfg)

        log.info("Setting up Trainer...")
        trainer = Trainer(
            **asdict(self.cfg.experiment.trainer),
            callbacks=callbacks,
            logger=logger,  # pass w&b logger
            use_distributed_sampler=False,
            enable_progress_bar=True,
            enable_model_summary=True,
            devices=self._train_device_ids,
            profiler=self.cfg.experiment.profiler,
        )

        # Try to compile model
        model = self._module
        if self.cfg.experiment.torch_compile:
            # torch dynamo explanation for debugging
            explanation = dynamo.explain(model)(
                next(iter(self._datamodule.train_dataloader()))
            )
            log.info(explanation)

            try:
                model: LightningModule = torch.compile(self._module)  # type: ignore
            except Exception as e:
                log.warning(
                    f"Failed to torch.compile model, continuing with original: {e}"
                )

        # Train
        log.info("Starting training...")
        trainer.fit(
            model=model,
            datamodule=self._datamodule,
            ckpt_path=self.cfg.experiment.warm_start,
        )


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(cfg: Config) -> None:
    log.info(f"Starting training. {cfg.experiment.num_devices} GPUs specified.")

    # Instantiate static config with dataclass as nested objects
    # to avoid dynamic DictConfig lookups for torch dynamo
    cfg = OmegaConf.to_object(cfg)

    trainer = Experiment(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    run()
