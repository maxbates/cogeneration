import copy
import os
from textwrap import dedent

import hydra
import torch
import torch._dynamo as dynamo
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from torch.profiler import ProfilerActivity, schedule

import wandb
from cogeneration.config.base import Config
from cogeneration.dataset.datasets import DatasetConstructor
from cogeneration.dataset.protein_dataloader import ProteinData
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils import (
    MemoryMonitorCallback,
    get_available_device,
    print_timing,
)
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.util.log import rank_zero_logger

log = rank_zero_logger(__name__)
torch.set_float32_matmul_precision("high")
torch.multiprocessing.set_sharing_strategy("file_system")


# Enable memory-efficient attention backends in PyTorch when available
if torch.cuda.is_available():
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)


class Experiment:
    def __init__(self, *, cfg: Config):
        self.cfg = cfg

        # Support warm starts
        if cfg.experiment.warm_start_ckpt is not None:
            ckpt_dir = os.path.dirname(cfg.experiment.warm_start_ckpt)
            log.info(f"🚩 Warm starting from {ckpt_dir}")
            assert os.path.exists(
                ckpt_dir
            ), f"Warm start directory {ckpt_dir} does not exist."

            # If specified, load model and merge in model config (to ensure all fields present)
            if cfg.experiment.warm_start_cfg_override:
                ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
                log.info(f"Loading warm start config from {ckpt_cfg_path}")

                merged_cfg, merged_ckpt_path = self.cfg.merge_checkpoint_cfg(
                    ckpt_cfg_path,
                    preserve_inference_cfg=False,
                )
                self.cfg = merged_cfg
                if merged_ckpt_path != ckpt_cfg_path:
                    log.info(f"Checkpoint path changed to {merged_ckpt_path}")
                    cfg.experiment.warm_start_ckpt = merged_ckpt_path

        # Handle DDP set up in case pytorch lightning doesn't handle it
        # (e.g. on mac laptop)
        setup_ddp(
            trainer_strategy=cfg.experiment.trainer.strategy,
            accelerator=cfg.experiment.trainer.accelerator,
            rank=str(DDPInfo.from_env().rank),
            world_size=str(cfg.experiment.num_devices),
        )

        # Dataset + Datamodule
        dataset_constructor = DatasetConstructor.from_cfg(self.cfg)
        train_dataset, valid_dataset = dataset_constructor.create_datasets()
        assert len(train_dataset) > 0, "Training dataset is empty"
        assert len(valid_dataset) > 0, "Validation dataset is empty"
        self._datamodule: LightningDataModule = ProteinData(
            data_cfg=self.cfg.data,
            dataset_cfg=self.cfg.dataset,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
        )

        # Handle devices. Folding may be assigned its own device.
        if self.cfg.experiment.trainer.accelerator == "cpu":
            folding_device_id = 0
            # set to number of processors, rather than list of devices, for Trainer().
            self._train_device_ids = torch.multiprocessing.cpu_count() // 2
        else:
            total_devices = self.cfg.experiment.num_devices
            device_ids = get_available_device(device_limit=total_devices)

            if self.cfg.folding.own_device:
                assert (
                    len(device_ids) > 1
                ), "Dedicated folding device requires at least 2 devices."
                folding_device_id = device_ids[0]
                self._train_device_ids = device_ids[1:]
            else:
                folding_device_id = 0
                self._train_device_ids = device_ids

        log.info(f"Folding device id: {folding_device_id}")
        log.info(f"Training with devices: {self._train_device_ids}")

        # Module
        self._module = FlowModule(
            self.cfg,
            folding_device=folding_device_id,
            offload_tools=not self.cfg.folding.own_device,
        )
        log.info("\n" + str(ModelSummary(self._module, max_depth=2)))

        # Load model state dict if provided
        if self.cfg.experiment.raw_state_dict_reload is not None:
            self._module.load_state_dict(
                torch.load(self.cfg.experiment.raw_state_dict_reload)["state_dict"]
            )

    @print_timing
    def train(self):
        callbacks = []

        callbacks.append(TQDMProgressBar(refresh_rate=1))

        # Add memory monitoring (signal-only mode, use kill -SIGUSR1 <PID> to trigger)
        callbacks.append(MemoryMonitorCallback())

        if self.cfg.experiment.debug:
            # Debug mode uses only one device and no workers
            self._train_device_ids = [self._train_device_ids[0]]
            self.cfg.data.loader.num_workers = 0

        if self.cfg.shared.local:
            # If local, use tensorboard logger
            local_tensorboard_logdir = "./tensorboard_logs"
            log.info(
                f"Local mode. Using Tensorboard logger @ {local_tensorboard_logdir}"
            )
            logger = TensorBoardLogger(
                save_dir=local_tensorboard_logdir,
                name=self.cfg.experiment.wandb.name,
            )
        else:
            # Set up w&b logging
            logger = WandbLogger(**self.cfg.experiment.wandb.asdict())
            # Model checkpoints
            checkpoint_cfg = self.cfg.experiment.checkpointer.asdict()
            callbacks.append(ModelCheckpoint(**checkpoint_cfg))
            # Save every n training steps
            # TODO - clean up, use cfg explicitly
            n_step_cfg = copy.deepcopy(checkpoint_cfg)
            del n_step_cfg["every_n_epochs"]
            n_step_cfg["every_n_train_steps"] = 2000
            n_step_cfg["monitor"] = "train/loss"
            callbacks.append(ModelCheckpoint(**n_step_cfg))

        # Save config if main process
        local_rank = DDPInfo.from_env().local_rank
        if local_rank == 0:
            # write locally
            ckpt_dir = self.cfg.experiment.checkpointer.dirpath
            log.info(
                f"Checkpoints, config, validations etc. will be saved to: {ckpt_dir}"
            )
            os.makedirs(ckpt_dir, exist_ok=True)
            cfg_path = os.path.join(ckpt_dir, "config.yaml")
            with open(cfg_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f.name)

            # write to w&b
            if logger is not None and isinstance(logger, WandbLogger):
                logger.experiment.config.update(self.cfg.flatdict())

        # set up profiler to save traces to wandb
        profiler = None
        if self.cfg.experiment.profile_chrome_trace:
            log.warning(
                dedent(
                    """
                    ⚠️ Setting up profiler to save chrome traces to wandb.
                    These are large (100+ MB) and only recommended for debugging if training is slow / failing.
                    """
                )
            )

            def wandb_trace_uploader(prof: torch.profiler.profile):
                # save chrome trace to wandb
                step = prof.step_num
                fname = f"trace_step{step}.json"
                prof.export_chrome_trace(fname)
                artifact_trace = wandb.Artifact(f"trace_{step}", type="trace")
                artifact_trace.add_file(fname)
                wandb.run.log_artifact(artifact_trace)

                # save memory snapshot
                # if torch.cuda.is_available():
                #     snapshot_path = f"snapshot_{step}.pickle"
                #     torch.cuda.memory._dump_snapshot(snapshot_path)
                #     artifact_snapshot = wandb.Artifact(
                #         f"snapshot_{step}", type="snapshot"
                #     )
                #     artifact_snapshot.add_file(snapshot_path)
                #     wandb.run.log_artifact(artifact_snapshot)

            profiler = PyTorchProfiler(
                # choose schedule based on observed problems
                schedule=schedule(wait=0, warmup=1, active=1, repeat=0),
                on_trace_ready=wandb_trace_uploader,
                # Including both CPU + GPU, esp with memory, may yield a very large trace (>> GB)
                # activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                with_stack=True,
                # disable to avoid large memory allocation
                # may lead to dataloader worker overload
                profile_memory=False,
                # disable to reduce profiler memory usage
                record_shapes=False,
            )

        # Try to compile model
        model = self._module
        if self.cfg.experiment.torch_compile:
            log.info("Running torch.compile()...")

            # torch dynamo explanation for debugging - note requires DDP set up for data module
            # explanation = dynamo.explain(model)(
            #     next(iter(self._datamodule.train_dataloader()))
            # )
            # log.info(explanation)

            try:
                model: LightningModule = torch.compile(self._module, mode="max-autotune")  # type: ignore
            except Exception as e:
                log.warning(
                    f"Failed to torch.compile model, continuing with original: {e}"
                )

        log.info("Setting up Trainer...")
        trainer = Trainer(
            **self.cfg.experiment.trainer.asdict(),
            callbacks=callbacks,
            logger=logger,  # pass w&b logger
            use_distributed_sampler=False,  # TODO - ddp
            devices=self._train_device_ids,
            profiler=profiler,
            enable_model_summary=False,  # manual model summary
        )

        # Train
        log.info("Starting training...")
        trainer.fit(
            model=model,
            datamodule=self._datamodule,
            ckpt_path=self.cfg.experiment.warm_start_ckpt,
        )

        # Save final ckpt. Defaults to symlink if possible, but can save final ckpt explicitly.
        if self.cfg.experiment.save_final_ckpt:
            log.info("Saving final checkpoint...")
            last_ckpt_path = os.path.join(
                self.cfg.experiment.checkpointer.dirpath, "last.ckpt"
            )
            final_ckpt_path = os.path.join(
                self.cfg.experiment.checkpointer.dirpath, "final.ckpt"
            )
            if self.cfg.experiment.final_ckpt_symlink:
                if not os.path.exists(last_ckpt_path):
                    log.error(
                        f"Final checkpoint symlink requested, but last checkpoint does not at exist: {last_ckpt_path}."
                        + f"Set `cfg.experiment.checkpointer.save_last == True`. Saving copy."
                    )
                    trainer.save_checkpoint(final_ckpt_path)
                else:
                    if os.path.exists(final_ckpt_path):
                        os.remove(final_ckpt_path)
                    os.link(last_ckpt_path, final_ckpt_path)
            else:
                trainer.save_checkpoint(final_ckpt_path)


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(cfg: Config) -> None:
    log.info(f"Starting training. {cfg.experiment.num_devices} GPUs specified.")

    cfg = OmegaConf.to_object(cfg)
    cfg = cfg.interpolate()

    trainer = Experiment(cfg=cfg)
    trainer.train()


if __name__ == "__main__":
    run()
