import os
from dataclasses import asdict

import hydra
import numpy as np
import pandas as pd
import torch
from config.base import Config, InferenceTaskEnum
from dataset.datasets import DatasetConstructor, LengthSamplingDataset
from models.module import FlowModule
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary
from scripts.utils import get_available_device, print_timing
from scripts.utils_ddp import DDPInfo, setup_ddp
from util.log import rank_zero_logger

torch.set_float32_matmul_precision("high")
log = rank_zero_logger(__name__)


class EvalRunner:
    def __init__(self, cfg: Config):
        self._input_cfg = cfg

        # Read in checkpoint and config
        if cfg.inference.task == InferenceTaskEnum.unconditional:
            ckpt_path = cfg.inference.unconditional_ckpt_path
        elif cfg.inference.task == InferenceTaskEnum.forward_folding:
            ckpt_path = cfg.inference.forward_folding_ckpt_path
        elif cfg.inference.task == InferenceTaskEnum.inverse_folding:
            ckpt_path = cfg.inference.inverse_folding_ckpt_path
        else:
            raise ValueError(f"Unknown task {cfg.inference.task}")
        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
        self._original_cfg = cfg.copy()

        # Merge configs
        OmegaConf.set_struct(cfg, False)
        OmegaConf.set_struct(ckpt_cfg, False)
        cfg = OmegaConf.merge(cfg, ckpt_cfg)
        cfg.experiment.checkpointer.dirpath = "./"
        self.cfg = cfg

        # Ensure DDP is set up for scenarios were pytorch lightning doesn't handle it
        # (e.g. debugging on Mac laptop)
        setup_ddp(
            trainer_strategy=cfg.experiment.trainer.strategy,
            accelerator=cfg.experiment.trainer.accelerator,
            rank=os.environ.get("LOCAL_RANK", "0"),
            world_size=str(cfg.experiment.num_devices),
        )

        # Set-up output directory only on rank 0, including writing `config.yaml`
        local_rank = os.environ.get("LOCAL_RANK", 0)
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            log.info(f"Saving results to {inference_dir}")
            self.cfg.experiment.inference_dir = inference_dir
            config_path = os.path.join(inference_dir, "config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f)
            log.info(f"Saving inference config to {config_path}")

        # Read checkpoint and initialize module
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self.cfg,
        )
        log.info(ModelSummary(self._flow_module))
        self._flow_module.eval()

    @property
    def inference_dir(self):
        return self._flow_module.inference_dir

    def setup_inference_dir(self, ckpt_path) -> str:
        ckpt_name = "/".join(ckpt_path.replace(".ckpt", "").split("/")[-3:])
        output_dir = os.path.join(
            self.cfg.inference.predict_dir,
            ckpt_name,
            self.cfg.inference.task,
            self.cfg.inference.inference_subdir,
        )
        os.makedirs(output_dir, exist_ok=True)
        return output_dir

    @print_timing
    def run_sampling(self):
        devices = get_available_device(device_limit=self.cfg.experiment.num_devices)
        log.info(f"Using devices: {devices}")
        log.info(f"Evaluating {self.cfg.inference.task}")

        if self.cfg.inference.task == InferenceTaskEnum.unconditional:
            eval_dataset = LengthSamplingDataset(self.cfg.samples)
        elif (
            self.cfg.inference.task == InferenceTaskEnum.forward_folding
            or self.cfg.inference.task == InferenceTaskEnum.inverse_folding
        ):
            # We want to use the inference settings for the pdb dataset, not what was in the ckpt config
            dataset_constructor = DatasetConstructor.pdb_test(
                dataset_cfg=self._original_cfg.pdb_post2021_dataset
            )
            eval_dataset, _ = dataset_constructor.create_datasets()
        else:
            raise ValueError(f"Unknown task {self.cfg.inference.task}")

        dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        trainer = Trainer(
            **asdict(self.cfg.experiment.trainer),
            devices=devices,
        )
        trainer.predict(self._flow_module, dataloaders=dataloader)

    def compute_unconditional_metrics(self):
        # TODO - need metrics class and to calculate task specific metrics. See Multiflow
        metrics_csv_path = os.path.join(self.inference_dir, "designable.csv")

    def compute_forward_folding_metrics(self):
        # TODO - need metrics class and to calculate task specific metrics. See Multiflow
        metrics_csv_path = os.path.join(self.inference_dir, "forward_fold_metrics.csv")

    def compute_inverse_folding_metrics(self):
        # TODO - need metrics class and to calculate task specific metrics. See Multiflow
        metrics_csv_path = os.path.join(self.inference_dir, "inverse_fold_metrics.csv")

    @print_timing
    def compute_metrics(self):
        log.info(f"Calculating metrics for {self.inference_dir}")

        if self.cfg.inference.task == InferenceTaskEnum.unconditional:
            self.compute_unconditional_metrics()
        elif self.cfg.inference.task == InferenceTaskEnum.forward_folding:
            self.compute_forward_folding_metrics()
        elif self.cfg.inference.task == InferenceTaskEnum.inverse_folding:
            self.compute_inverse_folding_metrics()
        else:
            raise ValueError(f"Unknown task {self.cfg.inference.task}")


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(cfg: Config) -> None:
    # Read model checkpoint.
    log.info(f"Starting inference with {cfg.inference.num_gpus} GPUs")

    sampler = EvalRunner(cfg=cfg)
    sampler.run_sampling()

    if DDPInfo.from_env().rank == 0:
        sampler.compute_metrics()


if __name__ == "__main__":
    run()
