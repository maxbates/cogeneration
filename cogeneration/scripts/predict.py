import copy
import os
from typing import Optional, Tuple

import hydra
import pandas as pd
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from cogeneration.config.base import Config
from cogeneration.dataset.datasets import DatasetConstructor, LengthSamplingDataset
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils import (
    get_available_device,
    print_timing,
    setup_cuequivariance_env,
)
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.type.task import InferenceTask
from cogeneration.util.log import rank_zero_logger

torch.set_float32_matmul_precision("high")
log = rank_zero_logger(__name__)


class EvalRunner:
    def __init__(self, cfg: Config):
        self._input_cfg: Config = copy.deepcopy(cfg)

        # singletons
        self._trainer: Optional[Trainer] = None
        self._dataloader: Optional[torch.utils.data.DataLoader] = None

        # Read in checkpoint config
        if cfg.inference.task == InferenceTask.unconditional:
            ckpt_path = cfg.inference.unconditional_ckpt_path
        elif cfg.inference.task == InferenceTask.inpainting:
            ckpt_path = cfg.inference.inpainting_ckpt_path
        elif cfg.inference.task == InferenceTask.forward_folding:
            ckpt_path = cfg.inference.forward_folding_ckpt_path
        elif cfg.inference.task == InferenceTask.inverse_folding:
            ckpt_path = cfg.inference.inverse_folding_ckpt_path
        else:
            raise ValueError(f"Unknown task {cfg.inference.task}")

        # Merge the checkpoint config into the current config
        log.info(f"Loading checkpoint from {ckpt_path}")
        merged_cfg, merged_ckpt_path = self._input_cfg.merge_checkpoint_cfg(
            ckpt_path=ckpt_path,
            preserve_inference_cfg=True,
        )
        self.cfg = merged_cfg
        if merged_ckpt_path != ckpt_path:
            log.info(f"Checkpoint path changed to {merged_ckpt_path}")
            ckpt_path = merged_ckpt_path

        local_rank = DDPInfo.from_env().local_rank

        # Setup for cuEquivariance
        setup_cuequivariance_env(
            kernels_enabled=cfg.shared.kernels,
            enable_bf16=cfg.shared.kernels_bf16
            and (cfg.experiment.trainer.precision == "bf16"),
        )

        # Ensure DDP is set up for scenarios were pytorch lightning doesn't handle it
        # (e.g. debugging on Mac laptop)
        setup_ddp(
            trainer_strategy=cfg.experiment.trainer.strategy,
            accelerator=cfg.experiment.trainer.accelerator,
            rank=str(local_rank),
            world_size=str(cfg.experiment.num_devices),
        )

        # Set-up output directory only on rank 0, including writing `config.yaml`
        if local_rank == 0:
            inference_dir = self.setup_inference_dir(ckpt_path)
            log.info(f"Saving results to {inference_dir}")
            self.cfg.inference.predict_dir = inference_dir
            config_path = os.path.join(inference_dir, "config.yaml")
            with open(config_path, "w") as f:
                OmegaConf.save(config=self.cfg, f=f)
            log.info(f"Saving inference config to {config_path}")

        # Read checkpoint and initialize module
        self._flow_module = FlowModule.load_from_checkpoint(
            checkpoint_path=ckpt_path,
            cfg=self.cfg,
        )
        self._flow_module.folding_validator.set_device_id(0)
        log.info(ModelSummary(self._flow_module, max_depth=2))
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

    @property
    def dataloader(self):
        """
        Returns a dataloader singleton, i.e. with a fixed rng seed.
        Reset it with `reset_dataloader` for new sampling dataset.
        """
        if self._dataloader is not None:
            return self._dataloader

        if self.cfg.inference.task == InferenceTask.unconditional:
            eval_dataset = LengthSamplingDataset(self.cfg.inference.samples)
        elif (
            self.cfg.inference.task == InferenceTask.inpainting
            or self.cfg.inference.task == InferenceTask.forward_folding
            or self.cfg.inference.task == InferenceTask.inverse_folding
        ):
            # We want to use the input cfg inference settings for the pdb dataset,
            # not what was in the ckpt config
            # TODO(dataset) - actually read the config, rather than using constructor to get new instance
            pdb_test_cfg = self.cfg.dataset.PDBPost2021()

            # The dataset will behave differently depending on the task
            # i.e. for inpainting, we generate motifs.
            dataset_constructor = DatasetConstructor.pdb_dataset(
                dataset_cfg=pdb_test_cfg,
                task=InferenceTask.to_data_task(self.cfg.inference.task),
            )
            _, eval_dataset = dataset_constructor.create_datasets()
        else:
            raise ValueError(f"Unknown task {self.cfg.inference.task}")

        self._dataloader = torch.utils.data.DataLoader(
            eval_dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        return self._dataloader

    def reset_dataloader(self):
        self._dataloader = None

    @property
    def trainer(self) -> Trainer:
        if self._trainer is not None:
            return self._trainer

        devices = get_available_device(device_limit=self.cfg.experiment.num_devices)
        log.info(f"Using devices: {devices}")

        self._trainer = Trainer(
            **self.cfg.experiment.trainer.asdict(),
            devices=devices,
        )
        return self._trainer

    @print_timing
    def run_sampling(self):
        log.info(f"Evaluating {self.cfg.inference.task}")

        self.trainer.predict(
            model=self._flow_module,
            dataloaders=self.dataloader,
        )

    @print_timing
    def compute_metrics(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        log.info(f"Calculating metrics for samples in {self.inference_dir}")

        top_samples_df, top_samples_path = self._flow_module.concat_all_top_samples(
            output_dir=self.inference_dir,
            is_inference=True,
        )
        top_metrics_df, top_metrics_path = (
            self._flow_module.folding_validator.assess_all_top_samples(
                task=self.cfg.inference.task,
                top_samples_df=top_samples_df,
                output_dir=self.inference_dir,
            )
        )

        return top_samples_df, top_metrics_df


@hydra.main(version_base=None, config_path="../config", config_name="base")
def run(cfg: Config) -> None:
    log.info(f"Starting inference, using {cfg.inference.num_gpus} GPUs")

    cfg = cfg.interpolate()

    sampler = EvalRunner(cfg=cfg)
    sampler.run_sampling()

    if DDPInfo.from_env().rank == 0:
        sampler.compute_metrics()


if __name__ == "__main__":
    run()
