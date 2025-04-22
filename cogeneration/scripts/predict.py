import copy
import os
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional, Tuple

import hydra
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.model_summary import ModelSummary

from cogeneration.config.base import Config
from cogeneration.dataset.datasets import DatasetConstructor, LengthSamplingDataset
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils import get_available_device, print_timing
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.type.task import InferenceTaskEnum
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
        if cfg.inference.task == InferenceTaskEnum.unconditional:
            ckpt_path = cfg.inference.unconditional_ckpt_path
        elif cfg.inference.task == InferenceTaskEnum.inpainting:
            ckpt_path = cfg.inference.inpainting_ckpt_path
        elif cfg.inference.task == InferenceTaskEnum.forward_folding:
            ckpt_path = cfg.inference.forward_folding_ckpt_path
        elif cfg.inference.task == InferenceTaskEnum.inverse_folding:
            ckpt_path = cfg.inference.inverse_folding_ckpt_path
        else:
            raise ValueError(f"Unknown task {cfg.inference.task}")

        # Merge the checkpoint config into the current config
        merged_cfg, ckpt_path = self.merge_checkpoint_cfg(cfg=cfg, ckpt_path=ckpt_path)
        # Save the merged cfg
        # Note that it is a DictConfig, because it may contain keys not defined in the Config dataclass
        self.cfg: DictConfig = merged_cfg

        local_rank = DDPInfo.from_env().local_rank

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

    @staticmethod
    def merge_checkpoint_cfg(
        cfg: Config,
        ckpt_path: str,
    ) -> Tuple[DictConfig, str]:
        """
        Load and merge checkpoint config from `ckpt_path` into current Config `cfg`.
        Also maps state_dict to new module names for public MultiFlow compatibility.
        Returns:
            merged config, which must be a DictConfig, not Config dataclass, to handle new keys from MultiFlow
            ckpt_path, which may be the same as input, or a new path if the checkpoint was modified.

        Maintain some important behavior configuration specified in `cfg`, i.e. for the current run,
        and avoid being overwritten by however the checkpoint was configured.

        To support using Multiflow, we need to:
        1) Rename the keys in the state_dict to match new network module names.
        We attempt to map the state_dict to the new module names, and save a new checkpoint, which we then load.
        This requires that the model architecture matches MultiFlow, i.e. the same modules and shapes.

        2) Support merging a YAML-style config with a structured config.
        Since structured configs are dataclasses, and the MultiFlow config has some different fields,
        we need to merge into a DictConfig. So, in this file, we lose the "type-safety" of the structured config.

        We may have to deprecate this backwards compatibility in the future,
        but it's nice to support inference from the public checkpoint.
        """
        if not ckpt_path.endswith(".ckpt"):
            raise ValueError(
                f"Invalid checkpoint path {ckpt_path}, should end with .ckpt"
            )
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist."

        ckpt_dir = os.path.dirname(ckpt_path)
        log.info(f"Loading checkpoint from {ckpt_dir}")
        ckpt_cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))

        # Use `set_struct` to prevent creation of fields not defined in the configs
        OmegaConf.set_struct(OmegaConf.structured(cfg), False)
        OmegaConf.set_struct(ckpt_cfg, False)

        # TODO - better support for merging structured configs
        #   To merge into a dataclass, all fields must exist. Or we merge into a base DictConfig.
        #   We should merge `ckpt_cfg` into `cfg`.
        #   However, some fields, which determine behavior (like run `local`), should not be overridden.
        #   Note that interpolations don't apply here, since we are creating a dict (~already interpolated).
        merged_cfg = OmegaConf.merge(ckpt_cfg, cfg, ckpt_cfg)

        # Overwrite certain fields from the checkpoint config
        # Model size etc. should match, but explicitly override some fields with specified config.
        # TODO - we may need to add more here
        # inference, folding validation take priority
        merged_cfg.inference = OmegaConf.merge(merged_cfg.inference, cfg.inference)
        merged_cfg.folding = OmegaConf.merge(merged_cfg.folding, cfg.folding)
        # datasets not relevant, either sample lengths or `pdb_test` dataset
        # training + validation cfg
        merged_cfg.experiment.checkpointer.dirpath = cfg.experiment.checkpointer.dirpath
        merged_cfg.experiment.trainer.strategy = cfg.experiment.trainer.strategy

        # In an attempt (that probably won't last) to support MultiFlow,
        # if we got a config from MultiFlow, we need to map to our new module names.
        # We'll map, and save a new checkpoint, and then load that checkpoint.
        if "interpolant" in ckpt_cfg and "twisting" in ckpt_cfg.interpolant:
            log.info("Mapping MultiFlow state dict")
            ckpt = torch.load(
                ckpt_path, map_location=torch.device("cpu"), weights_only=False
            )

            # Define new checkpoint directory
            ckpt_dir = f"{merged_cfg.experiment.checkpointer.dirpath}_mapped_{merged_cfg.shared.id}"
            ckpt_path = os.path.join(ckpt_dir, "mapped.ckpt")

            # Map modules in state_dict
            # Assumes that these modules are active in the network, i.e. network shape is the same as MultiFlow.
            state_dict = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            replacements = {
                "model.trunk.": "model.attention_ipa_trunk.trunk.",
                "model.aatype_pred_net.": "model.aa_pred_net.aatype_pred_net.",
            }
            for key, value in state_dict.items():
                for old, new in replacements.items():
                    key = key.replace(old, new)
                new_state_dict[key] = value

            # Save new checkpoint
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt["state_dict"] = new_state_dict
            torch.save(ckpt, ckpt_path)
            log.info(f"Saved mapped checkpoint to {ckpt_path}")

        return merged_cfg, ckpt_path

    @property
    def dataloader(self):
        """
        Returns a dataloader singleton, i.e. with a fixed rng seed.
        Reset it with `reset_dataloader` for new sampling dataset.
        """
        if self._dataloader is not None:
            return self._dataloader

        if self.cfg.inference.task == InferenceTaskEnum.unconditional:
            eval_dataset = LengthSamplingDataset(self.cfg.inference.samples)
        elif (
            self.cfg.inference.task == InferenceTaskEnum.inpainting
            or self.cfg.inference.task == InferenceTaskEnum.forward_folding
            or self.cfg.inference.task == InferenceTaskEnum.inverse_folding
        ):
            # We want to use the input cfg inference settings for the pdb dataset,
            # not what was in the ckpt config
            # TODO - actually read the config, rather than using constructor to get new instance
            pdb_test_cfg = self._input_cfg.dataset.PDBPost2021()

            # The dataset will behave differently depending on the task
            # i.e. for inpainting, we generate motifs.
            dataset_constructor = DatasetConstructor.pdb_dataset(
                dataset_cfg=pdb_test_cfg,
                task=self.cfg.inference.task,
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

        # Due to cfg merge above, cfg may be a structured config (dataclass) or a DictConfig
        trainer_cfg = (
            asdict(self.cfg.experiment.trainer)
            if hasattr(self.cfg.experiment.trainer, "__dataclass_fields__")
            else self.cfg.experiment.trainer
        )
        self._trainer = Trainer(
            **trainer_cfg,
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
    # Read model checkpoint.
    log.info(f"Starting inference with {cfg.inference.num_gpus} GPUs")

    sampler = EvalRunner(cfg=cfg)
    sampler.run_sampling()

    if DDPInfo.from_env().rank == 0:
        sampler.compute_metrics()


if __name__ == "__main__":
    run()
