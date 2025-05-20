import glob
import json
import logging
import os
import time
import warnings
from collections import deque
from random import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.distributed as dist
from lightning_fabric.utilities.warnings import PossibleUserWarning
from lightning_utilities.core.apply_func import apply_to_collection
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from cogeneration.config.base import Config
from cogeneration.data import all_atom, metrics
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.folding_validation import (
    FoldingValidator,
    SavedFoldingValidation,
)
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import mask_blend_2d
from cogeneration.data.trajectory import SavedTrajectory, save_trajectory
from cogeneration.models.loss_calculator import (
    AuxiliaryMetrics,
    LossCalculator,
    TrainingLosses,
)
from cogeneration.models.model import FlowModel
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import InferenceFeatures
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp
from cogeneration.type.metrics import MetricName, OutputFileName
from cogeneration.type.task import DataTask, InferenceTask


def to_numpy(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x.detach().cpu().numpy()


class FlowModule(LightningModule):
    def __init__(self, cfg: Config, folding_device_id: int = 0):
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters("cfg")

        self.model = FlowModel(self.cfg.model)

        self.interpolant = Interpolant(self.cfg.interpolant)

        self.folding_validator = FoldingValidator(
            cfg=self.cfg.folding,
            device_id=folding_device_id,
        )

        # self.logger defined in LightningModule
        self._log = logging.getLogger(__name__)

        self._epoch_start_time = None
        # metrics generated during validation_step()
        self.validation_epoch_metrics: List[pd.DataFrame] = []
        # sample information tracked during validation_step()
        self.validation_epoch_samples: List[Tuple[str, int, wandb.Molecule]] = []

        self._checkpoint_dir = None
        self._inference_dir = None

        # batch / state tracking
        self._data_history = deque(maxlen=100)

    @property
    def checkpoint_dir(self) -> str:
        """
        directory for validation samples
        """
        if self._checkpoint_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    checkpoint_dir = [self.cfg.experiment.checkpointer.dirpath]
                else:
                    checkpoint_dir = [None]
                dist.broadcast_object_list(checkpoint_dir, src=0)
                checkpoint_dir = checkpoint_dir[0]
            else:
                checkpoint_dir = self.cfg.experiment.checkpointer.dirpath

            self._checkpoint_dir = checkpoint_dir
            os.makedirs(self._checkpoint_dir, exist_ok=True)

        return self._checkpoint_dir

    @property
    def inference_dir(self) -> str:
        if self._inference_dir is None:
            if dist.is_initialized():
                if dist.get_rank() == 0:
                    inference_dir = [self.cfg.inference.predict_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self.cfg.inference.predict_dir
                if inference_dir is None:
                    inference_dir = os.path.join(self.checkpoint_dir, "inference")

            assert (
                inference_dir is not None and len(inference_dir) > 0
            ), f"Invalid inference dir {inference_dir}"
            self._inference_dir = inference_dir
            os.makedirs(self._inference_dir, exist_ok=True)

        return self._inference_dir

    def configure_optimizers(self):
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self.cfg.experiment.optimizer.asdict(),
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """MPS doesn't support float64, so convert to float64 tensors to float32."""

        def convert_to_float32(tensor):
            if tensor.dtype == torch.float64:
                return tensor.to(torch.float32)
            return tensor

        # if we are on MPS, convert all tensors to float32
        # TODO - check the device itself, not the config (but should match)
        if self.cfg.experiment.trainer.accelerator == "mps":
            batch = apply_to_collection(batch, torch.Tensor, convert_to_float32)

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_train_start(self):
        self._epoch_start_time = time.time()

        # Log number of epochs
        module_id = self.cfg.shared.id
        epochs_done = getattr(self.trainer, "current_epoch", 0)
        epochs_total = getattr(self.trainer, "max_epochs", -1)

        # if total is missing or zero, we’re just starting fresh
        if epochs_total < 0:
            self._log.info(
                f"Module {module_id}: starting fresh, will run {epochs_total or 'unspecified'} epochs"
            )
        else:
            epochs_remaining = epochs_total - epochs_done
            self._log.info(
                f"Module {module_id}: resuming at epoch {epochs_done}/{epochs_total} ({epochs_remaining} left)"
            )

    def on_train_epoch_end(self):
        epoch_time = (time.time() - self._epoch_start_time) / 60.0
        self.log(
            "train/epoch_time_minutes",
            epoch_time,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self._epoch_start_time = time.time()

    def on_validation_epoch_end(self):
        # Log validation samples
        if len(self.validation_epoch_samples) > 0:
            if self.logger is not None and hasattr(self.logger, "log_table"):
                # wandb logging
                self.logger.log_table(
                    key="valid/samples",
                    columns=["sample_path", "global_step", "Protein"],
                    data=self.validation_epoch_samples,
                )
            self.validation_epoch_samples.clear()

        # Log validation metrics
        if len(self.validation_epoch_metrics) > 0:
            val_epoch_metrics = pd.concat(self.validation_epoch_metrics)

            # drop non-metrics columns, e.g. can't take `mean()`
            val_epoch_metrics = val_epoch_metrics.drop(
                columns=[
                    MetricName.sample_id,
                    MetricName.header,
                    MetricName.sequence,
                    MetricName.sample_pdb_path,
                    MetricName.folded_pdb_path,
                ]
            )

            for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f"valid/{metric_name}",
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                    sync_dist=True,
                    rank_zero_only=False,
                )
            self.validation_epoch_metrics.clear()

    def model_step(self, batch: NoisyFeatures) -> TrainingLosses:
        model_output = self.model(batch)

        loss_calculator = LossCalculator(
            cfg=self.cfg,
            batch=batch,
            pred=model_output,
        )
        losses, aux = loss_calculator.calculate()

        self._data_history.append(
            {
                "batch": batch,
                "loss": losses,
                "model_output": model_output,
                "aux": aux,
            }
        )

        return losses

    def training_step(self, batch: BatchFeatures):
        step_start_time = time.time()

        self.interpolant.set_device(batch[bp.res_mask].device)
        noisy_batch = self.interpolant.corrupt_batch(batch, task=self.cfg.data.task)

        # Enable self-conditioning
        if (
            self.cfg.interpolant.self_condition
            and random() > self.cfg.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                # Perform a model pass, and use predicted translations and aatypes for self-conditioning
                model_sc = self.model(noisy_batch)

                noisy_batch[nbp.trans_sc] = mask_blend_2d(
                    trans_t=model_sc[pbp.pred_trans],
                    trans_1=noisy_batch[bp.trans_1],
                    mask=noisy_batch[bp.diffuse_mask],
                )
                logits_1 = torch.nn.functional.one_hot(
                    batch[bp.aatypes_1].long(),
                    num_classes=self.cfg.model.aa_pred.aatype_pred_num_tokens,
                ).float()
                noisy_batch[nbp.aatypes_sc] = mask_blend_2d(
                    trans_t=model_sc[pbp.pred_logits],
                    trans_1=logits_1,
                    mask=noisy_batch[bp.diffuse_mask],
                )

        # Model pass, get losses
        batch_losses = self.model_step(noisy_batch)

        # Log the calculated and other losses we want to track

        num_batch = batch_losses.trans_loss.shape[0]
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # log t
        so3_t = torch.squeeze(noisy_batch[nbp.so3_t])
        self._log_scalar(
            "train/so3_t",
            np.mean(to_numpy(so3_t)),
            prog_bar=False,
            batch_size=num_batch,
        )
        r3_t = torch.squeeze(noisy_batch[nbp.r3_t])
        self._log_scalar(
            "train/r3_t", np.mean(to_numpy(r3_t)), prog_bar=False, batch_size=num_batch
        )
        cat_t = torch.squeeze(noisy_batch[nbp.cat_t])
        self._log_scalar(
            "train/cat_t",
            np.mean(to_numpy(cat_t)),
            prog_bar=False,
            batch_size=num_batch,
        )

        # Stratified losses across t.
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == "rots_vf_loss":
                batch_t = so3_t
            elif loss_name == "aatypes_loss":
                batch_t = cat_t
            elif loss_name == "trans_loss":
                batch_t = r3_t
            elif loss_name == "torsions_loss":
                # torsion angles are more impacted by rotation than translation (?)
                batch_t = so3_t
            elif loss_name == "auxiliary_loss":
                batch_t = r3_t
            else:
                continue

            stratified_losses = metrics.t_stratified_loss(
                batch_t, loss_dict, loss_name=loss_name
            )
            for k, v in stratified_losses.items():
                self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)

        # log training throughput
        self._log_scalar(
            "train/length",
            batch[bp.res_mask].shape[1],
            prog_bar=False,
            batch_size=num_batch,
        )
        self._log_scalar("train/batch_size", num_batch, prog_bar=False)
        step_time = time.time() - step_start_time
        self._log_scalar("train/examples_per_second", num_batch / step_time)

        # inpainting / scaffolding metrics
        # note that depending on `cfg.interpolant` some examples may be set to different subtasks.
        if self.cfg.data.task == DataTask.inpainting:
            motif_mask = batch[bp.motif_mask].float()
            scaffold_mask = 1 - motif_mask

            scaffold_percent = torch.mean(scaffold_mask).item()
            self._log_scalar(
                "train/scaffolding_percent",
                scaffold_percent,
                prog_bar=False,
                batch_size=num_batch,
            )
            num_motif_res = torch.sum(motif_mask, dim=-1)
            self._log_scalar(
                "train/motif_size",
                torch.mean(num_motif_res).item(),
                prog_bar=False,
                batch_size=num_batch,
            )

        # log final loss explicitly (though it's already logged as `train_loss`)
        train_loss = total_losses["train_loss"]
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def validation_step(self, batch: InferenceFeatures, batch_idx: int) -> pd.DataFrame:
        assert batch is not None, "batch is None"

        res_mask = batch[bp.res_mask]
        num_batch, num_res = res_mask.shape
        csv_idx = batch[bp.csv_idx]

        # Get inference task corresponding to training task
        inference_task = InferenceTask.from_data_task(task=self.cfg.data.task)

        # TODO(inpainting) consider running validation for inpainting and unconditional generation.
        #   However, don't jump between tasks - perhaps run both, if `prop` > 0.
        # Like training, some proportion of batches convert `inpainting` -> `unconditional`
        # if (
        #     inference_task == InferenceTask.inpainting and
        #     random() < self.cfg.interpolant.inpainting_unconditional_prop
        # ):
        #     inference_task = InferenceTask.unconditional

        # Validation can run either unconditional generation, or inpainting
        self.interpolant.set_device(res_mask.device)
        protein_traj, model_traj = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            task=inference_task,
            diffuse_mask=batch[bp.diffuse_mask],
            motif_mask=batch.get(bp.motif_mask, None),
            chain_idx=batch[bp.chain_idx],
            res_idx=batch[bp.res_idx],
            # t=0 values will be noise
            trans_1=batch[bp.trans_1],
            rotmats_1=batch[bp.rotmats_1],
            aatypes_1=batch[bp.aatypes_1],
            torsions_1=batch[bp.torsion_angles_sin_cos_1],
        )

        bb_trajs = to_numpy(protein_traj.structure)
        aa_trajs = to_numpy(protein_traj.amino_acids)

        model_bb_trajs = to_numpy(model_traj.structure)
        model_aa_trajs = to_numpy(model_traj.amino_acids)
        # For now, we only emit logits from direct model output (don't simulate logits)
        model_logits_trajs = to_numpy(model_traj.logits)

        # batch-level aatype metrics
        final_step = protein_traj[-1]
        generated_aatypes = to_numpy(final_step.amino_acids)
        assert generated_aatypes.shape == (num_batch, num_res)
        batch_level_aatype_metrics = metrics.calc_aatype_metrics(generated_aatypes)

        batch_metrics = []
        for i in range(num_batch):
            sample_id = f"sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}"
            sample_dir = os.path.join(
                self.checkpoint_dir,
                sample_id,
            )
            os.makedirs(sample_dir, exist_ok=True)

            sample_motif_mask = (
                batch[bp.motif_mask][i]
                if (bp.motif_mask in batch and batch[bp.motif_mask] is not None)
                else None
            )

            # Compute metrics, and inverse fold + fold the designed structure
            top_sample_metrics, saved_trajectory_files, saved_folding_validation = (
                self.compute_sample_metrics(
                    sample_id=sample_id,
                    sample_dir=sample_dir,
                    task=inference_task,
                    protein_structure_traj=bb_trajs[i],
                    protein_aa_traj=aa_trajs[i],
                    model_structure_traj=model_bb_trajs[i],
                    model_aa_traj=model_aa_trajs[i],
                    model_logits_traj=model_logits_trajs[i],
                    diffuse_mask=to_numpy(batch[bp.diffuse_mask][i]),
                    motif_mask=to_numpy(sample_motif_mask),
                    chain_idx=to_numpy(batch[bp.chain_idx][i]),
                    res_idx=to_numpy(batch[bp.res_idx][i]),
                    true_bb_pos=None,  # codesign  # TODO(inpainting) define
                    true_aa=None,  # codesign  # TODO(inpainting) define
                    also_fold_pmpnn_seq=True,  # always fold during validation
                    write_sample_trajectories=False,  # don't write trajectories during validation (just structures)
                    write_animations=False,
                    n_inverse_folds=1,  # only one during validation
                )
            )

            # save molecule to W&B
            if isinstance(self.logger, WandbLogger):
                pdb_path = saved_trajectory_files.sample_pdb_path
                assert pdb_path is not None
                self.validation_epoch_samples.append(
                    (pdb_path, self.global_step, wandb.Molecule(pdb_path))
                )

            top_sample_metrics.update(batch_level_aatype_metrics)
            batch_metrics.append(top_sample_metrics)

        batch_metrics_df = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics_df)

        return batch_metrics_df

    def predict_step(
        self, batch: InferenceFeatures, batch_idx: Any
    ) -> Optional[pd.DataFrame]:
        task = self.cfg.inference.task
        res_mask = batch[bp.res_mask]
        num_batch, sample_length = res_mask.shape

        # Create an inference-specific interpolant
        interpolant = Interpolant(self.cfg.inference.interpolant)
        interpolant.set_device(res_mask.device)

        # Pull out metadata and t=1 values, if defined
        trans_1 = batch.get(bp.trans_1, None)
        rotmats_1 = batch.get(bp.rotmats_1, None)
        torsions_1 = batch.get(bp.torsion_angles_sin_cos_1, None)
        aatypes_1 = batch.get(bp.aatypes_1, None)
        sample_pdb_name = batch.get(bp.pdb_name, [None])[0]

        diffuse_mask = batch[bp.diffuse_mask]
        motif_mask = batch.get(bp.motif_mask, None)

        # Handle single-sample and missing sample_id
        if bp.sample_id in batch:
            sample_ids = batch[bp.sample_id].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        assert num_batch == len(sample_ids)

        # Task-specific logic for creating output `sample_dirs`
        if task == InferenceTask.unconditional:
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    f"length_{sample_length}",
                    f"sample_{str(sample_id)}",
                )
                for sample_id in sample_ids
            ]
        elif task == InferenceTask.inpainting:
            # For inpainting, we take a known PDB, but may alter its scaffold lengths
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    sample_pdb_name,
                    f"{sample_length}_{str(sample_id)}",
                )
                for sample_id in sample_ids
            ]
        elif task == InferenceTask.forward_folding:
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    f"length_{sample_length}",
                    sample_pdb_name,
                )
            ]
        elif task == InferenceTask.inverse_folding:
            sample_dirs = [
                os.path.join(
                    self.inference_dir, f"length_{sample_length}", sample_pdb_name
                )
            ]
        else:
            raise ValueError(f"Unknown task {task}")

        # Create output directories
        for sample_dir in sample_dirs:
            os.makedirs(sample_dir, exist_ok=True)

        # For some tasks, we have a `true_aatypes` sequence for sample metrics
        # Note for inpainting, only the motifs are defined
        true_aatypes = None
        if (
            task == InferenceTask.inpainting
            or task == InferenceTask.forward_folding
            or task == InferenceTask.inverse_folding
        ):
            assert aatypes_1 is not None

            true_aatypes = aatypes_1

            # Ensure only have one valid sequence for the batch, extract it
            assert true_aatypes.shape == (
                1,
                sample_length,
            ), f"true_aa shape {true_aatypes.shape}"
            true_aatypes = true_aatypes[0]
            assert true_aatypes.shape == (sample_length,)

        # For some tasks, we have a `true_bb_pos` ground truth structure for sample metrics
        # Note for inpainting, `true_bb_pos` is masked and only defined at the motifs
        true_bb_pos = None
        if task == InferenceTask.inpainting or task == InferenceTask.forward_folding:
            assert trans_1 is not None
            assert rotmats_1 is not None
            assert torsions_1 is not None
            assert true_aatypes is not None

            # For inpainting, limit true structure to motif positions
            if task == InferenceTask.inpainting:
                true_mask = 1 - diffuse_mask
            else:
                true_mask = res_mask

            true_bb_pos = all_atom.atom37_from_trans_rot(
                trans=trans_1,
                rots=rotmats_1,
                torsions=torsions_1,
                res_mask=true_mask,
                aatype=true_aatypes,
                unknown_to_alanine=True,
            )

            # Ensure only have one valid structure for the batch, extract it
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            true_bb_pos = true_bb_pos[0]
            assert true_bb_pos.shape == (sample_length, 37, 3)

        # Skip runs if already exist
        top_sample_json_paths = [
            os.path.join(sample_dir, OutputFileName.top_sample_json)
            for sample_dir in sample_dirs
        ]
        if all([os.path.exists(path) for path in top_sample_json_paths]):
            self._log.error(
                f"instance {sample_ids} length {sample_length} already exists @ {sample_dirs}"
            )
            return

        # Prepare for sampling - zero out some values, depending on task
        if task == InferenceTask.unconditional:
            # Zero-out structure and sequence
            trans_1 = rotmats_1 = torsions_1 = aatypes_1 = None
        elif task == InferenceTask.inpainting:
            # Keep all t=1 values (though they may be masked to the motifs)
            pass
        elif task == InferenceTask.forward_folding:
            # Zero-out structure
            trans_1 = rotmats_1 = torsions_1 = None
        elif task == InferenceTask.inverse_folding:
            # Zero-out sequence
            aatypes_1 = None
        else:
            raise ValueError(f"Unknown task {task}")

        # Sample batch
        protein_traj, model_traj = interpolant.sample(
            num_batch=num_batch,
            num_res=sample_length,
            model=self.model,
            task=task,
            diffuse_mask=diffuse_mask,
            motif_mask=motif_mask,
            chain_idx=batch[bp.chain_idx],
            res_idx=batch[bp.res_idx],
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            torsions_1=torsions_1,
            aatypes_1=aatypes_1,
        )

        model_bb_trajs = to_numpy(model_traj.structure)
        model_aa_trajs = to_numpy(model_traj.amino_acids)
        model_logits_trajs = to_numpy(model_traj.logits)

        bb_trajs = to_numpy(protein_traj.structure)
        aa_trajs = to_numpy(protein_traj.amino_acids)
        # (We only emit logits from direct model output - don't simulate logits)

        # Check for remaining mask tokens in final step of interpolated trajectory and reset to 0 := alanine
        for i in range(aa_trajs.shape[0]):  # samples
            for j in range(aa_trajs.shape[2]):  # positions
                if aa_trajs[i, -1, j] == MASK_TOKEN_INDEX:
                    self._log.info("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0

        all_top_sample_metrics = []
        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            top_sample_metrics, _, _ = self.compute_sample_metrics(
                sample_id=sample_id,
                sample_dir=sample_dir,
                task=task,
                protein_structure_traj=bb_trajs[i],
                protein_aa_traj=aa_trajs[i],
                model_structure_traj=model_bb_trajs[i],
                model_aa_traj=model_aa_trajs[i],
                model_logits_traj=model_logits_trajs[i],
                diffuse_mask=to_numpy(diffuse_mask[i]),
                motif_mask=to_numpy(motif_mask[i]) if motif_mask is not None else None,
                chain_idx=to_numpy(batch[bp.chain_idx][i]),
                res_idx=to_numpy(batch[bp.res_idx][i]),
                true_bb_pos=to_numpy(true_bb_pos),
                true_aa=to_numpy(true_aatypes),
                also_fold_pmpnn_seq=self.cfg.inference.also_fold_pmpnn_seq,
                write_sample_trajectories=self.cfg.inference.write_sample_trajectories,
                write_animations=self.cfg.inference.write_animations,
                animation_max_frames=self.cfg.inference.animation_max_frames,
            )

            all_top_sample_metrics.append(top_sample_metrics)

        return pd.DataFrame(all_top_sample_metrics)

    def compute_sample_metrics(
        self,
        sample_id: Union[int, str],
        sample_dir: str,  # inference output directory for this sample
        task: InferenceTask,
        protein_structure_traj: npt.NDArray,
        protein_aa_traj: npt.NDArray,
        model_structure_traj: npt.NDArray,
        model_aa_traj: npt.NDArray,
        model_logits_traj: npt.NDArray,
        diffuse_mask: npt.NDArray,
        motif_mask: Optional[npt.NDArray],  # if relevant
        chain_idx: npt.NDArray,
        res_idx: npt.NDArray,
        true_bb_pos: Optional[npt.NDArray],  # if relevant
        true_aa: Optional[npt.NDArray],  # if relevant
        also_fold_pmpnn_seq: bool,
        write_sample_trajectories: bool,
        write_animations: bool,
        animation_max_frames: int = 50,
        n_inverse_folds: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], SavedTrajectory, SavedFoldingValidation]:
        """
        Takes a single sample.
        Saves trajectory. Runs self-consistency check running ProteinMPNN + folding to compute metrics.
        Returns metrics for the top sample.

        Note that this function expects numpy inputs. Tensors should be detached and converted.
        """
        # Noisy trajectory and model (clean) trajectory may not be the same number of steps.
        noisy_traj_length, sample_length, _, _ = protein_structure_traj.shape
        assert protein_structure_traj.shape == (
            noisy_traj_length,
            sample_length,
            37,
            3,
        ), f"bb_traj shape {protein_structure_traj.shape}"
        assert protein_aa_traj.shape == (
            noisy_traj_length,
            sample_length,
        ), f"aa_traj shape {protein_aa_traj.shape}"

        models_traj_length = model_structure_traj.shape[0]
        assert model_structure_traj.shape == (
            models_traj_length,
            sample_length,
            37,
            3,
        ), f"model_traj shape {model_structure_traj.shape}"
        assert model_aa_traj.shape == (
            models_traj_length,
            sample_length,
        ), f"model_aa_traj shape {model_aa_traj.shape}"

        if true_aa is not None:
            assert true_aa.shape == (sample_length,), f"true_aa shape {true_aa.shape}"
        if true_bb_pos is not None:
            assert true_bb_pos.shape == (
                sample_length,
                37,
                3,
            ), f"true_bb_pos shape {true_bb_pos.shape}"
            # require sequence if given structure, so can save accurately
            assert true_aa is not None, "true_aa required if true_bb_pos given"

        os.makedirs(sample_dir, exist_ok=True)

        # Save PDBs, trajectories, and a fasta of the final sequence
        saved_trajectory_files = save_trajectory(
            sample_name=sample_id,
            sample_atom37=protein_structure_traj[-1],
            protein_structure_traj=protein_structure_traj,
            model_structure_traj=model_structure_traj,
            diffuse_mask=diffuse_mask,
            motif_mask=motif_mask,
            chain_idx=chain_idx,
            res_idx=res_idx,
            output_dir=sample_dir,
            protein_aa_traj=protein_aa_traj,
            model_aa_traj=model_aa_traj,
            model_logits_traj=model_logits_traj,
            write_trajectories=write_sample_trajectories,
            write_animations=write_animations,
            animation_max_frames=animation_max_frames,
        )

        top_sample_metrics, folding_validation_paths = (
            self.folding_validator.assess_sample(
                task=task,
                sample_name=sample_id,
                sample_dir=sample_dir,
                pred_pdb_path=saved_trajectory_files.sample_pdb_path,
                pred_bb_positions=protein_structure_traj[-1],
                pred_aa=protein_aa_traj[-1],
                diffuse_mask=diffuse_mask,
                motif_mask=motif_mask,
                chain_idx=chain_idx,
                res_idx=res_idx,
                also_fold_pmpnn_seq=also_fold_pmpnn_seq,
                true_bb_positions=true_bb_pos,
                true_aa=true_aa,
                n_inverse_folds=n_inverse_folds,
            )
        )

        return top_sample_metrics, saved_trajectory_files, folding_validation_paths

    def concat_all_top_samples(
        self,
        output_dir: str,
        is_inference: bool = True,  # otherwise, validation
    ) -> Tuple[pd.DataFrame, str]:
        """
        Loads all top samples CSVs in `output_dir` and concatenates them into a single DataFrame.
        Writes `output_dir/all_top_samples.csv` and returns path.
        """
        assert os.path.exists(output_dir), f"Output dir {output_dir} does not exist"

        # "memoize"
        top_sample_csv_path = os.path.join(
            output_dir, OutputFileName.all_top_samples_df
        )
        if os.path.exists(top_sample_csv_path):
            self._log.info(f"Using existing top samples CSV {top_sample_csv_path}")
            return pd.read_csv(top_sample_csv_path), top_sample_csv_path

        # Inference: inference_dir/length/sample_name/top_sample.csv
        # Validation: checkpoint_dir/sample_name/top_sample.csv
        json_file_glob = (
            f"*/*/{OutputFileName.top_sample_json}"
            if is_inference
            else f"*/{OutputFileName.top_sample_json}"
        )

        # load and concat
        all_json_paths = glob.glob(
            os.path.join(output_dir, json_file_glob), recursive=True
        )
        assert len(all_json_paths) > 0, f"No top samples JSONs found in {output_dir}"

        def read_json(p):
            with open(p, "r") as f:
                return json.load(f)

        top_sample_df = pd.DataFrame([read_json(p) for p in all_json_paths])
        top_sample_df.to_csv(top_sample_csv_path, index=False)

        self._log.info(f"All top samples saved -> {top_sample_csv_path}")

        return top_sample_df, top_sample_csv_path

    def _log_scalar(
        self,
        key,
        value,
        on_step=True,
        on_epoch=False,
        prog_bar=True,
        batch_size=None,
        sync_dist=False,
        rank_zero_only=True,
    ):
        if sync_dist and rank_zero_only:
            raise ValueError("Unable to sync dist when rank_zero_only=True")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=PossibleUserWarning)
            self.log(
                key,
                value,
                on_step=on_step,
                on_epoch=on_epoch,
                prog_bar=prog_bar,
                batch_size=batch_size,
                sync_dist=sync_dist,
                rank_zero_only=rank_zero_only,
            )
