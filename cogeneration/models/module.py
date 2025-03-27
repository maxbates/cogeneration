import glob
import json
import logging
import os
import time
from collections import deque
from dataclasses import asdict, dataclass, fields
from random import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.distributed as dist
from lightning_utilities.core.apply_func import apply_to_collection
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger

import wandb
from cogeneration.config.base import Config, InferenceTaskEnum
from cogeneration.data import all_atom, so3_utils
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.batch_props import PredBatchProps as pbp
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.enum import MetricName, OutputFileName
from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.io import write_numpy_json
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.trajectory import SavedTrajectory, save_trajectory
from cogeneration.models import metrics
from cogeneration.models.model import FlowModel


def to_numpy(x: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if x is None:
        return None
    return x.detach().cpu().numpy()


@dataclass
class TrainingLosses:
    """Struct to collect losses from model training step."""

    trans_loss: torch.Tensor
    rots_vf_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    aatypes_loss: torch.Tensor
    train_loss: torch.Tensor  # aggregated loss

    def __getitem__(self, key: str) -> Any:
        # TODO - deprecate and use properties directly
        return getattr(self, key)

    def items(self):
        # avoid using asdict() because deepcopy on Tensors creates issues
        # instead, return iterater over fields
        return ((f.name, getattr(self, f.name)) for f in fields(self))


class FlowModule(LightningModule):
    def __init__(self, cfg: Config, folding_device_id: int = 0):
        super().__init__()
        self.cfg = cfg

        self.model = FlowModel(self.cfg.model)

        self.interpolant = Interpolant(self.cfg.interpolant)

        self.folding_validator = FoldingValidator(
            cfg=self.cfg.folding,
            device_id=folding_device_id,
        )

        # self.logger defined in LightningModule
        self._log = logging.getLogger(__name__)

        self.save_hyperparameters()

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
            **asdict(self.cfg.experiment.optimizer),
        )

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        """MPS doesn't support float64, so convert to float64 tensors to float32."""

        def convert_to_float32(tensor):
            if tensor.dtype == torch.float64:
                return tensor.to(torch.float32)
            return tensor

        # if we are on MPS, convert all tensors to float32
        # TODO - check the device itself, not the config
        if self.cfg.experiment.trainer.accelerator == "mps":
            batch = apply_to_collection(batch, torch.Tensor, convert_to_float32)

        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def on_train_start(self):
        self._epoch_start_time = time.time()

    # def on_after_backward(self):
    #     # debugging, check for unused parameters
    #     for name, param in self.named_parameters():
    #         if param.grad is None:
    #             print(f"Parameter {name} is unused (no gradient)")

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

    def model_step(self, noisy_batch: Any) -> TrainingLosses:
        training_cfg = self.cfg.experiment.training

        bb_mask = noisy_batch[bp.res_mask]
        loss_mask = bb_mask * noisy_batch[bp.diffuse_mask]
        if training_cfg.mask_plddt:
            loss_mask *= noisy_batch[bp.plddt_mask]
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")
        num_batch, num_res = loss_mask.shape
        loss_denom_num_res = torch.sum(loss_mask, dim=-1)
        batch_loss_mask = torch.any(bb_mask, dim=-1)

        # Number of backbone atoms to consider for loss
        # 3 for C-alpha, N, C atoms, 5 if we also consider psi angles
        n_bb_atoms = 5 if self.cfg.model.predict_psi_torsions else 3

        # Ground truth labels
        gt_trans_1 = noisy_batch[bp.trans_1]
        gt_rotmats_1 = noisy_batch[bp.rotmats_1]
        gt_aatypes_1 = noisy_batch[bp.aatypes_1]
        gt_psi_torsions_1 = noisy_batch[bp.torsion_angles_sin_cos_1][..., 2, :]
        gt_bb_atoms, atom37_mask, _, _ = all_atom.compute_backbone(
            all_atom.create_rigid(gt_rotmats_1, gt_trans_1),
            psi_torsions=gt_psi_torsions_1,
        )
        gt_bb_atoms = gt_bb_atoms[:, :, :n_bb_atoms]
        atom37_mask = atom37_mask[:, :, :n_bb_atoms]

        rotmats_t = noisy_batch[nbp.rotmats_t]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))

        # Timestep used for normalization.
        r3_t = noisy_batch[nbp.r3_t]  # (B, 1)
        so3_t = noisy_batch[nbp.so3_t]  # (B, 1)
        cat_t = noisy_batch[nbp.cat_t]  # (B, 1)

        # losses for each domain scaled at `1 - min(t, clip)` (i.e. higher as t -> 1)
        r3_norm_scale = 1 - torch.min(
            r3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )  # (B, 1, 1)
        so3_norm_scale = 1 - torch.min(
            so3_t[..., None], torch.tensor(training_cfg.t_normalize_clip)
        )  # (B, 1, 1)
        if training_cfg.aatypes_loss_use_likelihood_weighting:
            cat_norm_scale = 1 - torch.min(
                cat_t, torch.tensor(training_cfg.t_normalize_clip)
            )  # (B, 1)
            assert cat_norm_scale.shape == (num_batch, 1)
        else:
            cat_norm_scale = 1.0

        # Model output predictions
        # Basically, predict translations, rotations, and sequence and compare to ground truth.
        # Also compute the vector field, from the sampled `t` to predicted trans + rots.
        # To learn/refine vector field at t (for trans and rots) for init_rigids -> pred_rigids.
        model_output = self.model(noisy_batch)
        pred_trans_1 = model_output[pbp.pred_trans]
        pred_rotmats_1 = model_output[pbp.pred_rotmats]
        pred_psi_1 = model_output[pbp.pred_psi]  # might be None
        pred_logits = model_output[pbp.pred_logits]  # (B, N, aatype_pred_num_tokens)

        pred_rots_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        if torch.any(torch.isnan(pred_rots_vf)):
            raise ValueError("NaN encountered in pred_rots_vf")

        # aatypes loss
        ce_loss = (
            torch.nn.functional.cross_entropy(
                pred_logits.reshape(-1, self.cfg.model.aa_pred.aatype_pred_num_tokens),
                gt_aatypes_1.flatten().long(),
                reduction="none",
            ).reshape(num_batch, num_res)
            / cat_norm_scale
        )
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / loss_denom_num_res
        aatypes_loss *= training_cfg.aatypes_loss_weight

        # TODO - translation vector field loss
        #   See FoldFlow2
        #   compute vf from t -> gt t=1 vs t -> pred t=1

        # Translation loss
        trans_error = (
            (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        )
        trans_loss = (
            training_cfg.translation_loss_weight
            * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / (loss_denom_num_res * 3)
        )

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = (
            training_cfg.rotation_loss_weights
            * torch.sum(rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2))
            / (loss_denom_num_res * 3)
        )

        # TODO consider explicit rotation loss (see FoldFlow2)

        # TODO consider explicit `psi` loss?

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1, pred_psi_1)[
            :, :, :n_bb_atoms
        ]
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        bb_atom_loss_mask = atom37_mask * loss_mask[..., None]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * bb_atom_loss_mask[..., None],
            dim=(-1, -2, -3),
        ) / (bb_atom_loss_mask.sum(dim=(-1, -2)) + 1e-10)

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * n_bb_atoms, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * n_bb_atoms, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_bb_atoms))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * n_bb_atoms])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, n_bb_atoms))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * n_bb_atoms])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # No loss on anything >6A
        proximity_mask = gt_pair_dists < 6
        pair_dist_mask = pair_dist_mask * proximity_mask

        dist_mat_loss = torch.sum(
            (gt_pair_dists - pred_pair_dists) ** 2 * pair_dist_mask, dim=(1, 2)
        )
        dist_mat_loss /= torch.sum(pair_dist_mask, dim=(1, 2)) + 1

        # Auxiliary loss
        auxiliary_loss = (
            bb_atom_loss * training_cfg.aux_loss_use_bb_loss
            + dist_mat_loss * training_cfg.aux_loss_use_pair_loss
        )
        # TODO consider separate t threshold for dist loss vs bb atom loss, add to config
        auxiliary_loss *= (r3_t[:, 0] > training_cfg.aux_loss_t_pass) & (
            so3_t[:, 0] > training_cfg.aux_loss_t_pass
        )
        auxiliary_loss *= training_cfg.aux_loss_weight

        # clamp certain loss terms
        trans_loss = torch.clamp(trans_loss, max=5)
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        # Total loss
        train_loss = trans_loss + rots_vf_loss + auxiliary_loss + aatypes_loss

        if torch.any(torch.isnan(train_loss)):
            raise ValueError("NaN loss encountered")
        assert train_loss.shape == (num_batch,)

        losses = TrainingLosses(
            trans_loss=trans_loss,
            rots_vf_loss=rots_vf_loss,
            auxiliary_loss=auxiliary_loss,
            aatypes_loss=aatypes_loss,
            train_loss=train_loss,
        )

        def normalize_loss(x):
            return x.sum() / (batch_loss_mask.sum() + 1e-10)

        aux = {
            "batch_train_loss": train_loss,
            "batch_rot_loss": rots_vf_loss,
            "batch_trans_loss": trans_loss,
            "batch_bb_atom_loss": bb_atom_loss,
            "batch_dist_mat_loss": dist_mat_loss,
            # normalized
            "train_loss": normalize_loss(train_loss),
            "rots_vf_loss": normalize_loss(rots_vf_loss),
            "trans_loss": normalize_loss(trans_loss),
            "bb_atom_loss": normalize_loss(bb_atom_loss),
            "dist_mat_loss": normalize_loss(dist_mat_loss),
            # meta
            "loss_denom_num_res": loss_denom_num_res,
            "examples_per_step": torch.tensor(num_batch),
            "res_length": torch.mean(torch.sum(bb_mask, dim=-1).float()),
        }

        self._data_history.append(
            {
                "batch": noisy_batch,
                "loss": losses,
                "model_output": model_output,
                "aux": aux,
            }
        )

        return losses

    def training_step(self, batch: Any):
        step_start_time = time.time()

        self.interpolant.set_device(batch[bp.res_mask].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        # Enable self-conditioning during training
        if (
            self.cfg.interpolant.self_condition
            and random() > self.cfg.interpolant.self_condition_prob
        ):
            with torch.no_grad():
                # Perform a model pass, and get predicted translations and aatypes
                model_sc = self.model(noisy_batch)
                noisy_batch[nbp.trans_sc] = model_sc[pbp.pred_trans] * noisy_batch[
                    bp.diffuse_mask
                ][..., None] + noisy_batch[bp.trans_1] * (
                    1 - noisy_batch[bp.diffuse_mask][..., None]
                )
                logits_1 = torch.nn.functional.one_hot(
                    batch[bp.aatypes_1].long(),
                    num_classes=self.cfg.model.aa_pred.aatype_pred_num_tokens,
                ).float()
                noisy_batch[nbp.aatypes_sc] = model_sc[pbp.pred_logits] * noisy_batch[
                    bp.diffuse_mask
                ][..., None] + logits_1 * (1 - noisy_batch[bp.diffuse_mask][..., None])

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
            elif loss_name == "psi_loss":
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

        # log final loss explicitly (though it's already logged as `train_loss`)
        train_loss = total_losses["train_loss"]
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def validation_step(self, batch: Any, batch_idx: int) -> pd.DataFrame:
        assert batch is not None, "batch is None"

        res_mask = batch[bp.res_mask]
        num_batch, num_res = res_mask.shape
        csv_idx = batch[bp.csv_idx]
        diffuse_mask = batch[bp.diffuse_mask]
        assert (diffuse_mask == 1.0).all()  # no partial masking yet

        # validation mostly just runs "codesign" but will corrupt trans/rots/aa based on interpolant config.
        self.interpolant.set_device(res_mask.device)
        protein_traj, model_traj = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            task=InferenceTaskEnum.unconditional,
            # t=0 values will be noise
            trans_1=batch[bp.trans_1],
            rotmats_1=batch[bp.rotmats_1],
            aatypes_1=batch[bp.aatypes_1],
            diffuse_mask=diffuse_mask,
            chain_idx=batch[bp.chain_idx],
            res_idx=batch[bp.res_idx],
        )

        bb_trajs = to_numpy(protein_traj.structure)
        aa_trajs = to_numpy(protein_traj.amino_acids)

        model_bb_trajs = to_numpy(model_traj.structure)
        model_aa_trajs = to_numpy(model_traj.amino_acids)
        # For now, we only emit logits from direct model output (don't simulate logits)
        model_logits_trajs = to_numpy(model_traj.logits)

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

            # Compute metrics, inverse fold + fold
            top_sample_metrics, saved_trajectory_files = self.compute_sample_metrics(
                sample_id=sample_id,
                sample_dir=sample_dir,
                task=InferenceTaskEnum.unconditional,
                bb_traj=bb_trajs[i],
                aa_traj=aa_trajs[i],
                model_bb_traj=model_bb_trajs[i],
                model_aa_traj=model_aa_trajs[i],
                model_logits_traj=model_logits_trajs[i],
                true_bb_pos=None,  # codesign
                true_aa=None,  # codesign
                diffuse_mask=to_numpy(diffuse_mask)[0],
                aatypes_corrupt=self.cfg.interpolant.aatypes.corrupt,
                also_fold_pmpnn_seq=True,  # always fold during validation
                write_sample_trajectories=False,  # don't write trajectories during validation (just structures)
                n_inverse_folds=1,  # only one during validation
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

    def predict_step(self, batch: Any, batch_idx: Any) -> pd.DataFrame:
        # Create an inference-specific interpolant
        interpolant = Interpolant(self.cfg.inference.interpolant)

        device = (
            batch[bp.num_res].device
            if bp.num_res in batch
            else (
                batch[bp.trans_1].device
                if bp.trans_1 in batch
                else f"cuda:{torch.cuda.current_device()}"
            )
        )
        interpolant.set_device(device)

        # Handle single-sample and missing sample_id
        if bp.sample_id in batch:
            sample_ids = batch[bp.sample_id].squeeze().tolist()
        else:
            sample_ids = [0]
        sample_ids = [sample_ids] if isinstance(sample_ids, int) else sample_ids
        num_batch = len(sample_ids)

        if self.cfg.inference.task == InferenceTaskEnum.unconditional:
            sample_length = batch[bp.num_res].item()
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    f"length_{sample_length}",
                    f"sample_{str(sample_id)}",
                )
                for sample_id in sample_ids
            ]
            true_aatypes = true_bb_pos = None
            trans_1 = rotmats_1 = diffuse_mask = aatypes_1 = None
        elif self.cfg.inference.task == InferenceTaskEnum.forward_folding:
            sample_length = batch[bp.trans_1].shape[1]
            sample_dirs = [
                os.path.join(
                    self.inference_dir, f"length_{sample_length}", batch[bp.pdb_name][0]
                )
            ]
            for sample_dir in sample_dirs:
                os.makedirs(sample_dir, exist_ok=True)
            true_bb_pos = all_atom.atom37_from_trans_rot(
                batch[bp.trans_1], batch[bp.rotmats_1]
            )
            # Ensure only have one valid structure for the batch
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            true_bb_pos = true_bb_pos[0]
            # save the ground truth as a pdb
            write_prot_to_pdb(
                prot_pos=to_numpy(true_bb_pos),
                file_path=os.path.join(
                    sample_dirs[0], batch[bp.pdb_name][0] + "_gt.pdb"
                ),
                aatype=to_numpy(batch[bp.aatypes_1][0]),
            )
            aatypes_1 = batch[bp.aatypes_1]
            trans_1 = rotmats_1 = diffuse_mask = true_aatypes = None
        elif self.cfg.inference.task == InferenceTaskEnum.inverse_folding:
            sample_length = batch[bp.trans_1].shape[1]
            trans_1 = batch[bp.trans_1]
            rotmats_1 = batch[bp.rotmats_1]
            true_aatypes = batch[bp.aatypes_1]
            sample_dirs = [
                os.path.join(
                    self.inference_dir, f"length_{sample_length}", batch[bp.pdb_name][0]
                )
            ]
            aatypes_1 = diffuse_mask = true_bb_pos = None
        else:
            raise ValueError(f"Unknown task {self.cfg.inference.task}")

        # Skip runs if already exist
        top_sample_csv_paths = [
            os.path.join(sample_dir, "top_sample.csv") for sample_dir in sample_dirs
        ]
        if all(
            [
                os.path.exists(top_sample_csv_path)
                for top_sample_csv_path in top_sample_csv_paths
            ]
        ):
            self._log.error(
                f"instance {sample_ids} length {sample_length} already exists @ {sample_dirs}"
            )
            return

        # Sample batch
        protein_traj, model_traj = interpolant.sample(
            num_batch=num_batch,
            num_res=sample_length,
            model=self.model,
            task=self.cfg.inference.task,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            diffuse_mask=diffuse_mask,
            separate_t=self.cfg.inference.interpolant.codesign_separate_t,
        )
        diffuse_mask = (
            diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        )

        bb_trajs = to_numpy(protein_traj.structure)
        aa_trajs = to_numpy(protein_traj.amino_acids)

        model_bb_trajs = to_numpy(model_traj.structure)
        model_aa_trajs = to_numpy(model_traj.amino_acids)
        # For now, we only emit logits from direct model output (don't simulate logits)
        model_logits_trajs = to_numpy(model_traj.logits)

        # Check for remaining mask tokens in final step of interpolated trajectory and reset to 0 := alanine
        for i in range(aa_trajs.shape[0]):  # samples
            for j in range(aa_trajs.shape[2]):  # positions
                if aa_trajs[i, -1, j] == MASK_TOKEN_INDEX:
                    self._log.info("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0

        all_top_sample_metrics = []
        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            top_sample_metrics, _ = self.compute_sample_metrics(
                sample_id=sample_id,
                sample_dir=sample_dir,
                task=self.cfg.inference.task,
                bb_traj=bb_trajs[i],
                aa_traj=aa_trajs[i],
                model_bb_traj=model_bb_trajs[i],
                model_aa_traj=model_aa_trajs[i],
                model_logits_traj=model_logits_trajs[i],
                true_bb_pos=to_numpy(true_bb_pos),
                true_aa=to_numpy(true_aatypes),
                diffuse_mask=to_numpy(diffuse_mask)[0],
                aatypes_corrupt=self.cfg.inference.interpolant.aatypes.corrupt,
                also_fold_pmpnn_seq=self.cfg.inference.also_fold_pmpnn_seq,
                write_sample_trajectories=self.cfg.inference.write_sample_trajectories,
            )

            all_top_sample_metrics.append(top_sample_metrics)

        return pd.DataFrame(all_top_sample_metrics)

    def compute_sample_metrics(
        self,
        sample_id: Union[int, str],
        sample_dir: str,  # inference output directory for this sample
        task: InferenceTaskEnum,
        bb_traj: npt.NDArray,
        aa_traj: npt.NDArray,
        model_bb_traj: npt.NDArray,
        model_aa_traj: npt.NDArray,
        model_logits_traj: npt.NDArray,
        true_bb_pos: Optional[npt.NDArray],  # if relevant. None for unconditional.
        true_aa: Optional[npt.NDArray],  # if relevant. None for unconditional.
        diffuse_mask: npt.NDArray,
        aatypes_corrupt: bool = True,
        also_fold_pmpnn_seq: bool = True,
        write_sample_trajectories: bool = True,
        n_inverse_folds: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], SavedTrajectory]:
        """
        Takes a single sample.
        Saves trajectory. Runs self-consistency check running ProteinMPNN + folding to compute metrics.
        Returns metrics for the top sample.

        Note that this function expects numpy inputs. Tensors should be detached and converted.
        """
        # Noisy trajectory and model (clean) trajectory may not be the same number of steps.
        noisy_traj_length, sample_length, _, _ = bb_traj.shape
        assert bb_traj.shape == (
            noisy_traj_length,
            sample_length,
            37,
            3,
        ), f"bb_traj shape {bb_traj.shape}"
        if true_bb_pos is not None:
            assert true_bb_pos.shape == (
                sample_length,
                37,
                3,
            ), f"true_bb_pos shape {true_bb_pos.shape}"
        assert aa_traj.shape == (
            noisy_traj_length,
            sample_length,
        ), f"aa_traj shape {aa_traj.shape}"

        models_traj_length = model_bb_traj.shape[0]
        assert model_bb_traj.shape == (
            models_traj_length,
            sample_length,
            37,
            3,
        ), f"model_traj shape {model_bb_traj.shape}"
        assert model_aa_traj.shape == (
            models_traj_length,
            sample_length,
        ), f"model_aa_traj shape {model_aa_traj.shape}"
        if true_aa is not None:
            assert true_aa.shape == (1, sample_length), f"true_aa shape {true_aa.shape}"
            # unwrap single row
            true_aa = true_aa[0]

        os.makedirs(sample_dir, exist_ok=True)

        # Save PDBs, trajectories, and a fasta of the final sequence
        saved_trajectory_files = save_trajectory(
            sample_name=sample_id,
            sample=bb_traj[-1],
            bb_prot_traj=bb_traj,
            x0_traj=np.flip(model_bb_traj, axis=0),
            diffuse_mask=diffuse_mask,
            output_dir=sample_dir,
            aa_traj=aa_traj,
            model_aa_traj=model_aa_traj,
            model_logits_traj=model_logits_traj,
            write_trajectories=write_sample_trajectories,
        )

        # TODO - metrics about the trajectory?

        top_sample_metrics = self.folding_validator.assess_sample(
            sample_name=sample_id,
            sample_dir=sample_dir,
            pred_pdb_path=saved_trajectory_files.sample_pdb_path,
            pred_bb_positions=bb_traj[-1],
            pred_aa=aa_traj[-1],
            diffuse_mask=diffuse_mask,
            also_fold_pmpnn_seq=also_fold_pmpnn_seq,
            true_bb_positions=true_bb_pos,
            true_aa=true_aa,
            n_inverse_folds=n_inverse_folds,
        )

        # TODO - return other written file paths, not just trajectory

        return top_sample_metrics, saved_trajectory_files

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
