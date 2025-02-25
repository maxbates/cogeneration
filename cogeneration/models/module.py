import logging
import os
import time
from dataclasses import asdict, dataclass, fields
from random import random
from typing import Any, Dict, Optional, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.distributed as dist
import wandb
from lightning_utilities.core.apply_func import apply_to_collection
from omegaconf import OmegaConf
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers.wandb import WandbLogger

from cogeneration.config.base import Config, DataTaskEnum, InferenceTaskEnum
from cogeneration.data import all_atom, so3_utils
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.batch_props import PredBatchProps as pbp
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restypes, restypes_with_x
from cogeneration.data.trajectory import save_traj
from cogeneration.models import metrics
from cogeneration.models.model import FlowModel

to_numpy = lambda x: x.detach().cpu().numpy()


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
    def __init__(self, cfg: Config, folding_device_id=None):
        super().__init__()
        self.cfg = cfg

        self.model = FlowModel(cfg.model)

        self.interpolant = Interpolant(cfg.interpolant)

        # self.logger defined in LightningModule
        self._log = logging.getLogger(__name__)

        self.save_hyperparameters()

        self.validation_epoch_metrics = []
        self.validation_epoch_samples = []
        self.save_hyperparameters()

        self._epoch_start_time = None

        self._checkpoint_dir = None
        self._inference_dir = None

        # TODO - move folding / validation to its own class/module
        self._folding_model = None
        self._folding_cfg = cfg.folding
        self._folding_device_id = folding_device_id

    @property
    def checkpoint_dir(self) -> str:
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
                    inference_dir = [self.cfg.experiment.inference_dir]
                else:
                    inference_dir = [None]
                dist.broadcast_object_list(inference_dir, src=0)
                inference_dir = inference_dir[0]
            else:
                inference_dir = self.cfg.experiment.inference_dir
                if inference_dir is None:
                    inference_dir = os.path.join(self.checkpoint_dir, "inference")

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
            for metric_name, metric_val in val_epoch_metrics.mean().to_dict().items():
                self._log_scalar(
                    f"valid/{metric_name}",
                    metric_val,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                    batch_size=len(val_epoch_metrics),
                )
            self.validation_epoch_metrics.clear()

    def model_step(self, noisy_batch: Any) -> TrainingLosses:
        training_cfg = self.cfg.experiment.training

        loss_mask = noisy_batch[bp.res_mask] * noisy_batch[bp.diffuse_mask]
        if training_cfg.mask_plddt:
            loss_mask *= noisy_batch[bp.plddt_mask]
        if torch.any(torch.sum(loss_mask, dim=-1) < 1):
            raise ValueError("Empty batch encountered")
        loss_denom = torch.sum(loss_mask, dim=-1) * 3
        num_batch, num_res = loss_mask.shape

        # Ground truth labels
        gt_trans_1 = noisy_batch[bp.trans_1]
        gt_rotmats_1 = noisy_batch[bp.rotmats_1]
        gt_aatypes_1 = noisy_batch[bp.aatypes_1]
        rotmats_t = noisy_batch[nbp.rotmats_t]
        gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, gt_rotmats_1.type(torch.float32))
        gt_bb_atoms = all_atom.to_atom37(gt_trans_1, gt_rotmats_1)[:, :, :3]

        # Timestep used for normalization.
        r3_t = noisy_batch[nbp.r3_t]  # (B, 1)
        so3_t = noisy_batch[nbp.so3_t]  # (B, 1)
        cat_t = noisy_batch[nbp.cat_t]  # (B, 1)
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
        aatypes_loss = torch.sum(ce_loss * loss_mask, dim=-1) / (loss_denom / 3)
        aatypes_loss *= training_cfg.aatypes_loss_weight

        # Backbone atom loss
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        gt_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        pred_bb_atoms *= training_cfg.bb_atom_scale / r3_norm_scale[..., None]
        bb_atom_loss = (
            torch.sum(
                (gt_bb_atoms - pred_bb_atoms) ** 2 * loss_mask[..., None, None],
                dim=(-1, -2, -3),
            )
            / loss_denom
        )

        # TODO - translation vector field loss
        #   requires calculating and emitting from model. See FoldFlow2

        # Translation loss
        trans_error = (
            (gt_trans_1 - pred_trans_1) / r3_norm_scale * training_cfg.trans_scale
        )
        trans_loss = (
            training_cfg.translation_loss_weight
            * torch.sum(trans_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )
        trans_loss = torch.clamp(trans_loss, max=5)

        # Rotation VF loss
        rots_vf_error = (gt_rot_vf - pred_rots_vf) / so3_norm_scale
        rots_vf_loss = (
            training_cfg.rotation_loss_weights
            * torch.sum(rots_vf_error**2 * loss_mask[..., None], dim=(-1, -2))
            / loss_denom
        )

        # TODO consider explicit rotation loss (see FoldFlow2)

        # Pairwise distance loss
        gt_flat_atoms = gt_bb_atoms.reshape([num_batch, num_res * 3, 3])
        gt_pair_dists = torch.linalg.norm(
            gt_flat_atoms[:, :, None, :] - gt_flat_atoms[:, None, :, :], dim=-1
        )
        pred_flat_atoms = pred_bb_atoms.reshape([num_batch, num_res * 3, 3])
        pred_pair_dists = torch.linalg.norm(
            pred_flat_atoms[:, :, None, :] - pred_flat_atoms[:, None, :, :], dim=-1
        )

        flat_loss_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_loss_mask = flat_loss_mask.reshape([num_batch, num_res * 3])
        flat_res_mask = torch.tile(loss_mask[:, :, None], (1, 1, 3))
        flat_res_mask = flat_res_mask.reshape([num_batch, num_res * 3])

        gt_pair_dists = gt_pair_dists * flat_loss_mask[..., None]
        pred_pair_dists = pred_pair_dists * flat_loss_mask[..., None]
        pair_dist_mask = flat_loss_mask[..., None] * flat_res_mask[:, None, :]

        # TODO - enable proximity mask, or weighting, add distance + weight to cfg
        # # No loss on anything >6A
        # proximity_mask = gt_pair_dists < 6
        # pair_dist_mask = pair_dist_mask * proximity_mask

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
        auxiliary_loss = torch.clamp(auxiliary_loss, max=5)

        # Total loss
        # se3_vf_loss = trans_loss + rots_vf_loss
        train_loss = trans_loss + rots_vf_loss + auxiliary_loss + aatypes_loss

        if torch.any(torch.isnan(train_loss)):
            raise ValueError("NaN loss encountered")

        self._prev_batch = noisy_batch
        self._prev_loss_denom = loss_denom
        self._prev_loss = TrainingLosses(
            trans_loss=trans_loss,
            rots_vf_loss=rots_vf_loss,
            auxiliary_loss=auxiliary_loss,
            aatypes_loss=aatypes_loss,
            train_loss=train_loss,
        )
        return self._prev_loss

    def validation_step(self, batch: Any, batch_idx: int):
        # TODO - run ProteinMPNN + folding to determine designability, using a separate FoldingModule

        assert batch is not None, "batch is None"

        res_mask = batch[bp.res_mask]
        num_batch, num_res = res_mask.shape

        self.interpolant.set_device(res_mask.device)

        csv_idx = batch[bp.csv_idx]
        diffuse_mask = batch[bp.diffuse_mask]
        assert (
            diffuse_mask == 1.0
        ).all()  # TODO - support partial masking (inpainting)

        prot_traj, model_traj = self.interpolant.sample(
            num_batch,
            num_res,
            self.model,
            trans_1=batch[bp.trans_1],
            rotmats_1=batch[bp.rotmats_1],
            aatypes_1=batch[bp.aatypes_1],
            diffuse_mask=diffuse_mask,
            chain_idx=batch[bp.chain_idx],
            res_idx=batch[bp.res_idx],
        )
        samples = prot_traj[-1][0].numpy()
        assert samples.shape == (num_batch, num_res, 37, 3)
        # assert False, "need to separate aatypes from atom37_traj"

        generated_aatypes = prot_traj[-1][1].numpy()
        assert generated_aatypes.shape == (num_batch, num_res)
        batch_level_aatype_metrics = metrics.calc_aatype_metrics(generated_aatypes)

        batch_metrics = []
        for i in range(num_batch):
            sample_dir = os.path.join(
                self.checkpoint_dir,
                f"sample_{csv_idx[i].item()}_idx_{batch_idx}_len_{num_res}",
            )
            os.makedirs(sample_dir, exist_ok=True)

            # Write out sample to PDB file
            final_pos = samples[i]
            saved_path = write_prot_to_pdb(
                final_pos, os.path.join(sample_dir, "sample.pdb"), no_indexing=True
            )
            if isinstance(self.logger, WandbLogger):
                self.validation_epoch_samples.append(
                    [saved_path, self.global_step, wandb.Molecule(saved_path)]
                )

            # TODO - run designability with ProteinMPNN + folding, calculate metrics of sequence, structure
            #   it should be able to be hidden behind a config flag

            try:
                mdtraj_metrics = metrics.calc_mdtraj_metrics(saved_path)
                batch_metric = batch_level_aatype_metrics | mdtraj_metrics
                batch_metrics.append(batch_metric)
            except Exception as e:
                print(e)
                continue

        batch_metrics = pd.DataFrame(batch_metrics)
        self.validation_epoch_metrics.append(batch_metrics)

    def training_step(self, batch: Any):
        step_start_time = time.time()

        self.interpolant.set_device(batch[bp.res_mask].device)
        noisy_batch = self.interpolant.corrupt_batch(batch)

        # Enable self-conditioning during training
        if self.cfg.interpolant.self_condition and random() > 0.5:
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

        # TODO - move all loss terms to use struct rather than property accessors

        num_batch = batch_losses["trans_loss"].shape[0]
        total_losses = {k: torch.mean(v) for k, v in batch_losses.items()}

        # Log the calculated and other losses we want to track
        for k, v in total_losses.items():
            self._log_scalar(f"train/{k}", v, prog_bar=False, batch_size=num_batch)
        # Stratified across t.
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
        for loss_name, loss_dict in batch_losses.items():
            if loss_name == "rots_vf_loss":
                batch_t = so3_t
            elif loss_name == "train_loss":
                continue
            elif loss_name == "aatypes_loss":
                batch_t = cat_t
            else:
                batch_t = r3_t
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

        train_loss = total_losses["train_loss"]
        self._log_scalar("train/loss", train_loss, batch_size=num_batch)

        return train_loss

    def predict_step(self, batch: Any, batch_idx: Any) -> Dict[str, str]:
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
            true_bb_pos = None
            sample_dirs = [
                os.path.join(
                    self.inference_dir,
                    f"length_{sample_length}",
                    f"sample_{str(sample_id)}",
                )
                for sample_id in sample_ids
            ]
            trans_1 = rotmats_1 = diffuse_mask = aatypes_1 = true_aatypes = None
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
            assert true_bb_pos.shape == (1, sample_length, 37, 3)
            # save the ground truth as a pdb
            write_prot_to_pdb(
                prot_pos=to_numpy(true_bb_pos[0]),
                file_path=os.path.join(
                    sample_dirs[0], batch[bp.pdb_name][0] + "_gt.pdb"
                ),
                aatype=to_numpy(batch[bp.aatypes_1][0]),
            )
            true_bb_pos = true_bb_pos[..., :3, :].reshape(-1, 3).cpu().numpy()
            assert true_bb_pos.shape == (sample_length * 3, 3)
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
        prot_traj, model_traj = interpolant.sample(
            num_batch=num_batch,
            num_res=sample_length,
            model=self.model,
            trans_1=trans_1,
            rotmats_1=rotmats_1,
            aatypes_1=aatypes_1,
            diffuse_mask=diffuse_mask,
            forward_folding=self.cfg.inference.task
            == InferenceTaskEnum.forward_folding,
            inverse_folding=self.cfg.inference.task
            == InferenceTaskEnum.inverse_folding,
            separate_t=self.cfg.inference.interpolant.codesign_separate_t,
        )
        diffuse_mask = (
            diffuse_mask if diffuse_mask is not None else torch.ones(1, sample_length)
        )
        atom37_traj = [x[0] for x in prot_traj]
        atom37_model_traj = [x[0] for x in model_traj]

        bb_trajs = to_numpy(torch.stack(atom37_traj, dim=0).transpose(0, 1))
        noisy_traj_length = bb_trajs.shape[1]
        assert bb_trajs.shape == (num_batch, noisy_traj_length, sample_length, 37, 3)

        model_trajs = to_numpy(torch.stack(atom37_model_traj, dim=0).transpose(0, 1))
        clean_traj_length = model_trajs.shape[1]
        assert model_trajs.shape == (num_batch, clean_traj_length, sample_length, 37, 3)

        aa_traj = [x[1] for x in prot_traj]
        clean_aa_traj = [x[1] for x in model_traj]

        aa_trajs = to_numpy(torch.stack(aa_traj, dim=0).transpose(0, 1).long())
        assert aa_trajs.shape == (num_batch, noisy_traj_length, sample_length)
        # Check for remaining mask tokens in interpolated trajectory, reset if present (to 0 := alanine)
        for i in range(aa_trajs.shape[0]):
            for j in range(aa_trajs.shape[2]):
                if aa_trajs[i, -1, j] == MASK_TOKEN_INDEX:
                    self._log.info("WARNING mask in predicted AA")
                    aa_trajs[i, -1, j] = 0

        clean_aa_trajs = to_numpy(
            torch.stack(clean_aa_traj, dim=0).transpose(0, 1).long()
        )
        assert clean_aa_trajs.shape == (num_batch, clean_traj_length, sample_length)

        top_sample_paths = {}
        for i, sample_id in zip(range(num_batch), sample_ids):
            sample_dir = sample_dirs[i]
            top_sample_df = self.compute_sample_metrics(
                batch=batch,
                model_traj=model_trajs[i],
                bb_traj=bb_trajs[i],
                aa_traj=aa_trajs[i],
                clean_aa_traj=clean_aa_trajs[i],
                true_bb_pos=true_bb_pos,
                true_aa=true_aatypes,
                diffuse_mask=diffuse_mask,
                sample_id=sample_id,
                sample_length=sample_length,
                sample_dir=sample_dir,
                aatypes_corrupt=self.cfg.inference.interpolant.aatypes.corrupt,
                also_fold_pmpnn_seq=self.cfg.inference.also_fold_pmpnn_seq,
                write_sample_trajectories=self.cfg.inference.write_sample_trajectories,
            )
            top_sample_csv_path = os.path.join(sample_dir, "top_sample.csv")
            top_sample_df.to_csv(top_sample_csv_path, index=False)
            top_sample_paths[sample_id] = top_sample_csv_path

        return top_sample_paths

    def compute_sample_metrics(
        self,
        batch: Any,
        model_traj: npt.NDArray,
        bb_traj: npt.NDArray,
        aa_traj: npt.NDArray,
        clean_aa_traj: npt.NDArray,
        true_bb_pos: Optional[npt.NDArray],
        true_aa: Optional[npt.NDArray],
        diffuse_mask: torch.Tensor,
        sample_id: Union[int, str],
        sample_length: int,
        sample_dir: str,
        aatypes_corrupt: bool = True,
        also_fold_pmpnn_seq: bool = True,
        write_sample_trajectories: bool = True,
    ) -> pd.DataFrame:
        """
        Compute metrics - trajectory, folding validation, etc. - for a single sample.
        TODO - folding validation with ProteinMPNN and folding module
        """
        noisy_traj_length, sample_length, _, _ = bb_traj.shape
        clean_traj_length = model_traj.shape[0]
        assert bb_traj.shape == (noisy_traj_length, sample_length, 37, 3)
        assert model_traj.shape == (clean_traj_length, sample_length, 37, 3)
        assert aa_traj.shape == (noisy_traj_length, sample_length)
        assert clean_aa_traj.shape == (clean_traj_length, sample_length)

        if true_aa is not None:
            assert true_aa.shape == (1, sample_length)

        os.makedirs(sample_dir, exist_ok=True)

        # Save trajectory
        traj_paths = save_traj(
            bb_traj[-1],
            bb_traj,
            np.flip(model_traj, axis=0),
            to_numpy(diffuse_mask)[0],
            output_dir=sample_dir,
            aa_traj=aa_traj,
            clean_aa_traj=clean_aa_traj,
            write_trajectories=write_sample_trajectories,
        )

        # TODO - support ProteinMPNN + folding + designability + secondary structure metrics
        # all metrics should be defined by dataclass / enum
        # calculation of metrics should probably be its own class. Should also handle inference stats.

        # TODO - actually select the top sample to return
        top_sample_df = pd.DataFrame(
            {
                "sample_id": [sample_id],
                "length": [sample_length],
                "seq_codesign": "".join([restypes[x] for x in aa_traj[-1]]),
                "seq_true_aa": (
                    "".join([restypes_with_x[i] for i in true_aa[0]])
                    if true_aa is not None
                    else None
                ),
                "pdb_path": traj_paths["sample_path"],
            }
        )
        return top_sample_df

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
