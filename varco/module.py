import gc
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch import distributed as dist

from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.models.esm_ckpt_loading import plan_esm_warm_start_state_dict_load
from cogeneration.scripts.utils_ddp import DDPInfo
from cogeneration.type.metrics import MetricName, OutputFileName
from cogeneration.type.task import InferenceTask
from cogeneration.util.log import rank_zero_logger
from varco.config import VarcoConfig
from varco.data import DataBatch, DataCorrupted, ModelPrediction
from varco.interpolant import TreeInterpolant
from varco.loss import BranchFlowLossCalculator
from varco.model import BranchFlowModel
from varco.viz import BranchingFlowVisualizer

logger = rank_zero_logger(__name__)


class BranchFlowModule(pl.LightningModule):
    def __init__(self, cfg: VarcoConfig):
        super().__init__()

        self.cfg = cfg
        self.save_hyperparameters("cfg")
        self._log = rank_zero_logger(__name__)

        self.model = BranchFlowModel(cfg=self.cfg.model)
        self.loss_calculator = BranchFlowLossCalculator(cfg=self.cfg.loss)
        self.interpolant = TreeInterpolant(cfg=self.cfg.interpolant)

        # track EMA of training loss, hacky init at 10 to avoid nan handling
        self.register_buffer("_train_loss_ema", torch.tensor(10.0), persistent=False)

        self._folding_validator: Optional[FoldingValidator] = None
        self._val_top_samples: List[Dict[str, Any]] = []
        self._predict_top_samples: List[Dict[str, Any]] = []

    def _get_folding_validator(self) -> FoldingValidator:
        if self._folding_validator is None:
            self._folding_validator = FoldingValidator(
                cfg=self.cfg.folding, device="cpu"
            )
        return self._folding_validator

    def _gather_top_samples(
        self, local_top_samples: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        if not dist.is_available() or not dist.is_initialized():
            return local_top_samples

        gathered: List[Optional[List[Dict[str, Any]]]] = [
            None for _ in range(dist.get_world_size())
        ]
        dist.all_gather_object(gathered, local_top_samples)

        merged: List[Dict[str, Any]] = []
        for part in gathered:
            if part:
                merged.extend(part)
        return merged

    def on_validation_start(self) -> None:
        if self.cfg.inference.folding_validation.enabled:
            self._get_folding_validator().set_device_id(str(self.device))

    def on_predict_start(self) -> None:
        if self.cfg.inference.folding_validation.enabled:
            self._get_folding_validator().set_device_id(str(self.device))

    def on_validation_epoch_start(self) -> None:
        self._val_top_samples = []

    def on_predict_epoch_start(self) -> None:
        self._predict_top_samples = []

    def load_state_dict(self, state_dict, strict: bool = True):
        if strict:
            plan = plan_esm_warm_start_state_dict_load(
                checkpoint_state_dict=state_dict,
                current_state_dict=self.state_dict(),
                strict=True,
                allow_missing_esm_combiner_pair=True,
            )
            for note in plan.notes:
                self._log.info(note)
            strict = plan.strict

        return super().load_state_dict(state_dict, strict=strict)

    def load_cogeneration_weights(self, ckpt_path: str):
        """
        Load compatible weights from a cogeneration checkpoint.
        Copies specified modules if shapes match. Fails on shape mismatch.
        """
        logger.info(f"⚡️ Loading cogeneration weights from: {ckpt_path}")

        cogen_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)[
            "state_dict"
        ]
        varco_state = self.state_dict()

        # Modules to copy: cogen prefix -> varco prefix(es)
        module_map = {
            "model.esm_combiner.": "model.esm_combiner.",
            "model.edge_feature_net.": "model.edge_feature_net.",
            "model.trunk.": "model.trunk.",
            "model.ipa_trunk.": "model.ipa_trunk.",
            "model.seq_trunk.": "model.seq_trunk.",
            "model.aa_pred_net.": "model.aatype_pred.",
            "model.bfactor_net.": "model.bfactor_net.",
            "model.plddt_net.": "model.plddt_net.",
        }

        mapped = {}
        for cogen_key, value in cogen_state.items():
            # Skip frozen ESM (loaded separately)
            if ".esm_combiner.esm." in cogen_key:
                continue
            for cogen_prefix, varco_prefix in module_map.items():
                if cogen_key.startswith(cogen_prefix):
                    suffix = cogen_key[len(cogen_prefix) :]
                    targets = (
                        [varco_prefix]
                        if isinstance(varco_prefix, str)
                        else varco_prefix
                    )
                    for t in targets:
                        varco_key = t + suffix
                        if varco_key in varco_state:
                            if varco_state[varco_key].shape != value.shape:
                                raise ValueError(
                                    f"Shape mismatch: {cogen_key} {value.shape} vs {varco_key} {varco_state[varco_key].shape}"
                                )
                            mapped[varco_key] = value
                    break

        self.load_state_dict(mapped, strict=False)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            params=self.model.parameters(),
            **self.cfg.experiment.optimizer.asdict(),
        )

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        """Override to time"""
        backward_start = time.perf_counter()
        optimizer.step(closure=optimizer_closure)
        backward_time = time.perf_counter() - backward_start
        self.log("t/backward_ms", backward_time * 1000)

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        return self.model(batch)

    def training_step(self, batch: DataBatch, batch_idx: int) -> torch.Tensor:
        self.interpolant.set_device(self.device)

        # Corruption (timed) runs on GPU, not in dataloader (CPU)
        corrupt_start = time.perf_counter()
        bridged, couplings = self.interpolant.corrupt_batch(batch=batch)
        corrupted = bridged.pack_present()
        corrupt_time = time.perf_counter() - corrupt_start

        # Forward pass
        forward_start = time.perf_counter()
        pred = self.forward(corrupted)
        forward_time = time.perf_counter() - forward_start

        # Loss and metrics
        loss_start = time.perf_counter()
        losses, metrics = self.loss_calculator.calculate(
            batch=corrupted, pred=pred, couplings=couplings, bridged=bridged
        )
        loss_time = time.perf_counter() - loss_start

        # bin t value (bucketed into 0.0-0.2, 0.2-0.4, ..., 0.8-1.0)
        mean_t = bridged.t.mean().detach().item()
        t_bin_idx = min(
            int(mean_t * 5), 4
        )  # 0-4 for bins [0.0-0.2), [0.2-0.4), ..., [0.8-1.0]
        t_bin_start = t_bin_idx * 0.2
        t_bin_end = t_bin_start + 0.2
        t_bin_key = f"{t_bin_start:.1f}-{t_bin_end:.1f}"

        # primary losses
        self.log("L/train", losses.total_loss, prog_bar=True)
        self.log("L/trans", losses.trans_loss)
        self.log("L/rot", losses.rot_vf_loss)
        self.log("L/cdist", losses.pairwise_loss)
        self.log("L/seq", losses.base_seq_loss)
        self.log("L/seq_prob", losses.base_seq_prob_loss)
        self.log("L/seq_tok", losses.base_seq_token_loss)
        self.log("L/seq_ins", losses.insertion_seq_loss)
        self.log("L/split", losses.split_token_loss)
        self.log("L/split_pooled", losses.split_pooled_loss)
        self.log("L/del", losses.del_loss)
        self.log("L/bfactor", losses.bfactor_loss)
        self.log("L/plddt", losses.plddt_loss)

        # EMA of training loss
        train_loss = losses.total_loss.detach().item()
        beta = float(self.cfg.experiment.train_loss_ema_beta)
        self._train_loss_ema = beta * self._train_loss_ema + (1.0 - beta) * train_loss
        self.log("L/train_ema", self._train_loss_ema, prog_bar=True)

        # aux metrics
        self.log("A/base_seq_ce", metrics.base_seq_ce)
        self.log("A/base_seq_acc", metrics.base_seq_acc)
        self.log("A/base_seq_ppl", torch.exp(metrics.base_seq_ce))
        self.log("A/insertion_seq_ce", metrics.insertion_seq_ce)
        self.log("A/insertion_target_entropy", metrics.insertion_target_entropy)
        self.log("A/insertion_ce_over_entropy", metrics.insertion_ce_over_entropy)
        self.log("A/insertion_ce_minus_entropy", metrics.insertion_ce_minus_entropy)
        self.log("A/insertion_seq_kl", metrics.insertion_seq_kl)
        self.log("A/trans_rmse_ang", metrics.trans_rmse_ang)
        self.log("A/trans_mae_ang", metrics.trans_mae_ang)
        self.log("A/rot_mae_deg", metrics.rot_mae_deg)
        self.log("A/rot_rmse_deg", metrics.rot_rmse_deg)
        self.log("A/split_event_ce", metrics.split_event_ce)
        self.log("A/split_event_precision", metrics.split_event_precision)
        self.log("A/split_event_recall", metrics.split_event_recall)
        self.log("A/split_event_f1", metrics.split_event_f1)
        self.log("A/split_event_auprc", metrics.split_event_auprc)
        self.log("A/split_event_pos_rate", metrics.split_event_pos_rate)
        self.log("A/split_rate_mae", metrics.split_rate_mae)
        self.log("A/split_rate_mae_pos", metrics.split_rate_mae_pos)
        self.log("A/split_rate_corr", metrics.split_rate_corr)
        self.log("A/del_event_ce", metrics.del_event_ce)
        self.log("A/del_event_precision", metrics.del_event_precision)
        self.log("A/del_event_recall", metrics.del_event_recall)
        self.log("A/del_event_f1", metrics.del_event_f1)
        self.log("A/del_event_auprc", metrics.del_event_auprc)
        self.log("A/del_event_pos_rate", metrics.del_event_pos_rate)
        self.log("A/del_prob_mean", metrics.del_prob_mean)
        self.log("A/del_true_rate", metrics.del_true_rate)
        self.log("A/del_brier", metrics.del_brier)
        self.log("A/lddt_mean", metrics.lddt_mean)
        self.log("A/plddt_ce", metrics.plddt_ce)
        self.log("A/plddt_bin_acc", metrics.plddt_bin_acc)
        self.log("A/plddt_bin_acc_pm1", metrics.plddt_bin_acc_pm1)
        self.log("A/plddt_bin_mae", metrics.plddt_bin_mae)

        # t-stratified losses for primary losses
        self.log(f"L_t/trans_t{t_bin_key}", losses.trans_loss)
        self.log(f"L_t/rot_t{t_bin_key}", losses.rot_vf_loss)
        self.log(f"L_t/seq_t{t_bin_key}", losses.base_seq_loss)

        # t-stratified metrics
        self.log(f"A_t/trans_rmse_ang_t{t_bin_key}", metrics.trans_rmse_ang)
        self.log(f"A_t/rot_rmse_deg_t{t_bin_key}", metrics.rot_rmse_deg)
        self.log(f"A_t/base_seq_acc_t{t_bin_key}", metrics.base_seq_acc)
        self.log(f"A_t/insertion_seq_kl_t{t_bin_key}", metrics.insertion_seq_kl)
        self.log(f"A_t/split_rate_mae_t{t_bin_key}", metrics.split_rate_mae)
        self.log(f"A_t/split_event_auprc_t{t_bin_key}", metrics.split_event_auprc)
        self.log(f"A_t/del_event_auprc_t{t_bin_key}", metrics.del_event_auprc)
        self.log(f"A_t/lddt_mean_t{t_bin_key}", metrics.lddt_mean)

        # Timing statistics
        batch_size = corrupted.trans_t.shape[0]
        self.log("t/t", bridged.t.mean())
        self.log("t/batch_size", float(batch_size))
        # skip startup noise
        if batch_idx > 3:
            self.log("t/forward", forward_time * 1000)
            self.log("t/loss", loss_time * 1000)
        # Corruption time as function of batch size / t
        self.log("t/corrupt_ms_per_batch", corrupt_time * 1000 / batch_size)
        self.log(f"t/corrupt_ms_t{t_bin_key}", corrupt_time * 1000)

        # MPS clean up
        if batch_idx % 100 == 0 and torch.backends.mps.is_available():
            alloc = torch.mps.current_allocated_memory() / 1e9
            drv = torch.mps.driver_allocated_memory() / 1e9
            logger.debug(f"step {batch_idx} mps alloc={alloc:.2f}GB driver={drv:.2f}GB")
            gc.collect()
            torch.mps.empty_cache()

        return losses.total_loss

    def validation_step(self, batch: DataBatch, batch_idx: int) -> None:
        self.interpolant.set_device(self.device)
        sample_name = f"val_sample_{batch_idx}"

        sample_traj = self.interpolant.sample(
            model=self.model,
            data=batch,
        )

        viz = BranchingFlowVisualizer(sigma=1.0)
        val_dir = os.path.join(
            self.cfg.inference.predict_dir, "val", f"epoch{self.current_epoch:03d}"
        )
        viz.plot_sampling_trajectory(
            traj=sample_traj,
            out_dir=val_dir,
            filename=sample_name,
        )

        # run folding validation (refold/designability)for max_batches samples
        fold_val_cfg = self.cfg.inference.folding_validation
        if (
            fold_val_cfg.enabled
            and batch_idx < fold_val_cfg.max_batches
            and DDPInfo.from_env().local_rank == 0
        ):
            sample_dir = os.path.join(val_dir, sample_name)
            os.makedirs(sample_dir, exist_ok=True)

            final_sample = sample_traj.samples[-1]
            pred_atom37 = final_sample.to_atom37()[0].detach().cpu().numpy()
            pred_aa = final_sample.aatypes_t[0].detach().cpu().numpy()
            chain_idx = final_sample.chain_idx[0].detach().cpu().numpy()
            res_idx = np.arange(pred_aa.shape[0], dtype=np.int32)

            pred_pdb_path = os.path.join(sample_dir, OutputFileName.sample_pdb)
            write_prot_to_pdb(
                prot_pos=pred_atom37,
                file_path=pred_pdb_path,
                aatype=pred_aa,
                chain_idx=chain_idx,
                res_idx=res_idx,
                no_indexing=True,
                overwrite=True,
            )

            top_sample_metrics, _ = self._get_folding_validator().assess_sample(
                task=InferenceTask.unconditional,
                sample_name=sample_name,
                sample_dir=sample_dir,
                pred_pdb_path=pred_pdb_path,
                pred_bb_positions=pred_atom37,
                pred_aa=pred_aa,
                sample_aa_traj=np.expand_dims(pred_aa, axis=0),
                diffuse_mask=np.ones_like(pred_aa, dtype=np.int8),
                motif_mask=None,
                chain_idx=chain_idx,
                res_idx=res_idx,
                true_bb_positions=None,
                true_aa=None,
                inverse_fold=fold_val_cfg.assess_designability,
                also_fold_pmpnn_seq=fold_val_cfg.assess_designability,
                n_inverse_folds=self.cfg.folding.protein_mpnn.seq_per_sample,
            )

            self._val_top_samples.append(top_sample_metrics)
            if MetricName.plddt_mean in top_sample_metrics:
                self.log("val/plddt_mean", top_sample_metrics[MetricName.plddt_mean])
            if MetricName.bb_rmsd_folded in top_sample_metrics:
                self.log("val/bb_rmsd", top_sample_metrics[MetricName.bb_rmsd_folded])

    def predict_step(self, batch: DataBatch, batch_idx: int, dataloader_idx: int = 0):
        self.interpolant.set_device(self.device)

        sample_traj = self.interpolant.sample(
            model=self.model,
            data=batch,
        )

        rank = DDPInfo.from_env().rank
        sample_name = f"predict_rank{rank:03d}_idx{batch_idx:05d}"

        sample_dir = os.path.join(
            self.cfg.inference.predict_dir,
            self.cfg.inference.inference_subdir,
            sample_name,
        )
        os.makedirs(sample_dir, exist_ok=True)

        # write PDB
        final_sample = sample_traj.samples[-1]
        pred_atom37 = final_sample.to_atom37()[0].detach().cpu().numpy()
        pred_aa = final_sample.aatypes_t[0].detach().cpu().numpy()
        chain_idx = final_sample.chain_idx[0].detach().cpu().numpy()
        res_idx = np.arange(pred_aa.shape[0], dtype=np.int32)

        # write predicted PDB file
        pred_pdb_path = os.path.join(sample_dir, OutputFileName.sample_pdb)
        write_prot_to_pdb(
            prot_pos=pred_atom37,
            file_path=pred_pdb_path,
            aatype=pred_aa,
            chain_idx=chain_idx,
            res_idx=res_idx,
            no_indexing=True,
            overwrite=True,
        )

        # plot trajectory
        if self.cfg.inference.plot.enabled:
            viz = BranchingFlowVisualizer()
            viz.plot_sampling_trajectory(
                traj=sample_traj,
                out_dir=sample_dir,
                filename="sampling_trajectory",
                max_frames=self.cfg.inference.plot.max_frames,
                only_alpha_carbons=self.cfg.inference.plot.only_alpha_carbons,
                show_residue_letters=self.cfg.inference.plot.show_residue_letters,
                color_by=self.cfg.inference.plot.color_by,
            )

        # folding validation - either refolding, or designability
        fold_val_cfg = self.cfg.inference.folding_validation
        if fold_val_cfg.enabled and batch_idx < fold_val_cfg.max_batches:
            top_sample_metrics, _ = self._get_folding_validator().assess_sample(
                task=InferenceTask.unconditional,
                sample_name=sample_name,
                sample_dir=sample_dir,
                pred_pdb_path=pred_pdb_path,
                pred_bb_positions=pred_atom37,
                pred_aa=pred_aa,
                sample_aa_traj=np.expand_dims(pred_aa, axis=0),
                diffuse_mask=np.ones_like(pred_aa, dtype=np.int8),
                motif_mask=None,
                chain_idx=chain_idx,
                res_idx=res_idx,
                true_bb_positions=None,
                true_aa=None,
                inverse_fold=fold_val_cfg.assess_designability,
                also_fold_pmpnn_seq=fold_val_cfg.assess_designability,
                n_inverse_folds=self.cfg.folding.protein_mpnn.seq_per_sample,
            )
            self._predict_top_samples.append(top_sample_metrics)

    def on_validation_epoch_end(self) -> None:
        if not self.cfg.inference.folding_validation.enabled:
            return
        if DDPInfo.from_env().local_rank != 0:
            return

        all_top_samples = self._gather_top_samples(self._val_top_samples)
        if len(all_top_samples) == 0:
            return

        val_dir = os.path.join(
            self.cfg.inference.predict_dir, "val", f"epoch{self.current_epoch:03d}"
        )
        os.makedirs(val_dir, exist_ok=True)
        pd.DataFrame(all_top_samples).to_csv(
            os.path.join(val_dir, OutputFileName.all_top_samples_df), index=False
        )

    def on_predict_epoch_end(self) -> None:
        if not self.cfg.inference.folding_validation.enabled:
            return
        if DDPInfo.from_env().local_rank != 0:
            return

        all_top_samples = self._gather_top_samples(self._predict_top_samples)
        if len(all_top_samples) == 0:
            return

        predict_dir = os.path.join(self.cfg.inference.predict_dir, "predict")
        os.makedirs(predict_dir, exist_ok=True)
        pd.DataFrame(all_top_samples).to_csv(
            os.path.join(predict_dir, OutputFileName.all_top_samples_df), index=False
        )
