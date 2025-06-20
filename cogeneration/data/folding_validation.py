import glob
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from Bio import SeqIO

from cogeneration.config.base import FoldingConfig
from cogeneration.data import residue_constants
from cogeneration.data.const import CA_IDX, aatype_to_seq
from cogeneration.data.io import write_numpy_json
from cogeneration.data.metrics import calc_ca_ca_metrics, calc_mdtraj_metrics
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.protein_mpnn_runner import ProteinMPNNRunner
from cogeneration.data.residue_constants import restype_order_with_x, restypes_with_x
from cogeneration.data.superimposition import superimpose
from cogeneration.dataset.process_pdb import get_uncompressed_pdb_path, process_pdb_file
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.metrics import MetricName, OutputFileName
from cogeneration.type.task import InferenceTask

# NOTE - would be nice to better make 3rd party software injectable so easier to mock
# However, it is not hard to patch the relevant functions for now.


@dataclass
class SavedFoldingValidation:
    """
    Struct for the results of folding validation.
    Entries should be None if not written.
    """

    true_pdb: Optional[str] = None
    true_fasta: Optional[str] = None
    sample_fasta: Optional[str] = None
    inverse_folded_fasta: Optional[str] = None
    codesign_df: Optional[str] = None
    designability_df: Optional[str] = None
    top_sample_json: Optional[str] = None


@dataclass
class FoldingValidator:
    """
    Class to support folding and inverse folding with 3rd party tools.

    For now, just support ProteinMPNN and AF2
    """

    cfg: FoldingConfig
    device_id: Optional[int] = None

    def __post_init__(self):
        self.log = logging.getLogger(__name__)
        # Initialize ProteinMPNN runner once to avoid repeated instantiation
        self.protein_mpnn_runner = ProteinMPNNRunner(self.cfg.protein_mpnn)

    def set_device_id(self, device_id: int):
        self.device_id = device_id

    def assess_sample(
        self,
        task: InferenceTask,  # task type to determine which metrics to compute
        sample_name: Union[int, str],
        sample_dir: str,  # directory to write intermediates / outputs to
        pred_pdb_path: str,  # PDB file for predicted / generated structure, atom37.
        pred_bb_positions: npt.NDArray,  # (N, n_bb_atoms, 3) where n_bb_atoms in {3, 5}
        pred_aa: npt.NDArray,  # (N)
        diffuse_mask: npt.NDArray,  # (N)
        motif_mask: Optional[npt.NDArray],  # (N) [inpainting only]
        chain_idx: npt.NDArray,  # (N)
        res_idx: npt.NDArray,  # (N)
        true_bb_positions: Optional[npt.NDArray],  # (N, 37, 3), motifs for inpainting
        true_aa: Optional[npt.NDArray],  # (N)
        also_fold_pmpnn_seq: bool = True,  # also fold inverse-folded sequences
        n_inverse_folds: Optional[int] = None,
    ) -> Tuple[Dict[str, Any], SavedFoldingValidation]:
        """
        Entrypoint validation function.

        Takes a generated sample, i.e. structure and sequence.
        Performs inverse folding and forward folding, and generates metrics.
        Returns the top sample.

        If model is folding/inverse folding (not hallucinating), can pass `true_aa` and/or `true_bb_positions`.

        Generates metrics for:
        - structure, e.g. secondary structure
        - designability, i.e. if we can inverse fold and then re-fold and get the same backbone structure
        - self-consistency (if true aa is provided), i.e. if inverse folding generates similar sequences

        For inpainting, additionally computes sequence recovery + RMSD of fixed motifs

        Also generates several intermediate fastas and dataframes.
        """
        assert (
            pred_bb_positions.ndim == 3
        ), f"Invalid pred_bb_positions shape {pred_bb_positions.shape}. No batch dim."
        sample_length = pred_bb_positions.shape[0]

        # Check files / directories
        assert os.path.exists(
            pred_pdb_path
        ), f"Predicted PDB path {pred_pdb_path} does not exist"
        os.makedirs(sample_dir, exist_ok=True)
        inverse_folding_dir = os.path.join(sample_dir, "inverse_folding")
        folding_dir = os.path.join(sample_dir, "folding")
        assert not os.path.exists(
            inverse_folding_dir
        ), f"{inverse_folding_dir} already exists"
        assert not os.path.exists(folding_dir), f"{folding_dir} already exists"

        # Check inputs
        assert pred_bb_positions.shape == (
            sample_length,
            37,
            3,
        ), f"Invalid pred_bb_positions shape {pred_bb_positions.shape}"
        assert pred_aa.shape == (
            sample_length,
        ), f"Invalid pred_aa shape {pred_aa.shape}"

        if true_aa is not None:
            assert true_aa.shape == (
                sample_length,
            ), f"Invalid true_aa shape {true_aa.shape}"
        if true_bb_positions is not None:
            assert true_bb_positions.shape == (
                sample_length,
                37,
                3,
            ), f"Invalid true_bb_positions shape {true_bb_positions.shape}"
            assert (
                true_aa is not None
            ), "true_aa must be provided if true_bb_positions is provided"

        # Write PDB for true structure, if provided
        true_pdb_path = None
        if true_bb_positions is not None:
            true_pdb_path = os.path.join(sample_dir, OutputFileName.true_structure_pdb)
            write_prot_to_pdb(
                prot_pos=true_bb_positions,
                file_path=true_pdb_path,
                aatype=true_aa,
                chain_idx=chain_idx,
                res_idx=res_idx,
            )

        # Write fasta for true sequence, if provided
        true_fasta_path = None
        if true_aa is not None:
            true_fasta_path = os.path.join(sample_dir, OutputFileName.true_sequence_fa)
            true_aa_seq = "".join([restypes_with_x[x] for x in true_aa])
            with open(true_fasta_path, "w") as f:
                f.write(f">{sample_name}\n{true_aa_seq}\n")

        # Write fasta files for predicted sequences
        sample_fasta_path = os.path.join(sample_dir, OutputFileName.sample_sequence_fa)
        pred_aa_seq = "".join([restypes_with_x[x] for x in pred_aa])
        with open(sample_fasta_path, "w") as f:
            f.write(f">{sample_name}\n{pred_aa_seq}\n")

        # Assessment depends on the task:
        #
        # - Co-design / hallucination
        #   - Run inverse folding on the structure, assess sequence recovery
        #   - Run folding on the generated sequence, assess RMSD + confidence.
        #   - (if `also_fold_pmpnn_seq`), run folding on the inverse folded sequences, assess designability.
        #
        # - Inpainting
        #    Conditioned on motif-only `true_bb_positions` and `true_aa`
        #   - Run inverse folding on the structure, assess sequence recovery of motifs
        #   - Run folding on the generated sequence, assess RMSD + confidence.
        #   - Run folding on the inverse folded sequences, assess designability.
        #
        # - Inverse Folding:
        #    Conditioned on `true_bb_positions`. We predict a sequence. We have `true_aa` to compare.
        #    - Run folding on the sequence, assess RMSD + confidence.
        #    - Run inverse folding, assess sequence recovery
        #
        # - Forward Folding:
        #    Conditioned on `true_aa`. We predict a structure. We have `true_bb_positions` to compare.
        #    - Run folding on the sequence, assess RMSD + confidence.
        #    - Run inverse folding on the folded structure, assess sequence recovery
        #
        # Thus in all cases we run inverse folding using the backbone, and folding using the sequence.
        # Only when doing codesign is inverse-folding-then-folding optional, and still likely recommended.

        # `is_codesign` determines how we pick a top sample.
        # For codesign, we pick the best sample we generated.
        # For forward/inverse folding, we pick the most designable sample (i.e. inverse folded then folded).
        is_codesign = (
            task == InferenceTask.unconditional or task == InferenceTask.inpainting
        )

        # Sequence recovery helpers
        def _seq_str_to_np(seq: str) -> npt.NDArray:
            return np.array([restype_order_with_x[x] for x in seq])

        def _calc_seq_recovery(ref_seq: npt.NDArray, pred_seq: npt.NDArray) -> float:
            return (ref_seq == pred_seq).mean()

        def _calc_df_seq_recovery(
            df: pd.DataFrame, ref_seq: npt.NDArray, col: str = MetricName.sequence
        ) -> pd.Series:
            if col not in df.columns:
                raise KeyError(f"Column '{col}' not found in DataFrame: {df.columns}")
            return df[col].apply(
                lambda seq: _calc_seq_recovery(ref_seq, _seq_str_to_np(seq)),
            )

        # Run inverse folding
        inverse_folded_fasta_path = self.inverse_fold_structure(
            pdb_input_path=pred_pdb_path,
            diffuse_mask=diffuse_mask,
            output_dir=inverse_folding_dir,
            num_sequences=n_inverse_folds,
        )

        # Run folding on the generated sequence
        codesign_df = self.fold_fasta(
            fasta_path=sample_fasta_path,
            output_dir=folding_dir,
        )
        codesign_df = FoldingValidator.assess_folded_structures(
            sample_pdb_path=pred_pdb_path,
            pdb_name=sample_name,
            folded_df=codesign_df,
            true_bb_positions=true_bb_positions,
            motif_mask=motif_mask,
            task=task,
        )

        # Run folding on the inverse folded sequences
        designability_df = None
        if also_fold_pmpnn_seq or not is_codesign:
            designability_df = self.fold_fasta(
                fasta_path=inverse_folded_fasta_path,
                output_dir=folding_dir,
            )
            designability_df = FoldingValidator.assess_folded_structures(
                sample_pdb_path=pred_pdb_path,
                pdb_name=sample_name,
                folded_df=designability_df,
                true_bb_positions=true_bb_positions,
                motif_mask=motif_mask,
                task=task,
            )

            # Calculate sequence recovery for each inverse folded sequence
            designability_df[MetricName.inverse_folding_sequence_recovery_pred] = (
                _calc_df_seq_recovery(
                    df=designability_df,
                    ref_seq=pred_aa,
                )
            )

        # Calculate sequence-recovery
        # Skip for forward_folding, since sequence is fixed
        if true_aa is not None and task != InferenceTask.forward_folding:
            codesign_df[MetricName.inverse_folding_sequence_recovery_gt] = (
                _calc_seq_recovery(true_aa, pred_aa)
            )

            # Determine sequence recovery for Inverse Folded sequences to GT
            if designability_df is not None:
                designability_df[MetricName.inverse_folding_sequence_recovery_gt] = (
                    _calc_df_seq_recovery(
                        df=designability_df,
                        ref_seq=true_aa,
                    )
                )

            # For inpainting task, compute sequence recovery for fixed motifs
            if task == InferenceTask.inpainting:
                if motif_mask.any():
                    motif_sel = motif_mask.astype(bool)

                    # Calculate sequence recovery for fixed motifs
                    codesign_df[MetricName.motif_sequence_recovery] = (
                        _calc_seq_recovery(true_aa[motif_sel], pred_aa[motif_sel])
                    )

                    # Calculate sequence recovery for fixed motifs in inverse folded sequences
                    if designability_df is not None:
                        designability_df[
                            MetricName.motif_inverse_folding_sequence_recovery
                        ] = designability_df[MetricName.sequence].apply(
                            lambda seq: _calc_seq_recovery(
                                true_aa[motif_sel],
                                _seq_str_to_np(seq)[motif_sel],
                            )
                        )

        # Summarize designability data and include in folding_df
        if designability_df is not None:
            # counts
            codesign_df[MetricName.num_inverse_folded] = len(designability_df)
            codesign_df[MetricName.num_designable] = len(
                designability_df[designability_df[MetricName.is_designable]]
            )

            # inverse folding metrics
            codesign_df[MetricName.inverse_folding_sequence_recovery_mean] = (
                designability_df[
                    MetricName.inverse_folding_sequence_recovery_pred
                ].mean()
            )
            codesign_df[MetricName.inverse_folding_sequence_recovery_max] = (
                designability_df[
                    MetricName.inverse_folding_sequence_recovery_pred
                ].max()
            )

            # folding / designability metrics
            codesign_df[MetricName.inverse_folding_bb_rmsd_single_seq] = (
                designability_df[MetricName.bb_rmsd_folded].iloc[0]
            )
            codesign_df[MetricName.inverse_folding_bb_rmsd_min] = designability_df[
                MetricName.bb_rmsd_folded
            ].min()
            codesign_df[MetricName.inverse_folding_bb_rmsd_mean] = designability_df[
                MetricName.bb_rmsd_folded
            ].mean()

            # inpainting summary metrics
            if task == InferenceTask.inpainting:
                codesign_df[MetricName.inverse_folding_motif_sequence_recovery_mean] = (
                    designability_df[
                        MetricName.motif_inverse_folding_sequence_recovery
                    ].mean()
                )
                codesign_df[MetricName.inverse_folding_motif_bb_rmsd_mean] = (
                    designability_df[MetricName.motif_bb_rmsd_folded].mean()
                )

        # Assign some information to both dataframes
        for df in [codesign_df, designability_df]:
            if df is None:
                continue
            df[MetricName.sample_length] = sample_length
            df[MetricName.sample_id] = sample_name

        # Write the DataFrames
        codesign_df_path = os.path.join(sample_dir, OutputFileName.codesign_df)
        codesign_df.to_csv(codesign_df_path, index=False)
        designability_df_path = None
        if designability_df is not None:
            designability_df_path = os.path.join(
                sample_dir, OutputFileName.designability_df
            )
            designability_df.to_csv(designability_df_path, index=False)

        # Candidates for top samples are described above. If codesign, just use what was generated + folded.
        candidates_df = codesign_df if is_codesign else designability_df
        # Sort, in case we have multiple samples, by RMSD
        candidates_df = candidates_df.sort_values(
            MetricName.bb_rmsd_folded, ascending=False
        )
        # Select the top sample
        top_sample = candidates_df.iloc[0].to_dict()

        # Compute information about secondary structure and other metrics
        top_sample.update(
            calc_mdtraj_metrics(pdb_path=top_sample[MetricName.sample_pdb_path])
        )
        top_sample.update(
            calc_ca_ca_metrics(
                ca_pos=pred_bb_positions[:, CA_IDX],
                residue_index=res_idx,
            )
        )
        # TODO(inpainting) - calculate scaffold-specific metrics for secondary structure, clashes

        # write top sample JSON
        top_sample_path = os.path.join(sample_dir, OutputFileName.top_sample_json)
        write_numpy_json(top_sample_path, top_sample)

        # track validation files
        folding_validation_paths = SavedFoldingValidation(
            true_pdb=true_pdb_path,
            true_fasta=true_fasta_path,
            sample_fasta=sample_fasta_path,
            inverse_folded_fasta=inverse_folded_fasta_path,
            codesign_df=codesign_df_path,
            designability_df=designability_df_path,
            top_sample_json=top_sample_path,
        )

        return top_sample, folding_validation_paths

    def assess_all_top_samples(
        self,
        task: InferenceTask,
        top_samples_df: pd.DataFrame,
        output_dir: str,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Compute task specific summary metrics for all top samples, writes DataFrame to output_dir
        """
        metrics = {
            "Total Samples": len(top_samples_df),
        }

        # TODO(inpainting) - add inpainting specific metrics, perhaps specific file name

        if task == InferenceTask.unconditional or task == InferenceTask.inpainting:
            metrics_csv_path = os.path.join(
                output_dir, OutputFileName.designable_metrics_df
            )

            metrics.update(
                {
                    "Total codesignable": top_samples_df[
                        MetricName.is_designable
                    ].sum(),
                    "Percent codesignable": top_samples_df[
                        MetricName.is_designable
                    ].mean(),
                    "Average Inv Fold Consistency": top_samples_df[
                        MetricName.inverse_folding_sequence_recovery_mean
                    ].mean(),
                    "Average Inv Fold Best Consistency": top_samples_df[
                        MetricName.inverse_folding_sequence_recovery_max
                    ].max(),
                    "Single Seq Inv Fold Designability": top_samples_df[
                        MetricName.inverse_folding_bb_rmsd_single_seq
                    ].mean(),
                    "Top Seq Inv Fold Designability": top_samples_df[
                        MetricName.inverse_folding_bb_rmsd_min
                    ].mean(),
                }
            )

            # Add inpainting-specific metrics if task is inpainting
            if task == InferenceTask.inpainting:
                metrics.update(
                    {
                        "Average Motif Sequence Recovery": top_samples_df[
                            MetricName.motif_sequence_recovery
                        ].mean(),
                        "Average Motif Folded RMSD": top_samples_df[
                            MetricName.motif_bb_rmsd_folded
                        ].mean(),
                    }
                )

            # TODO(metric) - calculate diversity using FoldSeek, see MultiFlow.

        elif task == InferenceTask.forward_folding:
            metrics_csv_path = os.path.join(
                output_dir, OutputFileName.forward_fold_metrics_df
            )
            valid_fold = top_samples_df[MetricName.bb_rmsd_gt] <= 2.0
            metrics.update(
                {
                    "Total Match Ground Truth": valid_fold.sum(),
                    "Percent Match Ground Truth": valid_fold.mean(),
                    "Average Sample RMSD to Ground Truth": top_samples_df[
                        MetricName.bb_rmsd_gt
                    ].mean(),
                    "Average Folded RMSD to Ground Truth": top_samples_df[
                        MetricName.bb_rmsd_folded_gt
                    ].mean(),
                }
            )
        elif task == InferenceTask.inverse_folding:
            metrics_csv_path = os.path.join(
                output_dir, OutputFileName.inverse_fold_metrics_df
            )
            metrics.update(
                {
                    "Total Designable": top_samples_df[MetricName.is_designable].sum(),
                    "Percent Designable": top_samples_df[
                        MetricName.is_designable
                    ].mean(),
                    "Average RMSD to Sample": top_samples_df[
                        MetricName.bb_rmsd_folded
                    ].mean(),
                    "Average RMSD to Ground Truth": top_samples_df[
                        MetricName.bb_rmsd_gt
                    ].mean(),
                    "Average Sequence Recovery to Ground Truth": top_samples_df[
                        MetricName.inverse_folding_sequence_recovery_gt
                    ].mean(),
                    "Average Inv Fold RMSD": top_samples_df[
                        MetricName.inverse_folding_bb_rmsd_mean
                    ].mean(),
                    "Average Inv Fold Best RMSD": top_samples_df[
                        MetricName.inverse_folding_bb_rmsd_min
                    ].mean(),
                    "Average Inv Fold Sequence Recovery": top_samples_df[
                        MetricName.inverse_folding_sequence_recovery_mean
                    ].mean(),
                    "Average Inv Fold Best Sequence Recovery": top_samples_df[
                        MetricName.inverse_folding_sequence_recovery_max
                    ].mean(),
                }
            )
        else:
            raise ValueError(f"Unsupported task {task}")

        self.log.info(f"Summary metrics for task {task} -> {metrics_csv_path}")
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(metrics_csv_path, index=False)
        return metrics_df, metrics_csv_path

    def fold_fasta(self, fasta_path: str, output_dir: str) -> pd.DataFrame:
        """
        Fold sequences in a fasta file.

        Recommend `output_dir` to be empty,
        since content inside it will be removed to prevent AF2 artifact bloat.

        Returns a dataframe with `header`, `sequence`, `folded_pdb_path`, `mean_plddt`
        """
        if self.cfg.folding_model == "af2":
            folded_output = self._run_alphafold2(
                fasta_path=fasta_path, output_dir=output_dir
            )
        else:
            raise ValueError(f"Unsupported folding model: {self.cfg.folding_model}")
        return folded_output

    def inverse_fold_structure(
        self,
        pdb_input_path: str,
        diffuse_mask: Optional[npt.NDArray],
        output_dir: str,
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> str:
        """
        Generates and returns a fasta of inverse folded sequences using ProteinMPNN.
        The number of sequences is determined by cfg.
        """
        # TODO(inpainting-fixed) - support pass fixed residues to ProteinMPNN
        #    However, for inpainting, likely want to pass an empty mask,
        #    since some of the metrics check for sequence conservation of motifs.
        # assert diffuse_mask is None or (diffuse_mask == 1.0).all()

        assert os.path.exists(
            pdb_input_path
        ), f"PDB path {pdb_input_path} does not exist"

        uncompressed_pdb_path, is_temp_file = get_uncompressed_pdb_path(pdb_input_path)

        # Run ProteinMPNN
        fasta_path = self.protein_mpnn_runner.generate_fasta(
            pdb_path=Path(uncompressed_pdb_path),
            output_dir=Path(output_dir),
            device_id=self.device_id,
            num_sequences=num_sequences,
            seed=seed,
        )

        if is_temp_file:
            os.remove(uncompressed_pdb_path)

        return str(fasta_path)

    def _run_alphafold2(self, fasta_path: str, output_dir: str) -> pd.DataFrame:
        """
        Run ColabFold AF2 folding on a fasta file
        Returns a DataFrame describing the outputs, where `header` column is the sequence name.

        TODO(tools) make this static. It should delegate to a AlphaFold2Runner class, that can be mocked.
        """
        assert self.device_id is not None, "Device ID must be set for AF2 folding"
        assert (
            self.cfg.colabfold_path is not None
        ), "ColabFold path must be set for AF2 folding"
        assert os.path.exists(fasta_path), f"ColabFold path {fasta_path} does not exist"
        assert os.path.exists(
            self.cfg.colabfold_path
        ), f"ColabFold path {self.cfg.colabfold_path} does not exist"

        # Require an empty output directory, so we can safely clean up outputs.
        assert (
            not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0
        ), f"Output folding directory {output_dir} is not empty"
        os.makedirs(output_dir, exist_ok=True)

        # NOTE - public MultiFlow only runs model 4, but may want to consider running others.
        # TODO(tools) - pass device

        af2_args = [
            str(self.cfg.colabfold_path),
            fasta_path,
            output_dir,
            "--msa-mode",
            "single_sequence",
            "--num-models",
            "1",
            "--random-seed",
            str(self.cfg.protein_mpnn.pmpnn_seed),
            "--model-order",
            "4",
            "--num-recycle",
            "3",
            "--model-type",
            "alphafold2_ptm",
        ]
        process = subprocess.Popen(
            af2_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        code = process.wait()
        if code != 0:
            raise RuntimeError(
                f"AlphaFold2 run.py failed with code {code}. Args: {' '.join(af2_args)}"
            )

        fasta_seqs = SeqIO.to_dict(SeqIO.parse(fasta_path, "fasta"))
        all_af2_files = glob.glob(os.path.join(output_dir, "*"))
        af2_model_4_pdbs = {}
        af2_model_4_jsons = {}
        for x in all_af2_files:
            # Only keep the model_4 PDB and json, delete everything else.
            if "model_4" in x:
                seq_name = os.path.basename(x)
                if x.endswith(".json"):
                    seq_name = seq_name.split("_scores")[0]
                    af2_model_4_jsons[seq_name] = x
                if x.endswith(".pdb"):
                    seq_name = seq_name.split("_unrelaxed")[0]
                    af2_model_4_pdbs[seq_name] = x
            else:
                os.remove(x)

        all_fold_metrics = []
        for header, seq_record in fasta_seqs.items():
            af2_folded_path = af2_model_4_pdbs[header]
            af2_json_path = af2_model_4_jsons[header]
            with open(af2_json_path, "r") as f:
                fold_metrics = json.load(f)

            all_fold_metrics.append(
                {
                    MetricName.header: header,
                    MetricName.sequence: seq_record.seq,
                    MetricName.folded_pdb_path: af2_folded_path,
                    MetricName.plddt_mean: np.mean(fold_metrics["plddt"]),
                }
            )

        return pd.DataFrame(all_fold_metrics)

    @staticmethod
    def calc_backbone_rmsd(
        mask: Optional[Union[torch.Tensor, npt.NDArray]],  # (N)
        pos_1: npt.NDArray,  # (N, 3, 3)
        pos_2: npt.NDArray,  # (N, 3, 3)
    ) -> float:
        if mask is None:
            mask = np.ones(pos_1.shape[0], dtype=bool)

        # tolerate either Ca (1) or Ca-C-N (3) atoms
        assert pos_1.shape == pos_2.shape, f"pos_1 {pos_1.shape} != pos_2 {pos_2.shape}"
        assert pos_1.shape[1] in [
            1,
            3,
        ], f"pos_1 {pos_1.shape} must have 1 or 3 backbone atoms"

        if pos_1.shape[1] == 3:
            aligned_rmsd = superimpose(
                torch.tensor(pos_1).reshape(-1, 3)[None],  # (1, N*3, 3)
                torch.tensor(pos_2).reshape(-1, 3)[None],  # (1, N*3, 3)
                mask=torch.tensor(mask)[:, None].repeat(1, 3).reshape(-1),  # (1, N*3)
            )
        else:
            aligned_rmsd = superimpose(
                torch.tensor(pos_1)[None],  # (1, N, 3)
                torch.tensor(pos_2)[None],  # (1, N, 3)
                mask=torch.tensor(mask)[:, None].repeat(1, 1),  # (1, N)
            )

        return aligned_rmsd[1].item()

    @staticmethod
    def assess_folded_structures(
        sample_pdb_path: str,
        pdb_name: str,
        folded_df: pd.DataFrame,
        true_bb_positions: Optional[npt.NDArray] = None,  # (N, 37, 3)
        motif_mask: Optional[npt.NDArray] = None,  # (N) inpainting
        task: InferenceTask = InferenceTask.unconditional,  # influences which metrics to compute
    ) -> pd.DataFrame:
        """
        Calculate RMSD, pLDDT, and other metrics, comparing folded structures in `folded_df`
        to the sample structure, and ground truth structure if provided.

        `folded_df` can be either the single generated sample, or ProteinMPNN re-folds.

        For inpainting, also computes metrics specific to fixed motifs.

        Extends the input `folded_df` rows with these metrics and returns DataFrame.

        edited from `process_folded_outputs()` in public multiflow
        """
        sample_feats = process_pdb_file(
            pdb_file_path=sample_pdb_path,
            pdb_name=pdb_name,
        )
        sample_bb_pos = sample_feats[dpc.atom_positions][:, :3, :]  # (N, 3, 3)

        num_res = sample_bb_pos.shape[0]
        res_mask = torch.ones(num_res)

        # Require meaningful motif_mask for inpainting (e.g. for RMSD calculations)
        if motif_mask is not None:
            assert motif_mask.shape == res_mask.shape
            if not motif_mask.any():
                motif_mask = None

        # Calculate RMSD etc. for each folded structured in folded_df
        all_metrics = []
        for _, row in folded_df.iterrows():
            folded_feats = process_pdb_file(
                pdb_file_path=row[MetricName.folded_pdb_path],
                pdb_name=row[MetricName.header],
            )
            sample_metrics = {
                # Include the original row. Includes `header`, `folded_pdb_path`, `plddt_mean` from AF2.
                **row.to_dict(),
                # Associate with the generated sample
                MetricName.sequence: aatype_to_seq(folded_feats[dpc.aatype]),
                MetricName.sample_pdb_path: sample_pdb_path,
            }

            # Calculate RMSD to generated sample
            folded_bb_pos = folded_feats[dpc.atom_positions][:, :3, :]  # (N, 3, 3)
            bb_rmsd = FoldingValidator.calc_backbone_rmsd(
                res_mask, sample_bb_pos, folded_bb_pos
            )
            sample_metrics[MetricName.bb_rmsd_folded] = bb_rmsd
            sample_metrics[MetricName.is_designable] = bb_rmsd <= 2.0

            # Calculate RMSD for fixed motifs in folded structures
            if task == InferenceTask.inpainting and motif_mask is not None:
                sample_metrics[MetricName.motif_bb_rmsd_folded] = (
                    FoldingValidator.calc_backbone_rmsd(
                        motif_mask, sample_bb_pos, folded_bb_pos
                    )
                )

            # If provided ground truth bb positions, also compare to them
            if true_bb_positions is not None:
                # Ideally we could reshape once, outside the loop, but safer to access to row's sample length here
                assert true_bb_positions.shape == (
                    num_res,
                    37,
                    3,
                ), f"Invalid true_bb_positions shape {true_bb_positions.shape}, expected ({num_res}, 37, 3)"
                true_bb_pos = true_bb_positions[:, :3, :]  # (N, 3, 3)
                assert true_bb_pos.shape == (
                    num_res,
                    3,
                    3,
                ), f"Invalid true_bb_positions shape {true_bb_pos.shape}, expected ({num_res}, 3, 3)"
                assert true_bb_pos.shape == sample_bb_pos.shape
                assert true_bb_pos.shape == folded_bb_pos.shape

                sample_metrics[MetricName.bb_rmsd_gt] = (
                    FoldingValidator.calc_backbone_rmsd(
                        res_mask, sample_bb_pos, true_bb_pos
                    )
                )
                sample_metrics[MetricName.bb_rmsd_folded_gt] = (
                    FoldingValidator.calc_backbone_rmsd(
                        res_mask, folded_bb_pos, true_bb_pos
                    )
                )

                # Calculate RMSD to GT for fixed motifs in folded structures
                if task == InferenceTask.inpainting and motif_mask is not None:
                    sample_metrics[MetricName.motif_bb_rmsd_gt] = (
                        FoldingValidator.calc_backbone_rmsd(
                            motif_mask, sample_bb_pos, true_bb_pos
                        )
                    )
                    sample_metrics[MetricName.motif_bb_rmsd_folded_gt] = (
                        FoldingValidator.calc_backbone_rmsd(
                            motif_mask, folded_bb_pos, true_bb_pos
                        )
                    )

            all_metrics.append(sample_metrics)

        return pd.DataFrame(all_metrics)
