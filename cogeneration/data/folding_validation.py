import glob
import json
import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from Bio import PDB, SeqIO

from cogeneration.config.base import FoldingConfig, InferenceTaskEnum
from cogeneration.data import residue_constants
from cogeneration.data.const import aatype_to_seq
from cogeneration.data.enum import DatasetProteinColumns as dpc
from cogeneration.data.enum import MetricName, OutputFileName
from cogeneration.data.io import write_numpy_json
from cogeneration.data.residue_constants import restype_order_with_x, restypes_with_x
from cogeneration.data.superimposition import superimpose
from cogeneration.dataset.data_utils import parse_pdb_feats
from cogeneration.models.metrics import calc_ca_ca_metrics, calc_mdtraj_metrics

# NOTE - would be nice to better make 3rd party software injectable so easier to mock
# However, it is not hard to patch the relevant functions for now.


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

    def set_device_id(self, device_id: int):
        self.device_id = device_id

    def assess_sample(
        self,
        sample_name: Union[int, str],
        sample_dir: str,  # directory to write intermediates / outputs to
        pred_pdb_path: str,  # PDB file for predicted / generated structure, atom37.
        pred_bb_positions: npt.NDArray,  # (n_residues, n_bb_atoms, 3) where n_bb_atoms in [3, 5, ?]
        pred_aa: npt.NDArray,  # (n_residues)
        diffuse_mask: npt.NDArray,  # (n_residues)
        also_fold_pmpnn_seq: bool,  # if generating aa sequences (codesign), also fold inverse folded sequences
        true_bb_positions: Optional[npt.NDArray],  # (n_residues, 37, 3)
        true_aa: Optional[npt.NDArray] = None,  # (n_residues)
        n_inverse_folds: Optional[int] = None,
    ) -> Dict[str, Any]:
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

        Also generates several intermediate fastas and dataframes.
        TODO enumerate + describe file structure. Should return written files. merge with SavedTrajectory?
        TODO minimally consider returning designability_df
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
        assert (diffuse_mask == 1.0).all(), "Diffuse mask must be all 1.0 for now"
        assert pred_bb_positions.shape == (
            sample_length,
            37,
            3,
        ), f"Invalid pred_bb_positions shape {pred_bb_positions.shape}"
        assert pred_aa.shape == (
            sample_length,
        ), f"Invalid pred_aa shape {pred_aa.shape}"

        if true_bb_positions is not None:
            assert true_bb_positions.shape == (
                sample_length,
                37,
                3,
            ), f"Invalid true_bb_positions shape {true_bb_positions.shape}"
        if true_aa is not None:
            assert true_aa.shape == (
                sample_length,
            ), f"Invalid true_aa shape {true_aa.shape}"

        # There are a few things we might want to do, depending on what we have sampled.
        #
        # 1) Co-design / hallucination:
        #   - Run inverse folding on the structure, assess sequence recovery
        #   - Run folding on the generated sequence, assess RMSD + confidence.
        #   - (if `also_fold_pmpnn_seq`), run folding on the inverse folded sequences, assess designability.
        #
        # 2) Inverse Folding:
        #    Given `true_bb_positions`. We predict a sequence. We have `true_aa` to compare.
        #    - Run folding on the sequence, assess RMSD + confidence.
        #    - Run inverse folding, assess sequence recovery
        #
        # 3) Forward Folding:
        #    Given `true_aa`. We predict a structure. We have `true_bb_positions` to compare.
        #    - Run folding on the sequence, assess RMSD + confidence.
        #    - Run inverse folding on the structure, assess sequence recovery
        #
        # Thus in all cases we run inverse folding using the backbone, and folding using the sequence.
        # Only when doing codesign is inverse-folding-then-folding optional, and still likely recommended.
        #
        # We also pick a top sample.
        # In codesign, it's the sample we generated.
        # For forward/inverse folding, we take the inverse-folded-then-folded sample with the highest RMSD to true_bb.
        is_codesign = true_bb_positions is None and true_aa is None

        # Write fasta files for predicted and true (if provided) sequences
        sample_fasta_path = os.path.join(sample_dir, OutputFileName.sample_sequence_fa)
        pred_aa_seq = "".join([restypes_with_x[x] for x in pred_aa])
        with open(sample_fasta_path, "w") as f:
            f.write(f">{sample_name}\n")
            f.write(pred_aa_seq)

        if true_aa is not None:
            true_fasta_path = os.path.join(sample_dir, OutputFileName.true_sequence_fa)
            true_aa_seq = "".join([restypes_with_x[x] for x in true_aa])
            with open(true_fasta_path, "w") as f:
                f.write(f">{sample_name}\n")
                f.write(true_aa_seq)

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

        # Run folding on the original sequence
        codesign_df = self.fold_fasta(
            fasta_path=sample_fasta_path,
            output_dir=folding_dir,
        )
        codesign_df = FoldingValidator.assess_folded_structures(
            sample_pdb_path=pred_pdb_path,
            folded_df=codesign_df,
            true_bb_positions=true_bb_positions,
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
                folded_df=designability_df,
                true_bb_positions=true_bb_positions,
            )

            # Calculate sequence recovery for each inverse folded sequence
            designability_df[MetricName.inverse_folding_sequence_recovery_pred] = (
                _calc_df_seq_recovery(
                    df=designability_df,
                    ref_seq=pred_aa,
                )
            )

        # If we have the actual sequence (e.g. not hallucinating), assess sequence self-consistency
        if true_aa is not None:
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

        # Summarize designability data and include in folding_df
        if designability_df is not None:
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
                designability_df[MetricName.bb_rmsd].iloc[0]
            )
            codesign_df[MetricName.inverse_folding_bb_rmsd_min] = designability_df[
                MetricName.bb_rmsd
            ].min()
            codesign_df[MetricName.inverse_folding_bb_rmsd_mean] = designability_df[
                MetricName.bb_rmsd
            ].mean()
            codesign_df[MetricName.num_inverse_folded] = len(designability_df)
            codesign_df[MetricName.num_designable] = len(
                designability_df[designability_df[MetricName.is_designable]]
            )

        # Assign some information to both dataframes
        for df in [codesign_df, designability_df]:
            df[MetricName.sample_length] = sample_length
            df[MetricName.sample_id] = sample_name

        # Write the DataFrames
        codesign_df.to_csv(os.path.join(sample_dir, OutputFileName.codesign_df))
        if designability_df is not None:
            designability_df.to_csv(
                os.path.join(sample_dir, OutputFileName.designability_df)
            )

        # Candidates for top samples are described above. If codesign, just use what was generated + folded.
        candidates_df = codesign_df if is_codesign else designability_df
        # Sort, in case we have multiple samples, by RMSD
        candidates_df = candidates_df.sort_values(MetricName.bb_rmsd, ascending=False)
        # Select the top sample
        top_sample = candidates_df.iloc[0].to_dict()

        # Compute information about secondary structure and other metrics
        top_sample.update(calc_mdtraj_metrics(top_sample[MetricName.sample_pdb_path]))
        top_sample.update(
            calc_ca_ca_metrics(pred_bb_positions[:, residue_constants.atom_order["CA"]])
        )

        # write top sample JSON
        top_sample_path = os.path.join(sample_dir, OutputFileName.top_sample_json)
        write_numpy_json(top_sample_path, top_sample)

        return top_sample

    def assess_all_top_samples(
        self,
        task: InferenceTaskEnum,
        top_samples_df: pd.DataFrame,
        output_dir: str,
    ) -> Tuple[pd.DataFrame, str]:
        """
        Compute task specific summary metrics for all top samples, writes DataFrame to output_dir
        """
        # TODO - consider a new summary enum for these fields.

        if task == InferenceTaskEnum.unconditional:
            metrics_csv_path = os.path.join(
                output_dir, OutputFileName.designable_metrics_df
            )

            # TODO - calculate diversity using FoldSeek, see MultiFlow.

            metrics_df = pd.DataFrame(
                [
                    {
                        "Total Samples": len(top_samples_df),
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
                ]
            )
        elif task == InferenceTaskEnum.forward_folding:
            metrics_csv_path = os.path.join(
                output_dir, OutputFileName.forward_fold_metrics_df
            )
            valid_fold = top_samples_df[MetricName.bb_rmsd_gt] <= 2.0
            metrics_df = pd.DataFrame(
                [
                    {
                        "Total Samples": len(top_samples_df),
                        "Total Match Ground Truth": valid_fold.sum(),
                        "Percent Match Ground Truth": valid_fold.mean(),
                        "Average Sample RMSD to Ground Truth": top_samples_df[
                            MetricName.bb_rmsd_gt
                        ].mean(),
                        "Average Folded RMSD to Ground Truth": top_samples_df[
                            MetricName.bb_rmsd_folded_gt
                        ].mean(),
                    }
                ]
            )
        elif task == InferenceTaskEnum.inverse_folding:
            metrics_csv_path = os.path.join(
                output_dir, OutputFileName.inverse_fold_metrics_df
            )
            metrics_df = pd.DataFrame(
                [
                    {
                        "Total Samples": len(top_samples_df),
                        "Total Designable": top_samples_df[
                            MetricName.is_designable
                        ].sum(),
                        "Percent Designable": top_samples_df[
                            MetricName.is_designable
                        ].mean(),
                        "Average RMSD to Sample": top_samples_df[
                            MetricName.bb_rmsd
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
                ]
            )
        else:
            raise ValueError(f"Unsupported task {task}")

        self.log.info(f"Summary metrics for task {task} -> {metrics_csv_path}")
        metrics_df.to_csv(metrics_csv_path, index=False)
        return metrics_df, metrics_csv_path

    def fold_fasta(self, fasta_path: str, output_dir: str) -> pd.DataFrame:
        """
        Fold sequences in a fasta file.
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
        diffuse_mask: npt.NDArray,
        output_dir: str,
        num_sequences: Optional[int] = None,
    ) -> str:
        """
        Generates and returns a fasta of inverse folded sequences using ProteinMPNN.
        The number of sequences is determined by cfg.
        """
        assert (diffuse_mask == 1.0).all(), "Diffuse mask must be all 1.0 for now"

        if num_sequences is None:
            num_sequences = self.cfg.seq_per_sample

        modified_fasta_path = self._run_protein_mpnn(
            input_pdb_path=pdb_input_path,
            output_dir=output_dir,
            num_sequences=num_sequences,
            seed=self.cfg.pmpnn_seed,
            device_id=self.device_id,
        )
        return modified_fasta_path

    def _run_alphafold2(self, fasta_path: str, output_dir: str) -> pd.DataFrame:
        """
        Run ColabFold AF2 folding on a fasta file
        Returns a DataFrame describing the outputs, where `header` column is the sequence name.

        TODO make this static. It should delegate to a AlphaFold2Runner class, that can be mocked.
        TODO - rename columns, include more information (e.g. all the pLDDTs)
        """
        assert self.device_id is not None, "Device ID must be set for AF2 folding"
        assert (
            self.cfg.colabfold_path is not None
        ), "ColabFold path must be set for AF2 folding"
        assert os.path.exists(fasta_path), f"ColabFold path {fasta_path} does not exist"
        assert os.path.exists(
            self.cfg.colabfold_path
        ), f"ColabFold path {self.cfg.colabfold_path} does not exist"

        os.makedirs(output_dir, exist_ok=True)

        # NOTE - public MultiFlow only runs model 4, but may want to consider running others.

        af2_args = [
            self.cfg.colabfold_path,
            fasta_path,
            output_dir,
            "--msa-mode",
            "single_sequence",
            "--num-models",
            "1",
            "--random-seed",
            "123",
            "--device",
            f"{self.device_id}",
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
        _ = process.wait()

        fasta_seqs = SeqIO.to_dict(SeqIO.read(fasta_path, "fasta"))
        all_af2_files = glob.glob(os.path.join(output_dir, "*"))
        af2_model_4_pdbs = {}
        af2_model_4_jsons = {}
        for x in all_af2_files:
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
        for header, _ in fasta_seqs.items():
            af2_folded_path = af2_model_4_pdbs[header]
            af2_json_path = af2_model_4_jsons[header]
            with open(af2_json_path, "r") as f:
                fold_metrics = json.load(f)

            # TODO - include other metrics from AlphaFold

            all_fold_metrics.append(
                {
                    MetricName.header: header,
                    MetricName.folded_pdb_path: af2_folded_path,
                    MetricName.plddt_mean: np.mean(fold_metrics["plddt"]),
                }
            )

        return pd.DataFrame(all_fold_metrics)

    def _run_protein_mpnn(
        self,
        input_pdb_path: str,
        output_dir: str,
        num_sequences: int,
        seed: int,
        device_id: int,
    ) -> str:
        """
        Run ProteinMPNN inverse folding on a PDB file
        Returns a fasta with reasonably named sequences

        TODO make this static. It should delegate to a ProteinMPNNRunner class, that can be mocked.
        """
        assert device_id is not None, "Device ID must be set for PMPNN folding"
        assert os.path.exists(
            self.cfg.pmpnn_path
        ), f"PMPNN path {self.cfg.pmpnn_path} does not exist"
        assert os.path.exists(
            input_pdb_path
        ), f"PDB path {input_pdb_path} does not exist"

        os.makedirs(output_dir, exist_ok=True)

        # ProteinMPNN takes a PDB as input, and writes a fasta as output, per PDB.
        # The output `.fa` file has the same basename as the input `.pdb` file.

        # Prep, copy the input structure to the directory where Protein MPNN needs it
        pdb_file_name = os.path.basename(input_pdb_path)
        pdb_name = pdb_file_name.split(".")[0]
        output_pdb_path = os.path.join(output_dir, pdb_file_name)
        shutil.copy(input_pdb_path, output_pdb_path)

        # Create a directory `seqs`, as required by the PMPNN code, into which we write each sequence as a file.
        os.makedirs(os.path.join(output_dir, "seqs"), exist_ok=True)

        # Prepare inputs
        process = subprocess.Popen(
            [
                "python",
                os.path.join(
                    self.cfg.pmpnn_path, "helper_scripts/parse_multiple_chains.py"
                ),
                f"--input_path={input_pdb_path}",
                f"--output_path={output_dir}",
            ]
        )
        _ = process.wait()

        # Run ProteinMPNN Inverse Folding
        pmpnn_args = [
            "python",
            os.path.join(self.cfg.pmpnn_path, "protein_mpnn_run.py"),
            "--out_folder",
            output_dir,
            "--jsonl_path",
            output_dir,
            "--num_seq_per_target",
            str(num_sequences),
            "--sampling_temp",
            "0.1",
            "--seed",
            str(seed),
            "--batch_size",
            "1",
            "--device",
            str(device_id),
        ]
        process = subprocess.Popen(
            pmpnn_args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        _ = process.wait()

        # Rename the entries into a new file.
        mpnn_fasta_path = os.path.join(
            output_dir,
            "seqs",
            pdb_name + ".fa",
        )
        modified_fasta_path = mpnn_fasta_path.replace(".fa", "_modified.fasta")
        fasta_seqs = SeqIO.to_dict(SeqIO.read(mpnn_fasta_path, "fasta"))
        with open(modified_fasta_path, "w") as f:
            for i, (header, seq) in enumerate(fasta_seqs.items()):
                # Skip the first, which is the input.
                if i == 0:
                    continue
                f.write(f">{pdb_name}_pmpnn_seq_{i}\n{seq}\n")

        return modified_fasta_path

    @staticmethod
    def assess_folded_structures(
        sample_pdb_path: str,
        folded_df: pd.DataFrame,
        true_bb_positions: Optional[npt.NDArray] = None,  # [n_residues, 37, 3]
    ) -> pd.DataFrame:
        """
        Calculate RMSD, pLDDT, and other metrics, comparing folded structures in `folded_df`
        to the sample structure, and ground truth structure if provided.

        `folded_df` can be either the single generated sample, or ProteinMPNN re-folds.

        Extends the input `folded_df` rows with these metrics and returns DataFrame.

        edited from `process_folded_outputs()` in public multiflow
        """
        sample_feats = parse_pdb_feats("sample", pdb_path=sample_pdb_path)
        sample_ca_pos = sample_feats[dpc.bb_positions]
        sample_bb_pos = sample_feats[dpc.atom_positions][:, :3].reshape(-1, 3)

        # Helpers to calculate RMSD
        def _calc_ca_rmsd(mask, folded_ca_pos):
            return (
                superimpose(
                    torch.tensor(sample_ca_pos)[None],
                    torch.tensor(folded_ca_pos[None]),
                    mask,
                )[1]
                .rmsd[0]
                .item()
            )

        def _calc_bb_rmsd(mask, sample_bb_pos, folded_bb_pos):
            aligned_rmsd = superimpose(
                torch.tensor(sample_bb_pos)[None],
                torch.tensor(folded_bb_pos[None]),
                mask[:, None].repeat(1, 3).reshape(-1),
            )
            return aligned_rmsd[1].item()

        # Calculate RMSD etc. for each folded structured in folded_df
        all_metrics = []
        for _, row in folded_df.iterrows():
            folded_feats = parse_pdb_feats(
                row[MetricName.header], row[MetricName.folded_pdb_path]
            )
            sample_metrics = {
                # Include the original row. Includes `header`, `folded_pdb_path`, `plddt_mean` from AF2.
                **row.to_dict(),
                # Associate with the generated sample
                MetricName.sequence: aatype_to_seq(folded_feats[dpc.aatype]),
                MetricName.sample_pdb_path: sample_pdb_path,
            }

            # Calculate RMSD to generated sample
            folded_ca_pos = folded_feats[dpc.bb_positions]
            folded_bb_pos = folded_feats[dpc.atom_positions][:, :3].reshape(-1, 3)
            res_mask = torch.ones(folded_ca_pos.shape[0])
            bb_rmsd = _calc_bb_rmsd(res_mask, sample_bb_pos, folded_bb_pos)
            sample_metrics[MetricName.bb_rmsd] = bb_rmsd
            sample_metrics[MetricName.is_designable] = bb_rmsd <= 2.0

            # If provided ground truth bb positions, also compare to them
            if true_bb_positions is not None:
                # Ideally we could reshape once, outside the loop, but safer to access to row's sample_length here
                sample_length = res_mask.shape[0]
                assert true_bb_positions.shape == (
                    sample_length,
                    37,
                    3,
                ), f"Invalid true_bb_positions shape {true_bb_positions.shape}"
                true_bb_pos = true_bb_positions[:, :3, :].reshape(
                    -1, 3
                )  # rename for reshape
                assert true_bb_pos.shape == (
                    sample_length * 3,
                    3,
                ), f"Invalid true_bb_positions shape {true_bb_pos.shape}"
                assert true_bb_pos.shape == sample_bb_pos.shape
                assert true_bb_pos.shape == folded_bb_pos.shape

                sample_metrics[MetricName.bb_rmsd_gt] = _calc_bb_rmsd(
                    res_mask, sample_bb_pos, true_bb_pos
                )
                sample_metrics[MetricName.bb_rmsd_folded_gt] = _calc_bb_rmsd(
                    res_mask, folded_bb_pos, true_bb_pos
                )

            all_metrics.append(sample_metrics)

        return pd.DataFrame(all_metrics)
