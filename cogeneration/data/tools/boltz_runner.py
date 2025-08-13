"""
Boltz-2 structure prediction module.

This module provides a BoltzRunner class that loads the Boltz-2 model once
and allows for efficient repeated inference. It supports single-sequence mode
(no templates, no MSA) with potentials support and batching.

TODO - boltz now supports `empty` MSA keyword, so we don't need to make a dummy one.
"""

import gc
import json
import logging
import os
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.types import Input, Manifest, Record, Target
from boltz.data.write.writer import BoltzWriter
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzSteeringParams,
    MSAModuleArgs,
    PairformerArgsV2,
    download_boltz2,
    get_cache_path,
    load_canonicals,
    process_input,
)
from boltz.model.models.boltz2 import Boltz2
from pytorch_lightning import Trainer

from cogeneration.config.base import BoltzConfig
from cogeneration.data import residue_constants
from cogeneration.data.const import CHAIN_BREAK_STR, aatype_to_seq
from cogeneration.data.tools.abc import FoldingDataFrame, FoldingTool, infer_device_id
from cogeneration.dataset.process_pdb import pdb_path_pdb_name, process_pdb_file
from cogeneration.models.utils import get_model_size_str
from cogeneration.type.dataset import DatasetProteinColumn, ProcessedFile
from cogeneration.type.metrics import MetricName
from cogeneration.util.log import rank_zero_logger

logger = rank_zero_logger(__name__)


@dataclass
class BoltzPrediction:
    """
    Output structure for Boltz predictions containing paths to generated files.

    The directory is something like `output_dir/predictions/record.id`
    """

    protein_id: str
    predictions_dir: Path
    path_pdb: Path
    path_plddt_npz: Path
    path_confidence_json: Path
    path_structure_npz: Optional[Path] = None

    def __post_init__(self):
        self._processed_file = None

    @classmethod
    def from_output_dir(cls, predictions_dir: Union[str, Path]) -> "BoltzPrediction":
        """
        Create a BoltzPrediction instance by scanning an output directory for Boltz files.

        Args:
            output_dir: Path to the directory containing Boltz output files.

        Returns:
            BoltzPrediction instance with paths to found files.
        """
        predictions_path = Path(predictions_dir)

        if not predictions_path.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {predictions_path}"
            )

        # Extract record_id from directory name (Note, assumes directory is named after record_id)
        record_id = predictions_path.name

        # Use rank 0 files (highest confidence prediction)

        pdb_file = predictions_path / f"{record_id}_model_0.pdb"
        assert pdb_file.exists(), f"PDB file does not exist: {pdb_file}"

        plddt_npz_file = predictions_path / f"plddt_{record_id}_model_0.npz"
        assert (
            plddt_npz_file.exists()
        ), f"Plddt NPZ file does not exist: {plddt_npz_file}"

        confidence_json_file = predictions_path / f"confidence_{record_id}_model_0.json"
        assert (
            confidence_json_file.exists()
        ), f"Confidence JSON file does not exist: {confidence_json_file}"

        # may only have PDB output
        structure_npz_file = predictions_path / f"{record_id}_model_0.npz"
        structure_npz_file = (
            None if not structure_npz_file.exists() else structure_npz_file
        )

        prediction = cls(
            protein_id=record_id,
            predictions_dir=predictions_path,
            path_pdb=pdb_file,
            path_plddt_npz=plddt_npz_file,
            path_confidence_json=confidence_json_file,
            path_structure_npz=structure_npz_file,
        )

        return prediction

    def parsed_structure(self) -> ProcessedFile:
        """
        Parse the PDB file at path_pdb using `process_pdb_file`
        """
        assert self.path_pdb is not None, "path_pdb is None"

        if self._processed_file is None:
            pdb_name = pdb_path_pdb_name(str(self.path_pdb))
            self._processed_file = process_pdb_file(
                pdb_file_path=str(self.path_pdb),
                pdb_name=pdb_name,
                write_dir=None,
                chain_id=None,
                scale_factor=1.0,
                max_combined_length=8192,
            )

        return self._processed_file

    def get_plddt_mean(self) -> Optional[float]:
        """
        Extract mean pLDDT confidence score from Boltz confidence files.

        Returns:
            Mean pLDDT score if available, None otherwise.
        """

        # Try to get pLDDT from NPZ file
        if self.path_plddt_npz and self.path_plddt_npz.exists():
            try:
                with np.load(self.path_plddt_npz) as plddt_data:
                    for key in ["plddt", "confidence", "plddt_scores"]:
                        if key in plddt_data:
                            scores = plddt_data[key]
                            return float(np.mean(scores))
            except (OSError, KeyError, ValueError):
                pass

        # Try to get pLDDT from confidence JSON
        if self.path_confidence_json and self.path_confidence_json.exists():
            try:
                with open(self.path_confidence_json, "r") as f:
                    confidence_data = json.load(f)
                # Boltz confidence is calculated as 0.8 * plddt + 0.2 * ipTM
                # We need to extract just the pLDDT component if available
                if "plddt" in confidence_data:
                    plddt_scores = confidence_data["plddt"]
                    if isinstance(plddt_scores, list):
                        return float(np.mean(plddt_scores))
                    elif isinstance(plddt_scores, (int, float)):
                        return float(plddt_scores)
                # Fallback to overall confidence score if pLDDT not available
                elif "confidence" in confidence_data:
                    return float(confidence_data["confidence"])
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

        return None

    def get_sequence(self) -> str:
        """
        Extract the amino acid sequence from the PDB structure.

        Returns:
            Amino acid sequence as a string if available, None otherwise.
        """
        processed_file = self.parsed_structure()

        sequence = aatype_to_seq(
            aatype=processed_file[DatasetProteinColumn.aatype],
            chain_idx=processed_file[DatasetProteinColumn.chain_index],
        )

        return sequence

    def clean_predictions_dir(self):
        """
        Remove outputs we don't need. Keep paths that are in the struct.
        """
        files_to_keep = [
            path
            for path in [
                self.path_pdb,
                self.path_structure_npz,
                self.path_plddt_npz,
                self.path_confidence_json,
            ]
            if path is not None and path.exists()
        ]

        # glob recursively
        for file in self.predictions_dir.glob("**/*"):
            if file.is_file() and file not in files_to_keep:
                try:
                    os.remove(file)
                except OSError:
                    pass


class BoltzPredictionSet:
    """
    A collection of BoltzPrediction objects returned from folding multiple proteins.
    """

    def __init__(self, predictions: List[BoltzPrediction]):
        self.predictions = predictions

    def __len__(self) -> int:
        return len(self.predictions)

    def __getitem__(self, index: int) -> BoltzPrediction:
        return self.predictions[index]

    def __iter__(self):
        return iter(self.predictions)

    def to_df(self) -> pd.DataFrame:
        """
        Convert to a DataFrame matching the AF2 folding format.

        Returns:
            DataFrame with columns: header, sequence, folded_pdb_path, plddt_mean
            matching the format returned by FoldingValidator._run_alphafold2()
        """
        rows = []
        for i, pred in enumerate(self.predictions):
            header = pred.protein_id
            sequence = pred.get_sequence()
            plddt_mean = pred.get_plddt_mean()

            row = {
                MetricName.header: header,
                MetricName.sequence: sequence,
                MetricName.folded_pdb_path: str(pred.path_pdb),
                MetricName.plddt_mean: plddt_mean,
            }
            rows.append(row)

        return pd.DataFrame(rows)


@dataclass(frozen=True, slots=True)
class BoltzPaths:
    """
    Boltz file layout

    cache_dir/
    ├── mols/
    ├── boltz2_conf.ckpt
    └── boltz2_aff.ckpt

    outputs_dir/
    ├── {id}.fasta        (generated input, Boltz-formatted FASTA)
    ├── processed/
    │   ├── constraints/
    │   ├── msa/
    │   ├── mols/
    │   ├── records/
    │   ├── templates/
    │   └── structures/
    ├── msa/              (dummy MSAs)
    ├── predictions/      (generated structures)
    └── logs/
    """

    cache_dir: Path
    outputs_dir: Path

    # derived paths
    processed_dir: Path = field(init=False)
    constraints_dir: Path = field(init=False)
    msa_dir: Path = field(init=False)
    mols_dir: Path = field(init=False)
    records_dir: Path = field(init=False)
    templates_dir: Path = field(init=False)
    structures_dir: Path = field(init=False)
    predictions_dir: Path = field(init=False)
    logs_dir: Path = field(init=False)
    processed_msa_dir: Path = field(init=False)
    processed_mols_dir: Path = field(init=False)

    def __post_init__(self) -> None:
        outputs = Path(self.outputs_dir).expanduser().resolve()
        cache = Path(self.cache_dir).expanduser().resolve()

        # Make cache_dir accessible as a resolved Path
        object.__setattr__(self, "cache_dir", cache)
        object.__setattr__(self, "mols_dir", cache / "mols")

        object.__setattr__(self, "processed_dir", outputs / "processed")
        object.__setattr__(self, "constraints_dir", self.processed_dir / "constraints")
        object.__setattr__(self, "msa_dir", outputs / "msa")
        object.__setattr__(self, "records_dir", self.processed_dir / "records")
        object.__setattr__(self, "templates_dir", self.processed_dir / "templates")
        object.__setattr__(self, "structures_dir", self.processed_dir / "structures")
        object.__setattr__(self, "predictions_dir", outputs / "predictions")
        object.__setattr__(self, "logs_dir", outputs / "logs")
        object.__setattr__(self, "processed_msa_dir", self.processed_dir / "msa")
        object.__setattr__(self, "processed_mols_dir", self.processed_dir / "mols")

    def mkdirs(self, exist_ok: bool = True) -> None:
        for d in (
            self.outputs_dir,
            self.processed_dir,
            self.constraints_dir,
            self.msa_dir,
            self.mols_dir,
            self.records_dir,
            self.templates_dir,
            self.structures_dir,
            self.predictions_dir,
            self.logs_dir,
            self.processed_msa_dir,
            self.processed_mols_dir,
        ):
            d.mkdir(parents=True, exist_ok=exist_ok)


class BoltzManifestBuilder:
    """
    Builder class for creating Boltz Manifest + Target objects from sequences.

    Supports:
    - Single and multiple sequences
    - Chain breaks using ':' separator
    - Multimeric structures
    - Tensor batch input with chain indices
    - FASTA file input
    - Uses Boltz's process_input function to create all necessary files
    """

    # Class-level counter for globally unique MSA IDs
    _msa_id_counter = -1
    _msa_id_lock = threading.Lock()

    def __init__(self, outputs_dir: Union[str, Path], cache_dir: Union[str, Path]):
        """
        Initialize ManifestBuilder.

        Args:
            outputs_dir: Directory to write FASTA files and processed data.
            cache_dir: Directory containing Boltz models and mols.
        """
        self.paths = BoltzPaths(cache_dir=cache_dir, outputs_dir=outputs_dir)
        self.paths.mkdirs()

    @classmethod
    def _get_next_msa_id(cls) -> int:
        """Get the next globally unique MSA ID (negative integer)."""
        with cls._msa_id_lock:
            msa_id = cls._msa_id_counter
            cls._msa_id_counter -= 1
            return msa_id

    def _validate_sequence(self, seq: str) -> str:
        """Validate and clean a protein sequence."""
        valid_aas = set(residue_constants.restypes + ["X"])
        if not all(aa.upper() in valid_aas for aa in seq):
            invalid_aas = [aa for aa in seq if aa.upper() not in valid_aas]
            raise ValueError(f"Invalid amino acids in sequence: {invalid_aas}")
        return seq.upper()

    def _parse_sequence_with_chain_breaks(self, seq: str) -> List[str]:
        """Parse a sequence string with ':' chain breaks into individual chain sequences."""
        return [
            self._validate_sequence(chain_seq)
            for chain_seq in seq.split(CHAIN_BREAK_STR)
        ]

    def _write_dummy_chain_msa(
        self, chain_seq: str, msa_dir: Optional[Path] = None
    ) -> Path:
        """
        Write a dummy MSA file for a single chain.
        """
        assert (
            CHAIN_BREAK_STR not in chain_seq
        ), f"Chain sequence {chain_seq} contains chain breaks"

        if msa_dir is None:
            msa_dir = self.paths.msa_dir

        # Create unique MSA_ID for this chain
        msa_taxonomy_id = self._get_next_msa_id()

        msa_filename = f"{msa_taxonomy_id}.csv"
        msa_path = msa_dir / msa_filename

        # Write dummy MSA with just the target sequence using MSA_ID
        with open(msa_path, "w") as msa_f:
            msa_f.write("key,sequence\n")
            msa_f.write(f"{msa_taxonomy_id},{chain_seq}\n")

        return msa_path

    def _write_fasta_file_and_msas(
        self,
        protein_id: str,
        seq: str,
        fasta_dir: Optional[Path] = None,
        msa_dir: Optional[Path] = None,
    ) -> Path:
        """
        Write FASTA file in Boltz format with proper headers for process_input.
        Creates dummy MSA files for single-sequence mode with unique MSA_ID per chain.

        Returns the path to the written FASTA file in Boltz format.

        Format: >CHAIN_ID|ENTITY_TYPE|MSA_PATH
        Example: >A|protein|1.csv
        """
        if fasta_dir is None:
            fasta_dir = self.paths.outputs_dir
        if msa_dir is None:
            msa_dir = self.paths.msa_dir

        fasta_path = fasta_dir / f"{protein_id}.fasta"

        chain_sequences = self._parse_sequence_with_chain_breaks(seq)

        with open(fasta_path, "w") as f:
            for i, chain_seq in enumerate(chain_sequences):
                chain_name = chr(ord("A") + i)  # A, B, C, etc.
                chain_msa_path = self._write_dummy_chain_msa(
                    chain_seq=chain_seq, msa_dir=msa_dir
                )
                f.write(f">{chain_name}|protein|{chain_msa_path}\n{chain_seq}\n")

        return fasta_path

    def _create_targets_from_sequences(
        self,
        sequences: List[str],
        protein_ids: List[str],
    ) -> List[Record]:
        """
        Internal method to create Target objects from sequences using process_input.
        This creates all necessary files that Boltz expects during inference.
        """

        if len(sequences) != len(protein_ids):
            raise ValueError("Number of sequences and protein IDs must match")

        # ensure mols dir exists and is not empty
        assert (
            self.paths.mols_dir.exists()
        ), f"Mols directory {self.paths.mols_dir} does not exist"
        # check a specific file exists, not empty directory
        # (brittle - may break if Boltz changes)
        assert (
            self.paths.mols_dir / "000.pkl"
        ).exists(), f"Mols directory {self.paths.mols_dir} is empty"

        ccd = load_canonicals(self.paths.mols_dir)

        records = []
        for seq, protein_id in zip(sequences, protein_ids):
            # Write FASTA file in Boltz format with dummy MSAs
            fasta_path = self._write_fasta_file_and_msas(
                protein_id=protein_id,
                seq=seq,
            )

            # Use process_input to create all necessary files
            try:
                process_input(
                    path=fasta_path,
                    ccd=ccd,
                    msa_dir=self.paths.msa_dir,
                    mol_dir=self.paths.mols_dir,
                    boltz2=True,
                    use_msa_server=False,
                    msa_server_url="",
                    msa_pairing_strategy="",
                    msa_server_username=None,
                    msa_server_password=None,
                    api_key_header=None,
                    api_key_value=None,
                    max_msa_seqs=0,
                    processed_msa_dir=self.paths.processed_msa_dir,
                    processed_constraints_dir=self.paths.constraints_dir,
                    processed_templates_dir=self.paths.templates_dir,
                    processed_mols_dir=self.paths.processed_mols_dir,
                    structure_dir=self.paths.structures_dir,
                    records_dir=self.paths.records_dir,
                )

                # Load the created record
                record_path = self.paths.records_dir / f"{protein_id}.json"
                if not record_path.exists():
                    raise FileNotFoundError(f"Record file not created: {record_path}")

                record = Record.load(record_path)

                # confirm id
                assert (
                    record.id == protein_id
                ), f"unexpected record ID, got {record.id}, expected {protein_id}"

                records.append(record)

            except Exception as e:
                raise RuntimeError(
                    f"Failed to process sequence for {protein_id}: {e}"
                ) from e

        return records

    def from_sequences(
        self, sequences: List[str], protein_ids: List[str] = None
    ) -> Manifest:
        """
        Create processed input files from multiple sequences with optional chain breaks.

        Args:
            sequences: List of protein sequences. Use ':' to separate chains in multimers.
            protein_ids: Optional list of protein IDs. If None, uses "protein_0", etc.

        Returns:
            Manifest object
        """
        records = self._create_targets_from_sequences(
            sequences=sequences, protein_ids=protein_ids
        )

        logger.debug(
            f"Prepared Boltz-2 manifest with {len(records)} records, lengths: {[len(seq) for seq in sequences]}"
        )
        return Manifest(records=records)

    def from_batch(
        self,
        aatypes: torch.Tensor,
        chain_idx: torch.Tensor,
        protein_ids: List[str],
    ) -> Manifest:
        """
        Create processed input files from batch tensors with chain indices.

        Args:
            aatypes: Tensor of shape (B, N) with amino acid type indices.
            chain_idx: Tensor of shape (B, N) with chain indices for each residue.
            protein_ids: Optional list of protein IDs. If None, uses "protein_0", etc.

        Returns:
            Manifest object
        """
        B, N = aatypes.shape
        if chain_idx.shape != (B, N):
            raise ValueError("aatypes and chain_idx must have the same shape")

        if len(protein_ids) != B:
            raise ValueError("Number of protein IDs must match batch size")

        # Convert batch tensors to sequences with chain breaks
        sequences_with_breaks = []
        for i in range(B):
            seq_indices = aatypes[i].detach().cpu().numpy()
            chains = chain_idx[i].detach().cpu().numpy()

            # Group residues by chain
            chain_sequences = {}
            for aa_idx, chain_id in zip(seq_indices, chains):
                if aa_idx >= len(residue_constants.restypes_with_x):
                    break  # Stop at padding

                aa = residue_constants.restypes_with_x[aa_idx]

                # convert X to A
                if aa == "X":
                    aa = "A"

                if chain_id not in chain_sequences:
                    chain_sequences[chain_id] = ""

                chain_sequences[chain_id] += aa

            # Sort chains by ID and join with ':'
            sorted_chains = [
                chain_sequences[cid] for cid in sorted(chain_sequences.keys())
            ]
            sequence_with_breaks = CHAIN_BREAK_STR.join(sorted_chains)
            sequences_with_breaks.append(sequence_with_breaks)

        # Use from_sequences to create processed files
        return self.from_sequences(
            sequences=sequences_with_breaks, protein_ids=protein_ids
        )

    def parse_fasta_records(self, fasta_path: Path) -> List[SeqRecord]:
        """
        Parse a FASTA file into a list of SeqRecord objects.
        """
        assert fasta_path.exists(), f"Fasta path {fasta_path} does not exist"

        records = list(SeqIO.parse(fasta_path, "fasta"))
        if not records:
            raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

        return records

    def from_fasta(
        self, fasta_path: Union[str, Path], protein_ids: Optional[List[str]] = None
    ) -> Manifest:
        """
        Create processed input files from a FASTA file.

        Args:
            fasta_path: Path to FASTA file containing sequences to fold
            protein_ids: Optional list of protein IDs. If None, uses headers from FASTA.

        Returns:
            Manifest object
        """
        fasta_path = Path(fasta_path)
        seq_records = self.parse_fasta_records(fasta_path=fasta_path)
        sequences = [str(seq_record.seq) for seq_record in seq_records]

        if protein_ids is not None:
            assert len(seq_records) == len(
                protein_ids
            ), f"Number of protein IDs must match number of sequences in FASTA"
        else:
            # Require ids are unique
            protein_ids = [record.id for record in seq_records]
            assert len(set(protein_ids)) == len(
                protein_ids
            ), f"Fasta file {fasta_path} has duplicate protein ids"

        return self.from_sequences(sequences=sequences, protein_ids=protein_ids)

    def prepare_fasta_for_cli(self, fasta_path: Union[str, Path]) -> Path:
        """
        Create a CLI-ready FASTA file with proper Boltz format and dummy MSA files.
        Creates a separate directory containing only FASTA files (as required by Boltz CLI).
        MSA files are placed in a sibling directory to avoid CLI directory scanning issues.

        Args:
            fasta_path: Path to input FASTA file

        Returns:
            Path to directory containing only Boltz-formatted FASTA files (for CLI)
        """
        seq_records = self.parse_fasta_records(fasta_path=fasta_path)

        # Require ids are unique
        assert len(set(seq_record.id for seq_record in seq_records)) == len(
            seq_records
        ), f"Fasta file {fasta_path} has duplicate protein ids"

        # Create a separate directory for just the FASTA files (CLI requirement)
        fasta_only_dir = self.paths.outputs_dir / "fasta"
        fasta_only_dir.mkdir(parents=True, exist_ok=True)

        # Create MSA directory as sibling to avoid Boltz CLI directory scanning issues
        cli_msa_dir = self.paths.outputs_dir / "msa"
        cli_msa_dir.mkdir(parents=True, exist_ok=True)

        # For each sequence, write a FASTA file in Boltz format with dummy MSAs
        for seq_record in seq_records:
            protein_id = seq_record.id
            sequence = str(seq_record.seq)
            fasta_path = self._write_fasta_file_and_msas(
                protein_id=protein_id,
                seq=sequence,
                fasta_dir=fasta_only_dir,
                msa_dir=cli_msa_dir,
            )

        return fasta_only_dir


class QuietBoltzWriter(BoltzWriter):
    """
    Quiet BoltzWriter that does not print the number of failed examples.
    """

    def on_predict_epoch_end(self, *args, **kwargs):
        pass


class BoltzRunner(FoldingTool):
    """
    Boltz-2 structure prediction class for efficient repeated inference.

    Loads the model once and allows for fast structure prediction from sequences
    or batches. Supports single-sequence mode (no templates, no MSA) with
    potentials support.
    """

    def __init__(
        self,
        cfg: BoltzConfig,
        device: Optional[Union[str, int]] = None,
    ):
        """
        Initialize the Boltz predictor.

        """
        self.cfg = cfg

        # Initialize model and trainer to None before setting device
        self.model = None
        self.trainer = None

        self.set_device_id(device)

        # Set up default paths using BoltzPaths
        self.default_paths = BoltzPaths(
            cache_dir=self.cfg.cache_dir, outputs_dir=self.cfg.outputs_path
        )
        self.default_paths.mkdirs()

        self.checkpoint_path = self.cfg.checkpoint_path

        # Optional periodic teardown to mitigate native MPS compiled-graph/cache growth.
        self.reload_every_n = getattr(self.cfg, "reload_every_n", None)
        self._predictions_since_load = 0

        # Configure model parameters
        self.diffusion_params = Boltz2DiffusionParams()
        self.diffusion_params.step_scale = self.cfg.step_scale

        self.pairformer_args = PairformerArgsV2()
        self.msa_args = MSAModuleArgs(
            subsample_msa=False,  # No MSA in single-sequence mode
            num_subsampled_msa=1,
            use_paired_feature=True,
        )

        self.steering_args = BoltzSteeringParams()
        if not self.cfg.use_potentials:
            self.steering_args.fk_steering = False
            self.steering_args.guidance_update = False

        # Model loaded in `predict` when first needed

    def set_device_id(self, device_id: Optional[Union[str, int]] = None):
        """Set the device ID for the folding tool."""
        self.device = infer_device_id(device_id=device_id)

        if self.model is not None:
            emoji = "🧊" if self.device.type == "cpu" else "🔥"
            logger.info(f"Moving Boltz-2 model to device {emoji} {self.device}")
            self.model.to(self.device)

        # if self.trainer is not None:
        #     # Use string instead of device object for trainer accelerator
        #     if self.device.type == "cuda":
        #         self.trainer.strategy.accelerator = CUDAAccelerator()
        #     elif self.device.type == "mps":
        #         self.trainer.strategy.accelerator = MPSAccelerator()
        #     else:
        #         self.trainer.strategy.accelerator = CPUAccelerator()

    def _download_model_if_needed(self):
        """Download Boltz-2 model, mols, etc if not present in cache_dir."""
        # Create cache directory if needed
        cache_dir = self.cfg.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)

        if not self.checkpoint_path.exists():
            download_boltz2(cache_dir)
            assert self.checkpoint_path.exists()
            assert self.default_paths.mols_dir.exists()

    def _load_model(self):
        """Load the Boltz-2 model from checkpoint."""

        # Download model if needed
        self._download_model_if_needed()

        predict_args = {
            "recycling_steps": self.cfg.recycling_steps,
            "sampling_steps": self.cfg.sampling_steps,
            "diffusion_samples": self.cfg.diffusion_samples,
            "max_parallel_samples": None,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }

        logger.debug(f"Loading Boltz-2 model (will persist in memory)...")

        # performance settings
        # TODO(boltz) use compile kwargs. Issue naming state keys.

        self.model = Boltz2.load_from_checkpoint(
            self.checkpoint_path,
            strict=True,
            predict_args=predict_args,
            map_location=self.device,
            diffusion_process_args=asdict(self.diffusion_params),
            ema=False,
            use_kernels=self.cfg.use_kernels,
            predict_bfactor=True,
            pairformer_args=asdict(self.pairformer_args),
            msa_args=asdict(self.msa_args),
            steering_args=asdict(self.steering_args),
        )
        self.model.eval()
        self.model.to(self.device)

        logger.info(
            f"✅ Successfully loaded Boltz-2 model on {self.device} (size: {get_model_size_str(self.model)})"
        )

        self.trainer = Trainer(
            accelerator=self.cfg.accelerator,
            precision=self.cfg.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

    def _get_manifest_builder(
        self, output_dir: Optional[Path] = None
    ) -> BoltzManifestBuilder:
        """
        Get a manifest builder for the current paths.
        """
        # Create an isolated outputs_dir with a timestamp + UUID
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            uuid_str = str(uuid.uuid4())
            output_dir = self.default_paths.outputs_dir / f"{timestamp}_{uuid_str}"

        return BoltzManifestBuilder(
            outputs_dir=output_dir, cache_dir=self.default_paths.cache_dir
        )

    @staticmethod
    def _cleanup_paths(paths: BoltzPaths) -> None:
        """
        Remove intermediate files/directories following predictions.

        This function cleans up the input artifacts created and intermediate processed files.
        Prediction directories are handled separately by `BoltzPrediction.clean_predictions_dir`.

        We leave the top-level directories in place, as they are expected to remain created.
        """
        for dir in [
            # remove boltz intermediate files
            paths.processed_dir,
            # Remove dummy MSAs
            paths.msa_dir,
        ]:
            if dir.exists():
                for item in dir.iterdir():
                    if item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                    else:
                        item.unlink(missing_ok=True)

    def predict(
        self,
        manifest: Manifest,
        paths: BoltzPaths,
    ) -> BoltzPredictionSet:
        """
        Fold structures from a manifest.

        Args:
            manifest: Input manifest containing records to fold.
            paths: BoltzPaths defining all paths for this run.

        Returns:
            BoltzPredictionSet containing BoltzPrediction objects for each protein in the manifest.
        """
        # Load model if not already loaded
        if self.model is None or self.trainer is None:
            self._load_model()

        start_time = time.time()
        paths.predictions_dir.mkdir(parents=True, exist_ok=True)

        # Require an empty output directory, so we can safely clean up outputs.
        for record in manifest.records:
            protein_output_dir = paths.predictions_dir / record.id
            if protein_output_dir.exists() and len(os.listdir(protein_output_dir)) > 0:
                raise ValueError(
                    f"Output directory {protein_output_dir} is not empty. "
                    "Please provide an empty directory for new targets."
                )

        # Use BoltzPaths processed directory
        processed_path = paths.processed_dir
        if not processed_path.exists():
            raise ValueError(
                "Processed files not found. "
                "Use ManifestBuilder.from_sequences() or .from_batch() to create processed files first."
            )

        # Verify record files exist
        records_dir = processed_path / "records"
        expected_record_files = [
            records_dir / f"{record.id}.json" for record in manifest.records
        ]
        missing_files = [f for f in expected_record_files if not f.exists()]
        if missing_files:
            raise FileNotFoundError(
                f"Missing record files in {records_dir}: {missing_files}"
            )

        # Set up data module using processed data directories
        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            target_dir=paths.structures_dir,
            msa_dir=paths.processed_msa_dir,
            mol_dir=paths.mols_dir,
            num_workers=self.cfg.num_workers,
            constraints_dir=paths.constraints_dir,
            template_dir=paths.templates_dir,
            extra_mols_dir=paths.processed_mols_dir,
            override_method=None,
            affinity=False,
        )

        # Set up writer
        writer = QuietBoltzWriter(
            data_dir=str(paths.structures_dir),
            output_dir=str(paths.predictions_dir),
            output_format=self.cfg.output_format,
            boltz2=True,
        )

        # Update trainer with writer callback
        self.trainer.callbacks = [writer]

        # Run prediction
        logger.debug(
            f"Running Boltz (native) prediction for {len(manifest.records)} records..."
        )
        self.trainer.predict(
            self.model,
            datamodule=data_module,
            return_predictions=False,
        )

        # Create BoltzPrediction objects for each protein
        predictions = []
        for record in manifest.records:
            protein_output_dir = paths.predictions_dir / record.id
            prediction = BoltzPrediction.from_output_dir(protein_output_dir)
            # remove unnecesary outputs
            prediction.clean_predictions_dir()
            predictions.append(prediction)

        # remove remaining intermediate Boltz-2 input / processing artifacts
        self._cleanup_paths(paths)

        # attempt cleanup / gc
        self._release_inference_state(writer=writer, data_module=data_module)

        # Periodically tear down model/trainer to reclaim native caches if configured
        try:
            self._predictions_since_load += len(manifest.records)
        except Exception:
            pass
        if (
            self.reload_every_n is not None
            and self.reload_every_n > 0
            and self._predictions_since_load >= self.reload_every_n
        ):
            logger.info(
                f"🗑️ Tearing down Boltz model after {self._predictions_since_load} predictions"
            )
            self._predictions_since_load = 0
            self._teardown_model()

        end_time = time.time()
        logger.debug(
            f"Boltz (native) prediction completed in {end_time - start_time:.2f} seconds"
        )

        return BoltzPredictionSet(predictions)

    def _release_inference_state(
        self, writer: QuietBoltzWriter, data_module: Boltz2InferenceDataModule
    ) -> None:
        """
        post-predict() cleanup to avoid reference retention.
        """
        # Detach callbacks to avoid callback list growth
        try:
            if self.trainer is not None:
                self.trainer.callbacks = []
                # Some PL versions keep a reference to the last used datamodule
                # Remove strong refs if present
                if hasattr(self.trainer, "datamodule"):
                    setattr(self.trainer, "datamodule", None)
        except Exception:
            pass

        # Drop local references
        if writer is not None:
            try:
                del writer
            except Exception:
                pass
        if data_module is not None:
            try:
                del data_module
            except Exception:
                pass

        # GC and empty caches
        try:
            gc.collect()
        except Exception:
            pass
        try:
            if (
                hasattr(torch, "cuda")
                and torch.cuda.is_available()
                and self.device.type == "cuda"
            ):
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            if (
                hasattr(torch, "mps")
                and torch.backends.mps.is_available()
                and self.device.type == "mps"
            ):
                torch.mps.empty_cache()
        except Exception:
            pass

    def _teardown_model(self) -> None:
        """
        Fully tear down model and trainer to encourage native cache release.
        Next prediction will lazily reload the model.
        """
        # Drop model and trainer reference
        try:
            if self.model is not None:
                del self.model
        except Exception:
            pass
        finally:
            self.model = None

        try:
            if self.trainer is not None:
                try:
                    self.trainer.callbacks = []
                except Exception:
                    pass
        finally:
            self.trainer = None

        self._release_inference_state(writer=None, data_module=None)

    def fold_fasta(
        self,
        fasta_path: Path,
        output_dir: Path,
    ) -> FoldingDataFrame:
        """
        Fold a protein sequence from a FASTA file and save the results to an output directory.
        Each fasta entry should be an entry in the DataFrame.

        Implements the FoldingTool interface.
        """
        if self.cfg.run_native:
            return self.fold_fasta_native(fasta_path=fasta_path, output_dir=output_dir)
        else:
            return self.fold_fasta_subprocess(
                fasta_path=fasta_path, output_dir=output_dir
            )

    def fold_fasta_native(
        self,
        fasta_path: Path,
        output_dir: Path,
    ) -> FoldingDataFrame:
        """
        Using in-memory model, fold a protein sequence from a FASTA file.
        Each fasta entry should be an entry in the DataFrame.

        Implements the FoldingTool interface.
        """
        manifest_builder = self._get_manifest_builder(output_dir=output_dir)
        manifest = manifest_builder.from_fasta(fasta_path=fasta_path)

        prediction_set = self.predict(manifest=manifest, paths=manifest_builder.paths)

        return prediction_set.to_df()

    def fold_fasta_subprocess(
        self,
        fasta_path: Path,
        output_dir: Path,
    ) -> FoldingDataFrame:
        """
        Using Boltz CLI as a subprocess, fold a protein sequence from a FASTA file.
        Each fasta entry should be an entry in the DataFrame.

        This method calls `boltz predict` as a CLI command instead of using the Python API.
        Useful for simpler execution or when the Python API has issues.

        Creates dummy MSA files for single-sequence mode, similar to the Python API.

        Note CLI Boltz outputs results to `outputs_dir/boltz_results_<input_dir>`

        Args:
            fasta_path: Path to the FASTA file containing sequences to fold.
            output_dir: Directory to save the prediction results.

        Returns:
            DataFrame with columns: header, sequence, folded_pdb_path, plddt_mean
        """
        start_time = time.time()

        # Use ManifestBuilder to prepare FASTA with MSA files
        manifest_builder = self._get_manifest_builder(output_dir=output_dir)

        # ensure can parse fasta
        records = manifest_builder.parse_fasta_records(fasta_path=fasta_path)
        logger.debug(
            f"Running Boltz (subprocess) prediction for {len(records)} records from {fasta_path}..."
        )

        # Ensure model is downloaded
        self._download_model_if_needed()

        # Get Boltz-formatted FASTA with dummy MSAs
        fasta_dir = manifest_builder.prepare_fasta_for_cli(fasta_path=fasta_path)

        # Build the CLI command
        cmd = [
            "boltz",
            "predict",
            str(fasta_dir),
            "--cache",
            str(self.cfg.cache_dir),
            "--out_dir",
            str(output_dir),
            "--output_format",
            "pdb",
            # config parameters
            "--sampling_steps",
            str(self.cfg.sampling_steps),
            "--diffusion_samples",
            str(self.cfg.diffusion_samples),
            "--recycling_steps",
            str(self.cfg.recycling_steps),
            "--step_scale",
            str(self.cfg.step_scale),
        ]

        logger.debug(
            f"Running Boltz (subprocess) prediction for {len(records)} records: {cmd}"
        )

        # Run the CLI command
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"Boltz CLI command failed with return code {e.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stdout: {e.stdout}\n"
                f"stderr: {e.stderr}"
            ) from e

        # Parse CLI output and create BoltzPrediction objects
        predictions = []
        boltz_results_dirs = list(output_dir.glob("boltz_results_*"))
        if not boltz_results_dirs:
            raise RuntimeError(
                f"No boltz_results_* directory found in output directory: {output_dir}"
            )

        # Process all boltz_results_* directories to handle multiple sequences
        for boltz_results_dir in boltz_results_dirs:
            predictions_dir = boltz_results_dir / "predictions"
            if not predictions_dir.exists():
                continue

            # Each sequence gets its own subdirectory in predictions/
            for item in predictions_dir.iterdir():
                if item.is_dir():
                    try:
                        prediction = BoltzPrediction.from_output_dir(item)
                        prediction.clean_predictions_dir()
                        predictions.append(prediction)
                    except (FileNotFoundError, AssertionError):
                        continue

        if not predictions:
            raise RuntimeError(
                f"No valid predictions found in output directory: {output_dir}"
            )

        end_time = time.time()

        logger.debug(
            f"Boltz (subprocess) prediction for {len(predictions)}/{len(records)} records completed in {end_time - start_time:.2f} seconds"
        )
        return BoltzPredictionSet(predictions).to_df()

    def fold_sequences(
        self,
        sequences: List[str],
        protein_ids: List[str],
    ) -> BoltzPredictionSet:
        """
        Fold sequences from a list of sequences and protein IDs.
        Specify chain breaks using CHAIN_BREAK_STR ':' in sequences.
        """
        manifest_builder = self._get_manifest_builder(output_dir=None)
        manifest = manifest_builder.from_sequences(
            sequences=sequences,
            protein_ids=protein_ids,
        )

        return self.predict(manifest=manifest, paths=manifest_builder.paths)

    def fold_batch(
        self,
        aatypes: torch.Tensor,  # (B, N)
        chain_idx: torch.Tensor,  # (B, N)
        protein_ids: List[str],  # len == B
    ) -> BoltzPredictionSet:
        """
        Fold sequences in a batch. Specify chain breaks in chain_idx.
        Returns one prediction per sequence
        """
        manifest_builder = self._get_manifest_builder(output_dir=None)
        manifest = manifest_builder.from_batch(
            aatypes=aatypes,
            chain_idx=chain_idx,
            protein_ids=protein_ids,
        )

        return self.predict(manifest=manifest, paths=manifest_builder.paths)
