"""
Boltz-2 structure prediction module.

This module provides a BoltzRunner class that loads the Boltz-2 model once
and allows for efficient repeated inference. It supports single-sequence mode
(no templates, no MSA) with potentials support and batching.
"""

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
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
    process_input,
)
from boltz.model.models.boltz2 import Boltz2
from pytorch_lightning import Trainer

from cogeneration.config.base import BoltzConfig
from cogeneration.data import residue_constants
from cogeneration.data.tools.abc import FoldingTool
from cogeneration.dataset.process_pdb import pdb_path_pdb_name, process_pdb_file
from cogeneration.type.dataset import DatasetProteinColumn, ProcessedFile
from cogeneration.type.metrics import MetricName


@dataclass
class BoltzPrediction:
    """
    Output structure for Boltz predictions containing paths to generated files.
    """

    protein_id: str
    output_dir: Path
    path_pdb: Path
    path_plddt_npz: Path
    path_confidence_json: Path
    path_structure_npz: Optional[Path] = None

    def __post_init__(self):
        self._processed_file = None

    @classmethod
    def from_output_dir(cls, output_dir: Union[str, Path]) -> "BoltzPrediction":
        """
        Create a BoltzPrediction instance by scanning an output directory for Boltz files.

        Args:
            output_dir: Path to the directory containing Boltz output files.

        Returns:
            BoltzPrediction instance with paths to found files.
        """
        output_path = Path(output_dir)

        if not output_path.exists():
            raise FileNotFoundError(f"Output directory does not exist: {output_path}")

        # Extract record_id from directory name (Note, assumes directory is named after record_id)
        record_id = output_path.name

        # Use rank 0 files (highest confidence prediction)

        pdb_file = output_path / f"{record_id}_model_0.pdb"
        assert pdb_file.exists(), f"PDB file does not exist: {pdb_file}"

        plddt_npz_file = output_path / f"plddt_{record_id}_model_0.npz"
        assert (
            plddt_npz_file.exists()
        ), f"Plddt NPZ file does not exist: {plddt_npz_file}"

        confidence_json_file = output_path / f"confidence_{record_id}_model_0.json"
        assert (
            confidence_json_file.exists()
        ), f"Confidence JSON file does not exist: {confidence_json_file}"

        # may only have PDB output
        structure_npz_file = output_path / f"{record_id}_model_0.npz"

        prediction = cls(
            protein_id=record_id,
            output_dir=output_path,
            path_pdb=pdb_file,
            path_plddt_npz=plddt_npz_file,
            path_confidence_json=confidence_json_file,
            path_structure_npz=(
                structure_npz_file if structure_npz_file.exists() else None
            ),
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
                plddt_data = np.load(self.path_plddt_npz)
                # Look for common pLDDT keys in the NPZ file
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

        aatype = processed_file[DatasetProteinColumn.aatype]
        # Convert aatype indices to sequence string
        sequence = "".join([residue_constants.restypes_with_x[aa] for aa in aatype])

        return sequence

    def clean_output_dir(self):
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

        for file in self.output_dir.glob("*"):
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

    def __init__(self, target_dir: Optional[Union[str, Path]] = None):
        """
        Initialize ManifestBuilder.

        Args:
            target_dir: Directory to write FASTA files and processed data.
                       If None, uses "./outputs/targets".
        """
        if target_dir is None:
            self.target_dir = Path("./outputs/targets")
        else:
            self.target_dir = Path(target_dir)
        self.target_dir.mkdir(parents=True, exist_ok=True)

        # Create processed data directories (as expected by process_input)
        self.processed_dir = self.target_dir / "processed"
        self.records_dir = self.processed_dir / "records"
        self.structure_dir = self.processed_dir / "structures"
        self.processed_msa_dir = self.processed_dir / "msa"
        self.processed_constraints_dir = self.processed_dir / "constraints"
        self.processed_templates_dir = self.processed_dir / "templates"
        self.processed_mols_dir = self.processed_dir / "mols"

        # Create all necessary directories
        for dir_path in [
            self.records_dir,
            self.structure_dir,
            self.processed_msa_dir,
            self.processed_constraints_dir,
            self.processed_templates_dir,
            self.processed_mols_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Create MSA directory if target_dir is provided
        if self.target_dir:
            self.msa_dir = self.target_dir / "msas"
            self.msa_dir.mkdir(exist_ok=True)

    @classmethod
    def _get_next_msa_id(cls) -> int:
        """Get the next globally unique MSA ID (negative integer)."""
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
        return [self._validate_sequence(chain_seq) for chain_seq in seq.split(":")]

    def _write_fasta_file_and_msas(
        self,
        protein_id: str,
        chains: List[str],
        msa_dir: Path,
    ) -> Path:
        """
        Write FASTA file in Boltz format with proper headers for process_input.
        Creates dummy MSA files for single-sequence mode with unique UUID-based MSA_ID per chain.

        Returns the path to the written FASTA file.
        """
        fasta_path = self.target_dir / f"{protein_id}.fasta"

        with open(fasta_path, "w") as f:
            for i, chain_seq in enumerate(chains):
                chain_name = chr(ord("A") + i)  # A, B, C, etc.

                # Create unique MSA_ID for this chain
                msa_taxonomy_id = self._get_next_msa_id()

                # Create dummy MSA file for this chain using the MSA_ID as filename
                msa_filename = f"{msa_taxonomy_id}.csv"  # Use MSA_ID as filename
                msa_path = msa_dir / msa_filename
                msa_path.parent.mkdir(parents=True, exist_ok=True)

                # Write dummy MSA with just the target sequence using MSA_ID
                with open(msa_path, "w") as msa_f:
                    msa_f.write("key,sequence\n")
                    msa_f.write(f"{msa_taxonomy_id},{chain_seq}\n")

                # Format: >CHAIN_ID|ENTITY_TYPE|MSA_PATH
                f.write(f">{chain_name}|protein|{msa_path}\n{chain_seq}\n")

        return fasta_path

    def _create_targets_from_sequences(
        self,
        sequences: List[str],
        protein_ids: List[str],
        ccd: Optional[Dict] = None,
        mol_dir: Optional[Path] = None,
    ) -> tuple[List[Record], Path]:
        """
        Internal method to create Target objects from sequences using process_input.
        This creates all necessary files that Boltz expects during inference.
        """

        if len(sequences) != len(protein_ids):
            raise ValueError("Number of sequences and protein IDs must match")

        if ccd is None:
            ccd = {}

        if mol_dir is None:
            # Use default cache directory
            mol_dir = Path.home() / ".boltz" / "mols"
            mol_dir.mkdir(parents=True, exist_ok=True)

        msa_dir = self.target_dir / "msa"
        msa_dir.mkdir(exist_ok=True)

        records = []
        for seq, protein_id in zip(sequences, protein_ids):
            # Parse chain breaks
            chain_sequences = self._parse_sequence_with_chain_breaks(seq)

            # Write FASTA file in Boltz format with MSA_IDs defined per chain
            fasta_path = self._write_fasta_file_and_msas(
                protein_id=protein_id, chains=chain_sequences, msa_dir=msa_dir
            )

            # Use process_input to create all necessary files
            try:
                process_input(
                    path=fasta_path,
                    ccd=ccd,
                    msa_dir=msa_dir,
                    mol_dir=mol_dir,
                    boltz2=True,
                    use_msa_server=False,
                    msa_server_url="",
                    msa_pairing_strategy="",
                    max_msa_seqs=0,
                    processed_msa_dir=self.processed_msa_dir,
                    processed_constraints_dir=self.processed_constraints_dir,
                    processed_templates_dir=self.processed_templates_dir,
                    processed_mols_dir=self.processed_mols_dir,
                    structure_dir=self.structure_dir,
                    records_dir=self.records_dir,
                )

                # Load the created record
                record_path = self.records_dir / f"{protein_id}.json"
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

        return records, self.processed_dir

    def from_sequences(
        self, sequences: List[str], protein_ids: List[str] = None
    ) -> tuple[Manifest, Path]:
        """
        Create processed input files from multiple sequences with optional chain breaks.

        Args:
            sequences: List of protein sequences. Use ':' to separate chains in multimers.
            protein_ids: Optional list of protein IDs. If None, uses "protein_0", etc.

        Returns:
            Tuple of (Manifest object, processed_data_dir_path)
        """
        records, processed_dir = self._create_targets_from_sequences(
            sequences, protein_ids
        )
        return Manifest(records=records), processed_dir

    def from_batch(
        self,
        aatypes: torch.Tensor,
        chain_idx: torch.Tensor,
        protein_ids: List[str],
    ) -> tuple[Manifest, Path]:
        """
        Create processed input files from batch tensors with chain indices.

        Args:
            aatypes: Tensor of shape (B, N) with amino acid type indices.
            chain_idx: Tensor of shape (B, N) with chain indices for each residue.
            protein_ids: Optional list of protein IDs. If None, uses "protein_0", etc.

        Returns:
            Tuple of (Manifest object, processed_data_dir_path)

        TODO should merge with AlphaFold function to create the same :-delimited multimer fasta
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
            sequence_with_breaks = ":".join(sorted_chains)
            sequences_with_breaks.append(sequence_with_breaks)

        # Use from_sequences to create processed files
        return self.from_sequences(sequences_with_breaks, protein_ids)

    def from_fasta(
        self, fasta_path: Union[str, Path], protein_ids: Optional[List[str]] = None
    ) -> tuple[Manifest, Path]:
        """
        Create processed input files from a FASTA file.

        Args:
            fasta_path: Path to FASTA file containing sequences to fold
            protein_ids: Optional list of protein IDs. If None, uses headers from FASTA.

        Returns:
            Tuple of (Manifest object, processed_data_dir_path)
        """
        assert os.path.exists(fasta_path), f"Fasta path {fasta_path} does not exist"

        fasta_path = Path(fasta_path)

        # Parse FASTA file to get sequences
        sequences = []
        headers = []

        with open(fasta_path, "r") as f:
            current_seq = ""
            current_header = None

            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    # Save previous sequence if exists
                    if current_header is not None and current_seq:
                        sequences.append(current_seq)
                        headers.append(current_header)

                    # Start new sequence
                    current_header = line[1:]  # Remove '>' prefix
                    current_seq = ""
                elif line:
                    current_seq += line

            # Save last sequence
            if current_header is not None and current_seq:
                sequences.append(current_seq)
                headers.append(current_header)

        if not sequences:
            raise ValueError(f"No sequences found in FASTA file: {fasta_path}")

        # Use provided protein_ids or default to FASTA headers
        if protein_ids is None:
            protein_ids = headers
        elif len(protein_ids) != len(sequences):
            raise ValueError(
                "Number of protein IDs must match number of sequences in FASTA"
            )

        # Use from_sequences to create processed files
        return self.from_sequences(sequences, protein_ids)


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
    ):
        """
        Initialize the Boltz predictor.

        """
        self.cfg = cfg

        # Set up precision with fallback logic
        if self.cfg.precision is None:
            self.precision = "bf16-mixed" if torch.cuda.is_available() else "32"
        else:
            self.precision = self.cfg.precision

        # Set up cache directory
        if self.cfg.cache_dir is None:
            cache_dir = get_cache_path()
        else:
            cache_dir = self.cfg.cache_dir
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Set up outputs directory
        if self.cfg.outputs_path is None:
            outputs_path = "./outputs"
        else:
            outputs_path = self.cfg.outputs_path
        self.outputs_path = Path(outputs_path).expanduser()
        self.outputs_path.mkdir(parents=True, exist_ok=True)

        # Set up checkpoint path
        if self.cfg.checkpoint_path is None:
            checkpoint_path = self.cache_dir / "boltz2_conf.ckpt"
        else:
            checkpoint_path = self.cfg.checkpoint_path
        self.checkpoint_path = Path(checkpoint_path)

        # Initialize model and trainer to None before setting device
        self.model = None
        self.trainer = None

        # setup default device
        self.set_device_id(None)

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

    def set_device_id(self, device_id: int):
        """Set the device ID for the folding tool."""
        if torch.cuda.is_available():
            if device_id is None:
                device = "cuda"
            else:
                device = f"cuda:{device_id}"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.device = torch.device(device)

        if self.model is not None:
            self.model.to(self.device)

        # TODO confirm trainer moved properly
        if self.trainer is not None:
            # Use string instead of device object for trainer accelerator
            if device.startswith("cuda"):
                self.trainer.accelerator = "gpu"
            elif device == "mps":
                self.trainer.accelerator = "mps"
            else:
                self.trainer.accelerator = "cpu"

    def _download_model_if_needed(self):
        """Download Boltz-2 model if not present in cache."""
        model_path = self.cache_dir / "boltz2_conf.ckpt"
        if not model_path.exists():
            download_boltz2(self.cache_dir)

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

        # TODO - upgrade to version 2.1.x that uses CUDA kernels,
        #   which defines `use_kernels` instead of `use_trifast`
        # (At writing, requires CUDA is available)

        self.model = Boltz2.load_from_checkpoint(
            self.checkpoint_path,
            strict=True,
            predict_args=predict_args,
            map_location="cpu",
            diffusion_process_args=asdict(self.diffusion_params),
            ema=False,
            use_trifast=self.device.type == "cuda",
            pairformer_args=asdict(self.pairformer_args),
            msa_args=asdict(self.msa_args),
            steering_args=asdict(self.steering_args),
        )
        self.model.eval()
        self.model.to(self.device)

        self.trainer = Trainer(
            accelerator=self.cfg.accelerator,
            precision=self.precision,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
        )

    def predict(
        self,
        manifest: Manifest,
        output_dir: Optional[Path] = None,
        processed_dir: Optional[Path] = None,
    ) -> BoltzPredictionSet:
        """
        Fold structures from a manifest.

        Args:
            manifest: Input manifest containing records to fold.
            output_dir: Output directory for results. If None, uses self.outputs_path.
            processed_dir: Directory containing processed input files (structures, records, etc.).
                          This should be the processed_dir returned by ManifestBuilder methods.

        Returns:
            BoltzPredictionSet containing BoltzPrediction objects for each protein in the manifest.
        """

        if output_dir is None:
            output_dir = self.outputs_path
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Require an empty output directory, so we can safely clean up outputs.
        for record in manifest.records:
            protein_output_dir = output_dir / record.id
            if protein_output_dir.exists() and len(os.listdir(protein_output_dir)) > 0:
                raise ValueError(
                    f"Output directory {protein_output_dir} is not empty. "
                    "Please provide an empty directory for new targets."
                )

        # Load model if not already loaded
        if self.model is None or self.trainer is None:
            self._load_model()

        # Use provided processed_dir or default to ManifestBuilder processed directory
        if processed_dir is None:
            # Default to the standard ManifestBuilder processed directory
            processed_path = Path("./outputs/targets/processed")
            if not processed_path.exists():
                raise ValueError(
                    "processed_dir must be provided when processed files are not in default location. "
                    "Use ManifestBuilder.from_sequences() or .from_batch() to create processed files, "
                    "then pass the returned processed_dir to this method."
                )
        else:
            processed_path = Path(processed_dir)

        # Verify processed directory structure exists
        required_subdirs = [
            "records",
            "structures",
            "msa",
            "constraints",
            "templates",
            "mols",
        ]
        for subdir in required_subdirs:
            subdir_path = processed_path / subdir
            if not subdir_path.exists():
                raise FileNotFoundError(f"Missing required subdirectory: {subdir_path}")

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

        # Create MSA directory (empty for sequence-only prediction)
        msa_dir = self.outputs_path / "msa"
        msa_dir.mkdir(exist_ok=True)

        # Ensure molecule directory exists (for CCD - Common Component Dictionary)
        mol_dir = self.cache_dir / "mols"
        mol_dir.mkdir(exist_ok=True)

        # Set up data module using processed data directories
        data_module = Boltz2InferenceDataModule(
            manifest=manifest,
            # Point to structures directory where NPZ files are
            target_dir=processed_path / "structures",
            msa_dir=processed_path / "msa",
            mol_dir=mol_dir,
            num_workers=1,
            constraints_dir=processed_path / "constraints",
            template_dir=processed_path / "templates",
            extra_mols_dir=processed_path / "mols",
            override_method=None,
            affinity=False,
        )

        # Set up writer
        writer = BoltzWriter(
            data_dir=str(processed_path / "structures"),
            output_dir=str(output_dir),
            output_format=self.cfg.output_format,
            boltz2=True,
        )

        # Update trainer with writer callback
        self.trainer.callbacks = [writer]

        # Run prediction
        self.trainer.predict(
            self.model,
            datamodule=data_module,
            return_predictions=False,
        )

        # Create BoltzPrediction objects for each protein
        predictions = []
        for record in manifest.records:
            protein_output_dir = output_dir / record.id
            prediction = BoltzPrediction.from_output_dir(protein_output_dir)
            # remove unnecesary outputs
            prediction.clean_output_dir()
            predictions.append(prediction)

        return BoltzPredictionSet(predictions)

    def fold_fasta(
        self,
        fasta_path: Path,
        output_dir: Path,
    ) -> pd.DataFrame:
        """
        Fold a protein sequence from a FASTA file and save the results to an output directory.
        Each fasta entry should be an entry in the DataFrame.

        Implements the FoldingTool interface.
        """
        manifest_builder = BoltzManifestBuilder(target_dir=output_dir / "processed")
        manifest, processed_dir = manifest_builder.from_fasta(fasta_path)

        prediction_set = self.predict(
            manifest=manifest, output_dir=output_dir, processed_dir=processed_dir
        )

        return prediction_set.to_df()
