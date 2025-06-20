"""
Unified ProteinMPNN runner

This module provides a unified interface for running ProteinMPNN inverse folding,
supporting both native (in-memory model) and subprocess execution modes.
The native mode loads the model once and keeps it in memory for fast inference,
while subprocess mode calls the original ProteinMPNN script.

Running natively requires `LigandMPNN` is installed, and location specified in the config.

Note: lots of Claude generated code here. Which I blame on LigandMPNN 
having an outrageously complicated `main()` function and no straightforward 
way to create a model, keep it in memory, and call the inference functions. 
So much of it's main() is ported here to support some of its features.
"""

import copy
import importlib.util
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from cogeneration.config.base import ModelType, ProteinMPNNRunnerConfig
from cogeneration.data import all_atom, residue_constants
from cogeneration.type.batch import PredBatchProp as pbp
from cogeneration.type.str_enum import StrEnum

logger = logging.getLogger(__name__)

# Type aliases for amino acid formats to clarify which alphabet ordering is used
# ProteinMPNN uses alphabetical ordering: [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y]
# Project uses AlphaFold ordering: [A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V]
MPNNAATypes = torch.Tensor  # Amino acid types in ProteinMPNN alphabetical ordering
MPNNLogits = torch.Tensor  # Logits in ProteinMPNN alphabetical ordering (21,)
CogenAATypes = torch.Tensor  # Amino acid types in project's AlphaFold ordering
CogenLogits = torch.Tensor  # Logits in project's AlphaFold ordering (21,)


@dataclass
class NativeMPNNResult:
    """
    Result structure from native ProteinMPNN inference.

    This provides a consistent interface for ProteinMPNN results with support
    for both single structure and batch processing.
    """

    logits: CogenLogits  # All sequence logits in project's AlphaFold ordering
    confidence_scores: torch.Tensor  # Confidence scores for each sequence
    _sequences: Optional[CogenAATypes] = None  # Cached sequences (computed on demand)

    @property
    def sequences(self) -> CogenAATypes:
        """
        Generated amino acid sequences in project's AlphaFold ordering.

        Computed on-demand from logits using argmax sampling.

        Returns:
            sequences: Amino acid sequences in project's AlphaFold ordering
        """
        if self._sequences is None:
            # Generate sequences from logits using argmax
            self._sequences = torch.argmax(self.logits, dim=-1)
        return self._sequences

    def set_sequences(self, sequences: CogenAATypes) -> None:
        """
        Explicitly set the sequences (useful when sequences were generated via sampling).

        Args:
            sequences: Amino acid sequences in project's AlphaFold ordering
        """
        self._sequences = sequences

    @property
    def averaged_logits(self) -> CogenLogits:
        """
        Averaged logits across all generated sequences.

        For compatibility with existing code that expects averaged_logits.

        Returns:
            averaged_logits: Mean logits across the sequence dimension
        """
        if len(self.logits.shape) >= 2:
            # Average across the sequence dimension (first non-batch dimension)
            return torch.mean(
                self.logits, dim=-3 if len(self.logits.shape) == 4 else -2
            )
        else:
            return self.logits

    @property
    def all_logits(self) -> CogenLogits:
        """
        All sequence logits (alias for logits property).

        For compatibility with existing code.
        """
        return self.logits


class ProteinMPNNRunner:
    """
    Unified ProteinMPNN runner supporting both native and subprocess modes.

    The runner can operate in a few modes:
    1. Native mode (`use_native_runner=True`): Loads model into memory for fast repeated inference
    2. Subprocess mode (`use_native_runner=False`): Calls the original ProteinMPNN run.py script
    3. Batch mode (requires `use_native_runner=True`): Runs inference on a Batch in parallel, returns sequences + logits

    For fast inference, e.g. during Feynman-Kac steering, use the `run_batch` method
    which operates directly on protein frames (translations and rotations)
    and returns logits without PDB I/O overhead.
    """

    def __init__(self, config: ProteinMPNNRunnerConfig):
        """
        Initialize the ProteinMPNN runner.

        Args:
            config: ProteinMPNN configuration object
        """
        self.config = config  # TODO rename to cfg
        self._model = None
        self._model_sc = None  # Side chain packing model
        self._device = None
        self._checkpoint_path = None
        self._ligandmpnn_modules = {}  # Cache for loaded modules

        # Set up device
        # TODO support its own device more explicitly
        if self.config.use_native_runner:
            self._device = torch.device(self.config.accelerator)
            logger.info(
                f"ProteinMPNN native runner initialized on device: {self._device}"
            )

    def _create_mpnn_to_cogen_conversion_map(self) -> torch.Tensor:
        """
        Create mapping tensor to convert from ProteinMPNN logits to project logits.

        ProteinMPNN uses alphabetical ordering: [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y]
        Project uses AlphaFold ordering: [A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V]

        Returns:
            conversion_map: Tensor of shape (21,) where conversion_map[mpnn_idx] = cogen_idx
                           Index 20 handles X (unknown) amino acids by mapping to A (alanine)
        """
        data_utils = self._load_ligandmpnn_module("data_utils")
        conversion_map = torch.zeros(
            21, dtype=torch.long
        )  # Include space for X at index 20

        # Map each ProteinMPNN index to corresponding project index
        for mpnn_idx, aa_letter in data_utils.restype_int_to_str.items():
            if aa_letter == "X":
                # Handle unknown amino acids by mapping to alanine (index 0 in project ordering)
                conversion_map[mpnn_idx] = 0  # A = 0 in project ordering
            elif aa_letter in residue_constants.restype_order:
                # Use standard 20 amino acid mapping (without X)
                cogen_idx = residue_constants.restype_order[aa_letter]
                conversion_map[mpnn_idx] = cogen_idx
            else:
                # Fallback: map unknown amino acids to alanine
                conversion_map[mpnn_idx] = 0  # A = 0 in project ordering

        return conversion_map

    def _convert_mpnn_logits_to_cogen(self, mpnn_logits: MPNNLogits) -> CogenLogits:
        """
        Convert logits from ProteinMPNN format to project format.

        Args:
            mpnn_logits: Logits in ProteinMPNN alphabetical ordering (may be 20 or 21 dimensional)

        Returns:
            cogen_logits: Logits reordered to project's AlphaFold ordering (21 dimensional)
        """
        conversion_map = self._create_mpnn_to_cogen_conversion_map()
        conversion_map = conversion_map.to(mpnn_logits.device)

        # Handle both 20 and 21 dimensional logits
        original_shape = mpnn_logits.shape
        last_dim = original_shape[-1]

        if last_dim == 20:
            # Standard 20 amino acids - pad with zeros for X
            pad_shape = list(original_shape)
            pad_shape[-1] = 21
            mpnn_logits_padded = torch.zeros(
                pad_shape, device=mpnn_logits.device, dtype=mpnn_logits.dtype
            )
            mpnn_logits_padded[..., :20] = mpnn_logits
            mpnn_logits = mpnn_logits_padded
        elif last_dim == 21:
            # Already includes X - use as is
            pass
        else:
            raise ValueError(
                f"Unsupported logits last dimension: {last_dim}, expected 20 or 21"
            )

        # Create the output tensor (always 21 dimensional for project format)
        output_shape = list(mpnn_logits.shape)
        cogen_logits = torch.zeros(
            output_shape, device=mpnn_logits.device, dtype=mpnn_logits.dtype
        )

        # Handle different input shapes
        if len(original_shape) == 1:
            # Single sequence logits (20 or 21,)
            for mpnn_idx in range(min(21, mpnn_logits.shape[0])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[cogen_idx] += mpnn_logits[
                    mpnn_idx
                ]  # Use += to handle duplicate mappings (X->A)
        elif len(original_shape) == 2:
            # Batch of sequence logits (N, 20 or 21)
            for mpnn_idx in range(min(21, mpnn_logits.shape[1])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[:, cogen_idx] += mpnn_logits[:, mpnn_idx]
        elif len(original_shape) == 3:
            # Batch of multiple sequence logits (B, N, 20 or 21) or (num_sequences, N, 20 or 21)
            for mpnn_idx in range(min(21, mpnn_logits.shape[2])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[:, :, cogen_idx] += mpnn_logits[:, :, mpnn_idx]
        elif len(original_shape) == 4:
            # Batch of multiple sequence logits (B, num_sequences, N, 20 or 21)
            for mpnn_idx in range(min(21, mpnn_logits.shape[3])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[:, :, :, cogen_idx] += mpnn_logits[:, :, :, mpnn_idx]
        else:
            raise ValueError(f"Unsupported logits shape: {original_shape}")

        return cogen_logits

    def _convert_cogen_aatypes_to_mpnn(
        self, cogen_aatypes: CogenAATypes
    ) -> MPNNAATypes:
        """
        Convert amino acid types from project format to ProteinMPNN format.

        Args:
            cogen_aatypes: Amino acid types in project's AlphaFold ordering

        Returns:
            mpnn_aatypes: Amino acid types in ProteinMPNN alphabetical ordering
        """
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Convert project integers to amino acid letters using project's mapping
        project_int_to_restype = {
            v: k for k, v in residue_constants.restype_order.items()
        }
        aa_letters = []
        for aa in cogen_aatypes.flatten():
            aa_int = int(aa)
            if aa_int == 20:  # X (unknown) amino acid in project format
                aa_letters.append("A")  # Convert X to A (alanine)
            elif aa_int in project_int_to_restype:
                aa_letters.append(project_int_to_restype[aa_int])
            else:
                aa_letters.append("A")  # Fallback to alanine for unknown indices

        # Convert amino acid letters to ProteinMPNN integers
        mpnn_aatypes_list = []
        for aa_letter in aa_letters:
            mpnn_int = data_utils.restype_str_to_int.get(
                aa_letter, 0
            )  # Default to A=0 if not found
            mpnn_aatypes_list.append(mpnn_int)

        mpnn_aatypes = torch.tensor(
            mpnn_aatypes_list, device=cogen_aatypes.device, dtype=torch.long
        )
        return mpnn_aatypes.reshape(cogen_aatypes.shape)

    def _load_ligandmpnn_module(self, module_name: str):
        """
        Load a LigandMPNN module using importlib without modifying sys.path.

        Args:
            module_name: Name of the module to load (e.g., 'data_utils', 'model_utils')

        Returns:
            Loaded module
        """
        if module_name in self._ligandmpnn_modules:
            return self._ligandmpnn_modules[module_name]

        ligandmpnn_path = self.config.pmpnn_path
        if not ligandmpnn_path.exists():
            raise FileNotFoundError(f"LigandMPNN path not found: {ligandmpnn_path}")

        module_file = ligandmpnn_path / f"{module_name}.py"
        if not module_file.exists():
            raise ImportError(f"Module {module_name}.py not found in {ligandmpnn_path}")

        # Load module using importlib
        spec = importlib.util.spec_from_file_location(module_name, module_file)
        if spec is None or spec.loader is None:
            raise ImportError(
                f"Could not load spec for {module_name} from {module_file}"
            )

        module = importlib.util.module_from_spec(spec)

        # Add LigandMPNN path to sys.modules temporarily for relative imports
        # but don't modify sys.path globally
        old_path = sys.path[:]
        try:
            if str(ligandmpnn_path) not in sys.path:
                sys.path.insert(0, str(ligandmpnn_path))
            spec.loader.exec_module(module)
        finally:
            sys.path[:] = old_path

        # Cache the module
        self._ligandmpnn_modules[module_name] = module

        return module

    def _load_model(self) -> torch.nn.Module:
        """
        Load the ProteinMPNN model into memory.

        Returns:
            Loaded ProteinMPNN model
        """
        if self._model is not None:
            return self._model

        try:
            # Load required modules
            model_utils = self._load_ligandmpnn_module("model_utils")
            ProteinMPNN = model_utils.ProteinMPNN

            # Get checkpoint path
            checkpoint_path = self._get_checkpoint_path()

            # Load checkpoint
            logger.info(f"Loading ProteinMPNN checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self._device)

            # Determine model parameters based on model type
            if self.config.model_type == ModelType.LIGAND_MPNN:
                atom_context_num = checkpoint.get("atom_context_num", 16)
                ligand_mpnn_use_side_chain_context = (
                    self.config.ligand_mpnn_use_side_chain_context
                )
                k_neighbors = checkpoint.get("num_edges", 48)
            else:
                atom_context_num = 1
                ligand_mpnn_use_side_chain_context = False
                k_neighbors = checkpoint.get("num_edges", 48)

            # Initialize model
            self._model = ProteinMPNN(
                node_features=128,
                edge_features=128,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                k_neighbors=k_neighbors,
                device=self._device,
                atom_context_num=atom_context_num,
                model_type=self.config.model_type,
                ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
            )

            # Load state dict
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Successfully loaded {self.config.model_type} model")
            return self._model

        except ImportError as e:
            logger.error(f"Failed to import LigandMPNN modules: {e}")
            logger.error(
                "Make sure LigandMPNN is installed and available in the specified path"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load ProteinMPNN model: {e}")
            raise

    def _load_side_chain_model(self) -> Optional[torch.nn.Module]:
        """
        Load the side chain packing model if needed.

        Returns:
            Loaded side chain packing model or None
        """
        if not self.config.pack_side_chains:
            return None

        if self._model_sc is not None:
            return self._model_sc

        try:
            # Temporary fix for NumPy compatibility with LigandMPNN
            # LigandMPNN uses deprecated np.int which was removed in newer NumPy versions
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                if not hasattr(np, "int"):
                    np.int = int
                if not hasattr(np, "float"):
                    np.float = float
                if not hasattr(np, "bool"):
                    np.bool = bool
                if not hasattr(np, "complex"):
                    np.complex = complex

            # Load required modules
            sc_utils = self._load_ligandmpnn_module("sc_utils")
            Packer = sc_utils.Packer

            checkpoint_path_sc = self.config.checkpoint_path_sc
            if checkpoint_path_sc is None:
                weights_path = self.config.pmpnn_path / self.config.pmpnn_weights_dir

                # Default side chain checkpoint filename
                checkpoint_filename = "ligandmpnn_sc_v_32_002_16.pt"

                # Try weights directory first if specified
                if weights_path is not None:
                    possible_sc_paths = [
                        weights_path / checkpoint_filename,
                        self.config.pmpnn_path / "model_params" / checkpoint_filename,
                    ]
                    for path in possible_sc_paths:
                        if path.exists():
                            checkpoint_path_sc = path
                            break

                # Fallback to ligandmpnn_path if not found in weights directory
                if checkpoint_path_sc is None:
                    checkpoint_path_sc = self.config.pmpnn_path / checkpoint_filename

            if not checkpoint_path_sc.exists():
                logger.warning(
                    f"Side chain packing checkpoint not found: {checkpoint_path_sc}"
                )
                return None

            logger.info(f"Loading side chain packing model: {checkpoint_path_sc}")

            self._model_sc = Packer(
                node_features=128,
                edge_features=128,
                num_positional_embeddings=16,
                num_chain_embeddings=16,
                num_rbf=16,
                hidden_dim=128,
                num_encoder_layers=3,
                num_decoder_layers=3,
                atom_context_num=16,
                lower_bound=0.0,
                upper_bound=20.0,
                top_k=32,
                dropout=0.0,
                augment_eps=0.0,
                atom37_order=False,
                device=self._device,
                num_mix=3,
            )

            checkpoint_sc = torch.load(checkpoint_path_sc, map_location=self._device)
            self._model_sc.load_state_dict(checkpoint_sc["model_state_dict"])
            self._model_sc.to(self._device)
            self._model_sc.eval()

            logger.info("Successfully loaded side chain packing model")
            return self._model_sc

        except Exception as e:
            logger.warning(f"Failed to load side chain packing model: {e}")
            return None

    def _get_checkpoint_path(self) -> Path:
        """
        Get the checkpoint path for the specified model type.

        Returns:
            Path to the model checkpoint
        """
        if self._checkpoint_path is not None:
            return self._checkpoint_path

        ligandmpnn_path = self.config.pmpnn_path
        weights_path = ligandmpnn_path / self.config.pmpnn_weights_dir

        # Map model types to checkpoint files
        checkpoint_mapping = {
            ModelType.PROTEIN_MPNN: "proteinmpnn_v_48_020.pt",
            ModelType.LIGAND_MPNN: "ligandmpnn_v_32_010_25.pt",
            ModelType.SOLUBLE_MPNN: "solublempnn_v_48_020.pt",
            ModelType.MEMBRANE_MPNN: "proteinmpnn_v_48_020.pt",  # Same as protein_mpnn
            ModelType.GLOBAL_MEMBRANE_MPNN: "global_membrane_mpnn_v_48_020.pt",
        }

        checkpoint_file = checkpoint_mapping.get(self.config.model_type)
        if checkpoint_file is None:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        # Try different locations for the checkpoint
        possible_paths = []

        # If weights path is specified, search there first
        if weights_path is not None:
            possible_paths.extend(
                [
                    weights_path / checkpoint_file,
                    ligandmpnn_path / "model_params" / checkpoint_file,
                ]
            )

        # Fallback to original locations
        possible_paths.extend(
            [
                ligandmpnn_path / checkpoint_file,
                ligandmpnn_path / "model_params" / checkpoint_file,
                Path(checkpoint_file),  # Current directory
            ]
        )

        for path in possible_paths:
            if path.exists():
                self._checkpoint_path = path
                return path

        raise FileNotFoundError(
            f"Could not find checkpoint for {self.config.model_type}. "
            f"Tried: {[str(p) for p in possible_paths]}"
        )

    def _parse_bias_AA(self, bias_str: Optional[str]) -> torch.Tensor:
        """
        Parse amino acid bias string into tensor.

        Args:
            bias_str: Bias string in format "A:-1.024,P:2.34"

        Returns:
            Bias tensor of shape [21]
        """
        # Load required module
        data_utils = self._load_ligandmpnn_module("data_utils")

        bias_AA = torch.zeros([21], device=self._device, dtype=torch.float32)
        if bias_str:
            try:
                items = [item.split(":") for item in bias_str.split(",")]
                for aa, bias_val in items:
                    if aa in data_utils.restype_str_to_int:
                        bias_AA[data_utils.restype_str_to_int[aa]] = float(bias_val)
            except Exception as e:
                logger.warning(f"Failed to parse bias_AA '{bias_str}': {e}")
        return bias_AA

    def _parse_omit_AA(self, omit_str: str) -> torch.Tensor:
        """
        Parse amino acid omission string into tensor.

        Args:
            omit_str: String of amino acids to omit, e.g., "ACG"

        Returns:
            Omission tensor of shape [21]
        """
        # Load required module
        data_utils = self._load_ligandmpnn_module("data_utils")

        omit_AA = torch.tensor(
            np.array([AA in omit_str for AA in data_utils.alphabet]).astype(np.float32),
            device=self._device,
        )
        return omit_AA

    def generate_fasta(
        self,
        pdb_path: Path,
        output_dir: Path,
        device_id: Optional[int] = None,
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        chains_to_design: Optional[List[str]] = None,
        fixed_residues: Optional[List[str]] = None,
        redesigned_residues: Optional[List[str]] = None,
        bias_AA: Optional[str] = None,
        omit_AA: Optional[str] = None,
        parse_these_chains_only: Optional[List[str]] = None,
        **kwargs,
    ) -> Path:
        """
        Run ProteinMPNN inverse folding on a PDB file.

        Args:
            pdb_path: Path to input PDB file
            output_dir: Directory to save output files
            device_id: GPU device ID (required for subprocess mode)
            num_sequences: Number of sequences to generate
            seed: Random seed
            temperature: Sampling temperature
            chains_to_design: List of chain IDs to design
            fixed_residues: List of residue positions to keep fixed
            redesigned_residues: List of residue positions to redesign
            bias_AA: Amino acid bias string (format: "A:-1.024,P:2.34")
            omit_AA: Amino acids to omit (format: "ACG")
            parse_these_chains_only: Chains to parse from PDB
            **kwargs: Additional arguments

        Returns:
            Path to output FASTA file
        """
        if self.config.use_native_runner:
            return self.run_native(
                pdb_path=pdb_path,
                output_dir=output_dir,
                device_id=device_id,
                num_sequences=num_sequences,
                seed=seed,
                temperature=temperature,
                chains_to_design=chains_to_design,
                fixed_residues=fixed_residues,
                redesigned_residues=redesigned_residues,
                bias_AA=bias_AA,
                omit_AA=omit_AA,
                parse_these_chains_only=parse_these_chains_only,
                **kwargs,
            )
        else:
            return self.run_subprocess(
                pdb_path=pdb_path,
                output_dir=output_dir,
                device_id=device_id,
                num_sequences=num_sequences,
                seed=seed,
                **kwargs,
            )

    def _process_fasta_output(
        self, original_fasta: Path, output_dir: Path, pdb_stem: str
    ) -> Path:
        """
        Process and clean up FASTA output from ProteinMPNN.

        This method skips the native sequence (first record) and renames the
        generated sequences with a cleaner ID format.

        Returns:
            Path to processed FASTA file
        """
        output_fasta = output_dir / f"{pdb_stem}_sequences.fa"

        records = []
        seq_counter = 1  # Counter for generated sequences
        for i, record in enumerate(SeqIO.parse(original_fasta, "fasta")):
            # skip the first record (native sequence)
            if i == 0:
                continue

            # Rename to a cleaner and unique sequence id (assuming unique pdb_names)
            new_id = f"{pdb_stem}_seq_{seq_counter}"
            new_record = SeqRecord(
                seq=record.seq,
                id=new_id,
                description=f"ProteinMPNN generated sequence {seq_counter}",
            )
            records.append(new_record)
            seq_counter += 1

        # Write processed sequences
        SeqIO.write(records, output_fasta, "fasta")
        logger.info(f"Processed {len(records)} sequences saved to {output_fasta}")

        return output_fasta

    def run_native(
        self,
        pdb_path: Path,
        output_dir: Path,
        device_id: Optional[int] = None,
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        chains_to_design: Optional[List[str]] = None,
        fixed_residues: Optional[List[str]] = None,
        redesigned_residues: Optional[List[str]] = None,
        bias_AA: Optional[str] = None,
        omit_AA: Optional[str] = None,
        parse_these_chains_only: Optional[List[str]] = None,
        **kwargs,
    ) -> Path:
        """
        Run ProteinMPNN using native (in-memory) model.

        Returns:
            Path to output FASTA file
        """
        # Import required modules
        try:
            # Load required modules using importlib
            data_utils = self._load_ligandmpnn_module("data_utils")

        except ImportError as e:
            logger.error(f"Failed to import LigandMPNN data_utils: {e}")
            raise

        # Validate inputs
        if not pdb_path.exists():
            raise FileNotFoundError(f"PDB file not found: {pdb_path}")

        # Set defaults
        num_sequences = num_sequences or self.config.seq_per_sample
        seed = seed or self.config.pmpnn_seed
        temperature = temperature or self.config.temperature
        omit_AA = omit_AA or self.config.omit_AA
        bias_AA = bias_AA or self.config.bias_AA

        # Set random seeds
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load model and move to specified device if provided
        model = self._load_model()
        if device_id is not None:
            device = torch.device(
                f"cuda:{device_id}" if torch.cuda.is_available() else "cpu"
            )
            model = model.to(device)
            self._device = device
            logger.info(f"Moved model to device: {device}")

        # Load side chain model if needed and move to device
        model_sc = None
        if self.config.pack_side_chains:
            model_sc = self._load_side_chain_model()
            if model_sc is not None and device_id is not None:
                model_sc = model_sc.to(device)
                logger.info(f"Moved side chain model to device: {device}")

        logger.info(f"Running native ProteinMPNN on {pdb_path}")
        logger.info(
            f"Model type: {self.config.model_type}, Temperature: {temperature}, Sequences: {num_sequences}"
        )

        # Parse PDB and create protein_dict
        protein_dict = self._parse_pdb_to_protein_dict(
            pdb_path=pdb_path,
            parse_these_chains_only=parse_these_chains_only,
        )

        # Run inference using shared method
        inference_result = self._run_inference_on_protein_dict(
            protein_dict=protein_dict,
            temperature=temperature,
            num_sequences=num_sequences,
            bias_AA=bias_AA,
            omit_AA=omit_AA,
            chains_to_design=chains_to_design,
            fixed_residues=fixed_residues,
            redesigned_residues=redesigned_residues,
            model=model,
        )

        # Save results to FASTA using shared method
        output_fasta = self._save_inference_results_to_fasta(
            inference_result=inference_result,
            protein_dict=protein_dict,
            output_dir=output_dir,
            pdb_path=pdb_path,
            temperature=temperature,
            seed=seed,
            num_sequences=num_sequences,
        )

        # Apply side chain packing if enabled and model is available
        if self.config.pack_side_chains and model_sc is not None:
            self._apply_side_chain_packing(
                inference_result=inference_result,
                protein_dict=protein_dict,
                model_sc=model_sc,
                output_dir=output_dir,
                pdb_path=pdb_path,
                num_sequences=num_sequences,
            )

        # Process FASTA output
        processed_fasta = self._process_fasta_output(
            output_fasta, output_dir, pdb_path.stem
        )

        logger.info(f"Native ProteinMPNN completed successfully")
        return processed_fasta

    def run_subprocess(
        self,
        pdb_path: Path,
        output_dir: Path,
        device_id: Optional[int],
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
        **kwargs,
    ) -> Path:
        """
        Run ProteinMPNN using subprocess (original script).

        Returns:
            Path to output FASTA file
        """
        # Validate inputs
        assert pdb_path.exists(), f"PDB path does not exist: {pdb_path}"
        assert device_id is not None, "device_id is required for subprocess mode"

        # Set defaults
        num_sequences = num_sequences or self.config.seq_per_sample
        seed = seed or self.config.pmpnn_seed

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Copy PDB to output directory
        pdb_copy_path = output_dir / pdb_path.name
        shutil.copy2(pdb_path, pdb_copy_path)

        # Construct ProteinMPNN command
        run_script = self.config.pmpnn_path / "run.py"
        assert run_script.exists(), f"ProteinMPNN run.py not found: {run_script}"

        cmd = [
            "python",
            str(run_script),
            "--model_type",
            self.config.model_type,
            "--pdb_path",
            str(pdb_copy_path),
            "--out_folder",
            str(output_dir),
            "--num_seq_per_target",
            str(num_sequences),
            "--sampling_temp",
            str(self.config.temperature),
            "--seed",
            str(seed),
            "--batch_size",
            "1",
        ]

        # Add additional arguments
        if self.config.bias_AA:
            cmd.extend(["--bias_AA", self.config.bias_AA])
        if self.config.omit_AA:
            cmd.extend(["--omit_AA", self.config.omit_AA])

        logger.info(f"Running ProteinMPNN subprocess: {' '.join(cmd)}")

        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(device_id)

        # Execute command
        try:
            result = subprocess.run(
                cmd,
                cwd=self.config.pmpnn_path,
                env=env,
                capture_output=True,
                text=True,
                timeout=600,  # 10 min timeout
            )

            if result.returncode != 0:
                logger.error(
                    f"ProteinMPNN subprocess failed with return code {result.returncode}"
                )
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, cmd)

            logger.info("ProteinMPNN subprocess completed successfully")

        except subprocess.TimeoutExpired:
            logger.error("ProteinMPNN subprocess timed out")
            raise
        except Exception as e:
            logger.error(f"ProteinMPNN subprocess failed: {e}")
            raise

        # Find and process output FASTA file
        seqs_dir = output_dir / "seqs"
        if not seqs_dir.exists():
            raise FileNotFoundError(
                f"ProteinMPNN output directory not found: {seqs_dir}"
            )

        # Find the output FASTA file
        fasta_files = list(seqs_dir.glob("*.fa"))
        if not fasta_files:
            raise FileNotFoundError(f"No FASTA files found in {seqs_dir}")

        original_fasta = fasta_files[0]

        # Process FASTA output
        processed_fasta = self._process_fasta_output(
            original_fasta, output_dir, pdb_path.stem
        )

        return processed_fasta

    # TODO run_batch should take `num_logits` or `is_masking` to determine whether 20 or 21 logits are used

    def run_batch(
        self,
        pred_trans: torch.Tensor,  # (B, N, 3)
        pred_rotmats: torch.Tensor,  # (B, N, 3, 3)
        pred_aatypes: CogenAATypes,  # (B, N) - amino acid types in project's AlphaFold ordering
        res_mask: torch.Tensor,  # (B, N) - residue mask
        pred_torsions: Optional[torch.Tensor] = None,  # (B, N, 7, 2)
        chain_idx: Optional[torch.Tensor] = None,  # (B, N) - chain indices
        num_sequences: Optional[int] = 1,
        temperature: Optional[float] = None,
        chains_to_design: Optional[List[str]] = None,
        fixed_residues: Optional[List[str]] = None,
        redesigned_residues: Optional[List[str]] = None,
        bias_AA: Optional[str] = None,
        omit_AA: Optional[str] = None,
        batch_size_limit: Optional[int] = None,  # memory management
        **kwargs,
    ) -> NativeMPNNResult:
        """
        Run ProteinMPNN on a batch of protein frames directly.

        This method is optimized for use in Feynman-Kac steering where ProteinMPNN
        is called frequently during inference. It avoids PDB I/O and operates directly
        on the model's frame representation.

        Args:
            pred_trans: Predicted frame translations (B, N, 3)
            pred_rotmats: Predicted frame rotations (B, N, 3, 3)
            pred_aatypes: Predicted amino acid types in project's AlphaFold ordering (B, N)
            res_mask: Residue mask indicating valid positions (B, N)
            pred_torsions: Optional predicted torsion angles (B, N, 7, 2)
            chain_idx: Optional chain indices (B, N), defaults to single chain
            num_sequences: Number of sequences to generate per batch element (per structure)
            temperature: Sampling temperature, defaults to config value
            chains_to_design: List of chain IDs to design
            fixed_residues: List of residue positions to keep fixed
            redesigned_residues: List of residue positions to redesign
            bias_AA: Amino acid bias string (format: "A:-1.024,P:2.34")
            omit_AA: Amino acids to omit (format: "ACG")
            batch_size_limit: Maximum batch size for processing (for memory management)
            **kwargs: Additional arguments

        Returns:
            NativeMPNNResult containing:
                - sequences: Generated amino acid sequences in project's AlphaFold ordering (B, num_sequences, N)
                - logits: All sequence logits in project's AlphaFold ordering (B, num_sequences, N, 21)
                - confidence_scores: Confidence scores for each sequence (B, num_sequences)
        """
        if not self.config.use_native_runner:
            raise ValueError("run_batch only supports native runner mode")

        # Import required modules
        try:
            data_utils = self._load_ligandmpnn_module("data_utils")
        except ImportError as e:
            logger.error(f"Failed to import LigandMPNN data_utils: {e}")
            raise

        # Move all input tensors to the runner's device and handle MPS compatibility
        def move_tensor_to_device(tensor: torch.Tensor) -> torch.Tensor:
            """Move tensor to runner's device with MPS float64 compatibility."""
            if self._device.type == "mps" and tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor.to(self._device)

        pred_trans = move_tensor_to_device(pred_trans)
        pred_rotmats = move_tensor_to_device(pred_rotmats)
        pred_aatypes = move_tensor_to_device(pred_aatypes)
        res_mask = move_tensor_to_device(res_mask)

        if pred_torsions is not None:
            pred_torsions = move_tensor_to_device(pred_torsions)

        if chain_idx is not None:
            chain_idx = move_tensor_to_device(chain_idx)
        else:
            chain_idx = torch.ones_like(res_mask)

        B, N = pred_trans.shape[:2]

        # Set defaults
        temperature = temperature or self.config.temperature
        omit_AA = omit_AA or self.config.omit_AA
        bias_AA = bias_AA or self.config.bias_AA
        batch_size_limit = batch_size_limit or max(
            8, B // 4
        )  # Process in chunks if large batch

        # Load model
        model = self._load_model()

        # Process batch in chunks if needed for memory management
        if B > batch_size_limit:
            logger.info(
                f"Processing large batch of {B} structures in chunks of {batch_size_limit}"
            )
            chunk_results = []

            for chunk_start in range(0, B, batch_size_limit):
                chunk_end = min(chunk_start + batch_size_limit, B)

                # Extract chunk
                chunk_trans = pred_trans[chunk_start:chunk_end]
                chunk_rotmats = pred_rotmats[chunk_start:chunk_end]
                chunk_aatypes = pred_aatypes[chunk_start:chunk_end]
                chunk_res_mask = res_mask[chunk_start:chunk_end]
                chunk_chain_idx = chain_idx[chunk_start:chunk_end]
                chunk_torsions = (
                    pred_torsions[chunk_start:chunk_end]
                    if pred_torsions is not None
                    else None
                )

                # Process chunk recursively
                chunk_result = self.run_batch(
                    pred_trans=chunk_trans,
                    pred_rotmats=chunk_rotmats,
                    pred_aatypes=chunk_aatypes,
                    res_mask=chunk_res_mask,
                    pred_torsions=chunk_torsions,
                    chain_idx=chunk_chain_idx,
                    temperature=temperature,
                    num_sequences=num_sequences,
                    chains_to_design=chains_to_design,
                    fixed_residues=fixed_residues,
                    redesigned_residues=redesigned_residues,
                    bias_AA=bias_AA,
                    omit_AA=omit_AA,
                    batch_size_limit=None,  # Prevent infinite recursion
                )
                chunk_results.append(chunk_result)

            # Combine chunk results
            sequences = torch.cat([r.sequences for r in chunk_results], dim=0)
            all_logits = torch.cat([r.logits for r in chunk_results], dim=0)
            confidence_scores = torch.cat(
                [r.confidence_scores for r in chunk_results], dim=0
            )
            return NativeMPNNResult(
                logits=all_logits,
                confidence_scores=confidence_scores,
                _sequences=sequences,
            )

        # Convert frames to atom37 coordinates for the entire batch
        atom37 = all_atom.atom37_from_trans_rot(
            trans=pred_trans,  # (B, N, 3)
            rots=pred_rotmats,  # (B, N, 3, 3)
            torsions=pred_torsions,  # (B, N, 7, 2) or None
            aatype=pred_aatypes,  # (B, N)
            res_mask=res_mask,  # (B, N)
            unknown_to_alanine=True,
        )  # (B, N, 37, 3)

        # Convert amino acid types from project format to MPNN format
        mpnn_aatypes = self._convert_cogen_aatypes_to_mpnn(pred_aatypes)  # (B, N)

        # Create batch protein_dict for all structures
        batch_protein_dict = self._create_protein_dict_from_frames(
            atom37=atom37,  # (B, N, 37, 3)
            aatypes=mpnn_aatypes,  # (B, N)
            chain_idx=chain_idx,  # (B, N)
            device=self._device,
            res_mask=res_mask,  # (B, N)
        )

        # Run inference on the entire batch
        batch_result = self._run_inference_on_protein_dict(
            protein_dict=batch_protein_dict,
            temperature=temperature,
            num_sequences=num_sequences,
            bias_AA=bias_AA,
            omit_AA=omit_AA,
            chains_to_design=chains_to_design,
            fixed_residues=fixed_residues,
            redesigned_residues=redesigned_residues,
            model=model,
        )

        # Extract results
        logits = batch_result.logits  # (B, num_sequences, N, 21)
        confidence_scores = batch_result.confidence_scores  # (B, num_sequences)
        sequences = batch_result.sequences  # (B, num_sequences, N)

        # Convert from MPNN format to Cogen format
        logits = self._convert_mpnn_logits_to_cogen(logits)
        sequences = self._convert_mpnn_sequences_to_cogen(sequences)

        return NativeMPNNResult(
            logits=logits,
            confidence_scores=confidence_scores,
            _sequences=sequences,
        )

    def _create_protein_dict_from_frames(
        self,
        atom37: torch.Tensor,  # (N, 37, 3) or (B, N, 37, 3)
        aatypes: MPNNAATypes,  # (N,) or (B, N) - amino acid types in ProteinMPNN alphabetical ordering
        chain_idx: torch.Tensor,  # (N,) or (B, N)
        device: torch.device,
        res_mask: Optional[
            torch.Tensor
        ] = None,  # (B, N) - only required for batch inputs
    ) -> Dict:
        """
        Create a protein_dict that mimics the output of parse_PDB.

        This method creates ProteinMPNN-compatible data structures from frame coordinates
        and amino acid types. The input amino acid types should already be in ProteinMPNN's
        alphabetical ordering [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y].

        Supports both single structure and batch inputs by automatically detecting dimensionality.

        Args:
            atom37: Atom coordinates in atom37 format
                   - Single: (N, 37, 3)
                   - Batch: (B, N, 37, 3)
            aatypes: Amino acid types in ProteinMPNN alphabetical ordering
                    - Single: (N,)
                    - Batch: (B, N)
            chain_idx: Chain indices
                      - Single: (N,)
                      - Batch: (B, N)
            device: Device to put tensors on
            res_mask: Residue mask (only required for batch inputs)
                     - Batch: (B, N)

        Returns:
            protein_dict: Dictionary similar to parse_PDB output with ProteinMPNN-compatible data
                         For batch inputs, includes 'batch_size' field to indicate batch processing
        """
        # Load the required modules
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Detect if input is batched based on atom37 dimensions
        is_batched = len(atom37.shape) == 4  # (B, N, 37, 3) vs (N, 37, 3)

        if is_batched:
            return self._create_batch_protein_dict_impl(
                atom37, aatypes, chain_idx, res_mask, device, data_utils
            )
        else:
            return self._create_single_protein_dict_impl(
                atom37, aatypes, chain_idx, device, data_utils
            )

    def _create_single_protein_dict_impl(
        self,
        atom37: torch.Tensor,  # (N, 37, 3)
        aatypes: MPNNAATypes,  # (N,) - amino acid types in ProteinMPNN alphabetical ordering
        chain_idx: torch.Tensor,  # (N,)
        device: torch.device,
        data_utils,
    ) -> Dict:
        """Implementation for single structure protein_dict creation."""
        N = atom37.shape[0]

        # Convert ProteinMPNN amino acid integers to sequence string
        sequence_chars = []
        for aa_int in aatypes:
            # Convert ProteinMPNN integer to string
            mpnn_char = data_utils.restype_int_to_str.get(int(aa_int), "A")
            sequence_chars.append(mpnn_char)

        sequence = "".join(sequence_chars)

        # Create chain labels (letters A, B, C, etc.)
        unique_chains = torch.unique(chain_idx, sorted=True)
        chain_mapping = {}
        for i, chain_id in enumerate(unique_chains):
            chain_mapping[int(chain_id)] = chr(ord("A") + i)

        chain_letters_list = [chain_mapping[int(idx)] for idx in chain_idx]
        chain_letters = chain_letters_list  # Both field names are used

        # Convert to tensors/formats expected by featurize
        # Chain labels should be encoded as integers for featurize
        chain_label_to_int = {
            label: i for i, label in enumerate(set(chain_letters_list))
        }
        chain_labels_tensor = torch.tensor(
            [chain_label_to_int[label] for label in chain_letters_list], device=device
        )

        # Prepare the coordinates in the expected format
        # featurize expects X with shape (N, 4, 3) for N, CA, C, O atoms
        # atom37 format: [N, CA, C, O, CB, ...] - we need first 4
        X_backbone = atom37[:, :4, :]  # (N, 4, 3)

        protein_dict = {
            "coords": atom37,  # (N, 37, 3) - full atom coordinates
            "X": X_backbone,  # (N, 4, 3) - backbone coordinates for featurize
            "S": aatypes,  # (N,) - sequence as ProteinMPNN integers
            "seq": sequence,  # sequence as string using ProteinMPNN alphabet
            "mask": torch.ones(N, device=device),  # All residues are valid
            "R_idx": torch.arange(
                1, N + 1, device=device
            ),  # Residue indices starting from 1
            "chain_labels": chain_labels_tensor,  # Tensor of chain labels (for featurize)
            "chain_letters": chain_letters,  # List of chain letters (for inference)
            "chain_idx": chain_idx,  # Original chain indices
            "chain_mask": torch.ones(
                N, device=device
            ),  # All residues can be designed by default
        }

        return protein_dict

    def _create_batch_protein_dict_impl(
        self,
        atom37: torch.Tensor,  # (B, N, 37, 3)
        aatypes: MPNNAATypes,  # (B, N) - amino acid types in ProteinMPNN alphabetical ordering
        chain_idx: torch.Tensor,  # (B, N)
        res_mask: torch.Tensor,  # (B, N)
        device: torch.device,
        data_utils,
    ) -> Dict:
        """Implementation for batch protein_dict creation."""
        if res_mask is None:
            raise ValueError("res_mask is required for batch inputs")

        B, N = atom37.shape[:2]

        # Convert ProteinMPNN amino acid integers to sequence strings for each batch item
        batch_sequences = []
        batch_chain_letters_list = []

        for b in range(B):
            # Convert sequence for this batch item
            sequence_chars = []
            for aa_int in aatypes[b]:
                mpnn_char = data_utils.restype_int_to_str.get(int(aa_int), "A")
                sequence_chars.append(mpnn_char)
            batch_sequences.append("".join(sequence_chars))

            # Create chain labels (letters A, B, C, etc.) for this batch item
            unique_chains = torch.unique(chain_idx[b], sorted=True)
            chain_mapping = {}
            for i, chain_id in enumerate(unique_chains):
                chain_mapping[int(chain_id)] = chr(ord("A") + i)

            chain_letters_list = [chain_mapping[int(idx)] for idx in chain_idx[b]]
            batch_chain_letters_list.append(chain_letters_list)

        # Convert to batch tensors/formats expected by featurize
        # For batch processing, we'll use the first batch item's chain structure as template
        # This assumes all batch items have similar chain structure (common in practice)
        chain_label_to_int = {
            label: i for i, label in enumerate(set(batch_chain_letters_list[0]))
        }

        # Create chain labels tensor for each batch item
        batch_chain_labels = []
        for b in range(B):
            chain_labels_tensor = torch.tensor(
                [
                    chain_label_to_int.get(label, 0)
                    for label in batch_chain_letters_list[b]
                ],
                device=device,
            )
            batch_chain_labels.append(chain_labels_tensor)
        batch_chain_labels_tensor = torch.stack(batch_chain_labels, dim=0)  # (B, N)

        # Prepare the coordinates in the expected format
        # featurize expects X with shape (B, N, 4, 3) for N, CA, C, O atoms
        X_backbone = atom37[:, :, :4, :]  # (B, N, 4, 3)

        # For batch processing, we'll handle each structure separately but use the same protein_dict structure
        batch_protein_dict = {
            "coords": atom37,  # (B, N, 37, 3) - full atom coordinates
            "X": X_backbone,  # (B, N, 4, 3) - backbone coordinates for featurize
            "S": aatypes,  # (B, N) - sequence as ProteinMPNN integers
            "seq": batch_sequences,  # List of sequences as strings using ProteinMPNN alphabet
            "mask": res_mask.float(),  # (B, N) - residue mask converted to float for LigandMPNN compatibility
            "R_idx": torch.arange(1, N + 1, device=device)
            .unsqueeze(0)
            .repeat(B, 1),  # (B, N) - residue indices starting from 1
            "chain_labels": batch_chain_labels_tensor,  # (B, N) - tensor of chain labels (for featurize)
            "chain_letters": batch_chain_letters_list,  # List of lists of chain letters (for inference)
            "chain_idx": chain_idx,  # (B, N) - original chain indices
            "chain_mask": torch.ones_like(
                res_mask, device=device, dtype=torch.float32
            ),  # (B, N) - all residues can be designed by default, float format
            "batch_size": B,  # Mark as batch data
        }

        return batch_protein_dict

    def _run_inference_on_protein_dict(
        self,
        protein_dict: Dict,
        temperature: float,
        num_sequences: int,
        bias_AA: Optional[str],
        omit_AA: Optional[str],
        chains_to_design: Optional[List[str]],
        fixed_residues: Optional[List[str]],
        redesigned_residues: Optional[List[str]],
        model: torch.nn.Module,
    ) -> NativeMPNNResult:
        """
        Run inference on a protein_dict.

        ProteinMPNN generates each batch autoregressively, and it is recommended to generate multiple independent batches per input.
        --num_sequences is roughly equivalent to --number_of_batches in LigandMPNN.
        Each is generated independently.

        Args:
            protein_dict: Dictionary containing protein structures
            temperature: Sampling temperature
            num_sequences: Number of independent autoregressive sequences to generate per structure (equivalent to --number_of_batches in LigandMPNN)
            bias_AA: Amino acid bias string (optional)
            omit_AA: Amino acids to omit (optional)
            chains_to_design: Chains to design (optional)
            fixed_residues: Fixed residues (optional)
            redesigned_residues: Redesigned residues (optional)
            model: ProteinMPNN model

        Returns:
            NativeMPNNResult containing logits, confidence scores, and sequences
        """
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Check if this is a batch of structures or a single structure
        if "X" in protein_dict:
            X = protein_dict["X"]
            if X.dim() == 4:  # (B, N, 4, 3) - batch of structures
                B, N = X.shape[:2]
                is_batch = True
            elif X.dim() == 3:  # (N, 4, 3) - single structure
                B, N = 1, X.shape[0]
                is_batch = False
            else:
                raise ValueError(f"Unexpected X shape: {X.shape}")
        else:
            raise ValueError(
                "protein_dict must contain 'X' key with backbone coordinates"
            )

        logger.info(
            f"Generating {num_sequences} sequences with temperature {temperature}"
        )

        # Collect results for all structures
        all_batch_logits = []
        all_batch_sequences = []
        all_batch_confidence = []

        # Process each structure individually
        for struct_idx in range(B):
            # Extract single structure from batch
            if is_batch:
                single_protein_dict = {}
                for key, value in protein_dict.items():
                    if key == "_parsed_metadata":
                        single_protein_dict[key] = value
                        continue
                    if isinstance(value, torch.Tensor) and value.dim() >= 2:
                        single_protein_dict[key] = value[struct_idx]
                    else:
                        single_protein_dict[key] = value
            else:
                single_protein_dict = protein_dict

            # Featurize the single structure
            feature_dict = data_utils.featurize(
                single_protein_dict,
                cutoff_for_score=self.config.ligand_mpnn_cutoff_for_score,
                use_atom_context=True,
            )

            # NOTE: batch_size=1 always for autoregressive sampling, num_sequences controls number of independent runs
            # TODO - however, if we have a batch input, we can run each member of the batch simultaneously,
            #   so there are only `num_sequences` independent runs

            # Set up the feature dict for sampling sequences
            feature_dict["batch_size"] = 1
            feature_dict["temperature"] = temperature

            # Set up bias and omit tensors
            bias_tensor = torch.zeros(1, N, 21, device=self._device)
            omit_tensor = torch.zeros(1, N, 21, device=self._device)

            if bias_AA:
                bias_values = self._parse_bias_AA(
                    bias_AA
                )  # Returns tensor of shape [21]
                # Apply the same bias to all positions
                bias_tensor[0, :, :] = bias_values.unsqueeze(0).expand(N, -1)

            if omit_AA:
                omit_values = self._parse_omit_AA(
                    omit_AA
                )  # Returns tensor of shape [21]
                # Apply the same omit mask to all positions
                omit_tensor[0, :, :] = omit_values.unsqueeze(0).expand(N, -1)

            # Combine bias and omit
            feature_dict["bias"] = bias_tensor - 1e8 * omit_tensor

            # Set up chain mask
            if "chain_mask" in single_protein_dict:
                feature_dict["chain_mask"] = single_protein_dict[
                    "chain_mask"
                ].unsqueeze(0)
            else:
                feature_dict["chain_mask"] = torch.ones(1, N, device=self._device)

            # Set up symmetry (none for now)
            feature_dict["symmetry_residues"] = [[]]
            feature_dict["symmetry_weights"] = [[]]

            # Generate multiple sequences through independent autoregressive runs
            # This follows LigandMPNN's approach of running model.sample() multiple times
            all_S_samples = []
            all_log_probs = []
            all_sampling_probs = []

            with torch.no_grad():
                for seq_idx in range(num_sequences):
                    # Generate different random seed for each autoregressive sequence generation
                    # This is critical for ensuring independent samples from the autoregressive model
                    feature_dict["randn"] = torch.randn(1, N, device=self._device)

                    # Run autoregressive model inference - generates 1 sequence
                    output_dict = model.sample(feature_dict)

                    # Extract results
                    S_sample = output_dict["S"]  # (1, N)
                    log_probs = output_dict["log_probs"]  # (1, N, 21)
                    sampling_probs = output_dict.get(
                        "sampling_probs", log_probs
                    )  # (1, N, 21)

                    all_S_samples.append(S_sample)
                    all_log_probs.append(log_probs)
                    all_sampling_probs.append(sampling_probs)

            # Combine all sequences from independent autoregressive runs
            S_sample = torch.cat(all_S_samples, dim=0)  # (num_sequences, N)
            log_probs = torch.cat(all_log_probs, dim=0)  # (num_sequences, N, 21)
            sampling_probs = torch.cat(
                all_sampling_probs, dim=0
            )  # (num_sequences, N, 21)

            # Compute confidence scores for each generated sequence
            mask_for_score = feature_dict["mask"] * feature_dict["chain_mask"]
            confidence_scores = []

            for seq_idx in range(num_sequences):
                loss, _ = data_utils.get_score(
                    S_sample[seq_idx : seq_idx + 1],  # (1, N)
                    log_probs[seq_idx : seq_idx + 1],  # (1, N, 21)
                    mask_for_score,  # (1, N)
                )
                confidence = torch.exp(-loss).squeeze()  # Remove extra dimensions
                confidence_scores.append(confidence)

            # Store results for this structure
            all_batch_logits.append(log_probs)  # (num_sequences, N, 21)
            all_batch_sequences.append(S_sample)  # (num_sequences, N)
            all_batch_confidence.append(
                torch.stack(confidence_scores)
            )  # (num_sequences,)

        # Combine results from all structures
        batch_logits = torch.stack(all_batch_logits)  # (B, num_sequences, N, 21)
        batch_sequences = torch.stack(all_batch_sequences)  # (B, num_sequences, N)
        batch_confidence = torch.stack(all_batch_confidence)  # (B, num_sequences)

        logger.info(
            f"Successfully generated {num_sequences * B} independent autoregressive sequences across {B} structures"
        )

        if is_batch:
            # Return batch format
            return NativeMPNNResult(
                logits=batch_logits,
                confidence_scores=batch_confidence,
                _sequences=batch_sequences,
            )
        else:
            # Remove batch dimension (B=1) for single structure case
            all_logits_tensor = batch_logits[0]  # (num_sequences, N, 21)
            all_confidence_scores_tensor = batch_confidence[0]  # (num_sequences,)
            sequences = batch_sequences[0]  # (num_sequences, N)

            return NativeMPNNResult(
                logits=all_logits_tensor,
                confidence_scores=all_confidence_scores_tensor,
                _sequences=sequences,
            )

    def _parse_pdb_to_protein_dict(
        self,
        pdb_path: Path,
        parse_these_chains_only: Optional[List[str]] = None,
    ) -> Dict:
        """
        Parse PDB file into protein_dict format.

        This method handles the PDB parsing and initial setup that's common
        between run_native and potentially other methods.
        """
        # Load required module
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Parse PDB
        parse_all_atoms_flag = self.config.ligand_mpnn_use_side_chain_context or (
            self.config.pack_side_chains and not self.config.repack_everything
        )

        protein_dict, backbone, other_atoms, icodes, _ = data_utils.parse_PDB(
            str(pdb_path),
            device=str(self._device),
            chains=parse_these_chains_only or [],
            parse_all_atoms=parse_all_atoms_flag,
            parse_atoms_with_zero_occupancy=False,
        )

        # Add chain_mask for featurization - all residues can be designed by default
        if "S" in protein_dict:
            protein_dict["chain_mask"] = torch.ones_like(
                protein_dict["S"], device=self._device
            )
        elif "X" in protein_dict:
            # Use backbone coordinates to determine sequence length
            protein_dict["chain_mask"] = torch.ones(
                protein_dict["X"].shape[0], device=self._device
            )
        else:
            # Fallback - use any available mask
            for key in ["mask", "chain_labels"]:
                if key in protein_dict:
                    protein_dict["chain_mask"] = torch.ones_like(
                        protein_dict[key], device=self._device
                    )
                    break

        # Store additional info for side chain packing
        protein_dict["_parsed_metadata"] = {
            "backbone": backbone,
            "other_atoms": other_atoms,
            "icodes": icodes,
        }

        return protein_dict

    def _pad_result_to_original_length(
        self,
        structure_result: NativeMPNNResult,
        valid_mask: torch.Tensor,
        original_length: int,
        temperature: float,
    ) -> NativeMPNNResult:
        """
        Pad the inference result back to the original sequence length.

        This handles cases where only a subset of residues were processed.
        """
        device = valid_mask.device
        num_sequences = structure_result.logits.shape[0]

        # Create padded tensors
        padded_all_logits = torch.zeros(
            num_sequences, original_length, 21, device=device
        )
        padded_sequences = torch.zeros(
            num_sequences, original_length, device=device, dtype=torch.long
        )

        # Fill in valid positions
        padded_all_logits[:, valid_mask, :] = structure_result.logits
        padded_sequences[:, valid_mask] = structure_result.sequences

        # For invalid positions, set uniform probabilities (or very low confidence)
        invalid_mask = ~valid_mask
        if invalid_mask.any():
            # Set to uniform distribution in log space
            uniform_logits = torch.full(
                (21,), -torch.log(torch.tensor(21.0)), device=device
            )
            padded_all_logits[:, invalid_mask, :] = uniform_logits

        return NativeMPNNResult(
            logits=padded_all_logits,
            confidence_scores=structure_result.confidence_scores,
            _sequences=padded_sequences,
        )

    def _save_inference_results_to_fasta(
        self,
        inference_result: NativeMPNNResult,
        protein_dict: Dict,
        output_dir: Path,
        pdb_path: Path,
        temperature: float,
        seed: int,
        num_sequences: int,
    ) -> Path:
        """
        Save inference results to FASTA file.

        This method handles the FASTA output generation that's common
        between different inference methods.
        """
        # Load required module
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Extract results
        S_stack = inference_result.sequences  # (num_sequences, L)
        log_probs_stack = inference_result.logits  # (num_sequences, L, 21)
        confidence_scores = inference_result.confidence_scores  # (num_sequences,)

        # Create additional derived results for compatibility
        # Compute sequence recovery
        feature_dict_mask = torch.ones(1, S_stack.shape[1], device=self._device)
        chain_mask = torch.ones(1, S_stack.shape[1], device=self._device)
        rec_mask = feature_dict_mask * chain_mask
        rec_stack = data_utils.get_seq_rec(S_stack[:1], S_stack, rec_mask)

        # Get native sequence from protein_dict
        if "S" in protein_dict:
            native_seq = "".join(
                [
                    data_utils.restype_int_to_str[AA]
                    for AA in protein_dict["S"].cpu().numpy()
                ]
            )
        else:
            # Fallback: use first generated sequence
            native_seq = "".join(
                [data_utils.restype_int_to_str[AA] for AA in S_stack[0].cpu().numpy()]
            )

        seq_np = np.array(list(native_seq))
        seq_out_str = []

        # Compute chain masks on-the-fly from chain_idx instead of using mask_c
        if "chain_idx" in protein_dict:
            chain_idx = protein_dict["chain_idx"]
            unique_chains = torch.unique(chain_idx, sorted=True)
            for unique_chain_id in unique_chains:
                chain_mask = chain_idx == unique_chain_id
                seq_out_str += list(seq_np[chain_mask.cpu().numpy()])
                seq_out_str += [":"]
        else:
            # Fallback: treat all residues as single chain
            seq_out_str += list(seq_np)
            seq_out_str += [":"]
        seq_out_str = "".join(seq_out_str)[:-1]

        # Compute combined mask for ligand confidence
        feature_dict_mask = torch.ones(1, len(native_seq), device=self._device)
        chain_mask = torch.ones(1, len(native_seq), device=self._device)
        if (
            self.config.model_type == ModelType.LIGAND_MPNN
            and "mask_XY" in protein_dict
        ):
            combined_mask = (
                feature_dict_mask
                * protein_dict.get("mask_XY", feature_dict_mask)
                * chain_mask
            )
        else:
            combined_mask = feature_dict_mask * chain_mask

        # Save results to FASTA
        name = pdb_path.stem
        output_fasta = output_dir / f"{name}.fa"

        logger.info(f"Saving {num_sequences} sequences to {output_fasta}")

        with open(output_fasta, "w") as f:
            # Write header
            f.write(
                f">{name}, T={temperature}, seed={seed}, "
                f"num_res={torch.sum(rec_mask).cpu().numpy()}, "
                f"num_ligand_res={torch.sum(chain_mask).cpu().numpy()}, "
                f"use_ligand_context={bool(self.config.ligand_mpnn_use_atom_context)}, "
                f"ligand_cutoff_distance={float(self.config.ligand_mpnn_cutoff_for_score)}, "
                f"batch_size=1, number_of_batches={num_sequences}, "
                f"model_path={self._get_checkpoint_path()}\n{seq_out_str}\n"
            )

            # Write sequences
            for ix in range(S_stack.shape[0]):
                ix_suffix = ix + 1
                seq_rec_print = np.format_float_positional(
                    rec_stack[ix].cpu().numpy(), unique=False, precision=4
                )
                confidence_print = np.format_float_positional(
                    confidence_scores[ix].cpu().numpy(), unique=False, precision=4
                )

                seq = "".join(
                    [
                        data_utils.restype_int_to_str[AA]
                        for AA in S_stack[ix].cpu().numpy()
                    ]
                )

                # Format output sequence
                seq_np = np.array(list(seq))
                seq_out_str = []

                # Compute chain masks on-the-fly from chain_idx instead of using mask_c
                if "chain_idx" in protein_dict:
                    chain_idx = protein_dict["chain_idx"]
                    unique_chains = torch.unique(chain_idx, sorted=True)
                    for unique_chain_id in unique_chains:
                        chain_mask = chain_idx == unique_chain_id
                        seq_out_str += list(seq_np[chain_mask.cpu().numpy()])
                        seq_out_str += [":"]
                else:
                    # Fallback: treat all residues as single chain
                    seq_out_str += list(seq_np)
                    seq_out_str += [":"]
                seq_out_str = "".join(seq_out_str)[:-1]

                if ix == S_stack.shape[0] - 1:
                    # Final line without newline
                    f.write(
                        f">{name}, id={ix_suffix}, T={temperature}, seed={seed}, "
                        f"overall_confidence={confidence_print}, ligand_confidence={confidence_print}, "
                        f"seq_rec={seq_rec_print}\n{seq_out_str}"
                    )
                else:
                    f.write(
                        f">{name}, id={ix_suffix}, T={temperature}, seed={seed}, "
                        f"overall_confidence={confidence_print}, ligand_confidence={confidence_print}, "
                        f"seq_rec={seq_rec_print}\n{seq_out_str}\n"
                    )

        return output_fasta

    def _apply_side_chain_packing(
        self,
        inference_result: NativeMPNNResult,
        protein_dict: Dict,
        model_sc: torch.nn.Module,
        output_dir: Path,
        pdb_path: Path,
        num_sequences: int,
    ) -> None:
        """
        Apply side chain packing to generated sequences.

        This method handles side chain packing that's common between
        different inference methods.
        """
        logger.info("Applying side chain packing...")

        # Import required functions for side chain packing
        try:
            data_utils = self._load_ligandmpnn_module("data_utils")
            sc_utils = self._load_ligandmpnn_module("sc_utils")
        except ImportError as e:
            logger.error(f"Failed to import side chain packing utilities: {e}")
            logger.warning("Side chain packing skipped due to import error")
            return

        # Extract results
        S_stack = inference_result.sequences  # (num_sequences, L)

        # Get metadata from parsing
        parsed_metadata = protein_dict.get("_parsed_metadata", {})
        other_atoms = parsed_metadata.get("other_atoms")
        icodes = parsed_metadata.get("icodes", [])

        # Prepare feature dict for side chain packing
        # Re-featurize with ligand_mpnn settings for side chain packing
        sc_feature_dict_base = data_utils.featurize(
            protein_dict,
            cutoff_for_score=8.0,
            use_atom_context=True,  # Use atom context for side chain packing
            number_of_ligand_atoms=16,
            model_type="ligand_mpnn",
        )

        # Create lists to store packed results
        packed_structures = []

        # Pack side chains for each generated sequence
        for seq_idx in range(num_sequences):
            logger.info(
                f"Packing side chains for sequence {seq_idx + 1}/{num_sequences}"
            )

            # Prepare feature dict for this sequence
            sc_feature_dict = copy.deepcopy(sc_feature_dict_base)

            # Expand batch dimension if needed
            for k, v in sc_feature_dict.items():
                if k != "S" and hasattr(v, "shape"):
                    try:
                        num_dim = len(v.shape)
                        if num_dim == 2:
                            sc_feature_dict[k] = v.repeat(1, 1)
                        elif num_dim == 3:
                            sc_feature_dict[k] = v.repeat(1, 1, 1)
                        elif num_dim == 4:
                            sc_feature_dict[k] = v.repeat(1, 1, 1, 1)
                        elif num_dim == 5:
                            sc_feature_dict[k] = v.repeat(1, 1, 1, 1, 1)
                    except Exception as e:
                        logger.debug(f"Could not expand dimension for key {k}: {e}")
                        pass

            # Set the sequence for this iteration
            sc_feature_dict["S"] = S_stack[seq_idx : seq_idx + 1]

            # Pack side chains multiple times as configured
            sequence_packed_structures = []
            for pack_idx in range(self.config.number_of_packs_per_design):
                try:
                    # Apply side chain packing
                    packed_result = sc_utils.pack_side_chains(
                        sc_feature_dict,
                        model_sc,
                        self.config.sc_num_denoising_steps,
                        self.config.sc_num_samples,
                        self.config.repack_everything,
                    )

                    # Store packed coordinates and metadata
                    packed_info = {
                        "X": packed_result["X"].cpu(),
                        "X_m": packed_result["X_m"].cpu(),
                        "b_factors": packed_result["b_factors"].cpu(),
                        "sequence_idx": seq_idx,
                        "pack_idx": pack_idx,
                    }
                    sequence_packed_structures.append(packed_info)

                except Exception as e:
                    logger.error(
                        f"Failed to pack side chains for sequence {seq_idx + 1}, pack {pack_idx + 1}: {e}"
                    )
                    continue

            packed_structures.append(sequence_packed_structures)

        # Save packed structures as PDB files
        if packed_structures:
            logger.info(f"Saving packed structures to PDB files...")
            packed_dir = output_dir / "packed"
            packed_dir.mkdir(exist_ok=True)

            for seq_idx, seq_packed_list in enumerate(packed_structures):
                for pack_info in seq_packed_list:
                    pack_idx = pack_info["pack_idx"]

                    # Generate output filename
                    packed_pdb_path = (
                        packed_dir
                        / f"{pdb_path.stem}_seq_{seq_idx + 1}_pack_{pack_idx + 1}.pdb"
                    )

                    try:
                        # Write full PDB with side chains
                        data_utils.write_full_PDB(
                            str(packed_pdb_path),
                            pack_info["X"][0].numpy(),
                            pack_info["X_m"][0].numpy(),
                            pack_info["b_factors"][0].numpy(),
                            protein_dict["R_idx"][0].cpu().numpy(),
                            protein_dict["chain_letters"],
                            S_stack[seq_idx].cpu().numpy(),
                            other_atoms=other_atoms,
                            icodes=icodes,
                        )

                        logger.info(f"Saved packed structure: {packed_pdb_path}")

                    except Exception as e:
                        logger.error(
                            f"Failed to write packed PDB {packed_pdb_path}: {e}"
                        )
                        continue

            logger.info(
                f"Side chain packing completed. Saved {len([p for seq_list in packed_structures for p in seq_list])} packed structures."
            )
        else:
            logger.warning("No packed structures were generated successfully")

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
        if hasattr(self, "_model_sc") and self._model_sc is not None:
            del self._model_sc
        if hasattr(self, "_device"):
            if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                torch.cuda.empty_cache()

    def _convert_mpnn_sequences_to_cogen(
        self, mpnn_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert amino acid sequences from MPNN format to Cogen format.

        MPNN uses alphabetical ordering [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y]
        Cogen uses AlphaFold ordering [A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V]

        Args:
            mpnn_sequences: Sequences in MPNN format (..., N) where values are in [0, 19] or possibly higher for special tokens

        Returns:
            cogen_sequences: Sequences in Cogen format (..., N) where values are in [0, 19]
        """
        # Load required modules for mapping
        data_utils = self._load_ligandmpnn_module("data_utils")
        from cogeneration.data import residue_constants

        # Create mapping from MPNN int -> Cogen int
        mpnn_int_to_str = data_utils.restype_int_to_str
        cogen_str_to_int = residue_constants.restype_order

        # Determine the maximum index in the sequences to ensure conversion map covers all indices
        max_index = mpnn_sequences.max().item() if mpnn_sequences.numel() > 0 else 19
        conversion_size = max(
            21, max_index + 1
        )  # At least 21 for standard amino acids + unknown

        # Create conversion tensor
        conversion_map = torch.zeros(
            conversion_size, dtype=torch.long, device=mpnn_sequences.device
        )

        for mpnn_int, aa_str in mpnn_int_to_str.items():
            if aa_str in cogen_str_to_int:
                cogen_int = cogen_str_to_int[aa_str]
                conversion_map[mpnn_int] = cogen_int
            else:
                # Fallback to alanine (A=0) if amino acid not found
                conversion_map[mpnn_int] = 0

        # Handle any indices beyond the standard 20 amino acids (e.g., index 20 or higher)
        # These are likely special tokens or unknown amino acids, map them to alanine (A=0)
        for idx in range(20, conversion_size):
            conversion_map[idx] = 0  # Map to alanine (A=0 in Cogen format)

        # Apply conversion
        cogen_sequences = conversion_map[mpnn_sequences]

        return cogen_sequences
