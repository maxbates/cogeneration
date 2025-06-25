"""
Unified ProteinMPNN runner

This module provides a unified interface for running ProteinMPNN inverse folding:
`run_subprocess` (slow) - calls original ProteinMPNN run.py script for a PDB structure 
`run_native` (fast) - loads LigandMPNN model (cached in runner) and runs on a PDB structure
`run_batch` (fast) - on trans, rots, aatypes etc. tensors directly, bypassing PDB I/O overhead

Note that LigandMPNN does not support batching, so `run_native` and `run_batch` have similar run times.

Running natively requires `LigandMPNN` is installed, and location specified in the config.

In theory, we improve parallelism using a `ProteinMPNNRunnerPool`, which is a pool 
of ProteinMPNNRunner instances that can be used to run inference on multiple structures in parallel.
However, on MPS (on a Mac), MPS effectively serializes all inference requests. On CUDA, it may help some.

Note: lots of Claude generated code here. Which I blame on LigandMPNN 
having an outrageously complicated `main()` function and no straightforward 
way to create a model, keep it in memory, and call the inference functions. 
So much of it's main() is ported here to support some of its features.

================================================================================
Claude generated note on batching / vectorized execution:

ProteinMPNN/LigandMPNN models do NOT support true vectorized batching of multiple 
different protein structures. This is a fundamental architectural limitation of 
these models. Here's why:

1. **Single Structure Architecture**: The models are designed to process one protein 
   structure at a time. The "batch_size" parameter in the model refers to the number 
   of sequences to generate for a SINGLE structure, not the number of different 
   structures to process simultaneously.

2. **Structure-Specific Features**: Each protein structure has its own unique:
   - Sequence length (variable L)
   - Chain topology and connectivity
   - Residue-residue contact maps
   - Geometric features (distances, angles)
   - Ligand context (for LigandMPNN)
   
   These cannot be meaningfully batched together as they have different dimensions
   and semantic meanings.

3. **Featurization Limitations**: The featurization process (in data_utils.featurize())
   creates structure-specific features that assume a single protein structure. The
   resulting feature tensors have shapes like (1, L, ...) where the first dimension
   is always 1 (single structure).

4. **Model Sample Method**: The model.sample() method expects:
   - feature_dict["batch_size"] = number of sequences to generate for ONE structure
   - All input tensors shaped for a single structure
   - Returns sequences shaped as (batch_size, L) where batch_size is the number
     of sequences generated for the single input structure

5. **Memory Layout**: The model's internal computations assume that all sequences
   in a batch correspond to the same underlying protein structure, sharing the
   same backbone coordinates, contact maps, and geometric features.

This limitation is inherent to the ProteinMPNN model family and cannot be
circumvented without fundamental changes to the model architecture.
"""

import asyncio
import copy
import importlib.util
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from cogeneration.config.base import ModelType, ProteinMPNNRunnerConfig
from cogeneration.data import all_atom, residue_constants
from cogeneration.data.const import CHAIN_BREAK_STR
from cogeneration.data.tools.abc import (
    InverseFoldingFasta,
    InverseFoldingTool,
    infer_device_id,
)

logger = logging.getLogger(__name__)

# Type aliases for amino acid formats to clarify which alphabet ordering is used
# ProteinMPNN uses alphabetical ordering: [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y]
# Project uses AlphaFold ordering: [A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V]
MPNNAATypes = torch.Tensor  # Amino acid types in ProteinMPNN alphabetical ordering
MPNNLogits = torch.Tensor  # Logits in ProteinMPNN alphabetical ordering (21,)
CogenAATypes = torch.Tensor  # Amino acid types in project's AlphaFold ordering
CogenLogits = torch.Tensor  # Logits in project's AlphaFold ordering (21,)

MPNNProteinDict = Dict  # Dictionary containing protein structure data (coordinates, sequences, chains, etc.)


@dataclass
class NativeMPNNResult:
    """
    Result structure from native ProteinMPNN inference.
    This provides a consistent interface for ProteinMPNN results.

    Note this struct is also being used for `MPNNLogits` and `MPNNSequences`,
    conversions are handled by ProteinMPNNRunner (need access to data_utils).
    """

    logits: CogenLogits  # (B, num_passes, sequences_per_pass, N, 21)
    confidence_scores: torch.Tensor  # (B, num_passes, sequences_per_pass)
    _sequences: Optional[CogenAATypes] = None  # (B, num_passes, sequences_per_pass, N)

    @property
    def sequences(self) -> CogenAATypes:
        """
        Generated amino acid sequences in project's AlphaFold ordering.

        Computed on-demand from logits using argmax sampling.

        Returns:
            sequences: Amino acid sequences in project's AlphaFold ordering (B, num_passes, sequences_per_pass, N)
        """
        if self._sequences is None:
            # Generate sequences from logits using argmax
            self._sequences = torch.argmax(self.logits, dim=-1)
        return self._sequences

    def set_sequences(self, sequences: CogenAATypes) -> None:
        """
        Explicitly set the sequences (useful when sequences were generated via sampling).

        Args:
            sequences: Amino acid sequences in project's AlphaFold ordering (B, num_passes, sequences_per_pass, N)
        """
        self._sequences = sequences

    @property
    def averaged_logits(self) -> CogenLogits:
        """
        Averaged logits across all generated sequences.

        For compatibility with existing code that expects averaged_logits.

        Returns:
            averaged_logits: Mean logits across num_passes and sequences_per_pass dimensions (B, N, 21)
        """
        # Average across both the num_passes and sequences_per_pass dimensions
        return torch.mean(self.logits, dim=(-4, -3))

    @property
    def average_logits_per_pass(self) -> CogenLogits:
        """
        Averaged logits per pass (averaging only across sequences_per_pass).

        Returns:
            average_logits_per_pass: Mean logits across sequences_per_pass dimension (B, num_passes, N, 21)
        """
        # Average across only the sequences_per_pass dimension
        return torch.mean(self.logits, dim=-3)

    @property
    def all_logits(self) -> CogenLogits:
        """
        All sequence logits (alias for logits property).

        For compatibility with existing code.

        Returns:
            all_logits: All sequence logits (B, num_passes, sequences_per_pass, N, 21)
        """
        return self.logits


class ProteinMPNNRunner(InverseFoldingTool):
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

    def __init__(self, cfg: ProteinMPNNRunnerConfig):
        """
        Initialize the ProteinMPNN runner.

        Args:
            cfg: ProteinMPNN configuration object
        """
        self.cfg = cfg  # TODO rename to cfg
        self._model = None
        self._model_sc = None  # Side chain packing model
        self._checkpoint_path = None
        self._ligandmpnn_modules = {}  # Cache for loaded modules

        self.set_device_id(None)

    @property
    def device(self) -> torch.device:
        """Public access to the device."""
        return self._device

    def set_device_id(self, device_id: Optional[int] = None):
        """Set the device ID"""
        self._device = infer_device_id(device_id=device_id)

        if self._model is not None:
            self._model.to(self._device)

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
            # Batch of multiple sequence logits (B, N, 20 or 21) or (num_passes, N, 20 or 21)
            for mpnn_idx in range(min(21, mpnn_logits.shape[2])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[:, :, cogen_idx] += mpnn_logits[:, :, mpnn_idx]
        elif len(original_shape) == 4:
            # Batch of multiple sequence logits (B, num_passes, N, 20 or 21)
            for mpnn_idx in range(min(21, mpnn_logits.shape[3])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[:, :, :, cogen_idx] += mpnn_logits[:, :, :, mpnn_idx]
        elif len(original_shape) == 5:
            # Batch of multiple sequence logits (B, num_passes, sequences_per_pass, N, 20 or 21)
            for mpnn_idx in range(min(21, mpnn_logits.shape[4])):
                cogen_idx = conversion_map[mpnn_idx]
                cogen_logits[:, :, :, :, cogen_idx] += mpnn_logits[:, :, :, :, mpnn_idx]
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

        ligandmpnn_path = self.cfg.ligand_mpnn_path
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
            if self.cfg.model_type == ModelType.LIGAND_MPNN:
                atom_context_num = checkpoint.get("atom_context_num", 16)
                ligand_mpnn_use_side_chain_context = (
                    self.cfg.ligand_mpnn_use_side_chain_context
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
                model_type=self.cfg.model_type,
                ligand_mpnn_use_side_chain_context=ligand_mpnn_use_side_chain_context,
            )

            # Load state dict
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Successfully loaded {self.cfg.model_type} model")
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
        if not self.cfg.pack_side_chains:
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

            checkpoint_path_sc = self.cfg.checkpoint_path_sc
            if checkpoint_path_sc is None:
                weights_path = self.cfg.ligand_mpnn_path / self.cfg.pmpnn_weights_dir

                # Default side chain checkpoint filename
                checkpoint_filename = "ligandmpnn_sc_v_32_002_16.pt"

                # Try weights directory first if specified
                if weights_path is not None:
                    possible_sc_paths = [
                        weights_path / checkpoint_filename,
                        self.cfg.ligand_mpnn_path
                        / "model_params"
                        / checkpoint_filename,
                    ]
                    for path in possible_sc_paths:
                        if path.exists():
                            checkpoint_path_sc = path
                            break

                # Fallback to ligandmpnn_path if not found in weights directory
                if checkpoint_path_sc is None:
                    checkpoint_path_sc = self.cfg.ligand_mpnn_path / checkpoint_filename

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

        ligandmpnn_path = self.cfg.ligand_mpnn_path
        weights_path = ligandmpnn_path / self.cfg.pmpnn_weights_dir

        # Map model types to checkpoint files
        checkpoint_mapping = {
            ModelType.PROTEIN_MPNN: "proteinmpnn_v_48_020.pt",
            ModelType.LIGAND_MPNN: "ligandmpnn_v_32_010_25.pt",
            ModelType.SOLUBLE_MPNN: "solublempnn_v_48_020.pt",
            ModelType.MEMBRANE_MPNN: "proteinmpnn_v_48_020.pt",  # Same as protein_mpnn
            ModelType.GLOBAL_MEMBRANE_MPNN: "global_membrane_mpnn_v_48_020.pt",
        }

        checkpoint_file = checkpoint_mapping.get(self.cfg.model_type)
        if checkpoint_file is None:
            raise ValueError(f"Unknown model type: {self.cfg.model_type}")

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
            f"Could not find checkpoint for {self.cfg.model_type}. "
            f"Tried: {[str(p) for p in possible_paths]}"
        )

    def inverse_fold_pdb(
        self,
        pdb_path: Path,
        output_dir: Path,
        diffuse_mask: torch.Tensor,
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> InverseFoldingFasta:
        """
        Run ProteinMPNN inverse folding on a PDB file.
        Implements InverseFoldingTool interface.

        Args:
            pdb_path: Path to input PDB file
            output_dir: Directory to save output files
            num_sequences: Number of sequences to generate
            diffuse_mask: Diffusion mask tensor for sequence generation
            seed: Random seed
            temperature: Sampling temperature

        Returns:
            InverseFoldingFasta
        """
        if self.cfg.use_native_runner:
            return self.inverse_fold_pdb_native(
                pdb_path=pdb_path,
                output_dir=output_dir,
                num_sequences=num_sequences,
                diffuse_mask=diffuse_mask,
                seed=seed,
                temperature=temperature,
            )
        else:
            return self.inverse_fold_pdb_subprocess(
                pdb_path=pdb_path,
                output_dir=output_dir,
                num_sequences=num_sequences,
                diffuse_mask=diffuse_mask,
                seed=seed,
                temperature=temperature,
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

    def inverse_fold_pdb_native(
        self,
        pdb_path: Path,
        output_dir: Path,
        num_sequences: Optional[int] = None,
        diffuse_mask: Optional[torch.Tensor] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> InverseFoldingFasta:
        """
        Run ProteinMPNN inference directly using the native model.

        This method loads the model directly and runs inference without subprocess.
        It supports side chain packing if configured.

        Returns fasta output path.
        """
        # Set defaults
        num_sequences = num_sequences or self.cfg.seq_per_sample
        seed = seed or self.cfg.pmpnn_seed
        temperature = temperature or self.cfg.temperature

        # TODO parse `diffuse_mask` into `chains_to_design` and `fixed_residues`

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load models
        model = self._load_model()
        model_sc = self._load_side_chain_model() if self.cfg.pack_side_chains else None

        logger.info(f"Running native ProteinMPNN on {pdb_path}")

        # Parse PDB and create protein_dict
        protein_dict = self._parse_pdb_to_protein_dict(
            pdb_path=pdb_path,
        )

        # Prepare feature dictionary for inference
        data_utils = self._load_ligandmpnn_module("data_utils")
        feature_dict = data_utils.featurize(
            protein_dict,
            cutoff_for_score=self.cfg.ligand_mpnn_cutoff_for_score,
            use_atom_context=True,
        )

        # Set up additional parameters
        N = protein_dict["X"].shape[0]
        bias_tensor = torch.zeros(1, N, 21, device=self._device)
        omit_tensor = torch.zeros(1, N, 21, device=self._device)
        feature_dict["bias"] = bias_tensor - 1e8 * omit_tensor

        # Set up chain mask
        if "chain_mask" in protein_dict:
            feature_dict["chain_mask"] = protein_dict["chain_mask"].unsqueeze(0)
        else:
            feature_dict["chain_mask"] = torch.ones(1, N, device=self._device)

        # Set up symmetry (none for now)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]

        # Run inference using shared method
        inference_result = self._run_inference_on_protein_dict(
            feature_dict=feature_dict,
            num_passes=num_sequences,
            sequences_per_pass=1,  # Set default for PDB-based methods
            temperature=temperature,
            seed=seed,
        )

        # Save results to FASTA using shared method
        output_fasta = self._save_inference_results_to_fasta(
            inference_result=inference_result,
            protein_dict=protein_dict,
            output_dir=output_dir,
            pdb_path=pdb_path,
            temperature=temperature,
            seed=seed,
            num_passes=num_sequences,
        )

        # Apply side chain packing if enabled and model is available
        if self.cfg.pack_side_chains and model_sc is not None:
            self._apply_side_chain_packing(
                inference_result=inference_result,
                protein_dict=protein_dict,
                model_sc=model_sc,
                output_dir=output_dir,
                pdb_path=pdb_path,
                num_passes=num_sequences,
            )

        # Process FASTA output
        processed_fasta = self._process_fasta_output(
            output_fasta, output_dir, pdb_path.stem
        )

        logger.info(f"Native ProteinMPNN completed successfully")
        return processed_fasta

    def inverse_fold_pdb_subprocess(
        self,
        pdb_path: Path,
        output_dir: Path,
        diffuse_mask: torch.Tensor,
        num_sequences: Optional[int] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> InverseFoldingFasta:
        """
        Run ProteinMPNN using subprocess.

        Returns fasta output path.
        """
        # Validate inputs
        assert pdb_path.exists(), f"PDB path does not exist: {pdb_path}"

        # Set defaults
        num_sequences = num_sequences or self.cfg.seq_per_sample
        seed = seed or self.cfg.pmpnn_seed
        temperature = temperature or self.cfg.temperature

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # TODO - parse `diffuse_mask` into `chains_to_design` and `fixed_residues`
        chains_to_design = None
        fixed_residues = None

        # Copy PDB to output directory
        pdb_copy_path = output_dir / pdb_path.name
        shutil.copy2(pdb_path, pdb_copy_path)

        # Construct ProteinMPNN command
        run_script = self.cfg.protein_mpnn_path / "run.py"
        assert run_script.exists(), f"ProteinMPNN run.py not found: {run_script}"

        cmd = [
            "python",
            str(run_script),
            "--model_type",
            self.cfg.model_type,
            "--pdb_path",
            str(pdb_copy_path),
            "--out_folder",
            str(output_dir),
            "--num_seq_per_target",
            str(num_sequences),
            "--sampling_temp",
            str(temperature),
            "--seed",
            str(seed),
            "--batch_size",
            "1",
        ]

        # Add additional arguments
        if chains_to_design is not None:
            cmd.extend(["--pdb_path_chains", ",".join(chains_to_design)])

        # fixed positions requires jsonl file
        if fixed_residues is not None:
            fixed_positions_path = output_dir / "fixed_positions.jsonl"
            with open(fixed_positions_path, "w") as f:
                json.dump(fixed_residues, f)
            cmd.extend(["--fixed_positions", str(fixed_positions_path)])

        logger.info(f"Running ProteinMPNN subprocess: {' '.join(cmd)}")

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "0"  # Set to GPU 0 by default

        # Execute command
        try:
            result = subprocess.run(
                cmd,
                cwd=self.cfg.ligand_mpnn_path,
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

    def _protein_dict_from_batch(
        self,
        trans: torch.Tensor,  # (B, N, 3)
        rotmats: torch.Tensor,  # (B, N, 3, 3)
        aatypes: CogenAATypes,  # (B, N) - amino acid types in project's AlphaFold ordering
        res_mask: torch.Tensor,  # (B, N) - residue mask
        chain_idx: Optional[torch.Tensor] = None,  # (B, N) - chain indices
        torsions: Optional[torch.Tensor] = None,  # (B, N, 7, 2)
    ) -> List[MPNNProteinDict]:
        """
        Create protein dictionaries from batch tensor data.

        This method handles the conversion from batch tensors (translations, rotations, etc.)
        to ProteinMPNN-compatible protein dictionaries for inference.
        """

        # Move all input tensors to the runner's device and handle MPS compatibility
        def move_tensor_to_device(tensor: torch.Tensor) -> torch.Tensor:
            """Move tensor to runner's device with MPS float64 compatibility."""
            if self._device.type == "mps" and tensor.dtype == torch.float64:
                tensor = tensor.float()
            return tensor.to(self._device)

        trans = move_tensor_to_device(trans)
        rotmats = move_tensor_to_device(rotmats)
        aatypes = move_tensor_to_device(aatypes)
        res_mask = move_tensor_to_device(res_mask)

        if torsions is not None:
            torsions = move_tensor_to_device(torsions)

        if chain_idx is not None:
            chain_idx = move_tensor_to_device(chain_idx)
        else:
            chain_idx = torch.ones_like(res_mask)

        # Convert frames to atom37 coordinates for the entire batch
        atom37 = all_atom.atom37_from_trans_rot(
            trans=trans,  # (B, N, 3)
            rots=rotmats,  # (B, N, 3, 3)
            torsions=torsions,  # (B, N, 7, 2) or None
            aatype=aatypes,  # (B, N)
            res_mask=res_mask,  # (B, N)
            unknown_to_alanine=True,
        )  # (B, N, 37, 3)

        # Convert amino acid types from project format to MPNN format
        mpnn_aatypes = self._convert_cogen_aatypes_to_mpnn(aatypes)  # (B, N)

        # Create batch protein_dict for all structures
        protein_dicts: List[MPNNProteinDict] = self._create_protein_dict_from_frames(
            atom37=atom37,  # (B, N, 37, 3)
            aatypes=mpnn_aatypes,  # (B, N)
            chain_idx=chain_idx,  # (B, N)
            device=self._device,
            res_mask=res_mask,  # (B, N)
        )

        return protein_dicts

    # TODO run_batch should take `num_logits` or `is_masking` to determine whether 20 or 21 logits are used

    def run_batch(
        self,
        trans: torch.Tensor,  # (B, N, 3)
        rotmats: torch.Tensor,  # (B, N, 3, 3)
        aatypes: CogenAATypes,  # (B, N)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        chain_idx: torch.Tensor,  # (B, N)
        torsions: Optional[torch.Tensor] = None,  # (B, N, 7, 2)
        num_passes: Optional[int] = 1,
        sequences_per_pass: int = 1,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> NativeMPNNResult:
        """
        Run ProteinMPNN inference on a batch of protein structures.
        Return NativeMPNNResult
        """
        # Check that native runner is enabled
        if not self.cfg.use_native_runner:
            raise ValueError("run_batch only supports native runner mode")

        # Set defaults
        temperature = temperature or self.cfg.temperature
        seed = seed or self.cfg.pmpnn_seed

        # TODO parse `diffuse_mask` into `chains_to_design` and `fixed_residues`

        # Load model
        model = self._load_model()

        # Create protein dictionaries from batch data
        protein_dicts = self._protein_dict_from_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            chain_idx=chain_idx,
            torsions=torsions,
        )

        # Prepare batch feature dictionary for efficient inference
        data_utils = self._load_ligandmpnn_module("data_utils")
        B = len(protein_dicts)
        N = protein_dicts[0]["X"].shape[0]  # Assume all structures have same length

        # Stack all protein features into batch tensors
        batch_X = torch.stack(
            [pdict["X"] for pdict in protein_dicts], dim=0
        )  # (B, N, 4, 3)
        batch_mask = torch.stack(
            [pdict["mask"] for pdict in protein_dicts], dim=0
        )  # (B, N)
        batch_chain_mask = torch.stack(
            [pdict["chain_mask"] for pdict in protein_dicts], dim=0
        )  # (B, N)

        # Create a representative protein_dict for featurization
        # We'll use the first structure as template and then replace with batch data
        representative_dict = protein_dicts[0].copy()
        feature_dict = data_utils.featurize(
            representative_dict,
            cutoff_for_score=self.cfg.ligand_mpnn_cutoff_for_score,
            use_atom_context=True,
        )

        # Replace single-structure features with batch features
        feature_dict["X"] = batch_X
        feature_dict["mask"] = batch_mask
        feature_dict["chain_mask"] = batch_chain_mask

        # Set up batch bias and omit tensors
        bias_tensor = torch.zeros(B, N, 21, device=self._device)
        omit_tensor = torch.zeros(B, N, 21, device=self._device)
        feature_dict["bias"] = bias_tensor - 1e8 * omit_tensor

        # Set up symmetry (none for now)
        feature_dict["symmetry_residues"] = [[]]
        feature_dict["symmetry_weights"] = [[]]

        # Run inference on the batch feature dictionary
        batch_result = self._run_inference_on_protein_dict(
            feature_dict=feature_dict,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=temperature,
            seed=seed,
        )

        # Extract results
        logits = batch_result.logits  # (B, num_passes, sequences_per_pass, N, 21)
        confidence_scores = (
            batch_result.confidence_scores
        )  # (B, num_passes, sequences_per_pass, N)
        sequences = batch_result.sequences  # (B, num_passes, sequences_per_pass, N)

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
    ) -> Union[MPNNProteinDict, List[MPNNProteinDict]]:
        """
        Create protein_dict(s) that mimic the output of parse_PDB.

        This method creates ProteinMPNN-compatible data structures from frame coordinates
        and amino acid types. The input amino acid types should already be in ProteinMPNN's
        alphabetical ordering [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y].

        Args:
            atom37: Atom coordinates in atom37 format
                   - Single: (N, 37, 3) -> returns Dict
                   - Batch: (B, N, 37, 3) -> returns List[Dict]
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
            If single structure: protein_dict (ProteinDict)
            If batch: List of protein_dicts (List[ProteinDict])
        """
        # Load the required modules
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Detect if input is batched based on atom37 dimensions
        is_batched = len(atom37.shape) == 4  # (B, N, 37, 3) vs (N, 37, 3)

        if is_batched:
            # Return list of protein_dicts, one per batch element
            if res_mask is None:
                raise ValueError("res_mask is required for batch inputs")

            B = atom37.shape[0]
            protein_dicts = []

            for b in range(B):
                # Extract single structure from batch
                single_atom37 = atom37[b]  # (N, 37, 3)
                single_aatypes = aatypes[b]  # (N,)
                single_chain_idx = chain_idx[b]  # (N,)

                # Create single protein_dict
                protein_dict = self._create_single_protein_dict_impl(
                    atom37=single_atom37,
                    aatypes=single_aatypes,
                    chain_idx=single_chain_idx,
                    device=device,
                )
                protein_dicts.append(protein_dict)

            return protein_dicts
        else:
            # Return single protein_dict
            return self._create_single_protein_dict_impl(
                atom37=atom37,
                aatypes=aatypes,
                chain_idx=chain_idx,
                device=device,
            )

    def _create_single_protein_dict_impl(
        self,
        atom37: torch.Tensor,  # (N, 37, 3)
        aatypes: MPNNAATypes,  # (N,) - amino acid types in ProteinMPNN alphabetical ordering
        chain_idx: torch.Tensor,  # (N,)
        device: torch.device,
    ) -> MPNNProteinDict:
        """Implementation for single structure protein_dict creation."""
        data_utils = self._load_ligandmpnn_module("data_utils")

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

        # TODO - support LigandMPNN, which expects a different dict

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

    def _run_inference_on_protein_dict(
        self,
        feature_dict: Dict[str, torch.Tensor],
        num_passes: int,
        sequences_per_pass: int,
        temperature: float = 1.0,
        seed: Optional[int] = None,
    ) -> NativeMPNNResult:
        """Run inference on a protein feature dictionary.

        Args:
            feature_dict: Dictionary containing protein features
            num_passes: Number of sampling passes to perform
            sequences_per_pass: Number of sequences to generate per pass
            temperature: Sampling temperature
            seed: Random seed for reproducibility

        Returns:
            NativeMPNNResult containing logits, sequences, and confidence scores
        """
        B = feature_dict["mask"].shape[0]  # batch size (number of structures)
        N = feature_dict["mask"].shape[1]  # sequence length

        # Process each structure individually but use efficient batching for sequences_per_pass
        all_logits = []
        all_sequences = []
        all_confidence_scores = []

        for struct_idx in range(B):
            # Extract single structure features
            single_feature_dict = {}
            for key, value in feature_dict.items():
                if (
                    isinstance(value, torch.Tensor)
                    and len(value.shape) > 0
                    and value.shape[0] == B
                ):
                    # Extract single structure from batch
                    single_feature_dict[key] = value[
                        struct_idx : struct_idx + 1
                    ]  # Keep batch dim of 1
                else:
                    # Non-batched features (like symmetry lists)
                    single_feature_dict[key] = value

            struct_logits = []
            struct_sequences = []
            struct_confidence_scores = []

            for pass_idx in range(num_passes):
                # Set seed first for reproducible results
                if seed is not None:
                    torch.manual_seed(seed + struct_idx * num_passes + pass_idx)

                # Set up feature dict for this pass with batch_size for parallel generation
                pass_feature_dict = single_feature_dict.copy()
                pass_feature_dict["batch_size"] = sequences_per_pass
                pass_feature_dict["temperature"] = temperature

                # Generate random tensor for decoding order (required by LigandMPNN)
                pass_feature_dict["randn"] = torch.randn(
                    [sequences_per_pass, N], device=self._device
                )

                # Run model inference - the model will handle batching internally
                # by repeating input tensors sequences_per_pass times
                with torch.no_grad():
                    sample_dict = self._model.sample(pass_feature_dict)

                # Extract results - shape will be (sequences_per_pass, N) due to internal batching
                pass_logits = sample_dict["log_probs"]  # (sequences_per_pass, N, 21)
                pass_sequences = sample_dict["S"]  # (sequences_per_pass, N)
                pass_sampling_probs = sample_dict[
                    "sampling_probs"
                ]  # (sequences_per_pass, N, 20)

                # Compute confidence scores (mean probability of sampled amino acids)
                # Convert sequences to one-hot for indexing into sampling probabilities
                seq_one_hot = torch.nn.functional.one_hot(
                    pass_sequences, num_classes=20
                ).float()
                # Sum over amino acid dimension to get per-position confidence
                confidence = torch.sum(
                    pass_sampling_probs * seq_one_hot, dim=-1
                )  # (sequences_per_pass, N)

                # Aggregate to per-sequence confidence by taking mean across positions
                confidence_per_seq = torch.mean(
                    confidence, dim=-1
                )  # (sequences_per_pass,)

                struct_logits.append(pass_logits)
                struct_sequences.append(pass_sequences)
                struct_confidence_scores.append(confidence_per_seq)

            # Stack results from all passes for this structure
            # Final shapes: (num_passes, sequences_per_pass, N, ...)
            struct_logits = torch.stack(
                struct_logits, dim=0
            )  # (num_passes, sequences_per_pass, N, 21)
            struct_sequences = torch.stack(
                struct_sequences, dim=0
            )  # (num_passes, sequences_per_pass, N)
            struct_confidence_scores = torch.stack(
                struct_confidence_scores, dim=0
            )  # (num_passes, sequences_per_pass)

            all_logits.append(struct_logits)
            all_sequences.append(struct_sequences)
            all_confidence_scores.append(struct_confidence_scores)

        # Stack results from all structures
        # Final shapes: (B, num_passes, sequences_per_pass, N, ...)
        all_logits = torch.stack(
            all_logits, dim=0
        )  # (B, num_passes, sequences_per_pass, N, 21)
        all_sequences = torch.stack(
            all_sequences, dim=0
        )  # (B, num_passes, sequences_per_pass, N)
        all_confidence_scores = torch.stack(
            all_confidence_scores, dim=0
        )  # (B, num_passes, sequences_per_pass)

        total_sequences = B * num_passes * sequences_per_pass
        logger.info(
            f"Successfully generated {total_sequences} sequences across {B} structures"
        )

        # Create and return result with proper shapes
        result = NativeMPNNResult(
            logits=all_logits,  # (B, num_passes, sequences_per_pass, N, 21)
            confidence_scores=all_confidence_scores,  # (B, num_passes, sequences_per_pass)
            _sequences=all_sequences,  # (B, num_passes, sequences_per_pass, N)
        )

        return result

    def _parse_pdb_to_protein_dict(
        self,
        pdb_path: Path,
        parse_these_chains_only: Optional[List[str]] = None,
    ) -> MPNNProteinDict:
        """
        Parse PDB file into protein_dict format.

        This method handles the PDB parsing and initial setup that's common
        between run_native and potentially other methods.
        """
        # Load required module
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Parse PDB
        parse_all_atoms_flag = self.cfg.ligand_mpnn_use_side_chain_context or (
            self.cfg.pack_side_chains and not self.cfg.repack_everything
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

    def _save_inference_results_to_fasta(
        self,
        inference_result: NativeMPNNResult,
        protein_dict: MPNNProteinDict,
        output_dir: Path,
        pdb_path: Path,
        temperature: float,
        seed: int,
        num_passes: int,
    ) -> Path:
        """
        Save inference results to FASTA file.

        This method handles the FASTA output generation that's common
        between different inference methods.
        """
        # Load required module
        data_utils = self._load_ligandmpnn_module("data_utils")

        # Extract results from first batch element (B=1 for single structure)
        # Shape: (num_passes, sequences_per_pass, L) -> flatten to (num_passes*sequences_per_pass, L)
        S_stack = inference_result.sequences[0].view(
            -1, inference_result.sequences.shape[-1]
        )  # (num_passes*sequences_per_pass, L)
        log_probs_stack = inference_result.logits[0].view(
            -1, inference_result.logits.shape[-2], inference_result.logits.shape[-1]
        )  # (num_passes*sequences_per_pass, L, 21)
        confidence_scores = inference_result.confidence_scores[0].view(
            -1
        )  # (num_passes*sequences_per_pass,)

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
                seq_out_str += [CHAIN_BREAK_STR]
        else:
            # Fallback: treat all residues as single chain
            seq_out_str += list(seq_np)
            seq_out_str += [CHAIN_BREAK_STR]
        seq_out_str = "".join(seq_out_str)[:-1]

        # Compute combined mask for ligand confidence
        feature_dict_mask = torch.ones(1, len(native_seq), device=self._device)
        chain_mask = torch.ones(1, len(native_seq), device=self._device)
        if self.cfg.model_type == ModelType.LIGAND_MPNN and "mask_XY" in protein_dict:
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

        logger.info(f"Saving {num_passes} sequences to {output_fasta}")

        with open(output_fasta, "w") as f:
            # Write header
            f.write(
                f">{name}, T={temperature}, seed={seed}, "
                f"num_res={torch.sum(rec_mask).cpu().numpy()}, "
                f"num_ligand_res={torch.sum(chain_mask).cpu().numpy()}, "
                f"use_ligand_context={bool(self.cfg.ligand_mpnn_use_atom_context)}, "
                f"ligand_cutoff_distance={float(self.cfg.ligand_mpnn_cutoff_for_score)}, "
                f"batch_size=1, number_of_batches={num_passes}, "
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
                        seq_out_str += [CHAIN_BREAK_STR]
                else:
                    # Fallback: treat all residues as single chain
                    seq_out_str += list(seq_np)
                    seq_out_str += [CHAIN_BREAK_STR]
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
        protein_dict: MPNNProteinDict,
        model_sc: torch.nn.Module,
        output_dir: Path,
        pdb_path: Path,
        num_passes: int,
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
        # Flatten the sequences from (num_passes, sequences_per_pass, N) to (num_passes*sequences_per_pass, N)
        S_stack = inference_result.sequences[0].view(
            -1, inference_result.sequences.shape[-1]
        )  # (num_passes*sequences_per_pass, N)
        total_sequences = S_stack.shape[0]

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
        for seq_idx in range(total_sequences):
            logger.info(
                f"Packing side chains for sequence {seq_idx + 1}/{total_sequences}"
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
            for pack_idx in range(self.cfg.number_of_packs_per_design):
                try:
                    # Apply side chain packing
                    packed_result = sc_utils.pack_side_chains(
                        sc_feature_dict,
                        model_sc,
                        self.cfg.sc_num_denoising_steps,
                        self.cfg.sc_num_samples,
                        self.cfg.repack_everything,
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

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
        if hasattr(self, "_model_sc") and self._model_sc is not None:
            del self._model_sc
        if hasattr(self, "_device"):
            if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                torch.cuda.empty_cache()


class ProteinMPNNRunnerPool:
    """
    Pool of ProteinMPNNRunner instances for parallel inference.

    Since ProteinMPNN doesn't support true batching of different structures,
    this pool manages multiple model instances to enable parallel processing.
    Models are loaded lazily when first used to reduce initialization time.

    TODO - better seed handling.
    Seeds are set using `torch.manual_seed` to generate `randn` and run `model.sample()`,
    but this module is not running across independent threads ,
    and it is likely the `torch.manual_seed` calls are stomping on each other.

    On CUDA systems, runners can be distributed across multiple GPUs for better scaling.
    """

    def __init__(
        self,
        cfg: ProteinMPNNRunnerConfig,
        num_models: int,
        device_ids: Optional[List[int]] = None,
    ):
        """
        Initialize the pool with multiple ProteinMPNN runners.

        Args:
            cfg: ProteinMPNN configuration
            num_models: Number of model instances to create
            device_ids: Optional list of CUDA device IDs to distribute runners across.
                       If None, all runners use the same device from cfg.
                       If provided, runners are distributed round-robin across devices.
        """
        if not cfg.use_native_runner:
            raise ValueError("ProteinMPNNRunnerPool only supports native runner mode")

        self.cfg = cfg
        self.num_models = num_models
        self.device_ids = device_ids

        # Create pool of runners (without loading models yet)
        self.runners = []
        for i in range(num_models):
            runner_cfg = copy.deepcopy(cfg)  # Create separate config for each runner
            runner = ProteinMPNNRunner(runner_cfg)

            # Assign specific device if device_ids provided (for multi-GPU CUDA)
            if device_ids and len(device_ids) > 0:
                device_id = device_ids[i % len(device_ids)]
                if torch.cuda.is_available():
                    runner._device = torch.device(f"cuda:{device_id}")
                    logger.info(f"Runner {i} assigned to cuda:{device_id}")
                else:
                    logger.warning(f"CUDA not available, ignoring device_ids")

            self.runners.append(runner)

        # Track which runners are available and which have models loaded
        self.available_runners = deque(self.runners)
        self.busy_runners = set()
        self.loaded_runners = set()  # Track which runners have loaded models

        # Simple queue for waiting requests
        self.waiting_queue = deque()

        # Lock for thread safety
        self.lock = threading.Lock()

    def run_batch(
        self,
        trans: torch.Tensor,  # (B, N, 3)
        rotmats: torch.Tensor,  # (B, N, 3, 3)
        aatypes: CogenAATypes,  # (B, N)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        chain_idx: torch.Tensor,  # (B, N)
        torsions: Optional[torch.Tensor] = None,  # (B, N, 7, 2)
        num_passes: Optional[int] = 1,
        sequences_per_pass: int = 1,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> NativeMPNNResult:
        """
        Run ProteinMPNN inference using available models from the pool.

        If there's only one model, delegates directly to the single runner.
        If there are multiple models, splits the batch and runs in parallel.

        Args:
            Same as ProteinMPNNRunner.run_batch

        Returns:
            NativeMPNNResult from the inference
        """
        B = trans.shape[0]

        if seed is not None:
            raise ValueError("Seed not recommended for ProteinMPNNRunnerPool")

        # If only one model or single batch element, use direct delegation
        if self.num_models == 1 or B == 1:
            return self._run_batch_single(
                trans=trans,
                rotmats=rotmats,
                aatypes=aatypes,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                torsions=torsions,
                num_passes=num_passes,
                sequences_per_pass=sequences_per_pass,
                temperature=temperature,
                seed=seed,
            )

        # Multiple models and multiple batch elements - run in parallel
        return asyncio.run(
            self._run_batch_parallel(
                trans=trans,
                rotmats=rotmats,
                aatypes=aatypes,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                torsions=torsions,
                num_passes=num_passes,
                sequences_per_pass=sequences_per_pass,
                temperature=temperature,
                seed=seed,
            )
        )

    async def _run_batch_parallel(
        self,
        trans: torch.Tensor,  # (B, N, 3)
        rotmats: torch.Tensor,  # (B, N, 3, 3)
        aatypes: CogenAATypes,  # (B, N)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        chain_idx: torch.Tensor,  # (B, N)
        torsions: Optional[torch.Tensor] = None,  # (B, N, 7, 2)
        num_passes: int = 1,
        sequences_per_pass: int = 1,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> NativeMPNNResult:
        """
        Run batch items in parallel using async/await pattern.
        """
        B = trans.shape[0]

        # Create tasks for each batch element
        tasks = []
        for b in range(B):
            task = self._run_batch_single_async(
                trans=trans[b : b + 1],
                rotmats=rotmats[b : b + 1],
                aatypes=aatypes[b : b + 1],
                res_mask=res_mask[b : b + 1],
                diffuse_mask=diffuse_mask[b : b + 1],
                chain_idx=chain_idx[b : b + 1],
                torsions=torsions[b : b + 1] if torsions is not None else None,
                num_passes=num_passes,
                sequences_per_pass=sequences_per_pass,
                temperature=temperature,
                seed=seed,
            )
            tasks.append(task)

        # Run all tasks in parallel
        results = await asyncio.gather(*tasks)

        # Collate results
        return self._collate_mpnn_results(results)

    async def _run_batch_single_async(
        self,
        trans: torch.Tensor,  # (1, N, 3)
        rotmats: torch.Tensor,  # (1, N, 3, 3)
        aatypes: CogenAATypes,  # (1, N)
        res_mask: torch.Tensor,  # (1, N)
        diffuse_mask: torch.Tensor,  # (1, N)
        chain_idx: torch.Tensor,  # (1, N)
        torsions: Optional[torch.Tensor] = None,  # (1, N, 7, 2)
        num_passes: int = 1,
        sequences_per_pass: int = 1,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> NativeMPNNResult:
        """
        Async wrapper for running a single batch element.
        """
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self._run_batch_single,
            trans,
            rotmats,
            aatypes,
            res_mask,
            diffuse_mask,
            chain_idx,
            torsions,
            num_passes,
            sequences_per_pass,
            temperature,
            seed,
        )

    def _run_batch_single(
        self,
        trans: torch.Tensor,  # (1, N, 3)
        rotmats: torch.Tensor,  # (1, N, 3, 3)
        aatypes: CogenAATypes,  # (1, N)
        res_mask: torch.Tensor,  # (1, N)
        diffuse_mask: torch.Tensor,  # (1, N)
        chain_idx: torch.Tensor,  # (1, N)
        torsions: Optional[torch.Tensor] = None,  # (1, N, 7, 2)
        num_passes: int = 1,
        sequences_per_pass: int = 1,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> NativeMPNNResult:
        """
        Run ProteinMPNN inference on a single batch element (batch size 1).

        Args:
            All tensors should have batch dimension 1 (1, ...)
            Other args same as run_batch

        Returns:
            NativeMPNNResult for the single structure
        """
        # Validate batch size is 1
        assert trans.shape[0] == 1, f"Expected batch size 1, got {trans.shape[0]}"

        # Get an available runner
        runner = self._get_runner()

        try:
            # Run inference
            result = runner.run_batch(
                trans=trans,
                rotmats=rotmats,
                aatypes=aatypes,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                torsions=torsions,
                num_passes=num_passes,
                sequences_per_pass=sequences_per_pass,
                temperature=temperature,
                seed=seed,
            )
            return result
        finally:
            # Return runner to pool
            self._return_runner(runner)

    def _collate_mpnn_results(
        self, results: List[NativeMPNNResult]
    ) -> NativeMPNNResult:
        """
        Collate multiple NativeMPNNResult instances into a single batched result.

        Args:
            results: List of NativeMPNNResult, each with batch dimension 1

        Returns:
            Single NativeMPNNResult with batch dimension len(results)
        """
        if not results:
            raise ValueError("Cannot collate empty results list")

        # Extract components from all results
        all_logits = []
        all_confidence_scores = []
        all_sequences = []

        for result in results:
            # Each result should have batch dimension 1
            assert (
                result.logits.shape[0] == 1
            ), f"Expected batch size 1, got {result.logits.shape[0]}"

            all_logits.append(result.logits)
            all_confidence_scores.append(result.confidence_scores)

            # Get sequences (computed on demand if needed)
            if result._sequences is not None:
                all_sequences.append(result._sequences)
            else:
                all_sequences.append(result.sequences)

        # Concatenate along batch dimension
        collated_logits = torch.cat(
            all_logits, dim=0
        )  # (B, num_passes, sequences_per_pass, N, 21)
        collated_confidence = torch.cat(
            all_confidence_scores, dim=0
        )  # (B, num_passes, sequences_per_pass)
        collated_sequences = torch.cat(
            all_sequences, dim=0
        )  # (B, num_passes, sequences_per_pass, N)

        return NativeMPNNResult(
            logits=collated_logits,
            confidence_scores=collated_confidence,
            _sequences=collated_sequences,
        )

    def _get_runner(self) -> ProteinMPNNRunner:
        """
        Get an available runner from the pool.
        Loads the model if not already loaded.
        Blocks until one becomes available.
        """
        while True:
            with self.lock:
                if self.available_runners:
                    runner = self.available_runners.popleft()
                    self.busy_runners.add(runner)

                    # Check if model needs to be loaded
                    if runner not in self.loaded_runners:
                        runner._load_model()
                        self.loaded_runners.add(runner)

                    return runner

            # No runners available, yield and wait a bit and try again
            time.sleep(0.01)

    def _return_runner(self, runner: ProteinMPNNRunner) -> None:
        """
        Return a runner to the available pool.
        """
        with self.lock:
            if runner in self.busy_runners:
                self.busy_runners.remove(runner)
                self.available_runners.append(runner)

    def __len__(self) -> int:
        """Return the number of runners in the pool."""
        return self.num_models

    def __del__(self):
        """Clean up all runners."""
        for runner in self.runners:
            del runner
