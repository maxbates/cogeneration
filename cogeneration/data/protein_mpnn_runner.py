"""
Unified ProteinMPNN runner

This module provides a unified interface for running ProteinMPNN inverse folding,
supporting both native (in-memory model) and subprocess execution modes.
The native mode loads the model once and keeps it in memory for fast inference,
while subprocess mode calls the original ProteinMPNN script.

This module has a bunch of AI-slop, which I blame on LigandMPNN 
having an outrageously complicated `main()` function and no straightforward 
way to create a model, keep it in memory, and call the inference functions. 
So much of it's main() is ported here to support some of its features.
"""

import importlib.util
import logging
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord

from cogeneration.config.base import ModelType, ProteinMPNNRunnerConfig
from cogeneration.type.str_enum import StrEnum

logger = logging.getLogger(__name__)


class ProteinMPNNRunner:
    """
    Unified ProteinMPNN runner supporting both native and subprocess modes.

    The runner can operate in two modes:
    1. Native mode: Loads model into memory for fast repeated inference
    2. Subprocess mode: Calls the original ProteinMPNN run.py script

    Mode is controlled by the `use_native_runner` configuration option.
    """

    def __init__(self, config: ProteinMPNNRunnerConfig):
        """
        Initialize the ProteinMPNN runner.

        Args:
            config: ProteinMPNN configuration object
        """
        self.config = config
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
            import warnings

            # Temporarily suppress warnings while adding deprecated aliases
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
            bias_str: Bias string in format "A:-1.024,P:2.34,C:-12.34"

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

    def run(
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

        # Process residue encoding
        R_idx_list = list(protein_dict["R_idx"].cpu().numpy())
        chain_letters_list = list(protein_dict["chain_letters"])
        encoded_residues = []
        for i, R_idx_item in enumerate(R_idx_list):
            tmp = str(chain_letters_list[i]) + str(R_idx_item) + icodes[i]
            encoded_residues.append(tmp)

        encoded_residue_dict = dict(zip(encoded_residues, range(len(encoded_residues))))

        # Use encoded_residue_dict for residue mapping if fixed_residues or redesigned_residues are provided
        if fixed_residues or redesigned_residues:
            logger.info(
                f"Using encoded residue mapping: {len(encoded_residue_dict)} residues"
            )
            if fixed_residues:
                # Map fixed_residues using encoded_residue_dict
                mapped_fixed = []
                for res in fixed_residues:
                    if res in encoded_residue_dict:
                        mapped_fixed.append(encoded_residue_dict[res])
                    else:
                        logger.warning(f"Fixed residue {res} not found in structure")
                logger.info(f"Mapped {len(mapped_fixed)} fixed residues")

            if redesigned_residues:
                # Map redesigned_residues using encoded_residue_dict
                mapped_redesigned = []
                for res in redesigned_residues:
                    if res in encoded_residue_dict:
                        mapped_redesigned.append(encoded_residue_dict[res])
                    else:
                        logger.warning(
                            f"Redesigned residue {res} not found in structure"
                        )
                logger.info(f"Mapped {len(mapped_redesigned)} redesigned residues")

        # Set up bias and omit tensors
        bias_AA_tensor = self._parse_bias_AA(bias_AA)
        omit_AA_tensor = self._parse_omit_AA(omit_AA)

        # Set up per-residue bias and omit (placeholder for now)
        bias_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self._device, dtype=torch.float32
        )
        omit_AA_per_residue = torch.zeros(
            [len(encoded_residues), 21], device=self._device, dtype=torch.float32
        )

        # Set up fixed/redesigned positions
        fixed_residues = fixed_residues or []
        redesigned_residues = redesigned_residues or []

        fixed_positions = torch.tensor(
            [int(item not in fixed_residues) for item in encoded_residues],
            device=self._device,
        )
        redesigned_positions = torch.tensor(
            [int(item not in redesigned_residues) for item in encoded_residues],
            device=self._device,
        )

        # Set up membrane labels (for membrane variants)
        buried_positions = torch.zeros_like(fixed_positions)
        interface_positions = torch.zeros_like(fixed_positions)
        protein_dict["membrane_per_residue_labels"] = 2 * buried_positions * (
            1 - interface_positions
        ) + 1 * interface_positions * (1 - buried_positions)

        if self.config.model_type == ModelType.GLOBAL_MEMBRANE_MPNN:
            protein_dict["membrane_per_residue_labels"] = (
                self.config.global_transmembrane_label + 0 * fixed_positions
            )

        # Set up chain mask
        chains_to_design = chains_to_design or protein_dict["chain_letters"]
        chain_mask = torch.tensor(
            np.array(
                [item in chains_to_design for item in protein_dict["chain_letters"]],
                dtype=np.int32,
            ),
            device=self._device,
        )

        # Create chain_mask for design
        if redesigned_residues:
            protein_dict["chain_mask"] = chain_mask * (1 - redesigned_positions)
        elif fixed_residues:
            protein_dict["chain_mask"] = chain_mask * fixed_positions
        else:
            protein_dict["chain_mask"] = chain_mask

        # Featurize protein
        if self.config.model_type == ModelType.LIGAND_MPNN:
            atom_context_num = 16  # Default for ligand_mpnn
        else:
            atom_context_num = 1

        feature_dict = data_utils.featurize(
            protein_dict,
            cutoff_for_score=self.config.ligand_mpnn_cutoff_for_score,
            use_atom_context=self.config.ligand_mpnn_use_atom_context,
            number_of_ligand_atoms=atom_context_num,
            model_type=self.config.model_type,
        )

        # Set up feature dict for sampling
        feature_dict["batch_size"] = 1
        B, L, _, _ = feature_dict["X"].shape
        feature_dict["temperature"] = temperature
        feature_dict["bias"] = (
            (-1e8 * omit_AA_tensor[None, None, :] + bias_AA_tensor).repeat([1, L, 1])
            + bias_AA_per_residue[None]
            - 1e8 * omit_AA_per_residue[None]
        )
        feature_dict["symmetry_residues"] = [[]]  # No symmetry for now
        feature_dict["symmetry_weights"] = [[]]

        # Generate sequences
        logger.info("Generating sequences...")
        S_list = []
        log_probs_list = []
        sampling_probs_list = []
        decoding_order_list = []
        loss_list = []
        loss_per_residue_list = []
        loss_XY_list = []

        with torch.no_grad():
            for batch_idx in range(num_sequences):
                feature_dict["randn"] = torch.randn(
                    [feature_dict["batch_size"], feature_dict["mask"].shape[1]],
                    device=self._device,
                )

                output_dict = model.sample(feature_dict)

                # Compute confidence scores
                loss, loss_per_residue = data_utils.get_score(
                    output_dict["S"],
                    output_dict["log_probs"],
                    feature_dict["mask"] * feature_dict["chain_mask"],
                )

                if self.config.model_type == ModelType.LIGAND_MPNN:
                    combined_mask = (
                        feature_dict["mask"]
                        * feature_dict["mask_XY"]
                        * feature_dict["chain_mask"]
                    )
                else:
                    combined_mask = feature_dict["mask"] * feature_dict["chain_mask"]

                loss_XY, _ = data_utils.get_score(
                    output_dict["S"], output_dict["log_probs"], combined_mask
                )

                S_list.append(output_dict["S"])
                log_probs_list.append(output_dict["log_probs"])
                sampling_probs_list.append(output_dict["sampling_probs"])
                decoding_order_list.append(output_dict["decoding_order"])
                loss_list.append(loss)
                loss_per_residue_list.append(loss_per_residue)
                loss_XY_list.append(loss_XY)

        # Stack results
        S_stack = torch.cat(S_list, 0)
        log_probs_stack = torch.cat(log_probs_list, 0)
        loss_stack = torch.cat(loss_list, 0)
        loss_per_residue_stack = torch.cat(loss_per_residue_list, 0)
        loss_XY_stack = torch.cat(loss_XY_list, 0)

        # Compute sequence recovery
        rec_mask = feature_dict["mask"][:1] * feature_dict["chain_mask"][:1]
        rec_stack = data_utils.get_seq_rec(feature_dict["S"][:1], S_stack, rec_mask)

        # Get native sequence
        native_seq = "".join(
            [
                data_utils.restype_int_to_str[AA]
                for AA in feature_dict["S"][0].cpu().numpy()
            ]
        )
        seq_np = np.array(list(native_seq))
        seq_out_str = []
        for mask in protein_dict["mask_c"]:
            seq_out_str += list(seq_np[mask.cpu().numpy()])
            seq_out_str += [":"]
        seq_out_str = "".join(seq_out_str)[:-1]

        # Save results to FASTA
        name = pdb_path.stem
        output_fasta = output_dir / f"{name}.fa"

        logger.info(f"Saving {num_sequences} sequences to {output_fasta}")

        with open(output_fasta, "w") as f:
            # Write header
            f.write(
                f">{name}, T={temperature}, seed={seed}, "
                f"num_res={torch.sum(rec_mask).cpu().numpy()}, "
                f"num_ligand_res={torch.sum(combined_mask[:1]).cpu().numpy()}, "
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
                loss_np = np.format_float_positional(
                    np.exp(-loss_stack[ix].cpu().numpy()), unique=False, precision=4
                )
                loss_XY_np = np.format_float_positional(
                    np.exp(-loss_XY_stack[ix].cpu().numpy()),
                    unique=False,
                    precision=4,
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
                for mask in protein_dict["mask_c"]:
                    seq_out_str += list(seq_np[mask.cpu().numpy()])
                    seq_out_str += [":"]
                seq_out_str = "".join(seq_out_str)[:-1]

                if ix == S_stack.shape[0] - 1:
                    # Final line without newline
                    f.write(
                        f">{name}, id={ix_suffix}, T={temperature}, seed={seed}, "
                        f"overall_confidence={loss_np}, ligand_confidence={loss_XY_np}, "
                        f"seq_rec={seq_rec_print}\n{seq_out_str}"
                    )
                else:
                    f.write(
                        f">{name}, id={ix_suffix}, T={temperature}, seed={seed}, "
                        f"overall_confidence={loss_np}, ligand_confidence={loss_XY_np}, "
                        f"seq_rec={seq_rec_print}\n{seq_out_str}\n"
                    )

        # Apply side chain packing if enabled and model is available
        # TODO confirm this works, ligandMPNN is compatible with other packages, etc.
        if self.config.pack_side_chains and model_sc is not None:
            logger.info("Applying side chain packing...")

            # Import required functions for side chain packing
            try:
                import copy

                sc_utils = self._load_ligandmpnn_module("sc_utils")
            except ImportError as e:
                logger.error(f"Failed to import side chain packing utilities: {e}")
                logger.warning("Side chain packing skipped due to import error")
            else:
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
                                logger.debug(
                                    f"Could not expand dimension for key {k}: {e}"
                                )
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
                                / f"{name}_seq_{seq_idx + 1}_pack_{pack_idx + 1}.pdb"
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
                                    other_atoms=(
                                        other_atoms
                                        if "other_atoms" in locals()
                                        else None
                                    ),
                                    icodes=icodes if "icodes" in locals() else None,
                                )

                                logger.info(
                                    f"Saved packed structure: {packed_pdb_path}"
                                )

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

        # Process FASTA output for consistency with subprocess mode
        processed_fasta = self._process_fasta_output(output_fasta, output_dir, name)

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
                timeout=3600,  # 1 hour timeout
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

        # Process FASTA output using shared method
        processed_fasta = self._process_fasta_output(
            original_fasta, output_dir, pdb_path.stem
        )

        return processed_fasta

    def __del__(self):
        """Clean up resources."""
        if hasattr(self, "_model") and self._model is not None:
            del self._model
        if hasattr(self, "_model_sc") and self._model_sc is not None:
            del self._model_sc
        if hasattr(self, "_device"):
            if torch.cuda.is_available() and str(self._device).startswith("cuda"):
                torch.cuda.empty_cache()
