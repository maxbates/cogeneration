from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch

# Folding tools output a DataFrame with the following columns:
#  `MetricName.header`
#  `MetricName.sequence`
#  `MetricName.folded_pdb_path`
#  `MetricName.mean_plddt`
FoldingDataFrame = pd.DataFrame

# Inverse folding tools output a fasta file with the inverse folded sequences
InverseFoldingFasta = Path


class FoldingTool(ABC):
    """Base class for folding tools."""

    @abstractmethod
    def fold_fasta(
        self,
        fasta_path: Path,
        output_dir: Path,
    ) -> FoldingDataFrame:
        """
        Fold a protein sequence from a FASTA file and save the results to an output directory
        Each fasta entry should be an entry in the DataFrame.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def set_device_id(self, device_id: Optional[int] = None):
        """Set the device ID for the folding tool."""
        raise NotImplementedError("Subclasses must implement this method.")


class InverseFoldingTool(ABC):
    """Base class for inverse folding tools."""

    @abstractmethod
    def inverse_fold_pdb(
        self,
        pdb_path: Path,
        output_dir: Path,
        diffuse_mask: Optional[npt.NDArray],
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> InverseFoldingFasta:
        """
        Inverse fold a PDB protein structure, return a fasta of sequences.
        Each fasta entry must have a unique name.

        Args:
            pdb_path: Path to input PDB file
            output_dir: Directory to save output files
            num_sequences: Number of sequences to generate
            diffuse_mask: Diffusion mask NDArray, to fix undiffused residues
            device_id: GPU device ID (required for subprocess mode)
            seed: Random seed
            temperature: Sampling temperature

        """
        raise NotImplementedError("Subclasses must implement this method.")

    @abstractmethod
    def set_device_id(self, device_id: Optional[int] = None):
        """Set the device ID for the folding tool."""
        raise NotImplementedError("Subclasses must implement this method.")


def infer_device_id(device_id: Optional[int] = None) -> torch.device:
    if torch.cuda.is_available():
        if device_id is None:
            device = "cuda"
        else:
            device = f"cuda:{device_id}"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    return torch.device(device)
