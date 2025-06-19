"""
ProteinMPNN runner module for executing ProteinMPNN inverse folding.
"""

import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

from Bio import SeqIO

from cogeneration.config.base import ProteinMPNNRunnerConfig

logger = logging.getLogger(__name__)


class ProteinMPNNRunner:
    """
    Encapsulates the logic for running ProteinMPNN inverse folding.

    This class is responsible for running ProteinMPNN on a given PDB file
    and returning the path to the generated sequences file.
    """

    def __init__(self, config: ProteinMPNNRunnerConfig):
        """
        Initialize the ProteinMPNN runner with configuration.

        Args:
            config: ProteinMPNNRunnerConfig containing paths and parameters
        """
        self.config = config

    def run(
        self,
        pdb_path: Path,
        output_dir: Path,
        device_id: int,
        num_sequences: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """
        Run ProteinMPNN inverse folding on a PDB file.

        Args:
            pdb_path: Path to the input PDB file
            output_dir: Directory where output files will be saved
            device_id: CUDA device ID to use
            num_sequences: Number of sequences to generate (overrides config if provided)
            seed: Random seed (overrides config if provided)

        Returns:
            Path to the generated FASTA file

        Raises:
            AssertionError: If PDB path doesn't exist or device_id is None
            subprocess.CalledProcessError: If ProteinMPNN execution fails
        """
        assert pdb_path.exists(), f"PDB path {pdb_path} does not exist"
        assert device_id is not None, "Device ID must be set for PMPNN folding"
        assert os.path.exists(
            self.config.pmpnn_path
        ), f"PMPNN path {self.config.pmpnn_path} does not exist"

        # Use provided values or fall back to config defaults
        num_sequences = (
            num_sequences if num_sequences is not None else self.config.seq_per_sample
        )
        seed = seed if seed is not None else self.config.pmpnn_seed

        logger.info(f"Running ProteinMPNN on {pdb_path}")

        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # ProteinMPNN takes a PDB as input, and writes a fasta as output, per PDB.
        # The output `.fa` file has the same basename as the input `.pdb` file.

        # Prep, copy the input structure to the directory where Protein MPNN needs it
        pdb_file_name = pdb_path.name
        pdb_name = pdb_path.stem
        output_pdb_path = output_dir / pdb_file_name
        shutil.copy(pdb_path, output_pdb_path)

        # Create a directory `seqs`, as required by the PMPNN code
        seqs_dir = output_dir / "seqs"
        seqs_dir.mkdir(exist_ok=True)

        # Prepare ProteinMPNN command
        cmd = [
            "python3",
            str(self.config.pmpnn_path / "protein_mpnn_run.py"),
            "--out_folder",
            str(output_dir),
            "--pdb_path",
            str(output_pdb_path),
            "--num_seq_per_target",
            str(num_sequences),
            "--sampling_temp",
            "0.1",
            "--seed",
            str(seed),
            "--batch_size",
            "1",
        ]

        logger.info(f"Running command: {' '.join(cmd)}")

        # Execute ProteinMPNN
        process = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT
        )
        code = process.wait()

        if code != 0:
            logger.error(f"ProteinMPNN failed with return code {code}")
            raise RuntimeError(
                f"ProteinMPNN run.py failed with code {code}. Args: {' '.join(cmd)}"
            )

        logger.info("ProteinMPNN completed successfully")

        # Find and rename the sequences into a new file
        mpnn_fasta_path = seqs_dir / f"{pdb_name}.fa"
        modified_fasta_path = str(mpnn_fasta_path).replace(".fa", "_modified.fasta")

        if not mpnn_fasta_path.exists():
            raise FileNotFoundError(
                f"Expected FASTA output not found at {mpnn_fasta_path}"
            )

        # Rename the entries into a readable format, with unique names and omitting most metrics
        fasta_records = SeqIO.parse(mpnn_fasta_path, "fasta")
        with open(modified_fasta_path, "w") as f:
            for i, record in enumerate(fasta_records):
                # Skip the first, which is the input.
                if i == 0:
                    continue
                # record names are like `T=0.1, sample=1, score=0.4789, global_score=0.4789, seq_recovery=0.2188`
                # but drop and we'll compute ourselves.
                f.write(f">{pdb_name}_pmpnn_seq_{i}\n")
                # replace chain breaks ProteinMPNN serializes with "/" -> ":" for AF2 compatibility
                f.write(str(record.seq).replace("/", ":") + "\n")

        logger.info(f"ProteinMPNN sequences saved to {modified_fasta_path}")

        return Path(modified_fasta_path)
