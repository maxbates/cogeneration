import glob
import json
import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from Bio import SeqIO

from cogeneration.config.base import AlphaFold2Config
from cogeneration.data.const import CHAIN_BREAK_STR
from cogeneration.data.tools.abc import FoldingDataFrame, FoldingTool
from cogeneration.type.metrics import MetricName


class AlphaFold2Tool(FoldingTool):
    """AlphaFold2 folding tool using ColabFold."""

    def __init__(
        self,
        cfg: AlphaFold2Config,
    ):
        """
        Initialize AlphaFold2 tool.

        Args:
            colabfold_path: Path to ColabFold executable
            device_id: GPU device ID to use
            seed: Random seed for reproducibility
        """
        self.cfg = cfg

        self.log = logging.getLogger(__name__)

        assert os.path.exists(
            self.cfg.colabfold_path
        ), f"ColabFold path {self.cfg.colabfold_path} does not exist"

    def set_device_id(self, device_id: Optional[int] = None):
        """Set the GPU device ID."""
        if device_id is None:
            device_id = 0
        self.device_id = device_id

    def fold_fasta(self, fasta_path: Path, output_dir: Path) -> FoldingDataFrame:
        """
        Run ColabFold AF2 folding on a fasta file
        Returns a DataFrame describing the outputs, where `header` column is the sequence name.
        """
        assert self.device_id is not None, "Device ID must be set for AF2 folding"
        assert os.path.exists(fasta_path), f"Fasta path {fasta_path} does not exist"

        # Require an empty output directory, so we can safely clean up outputs.
        assert (
            not os.path.exists(output_dir) or len(os.listdir(output_dir)) == 0
        ), f"Output folding directory {output_dir} is not empty"
        os.makedirs(output_dir, exist_ok=True)

        # If `af2_model_type` not clearly specified,
        # inspect fasta to determine if multimer or not, looking for chain breaks.
        if self.cfg.af2_model_type == "auto":
            is_multimer = False
            with open(fasta_path, "r") as f:
                for line in f:
                    if line.startswith(">"):
                        continue
                    if CHAIN_BREAK_STR in line:
                        is_multimer = True
                        break

            af2_model_type = (
                "alphafold2_ptm" if not is_multimer else "alphafold2_multimer_v3"
            )
        else:
            af2_model_type = self.cfg.af2_model_type

        # NOTE - public MultiFlow only runs model 4 for speed,
        # but may want to support running others.

        af2_args = [
            str(self.cfg.colabfold_path),
            str(fasta_path),
            str(output_dir),
            "--msa-mode",
            "single_sequence",
            "--num-models",
            "1",
            "--random-seed",
            str(self.cfg.seed),
            "--model-order",
            "4",
            "--num-recycle",
            "3",
            "--model-type",
            af2_model_type,
            "--gpu-id",
            str(self.device_id),
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
