#! /usr/bin/env python3

"""
CLI program to generate redesigned sequences using ProteinMPNN for structures in a metadata CSV.

Fine-tuning on ProteinMPNN sequences was necessary in MultiFlow to generate high-designability samples.

Example usage:
    python redesign_structures.py --metadata_csv pdb_metadata.csv --output_dir pdb_redesigned

You may wish to first seed with MultiFlow redesigns:
    cat datasets/metadata/pdb_redesigned.csv > {output_dir}/redesigned.csv

Or after running, you may wish to merge with MultiFlow redesigns.
"""

import argparse
import csv
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy.typing as npt
import pandas as pd
from tqdm.auto import tqdm

from cogeneration.config.base import FoldingConfig
from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.dataset.process_pdb import process_pdb_file, read_processed_file
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataCSVRow, ProcessedFile, RedesignColumn
from cogeneration.type.metrics import MetricName

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s"
)


@dataclass
class Args:
    metadata_csv: Path
    output_dir: Path
    seqs_per_sample: int = 4
    skip_existing: bool = True
    pmpnn_seed: int = 123
    all_csv: str = "redesigned_all.csv"
    best_csv: str = "redesigned.csv"
    device_id: int = 0

    @classmethod
    def parse_args(cls) -> "Args":
        p = argparse.ArgumentParser(description="Protein sequence redesign CLI")
        p.add_argument(
            "--metadata_csv",
            type=Path,
            default=os.path.join(
                os.path.expanduser("~"), "rcsb_pdb", "processed", "metadata.csv"
            ),
            help="Metadata CSV with processed PDB info",
        )
        p.add_argument(
            "--output_dir",
            type=Path,
            default=os.path.join(os.path.expanduser("~"), "rcsb_pdb", "redesigned"),
            help="Directory to write redesign files",
        )
        p.add_argument(
            "--seqs_per_sample",
            type=int,
            default=4,
            help="Sequences generated per example",
        )
        p.add_argument(
            "--skip_existing",
            type=bool,
            default=True,
            help="Skip existing redesigned sequences",
        )
        p.add_argument(
            "--pmpnn_seed", type=int, default=123, help="Random seed for ProteinMPNN"
        )
        p.add_argument("--all_csv", default="redesigned_all.csv")
        p.add_argument("--best_csv", default="redesigned.csv")
        p.add_argument("--device_id", type=int, default=0)
        args = p.parse_args()

        return cls(
            metadata_csv=args.metadata_csv,
            output_dir=args.output_dir,
            seqs_per_sample=args.seqs_per_sample,
            skip_existing=args.skip_existing,
            pmpnn_seed=args.pmpnn_seed,
            all_csv=args.all_csv,
            best_csv=args.best_csv,
            device_id=args.device_id,
        )


def aatype_to_sequence(aatype: npt.NDArray) -> str:
    """Convert a numpy array of aatypes to a string sequence."""
    return "".join([restypes_with_x[aa] for aa in aatype])


@dataclass
class Redesign:
    """Single redesign, i.e. single sequence in a fasta of redesigned sequences."""

    # original
    metadata: MetadataCSVRow
    processed_file: ProcessedFile
    # redesign
    sequence_id: str  # modified ProteinMPNN sequence name
    sequence: str  # redesigned sequence
    fasta_path: str  # redesign fasta
    fasta_idx: int  # index within the fasta (0‑based)
    pred_pdb_path: str  # path to predicted structure
    rmsd: float  # RMSD to the original structure

    @property
    def wildtype_seq(self) -> str:
        """Get the wildtype sequence from the processed file."""
        return aatype_to_sequence(self.processed_file[dpc.aatype])

    def to_dict(self) -> dict:
        return {
            RedesignColumn.example: self.metadata[mc.pdb_name],
            "sequence_id": self.sequence_id,
            "sequence": self.sequence,
            "fasta_path": self.fasta_path,
            "fasta_idx": self.fasta_idx,
            "pred_pdb_path": self.pred_pdb_path,
            "rmsd": self.rmsd,
            "wildtype_seq": self.wildtype_seq,
        }


@dataclass
class BestRedesign:
    """Summary for the best redesign"""

    example: str
    wildtype_seq: str
    wildtype_rmsd: float
    best_seq: str
    best_rmsd: float

    @classmethod
    def from_redesign(cls, redesign: Redesign) -> "BestRedesign":
        """Create a BestRedesign from a Redesign instance."""
        return cls(
            example=redesign.metadata[mc.pdb_name],
            wildtype_seq=redesign.wildtype_seq,
            wildtype_rmsd=0.0,  # TODO - refold WT
            best_seq=redesign.sequence,
            best_rmsd=redesign.rmsd,
        )

    def to_dict(self) -> dict:
        return {
            RedesignColumn.example: self.example,
            RedesignColumn.wildtype_seq: self.wildtype_seq,
            RedesignColumn.wildtype_rmsd: self.wildtype_rmsd,
            RedesignColumn.best_seq: self.best_seq,
            RedesignColumn.best_rmsd: self.best_rmsd,
        }


@dataclass
class SequenceRedesigner:
    args: Args
    validator: FoldingValidator

    def __post_init__(self):
        self.args.output_dir.mkdir(parents=True, exist_ok=True)

        self.log = logging.getLogger(__name__)

        if self.args.device_id is not None:
            self.validator.set_device_id(self.args.device_id)

    def redesign_structure(
        self, metadata_row: MetadataCSVRow, processed_file: ProcessedFile
    ) -> Tuple[List[Redesign], BestRedesign]:
        """Generate multiple redesigns for one structure and pick the best."""
        work_dir = self.args.output_dir / "work" / metadata_row[mc.pdb_name]
        work_dir.mkdir(parents=True, exist_ok=True)

        # TODO write fasta for original sequence, fold it, calculate RMSD to original structure

        # Redesign the structure using inverse folding
        redesign_fasta_path = self.validator.inverse_fold_structure(
            pdb_input_path=metadata_row[mc.raw_path],
            diffuse_mask=None,
            output_dir=str(work_dir),
            num_sequences=self.args.seqs_per_sample,
        )

        # Fold the redesigned sequences
        folding_df = self.validator.fold_fasta(
            fasta_path=str(redesign_fasta_path),
            output_dir=str(work_dir / "folding"),
        )
        assert (
            MetricName.header in folding_df.columns
        ), "fold_fasta output must include 'header' column"
        assert (
            MetricName.sequence in folding_df.columns
        ), "fold_fasta output must include 'sequence' column"
        assert (
            MetricName.folded_pdb_path in folding_df.columns
        ), "fold_fasta output must include 'folded_pdb_path' column"
        assert (
            MetricName.plddt_mean in folding_df.columns
        ), "fold_fasta output must include 'plddt_mean' column"

        wt_backbone_positions = processed_file[dpc.atom_positions][:, :3, :]

        def get_pred_backbone_positions(folding_row: pd.Series) -> npt.NDArray:
            # load folded structure and extract backbone positions
            folded_feats = process_pdb_file(
                pdb_file_path=folding_row[MetricName.folded_pdb_path],
                pdb_name=folding_row[MetricName.header],
            )
            return folded_feats[dpc.atom_positions][:, :3, :]

        redesigns: List[Redesign] = []
        for idx, folding_row in folding_df.iterrows():
            # Calculate RMSD to original
            rmsd = self.validator.calc_backbone_rmsd(
                mask=None,
                pos_1=wt_backbone_positions,
                pos_2=get_pred_backbone_positions(folding_row),
            )

            # track redesign
            redesigns.append(
                Redesign(
                    metadata=metadata_row,
                    processed_file=processed_file,
                    sequence_id=folding_row[MetricName.header],
                    sequence=folding_row[MetricName.sequence],
                    fasta_path=redesign_fasta_path,
                    fasta_idx=int(idx),
                    pred_pdb_path=folding_row[MetricName.folded_pdb_path],
                    rmsd=rmsd,
                )
            )

        # Track best redesign
        best_redesign = sorted(redesigns, key=lambda x: x.rmsd)[0]
        best = BestRedesign.from_redesign(redesign=best_redesign)

        return redesigns, best

    def run(self) -> None:
        self.log.info(f"Loading structures in {self.args.metadata_csv}")
        metadata = pd.read_csv(self.args.metadata_csv)
        self.log.info(f"Found {len(metadata)} structures")

        all_redesigns_path = self.args.output_dir / self.args.all_csv
        best_redesigns_path = self.args.output_dir / self.args.best_csv
        self.log.info(
            f"Writing redesigns to {all_redesigns_path} and {best_redesigns_path}"
        )

        # Check for existing redesigns. If they exist, optionally skip existing.
        if (
            os.path.exists(best_redesigns_path)
            and os.path.getsize(best_redesigns_path) > 0
        ):
            if not self.args.skip_existing:
                raise Exception(
                    f"Redesigns already exist at {best_redesigns_path}. Delete or allow --skip_existing=True"
                )

            existing_redesigns = pd.read_csv(best_redesigns_path)
            redesigned_pdbs = set(existing_redesigns[RedesignColumn.example].unique())
            self.log.info(
                f"Found {len(redesigned_pdbs)} redesigned PDBs in {best_redesigns_path}, which will be skipped."
            )
            metadata = metadata[~metadata[mc.pdb_name].isin(redesigned_pdbs)]

        # Results will be streamed to CSV files as they are generated
        all_redesigns_handle = open(all_redesigns_path, "a", newline="")
        all_redesigns_writer: Optional[csv.DictWriter] = None
        best_redesigns_handle = open(best_redesigns_path, "a", newline="")
        best_redesigns_writer: Optional[csv.DictWriter] = None

        # Iterate to redesign structures
        with tqdm(metadata.iterrows(), total=len(metadata), desc="Redesigning") as pbar:
            for idx, row in pbar:
                row: MetadataCSVRow = row

                processed_file = read_processed_file(
                    processed_file_path=row[mc.processed_path],
                    # enable trimming and centering
                    center=True,
                    trim_to_modeled_residues=True,
                    trim_chains_independently=True,
                )

                pbar.set_postfix(
                    {
                        "pdb": row[mc.pdb_name],
                        "length": len(processed_file[dpc.aatype]),
                    }
                )

                redesigns, best = self.redesign_structure(
                    metadata_row=row,
                    processed_file=processed_file,
                )

                # Initialize CSV writers if necessary and write redesigns
                for redesign in redesigns:
                    if all_redesigns_writer is None:
                        all_redesigns_writer = csv.DictWriter(
                            all_redesigns_handle,
                            fieldnames=list(redesign.to_dict().keys()),
                        )
                        if all_redesigns_handle.tell() == 0:
                            all_redesigns_writer.writeheader()
                    all_redesigns_writer.writerow(redesign.to_dict())

                if best_redesigns_writer is None:
                    best_redesigns_writer = csv.DictWriter(
                        best_redesigns_handle, fieldnames=list(best.to_dict().keys())
                    )
                    if best_redesigns_handle.tell() == 0:
                        best_redesigns_writer.writeheader()
                best_redesigns_writer.writerow(best.to_dict())

        # Close CSV files
        all_redesigns_handle.close()
        best_redesigns_handle.close()


def main() -> None:
    args = Args.parse_args()

    # TODO support hydra to get config
    validator = FoldingValidator(
        cfg=FoldingConfig(
            seq_per_sample=args.seqs_per_sample,
            pmpnn_seed=args.pmpnn_seed,
        )
    )

    redesigner = SequenceRedesigner(args, validator)
    redesigner.run()


if __name__ == "__main__":
    main()
