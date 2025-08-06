import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from mpmath.libmp.libintmath import ifac2
from numpy import typing as npt
from tqdm.asyncio import tqdm

from cogeneration.config.base import (
    DatasetConfig,
    FoldingModel,
    LigandMPNNModelType,
    RedesignConfig,
)
from cogeneration.data.const import aatype_to_seq
from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.process_pdb import process_pdb_file, read_processed_file
from cogeneration.dataset.scripts.process_pdb_files import safe_process_pdb
from cogeneration.type.dataset import BestRedesignColumn
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import (
    MetadataCSVRow,
    MetadataDataFrame,
    ProcessedFile,
    RedesignColumn,
)
from cogeneration.type.metrics import MetricName
from cogeneration.type.structure import StructureExperimentalMethod
from cogeneration.type.task import DataTask
from cogeneration.util.log import rank_zero_logger

"""
There are two notions of redesigns:

1) Redesign - any redesigned sequence from an original structure, with predicted structure and RMSD.
2) BestRedesigns (MulitFlow style) - single sequence that replaces original sequence.

SequenceRedesigner generates both:
1) A new dataset of redesigns, which contains all redesigns meeting RMSD threshold (or can be filtered later)
2) Best rdesigns CSV, which augments original dataset by replacing the original sequence with the best redesign sequence.
"""


logger = rank_zero_logger(__name__)


@dataclass
class Redesign:
    """Single redesign, i.e. single sequence in a fasta of redesigned sequences."""

    # original
    orig_metadata: MetadataCSVRow
    orig_processed_file: ProcessedFile
    # redesign
    sequence_id: str  # modified ProteinMPNN sequence name
    sequence: str  # redesigned sequence
    fasta_path: str  # redesign fasta
    fasta_idx: int  # index within the fasta (0‑based)
    pred_pdb_path: str  # path to predicted structure
    rmsd: float  # RMSD to the original structure
    # generation metadata
    inverse_folding_tool: LigandMPNNModelType  # NOTE assumes using LigandMPNN
    folding_tool: FoldingModel

    @property
    def example_seq(self) -> str:
        """Get the original example's wildtype sequence from the processed file."""
        return aatype_to_seq(self.orig_processed_file[dpc.aatype])

    @property
    def structure_method(self) -> StructureExperimentalMethod:
        if self.folding_tool == FoldingModel.boltz2:
            return StructureExperimentalMethod.BOLTZ_2
        elif self.folding_tool == FoldingModel.alphafold2:
            return StructureExperimentalMethod.AFDB
        else:
            return StructureExperimentalMethod.UNKNOWN

    def process(self, write_dir: str) -> Optional[MetadataCSVRow]:
        metadata, err = safe_process_pdb(
            raw_pdb_path=self.pred_pdb_path,
            pdb_name=self.sequence_id,
            write_dir=write_dir,
            delete_original=False,
            max_combined_length=8192,
        )
        if err is not None:
            logger.error(f"{self.orig_metadata[mc.pdb_name]} | {err}")
            return None

        return metadata

    def to_dict(self) -> dict:
        """CSV row"""
        source_pdb_name = self.orig_metadata[mc.pdb_name]

        return {
            # redesign gets unique name
            mc.pdb_name: f"{source_pdb_name}_{self.sequence_id}",
            # source PDB
            RedesignColumn.example: source_pdb_name,
            # structure
            RedesignColumn.pred_pdb_path: self.pred_pdb_path,
            RedesignColumn.rmsd: self.rmsd,
            # override structure method
            mc.structure_method: self.structure_method,
            # sequence
            RedesignColumn.sequence: self.sequence,
            RedesignColumn.sequence_id: self.sequence_id,
            # provenance
            RedesignColumn.example_sequence: self.example_seq,
            RedesignColumn.example_pdb_path: self.orig_metadata[mc.raw_path],
            RedesignColumn.fasta_path: self.fasta_path,
            RedesignColumn.fasta_idx: self.fasta_idx,
            RedesignColumn.inverse_folding_tool: self.inverse_folding_tool,
            RedesignColumn.folding_tool: self.folding_tool,
        }

    def to_processed_dict(self, write_dir: str) -> Optional[dict]:
        """Processed file metadata for this redesign."""
        metadata = self.process(write_dir=write_dir)

        if metadata is None:
            return None

        return {
            **metadata,
            **self.to_dict(),
        }


@dataclass
class BestRedesign:
    """Summary for the best redesign, i.e. MultiFlow style specification of redesign."""

    example: str
    wildtype_seq: str
    wildtype_rmsd: float
    best_seq: str
    best_rmsd: float

    @classmethod
    def from_redesign(cls, redesign: Redesign) -> "BestRedesign":
        """Create a BestRedesign from a Redesign instance."""
        return cls(
            example=redesign.orig_metadata[mc.pdb_name],
            wildtype_seq=redesign.example_seq,
            wildtype_rmsd=-1.0,  # TODO(validation) - refold WT
            best_seq=redesign.sequence,
            best_rmsd=redesign.rmsd,
        )

    def to_dict(self) -> dict:
        """CSV row"""
        return {
            BestRedesignColumn.example: self.example,
            BestRedesignColumn.wildtype_seq: self.wildtype_seq,
            BestRedesignColumn.wildtype_rmsd: self.wildtype_rmsd,
            BestRedesignColumn.best_seq: self.best_seq,
            BestRedesignColumn.best_rmsd: self.best_rmsd,
        }


@dataclass
class SequenceRedesigner:
    """
    SequenceRedesigner takes a BaseDataset of PDBs and generates a new dataset of redesigned structures + sequences.

    Can specify either metadata CSV in redesign config, or will use load datasets using dataset config.

    This is a slow process -- each structure must be inverse folded then folded, which takes time.
    Results are streamed into the output CSVs as they are ready. Supports resuming.

    Fine-tuning on ProteinMPNN sequences was necessary in MultiFlow to generate high-designability samples.

    Like MultiFlow, we also generate a "best" redesign for each structure with a matching CSV structure.
    This file is mostly for historical reasons...

    Unlike MultiFlow, we generate a list of all redesigns + predicted structures + RMSD.
    We require the new predicted structure, because we predict torsions etc., not only the backbone.
    This file functions as a new dataset.
    """

    cfg: RedesignConfig
    validator: FoldingValidator
    dataset_cfg: DatasetConfig

    def __post_init__(self):
        # output files
        self.all_redesigns_path = self.cfg.output_dir / self.cfg.all_csv
        self.best_redesigns_path = self.cfg.output_dir / self.cfg.best_csv

        # directory for inverse folding / folding intermediates
        self.work_dir = self.cfg.output_dir / "work"
        # directory for processing resdesigns
        self.processed_dir = self.cfg.output_dir / "processed"

        self.log = logger

    def _get_pred_backbone_positions(self, folding_row: pd.Series) -> npt.NDArray:
        """Helper to load folded structure and extract backbone positions."""
        folded_feats = process_pdb_file(
            pdb_file_path=folding_row[MetricName.folded_pdb_path],
            pdb_name=folding_row[MetricName.header],
        )
        return folded_feats[dpc.atom_positions][:, :3, :]

    def redesign_structure(
        self, metadata_row: MetadataCSVRow, processed_file: ProcessedFile
    ) -> Tuple[List[Redesign], Optional[BestRedesign]]:
        """Generate multiple redesigns for one structure and pick the best."""
        pdb_name = metadata_row[mc.pdb_name]
        work_dir = self.work_dir / pdb_name

        # may exist if already attempted to redesign, but none kept
        if work_dir.exists():
            self.log.warning(
                f"Redesign for {pdb_name} work directory {work_dir} already exists. Skipping"
            )
            return [], None

        work_dir.mkdir(parents=True, exist_ok=True)

        # Redesign the structure using inverse folding
        redesign_fasta_path = self.validator.inverse_fold_pdb(
            pdb_path=metadata_row[mc.raw_path],
            diffuse_mask=None,
            output_dir=work_dir,
            num_sequences=self.cfg.seqs_per_sample,
        )

        # Fold the redesigned sequences
        folding_df = self.validator.fold_fasta(
            fasta_path=redesign_fasta_path,
            output_dir=(work_dir / "folding"),
        )

        # sanity check
        for col in [
            MetricName.header,
            MetricName.sequence,
            MetricName.folded_pdb_path,
            MetricName.plddt_mean,
        ]:
            if col not in folding_df.columns:
                raise ValueError(f"fold_fasta output must include '{col}' column")

        orig_backbone_positions = processed_file[dpc.atom_positions][:, :3, :]

        redesigns: List[Redesign] = []
        for idx, folding_row in folding_df.iterrows():
            # Calculate RMSD to original
            rmsd = self.validator.calc_backbone_rmsd(
                mask=None,
                pos_1=orig_backbone_positions,
                pos_2=self._get_pred_backbone_positions(folding_row),
            )

            # Optionally filter out high RMSD redesigns
            if rmsd > self.cfg.rmsd_max and self.cfg.rmsd_max > 0.01:
                continue

            # track redesign
            redesigns.append(
                Redesign(
                    orig_metadata=metadata_row,
                    orig_processed_file=processed_file,
                    sequence_id=folding_row[MetricName.header],
                    sequence=folding_row[MetricName.sequence],
                    fasta_path=str(redesign_fasta_path),
                    fasta_idx=int(idx),
                    pred_pdb_path=folding_row[MetricName.folded_pdb_path],
                    rmsd=rmsd,
                    inverse_folding_tool=self.validator.cfg.protein_mpnn.model_type,
                    folding_tool=self.validator.cfg.folding_model,
                )
            )

        if not redesigns:
            self.log.info(
                f"No redesigns for {pdb_name} with RMSD <= {self.cfg.rmsd_max}. Skipping."
            )
            return [], None

        # Track best redesign
        best_redesign = sorted(redesigns, key=lambda x: x.rmsd)[0]
        best = BestRedesign.from_redesign(redesign=best_redesign)

        return redesigns, best

    def load_metadata(self) -> MetadataDataFrame:
        if self.cfg.metadata_csv is not None:
            self.log.info(f"Loading metadata from CSV {self.cfg.metadata_csv}...")
            metadata = pd.read_csv(Path(self.cfg.metadata_csv).expanduser())
        else:
            self.log.info(f"Loading structures from dataset...")
            dataset = BaseDataset(
                cfg=self.dataset_cfg,
                task=DataTask.hallucination,
                eval=False,
                use_test=False,
            )
            metadata = dataset.csv.copy()

        self.log.info(f"Found {len(metadata)} structures to redesign")
        assert (
            mc.raw_path in metadata.columns
        ), f"metadata must contain '{mc.raw_path}' column with paths to input PDB files."

        # Check for existing redesigns. If they exist, optionally skip existing.
        if (
            os.path.exists(self.all_redesigns_path)
            and os.path.getsize(self.all_redesigns_path) > 0
        ):
            if not self.cfg.skip_existing:
                raise Exception(
                    f"Redesigns already exist at {self.all_redesigns_path}. Delete or allow --skip_existing=True"
                )

            existing_redesigns = pd.read_csv(self.all_redesigns_path)
            redesigned_pdbs = set(
                existing_redesigns[BestRedesignColumn.example].unique()
            )
            num_good_redesigns = len(
                existing_redesigns[
                    existing_redesigns["rmsd"].apply(float) <= self.cfg.rmsd_good
                ]
            )
            self.log.info(
                f"Found {len(existing_redesigns)} redesigns ({num_good_redesigns} < {self.cfg.rmsd_good}) for {len(redesigned_pdbs)} PDBs in {self.all_redesigns_path}, which will be skipped."
            )
            metadata = metadata[~metadata[mc.pdb_name].isin(redesigned_pdbs)]

        return metadata

    def run(self) -> None:
        if self.work_dir.exists():
            existing_redesign_pdbs = self.work_dir.glob("*")
            self.log.warning(
                f"⚠️ Work directory {self.work_dir} contains {len(list(existing_redesign_pdbs))} redesigned PDBs. Assuming we are resuming..."
            )

        metadata = self.load_metadata()

        # setup directories
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.log.info(f"Writing redesigns to {self.all_redesigns_path}...")

        # Results will be streamed to CSV files as they are generated
        all_redesigns_handle = open(self.all_redesigns_path, "a", newline="")
        all_redesigns_writer: Optional[csv.DictWriter] = None
        best_redesigns_handle = open(self.best_redesigns_path, "a", newline="")
        best_redesigns_writer: Optional[csv.DictWriter] = None

        # Counter for tracking good / total redesigns
        good_redesigns = 0
        total_redesigns = 0

        # Iterate to redesign structures
        with tqdm(metadata.iterrows(), total=len(metadata), desc="Redesigning") as pbar:
            for idx, row in pbar:
                row: MetadataCSVRow = row

                pdb_name = row[mc.pdb_name]
                pdb_length = (
                    row[mc.modeled_indep_seq_len]
                    if mc.modeled_indep_seq_len in row
                    else row[mc.modeled_seq_len]
                )

                pbar.set_postfix(
                    {
                        "pdb": pdb_name,
                        "length": pdb_length,
                        "redesigns": f"{good_redesigns}/{total_redesigns} < {self.cfg.rmsd_good}",
                    }
                )

                try:
                    processed_file = read_processed_file(
                        processed_file_path=row[mc.processed_path],
                        # enable trimming and centering
                        center=True,
                        trim_to_modeled_residues=True,
                        trim_chains_independently=True,
                    )

                    redesigns, best = self.redesign_structure(
                        metadata_row=row,
                        processed_file=processed_file,
                    )
                except Exception as e:
                    self.log.error(f"⚠️ Error redesigning {pdb_name}: {e}")
                    continue

                if not redesigns or len(redesigns) == 0:
                    self.log.info(f"No redesigns retained for {pdb_name}.")
                    continue

                # Initialize CSV writers if necessary and write redesigns
                for redesign in redesigns:
                    processed_redesign = redesign.to_processed_dict(
                        write_dir=str(self.processed_dir)
                    )

                    # skip if error processing
                    if processed_redesign is None:
                        continue

                    # setup writer if needed
                    if all_redesigns_writer is None:
                        all_redesigns_writer = csv.DictWriter(
                            all_redesigns_handle,
                            fieldnames=list(processed_redesign.keys()),
                        )
                        if all_redesigns_handle.tell() == 0:
                            all_redesigns_writer.writeheader()
                    all_redesigns_writer.writerow(processed_redesign)

                    # Increment redesign counter and update progress bar
                    total_redesigns += 1
                    if redesign.rmsd < self.cfg.rmsd_good:
                        good_redesigns += 1

                self.log.info(
                    f"{row[mc.pdb_name]} Redesigns RMSD = {', '.join([str(r.rmsd) for r in redesigns])}"
                )

                if best is not None:
                    if best_redesigns_writer is None:
                        best_redesign = best.to_dict()
                        # setup writer if needed
                        best_redesigns_writer = csv.DictWriter(
                            best_redesigns_handle, fieldnames=list(best_redesign.keys())
                        )
                        if best_redesigns_handle.tell() == 0:
                            best_redesigns_writer.writeheader()
                    best_redesigns_writer.writerow(best_redesign)

        # Close CSV files
        all_redesigns_handle.close()
        best_redesigns_handle.close()
