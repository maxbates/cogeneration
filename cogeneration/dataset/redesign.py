import csv
import gc
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
from mpmath.libmp.libintmath import ifac2
from numpy import typing as npt
from tqdm.auto import tqdm

from cogeneration.config.base import (
    DatasetConfig,
    FoldingModel,
    LigandMPNNModelType,
    RedesignConfig,
)
from cogeneration.data.const import CA_IDX, aatype_to_seq
from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.data.superimposition import calc_tm_score
from cogeneration.data.tools.boltz_runner import BoltzPrediction, BoltzPredictionSet
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.process_pdb import process_pdb_file, read_processed_file
from cogeneration.dataset.scripts.process_pdb_files import safe_process_pdb
from cogeneration.type.dataset import BestRedesignColumn
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn
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
Redesigns represent sequences folded from an original structure with predicted structure and RMSD.

BestRedesigns (MultiFlow style) can be provided as an input, to bypass inverse folding. 
BestRedesigns are a deprecated output. We only write the full redesigns dataset. Filter using RMSD.
"""


logger = rank_zero_logger(__name__)


def get_existing_fieldnames(csv_path):
    """Helper to check existing CSV fieldnames."""
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
        with open(csv_path, "r", newline="") as f:
            reader = csv.reader(f)
            return next(reader)
    return None


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
    tm_score: float  # TM-score to the original structure (normalized by example)
    # generation metadata
    inverse_folding_tool: Union[LigandMPNNModelType, str]
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
        # redesign gets unique name (from inverse folding) that should differ from source PDB
        redesign_pdb_name = self.sequence_id
        source_pdb_name = self.orig_metadata[mc.pdb_name]
        assert (
            redesign_pdb_name != source_pdb_name
        ), f"Sequence ID {redesign_pdb_name} should differ from source PDB {source_pdb_name}"

        return {
            mc.pdb_name: redesign_pdb_name,
            # source PDB
            RedesignColumn.example: source_pdb_name,
            # structure
            RedesignColumn.pred_pdb_path: self.pred_pdb_path,
            RedesignColumn.rmsd: self.rmsd,
            RedesignColumn.tm_score: self.tm_score,
            # override structure method
            mc.structure_method: self.structure_method,
            # sequence
            RedesignColumn.sequence: self.sequence,
            RedesignColumn.sequence_id: self.sequence_id,
            # provenance
            mc.date: datetime.now().strftime("%Y-%m-%d"),  # YYYY-MM-DD
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
        # output file for all redesigns
        self.redesigns_path = self.cfg.output_dir / self.cfg.redesigns_csv

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
        self,
        metadata_row: MetadataCSVRow,
        processed_file: ProcessedFile,
        provided_seq: Optional[str] = None,
    ) -> List[Redesign]:
        """Generate redesigns for one structure.

        If `provided_seq` is given, bypass inverse folding and fold only the provided sequence.
        """
        pdb_name = metadata_row[mc.pdb_name]
        work_dir = self.work_dir / pdb_name

        # may exist if already attempted to redesign, but failed or interrupted
        if work_dir.exists():
            # If using Boltz2, hacky try to parse the predictions for a more meaningful message
            # We probably don't want to just delete the work dir if it exists.
            # But we can't continue, because it needs to be empty.
            if self.validator.cfg.folding_model == FoldingModel.boltz2:
                try:
                    predictions_dir = work_dir / "folding" / "predictions"
                    if (
                        os.path.exists(predictions_dir)
                        and len(os.listdir(predictions_dir)) > 0
                    ):
                        self.log.warning(
                            f"⏩ Skipping {pdb_name}: found {len(os.listdir(predictions_dir))} unprocessed Boltz2 predictions, which will be ignored! You may wish to delete {work_dir}"
                        )
                    else:
                        self.log.warning(
                            f"⏩ Skipping {pdb_name}: processed but no predictions found. You should delete {work_dir}"
                        )
                except Exception as e:
                    self.log.warning(
                        f"⏩ Skipping {pdb_name}: incomplete Boltz2 predictions (may have failed or been interrupted). You should delete {work_dir}"
                    )
            else:
                self.log.warning(
                    f"⏩ Skipping {pdb_name}: redesign work directory already exists. You may wish to delete {work_dir}"
                )

            return []

        work_dir.mkdir(parents=True, exist_ok=True)

        # Either inverse fold to generate sequences, or use provided sequence
        if provided_seq is None:
            # Redesign the structure using inverse folding
            # Keep top `fold_per_sample` of `seqs_per_sample` in fasta to fold. The others will be dropped.
            redesign_fasta_path = self.validator.inverse_fold_pdb(
                pdb_path=metadata_row[mc.raw_path],
                diffuse_mask=None,
                output_dir=work_dir,
                num_sequences=self.cfg.seqs_per_sample,
                retain_top_n=self.cfg.fold_per_sample,
            )
        else:
            # Create a fasta file containing only the provided sequence
            # The name in the fasta will become the pdb_name for the redesign.
            redesign_fasta_path = work_dir / f"{pdb_name}.fa"
            with open(redesign_fasta_path, "w") as f:
                f.write(f">{pdb_name}_redesign\n{provided_seq}\n")

        # Fold the sequence(s)
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

        orig_backbone_positions = processed_file[dpc.atom_positions][
            :, :3, :
        ]  # (N, 3, 3)

        redesigns: List[Redesign] = []
        for idx, folding_row in folding_df.iterrows():
            pred_backbone_positions = self._get_pred_backbone_positions(
                folding_row
            )  # (N, 3, 3)

            # Calculate RMSD to original
            rmsd = self.validator.calc_backbone_rmsd(
                mask=None,
                pos_1=orig_backbone_positions,
                pos_2=self._get_pred_backbone_positions(folding_row),
            )

            # Calculate TM-score to original
            try:
                # Use CA positions and sequences for alignment
                orig_ca = orig_backbone_positions[:, CA_IDX, :]
                pred_ca = pred_backbone_positions[:, CA_IDX, :]
                orig_seq = aatype_to_seq(
                    processed_file[dpc.aatype],
                    chain_idx=processed_file[dpc.chain_index],
                ).replace("X", "A")
                pred_seq = folding_row[MetricName.sequence].replace("X", "A")
                tm_norm_chain1, _ = calc_tm_score(
                    pos_1=orig_ca,
                    pos_2=pred_ca,
                    seq_1=orig_seq,
                    seq_2=pred_seq,
                )
            except Exception as e:
                self.log.error(f"Error calculating TM-score for {pdb_name}: {e}")
                tm_norm_chain1 = 0.0

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
                    tm_score=float(tm_norm_chain1),
                    inverse_folding_tool=(
                        self.validator.cfg.protein_mpnn.model_type
                        if provided_seq is None
                        else "provided"
                    ),
                    folding_tool=self.validator.cfg.folding_model,
                )
            )

        return redesigns

    def load_metadata(self) -> MetadataDataFrame:
        if self.cfg.metadata_csv is not None:
            self.log.info(f"Loading metadata from CSV {self.cfg.metadata_csv}...")
            metadata = pd.read_csv(Path(self.cfg.metadata_csv).expanduser())
        else:
            self.log.info(f"Loading structures from datasets...")
            dataset = BaseDataset(
                cfg=self.dataset_cfg,
                task=DataTask.hallucination,
                eval=False,
                use_test=False,
            )
            metadata = dataset.csv.copy()

        # Optionally shuffle before processing
        if self.cfg.shuffle:
            self.log.info(f"Shuffling metadata...")
            metadata = metadata.sample(
                frac=1, random_state=self.dataset_cfg.seed
            ).reset_index(drop=True)
        elif self.cfg.sort_by_length:
            self.log.info(f"Sorting metadata by length...")
            metadata = metadata.sort_values(by=mc.modeled_seq_len, ascending=True)

        self.log.info(
            f"Filtered dataset contains {len(metadata)} structures to redesign"
        )
        assert (
            mc.raw_path in metadata.columns
        ), f"metadata must contain '{mc.raw_path}' column with paths to input PDB files."

        # Check for existing redesigns. If they exist, optionally skip existing.
        if (
            os.path.exists(self.redesigns_path)
            and os.path.getsize(self.redesigns_path) > 0
        ):
            if not self.cfg.skip_existing:
                raise Exception(
                    f"Redesigns already exist at {self.redesigns_path}. Delete or allow --skip_existing=True"
                )

            existing_redesigns = pd.read_csv(self.redesigns_path)
            redesigned_pdbs = set(existing_redesigns[RedesignColumn.example].unique())
            num_good_redesigns = len(
                existing_redesigns[
                    existing_redesigns["rmsd"].apply(float) <= self.cfg.rmsd_good
                ]
            )
            self.log.info(
                f"Found {len(existing_redesigns)} redesigns ({num_good_redesigns} < {self.cfg.rmsd_good}) for {len(redesigned_pdbs)} PDBs in {self.redesigns_path}, which will be skipped."
            )

            # Basic compatibility check - assume columns match current redesign columns + metadata columns
            df_columns = set(existing_redesigns.columns)
            expected_columns = set(list(RedesignColumn) + list(MetadataColumn))
            assert (
                df_columns == expected_columns
            ), f"Existing redesigns CSV has new ({df_columns - expected_columns}) or missing ({expected_columns - df_columns}) columns."

            metadata = metadata[~metadata[mc.pdb_name].isin(redesigned_pdbs)]

        return metadata

    def _maybe_load_best_redesigns(self) -> Optional[pd.DataFrame]:
        """Load BestRedesigns CSV if configured, adjusting the example casing."""
        if self.cfg.best_redesigns_csv is None:
            return None
        best_path = Path(self.cfg.best_redesigns_csv).expanduser()
        assert best_path.exists(), f"BestRedesigns CSV not found: {best_path}"

        df = pd.read_csv(best_path)
        self.log.info(f"Found {len(df)} pre-defined redesigns from {best_path}")

        # filter to good RMSD options
        df = df[df[BestRedesignColumn.best_rmsd].apply(float) <= self.cfg.rmsd_good]
        self.log.info(
            f"Using {len(df)} pre-defined redesigns with RMSD < {self.cfg.rmsd_good}"
        )

        return df

    def run(self) -> None:
        self.log.info(f"Writing redesigns to {self.redesigns_path}...")

        if self.work_dir.exists():
            existing_redesign_pdbs = self.work_dir.glob("*")
            self.log.warning(
                f"Work directory {self.work_dir} contains {len(list(existing_redesign_pdbs))} redesigned PDBs. Assuming we are resuming..."
            )

        metadata = self.load_metadata()

        # BestRedesigns CSV supersedes inverse folding.
        best_redesigns_df = self._maybe_load_best_redesigns()
        # map pdb_name -> sequence
        provided_seq_map = {}
        if best_redesigns_df is not None:
            # Normalize `BestRedesignColumn.example` and `mc.pdb_name` to uppercase
            best_redesigns_df[BestRedesignColumn.example] = (
                best_redesigns_df[BestRedesignColumn.example].astype(str).str.upper()
            )
            redesign_source_pdb_names = set(
                best_redesigns_df[BestRedesignColumn.example]
            )
            dataset_pdb_names = metadata[mc.pdb_name].astype(str).str.upper()

            # Downselect dataset to redesign targets
            metadata = metadata[dataset_pdb_names.isin(redesign_source_pdb_names)]

            # Track found / missing redesign targets
            redesign_source_present_pdb_names = set(metadata[mc.pdb_name].unique())
            redesign_source_missing_pdb_names = (
                redesign_source_pdb_names - redesign_source_present_pdb_names
            )
            if len(redesign_source_missing_pdb_names) > 0:
                self.log.warning(
                    f"ℹ️ {len(redesign_source_present_pdb_names)} redesign targets present in dataset, but {len(redesign_source_missing_pdb_names)} missing"
                )
                example_missing = (
                    list(redesign_source_missing_pdb_names)[:20]
                    if len(redesign_source_missing_pdb_names) > 20
                    else list(redesign_source_missing_pdb_names)
                )
                self.log.warning(f"Some missing PDB names: {example_missing}")
            else:
                self.log.info(
                    f"ℹ️ {len(metadata[mc.pdb_name].unique())} structures of {len(redesign_source_pdb_names)} redesign targets present in dataset"
                )

            provided_seq_map = dict(
                zip(
                    best_redesigns_df[BestRedesignColumn.example],
                    best_redesigns_df[BestRedesignColumn.best_seq],
                )
            )

        # setup directories
        self.cfg.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        # Results will be streamed to CSV file as they are generated
        all_redesigns_handle = open(self.redesigns_path, "a", newline="")
        all_redesigns_writer: Optional[csv.DictWriter] = None

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

                    # check for pre-defined sequence
                    provided_seq = provided_seq_map.get(str(row[mc.pdb_name]).upper())

                    redesigns = self.redesign_structure(
                        metadata_row=row,
                        processed_file=processed_file,
                        provided_seq=provided_seq,
                    )
                except Exception as e:
                    self.log.error(f"❌ Error redesigning {pdb_name}")
                    self.log.error(e)
                    continue

                if not redesigns or len(redesigns) == 0:
                    self.log.info(f"No redesigns for {pdb_name}.")
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
                        processed_redesign_fields = list(processed_redesign.keys())
                        # handle existing file, ensure fields match
                        existing_fieldnames = get_existing_fieldnames(
                            self.redesigns_path
                        )
                        if existing_fieldnames is not None:
                            if set(existing_fieldnames) != set(
                                processed_redesign_fields
                            ):
                                raise ValueError(
                                    f"Existing redesigns CSV at {self.redesigns_path} has different fields than current redesign. Existing: {existing_fieldnames}, Current: {processed_redesign_fields}"
                                )
                            processed_redesign_fields = existing_fieldnames

                        all_redesigns_writer = csv.DictWriter(
                            all_redesigns_handle,
                            fieldnames=processed_redesign_fields,
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

                # garbage collect between iterations
                del redesigns, processed_file
                gc.collect()

        # Close CSV file
        all_redesigns_handle.close()
