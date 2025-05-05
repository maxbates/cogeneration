"""Script to update dataset metadata CSV with new columns."""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, List, Union

import pandas as pd
from tqdm.auto import tqdm

from cogeneration.data.io import read_pkl
from cogeneration.dataset.datasets import read_metadata_file
from cogeneration.dataset.process_pdb import _process_chain_feats, read_processed_file
from cogeneration.type.dataset import DatasetProteinColumn, MetadataColumn


class MetadataUpdater:
    """
    Helper to update dataset metadata DataFrame by adding missing columns and verifying completeness.
    """

    def __init__(self, metadata_csv_path: str):
        self.csv_path = Path(metadata_csv_path)
        self.logger = logging.getLogger(__name__)

        self.df = read_metadata_file(metadata_csv_path)
        # Load existing metadata
        self.logger.info(
            f"Loaded metadata CSV with {len(self.df)} entries from {metadata_csv_path}."
        )

    def abs_processed_path(self, processed_path: str) -> str:
        """Get complete processed_path given potentially relative path in metadata CSV"""
        if processed_path.startswith("/"):
            return processed_path
        else:
            # assume data directories are siblings to metadata dir containing metadata file
            metadata_dir = self.csv_path.parent
            return str((metadata_dir.parent / processed_path).absolute())

    def update(self):
        """
        Compute and add missing metadata columns: modeled_seq_len,
        modeled_indep_seq_len, and moduled_num_res.
        """
        # Compute updates per row by loading processed file once
        updates = []
        for idx, row in tqdm(
            self.df.iterrows(), desc="Updating rows", total=len(self.df)
        ):
            updates.append(self._update_row(row=row, idx=idx))

        # Convert [{col: val}, ...] to {col: [val, ...]}, assume same columns throughout
        column_updates = {
            col: [update.get(col) for update in updates]
            for col in set(updates[0].keys())
        }
        # Add new columns to DataFrame
        for col, vals in column_updates.items():
            self.df[col] = vals

        # Verify completeness
        self.check_all_columns_present()

    def _update_row(self, row: pd.Series, idx: Union[Hashable, str, int]) -> dict:
        """
        Compute missing values
        Load raw processed features once and compute all required metrics for a single entry.
        Returns a dict mapping column names to computed values.
        """
        row_updates: dict = {}

        processed_path = self.abs_processed_path(row[MetadataColumn.processed_path])
        raw_processed_file = read_pkl(processed_path)

        if MetadataColumn.moduled_num_res not in row:
            row_updates[MetadataColumn.moduled_num_res] = len(
                raw_processed_file[DatasetProteinColumn.modeled_idx]
            )

        if MetadataColumn.modeled_seq_len not in row:
            whole_complex_trimmed = _process_chain_feats(
                raw_processed_file,
                center=True,
                trim_to_modeled_residues=True,
                trim_chains_independently=False,
            )
            row_updates[MetadataColumn.modeled_seq_len] = len(
                whole_complex_trimmed[DatasetProteinColumn.aatype]
            )

        if MetadataColumn.modeled_indep_seq_len not in row:
            chain_independent_trimmed = _process_chain_feats(
                raw_processed_file,
                center=True,
                trim_to_modeled_residues=True,
                trim_chains_independently=True,
            )
            row_updates[MetadataColumn.modeled_indep_seq_len] = len(
                chain_independent_trimmed[DatasetProteinColumn.aatype]
            )

        return row_updates

    def check_all_columns_present(self):
        """Verify all DatasetColumns are present in the DataFrame."""
        expected = {col for col in MetadataColumn}
        present = set(self.df.columns)
        missing = expected - present
        extra = present - expected
        if missing:
            self.logger.error(f"Missing columns in metadata: {sorted(missing)}")
        if extra:
            self.logger.error(f"Extra columns in metadata: {sorted(extra)}")
        if not missing and not extra:
            self.logger.info("All DatasetColumns are present in the metadata.")


@dataclass
class Args:
    """Command-line arguments for updating dataset metadata."""

    metadata_csv_path: str
    updated_csv_path: str

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser(
            description="Update dataset metadata CSV by adding missing columns."
        )
        parser.add_argument(
            "--metadata_csv",
            help="Path to the input metadata CSV file.",
            type=str,
            required=True,
        )
        parser.add_argument(
            "--updated_csv",
            help="Path to write the updated metadata CSV file.",
            type=str,
            required=True,
        )
        args = parser.parse_args()

        return cls(
            metadata_csv_path=args.metadata_csv,
            updated_csv_path=args.updated_csv,
        )


def main(args: Args) -> None:
    # Check paths
    # TODO enable moving original and writing updated to original path
    assert args.metadata_csv_path.startswith("/"), "Metadata CSV path must be absolute."
    if args.metadata_csv_path == args.updated_csv_path:
        raise ValueError("Input and output paths must be different.")
    if os.path.exists(args.updated_csv_path):
        raise FileExistsError(f"Output path {args.updated_csv_path} already exists")

    # Load and update metadata
    updater = MetadataUpdater(metadata_csv_path=args.metadata_csv_path)
    updater.update()

    # Write updated metadata to file
    updater.df.to_csv(args.updated_csv_path, index=False)
    print(f"Updated metadata CSV written to {args.updated_csv_path}")


if __name__ == "__main__":
    args = Args.from_parser()
    main(args)
