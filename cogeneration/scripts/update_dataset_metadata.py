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
from cogeneration.dataset.process_pdb import (
    _process_chain_feats,
    detect_multimer_clashes,
    read_processed_file,
)
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as dc
from cogeneration.type.dataset import MetadataCSVRow


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
            updates.append(self._update_row(row_metadata=row, idx=idx))

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

    def _update_row(
        self, row_metadata: MetadataCSVRow, idx: Union[Hashable, str, int]
    ) -> dict:
        """
        Compute missing values
        Load raw processed features once and compute all required metrics for a single entry.
        Returns a dict mapping column names to computed values.
        """
        row_updates: dict = {}

        processed_path = self.abs_processed_path(row_metadata[dc.processed_path])
        raw_processed_file = read_pkl(processed_path)

        # memoize constructing chain_feats
        whole_complex_trimmed = None  # MultiFlow style trimming
        chain_independent_trimmed = None  # improved multimer trimming

        def get_whole_complex_trimmed():
            nonlocal whole_complex_trimmed
            if whole_complex_trimmed is None:
                whole_complex_trimmed = _process_chain_feats(
                    raw_processed_file,
                    center=True,
                    trim_to_modeled_residues=True,
                    trim_chains_independently=False,
                )
            return whole_complex_trimmed

        def get_chain_independent_trimmed():
            nonlocal chain_independent_trimmed
            if chain_independent_trimmed is None:
                chain_independent_trimmed = _process_chain_feats(
                    raw_processed_file,
                    center=True,
                    trim_to_modeled_residues=True,
                    trim_chains_independently=True,
                )
            return chain_independent_trimmed

        if dc.moduled_num_res not in row_metadata:
            row_updates[dc.moduled_num_res] = len(raw_processed_file[dpc.modeled_idx])

        if dc.modeled_seq_len not in row_metadata:
            whole_complex_trimmed = get_whole_complex_trimmed()
            row_updates[dc.modeled_seq_len] = len(whole_complex_trimmed[dpc.aatype])

        # independent trimming
        if dc.modeled_indep_seq_len not in row_metadata:
            chain_independent_trimmed = get_chain_independent_trimmed()
            row_updates[dc.modeled_indep_seq_len] = len(
                chain_independent_trimmed[dpc.aatype]
            )

        # plddts
        if dc.mean_plddt_all_atom not in row_metadata:
            chain_independent_trimmed = get_chain_independent_trimmed()
            row_updates[dc.mean_plddt_all_atom] = chain_independent_trimmed[
                dpc.b_factors
            ].mean()
        if dc.mean_plddt_modeled_bb not in row_metadata:
            chain_independent_trimmed = get_chain_independent_trimmed()
            row_updates[dc.mean_plddt_modeled_bb] = chain_independent_trimmed[
                dpc.b_factors
            ][:, :3][chain_independent_trimmed[dpc.modeled_idx]].mean()

        # clashes
        if dc.num_chains_clashing not in row_metadata:
            chain_independent_trimmed = get_chain_independent_trimmed()
            chain_clashes = detect_multimer_clashes(
                complex_feats=chain_independent_trimmed,
                metadata=row_metadata,
            )
            row_updates[dc.num_chains_clashing] = len(
                set(clash.chain_id for clash in chain_clashes)
            )

        return row_updates

    def check_all_columns_present(self):
        """Verify all DatasetColumns are present in the DataFrame."""
        expected = {col for col in dc}
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
