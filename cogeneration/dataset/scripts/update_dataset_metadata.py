"""
Script to update dataset metadata CSV with new columns.
To be run on an ad-hoc basis if certain fields are required.

NOTE The original multiflow datasets cannot be fully upgraded using this script
because they do not reference the source PDB - it's easier to just reprocess the PDB from scratch.

NOTE - does not support the `BestRedesigns` sequence replacement - only works on metadata file directly.

TODO(dataset) consider deprecating this file? Or improve how kept in sync with `process_pdb.py`

Example which writes to `_updated.csv` in same directory:
    python update_dataset_metadata.py --metadata_csv /abs/path/to/metadata.csv

Specify output path:
    python update_dataset_metadata.py --metadata_csv /abs/path/to/metadata.csv --updated_csv /abs/path/to/metadata_updated.csv
"""

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, List, Union

from Bio import PDB
from tqdm.auto import tqdm

from cogeneration.data.io import read_pkl
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.interaction import (
    MultimerInteractions,
    NonResidueInteractions,
)
from cogeneration.dataset.process_pdb import (
    _chain_lengths,
    _concat_np_features,
    aatypes_md5_hash,
    get_uncompressed_pdb_path,
    pdb_structure_to_chain_feats,
    process_chain_feats,
)
from cogeneration.type.dataset import ChainFeatures
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataCSVRow
from cogeneration.type.structure import (
    StructureExperimentalMethod,
    extract_structure_date,
)


class MetadataUpdater:
    """
    Helper to update dataset metadata DataFrame by adding missing columns and verifying completeness.
    """

    def __init__(self, metadata_csv_path: str):
        self.csv_path = Path(metadata_csv_path)
        self.logger = logging.getLogger(__name__)

        self.df = BaseDataset.read_metadata_file(self.csv_path)
        # Load existing metadata
        self.logger.info(
            f"Loaded metadata CSV with {len(self.df)} entries from {self.csv_path}"
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
        # Peek at first row and what will be updated
        first_row = self.df.iloc[0]
        first_row_updates = self._update_row(first_row, 0)
        self.logger.info(
            f"Updates first row:\n{'\n'.join([f'{k}: {v}' for k, v in first_row_updates.items()])}"
        )

        # Compute updates per row by loading processed file once
        updates = []
        for idx, row in tqdm(
            self.df.iterrows(), desc="Inspecting rows", total=len(self.df)
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

        processed_path = self.abs_processed_path(row_metadata[mc.processed_path])
        raw_processed_file = read_pkl(processed_path)

        # memoize constructing chain_feats
        _pdb_structure = None  # PDB structure
        _whole_complex_trimmed = None  # MultiFlow style trimming
        _chain_independent_trimmed = None  # improved multimer trimming

        def get_pdb_structure():
            nonlocal _pdb_structure
            if _pdb_structure is None:
                if mc.raw_path not in row_metadata:
                    raise ValueError(
                        f"Cannot add non-residue interactions to processed files - non-res chains are omitted. "
                        f"Require row.raw_path"
                    )

                uncompressed_pdb_path, is_temp_file = get_uncompressed_pdb_path(
                    row_metadata[mc.raw_path]
                )

                parser = PDB.PDBParser(QUIET=True)
                _pdb_structure = parser.get_structure(
                    row_metadata[mc.pdb_name], uncompressed_pdb_path
                )

                if is_temp_file:
                    # Remove temporary file after reading
                    os.remove(uncompressed_pdb_path)

            return _pdb_structure

        def get_whole_complex_trimmed():
            nonlocal _whole_complex_trimmed
            if _whole_complex_trimmed is None:
                _whole_complex_trimmed = process_chain_feats(
                    raw_processed_file,
                    center=True,
                    trim_to_modeled_residues=True,
                    trim_chains_independently=False,
                )
            return _whole_complex_trimmed

        def get_chain_independent_trimmed():
            nonlocal _chain_independent_trimmed
            if _chain_independent_trimmed is None:
                _chain_independent_trimmed = process_chain_feats(
                    raw_processed_file,
                    center=True,
                    trim_to_modeled_residues=True,
                    trim_chains_independently=True,
                )
            return _chain_independent_trimmed

        # modeled sequence lengths

        if mc.moduled_num_res not in row_metadata:
            row_updates[mc.moduled_num_res] = len(raw_processed_file[dpc.modeled_idx])

        if mc.modeled_seq_len not in row_metadata:
            _whole_complex_trimmed = get_whole_complex_trimmed()
            row_updates[mc.modeled_seq_len] = len(_whole_complex_trimmed[dpc.aatype])

        if mc.modeled_indep_seq_len not in row_metadata:
            _chain_independent_trimmed = get_chain_independent_trimmed()
            row_updates[mc.modeled_indep_seq_len] = len(
                _chain_independent_trimmed[dpc.aatype]
            )

        # sequence hashes
        if mc.seq_hash not in row_metadata:
            row_updates[mc.seq_hash] = aatypes_md5_hash(
                aatype=raw_processed_file[dpc.aatype],
                chain_idx=raw_processed_file[dpc.chain_index],
            )
        if mc.seq_hash_indep not in row_metadata:
            _chain_independent_trimmed = get_chain_independent_trimmed()
            row_updates[mc.seq_hash_indep] = aatypes_md5_hash(
                aatype=_chain_independent_trimmed[dpc.aatype],
                chain_idx=_chain_independent_trimmed[dpc.chain_index],
            )

        # structure method required for future steps (b-factor vs plddt)
        # HACK include in the updates...
        if mc.structure_method in row_metadata:
            row_updates[mc.structure_method] = StructureExperimentalMethod.from_value(
                row_metadata[mc.structure_method]
            )
        else:
            structure = get_pdb_structure()
            row_updates[mc.structure_method] = (
                StructureExperimentalMethod.from_structure(structure=structure)
            )

        # plddts, using b-factors if synthetic otherwise 100.0
        if (
            mc.mean_plddt_all_atom not in row_metadata
            or mc.mean_plddt_modeled_bb not in row_metadata
        ):
            if StructureExperimentalMethod.is_experimental(
                row_updates[mc.structure_method]
            ):
                row_updates[mc.mean_plddt_all_atom] = 100.0
                row_updates[mc.mean_plddt_modeled_bb] = 100.0
            else:
                _chain_independent_trimmed = get_chain_independent_trimmed()
                row_updates[mc.mean_plddt_all_atom] = _chain_independent_trimmed[
                    dpc.b_factors
                ].mean()
                row_updates[mc.mean_plddt_modeled_bb] = _chain_independent_trimmed[
                    dpc.b_factors
                ][:, :3].mean()

        # interactions / clashes
        if (
            mc.chain_interactions not in row_metadata
            or mc.num_backbone_interactions not in row_metadata
            or mc.num_backbone_res_interacting not in row_metadata
            or mc.chain_clashes not in row_metadata
            or mc.num_chains_clashing not in row_metadata
        ):
            interactions = MultimerInteractions.from_chain_feats(
                complex_feats=get_chain_independent_trimmed(),
                metadata=row_metadata,
            )
            interactions.update_metadata(row_updates)

        # chains / lengths (requires raw PDB)

        if mc.num_all_chains not in row_metadata:
            structure = get_pdb_structure()
            row_updates[mc.num_all_chains] = len(structure.get_chains())

        if mc.chain_lengths not in row_metadata:
            structure = get_pdb_structure()
            struct_feats = pdb_structure_to_chain_feats(structure)
            row_updates[mc.chain_lengths] = _chain_lengths(
                struct_feats, modeled_only=False
            )
        if mc.chain_lengths_modeled not in row_metadata:
            structure = get_pdb_structure()
            struct_feats = pdb_structure_to_chain_feats(structure)
            row_updates[mc.chain_lengths_modeled] = _chain_lengths(
                struct_feats, modeled_only=True
            )

        # structure metadata like date, resolution (requires raw PDB)

        if mc.date not in row_metadata:
            structure = get_pdb_structure()
            row_updates[mc.date] = extract_structure_date(structure=structure)
        if mc.resolution not in row_metadata:
            structure = get_pdb_structure()
            row_updates[mc.resolution] = structure.header.get("resolution", None)

        # non-residue interactions (requires raw PDB)
        # non-res chains are omitted when processed.
        if (
            mc.num_non_residue_chains not in row_metadata
            or mc.num_single_atom_chains not in row_metadata
            or mc.num_solution_molecules not in row_metadata
            or mc.num_metal_atoms not in row_metadata
            or mc.num_metal_interactions not in row_metadata
            or mc.num_macromolecule_interactions not in row_metadata
            or mc.num_mediated_interactions not in row_metadata
            or mc.num_small_molecules not in row_metadata
            or mc.num_nucleic_acid_polymers not in row_metadata
            or mc.num_other_polymers not in row_metadata
            or mc.num_small_molecule_interactions not in row_metadata
            or mc.num_nucleic_acid_interactions not in row_metadata
        ):
            structure = get_pdb_structure()
            struct_feats = pdb_structure_to_chain_feats(structure)
            complex_feats = _concat_np_features(struct_feats, False)

            non_res_interactions = NonResidueInteractions.from_chain_feats(
                complex_feats=complex_feats,  # noqa
                structure=structure,
            )
            non_res_interactions.update_metadata(row_updates)

        return row_updates

    def check_all_columns_present(self):
        """Verify all DatasetColumns are present in the DataFrame."""
        expected = {col for col in mc}
        present = set(self.df.columns)
        missing = expected - present
        extra = present - expected
        if missing:
            self.logger.error(
                f"⚠️ Missing columns in metadata: {sorted(missing)}. Continuing..."
            )
        if extra:
            self.logger.error(
                f"⚠️ Extra columns in metadata: {sorted(extra)}. Continuing..."
            )
        if not missing and not extra:
            self.logger.info("All DatasetColumns are present in the metadata.")


@dataclass
class Args:
    """Command-line arguments for updating dataset metadata."""

    metadata_csv_path: Path
    updated_csv_path: Path

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
            default=None,
        )
        args = parser.parse_args()

        args.metadata_csv = Path(args.metadata_csv).expanduser().absolute()

        if args.updated_csv is None:
            args.updated_csv = (
                args.metadata_csv.parent / f"{args.metadata_csv.stem}_updated.csv"
            )

        return cls(
            metadata_csv_path=args.metadata_csv,
            updated_csv_path=args.updated_csv,
        )


def main(args: Args) -> None:
    # Check paths
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

    print(
        f"🌟 Replace original with:\n\tmv {args.updated_csv_path} {args.metadata_csv_path}"
    )


if __name__ == "__main__":
    args = Args.from_parser()
    main(args)
