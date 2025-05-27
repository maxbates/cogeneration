#! /usr/bin/env python3

"""
Script to preprocessing PDB files, generating metadata file and pkls expected by DataLoader
"""

import argparse
import logging
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from cogeneration.dataset.process_pdb import DataError, _process_pdb, pdb_path_pdb_name
from cogeneration.type.dataset import MetadataColumn, MetadataCSVRow, ProcessedFile

# Quiet mdtraj/formats/pdb/pdbfile.py
# UserWarning: Unlikely unit cell vectors detected in PDB file likely resulting from a dummy CRYST1 record. Discarding unit cell vectors.
warnings.filterwarnings("ignore", message="Unlikely unit cell vectors detected.*")


@dataclass
class Args:
    pdb_dir: str
    write_dir: str = "preprocessed"
    num_processes: Optional[int] = None
    delete_original: bool = False
    debug: bool = False
    max_structures: int = -1
    verbose: bool = False

    def __post_init__(self):
        os.makedirs(self.write_dir, exist_ok=True)

        if self.debug:
            self.num_processes = 1
            self.max_structures = 100
            self.verbose = True

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser(description="PDB processing script.")
        parser.add_argument(
            "--pdb_dir",
            help="Path to directory with PDB files.",
            type=str,
            default=os.path.join(os.path.expanduser("~"), "rcsb_pdb", "raw"),
        )
        parser.add_argument(
            "--write_dir",
            help="Path to write results to.",
            type=str,
            default=os.path.join(os.path.expanduser("~"), "rcsb_pdb", "processed"),
        )
        parser.add_argument(
            "--num_processes",
            help="Number of processes.",
            type=int,
            default=max(1, mp.cpu_count() // 2, mp.cpu_count() - 4),
        )
        parser.add_argument(
            "--delete_original",
            help="Whether to delete original PDB files.",
            action="store_true",
        )
        parser.add_argument(
            "--debug", help="Turn on for debugging.", action="store_true"
        )
        parser.add_argument(
            "--max_structures",
            help="Debugging, max number of files to process",
            type=int,
            default=-1,
        )
        parser.add_argument(
            "--verbose", help="Whether to log everything.", action="store_true"
        )
        args = parser.parse_args()

        return cls(
            pdb_dir=args.pdb_dir,
            num_processes=args.num_processes,
            write_dir=args.write_dir,
            delete_original=args.delete_original,
            debug=args.debug,
            max_structures=args.max_structures,
            verbose=args.verbose,
        )


def process_pdb_with_metadata(
    pdb_file_path: str,
    write_dir: str,
    chain_id: Optional[str] = None,
    pdb_name: Optional[str] = None,
    scale_factor: float = 1.0,
) -> Tuple[MetadataCSVRow, ProcessedFile]:
    """
    Process PDB file into concatenated chain features `ProcessedFile` and metadata `MetadataCSVRow`.
    Saves ProcessedFile pickle to `{write_dir}/{pdb_name}.pkl`.

    If `chain_id` is provided, only that chain is processed.
    `pdb_name` if provided is used for metadata and written file name.

    Raises DataError if a known filtering rule is hit. All other errors propogated.
    """
    return _process_pdb(
        pdb_file_path=pdb_file_path,
        chain_id=chain_id,
        pdb_name=pdb_name,
        scale_factor=scale_factor,
        write_dir=write_dir,
        generate_metadata=True,
    )


def process_fn(
    raw_pdb_path: str,
    write_dir: str,
    delete_original: bool = False,
    verbose: bool = False,
) -> Tuple[Optional[MetadataCSVRow], Optional[str]]:
    try:
        metadata, _ = process_pdb_with_metadata(
            pdb_file_path=raw_pdb_path, write_dir=write_dir
        )

        # TODO - clean up file when errors are encountered?
        if delete_original:
            os.remove(raw_pdb_path)

        return metadata, None

    # known DataError exceptions
    except DataError as e:
        return None, f"[{raw_pdb_path}] DataError: {e}"

    # unhandled processing errors
    except ValueError as e:
        return None, f"[{raw_pdb_path}] ⚠️ Processing Error: {e}"

    # catch and log unhandled errors
    except Exception as e:
        return None, f"[{raw_pdb_path}] 🔴 Unhandled Error: {e}"


def main(args: Args):
    pdb_dir = args.pdb_dir
    write_dir = args.write_dir

    # support divided directories and compressed files (e.g. `07/107L.ent.gz`)
    all_paths = []
    for root, _, files in os.walk(pdb_dir):
        for file_name in files:
            if file_name.endswith(".ent.gz") or file_name.endswith(".pdb"):
                all_paths.append(os.path.join(root, file_name))
    os.makedirs(write_dir, exist_ok=True)
    total_num_files = len(all_paths)
    print(f"Processing {total_num_files} files, writing to {write_dir}")

    if args.debug:
        metadata_file_name = "metadata_debug.csv"
    else:
        metadata_file_name = "metadata.csv"
    metadata_path = os.path.join(write_dir, metadata_file_name)

    # If metadata file exists, check it is valid, and re-use existing files.
    # Skip already processed files if in metadata and processed file exists.
    # Avoid holding onto DF, in case it is large.
    existing_metadata_columns = None
    existing_processed_files = 0
    if os.path.exists(metadata_path):
        existing_metadata_df = pd.read_csv(metadata_path)

        missing_columns = set(MetadataColumn).difference(existing_metadata_df.columns)
        extra_columns = set(existing_metadata_df.columns).difference(MetadataColumn)
        # ensure all required columns exist, otherwise regenerate
        if len(missing_columns) > 0 or len(extra_columns) > 0:
            print(f"Missing columns: {missing_columns}")
            print(f"Extra columns: {extra_columns}")
            raise Exception(
                f"⚠️ Existing Metadata file @ {metadata_path} has column mismatch.\nDelete it or rename it and rerun.\nYou may wish to also delete the processed PDB files."
            )
        else:
            # For rows that do exist, check if the processed file exists
            # If it does, no need to re-process
            existing_metadata_df = existing_metadata_df[
                existing_metadata_df[MetadataColumn.processed_path].apply(
                    lambda x: os.path.exists(os.path.join(write_dir, x))
                )
            ]
            existing_processed_files = len(existing_metadata_df)
            all_paths = [
                path
                for path in all_paths
                if pdb_path_pdb_name(path)
                not in existing_metadata_df[MetadataColumn.pdb_name].values
            ]
            # save the columns to ensure new metadata written the same way
            existing_metadata_columns = existing_metadata_df.columns
            print(
                f"Dropping already processed files, have {len(existing_metadata_df)}, processing {len(all_paths)} remaining."
            )

    if args.max_structures > 0:
        all_paths = all_paths[: args.max_structures]
        print(f"Debugging, limiting to {args.max_structures} files")

    all_metadata = []

    # set up progress bar
    pbar = tqdm(
        total=len(all_paths), desc="Processing PDBs", smoothing=0.01, unit="pdbs"
    )

    # track failures in a log
    error_log_path = os.path.join(write_dir, "_errors.log")
    logging.basicConfig(
        filename=error_log_path,
        level=logging.ERROR,
        format="%(asctime)s\t| %(message)s",
        filemode="a",
    )

    # Series
    if args.num_processes == 1 or args.debug:
        for raw_pdb_path in all_paths:
            metadata_row, error_msg = process_fn(
                raw_pdb_path,
                write_dir=write_dir,
                delete_original=args.delete_original,
                verbose=args.verbose,
            )

            pbar.update(1)

            if error_msg is None:
                all_metadata.append(metadata_row)
            else:
                logging.error(error_msg)
                pbar.write(error_msg)

    # Parallel
    else:
        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            future_to_path = {
                executor.submit(
                    process_fn,
                    raw_pdb_path=raw_pdb_path,
                    write_dir=write_dir,
                    delete_original=args.delete_original,
                    verbose=args.verbose,
                ): raw_pdb_path
                for raw_pdb_path in all_paths
            }

            for future in as_completed(future_to_path):
                pbar.update(1)
                metadata_row, error_msg = future.result()

                if error_msg is None:
                    all_metadata.append(metadata_row)
                else:
                    logging.error(error_msg)
                    pbar.write(error_msg)

    pbar.close()

    # filter if empty/failed
    all_metadata = [x for x in all_metadata if x is not None]
    metadata_df = pd.DataFrame(all_metadata)

    # append to existing metadata
    if existing_metadata_columns is not None:
        metadata_df = metadata_df.reindex(columns=existing_metadata_columns)
        metadata_df.to_csv(metadata_path, mode="a", header=False, index=False)
        print(
            f"Finished processing {len(metadata_df)} more -> {len(metadata_df) + existing_processed_files} of {total_num_files} files"
        )
    else:
        metadata_df.to_csv(metadata_path, index=False)
        print(
            f"Finished processing {len(metadata_df)} structures of {total_num_files} files"
        )
    print(f"Metadata written to {metadata_path}")
    print(f"Errors tracked in {error_log_path}")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args = Args.from_parser()
    main(args)
