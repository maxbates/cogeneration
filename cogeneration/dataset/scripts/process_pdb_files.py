#! /usr/bin/env python3

"""
Script to preprocessing PDB files, generating metadata file and pkls expected by DataLoader.
Supports partial processing and restarts and appending metadata, if has the same fields.

Pass `--alphafold` to use default AlphaFold directory structure and metadata file naming.

Pass `--max_structures` for debugging to limit number of files processed.

Usage:
    python process_pdb_files.py
    python process_pdb_files.py --alphafold
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from cogeneration.config.base import (
    DatasetConfig,
    DatasetFilterConfig,
    DatasetTrimMethod,
)
from cogeneration.dataset.filterer import DatasetFilterer
from cogeneration.dataset.process_pdb import (
    DataError,
    pdb_path_pdb_name,
    process_pdb_with_metadata,
)
from cogeneration.type.dataset import MetadataColumn, MetadataCSVRow

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
    alphafold: bool = False

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
            "--alphafold",
            help="Use AlphaFold directory structure and metadata naming.",
            action="store_true",
        )
        parser.add_argument(
            "--pdb_dir",
            help="Path to directory with PDB files.",
            type=str,
            default=None,  # Will be set based on --alphafold flag
        )
        parser.add_argument(
            "--write_dir",
            help="Path to write results to.",
            type=str,
            default=None,  # Will be set based on --alphafold flag
        )
        parser.add_argument(
            "--num_processes",
            help="Number of processes.",
            type=int,
            default=max(1, mp.cpu_count() // 2, mp.cpu_count() - 2),
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

        # Set default paths based on alphafold flag
        if args.pdb_dir is None:
            if args.alphafold:
                args.pdb_dir = os.path.join(
                    os.path.expanduser("~"), "pdb", "alphafold", "raw"
                )
            else:
                args.pdb_dir = os.path.join(
                    os.path.expanduser("~"), "pdb", "rcsb", "raw"
                )

        if args.write_dir is None:
            if args.alphafold:
                args.write_dir = os.path.join(
                    os.path.expanduser("~"), "pdb", "alphafold", "processed"
                )
            else:
                args.write_dir = os.path.join(
                    os.path.expanduser("~"), "pdb", "rcsb", "processed"
                )

        return cls(
            pdb_dir=args.pdb_dir,
            num_processes=args.num_processes,
            write_dir=args.write_dir,
            delete_original=args.delete_original,
            debug=args.debug,
            max_structures=args.max_structures,
            verbose=args.verbose,
            alphafold=args.alphafold,
        )


def safe_process_pdb(
    raw_pdb_path: str,
    write_dir: str,
    delete_original: bool = False,
    verbose: bool = False,
    max_combined_length: int = 8192,
    pdb_name: Optional[str] = None,
) -> Tuple[Optional[MetadataCSVRow], Optional[str]]:
    try:
        # if greater than 7 MB, skip
        if os.stat(raw_pdb_path).st_size > 7 * 1024 * 1024:
            return None, f"{raw_pdb_path} | Skipped: file size > 7 MB"

        metadata, _ = process_pdb_with_metadata(
            pdb_file_path=raw_pdb_path,
            pdb_name=pdb_name,
            write_dir=write_dir,
            max_combined_length=max_combined_length,
        )

        # TODO(dataset) - clean up file when errors are encountered?
        if delete_original:
            os.remove(raw_pdb_path)

        return metadata, None

    # known DataError exceptions
    except DataError as e:
        return None, f"{raw_pdb_path} | DataError: {e}"

    # unhandled processing errors
    except ValueError as e:
        return None, f"{raw_pdb_path} | ⚠️ Processing Error: {e}"

    # catch and log unhandled errors
    except Exception as e:
        return None, f"{raw_pdb_path} | 🔴 Unhandled Error: {e}"


def main(args: Args):
    write_dir = args.write_dir

    # track failures in a log
    error_log_path = os.path.join(write_dir, "_errors.log")
    logging.basicConfig(
        filename=error_log_path,
        level=logging.ERROR,
        format="%(asctime)s\t| %(message)s",
        filemode="a",
    )

    # support divided directories and compressed files (e.g. `07/107L.ent.gz`, `AF-P12345-F1-model_v4.pdb.gz`)
    all_paths = []
    for root, _, files in os.walk(args.pdb_dir):
        for file_name in files:
            if (
                file_name.endswith(".ent.gz")
                or file_name.endswith(".pdb")
                or file_name.endswith(".pdb.gz")
            ):
                all_paths.append(os.path.join(root, file_name))
    os.makedirs(write_dir, exist_ok=True)
    total_num_files = len(all_paths)
    print(f"Processing {total_num_files} files, writing to {write_dir}")

    # Determine metadata file path
    if args.debug:
        metadata_file_name = "metadata_debug.csv"
    else:
        metadata_file_name = "metadata.csv"
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f"Metadata will be written to {metadata_path}")

    # If metadata file exists, check it is valid, and re-use existing files.
    # Skip already processed files if in metadata and processed file exists.
    # Avoid holding onto DF, in case it is large.
    existing_metadata_columns = None
    existing_processed_files = 0
    if os.path.exists(metadata_path):
        print(
            f"Found existing metadata file at {metadata_path}, checking format + existing files..."
        )
        existing_metadata_df = pd.read_csv(metadata_path)

        if len(existing_metadata_df) == 0:
            raise Exception(
                f"⚠️ Existing Metadata file @ {metadata_path} is empty. Remove it."
            )

        missing_columns = set(MetadataColumn).difference(existing_metadata_df.columns)
        extra_columns = set(existing_metadata_df.columns).difference(MetadataColumn)
        # ensure all required columns exist, otherwise regenerate
        if len(missing_columns) > 0 or len(extra_columns) > 0:
            print(f"Missing columns: {missing_columns}")
            print(f"Extra columns: {extra_columns}")
            raise Exception(
                f"⚠️ Existing Metadata file @ {metadata_path} has column mismatch.\nDelete it or rename it and rerun.\nYou may wish to also delete the processed PDB files."
            )

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

        # drop reference to DF
        del existing_metadata_df

    # Rows will be written as available using CSV DictWriter
    # open CSV for streaming
    csv_file = open(
        metadata_path,
        mode="a" if existing_metadata_columns is not None else "w",
        newline="",
    )
    writer = csv.DictWriter(
        csv_file,
        fieldnames=(
            existing_metadata_columns
            if existing_metadata_columns is not None
            else [c for c in MetadataColumn]
        ),
    )
    if existing_metadata_columns is None:
        writer.writeheader()

    if args.max_structures > 0:
        all_paths = all_paths[: args.max_structures]
        print(f"--max_structures limiting to {args.max_structures} structures")

    # Set up DatasetFilterer to filter bad structures, but use lenient criteria
    dataset_filterer = DatasetFilterer(
        cfg=DatasetFilterConfig.lenient(),
        modeled_trim_method=DatasetTrimMethod.chains_independently,
    )

    # set up progress bar
    pbar = tqdm(
        total=len(all_paths), desc="Processing PDBs", smoothing=0.01, unit="pdbs"
    )

    def handle_result(metadata_row, error_msg):
        nonlocal processed_count

        # Processed successfully
        if metadata_row is not None and error_msg is None:
            # Filter out bad structures
            try:
                dataset_filterer.check_row(csv_row=metadata_row)
            except DataError as e:
                error_msg = f"{metadata_row[MetadataColumn.raw_path]} | Filtered | {e}"
                logging.error(error_msg)
                pbar.write(error_msg)

                # Original file already deleted, but delete the processed file
                if args.delete_original:
                    os.remove(metadata_row[MetadataColumn.processed_path])

                return
            except Exception as e:
                error_msg = (
                    f"{metadata_row[MetadataColumn.raw_path]} |  🔴 Filter Error | {e}"
                )
                logging.error(error_msg)
                pbar.write(error_msg)
                return

            # Otherwise, valid! Write metadata row to CSV
            writer.writerow(metadata_row)
            processed_count += 1

        # Encountered error
        elif error_msg is not None:
            logging.error(error_msg)
            pbar.write(error_msg)
        else:
            logging.error(f"Unknown error for {metadata_row[MetadataColumn.raw_path]}")
            pbar.write(f"Unknown error for {metadata_row[MetadataColumn.raw_path]}")

    def print_write_dir_size(directory_path: str) -> float:
        """Calculate the total size of a directory in GB."""
        num_files = 0
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(directory_path):
            for filename in filenames:
                file_path = os.path.join(dirpath, filename)
                if os.path.exists(file_path):
                    total_size += os.path.getsize(file_path)
                    num_files += 1
        gb = total_size / (1024**3)
        print(f"write_dir {directory_path} size: {gb:.2f} GB, {num_files} files")

    print_write_dir_size(write_dir)

    # Series
    processed_count = 0
    if args.num_processes == 1 or args.debug:
        for raw_pdb_path in all_paths:
            metadata_row, error_msg = safe_process_pdb(
                raw_pdb_path,
                write_dir=write_dir,
                delete_original=args.delete_original,
                verbose=args.verbose,
                max_combined_length=dataset_filterer.cfg.max_num_res,
            )
            handle_result(metadata_row, error_msg)
            pbar.update(1)

    # Parallel
    else:
        print(f"Processing {len(all_paths)} files with {args.num_processes} processes")

        with ProcessPoolExecutor(max_workers=args.num_processes) as executor:
            future_to_path = {
                executor.submit(
                    safe_process_pdb,
                    raw_pdb_path=raw_pdb_path,
                    write_dir=write_dir,
                    delete_original=args.delete_original,
                    verbose=args.verbose,
                    max_combined_length=dataset_filterer.cfg.max_num_res,
                ): raw_pdb_path
                for raw_pdb_path in all_paths
            }

            for future in as_completed(future_to_path):
                pbar.update(1)
                metadata_row, error_msg = future.result()
                handle_result(metadata_row, error_msg)

    pbar.close()
    csv_file.close()

    # print final stats
    total_done = existing_processed_files + processed_count
    print(
        f"Finished processing {processed_count} structures -> {total_done} of {total_num_files} files"
    )
    print_write_dir_size(write_dir)

    print(f"Metadata written to {metadata_path}")
    print(f"Errors tracked in {error_log_path}")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args = Args.from_parser()
    main(args)
