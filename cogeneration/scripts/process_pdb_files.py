"""Script for preprocessing PDB files to generate pkls expected by DataLoader"""

import argparse
import functools as fn
import gzip
import multiprocessing as mp
import os
import tempfile
import time

# Quiet mdtraj/formats/pdb/pdbfile.py
# UserWarning: Unlikely unit cell vectors detected in PDB file likely resulting from a dummy CRYST1 record. Discarding unit cell vectors.
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from cogeneration.dataset.process_pdb import DataError, _process_pdb
from cogeneration.type.dataset import MetadataColumn, MetadataCSVRow, ProcessedFile

warnings.filterwarnings(
    "ignore", message="Unlikely unit cell vectors detected.*"
)  # turn off that specific warning


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


def pdb_path_pdb_name(pdb_path: str) -> str:
    """
    Convert a PDB file path to a PDB name.
    """
    # Remove the directory and file extension
    pdb_name = os.path.basename(pdb_path).upper()
    pdb_name = pdb_name.replace(".ENT.GZ", "")
    pdb_name = pdb_name.replace(".PDB", "")
    # remove leading pdb e.g. `pdb0000.ent.gz`
    if pdb_name.startswith("PDB"):
        pdb_name = pdb_name[3:]
    return pdb_name


def process_fn(
    file_path: str, write_dir: str, delete_original: bool = False, verbose: bool = False
) -> Optional[MetadataCSVRow]:
    try:
        start_time = time.time()

        is_compressed = file_path.endswith(".ent.gz")

        if is_compressed:
            pdb_name = pdb_path_pdb_name(file_path)

            with gzip.open(file_path, "rt") as gz:
                # write decompressed content to a temp file with correct pdb_name
                tmp_dir = tempfile.mkdtemp()
                tmp_path = os.path.join(tmp_dir, f"{pdb_name}.pdb")
                with open(tmp_path, "wb") as tmp:
                    tmp.write(gz.read().encode("utf-8"))

            metadata, _ = process_pdb_with_metadata(
                pdb_file_path=tmp_path, pdb_name=pdb_name, write_dir=write_dir
            )

            # remove the temporary file
            os.remove(tmp_path)
        else:
            metadata, _ = process_pdb_with_metadata(
                pdb_file_path=file_path, write_dir=write_dir
            )

        elapsed_time = time.time() - start_time
        if verbose:
            print(f"[{file_path}] Finished in {elapsed_time:2.2f}s")
        if delete_original:
            os.remove(file_path)
        return metadata

    # TODO - clean up file when errors are encountered?
    except DataError as e:
        if verbose:
            print(f"[{file_path}] DataError: {e}")
    except ValueError as e:
        # log unhandled errors
        print(f"[{file_path}] ⚠️ Unhandled Processing Error: {e}")


def process_serially(
    all_paths: List[str],
    write_dir: str,
    delete_original: bool = False,
    verbose: bool = False,
) -> List[MetadataCSVRow]:
    all_metadata = []
    for i, file_path in tqdm(enumerate(all_paths), desc="Processing PDBs"):
        metadata = process_fn(
            file_path=file_path,
            write_dir=write_dir,
            delete_original=delete_original,
            verbose=verbose,
        )
        all_metadata.append(metadata)
    return all_metadata


@dataclass
class Args:
    pdb_dir: str
    num_processes: Optional[int] = None
    write_dir: str = "preprocessed"
    delete_original: bool = False
    debug: bool = False
    max_structures: int = -1
    verbose: bool = False

    def __post_init__(self):
        if self.debug:
            self.num_processes = 1
            self.verbose = True
        if self.num_processes is None:
            self.num_processes = max(1, mp.cpu_count() - 4)
        if not os.path.exists(self.write_dir):
            os.makedirs(self.write_dir)

    @classmethod
    def from_parser(cls):
        parser = argparse.ArgumentParser(description="PDB processing script.")
        parser.add_argument(
            "--pdb_dir", help="Path to directory with PDB files.", type=str
        )
        parser.add_argument(
            "--num_processes", help="Number of processes.", type=int, default=None
        )
        parser.add_argument(
            "--write_dir",
            help="Path to write results to.",
            type=str,
            default="preprocessed",
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
            print(f"Existing Metadata file has column mismatch and will be overwritten")
            print(f"Missing columns: {missing_columns}")
            print(f"Extra columns: {extra_columns}")
            # TODO - consider moving it instead of overwriting?
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

    # Process each file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_paths,
            write_dir=write_dir,
            delete_original=args.delete_original,
            verbose=args.verbose,
        )
    else:
        _process_fn = fn.partial(
            process_fn,
            verbose=args.verbose,
            delete_original=args.delete_original,
            write_dir=write_dir,
        )
        with mp.Pool(processes=args.num_processes) as pool:
            all_metadata = []
            for metadata in tqdm(
                pool.imap_unordered(_process_fn, all_paths), total=len(all_paths)
            ):
                all_metadata.append(metadata)

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


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = Args.from_parser()
    main(args)
