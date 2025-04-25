"""Script for preprocessing PDB files to generate pkls expected by DataLoader"""

import argparse
import functools as fn
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from tqdm.auto import tqdm

from cogeneration.dataset.process_pdb import DataError, _process_pdb
from cogeneration.type.dataset import MetadataCSVRow, ProcessedFile


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


def process_serially(
    all_paths: List[str],
    write_dir: str,
    delete_original: bool = False,
    verbose: bool = False,
) -> List[MetadataCSVRow]:
    all_metadata = []
    for i, file_path in tqdm(enumerate(all_paths), desc="Processing PDBs"):
        try:
            start_time = time.time()
            metadata, _ = process_pdb_with_metadata(
                pdb_file_path=file_path, write_dir=write_dir
            )
            elapsed_time = time.time() - start_time
            if verbose:
                print(f"Finished {file_path} in {elapsed_time:2.2f}s")
            all_metadata.append(metadata)
            if delete_original:
                os.remove(file_path)
        except DataError as e:
            if verbose:
                print(f"Failed {file_path}: {e}")
    return all_metadata


def process_fn(
    file_path: str, write_dir: str, delete_original: bool = False, verbose: bool = False
):
    try:
        start_time = time.time()
        metadata, _ = process_pdb_with_metadata(
            pdb_file_path=file_path, write_dir=write_dir
        )
        elapsed_time = time.time() - start_time
        if verbose:
            print(f"Finished {file_path} in {elapsed_time:2.2f}s")
        if delete_original:
            os.remove(file_path)
        return metadata
    except DataError as e:
        if verbose:
            print(f"Failed {file_path}: {e}")


@dataclass
class Args:
    pdb_dir: str
    num_processes: Optional[int] = None
    write_dir: str = "preprocessed"
    delete_original: bool = False
    debug: bool = False
    verbose: bool = False

    def __post_init__(self):
        if self.debug:
            self.num_processes = 1
            self.verbose = True
        if self.num_processes is None:
            self.num_processes = mp.cpu_count()
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
            "--verbose", help="Whether to log everything.", action="store_true"
        )
        args = parser.parse_args()

        return cls(
            pdb_dir=args.pdb_dir,
            num_processes=args.num_processes,
            write_dir=args.write_dir,
            delete_original=args.delete_original,
            debug=args.debug,
            verbose=args.verbose,
        )


def main(args: Args):
    pdb_dir = args.pdb_dir
    all_file_paths = [
        os.path.join(pdb_dir, x) for x in os.listdir(args.pdb_dir) if ".pdb" in x
    ]
    total_num_paths = len(all_file_paths)
    write_dir = args.write_dir

    if args.debug:
        metadata_file_name = "metadata_debug.csv"
    else:
        metadata_file_name = "metadata.csv"
    metadata_path = os.path.join(write_dir, metadata_file_name)
    print(f"Files will be written to {write_dir}")

    # Process each file
    if args.num_processes == 1 or args.debug:
        all_metadata = process_serially(
            all_file_paths,
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
                pool.imap_unordered(_process_fn, all_file_paths), total=total_num_paths
            ):
                all_metadata.append(metadata)

    # filter empty
    all_metadata = [x for x in all_metadata if x is not None]

    metadata_df = pd.DataFrame(all_metadata)
    metadata_df.to_csv(metadata_path, index=False)
    succeeded = len(all_metadata)
    print(f"Finished processing {succeeded}/{total_num_paths} files")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    args = Args.from_parser()
    main(args)
