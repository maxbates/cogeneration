#!/usr/bin/env python3

"""
Download AlphaFold PDB tarball for SwissProt from EBI and extract to local directory.
Only extracts .pdb.gz files, skipping .cif.gz files.

You can download more than one database by specifying the URL (and local path).
See https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/

Specify `extract_dir` to keep separate and treat as separate dataset.
Otherwise, structures from multiple sources can be mixed in the same directory.

Example:
    python download_alphafold.py \
    --url https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/UP000005640_9606_HUMAN_v4.tar \
    --tarball_path ~/pdb/alphafold/swissprot_human_pdb_v4.tar
    [--extract_dir ~/pdb/alphafold_human/raw]

"""

import argparse
import os
import shutil
import tarfile
from dataclasses import dataclass
from typing import Optional

import requests
from tqdm import tqdm


@dataclass
class Args:
    tarball_path: str  # Local path for downloaded tarball
    extract_dir: str  # Directory to extract PDB files to
    url: str  # URL to download from
    timeout: int  # Download timeout in seconds
    chunk_size: int  # Download chunk size
    remove_tarball: bool  # Remove tarball after extraction

    def __post_init__(self):
        self.tarball_path = os.path.expanduser(self.tarball_path)
        self.extract_dir = os.path.expanduser(self.extract_dir)

        # Create parent directories
        os.makedirs(os.path.dirname(self.tarball_path), exist_ok=True)
        os.makedirs(self.extract_dir, exist_ok=True)

    @classmethod
    def from_parser(cls) -> "Args":
        parser = argparse.ArgumentParser(
            description="Download AlphaFold PDB tarball and extract"
        )
        parser.add_argument(
            "--tarball_path",
            help="Local path for downloaded tarball",
            type=str,
            default=os.path.join(
                os.path.expanduser("~"), "pdb", "alphafold", "swissprot_pdb_v4.tar"
            ),
        )
        parser.add_argument(
            "--extract_dir",
            help="Directory to extract PDB files to",
            type=str,
            default=os.path.join(os.path.expanduser("~"), "pdb", "alphafold", "raw"),
        )
        parser.add_argument(
            "--url",
            help="URL to download tarball from",
            type=str,
            default="https://ftp.ebi.ac.uk/pub/databases/alphafold/v4/swissprot_pdb_v4.tar",
        )
        parser.add_argument(
            "--timeout",
            help="Download timeout in seconds",
            type=int,
            default=4 * 60 * 60,  # 4 hours
        )
        parser.add_argument(
            "--chunk_size",
            help="Download chunk size in bytes",
            type=int,
            default=8192,
        )
        parser.add_argument(
            "--remove_tarball",
            help="Remove tarball after extraction",
            action="store_true",
        )
        args = parser.parse_args()
        return cls(
            tarball_path=args.tarball_path,
            extract_dir=args.extract_dir,
            url=args.url,
            timeout=args.timeout,
            chunk_size=args.chunk_size,
            remove_tarball=args.remove_tarball,
        )


def download_tarball(
    url: str, tarball_path: str, timeout: int, chunk_size: int
) -> bool:
    """
    Download tarball from URL to local path with progress bar.
    Returns True on success, False on failure.
    """
    try:
        print(f"Downloading {url} to {tarball_path}...")

        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        # Get total file size if available
        total_size = int(response.headers.get("content-length", 0))

        with open(tarball_path, "wb") as f:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc="Downloading",
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

        print(f"Successfully downloaded to {tarball_path}")
        return True

    except Exception as e:
        print(f"Error downloading tarball: {e}")
        # Clean up partial download
        if os.path.exists(tarball_path):
            os.remove(tarball_path)
        return False


def extract_tarball(tarball_path: str, extract_dir: str) -> bool:
    """
    Extract tarball to specified directory, filtering out .cif.gz files.
    Returns True on success, False on failure.
    """
    try:
        print(f"Extracting {tarball_path} to {extract_dir}...")

        with tarfile.open(tarball_path, "r") as tar:
            # Get all members and filter to only .pdb.gz files
            all_members = tar.getmembers()
            pdb_members = [
                member
                for member in all_members
                if member.name.endswith(".pdb.gz")
                and not member.name.endswith(".cif.gz")
            ]

            print(
                f"Found {len(all_members)} total files, extracting {len(pdb_members)} .pdb.gz files (skipping .cif.gz files)"
            )

            with tqdm(total=len(pdb_members), desc="Extracting", unit="file") as pbar:
                for member in pdb_members:
                    tar.extract(member, path=extract_dir)
                    pbar.update(1)

        print(f"Successfully extracted {len(pdb_members)} PDB files to {extract_dir}")
        return True

    except Exception as e:
        print(f"Error extracting tarball: {e}")
        return False


def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB."""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 * 1024)
    return 0


def get_dir_size_mb(directory: str) -> float:
    """Get directory size in MB."""
    total_size_mb = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            total_size_mb += os.path.getsize(file_path) / (1024 * 1024)
    return total_size_mb


def count_pdb_files(directory: str) -> int:
    """Count PDB files in directory (.pdb and .pdb.gz)."""
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".pdb") or file.endswith(".pdb.gz"):
                count += 1
    return count


def main(args: Args) -> None:
    print(f"AlphaFold tarball URL: {args.url}")
    print(f"Local tarball path: {args.tarball_path}")
    print(f"Extract directory: {args.extract_dir}")

    # Check if tarball already exists
    if os.path.exists(args.tarball_path):
        size_mb = get_file_size_mb(args.tarball_path)
        print(f"Tarball already exists ({size_mb:.1f} MB), skipping download...")
    else:
        # Download the tarball
        if not download_tarball(
            args.url, args.tarball_path, args.timeout, args.chunk_size
        ):
            print("Failed to download tarball")
            return

    # Check if already extracted
    existing_pdb_count = count_pdb_files(args.extract_dir)
    if existing_pdb_count > 0:
        print(
            f"Found {existing_pdb_count} PDB files already extracted in {args.extract_dir}"
        )

    # Extract the tarball
    if not extract_tarball(args.tarball_path, args.extract_dir):
        print("Failed to extract tarball")
        return

    # Final stats
    final_pdb_count = count_pdb_files(args.extract_dir)
    final_size_mb = get_dir_size_mb(args.extract_dir)
    print(
        f"Total size of {final_pdb_count} PDBs in {args.extract_dir}: {final_size_mb:.1f} MB"
    )

    # Remove tarball
    if args.remove_tarball:
        print(f"Removing {args.tarball_path}...")
        os.remove(args.tarball_path)
    else:
        print(f"You can remove tarball with `rm {args.tarball_path}`")


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args = Args.from_parser()
    main(args)
