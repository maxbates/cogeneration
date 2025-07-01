#!/usr/bin/env python3

"""
Download RCSB PDB files to a local directory.
"""


import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os import cpu_count
from typing import List, Optional

import requests
from tqdm import tqdm


@dataclass
class Args:
    pdb_dir: str  # Local directory for downloaded PDB files
    target_path: str  # Remote subdirectory path (and URL suffix) of PDB files
    debug: bool  # If true, fetch only first 100 entries via HTTP
    # download + lookup configuration
    num_workers: Optional[int]
    timeout: int
    retries: int
    rsync_server: str
    port: int

    def __post_init__(self):
        self.pdb_dir = os.path.expanduser(self.pdb_dir)
        os.makedirs(self.pdb_dir, exist_ok=True)

        if self.debug:
            self.num_workers = 1

    @classmethod
    def from_parser(cls) -> "Args":
        parser = argparse.ArgumentParser(
            description="Sync RCSB PDB files and preprocess"
        )
        parser.add_argument(
            "--pdb_dir",
            help="Local directory for downloaded PDB files",
            type=str,
            default=os.path.join(os.path.expanduser("~"), "pdb", "rcsb", "raw"),
        )
        parser.add_argument(
            "--target_path",
            help="Remote subdirectory path and URL suffix (e.g. structures/divided/pdb)",
            type=str,
            default="structures/divided/pdb",
        )
        parser.add_argument(
            "--debug",
            help="Fetch only first 100 entries via HTTP",
            action="store_true",
        )
        parser.add_argument(
            "--num_workers",
            help="Number of parallel downloads",
            type=int,
            default=max(1, cpu_count() // 2, cpu_count() - 4),
        )
        parser.add_argument(
            "--timeout",
            help="Per-file timeout (sec)",
            type=int,
            default=300,
        )
        parser.add_argument(
            "--retries",
            help="Retries on failure",
            type=int,
            default=3,
        )
        parser.add_argument(
            "--rsync_server",
            help="Rsync server for dry-run",
            type=str,
            default="rsync.wwpdb.org::ftp",
        )
        parser.add_argument(
            "--port",
            help="Rsync port",
            type=int,
            default=33444,
        )
        args = parser.parse_args()
        return cls(
            pdb_dir=args.pdb_dir,
            target_path=args.target_path,
            debug=args.debug,
            num_workers=args.num_workers,
            timeout=args.timeout,
            retries=args.retries,
            rsync_server=args.rsync_server,
            port=args.port,
        )


def rsync_dryrun(
    server: str,
    target_path: str,
    pdb_dir: str,
    port: int,
) -> List[str]:
    """
    Perform an rsync dry-run to list files that would be updated or added.
    Returns a list of relative file paths (e.g., '07/107b.pdb').
    """
    cmd = [
        "rsync",
        "-rlptvn",
        "--port",
        str(port),
        f"{server}/data/{target_path}/",
        f"{pdb_dir}/",
        "--out-format=%n",
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def fetch_first_100(
    base_url: str,
    pdb_dir: str,
) -> None:
    """
    Fetch the first 100 PDB entry IDs from the RCSB API
    and download their .ent.gz files via HTTP.
    """
    resp = requests.get("https://data.rcsb.org/rest/v1/holdings/current/entry_ids")
    resp.raise_for_status()
    ids = resp.json()[:100]
    for eid in ids:
        eid = eid.lower()
        subdir = eid[1:3]
        fname = f"pdb{eid}.ent.gz"
        url = f"{base_url}/{subdir}/{fname}"
        dst = os.path.join(pdb_dir, subdir, fname)
        if os.path.exists(dst):
            continue
        try:
            r = requests.get(url, stream=True, timeout=30)
            r.raise_for_status()
            with open(dst, "wb") as f:
                for chunk in r.iter_content(16_384):
                    f.write(chunk)
        except Exception:
            print(f"[ERROR] failed to download {url}", file=sys.stderr)


def download_file_http(
    base_url: str,
    rel_path: str,
    pdb_dir: str,
    timeout: int,
) -> bool:
    """
    Download a single '.ent.gz' file from the RCSB PDB via HTTP.
    base_url + '/' + rel_path => full URL.
    Saves into pdb_dir/rel_path, creating subdirs as needed.
    Returns True on success, False on failure.
    """
    url = f"{base_url}/{rel_path}"
    dst = os.path.join(pdb_dir, rel_path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(dst, "wb") as f:
            for chunk in r.iter_content(16_384):
                f.write(chunk)
        return True
    except Exception:
        return False


def main(args: Args) -> None:
    pdb_http_base = f"https://files.wwpdb.org/pub/pdb/data/{args.target_path}"

    print(f"PDB base URL: {pdb_http_base}")
    print(f"Local PDB directory: {args.pdb_dir}")
    num_pdbs_downloaded = sum(
        1
        for root, _, files in os.walk(args.pdb_dir)
        for fn in files
        if fn.endswith((".ent.gz", ".pdb"))
    )
    print(f"Found {num_pdbs_downloaded} PDB files in {args.pdb_dir}")

    # If debuging, fetch the first 100 entries
    if args.debug:
        print("[Debug mode] Fetching first 100 PDB entries...")
        fetch_first_100(base_url=pdb_http_base, pdb_dir=args.pdb_dir)
        return

    # Otherwise, fetch a complete list and download in parallel
    print(f"Finding new PDB files @ {args.rsync_server} -> {args.pdb_dir}...")
    to_sync = rsync_dryrun(
        server=args.rsync_server,
        target_path=args.target_path,
        pdb_dir=args.pdb_dir,
        port=args.port,
    )

    print(f"Found {len(to_sync)} files to sync, beginning download...")
    failed: List[str] = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as exe:
        futures = {
            exe.submit(
                download_file_http,
                pdb_http_base,
                relative_path,
                args.pdb_dir,
                args.timeout,
            ): relative_path
            for relative_path in to_sync
        }
        for future in tqdm(as_completed(futures), total=len(futures), unit="file"):
            relative_path = futures[future]
            if not future.result():
                failed.append(relative_path)

    if failed:
        log_path = os.path.join(args.pdb_dir, "failed_syncs.log")
        with open(log_path, "w") as f:
            f.write("\n".join(failed))
        print(f"[WARN] {len(failed)} failures, see {log_path}", file=sys.stderr)


if __name__ == "__main__":
    # Don't use GPU
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    args = Args.from_parser()
    main(args)
