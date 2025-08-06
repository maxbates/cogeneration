#!/usr/bin/env python3

"""
Runs Foldseek's `easy-cluster` on one or more directories of structures and
writes a single `<out>.clusters` file (one line = one cluster).

Clustering may take some time, depending on the number of input structures.

https://github.com/steineggerlab/foldseek

TODO save DB for assessing designability later

Example:
    python cluster_pdbs.py \
    --dirs ~/pdb/rcsb/processed/raw ~/pdb/alphafold/raw
    --prefix cogeneration
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from tqdm.auto import tqdm

from cogeneration.dataset.process_pdb import pdb_path_pdb_name


class Args:
    dirs: List[Path]
    prefix: Path
    tm_score: float
    cov: float
    cov_mode: int
    threads: int
    copy_representatives: bool

    @classmethod
    def from_parser(cls):
        p = argparse.ArgumentParser()
        p.add_argument(
            "--dirs",
            nargs="+",
            required=True,
            type=Path,
            help="Directories with .pdb/.cif[.gz] files",
        )
        p.add_argument(
            "--prefix",
            required=True,
            type=Path,
            help="Output prefix (writes <prefix>.clusters)",
        )
        p.add_argument(
            "--tm-score",
            type=float,
            default=0.6,
            help="TM-score threshold (Foldseek --tmscore-threshold)",
        )
        p.add_argument(
            "--cov", type=float, default=0.9, help="-c coverage threshold (0-1)"
        )
        p.add_argument(
            "--cov-mode",
            type=int,
            choices=(0, 1, 2),
            default=0,
            help="Foldseek --cov-mode (0: long/long, 1: short/long, 2: long/short)",
        )
        p.add_argument(
            "--threads",
            type=int,
            default=max(1, os.cpu_count() - 2),
            help="CPU threads",
        )
        p.add_argument(
            "--copy-representatives",
            action="store_true",
            help="Copy representative PDBs next to <prefix>.clusters",
        )
        a = p.parse_args()

        return cls(
            dirs=a.dirs,
            prefix=a.prefix,
            tm_score=a.tm_score,
            cov=a.cov,
            cov_mode=a.cov_mode,
            threads=a.threads,
            copy_representatives=a.copy_representatives,
        )


def run_subprocess(cmd: str) -> None:
    """Run shell command, abort on error"""
    print("+", cmd, file=sys.stderr, flush=True)
    subprocess.run(cmd, shell=True, check=True)


def ensure_foldseek_installed() -> None:
    """Abort if Foldseek is not on PATH"""
    if shutil.which("foldseek") is None:
        sys.exit(
            "ERROR: Foldseek not found on PATH – see https://github.com/steineggerlab/foldseek"
        )


def symlink_input_dirs(dirs: List[Path], input_dir: Path) -> Path:
    """
    Symlink all structure files into a temp workspace,
    renaming to match processing script PDB names.
    """
    print("Creating symlinks...")
    input_dir.mkdir(parents=True, exist_ok=True)

    exts = {".pdb", ".cif", ".ent", ".pdb.gz", ".cif.gz", ".ent.gz"}
    for d in dirs:
        num_links = 0

        # collect all files, show single progress bar
        all_files = []
        for root, _, files in os.walk(d):
            for file in files:
                all_files.append((root, file))

        for root, file in tqdm(all_files, desc=f"Symlinking {d} -> {input_dir}"):
            f = Path(root) / file
            for ext in exts:
                if f.name.lower().endswith(ext):
                    pdb_name = pdb_path_pdb_name(f.name)
                    file_name = f"{pdb_name}{f.suffix}"

                    # handle renaming PDB .ent / .ent.gz files
                    file_name = file_name.replace(" ", "_")
                    file_name = file_name.replace(".ent", ".pdb")

                    try:
                        (input_dir / file_name).symlink_to(f.resolve())
                        num_links += 1
                    except FileExistsError:
                        print(
                            f"Skipping {f} ({pdb_name}), already exists @ {input_dir / pdb_name}"
                        )

                    break

        print(f"{num_links} symlinks in {d}")

    return input_dir


def main() -> None:
    args = Args.from_parser()

    ensure_foldseek_installed()

    out_dir = args.out_dir.resolve()
    print(f"Output directory: {out_dir}")

    # Create workspace
    ws = Path(tempfile.mkdtemp(prefix="foldseek_ws_"))
    structures = ws / "input"
    cluster_prefix = ws / "clu"
    tmp_dir = ws / "tmp"

    symlink_input_dirs(args.dirs, input_dir=structures)

    # Cluster
    run_subprocess(
        "foldseek easy-cluster "
        f"{structures} {cluster_prefix} {tmp_dir} "
        f"--alignment-type 1 "  # TM-align global metric
        f"--tmscore-threshold {args.tm_score} "
        f"--min-seq-id 0 "  # no minimum sequence identity
        f"-c {args.cov} "  # 80% coverage threshold
        f"--cov-mode {args.cov_mode} "
        f"--threads {args.threads} "
    )

    # Copy output clusters
    clu_tsv = Path(f"{cluster_prefix}_rep_clu.tsv")
    if not clu_tsv.exists():
        clu_tsv = Path(f"{cluster_prefix}_cluster.tsv")
    clusters_out = out_dir.with_suffix(".clusters")
    shutil.copy2(clu_tsv, clusters_out)
    print(f"Wrote clusters to: {clusters_out}")

    # Copy representative structures, if requested
    if args.copy_representatives:
        reps_dir = out_dir.parent / f"{out_dir.name}_reps"
        reps_dir.mkdir(parents=True, exist_ok=True)

        with open(clu_tsv) as fh:
            for line in fh:
                rep = line.split()[0]
                src = ws / rep
                if src.exists():
                    shutil.copy2(src, reps_dir / rep)


if __name__ == "__main__":
    main()
