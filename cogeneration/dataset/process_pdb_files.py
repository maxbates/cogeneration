"""Script for preprocessing PDB files to generate pkls expected by DataLoader"""

import argparse
import collections
import dataclasses
import functools as fn
import multiprocessing as mp
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import mdtraj as md
import numpy as np
import numpy.typing as npt
import pandas as pd
from Bio import PDB
from tqdm.auto import tqdm

from cogeneration.data.const import ALPHANUMERIC, CHAIN_TO_INT
from cogeneration.data.enum import DatasetColumns as dc
from cogeneration.data.enum import DatasetProteinColumns as dpc
from cogeneration.data.io import write_pkl
from cogeneration.data.protein import process_chain
from cogeneration.data.residue_constants import unk_restype_index
from cogeneration.dataset.data_utils import parse_chain_feats
from cogeneration.dataset.util import MetadataCSVRow

# TODO - support MMCIF files


class DataError(Exception):
    """Error raised when an invalid PDB file is encountered."""

    pass


def chain_str_to_int(chain_str: str):
    chain_int = 0
    if len(chain_str) == 1:
        return CHAIN_TO_INT[chain_str]
    for i, chain_char in enumerate(chain_str):
        chain_int += CHAIN_TO_INT[chain_char] + (i * len(ALPHANUMERIC))
    return chain_int


def concat_np_features(
    np_dicts: List[Dict[str, npt.NDArray]], add_batch_dim: bool
) -> Dict[str, npt.NDArray]:
    """Performs a nested concatenation of feature dicts.

    Args:
        np_dicts: list of dicts with the same structure.
            Each dict must have the same keys and numpy arrays as the values.
        add_batch_dim: whether to add a batch dimension to each feature.

    Returns:
        A single dict with all the features concatenated.
    """
    combined_dict = collections.defaultdict(list)
    for chain_dict in np_dicts:
        for feat_name, feat_val in chain_dict.items():
            if add_batch_dim:
                feat_val = feat_val[None]
            combined_dict[feat_name].append(feat_val)
    # Concatenate each feature
    for feat_name, feat_vals in combined_dict.items():
        combined_dict[feat_name] = np.concatenate(feat_vals, axis=0)
    return combined_dict


def oligomeric_count(
    struct_feats: List[Dict[str, np.ndarray]],
) -> str:
    """
    Generate comma-separated oligomeric counts for each unique sequence.
    For example, "1,2,3" means there are 1 monomeric, 2 dimeric, and 3 trimeric
    """
    seq_counts = {}
    for chain_dict in struct_feats:
        seq = tuple(chain_dict[dpc.aatype])
        if seq not in seq_counts:
            seq_counts[seq] = 0
        seq_counts[seq] += 1
    return ",".join([str(x) for x in sorted(set(seq_counts.values()))])


def oligomeric_detail(
    struct_feats: List[Dict[str, np.ndarray]],
) -> str:
    """
    Describe oligomeric detail of each unique sequence in the structure.
    Per-sequence details are comma-separated e.g. "monomeric,24-meric,trimeric"

    Note the descriptions vary more widely in the metadata provided by MultiFlow,
    perhaps taken from PDB directly, but this is fairly close.
    """
    count_to_prefix = {
        1: "mono",
        2: "di",
        3: "tri",
        4: "tetra",
        5: "penta",
        6: "hexa",
        7: "hepta",
        8: "octa",
        9: "nona",
        10: "deca",
        11: "undeca",
        12: "dodeca",  # pref to `duodeca`
        13: "trideca",
        14: "tetradeca",
        15: "pentadeca",
        16: "hexadeca",
        17: "heptadeca",
        18: "octadeca",
        19: "nonadeca",
        20: "eicosa",  # pref to `icosa`
    }
    counts = [int(x) for x in oligomeric_count(struct_feats).split(",")]
    return ",".join(
        [
            f"{count_to_prefix[x]}meric" if x in count_to_prefix else f"{x}-meric"
            for x in counts
        ]
    )


def process_file(file_path: str, write_dir: str) -> MetadataCSVRow:
    """Processes protein file into usable, smaller pickles.

    Args:
        file_path: Path to file to read.
        write_dir: Directory to write pickles to.

    Returns:
        Saves processed protein to pickle and returns metadata.

    Raises:
        DataError if a known filtering rule is hit.
        All other errors are unexpected and are propogated.
    """
    # TODO(multimer) reconcile functionality with data_utils `parse_pdb_feats`
    metadata = {}
    pdb_name = os.path.basename(file_path).replace(".pdb", "")
    metadata[dc.pdb_name] = pdb_name

    processed_path = os.path.join(write_dir, f"{pdb_name}.pkl")
    metadata[dc.processed_path] = os.path.abspath(processed_path)
    metadata[dc.raw_path] = file_path
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, file_path)

    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}

    # Generate chain features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        chain_id = chain_str_to_int(chain_id)
        chain_protein = process_chain(chain, chain_id=chain_id)
        chain_dict = dataclasses.asdict(chain_protein)
        chain_dict = parse_chain_feats(chain_dict, center=False)
        all_seqs.add(tuple(chain_dict[dpc.aatype]))
        struct_feats.append(chain_dict)

    # TODO - filter out chains which are small non-protein molecules
    #   For now, assume HETATMs ignored / not present in PDB file.
    #   More important for CIF.

    # Concat features for each chain in complex
    complex_feats = concat_np_features(struct_feats, False)

    # Track quaternary / oligomeric information
    metadata[dc.num_chains] = len(struct_chains)
    metadata[dc.oligomeric_count] = oligomeric_count(struct_feats)
    metadata[dc.oligomeric_detail] = oligomeric_detail(struct_feats)
    if len(all_seqs) == 1:
        metadata[dc.quaternary_category] = "homomer"
    else:
        metadata[dc.quaternary_category] = "heteromer"

    # Determine residues to be modeled
    complex_aatype = complex_feats[dpc.aatype]
    metadata[dc.seq_len] = len(complex_aatype)
    modeled_idx = np.where(complex_aatype != unk_restype_index)[0]
    if np.sum(complex_aatype != unk_restype_index) == 0:
        raise DataError("No modeled residues")
    min_modeled_idx = np.min(modeled_idx)
    max_modeled_idx = np.max(modeled_idx)
    metadata[dc.modeled_seq_len] = max_modeled_idx - min_modeled_idx + 1
    metadata[dc.moduled_num_res] = len(modeled_idx)
    complex_feats[dpc.modeled_idx] = modeled_idx

    # secondary structure, radius of gyration
    # TODO - consider tracking secondary structure to each chain, rather only than complex metrics
    try:
        # MDtraj
        traj = md.load(file_path)
        # secondary structure calculation
        pdb_ss = md.compute_dssp(traj, simplified=True)
        # radius of gyration calculation
        pdb_rg = md.compute_rg(traj)
    except Exception as e:
        raise DataError(f"Mdtraj failed with error {e}")
    metadata[dc.coil_percent] = np.sum(pdb_ss == "C") / metadata[dc.modeled_seq_len]
    metadata[dc.helix_percent] = np.sum(pdb_ss == "H") / metadata[dc.modeled_seq_len]
    metadata[dc.strand_percent] = np.sum(pdb_ss == "E") / metadata[dc.modeled_seq_len]
    metadata[dc.radius_gyration] = pdb_rg[0]

    write_pkl(processed_path, complex_feats)

    return metadata


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
            metadata = process_file(file_path=file_path, write_dir=write_dir)
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
        metadata = process_file(file_path=file_path, write_dir=write_dir)
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
