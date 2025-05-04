import collections
import dataclasses
import os
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import tree
from Bio import PDB
from numpy import typing as npt

from cogeneration.data.const import CA_IDX
from cogeneration.data.io import read_pkl, write_pkl
from cogeneration.data.protein import chain_str_to_int, process_chain
from cogeneration.data.residue_constants import unk_restype_index
from cogeneration.type.dataset import OLIGOMERIC_PREFIXES, ChainFeatures
from cogeneration.type.dataset import DatasetColumns as dc
from cogeneration.type.dataset import DatasetProteinColumns as dpc
from cogeneration.type.dataset import MetadataCSVRow, ProcessedFile


class DataError(Exception):
    """Error raised when an invalid PDB file is encountered."""

    pass


def _concat_np_features(
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


def _oligomeric_count(
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


def _oligomeric_detail(
    struct_feats: List[Dict[str, np.ndarray]],
) -> str:
    """
    Describe oligomeric detail of each unique sequence in the structure.
    Per-sequence details are comma-separated e.g. "monomeric,24-meric,trimeric"
    """

    counts = [int(x) for x in _oligomeric_count(struct_feats).split(",")]
    return ",".join(
        [
            (
                f"{OLIGOMERIC_PREFIXES[x]}meric"
                if x in OLIGOMERIC_PREFIXES
                else f"{x}-meric"
            )
            for x in counts
        ]
    )


def center_and_scale_chain_feats(
    chain_feats: ChainFeatures, scale_factor: float = 1.0
) -> ChainFeatures:
    """
    Updates `dpc.atom_positions` after centering (if `center` is True) and scaling by `scale_factor`.
    """
    # don't assume `dpc.bb_mask` exists yet
    bb_mask = chain_feats[dpc.atom_mask][:, CA_IDX]
    bb_pos = chain_feats[dpc.atom_positions][:, CA_IDX]
    bb_center = np.sum(bb_pos, axis=0) / (np.sum(bb_mask) + 1e-5)
    centered_pos = chain_feats[dpc.atom_positions] - bb_center[None, None, :]
    scaled_pos = centered_pos / scale_factor

    # update in place
    chain_feats[dpc.atom_positions] = scaled_pos * chain_feats[dpc.atom_mask][..., None]
    chain_feats[dpc.bb_positions] = chain_feats[dpc.atom_positions][:, CA_IDX]

    return chain_feats


def determine_modeled_residues(
    chain_feats: ChainFeatures,
) -> npt.NDArray:
    """
    Determines the modeled residues in `chain_feats`.
    Most appropriate for a chain, rather than complex, but can be used for either.
    However, it is 0-indexed given input, so don't use for multimer chains and then concat.

    raises DataError if no modeled residues are found.

    Returns an array like `[0, 1, 4, 5, ...]` (e.g. if 2, 3 are invalid) of valid residues.
    """
    modeled_residues = np.where(chain_feats[dpc.aatype] != unk_restype_index)[0]
    return modeled_residues


def determine_modeled_residues_mask(
    chain_feats: ChainFeatures,
    trim_chains_independently: bool = True,
) -> npt.NDArray:
    chain_idx = chain_feats[dpc.chain_index]
    modeled_idx = chain_feats[dpc.modeled_idx]
    is_monomer = len(np.unique(chain_feats[dpc.chain_index])) == 1

    # will set regions to keep to True and return
    keep_mask = np.zeros_like(chain_idx, dtype=bool)

    # Match public MultiFlow / FrameFlow behavior
    if is_monomer or not trim_chains_independently:
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        keep_mask[min_idx : (max_idx + 1)] = True
        return keep_mask

    # Trim each chain independently.
    # We want to preserve internal positions within chains, but if there is junk between chains (e.g. solvent)
    # then we drop it.
    # We'll determine the start / end positions of each chain that are modeled, and add to keep_mask
    modeled_mask = np.zeros_like(chain_idx, dtype=bool)
    modeled_mask[chain_feats[dpc.modeled_idx]] = True
    idx = np.arange(chain_idx.size)

    # for each chain, find first / last positions modeled, add to keep_mask
    for cid in np.unique(chain_idx):
        in_chain = chain_idx == cid
        modeled_in_chain = in_chain & modeled_mask
        if modeled_in_chain.any():
            first, last = np.flatnonzero(modeled_in_chain)[[0, -1]]
            keep_mask |= in_chain & (idx >= first) & (idx <= last)

    return keep_mask


def trim_chain_feats_to_modeled_residues(
    chain_feats: ChainFeatures,
    trim_chains_independently: bool = True,
) -> ChainFeatures:
    """
    Trims `chain_feats` to `dpc.modeled_idx` residues.
    Deletes the field, returns a new instance where each field is trimmed.
    Therefore, can only trim once! Should be called on reading a ProcessedFile, but not writing.

    There are two trimming strategies:

    1) Trim the complex.
    The FrameFlow/MultiFlow implementation just trims the ends using min/max modeled positions.
    In public MultiFlow / FrameFlow code, this occurs after processing all the chains, and only considers
    the ends of the concatenated chains.
    In the monomer setting, like the public versions of these two models, this is not a big deal,
    but leaves more junk in multimers, e.g. { chain ... junk ... chain }

    2) Trim chains independently.
    `trim_chains_independently` determines the min/max bounds for each chain independently,
    and keeps all residues in each chain between all non-UNK residues.

    In both cases, invalid residues are left in place, but `BatchProps.res_mask` should use `dpc.bb_mask` to ignore them.
    Remember we want to preserve positions for proteins with gaps i.e. uncaptured neighbors/gaps,
      so we don't impose a penalty on "neighbors" that aren't actually neighbors.
      So don't mask exclusively to known and modeled residues.
    """

    keep_mask = determine_modeled_residues_mask(
        chain_feats=chain_feats,
        trim_chains_independently=trim_chains_independently,
    )

    # del `modeled_idx`. All other features are (P, *), and are trimmed to (N, *)
    del chain_feats[dpc.modeled_idx]

    chain_feats = tree.map_structure(lambda x: x[keep_mask], chain_feats)
    return chain_feats


def _process_chain_feats(
    chain_feats: ChainFeatures,
    center: bool,
    trim_to_modeled_residues: bool,
    trim_chains_independently: bool,
    scale_factor: float = 1.0,
) -> ProcessedFile:
    """
    Parse chain features. Add position information and mask for backbone atoms.
    Returns a complete ProcessedFile.

    Used to load a processed file, which may have come from public MultiFlow data dump.
    Field additions should be idempotent.

    Note that positions are expected in angstroms (PDB style).
    """
    # Center and update positions.
    if center:
        chain_feats = center_and_scale_chain_feats(
            chain_feats,
            scale_factor=scale_factor,
        )

    # Trim to modeled residues.
    # This filters away residues outside range of consideration, yield length N := len(modeled_idx).
    # However, it does not filter away intermediates positions, i.e. if there is a gap in `modeled_idx`.
    if trim_to_modeled_residues:
        chain_feats = trim_chain_feats_to_modeled_residues(
            chain_feats,
            trim_chains_independently=trim_chains_independently,
        )

    return chain_feats


def _process_pdb(
    pdb_file_path: str,
    chain_id: Optional[str] = None,
    pdb_name: Optional[str] = None,
    scale_factor: float = 1.0,
    write_dir: Optional[str] = None,
    generate_metadata: bool = True,
) -> Tuple[MetadataCSVRow, ProcessedFile]:
    """
    Process PDB file into concatenated chain features `ProcessedFile` and metadata `MetadataCSVRow`.

    The metadata file requires access to some intermediates when generating `ProcessedFile`,
    and so we have a unified function for processing the file and generating the metadata.

    If `write_dir` is provided, saves `ProcessedFile` pickle to `{write_dir}/{pdb_name}.pkl`.
    If `chain_id` is provided, only that chain is processed.
    `pdb_name` if provided is used for metadata and written file name.

    During processing, we track and write `dpc.modeled_idx`.
    It can be used on read to trim complex.

    Raises DataError if a known filtering rule is hit. All other errors propogated.
    """
    if pdb_name is None:
        pdb_name = os.path.basename(pdb_file_path).replace(".pdb", "")

    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_file_path)
    struct_chains = {chain.id.upper(): chain for chain in structure.get_chains()}

    # Limit to `chain_id` if specified
    if chain_id is not None:
        if chain_id not in struct_chains:
            raise DataError(f"Chain {chain_id} not found in {pdb_name}")
        struct_chains = {chain_id: struct_chains[chain_id]}

    # Generate chain features
    struct_feats = []
    all_seqs = set()
    for chain_id, chain in struct_chains.items():
        chain_id = chain_str_to_int(chain_id)
        chain_protein = process_chain(chain, chain_id=chain_id)
        chain_feats: ChainFeatures = dataclasses.asdict(chain_protein)  # noqa

        # check for non-protein chains and skip them
        if np.sum(chain_feats[dpc.aatype] != unk_restype_index) == 0:
            continue

        # Add missing fields
        # `chain_feats` then has almost all the fields of a processed file, save for `concat`
        # During processing of individual chains, don't center them. We'll center the complex later.
        # Don't trim modeled residues. Write them. Can trim on load.
        chain_feats[dpc.bb_mask] = chain_feats[dpc.atom_mask][:, CA_IDX]
        chain_feats[dpc.bb_positions] = chain_feats[dpc.atom_positions][:, CA_IDX]

        all_seqs.add(tuple(chain_feats[dpc.aatype]))
        struct_feats.append(chain_feats)

    # Concat features for each chain in complex.
    # Note in AlphaFold2, the merging process is more complex, because it has MSAs, templates, etc.
    #   It separates the merges in `merge_chain_features()` -> `_merge_features_from_multiple_chains`
    #   using enums CHAIN_FEATURES, SEQ_FEATURES, MSA_FEATURES etc.
    #   However, we are really just working with `SEQ_FEATURES` and can concat them all.
    #   We track in the metadata `dc.seq_len` (the only CHAIN_FEATURE we consider) below.
    complex_feats: ChainFeatures = _concat_np_features(struct_feats, False)  # noqa

    # Determine modeled residues.
    complex_feats[dpc.modeled_idx] = determine_modeled_residues(complex_feats)

    # Center the complex
    complex_feats = center_and_scale_chain_feats(complex_feats)

    # Generate metadata file
    metadata: MetadataCSVRow = {}
    if generate_metadata:
        metadata[dc.pdb_name] = pdb_name
        metadata[dc.raw_path] = pdb_file_path

        # Track total sequence length, modeled sequence length
        metadata[dc.seq_len] = len(complex_feats[dpc.aatype])

        # Track modeled sequence lengths
        modeled_idx = complex_feats[dpc.modeled_idx]
        metadata[dc.moduled_num_res] = len(modeled_idx)
        # complex trimming, just determine start and ends
        complex_keep_mask = determine_modeled_residues_mask(
            chain_feats=complex_feats,
            trim_chains_independently=False,
        )
        metadata[dc.modeled_seq_len] = np.sum(complex_keep_mask)
        # chain independent trimming, trim each chain independently
        independent_keep_mask = determine_modeled_residues_mask(
            chain_feats=complex_feats,
            trim_chains_independently=True,
        )
        metadata[dc.modeled_indep_seq_len] = np.sum(independent_keep_mask)

        # Track quaternary / oligomeric information
        # TODO - consider differentiating number chains vs number valid chains
        #    Because some may be filtered out i.e. if they are not proteins
        metadata[dc.num_chains] = len(struct_chains)
        metadata[dc.oligomeric_count] = _oligomeric_count(struct_feats)
        metadata[dc.oligomeric_detail] = _oligomeric_detail(struct_feats)
        if len(all_seqs) == 1:
            metadata[dc.quaternary_category] = "homomer"
        else:
            metadata[dc.quaternary_category] = "heteromer"

        # secondary structure, radius of gyration
        try:
            # MDtraj
            traj = md.load(pdb_file_path)
            # secondary structure calculation
            pdb_ss = md.compute_dssp(traj, simplified=True)
            # radius of gyration calculation
            pdb_rg = md.compute_rg(traj)
        except Exception as e:
            raise DataError(f"Mdtraj failed with error {e}")

        # use denom `metadata[dc.modeled_seq_len]` to match public MultiFlow
        # though if use chain-independent filtering, values almost certainly higher.
        metadata[dc.coil_percent] = np.sum(pdb_ss == "C") / metadata[dc.modeled_seq_len]
        metadata[dc.helix_percent] = (
            np.sum(pdb_ss == "H") / metadata[dc.modeled_seq_len]
        )
        metadata[dc.strand_percent] = (
            np.sum(pdb_ss == "E") / metadata[dc.modeled_seq_len]
        )
        metadata[dc.radius_gyration] = pdb_rg[0]

    # Save pkl after processing is successful
    if write_dir is not None:
        processed_path = os.path.join(write_dir, f"{pdb_name}.pkl")
        metadata[dc.processed_path] = os.path.abspath(processed_path)
        write_pkl(processed_path, complex_feats)
    else:
        metadata[dc.processed_path] = ""

    return metadata, complex_feats


def process_pdb_file(
    pdb_file_path: str,
    pdb_name: str,
    chain_id: Optional[str] = None,
    scale_factor: float = 1.0,
) -> ProcessedFile:
    _, processed_file = _process_pdb(
        pdb_file_path=pdb_file_path,
        chain_id=chain_id,
        pdb_name=pdb_name,
        scale_factor=scale_factor,
        write_dir=None,
        generate_metadata=False,
    )
    return processed_file


def read_processed_file(
    processed_file_path: str,
    center: bool = True,
    trim_to_modeled_residues: bool = True,
    trim_chains_independently: bool = True,
) -> ProcessedFile:
    """
    Loads a processed PDB pkl from `process_pdb_file` and yields a ProcessedFile.
    """
    processed_feats = read_pkl(processed_file_path)
    processed_file = _process_chain_feats(
        processed_feats,
        center=center,
        trim_to_modeled_residues=trim_to_modeled_residues,
        trim_chains_independently=trim_chains_independently,
    )
    return processed_file
