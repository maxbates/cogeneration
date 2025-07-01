import collections
import gzip
import logging
import os
import tempfile
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import mdtraj as md
import numpy as np
import torch
import tree
from Bio import PDB
from Bio.PDB.Structure import Structure
from numpy import typing as npt

from cogeneration.data.const import CA_IDX, INT_TO_CHAIN
from cogeneration.data.io import read_pkl, write_pkl
from cogeneration.data.protein import chain_str_to_int, process_chain
from cogeneration.data.residue_constants import unk_restype_index
from cogeneration.dataset.interaction import (
    MultimerInteractions,
    NonResidueInteractions,
)
from cogeneration.type.dataset import OLIGOMERIC_PREFIXES, ChainFeatures
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataCSVRow, ProcessedFile
from cogeneration.type.structure import (
    StructureExperimentalMethod,
    extract_structure_date,
)


class DataError(Exception):
    """Error raised when an invalid PDB file is encountered."""

    pass


def pdb_path_pdb_name(pdb_path: str) -> str:
    """
    Convert a potentially compressed PDB file path to a PDB name.
    """
    # Remove the directory and file extension
    pdb_name = os.path.basename(pdb_path).upper()

    # for compressed PDBs, remove leading pdb e.g. `pdb0000.ent.gz`
    if pdb_name.startswith("PDB") and pdb_name.endswith(".ENT.GZ"):
        pdb_name = pdb_name[3:]

    pdb_name = pdb_name.replace(".ENT.GZ", "")
    pdb_name = pdb_name.replace(".PDB.GZ", "")
    pdb_name = pdb_name.replace(".PDB", "")

    # No special handling for AlphaFold names, keep as is

    return pdb_name


def get_uncompressed_pdb_path(
    file_path: str,
) -> Tuple[str, bool]:
    """
    Uncompress a PDB file if it is compressed, returns path and whether it is a temp path

    TODO(dataset) consider moving to context handler to close file more easily.
    """
    is_compressed = file_path.endswith(".ent.gz") or file_path.endswith(".pdb.gz")

    if is_compressed:
        pdb_name = pdb_path_pdb_name(file_path)

        with gzip.open(file_path, "rt") as gz:
            # write decompressed content to a temp file with correct pdb_name
            tmp_dir = tempfile.mkdtemp()
            uncompressed_path = os.path.join(tmp_dir, f"{pdb_name}.pdb")
            with open(uncompressed_path, "wb") as tmp:
                tmp.write(gz.read().encode("utf-8"))
    else:
        uncompressed_path = file_path

    return uncompressed_path, is_compressed


def _concat_np_features(
    np_dicts: List[ChainFeatures], add_batch_dim: bool
) -> ChainFeatures:
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
    struct_feats: List[ChainFeatures],
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
    struct_feats: List[ChainFeatures],
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


def _chain_lengths(
    struct_feats: List[ChainFeatures],
    modeled_only: bool,
) -> str:
    """
    Generate chain lengths in format "<chain_id>:<num_residues>,..."
    or "<chain_id>:<num_modeled_residues>,..." if `modeled_only` is True.
    """
    counts = {}
    for chain_dict in struct_feats:
        chain_id = chain_dict[dpc.chain_index][0]
        seq = chain_dict[dpc.aatype]
        if modeled_only:
            seq = seq[determine_modeled_residues(chain_dict)]
        counts[chain_id] = len(seq)

    return ",".join(
        [
            f"{INT_TO_CHAIN[chain_id]}:{count}"
            for chain_id, count in sorted(counts.items())
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
    It is 0-indexed given input, so don't use for multimer chains independently then concat.

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


def process_chain_feats(
    chain_feats: ChainFeatures,
    center: bool,
    trim_to_modeled_residues: bool,
    trim_chains_independently: bool,
    scale_factor: float = 1.0,
) -> ProcessedFile:
    """
    Parse chain features. Add position information and mask for backbone atoms.
    Returns a complete ProcessedFile.

    Called by `read_processed_file`.
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


def pdb_structure_to_chain_feats(
    structure: Structure,
    chain_ids: Optional[List[str]] = None,
) -> List[ChainFeatures]:
    """
    Convert PDB structure to collection of individual chains as ChainFeatures
    Optionally limit to `chain_ids`.
    """
    struct_chains = {}
    # Take first occurrence of each chain id
    # Handles trajectories by taking the first frame (assumes chain id preserved across frames)
    for chain in structure.get_chains():
        cid = chain.id.upper()
        if cid not in struct_chains:
            struct_chains[cid] = chain

    # Limit to `chain_ids` if specified
    if chain_ids is not None:
        chain_ids_upper = [cid.upper() for cid in chain_ids]
        missing_chains = set(chain_ids_upper) - set(struct_chains.keys())
        if missing_chains:
            raise DataError(f"Chain(s) {missing_chains} not found in {structure.id}")
        struct_chains = {cid: struct_chains[cid] for cid in chain_ids_upper}

    # struct_feats are individual chains, can be concat to complex_feats
    struct_feats: List[ChainFeatures] = []
    for chain_id, chain in struct_chains.items():
        chain_id = chain_str_to_int(chain_id)
        chain_protein = process_chain(chain, chain_id=chain_id)
        chain_feats: ChainFeatures = asdict(chain_protein)  # noqa

        # check for non-residue chains and skip them
        if (
            len(chain_feats[dpc.aatype]) == 0
            or np.sum(chain_feats[dpc.aatype] != unk_restype_index) == 0
        ):
            continue

        # Add missing fields
        # `chain_feats` then has almost all the fields of a processed file, save for `concat`
        # During processing of individual chains, don't center them. We'll center the complex later.
        # Don't trim modeled residues. Write them. Can trim on load.
        chain_feats[dpc.bb_mask] = chain_feats[dpc.atom_mask][:, CA_IDX]
        chain_feats[dpc.bb_positions] = chain_feats[dpc.atom_positions][:, CA_IDX]

        struct_feats.append(chain_feats)

    return struct_feats


def _process_pdb(
    pdb_file_path: str,
    chain_ids: Optional[List[str]] = None,
    pdb_name: Optional[str] = None,
    scale_factor: float = 1.0,
    write_dir: Optional[str] = None,
    generate_metadata: bool = True,
    max_combined_length: int = 8192,
) -> Tuple[MetadataCSVRow, ProcessedFile]:
    """
    Process PDB file into concatenated chain features `ProcessedFile` and metadata `MetadataCSVRow`.

    The metadata file requires access to some intermediates when generating `ProcessedFile`,
    and so we have a unified function for processing the file and generating the metadata.

    If `write_dir` is provided, saves `ProcessedFile` pickle to `{write_dir}/{pdb_name}.pkl`.
    If `chain_ids` is provided, only those chains are processed.
    `pdb_name` if provided is used for metadata and written file name.

    During processing, we track and write `dpc.modeled_idx`.
    It can be used on read to trim complex.

    Raises DataError if a known filtering rule is hit. All other errors propogated.
    """
    if pdb_name is None:
        pdb_name = pdb_path_pdb_name(pdb_file_path)

    uncompressed_pdb_path, is_temp_file = get_uncompressed_pdb_path(pdb_file_path)

    try:
        parser = PDB.PDBParser(QUIET=True)
        structure = parser.get_structure(pdb_name, uncompressed_pdb_path)

        struct_feats = pdb_structure_to_chain_feats(structure, chain_ids=chain_ids)

        if len(struct_feats) == 0:
            raise DataError(f"No valid chains found")

        # Concat features for each chain in complex.
        # Note in AlphaFold2, the merging process is more complex, because it has MSAs, templates, etc.
        #   It separates the merges in `merge_chain_features()` -> `_merge_features_from_multiple_chains`
        #   using enums CHAIN_FEATURES, SEQ_FEATURES, MSA_FEATURES etc.
        #   However, we are really just working with `SEQ_FEATURES` and can concat them all.
        #   We track in the metadata `dc.seq_len` (the only CHAIN_FEATURE we consider) below.
        complex_feats: ChainFeatures = _concat_np_features(struct_feats, False)  # noqa

        if complex_feats[dpc.aatype].shape[0] > max_combined_length:
            raise DataError(
                f"Combined length of {complex_feats[dpc.aatype].shape[0]} exceeds max_combined_length of {max_combined_length}"
            )

        # Determine modeled residues.
        complex_feats[dpc.modeled_idx] = determine_modeled_residues(complex_feats)
        if len(complex_feats[dpc.modeled_idx]) == 0:
            chain_desc = (
                f"chains {','.join(chain_ids)}" if chain_ids is not None else ""
            )
            raise DataError(f"No modeled residues found in {pdb_name} {chain_desc}")

        # Generate metadata file
        metadata: MetadataCSVRow = {}
        if generate_metadata:
            metadata[mc.pdb_name] = pdb_name
            metadata[mc.raw_path] = pdb_file_path

            # structure metadata
            metadata[mc.resolution] = structure.header.get("resolution")
            metadata[mc.date] = extract_structure_date(structure=structure)
            structure_method = StructureExperimentalMethod.from_structure(
                structure=structure
            )
            metadata[mc.structure_method] = structure_method

            # Track total sequence length
            metadata[mc.seq_len] = len(complex_feats[dpc.aatype])

            # Track modeled sequence lengths
            modeled_idx = complex_feats[dpc.modeled_idx]
            metadata[mc.moduled_num_res] = len(modeled_idx)
            # complex trimming, just determine start and ends
            complex_keep_mask = determine_modeled_residues_mask(
                chain_feats=complex_feats,
                trim_chains_independently=False,
            )
            metadata[mc.modeled_seq_len] = np.sum(complex_keep_mask)
            # chain independent trimming, trim each chain independently
            independent_keep_mask = determine_modeled_residues_mask(
                chain_feats=complex_feats,
                trim_chains_independently=True,
            )
            metadata[mc.modeled_indep_seq_len] = np.sum(independent_keep_mask)

            # Track quaternary / oligomeric information
            metadata[mc.num_chains] = len(struct_feats)
            metadata[mc.num_all_chains] = len(set(c.id for c in structure.get_chains()))
            metadata[mc.oligomeric_count] = _oligomeric_count(struct_feats)
            metadata[mc.oligomeric_detail] = _oligomeric_detail(struct_feats)
            uniq_seqs = set(
                [tuple(chain_feats[dpc.aatype]) for chain_feats in struct_feats]
            )
            metadata[mc.num_unique_seqs] = len(uniq_seqs)
            if len(uniq_seqs) == 1:
                metadata[mc.quaternary_category] = "homomer"
            else:
                metadata[mc.quaternary_category] = "heteromer"

            # chain lengths
            metadata[mc.chain_lengths] = _chain_lengths(
                struct_feats, modeled_only=False
            )
            metadata[mc.chain_lengths_modeled] = _chain_lengths(
                struct_feats, modeled_only=True
            )

            # pLDDT only present in synthetic structures.
            # Experimental ones have B-factors. We just set those to 100.0.
            if StructureExperimentalMethod.is_experimental(structure_method):
                plddts = np.full_like(complex_feats[dpc.b_factors], 100.0)
            else:
                plddts = complex_feats[dpc.b_factors]
            metadata[mc.mean_plddt_all_atom] = float(np.mean(plddts))
            metadata[mc.mean_plddt_modeled_bb] = float(
                np.mean(plddts[:, :3][modeled_idx])
            )

            # secondary structure, radius of gyration
            try:
                # MDtraj
                traj = md.load(uncompressed_pdb_path)
                # secondary structure calculation
                pdb_ss = md.compute_dssp(traj, simplified=True)
                # radius of gyration calculation
                pdb_rg = md.compute_rg(traj)
            except Exception as e:
                raise DataError(f"Mdtraj failed with error {e}")

            # track number of frames if trajectory
            metadata[mc.num_frames] = traj.n_frames

            # Secondary structure stats
            # Uses only the first frame `[0]` if there are multiple (since it used for the structure).
            secondary_structure = pdb_ss[0]

            # If no chains were dropped, shapes should match: limit to modeled residues.
            # This is unlike MultiFlow, which only considers over modeled residues.
            if secondary_structure.shape[0] == complex_feats[dpc.bb_mask].shape[0]:
                secondary_structure = secondary_structure[
                    complex_feats[dpc.bb_mask].astype(bool)
                ]
                secondary_structure_denom = np.sum(complex_feats[dpc.bb_mask])
            # If chains were dropped (no residues), skip the mask. More likely to drop due to high coil percent.
            else:
                secondary_structure_denom = secondary_structure.shape[0]

            metadata[mc.coil_percent] = (
                np.sum(secondary_structure == "C") / secondary_structure_denom
            )
            metadata[mc.helix_percent] = (
                np.sum(secondary_structure == "H") / secondary_structure_denom
            )
            metadata[mc.strand_percent] = (
                np.sum(secondary_structure == "E") / secondary_structure_denom
            )
            metadata[mc.radius_gyration] = pdb_rg[0]

            # check for interactions across chain pairs
            interactions = MultimerInteractions.from_chain_feats(
                complex_feats=complex_feats,
                metadata=metadata,
            )
            interactions.update_metadata(metadata)

            # non-residue chains and interactions
            non_residue_interactions = NonResidueInteractions.from_chain_feats(
                complex_feats=complex_feats,
                structure=structure,
            )
            non_residue_interactions.update_metadata(metadata)

        # Center the final complex
        complex_feats = center_and_scale_chain_feats(
            complex_feats, scale_factor=scale_factor
        )

        # Save pkl after processing is successful
        if write_dir is not None:
            processed_path = os.path.join(write_dir, f"{pdb_name}.pkl.gz")  # compressed
            metadata[mc.processed_path] = os.path.abspath(processed_path)
            write_pkl(processed_path, complex_feats)
        else:
            metadata[mc.processed_path] = ""

        # If we created a temp file, remove it
        if is_temp_file:
            os.remove(uncompressed_pdb_path)

        return metadata, complex_feats

    except Exception as e:
        # If we created a temp file, remove it
        if is_temp_file:
            os.remove(uncompressed_pdb_path)

        # re-raise original exception
        raise e


def process_pdb_file(
    pdb_file_path: str,
    pdb_name: str,
    write_dir: Optional[str] = None,
    chain_ids: Optional[List[str]] = None,
    chain_id: Optional[str] = None,  # backwards compatibility
    scale_factor: float = 1.0,
    max_combined_length: int = 8192,
) -> ProcessedFile:
    """
    Process PDB file into concatenated chain features `ProcessedFile`.

    If `write_dir` is provided, saves ProcessedFile pickle to `{write_dir}/{pdb_name}.pkl`.
    If `chain_ids` is provided, only those chains are processed.
    If `chain_id` is provided, only that chain is processed (backwards compatibility).
    `pdb_name` if provided is used for metadata and written file name.

    Raises DataError if a known filtering rule is hit. All other errors propogated.
    """
    # Handle backwards compatibility
    if chain_id is not None and chain_ids is not None:
        raise ValueError("Cannot specify both chain_id and chain_ids")
    if chain_id is not None:
        chain_ids = [chain_id]

    _, processed_file = _process_pdb(
        pdb_file_path=pdb_file_path,
        chain_ids=chain_ids,
        pdb_name=pdb_name,
        scale_factor=scale_factor,
        write_dir=write_dir,
        generate_metadata=False,
        max_combined_length=max_combined_length,
    )
    return processed_file


def process_pdb_with_metadata(
    pdb_file_path: str,
    write_dir: Optional[str] = None,
    chain_ids: Optional[List[str]] = None,
    chain_id: Optional[str] = None,  # backwards compatibility
    pdb_name: Optional[str] = None,
    scale_factor: float = 1.0,
    max_combined_length: int = 8192,
) -> Tuple[MetadataCSVRow, ProcessedFile]:
    """
    Process PDB file into concatenated chain features `ProcessedFile` and metadata `MetadataCSVRow`.

    If `write_dir` is provided, saves ProcessedFile pickle to `{write_dir}/{pdb_name}.pkl`.
    If `chain_ids` is provided, only those chains are processed.
    If `chain_id` is provided, only that chain is processed (backwards compatibility).
    `pdb_name` if provided is used for metadata and written file name.

    Raises DataError if a known filtering rule is hit. All other errors propogated.
    """
    # Handle backwards compatibility
    if chain_id is not None and chain_ids is not None:
        raise ValueError("Cannot specify both chain_id and chain_ids")
    if chain_id is not None:
        chain_ids = [chain_id]

    return _process_pdb(
        pdb_file_path=pdb_file_path,
        chain_ids=chain_ids,
        pdb_name=pdb_name,
        scale_factor=scale_factor,
        write_dir=write_dir,
        generate_metadata=True,
        max_combined_length=max_combined_length,
    )


def read_processed_file(
    processed_file_path: str,
    center: bool = True,
    trim_to_modeled_residues: bool = True,
    trim_chains_independently: bool = True,
) -> ProcessedFile:
    """
    Loads a processed PDB pkl from `process_pdb_file` and yields a ProcessedFile.
    Automatically handles both compressed (.pkl.gz) and uncompressed (.pkl) files.
    """
    processed_feats = read_pkl(processed_file_path)
    processed_file = process_chain_feats(
        processed_feats,
        center=center,
        trim_to_modeled_residues=trim_to_modeled_residues,
        trim_chains_independently=trim_chains_independently,
    )
    return processed_file
