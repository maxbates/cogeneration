import dataclasses
from typing import Any, Dict

import numpy as np
from Bio import PDB

from cogeneration.data.const import CA_IDX
from cogeneration.data.enum import DatasetProteinColumns as dpc
from cogeneration.data.protein import process_chain


def parse_chain_feats(
    chain_feats: Dict[dpc, Any],
    scale_factor: float = 1.0,
    center: bool = True,
):
    """
    Parse loaded chain features. Add position information and mask for backbone atoms.

    Note that positions are expected in angstroms (PDB style).
    """
    chain_feats[dpc.bb_mask] = chain_feats[dpc.atom_mask][:, CA_IDX]
    bb_pos = chain_feats[dpc.atom_positions][:, CA_IDX]
    if center:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats[dpc.bb_mask]) + 1e-5)
        centered_pos = chain_feats[dpc.atom_positions] - bb_center[None, None, :]
        scaled_pos = centered_pos / scale_factor
    else:
        scaled_pos = chain_feats[dpc.atom_positions] / scale_factor
    chain_feats[dpc.atom_positions] = scaled_pos * chain_feats[dpc.atom_mask][..., None]
    chain_feats[dpc.bb_positions] = chain_feats[dpc.atom_positions][:, CA_IDX]
    return chain_feats


def parse_pdb_feats(
    pdb_name: str,
    pdb_path: str,
    scale_factor: float = 1.0,
    # TODO: Make the default behaviour read all chains. Need to update return behavior.
    chain_id: str = "A",
) -> Dict[dpc, Any]:
    """
    Args:
        pdb_name: name of PDB to parse.
        pdb_path: path to PDB file to read.
        scale_factor: factor to scale atom positions.
        mean_center: whether to mean center atom positions.
    Returns:
        Dict with CHAIN_FEATS features extracted from PDB with specified
        preprocessing.
    """
    parser = PDB.PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_name, pdb_path)
    struct_chains = {chain.id: chain for chain in structure.get_chains()}

    def _process_chain_id(x):
        chain_prot = process_chain(struct_chains[x], x)
        chain_dict = dataclasses.asdict(chain_prot)

        # Take specific features needed for chain parsing
        feat_dict = {
            x: chain_dict[x]
            for x in [
                dpc.aatype,
                dpc.atom_positions,
                dpc.atom_mask,
                dpc.residue_index,
                dpc.b_factors,
            ]
        }
        return parse_chain_feats(feat_dict, scale_factor=scale_factor)

    if isinstance(chain_id, str):
        return _process_chain_id(chain_id)
    elif isinstance(chain_id, list):
        return {x: _process_chain_id(x) for x in chain_id}
    elif chain_id is None:
        return {x: _process_chain_id(x) for x in struct_chains}
    else:
        raise ValueError(f"Unrecognized chain list {chain_id}")
