from dataclasses import asdict

import numpy as np
from Bio import PDB

from cogeneration.data.const import CA_IDX
from cogeneration.data.protein import process_chain


def parse_chain_feats(chain_feats, scale_factor: float = 1.0, center: bool = True):
    chain_feats["bb_mask"] = chain_feats["atom_mask"][:, CA_IDX]
    bb_pos = chain_feats["atom_positions"][:, CA_IDX]
    if center:
        bb_center = np.sum(bb_pos, axis=0) / (np.sum(chain_feats["bb_mask"]) + 1e-5)
        centered_pos = chain_feats["atom_positions"] - bb_center[None, None, :]
        scaled_pos = centered_pos / scale_factor
    else:
        scaled_pos = chain_feats["atom_positions"] / scale_factor
    chain_feats["atom_positions"] = scaled_pos * chain_feats["atom_mask"][..., None]
    chain_feats["bb_positions"] = chain_feats["atom_positions"][:, CA_IDX]
    return chain_feats
