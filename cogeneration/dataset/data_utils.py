import numpy as np

from cogeneration.data.const import CA_IDX
from cogeneration.data.csv import DatasetProteinColumns as dpc


def parse_chain_feats(chain_feats, scale_factor: float = 1.0, center: bool = True):
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
