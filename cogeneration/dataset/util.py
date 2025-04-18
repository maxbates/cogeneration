from typing import Any, Dict, Union

import torch

from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.const import MASK_TOKEN_INDEX


def empty_feats(N: int) -> Dict[bp, Any]:
    """
    Create empty features for a protein of length N.
    """
    return {
        bp.res_mask: torch.ones(N),
        bp.aatypes_1: torch.ones(N) * MASK_TOKEN_INDEX,
        bp.trans_1: torch.zeros(N, 3),
        bp.rotmats_1: torch.eye(3).repeat(N, 1, 1),
        bp.torsion_angles_sin_cos_1: torch.zeros(N, 7, 2),
        bp.chain_idx: torch.ones(N),
        bp.res_idx: torch.arange(N),
        bp.res_plddt: torch.zeros(N),
        bp.diffuse_mask: torch.ones(N),
        bp.plddt_mask: torch.ones(N),
        bp.pdb_name: "empty",
        bp.csv_idx: torch.tensor([1], dtype=torch.long),
    }


def mock_noisy_feats(N: int, idx: int) -> Dict[Union[bp, nbp], Any]:
    """
    Create random + corrupted features for a protein of length N
    """
    feats = {}

    # N residue protein, random frames
    feats[bp.res_mask] = torch.ones(N)
    feats[bp.aatypes_1] = torch.randint(0, 20, (N,))
    feats[bp.trans_1] = torch.rand(N, 3)
    feats[bp.rotmats_1] = torch.rand(N, 3, 3)
    feats[bp.torsion_angles_sin_cos_1] = torch.rand(N, 7, 2)
    feats[bp.chain_idx] = torch.zeros(N)
    feats[bp.res_idx] = torch.arange(N)
    feats[bp.pdb_name] = f"test_{idx}"
    feats[bp.res_plddt] = torch.floor(torch.rand(N) + 0.5)
    feats[bp.plddt_mask] = feats[bp.res_plddt] > 0.6
    feats[bp.diffuse_mask] = torch.ones(N)
    feats[bp.csv_idx] = torch.tensor([0])

    # inference feats  # TODO remove, dedicated mock function
    feats[bp.num_res] = torch.tensor([N])
    feats[bp.sample_id] = f"test_{idx}"

    # generate corrupted noisy values for input_feats
    t = torch.rand(1)  # use same value as with unconditional + not separate_t
    feats[nbp.so3_t] = t
    feats[nbp.r3_t] = t
    feats[nbp.cat_t] = t
    feats[nbp.trans_t] = torch.rand(N, 3)
    feats[nbp.rotmats_t] = torch.rand(N, 3, 3)
    feats[nbp.aatypes_t] = torch.rand(N) * 20  # AA seq as floats
    feats[nbp.trans_sc] = torch.rand(N, 3)
    feats[nbp.aatypes_sc] = torch.rand(N, 21)  # include mask token

    return feats
