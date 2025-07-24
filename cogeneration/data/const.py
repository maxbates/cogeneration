import string

import torch

from cogeneration.data import residue_constants

# Global map from chain characters to integers.
ALPHANUMERIC = string.ascii_letters + string.digits + " "
CHAIN_TO_INT = {chain_char: i for i, chain_char in enumerate(ALPHANUMERIC)}
INT_TO_CHAIN = {i: chain_char for i, chain_char in enumerate(ALPHANUMERIC)}

# Size scales
NM_TO_ANG_SCALE = 10.0
ANG_TO_NM_SCALE = 1 / NM_TO_ANG_SCALE


def rigids_ang_to_nm(x: torch.tensor):
    return x.apply_trans_fn(lambda x: x * ANG_TO_NM_SCALE)


def rigids_nm_to_ang(x: torch.tensor):
    return x.apply_trans_fn(lambda x: x * NM_TO_ANG_SCALE)


# Complete sequence of chain IDs supported by the PDB format.
PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)  # := 62.

NUM_TOKENS = residue_constants.restype_num  # := 20.
MASK_TOKEN_INDEX = residue_constants.restypes_with_x.index("X")  # := 20

CA_IDX = residue_constants.atom_order["CA"]


def aatype_to_seq(aatype, chain_idx=None):
    if chain_idx is None:
        return "".join([residue_constants.restypes_with_x[x] for x in aatype])

    # Otherwise, include chain breaks where `chain_idx` changes
    seq = ""
    last_chain_id = None
    for aa, chain_id in zip(aatype, chain_idx):
        if chain_id != last_chain_id and last_chain_id is not None:
            seq += CHAIN_BREAK_STR
        seq += residue_constants.restypes_with_x[aa]
        last_chain_id = chain_id

    return seq


def seq_to_aatype(seq):
    return [
        residue_constants.restypes_with_x.index(x) for x in seq if x != CHAIN_BREAK_STR
    ]


CHAIN_BREAK_STR = ":"
