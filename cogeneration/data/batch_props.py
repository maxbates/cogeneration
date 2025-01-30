from enum import Enum


class BatchProps(str, Enum):
    """
    Dataloader batches cannot be made type-safe, so we define an enum of batch properties.

    Properties at time `1` (i.e. `"_1"`) are the ground truth, i.e. the original data.
    """

    res_mask = "res_mask"  # backbone positions in the original structure
    # ground truth properties
    aatypes_1 = "aatypes_1"  # amino acid sequence
    trans_1 = "trans_1"  # frame translations
    rotmats_1 = "rotmats_1"  # frame rotations
    # structure metadata
    num_res = "num_res"  # number of residues in the protein. Important for inference.
    chain_idx = "chain_idx"  # re-indexed chain index (chains are shuffled)
    res_idx = "res_idx"  # re-indexed residue index (residues are re-numbered contiguously 1-indexed)
    res_plddt = "res_plddt"  # aka b-factors
    # defined / computed properties
    diffuse_mask = (
        "diffuse_mask"  # hallucination mask, residue positions that are noised
    )
    plddt_mask = "plddt_mask"  # mask, residue positions above pLDDT threshold
    csv_idx = "csv_idx"  # index of the protein in the csv file, for debugging
    sample_id = "sample_id"
    pdb_name = "pdb_name"

    def __str__(self):
        return self.value


class NoisyBatchProps(str, Enum):
    """
    Properties of a noised batch

    Properties at time `t` (i.e. `"_t"`) are data with noise added, scaled by `t`. t=0 is pure noise.
    """

    cat_t = "cat_t"  # t for amino acids (categoricals)
    so3_t = "so3_t"  # t for SO3 (rotations)
    r3_t = "r3_t"  # t for R3 (translations)
    trans_t = "trans_t"  # (N, 3) tensor, translations @ t
    rotmats_t = "rotmats_t"  # (N, 3, 3) tensor, rotations @ t
    aatypes_t = "aatypes_t"  # (N) tensor, predicted amino acids @ t
    trans_sc = "trans_sc"  # (N, 3) tensor, self-conditioned pred translations @ t
    aatypes_sc = "aatypes_sc"  # (N, 21) tensor, self-conditioned pred sequence @ t, including mask token

    def __str__(self):
        return self.value


class PredBatchProps(str, Enum):
    """
    Properties of a predicted batch
    """

    pred_trans = "pred_trans"
    pred_rotmats = "pred_rotmats"
    pred_logits = "pred_logits"
    pred_aatypes = "pred_aatypes"

    def __str__(self):
        return self.value
