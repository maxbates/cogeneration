from cogeneration.util.base_classes import StrEnum


class BatchProps(StrEnum):
    """
    Dataloader batches cannot be made type-safe, so we define an enum of batch properties.

    Properties at time `1` (i.e. `"_1"`) are the ground truth, i.e. the original data.
    """

    res_mask = "res_mask"  # backbone positions in the original structure
    # ground truth properties
    aatypes_1 = "aatypes_1"  # amino acid sequence
    trans_1 = "trans_1"  # frame translations
    rotmats_1 = "rotmats_1"  # frame rotations
    # torsion angles: only predict 1 in model to guide frames; take `[..., 2, :]` from ground truth.
    torsion_angles_sin_cos_1 = "torsion_angles_sin_cos_1"
    # structure metadata
    num_res = "num_res"  # number of residues in the protein. Defined for unconditional batches
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


class NoisyBatchProps(StrEnum):
    """
    Properties of a noised batch

    Properties at time `t` (i.e. `"_t"`) are data with noise added, scaled by `t`. t=0 is pure noise.
    """

    cat_t = "cat_t"  # (bs, N, 1) tensor, t for amino acids (categoricals)
    so3_t = "so3_t"  # (bs, N, 1) tensor, t for SO3 (rotations)
    r3_t = "r3_t"  # (bs, N, 1) tensor, t for R3 (translations)
    trans_t = "trans_t"  # (bs, N, 3) tensor, translations @ t
    rotmats_t = "rotmats_t"  # (bs, N, 3, 3) tensor, rotations @ t
    aatypes_t = "aatypes_t"  # (bs, N) tensor, predicted amino acids @ t
    trans_sc = "trans_sc"  # (bs, N, 3) tensor, self-conditioned pred translations @ t
    aatypes_sc = "aatypes_sc"  # (bs, N, 21) tensor, self-conditioned pred sequence @ t, including mask token

    def __str__(self):
        return self.value


class PredBatchProps(StrEnum):
    """
    Properties of a predicted batch
    """

    pred_trans = "pred_trans"
    pred_rotmats = "pred_rotmats"
    pred_psi = "pred_psi"  # optionally output
    pred_logits = "pred_logits"
    pred_aatypes = "pred_aatypes"

    # other model outputs
    node_embed = "node_embed"
    edge_embed = "edge_embed"

    def __str__(self):
        return self.value
