from typing import Dict, Union

import torch

from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.type.str_enum import StrEnum


class BatchProps(StrEnum):
    """
    Dataloader batches cannot be structs, so we define an enum of batch properties.

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
    chain_idx = "chain_idx"  # re-indexed chain index (chains are shuffled)
    res_idx = "res_idx"  # re-indexed residue index (residues are re-numbered contiguously 1-indexed)
    res_plddt = "res_plddt"  # aka b-factors
    # defined / computed properties
    diffuse_mask = (
        "diffuse_mask"  # hallucination mask, residue positions that are noised
    )
    plddt_mask = "plddt_mask"  # mask, residue positions above pLDDT threshold
    # metadata
    pdb_name = "pdb_name"  # source PDB id
    csv_idx = "csv_idx"  # index of the protein in the csv file, for debugging
    # inference only
    num_res = (
        "num_res"  # number of residues in the protein  # TODO drop, just use res_mask
    )
    sample_id = "sample_id"  # inference sample id


class NoisyBatchProps(StrEnum):
    """
    Properties of a noised batch

    Properties at time `t` (i.e. `"_t"`) are data with noise added, scaled by `t`. t=0 is pure noise.
    """

    cat_t = "cat_t"  # (B, N, 1) tensor, t for amino acids (categoricals)
    so3_t = "so3_t"  # (B, N, 1) tensor, t for SO3 (rotations)
    r3_t = "r3_t"  # (B, N, 1) tensor, t for R3 (translations)
    trans_t = "trans_t"  # (B, N, 3) tensor, translations @ t
    rotmats_t = "rotmats_t"  # (B, N, 3, 3) tensor, rotations @ t
    aatypes_t = "aatypes_t"  # (B, N) tensor, predicted amino acids @ t
    trans_sc = "trans_sc"  # (B, N, 3) tensor, self-conditioned pred translations @ t
    aatypes_sc = "aatypes_sc"  # (B, N, 21) tensor, self-conditioned pred sequence @ t, including mask token


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


"""TensorFeat is a feature in primitive or pytorch"""
TensorFeat = Union[torch.Tensor, str, int]

"""BatchFeatures is a batch, features after featurizing ProcessedFile"""
BatchFeatures = Dict[BatchProps, TensorFeat]

"""NoisyFeatures is an item of a batch corrupted by Interpolant"""
NoisyFeatures = Dict[Union[BatchProps, NoisyBatchProps], TensorFeat]

"""InferenceFeatures is an inference (validation / prediction) batch"""
InferenceFeatures = Dict[BatchProps, TensorFeat]

"""ModelPrediction is the output of the model, i.e. the predicted features"""
ModelPrediction = Dict[PredBatchProps, TensorFeat]


def empty_feats(N: int) -> BatchFeatures:
    """
    Create empty features for a protein of length N.
    """
    return {
        BatchProps.res_mask: torch.ones(N),
        BatchProps.aatypes_1: torch.ones(N) * MASK_TOKEN_INDEX,
        BatchProps.trans_1: torch.zeros(N, 3),
        BatchProps.rotmats_1: torch.eye(3).repeat(N, 1, 1),
        BatchProps.torsion_angles_sin_cos_1: torch.zeros(N, 7, 2),
        BatchProps.chain_idx: torch.ones(N),
        BatchProps.res_idx: torch.arange(N),
        BatchProps.res_plddt: torch.zeros(N),
        BatchProps.diffuse_mask: torch.ones(N),
        BatchProps.plddt_mask: torch.ones(N),
        BatchProps.pdb_name: "",
        BatchProps.csv_idx: torch.tensor([1], dtype=torch.long),
    }
