from typing import Dict, Union

import torch

from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.type.str_enum import StrEnum
from cogeneration.type.task import DataTask


class BatchProp(StrEnum):
    """
    Enum of batch properties.
    "_1" properties @ time == 1 the original data.
    """

    # ground truth properties
    aatypes_1 = "aatypes_1"  # (B, N) amino acid sequence, as ints (0-20, 20 for UNK)
    trans_1 = "trans_1"  # (B, N, 3) frame translations
    rotmats_1 = "rotmats_1"  # (B, N, 3, 3) frame rotations
    torsion_angles_sin_cos_1 = "torsion_angles_sin_cos_1"  # (B, N, 7, 2)
    # structure metadata
    chain_idx = "chain_idx"  # (B, N) chain index (chains are shuffled, 1-indexed)
    res_idx = "res_idx"  # (B, N) residue index (residues are re-numbered contiguously 1-indexed)
    res_plddt = "res_plddt"  # (B, N) aka b-factors
    # masks
    res_mask = "res_mask"  # (B, N) residues under consideration (uses `dpc.bb_mask`)
    diffuse_mask = "diffuse_mask"  # (B, N) hallucination mask, residue positions that are corrupted/sampled
    motif_mask = "motif_mask"  # (B, N) [inpainting only] mask for fixed motif residue
    plddt_mask = (
        "plddt_mask"  # (B, N) pLDDT mask, residue positions above pLDDT threshold
    )
    # metadata
    pdb_name = "pdb_name"  # (B) source PDB id string/int
    csv_idx = "csv_idx"  # (B) index of the protein in the csv file, for debugging
    # inference only
    sample_id = "sample_id"  # (B) [inference only] sample id


# datum level metadata, i.e. `(B)` rather than `(B, N)`
METADATA_BATCH_PROPS = [
    BatchProp.pdb_name,
    BatchProp.csv_idx,
    BatchProp.sample_id,
]


class NoisyBatchProp(StrEnum):
    """
    Corrupted batch property enum.
    "_t" properties @ time == t are data corrupted to time `t`, where t=0 is noise.
    """

    cat_t = "cat_t"  # (B, 1) tensor, t for amino acids (categoricals)
    so3_t = "so3_t"  # (B, 1) tensor, t for SO3 (rotations)
    r3_t = "r3_t"  # (B, 1) tensor, t for R3 (translations)
    trans_t = "trans_t"  # (B, N, 3) tensor, translations @ t
    rotmats_t = "rotmats_t"  # (B, N, 3, 3) tensor, rotations @ t
    aatypes_t = "aatypes_t"  # (B, N) tensor, predicted amino acids @ t as ints (0-20)
    trans_sc = "trans_sc"  # (B, N, 3) tensor, self-conditioned pred translations @ t
    aatypes_sc = "aatypes_sc"  # (B, N, 21) tensor, self-conditioned pred sequence @ t, including mask token


class PredBatchProp(StrEnum):
    """
    Model output enum
    """

    pred_trans = "pred_trans"  # (B, N, 3)
    pred_rotmats = "pred_rotmats"  # (B, N, 3, 3)
    pred_torsions = "pred_torsions"  # Optional (B, N, K, 2), K=1 (psi) or K=7 (all)
    pred_logits = "pred_logits"  # (B, N, S) where S=21 if masking else S=20
    pred_aatypes = "pred_aatypes"  # (B, N)

    # other model outputs
    node_embed = "node_embed"  # (B, N, c_s)
    edge_embed = "edge_embed"  # (B, N, N, c_p)


"""TensorFeat is a feature in primitive or pytorch"""
TensorFeat = Union[torch.Tensor, str, int]

"""BatchFeatures is a batch, features after featurizing ProcessedFile"""
BatchFeatures = Dict[BatchProp, TensorFeat]

"""NoisyFeatures is an item of a batch corrupted by Interpolant"""
NoisyFeatures = Dict[Union[BatchProp, NoisyBatchProp], TensorFeat]

"""InferenceFeatures is an inference (validation / prediction) batch"""
InferenceFeatures = Dict[BatchProp, TensorFeat]

"""ModelPrediction is the output of the model, i.e. the predicted features"""
ModelPrediction = Dict[PredBatchProp, TensorFeat]


def empty_feats(N: int, task: DataTask = DataTask.hallucination) -> BatchFeatures:
    """
    Create empty features for a protein of length N.
    """
    feats = {
        BatchProp.res_mask: torch.ones(N),
        # assume masking interpolant
        BatchProp.aatypes_1: (torch.ones(N) * MASK_TOKEN_INDEX).long(),
        BatchProp.trans_1: torch.zeros(N, 3),
        BatchProp.rotmats_1: torch.eye(3).repeat(N, 1, 1),
        BatchProp.torsion_angles_sin_cos_1: torch.stack(
            [torch.zeros(N, 7), torch.ones(N, 7)], dim=-1  # sin(0) = 0  # cos(0) = 1
        ),
        BatchProp.chain_idx: torch.ones(N),
        BatchProp.res_idx: torch.arange(1, N + 1),
        BatchProp.res_plddt: torch.full((N,), 100.0),
        BatchProp.diffuse_mask: torch.ones(N),
        BatchProp.plddt_mask: torch.ones(N),
        BatchProp.pdb_name: "",
        BatchProp.csv_idx: torch.tensor([1], dtype=torch.long),
        BatchProp.sample_id: 0,  # inference only but no impact to training / model
    }

    if task == DataTask.inpainting:
        feats[BatchProp.motif_mask] = torch.zeros(N)

    return feats
