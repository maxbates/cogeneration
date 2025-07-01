from typing import Dict, Union

import pandas as pd
from numpy import typing as npt

from cogeneration.type.str_enum import StrEnum


class MetadataColumn(StrEnum):
    """
    Columns in the pre-processed PDB metadata CSVs

    Note that any column addition may require reprocessing or updating the existing metadata CSV.
    """

    pdb_name = "pdb_name"
    # (optional) original PDB file, may be compressed
    raw_path = "raw_path"
    # pkl file for processed structure/sequence
    processed_path = "processed_path"
    quaternary_category = "quaternary_category"
    # num non-unique sequences
    oligomeric_count = "oligomeric_count"
    # per-unique seq details
    oligomeric_detail = "oligomeric_detail"
    # (new) number of unique sequences
    num_unique_seqs = "num_unique_seqs"
    # (new) residue chain lengths, "<chain_id>:<num_res>,..." format
    chain_lengths = "chain_lengths"
    # (new) residue chain lengths, "<chain_id>:<num_modeled_residues>,..." format
    chain_lengths_modeled = "chain_lengths_modeled"
    # number of chains in protein (ignores non-residue chains)
    num_chains = "num_chains"
    # (new) number of all chains (includes non-residue chains)
    num_all_chains = "num_all_chains"
    # total number of residues
    seq_len = "seq_len"
    # (new) count of num residues in modeled structure, not including gaps
    moduled_num_res = "moduled_num_res"
    # modeled residues, whole complex trimming (i.e. max - min res number)
    modeled_seq_len = "modeled_seq_len"
    # (new) modeled residues, chains independently trimmed
    modeled_indep_seq_len = "modeled_indep_seq_len"
    # (new) mean pLDDT of all atoms, if synthetic else 100
    mean_plddt_all_atom = "mean_plddt_all_atom"
    # (new) mean pLDDT of modeled backbone atoms, if synthetic else 100
    mean_plddt_modeled_bb = "mean_plddt_modeled_bb"

    # trajectory metadata
    num_frames = "num_frames"

    # secondary structure stats
    coil_percent = "coil_percent"
    helix_percent = "helix_percent"
    strand_percent = "strand_percent"
    radius_gyration = "radius_gyration"

    # structure metadata if provided
    date = "date"  # date of structure deposition
    resolution = "resolution"
    structure_method = "structure_method"  # StructureExperimentalMethod enum

    # (new, multimer-only) interactions / clashes
    # chain-chain interactions: "<chain_a>:<chain_b>:<num_bb_res_xing_a>:<num_bb_res_xing_b>,...."
    chain_interactions = "chain_interactions"
    # backbone interactions across chains
    num_backbone_interactions = "num_backbone_interactions"
    num_backbone_res_interacting = "num_backbone_res_interacting"
    # atomic interactions across chains
    # num_atom_interactions = "num_atom_interactions"
    # serialized all chain clashes: "<chain_a>:<chain_b>:<num_bb_clashes>,..."
    chain_clashes = "chain_clashes"
    # clashes across chains exceeding clash threshold
    num_chains_clashing = "num_chains_clashing"

    # (new, multimer-only) Interaction hot spots
    # serialized hot spot residues: "<chain_id>:<res_index>:<num_interactions>,..."
    hot_spots = "hot_spots"

    # (new) Non-residue chains + interactions
    # [Can only be added on initial processing, because processed feats omit such chains.]
    # count of non-residue or empty chains (atoms, metals, molecules, DNA)
    num_non_residue_chains = "num_non_residue_chains"
    num_single_atom_chains = "num_single_atom_chains"
    num_solution_molecules = "num_solution_molecules"
    # interactions require the chain in proximity to 3+ residues
    num_metal_interactions = "num_metal_interactions"
    num_macromolecule_interactions = "num_macromolecule_interactions"
    # count of protein chain-chain interactions potentially mediated by non-residue chains
    num_mediated_interactions = "num_mediated_interactions"


class RedesignColumn(StrEnum):
    """Columns in the redesign metadata CSVs"""

    example = "example"  # pdb_name
    wildtype_seq = "wildtype_seq"  # original sequence
    wildtype_rmsd = "wildtype_rmsd"  # RMSD of original sequence to reference structure
    best_seq = "best_seq"  # best 1 redesign per structure
    best_rmsd = "best_rmsd"  # RMSD of best redesign to reference structure


class DatasetColumn(StrEnum):
    """Columns added by BaseDataset"""

    # cluster metadata (added by loading clusters csv)
    cluster = "cluster"

    # added during load
    index = "index"


class DatasetProteinColumn(StrEnum):
    """
    Information about the protein, pickled in `processed_path`, or from parsing a Protein / Chain.
    Most of these values are defined by `process_chain` and `parse_chain_feats`.
    A few (like `bb_mask` and `bb_positions` are added during processing).
    Only protein chains are saved in the ProcessedFile.

    The saved proteins contain all molecules / residues, and is of length P.
    `modeled_idx` is length N, and defines which residues are modeled.
    It is used to select windows of the structure, then dropped in loading i.e. `read_processed_file()`.
    Also note that during loading, the structure is centered and re-indexed, so many of these values change.
    """

    aatype = "aatype"  # (P, ) AA sequence residue indices
    atom_mask = "atom_mask"  # (P, 37) all atoms to consider
    bb_mask = "bb_mask"  # (P, ) alpha carbons considered
    atom_positions = "atom_positions"  # (P, 37, 3) all atom positions
    bb_positions = "bb_positions"  # (P, 3) alpha carbon
    residue_index = "residue_index"  # (P, ) residue index
    chain_index = "chain_index"  # (P, ) chain index
    b_factors = "b_factors"  # (P, 37) b factors
    modeled_idx = "modeled_idx"  # (N, ) index of modeled residues


class DatasetTransformColumn(StrEnum):
    """
    Columns generated by OpenFold data transform
    """

    # atom37_to_frames()
    rigidgroups_gt_frames = "rigidgroups_gt_frames"
    rigidgroups_gt_exists = "rigidgroups_gt_exists"
    rigidgroups_group_exists = "rigidgroups_group_exists"
    rigidgroups_group_is_ambiguous = "rigidgroups_group_is_ambiguous"
    rigidgroups_alt_gt_frames = "rigidgroups_alt_gt_frames"
    # atom37_to_torsion_angles()
    torsion_angles_sin_cos = "torsion_angles_sin_cos"


"""NumpyPrimitiveFeat is a feature in primitive or numpy"""
NumpyPrimitiveFeat = Union[npt.NDArray, str, int, float]

"""MetadataCSVRow type alias for a single row of the metadata CSV file"""
MetadataCSVRow = Dict[MetadataColumn, NumpyPrimitiveFeat]

"""MetadataDataFrame type alias for the metadata CSV file, composed of MetadataCSVRow"""
MetadataDataFrame = pd.DataFrame

"""DatasetCSVRow is row of MetadatDataFrame augmented with clusters, redesigned data, etc."""
DatasetCSVRow = Dict[
    Union[MetadataColumn, RedesignColumn, DatasetColumn], NumpyPrimitiveFeat
]

"""DatasetDataFrame is the full dataset, composed of DatasetCSVRow"""
DatasetDataFrame = pd.DataFrame

"""
ChainFeatures represents an intermediate state converting `asdict(Protein)` to `ProcessedFile`.
Use when only a subset of fields of `DatasetProteinColumns` are present in `ChainFeatures`.
"""
ChainFeatures = Dict[DatasetProteinColumn, npt.NDArray]

"""
ProcessedFile for pre-processed pkl, produced by `parse_pdb_files.py`
It contains 1+ chains, concatenated into a flat `chain_features`.
"""
ProcessedFile = Dict[DatasetProteinColumn, npt.NDArray]


"""
Note the descriptions vary more widely in the metadata provided by MultiFlow,
perhaps taken from PDB directly, but this is fairly close.
"""
OLIGOMERIC_PREFIXES = {
    1: "mono",
    2: "di",
    3: "tri",
    4: "tetra",
    5: "penta",
    6: "hexa",
    7: "hepta",
    8: "octa",
    9: "nona",
    10: "deca",
    11: "undeca",
    12: "dodeca",  # pref to `duodeca`
    13: "trideca",
    14: "tetradeca",
    15: "pentadeca",
    16: "hexadeca",
    17: "heptadeca",
    18: "octadeca",
    19: "nonadeca",
    20: "eicosa",  # pref to `icosa`
}
