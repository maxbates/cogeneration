import datetime
import itertools
import os
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cogeneration.config.dict_utils import (
    deep_merge_dicts,
    flatten_dict,
    prune_unknown_dataclass_fields,
)
from cogeneration.type.dataset import OLIGOMERIC_PREFIXES, MetadataColumn
from cogeneration.type.embed import PositionalEmbeddingMethod
from cogeneration.type.str_enum import StrEnum
from cogeneration.type.task import DataTask, InferenceTask

"""
Structured configurations for cogeneration.

# Public Multiflow
Several parameters from Multiflow appear to be unused in the public code.
Many are marked with `# DROP`.

# extensions / resolvers 
See __init__.py for extensions to OmegaConf, e.g. `ternary`.

# Enums 
hydra does not currently support `Literal` and suggests using Enums instead.
https://github.com/omry/omegaconf/issues/422
"""


# TODO - support default configs for `forward_folding` and `inverse folding`
#   There are a handful of values across the config that need to have other defaults.
#   i.e. instead of the current `ternary` paradigm
#   to better differentate the training vs inference interpolant

PATH_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
PATH_PUBLIC_WEIGHTS = PATH_PROJECT_ROOT / "multiflow_weights"

GenericConfig = TypeVar("GenericConfig", bound="BaseConfig")


@dataclass
class BaseClassConfig:
    """
    Base class for *Config objects. Can be nested.
    """

    def interpolate(self: GenericConfig) -> GenericConfig:
        """
        Interpolate Config with OmegaConf, yielding an interpolated Config class

        OmegaConf supports interpolated fields like `my_field: int = "${my_other_field}"`.
        This function interpolates those fields, yielding a new Config object.

        Intermediate fields are dataclasses, not DictConfig.
        """
        # use `to_object()` so intermediate fields are dataclasses, not DictConfig
        # `create()` interpolates the fields
        return OmegaConf.to_object(OmegaConf.create(self))

    def asdict(self, interpolate: bool = False) -> Dict[str, Any]:
        if interpolate:
            return asdict(self.interpolate())
        return asdict(self)

    def flatdict(self, interpolate: bool = False) -> Dict[str, Any]:
        """
        Flatten the dataclass into a dictionary.
        """
        # Use flatten_dict to flatten the dataclass
        return dict(flatten_dict(self.asdict(interpolate=interpolate)))

    def merge_dict(
        self: GenericConfig,
        other: Dict[str, Any],
        interpolate: bool = True,
    ) -> GenericConfig:
        """
        Merge a dictionary config with the current config, optionally interpolating `self` first.
        Returns a new config of the same type as self.
        """
        # merge configs as dictionary, interpolating self if requested
        merged_dict = deep_merge_dicts(self.asdict(interpolate=interpolate), other)

        # Rebuild a typed config of the same class as self
        schema = OmegaConf.structured(type(self))
        merged_conf = OmegaConf.merge(schema, OmegaConf.create(merged_dict))
        return OmegaConf.to_object(merged_conf)

    def merge(
        self: GenericConfig,
        other: GenericConfig,
        interpolate: bool = True,
    ) -> GenericConfig:
        """
        Standard overwriting merge of two configs: fields in `other` override those in `self`.
        Returns new instance of same class as `self`.
        """
        other_cfg = other.interpolate() if interpolate else other
        return self.merge_dict(
            other=other_cfg.asdict(),
        )


@dataclass
class SharedConfig(BaseClassConfig):
    """
    shared config, system "flags", metadata
    """

    # identifier
    id: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    # `local` to train locally on a Mac. use `mps` not `gpu` with `ddp`, etc.
    local: bool = True
    # randomness / stochastic paths
    stochastic: bool = True
    # project root path
    project_root: str = str(PATH_PROJECT_ROOT)
    # random number generator shared seed
    seed: int = 123


@dataclass
class ModelHyperParamsConfig(BaseClassConfig):
    """
    Shared hyperparameters for the model.
    Use a structured config for hyperparameters so easy to reference in templates.
    Default values match those from Multiflow. Use factory methods for different configurations.

    TODO register hydra config group to enable easy switching
    https://hydra.cc/docs/tutorials/structured_config/defaults/
    """

    node_embed_size: int = 256
    edge_embed_size: int = 128

    aa_num_tokens: int = 21  # number of amino acid types (if masking), 21 = mask

    pos_embed_size: int = 128
    pos_embed_method: PositionalEmbeddingMethod = PositionalEmbeddingMethod.rotary
    pos_embed_max_len: int = 2048  # 2056 in public multiflow

    timestep_embed_size: int = 128

    @classmethod
    def tiny(cls):
        """Factory for tiny configuration, e.g. for testing"""
        return cls(
            node_embed_size=4,
            edge_embed_size=4,
            pos_embed_size=4,
            timestep_embed_size=4,
        )

    @classmethod
    def public_multiflow(cls):
        """Factory for configuration compatible with public MultiFlow"""
        return cls(
            node_embed_size=256,
            edge_embed_size=128,
            pos_embed_size=128,
            timestep_embed_size=128,
        )

    @classmethod
    def esm2_150m(cls):
        """Factory for configuration compatible with ESM-2 150M"""
        return cls(
            node_embed_size=640,
            edge_embed_size=256,
            pos_embed_size=128,
            timestep_embed_size=128,
        )


@dataclass
class ModelNodeFeaturesConfig(BaseClassConfig):
    """
    Node features (residue, sequence)
    """

    # c_s: output node embedding width, and internal MLP dimension
    c_s: int = "${model.hyper_params.node_embed_size}"
    # c_pos_emb: position embedding size
    c_pos_emb: int = "${model.hyper_params.pos_embed_size}"
    # c_timestep_emb: timestep embedding size
    c_timestep_emb: int = "${model.hyper_params.timestep_embed_size}"
    # aatype_pred_num_tokens: number of amino acid types (21 = mask if masking)
    aatype_pred_num_tokens: int = "${model.hyper_params.aa_num_tokens}"
    # positional embedding parameters
    pos_embed_method: PositionalEmbeddingMethod = (
        "${model.hyper_params.pos_embed_method}"
    )
    pos_embed_max_len: int = "${model.hyper_params.pos_embed_max_len}"
    # embed_chain: whether to embed chain index.
    embed_chain: bool = True
    # embed_aatype: whether to embed amino acid type
    embed_aatype: bool = True
    # use_mlp: whether to use MLP for embedding, otherwise linear layer
    use_mlp: bool = True


@dataclass
class ModelEdgeFeaturesConfig(BaseClassConfig):
    """
    Edge features (structure, residue distances / orientations)
    """

    single_bias_transition_n: int = 2  # DROP
    # c_s: node embedding size (an input to edge network)
    c_s: int = "${model.hyper_params.node_embed_size}"
    # c_p: output embedding size + MLP width
    c_p: int = "${model.hyper_params.edge_embed_size}"
    # feat_dim: partial internal dimension (actual width is multiple)
    feat_dim: int = 64
    # num_bins: number of bins for edge distrogram
    num_bins: int = 22
    # positional embedding parameters
    pos_embed_method: PositionalEmbeddingMethod = (
        "${model.hyper_params.pos_embed_method}"
    )
    pos_embed_max_len: int = "${model.hyper_params.pos_embed_max_len}"
    # self_condition: used by interpolant config.
    self_condition: bool = True
    # embed_chain: whether to embed chain index.
    embed_chain: bool = True
    # embed_diffuse_mask: whether to embed diffuse mask. important esp. for `scaffolding` task.
    embed_diffuse_mask: bool = True


@dataclass
class ModelIPAConfig(BaseClassConfig):
    """
    Invariant Point Attention configuration.
    Keys match ported OpenFold's IPA module.
    """

    # IPA parameters
    # s channel. `s` is the single representation, i.e. node embedding
    c_s: int = "${model.hyper_params.node_embed_size}"
    # z channel. `z` is the pair representation, i.e. edge embedding
    c_z: int = "${model.hyper_params.edge_embed_size}"
    c_hidden: int = 16
    no_heads: int = 8
    no_qk_points: int = 8
    no_v_points: int = 12
    # Attention trunk parameters
    num_blocks: int = 8
    dropout: float = 0.0
    seq_tfmr_num_heads: int = 4
    seq_tfmr_num_layers: int = 4
    transformer_dropout: float = 0.2


@dataclass
class ModelAAPredConfig(BaseClassConfig):
    """
    Amino acid prediction configuration using simple linear layers.
    """

    # c_s: node embedding size (input)
    c_s: int = "${model.hyper_params.node_embed_size}"
    # aatype_pred_num_tokens: number of amino acid types => logits / rate-matrix shape
    aatype_pred_num_tokens: int = "${model.hyper_params.aa_num_tokens}"


@dataclass
class ModelSequenceIPANetConfig(BaseClassConfig):
    """
    IPA style transformer, predicting logits instead of backbone update.

    Public MultiFlow code uses minimal AAPred linear network.
    The public MultiFlow config alludes to a `sequence_net` not in code, with fields:

    'model:sequence_net:init_edge_embed',
    'model:sequence_net:init_node_embed',
    'model:sequence_net:ipa:c_hidden',
    'model:sequence_net:ipa:c_s',
    'model:sequence_net:ipa:c_z',
    'model:sequence_net:ipa:dropout',
    'model:sequence_net:ipa:no_heads',
    'model:sequence_net:ipa:no_qk_points',
    'model:sequence_net:ipa:no_v_points',
    'model:sequence_net:num_layers',
    'model:sequence_net:use_init_embed',
    'model:sequence_net:use_init_rigid',
    'model:sequence_net:use_local_attention',
    'model:use_sequence_net',
    """

    # aatype_pred_num_tokens: number of amino acid types => logits / rate-matrix shape
    aatype_pred_num_tokens: int = "${model.hyper_params.aa_num_tokens}"
    # c_s: internal embedding size for ReLU
    c_s: int = "${model.hyper_params.node_embed_size}"
    # add initial node + edge embeddings to post-IPA trunk embeddings
    # FoldFlow-2 claimed this was important to pass through time + positional embeddings to logit prediction
    use_init_embed: bool = True
    # IPA parameters
    # We use fewer blocks, because no backbone updates are performed
    ipa: ModelIPAConfig = field(
        default_factory=lambda: ModelIPAConfig(
            num_blocks=2,
        )
    )


class ModelSequencePredictionEnum(StrEnum):
    NOOP = "noop"  # simply emit aatypes
    aa_pred = "aa_pred"  # public MultiFlow. Simple MLP.
    sequence_ipa_net = "sequence_ipa_net"  # IPA transformer architecture


@dataclass
class ModelConfig(BaseClassConfig):
    hyper_params: ModelHyperParamsConfig = field(default_factory=ModelHyperParamsConfig)
    node_features: ModelNodeFeaturesConfig = field(
        default_factory=ModelNodeFeaturesConfig
    )
    edge_features: ModelEdgeFeaturesConfig = field(
        default_factory=ModelEdgeFeaturesConfig
    )
    ipa: ModelIPAConfig = field(default_factory=ModelIPAConfig)

    # predict torsion angles
    # note does not impact frames (trans/rots), just the rigid we construct (from trans/rot/psi)
    predict_psi_torsions: bool = True

    # sequence prediction, default is simple aa_pred matching MultiFlow
    sequence_pred_type: ModelSequencePredictionEnum = (
        ModelSequencePredictionEnum.sequence_ipa_net
    )
    aa_pred: ModelAAPredConfig = field(default_factory=ModelAAPredConfig)
    sequence_ipa_net: ModelSequenceIPANetConfig = field(
        default_factory=ModelSequenceIPANetConfig
    )


class InterpolantRotationsScheduleEnum(StrEnum):
    linear = "linear"
    # Note that structures seem to generate better when rotations settle first.
    # Therefore, rotations schedule should scale t faster than translations.
    # For example, it might make sense for rotations=exponential, translations=linear.
    exp = "exp"


@dataclass
class InterpolantRotationsConfig(BaseClassConfig):
    # corrupt rotations (unless inverse folding)
    corrupt: bool = (
        "${ternary:${equals: ${inference.task}, 'inverse_folding'}, False, True}"
    )
    # sampled noise std dev
    igso3_sigma: float = 1.5  # 1.5 in public multiflow
    train_schedule: InterpolantRotationsScheduleEnum = (
        InterpolantRotationsScheduleEnum.linear
    )
    sample_schedule: InterpolantRotationsScheduleEnum = (
        InterpolantRotationsScheduleEnum.exp
    )
    exp_rate: float = 10
    # stochastic paths
    # TODO consider min_t for stochastic rotations, so field can settle before injecting noise.
    stochastic: bool = "${shared.stochastic}"
    # sigma scaled by sqrt(t * (1-t)) * stochastic_noise_intensity
    # Roughly, 0.5 => 11°, 1.0 => 23°, 2.0 => 34° over 500 timesteps
    stochastic_noise_intensity: float = 0.5  # `g` in FoldFlow SO3SFM


class InterpolantTranslationsNoiseTypeEnum(StrEnum):
    centered_gaussian = "centered_gaussian"
    centered_harmonic = "centered_harmonic"


class InterpolantTranslationsScheduleEnum(StrEnum):
    linear = "linear"
    # variance-preserving SDE. sampling only.
    # does not itself inject noise into path, but variance controlled to remain stable throughout process.
    # helps "smooth" intermediate samples to avoid collapse (low variance) or divergence (high variance).
    vpsde = "vpsde"


@dataclass
class InterpolantTranslationsConfig(BaseClassConfig):
    # corrupt translations (unless inverse folding)
    corrupt: bool = (
        "${ternary:${equals: ${inference.task}, 'inverse_folding'}, False, True}"
    )
    # noise source distribution
    noise_type: InterpolantTranslationsNoiseTypeEnum = (
        InterpolantTranslationsNoiseTypeEnum.centered_harmonic
    )
    # batch_ot: enable minibatch optimal transport, otherwise enable aligning noise to sampled data
    batch_ot: bool = True
    batch_align: bool = True
    # train_schedule: training schedule for interpolant
    train_schedule: InterpolantTranslationsScheduleEnum = (
        InterpolantTranslationsScheduleEnum.linear
    )
    # sample_schedule: sampling schedule for interpolant.
    sample_schedule: InterpolantTranslationsScheduleEnum = (
        "${ternary:${equals: ${shared.stochastic}, True}, 'vpsde', 'linear'}"
    )
    # sample_temp: sampling temperature
    sample_temp: float = 1.0  # TODO - drop or check FrameDiff for usage
    # vpsde_bmin: variance-preserving SDE minimum
    vpsde_bmin: float = 0.1
    # vpsde_bmax: variance-preserving SDE maximum
    vpsde_bmax: float = 20.0
    # potentials and radius of gyration (rog) not used in public multiflow code (but in config)
    # potential: str = 'null'
    # potential_t_scaling: bool = False
    # rog:
    #   weight: 10.0
    #   cutoff: 5.0
    # stochastic paths
    stochastic: bool = "${shared.stochastic}"
    # sigma scaled by sqrt(t * (1-t)) * stochastic_noise_intensity
    # Roughly, 0.5 => 0.2Å, 1.0 => 0.4Å, 2.0 => 0.6Å over 500 timesteps
    stochastic_noise_intensity: float = 1.0  # `g` in FoldFlow SO3SFM


class InterpolantAATypesScheduleEnum(StrEnum):
    linear = "linear"
    exp = "exp"  # TODO re-introduce, 'exp' not used in public MultiFlow code


class InterpolantAATypesInterpolantTypeEnum(StrEnum):
    masking = "masking"
    uniform = "uniform"


@dataclass
class InterpolantAATypesConfig(BaseClassConfig):
    """
    Interpolant for amino acids.

    Note that there are two interpolants: one for training and one for inference
    TODO reconsider ternary use to differentiate training and inference interpolants
        or, consider a subclass which handles default arguments accordingly
    """

    # corrupt amino acid types (unless forward folding)
    corrupt: bool = (
        "${ternary:${equals: ${inference.task}, 'forward_folding'}, False, True}"
    )
    # schedule: training schedule for interpolant
    schedule: InterpolantAATypesScheduleEnum = InterpolantAATypesScheduleEnum.linear
    # schedule_exp_rate: exponential rate for schedule
    # in multiflow, exp rate was 10 in base.yaml, but -3 elsewhere
    schedule_exp_rate: float = -3
    # interpolant_type: noise distribution and type of interpolant
    interpolant_type: InterpolantAATypesInterpolantTypeEnum = (
        InterpolantAATypesInterpolantTypeEnum.masking
    )
    # temp: temperature
    temp: float = 0.1
    # noise: AA type change noise. No noise for forward_folding.
    noise: float = (
        "${ternary:${equals: ${inference.task}, 'forward_folding'}, 0.0, 20.0}"
    )
    # do_purity: enable purity, allows for unmasking by max log probs and for re-masking by `noise`
    # purity requires masking interpolant
    do_purity: bool = (
        "${ternary:${equals: ${inference.task}, 'forward_folding'}, False, True}"
    )
    # stochastic CTMC
    stochastic: bool = "${shared.stochastic}"
    # sigma scaled by sqrt(t * (1-t)) * stochastic_noise_intensity
    # Roughly, 0.5 => 0.2 jumps/residue, 1.0 = > 0.4, 1.5 => 0.6 over 500 timesteps  TODO doublecheck
    stochastic_noise_intensity: float = 0.25


class InterpolantTrainTimeSamplingEnum(StrEnum):
    uniform = "uniform"
    late_biased = "late_biased"


@dataclass
class InterpolantSamplingConfig(BaseClassConfig):
    # training takes a random t. Sampling runs over t timestemps.
    num_timesteps: int = 500


@dataclass
class InterpolantConfig(BaseClassConfig):
    """
    Interpolant configuration.
    Note there is a training interpolant, and a sampling interpolant, with their own configs.

    Note on `codesign_separate_t`:
    During training we can "overload" the training tasks by specifying `codesign_separate_t`,
    such that some proportion of the batch is allocated to other tasks like `forward_folding` or `inverse_folding`,
    by fixing the sequence or structure, respectively, at t=1.
    During training in `corrupt_batch()`, effectively sets some proportion of the batch to be
        `forward_folding` or `inverse_folding` (i.e. structure if inverse, sequence if forward)
        to enable these tasks in inference (proportions cfg below).
        In addition, the remainder of the time, the structure and sequence are given a different `t`.
    During sampling in `sample()`, for tasks with fixed domains (i.e. forward/inverse folding),
        those domains are fixed at ~t=1. Value should match how the model was trained for these tasks.
    """

    train_time_sampling_method: InterpolantTrainTimeSamplingEnum = (
        InterpolantTrainTimeSamplingEnum.late_biased
    )
    min_t: float = 1e-2
    # `codesign_separate_t` allows separate `t` times for rots / trans / aatypes so fixed domains are at ~t=1.
    codesign_separate_t: bool = True
    # `forward_folding` proportion of samples; requires `codesign_separate_t`
    codesign_forward_fold_prop: float = 0.1  # default 0.1 in public MultiFlow
    # `inverse_folding` proportion of samples; requires `codesign_separate_t`
    codesign_inverse_fold_prop: float = 0.1  # default 0.1 in public MultiFlow
    # `inpainting_unconditional_prop` in training converts some `inpainting` to `unconditional` batches.
    inpainting_unconditional_prop: float = 0.2
    # enable self-conditioning
    self_condition: bool = "${model.edge_features.self_condition}"
    self_condition_prob: float = 0.5  # 0.5 in public MultiFlow
    # kappa allows scaling rotation t exponentially during sampling
    provide_kappa: bool = True

    # sub-modules
    rots: InterpolantRotationsConfig = field(default_factory=InterpolantRotationsConfig)
    trans: InterpolantTranslationsConfig = field(
        default_factory=InterpolantTranslationsConfig
    )
    aatypes: InterpolantAATypesConfig = field(default_factory=InterpolantAATypesConfig)
    sampling: InterpolantSamplingConfig = field(
        default_factory=InterpolantSamplingConfig
    )


@dataclass
class DataLoaderConfig(BaseClassConfig):
    num_workers: int = 8
    prefetch_factor: int = 10


@dataclass
class DataSamplerConfig(BaseClassConfig):
    # Setting for 40GB GPUs
    max_batch_size: int = 80  # 128 for 80GB GPUs
    max_num_res_squared: int = 400_000  # 1_000_000 for 80GB GPUs


class DatasetEnum(StrEnum):
    """dataset for training"""

    pdb = "pdb"


@dataclass
class DataConfig(BaseClassConfig):
    task: DataTask = DataTask.hallucination
    dataset: DatasetEnum = DatasetEnum.pdb
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    sampler: DataSamplerConfig = field(default_factory=DataSamplerConfig)


@dataclass
class DatasetFilterConfig(BaseClassConfig):
    """Config for filtering metadata CSV. requires data to be in metadata CSV."""

    max_num_res: int = 384
    min_num_res: int = 60
    max_coil_percent: float = 0.667  # was 0.5 in public MultiFlow
    rog_quantile: float = 0.96
    # minimum percent of known and modelable residues in the structure of total sequence.
    max_percent_residues_unknown: float = 0.5
    oligomeric: List[str] = field(
        default_factory=lambda: [
            "monomer",
            "monomeric",
            "monomeric,monomeric",
            "homomer",
        ]
    )
    num_chains: List[int] = field(default_factory=lambda: [1])

    @classmethod
    def multimeric(cls) -> "DatasetFilterConfig":
        """Factory for multimeric configuration"""

        # generate reasonable combos: {name}, {name,name}, {name,name,name}
        oligomeric_names = [
            f"{prefix}meric" for i, prefix in OLIGOMERIC_PREFIXES.items()
        ]
        oligomeric_combos = [
            ",".join(combo)
            for i in range(1, 4)
            for combo in itertools.combinations(oligomeric_names, i)
        ]

        return cls(
            oligomeric=oligomeric_combos,
            num_chains=list(range(2, 20)),
        )


class DatasetInpaintingMotifStrategy(StrEnum):
    # Enable picking from all strategies
    ALL = "ALL"

    # single motif
    single_motif = "single_motif"
    # [min_num_motifs, max_num_motifs], skewed toward fewer
    variable_motifs = "variable_motifs"
    # sample 1 res position, sample N, mask closest N neighbors
    random_neighbors = "random_neighbors"
    # sample highly interacting res, mask neighbors < distance threshold
    densest_neighbors = "densest_neighbors"
    # (multimer) binding interface, mask only one chain
    binding_interface_single_chain = "binding_interface_single_chain"
    # (multimer) pick all residues within interaction threshold
    binding_interface = "binding_interface"
    # (multimer) pick any chain and consider it a "binder"
    binder = "binder"

    @staticmethod
    def is_multimeric(strategy: "DatasetInpaintingMotifStrategy") -> bool:
        """Check if `strategy` is multimeric"""
        return strategy in [
            DatasetInpaintingMotifStrategy.binding_interface_single_chain,
            DatasetInpaintingMotifStrategy.binding_interface,
            DatasetInpaintingMotifStrategy.binder,
        ]


@dataclass
class DatasetInpaintingConfig(BaseClassConfig):
    """
    Configuration for generating motifs / scaffolding
    """

    # try to trim low plddt ends from structures, see `dataset.min_plddt_threshold`
    trim_low_plddt_ends: bool = True
    # Specify motif selection strategy, or `ALL` to enable all.
    strategy: DatasetInpaintingMotifStrategy = DatasetInpaintingMotifStrategy.ALL

    # Strategy-dependent parameters
    # target fraction of residues to be in motif (remainder to be diffused)
    min_percent_motifs: float = 0.10
    max_percent_motifs: float = 0.70
    # motif length bounds
    min_motif_len: int = 8
    max_motif_len: int = 768
    # minimal spacing between motifs (i.e. min scaffold length)
    min_padding: int = 3
    # for variable number of motifs
    min_num_motifs: int = 1
    max_num_motifs: int = 5
    # for multimers, determining interacting residues
    interaction_dist_threshold_ang: float = 6.0
    proximity_dist_threshold_ang: float = 10.0


# dataset_metadata_dir_path is the root directory for dataset / metadata files
dataset_metadata_dir_path = PATH_PROJECT_ROOT / "cogeneration" / "datasets" / "metadata"


class DatasetTrimMethod(StrEnum):
    """
    Methods for trimming ChainFeatures to modeled residues.
    Relevant to multimers.
    """

    # concat chains and trim ends
    whole_complex = "whole_complex"
    # trim chains independently
    chains_independently = "chains_independently"

    def to_dataset_column(self) -> MetadataColumn:
        """
        Convert to DatasetColumn column name
        """
        if self == DatasetTrimMethod.whole_complex:
            return MetadataColumn.modeled_seq_len
        elif self == DatasetTrimMethod.chains_independently:
            return MetadataColumn.modeled_indep_seq_len
        else:
            raise ValueError(f"Unknown trim method: {self}")


@dataclass
class DatasetConfig(BaseClassConfig):
    """
    Information about the Dataset of protein structures and sequences.
    Assumes a structure matching the data provided by public Multiflow,
    but the pipeline to generate the data is not included.

    Note that a dataset comprises:
    - main data (i.e. from PDB)
    - redesigned data (to fine tune using ProteinMPNN sequences)
    - synthetic data (from AF2 predictions)
    and each of these shards may have metadata and clustering associated.

    Default dataset is PDB training dataset.

    use PDBPost2021() for test dataset

    # TODO - support scope and swissprot datasets
    # MultiFlow public config lists: `pdb`, `pdb_mixed`, `pdb_post2021`, `scope`, `scope_mixed`, `swissprot`
    """

    seed: int = "${shared.seed}"
    processed_data_path: str = os.path.dirname(dataset_metadata_dir_path)
    csv_path: Path = dataset_metadata_dir_path / "pdb_metadata.csv"
    cluster_path: Optional[Path] = dataset_metadata_dir_path / "pdb.clusters"
    max_cache_size: int = 100_000
    cache_num_res: int = 0  # min size to enable caching
    # plddt [0, 100]. Minimum threshold, per residue, masked if below and add_plddt_mask=True
    add_plddt_mask: bool = True
    min_plddt_threshold: float = 0.0
    # trim chains independently to modeled positions
    # removes non-residues between chains, more important for multimers
    modeled_trim_method: DatasetTrimMethod = DatasetTrimMethod.chains_independently
    # add gaussian noise to atom positions
    # TODO cfg to only add noise if t below some threshold (requires moving out of dataset)
    # TODO ensure noise added each time accessed and not cached
    noise_atom_positions_angstroms: float = 0.1

    # Redesigned, i.e. use ProteinMPNN to generate sequences for a structure
    use_redesigned: bool = True
    redesigned_csv_path: Optional[Path] = (
        dataset_metadata_dir_path / "pdb_redesigned.csv"
    )
    redesigned_rmsd_threshold: float = 2.0

    # Synthetic, e.g. AlphaFold structures?
    use_synthetic: bool = True
    synthetic_csv_path: Optional[Path] = (
        dataset_metadata_dir_path / "distillation_metadata.csv"
    )
    synthetic_cluster_path: Optional[Path] = (
        dataset_metadata_dir_path / "distillation.clusters"
    )

    # Eval parameters
    test_set_pdb_ids_path: Optional[Path] = None
    max_eval_length: Optional[int] = 256
    samples_per_eval_length: int = 5
    num_eval_lengths: int = 8

    # Scaffolding / inpainting parameters
    inpainting: DatasetInpaintingConfig = field(default_factory=DatasetInpaintingConfig)

    # Filtering
    filter: DatasetFilterConfig = field(default_factory=DatasetFilterConfig)

    @classmethod
    def PDBPost2021(cls):
        # Config for test-set of PDB structures from 2021 and later
        return cls(
            csv_path=dataset_metadata_dir_path / "test_set_metadata.csv",
            cluster_path=dataset_metadata_dir_path / "test_set_clusters.csv",
            cache_num_res=0,
            add_plddt_mask=False,
            # disable Redesigned and Synthetic for test set
            use_redesigned=False,
            redesigned_csv_path=None,
            use_synthetic=False,
            synthetic_csv_path=None,
            synthetic_cluster_path=None,
            # Eval parameters
            test_set_pdb_ids_path=dataset_metadata_dir_path / "test_set_pdb_ids.csv",
            # slightly relaxed filters to match original multiflow parameters
            filter=DatasetFilterConfig(
                max_num_res=400,
                min_num_res=50,
            ),
        )


@dataclass
class ExperimentTrainingConfig(BaseClassConfig):
    mask_plddt: bool = True
    bb_atom_scale: float = 0.1
    trans_scale: float = 0.1
    aatypes_loss_weight: float = 0.5  # default 0.0 in multiflow
    aatypes_loss_mean_or_sum: str = "mean"
    aatypes_loss_use_likelihood_weighting: bool = False
    translation_loss_weight: float = 2.0
    # losses scaling normalized up to t
    t_normalize_clip: float = 0.9
    rotation_loss_weights: float = 1.0
    aux_loss_weight: float = 0.5  # default 0.0 in multiflow
    aux_loss_use_bb_loss: bool = True
    aux_loss_use_pair_loss: bool = True
    aux_loss_t_pass: float = 0.5  # minimum t for aux loss


@dataclass
class ExperimentWandbConfig(BaseClassConfig):
    """W&B configuration. Some properties are kwargs to logger"""

    name: str = "${data.task}_${data.dataset}_${shared.id}"
    project: str = "cogeneration"
    # offline if `local` (not copied to W&B)
    offline: bool = "${ternary:${equals: ${shared.local}, True}, True, False}"


@dataclass
class ExperimentOptimizerConfig(BaseClassConfig):
    lr: float = 1e-4


@dataclass
class ExperimentTrainerConfig(BaseClassConfig):
    """
    Arguments to Pytorch Lightning Trainer()
    """

    # probably want "gpu" if not on a Mac, "mps" for Mac M# GPU
    # `accelerator` is argument name to Trainer()
    accelerator: str = "${ternary:${equals: ${shared.local}, True}, 'mps', 'gpu'}"
    # `strategy` is argument name to Trainer(), ddp = distributed data parallel
    strategy: Optional[str] = (
        "${ternary:${equals: ${shared.local}, True}, 'auto', 'ddp'}"
    )
    overfit_batches: int = 0
    min_epochs: int = 1  # prevents early stopping
    max_epochs: int = 200
    deterministic: bool = False
    check_val_every_n_epoch: int = 4
    accumulate_grad_batches: int = 2
    # logging
    log_every_n_steps: int = 1
    # TODO put somewhere else, invalid argument to Trainer()
    # local_tensorboard_logdir: str = "./tensorboard_logs"

    def __post_init__(self):
        # distributed training (ddp) not currently supported with MPS
        if self.accelerator == "mps" and self.strategy is not None:
            self.strategy = "auto"


@dataclass
class ExperimentCheckpointerConfig(BaseClassConfig):
    """
    Arguments to ModelCheckpoint()
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html

    By default, save a `last.ckpt` `every_n_epochs` into `dirpath`.
    `ExperimentConfig` can save a `final.ckpt` .
    """

    dirpath: str = (
        "ckpt/${experiment.wandb.project}/${experiment.wandb.name}/${shared.id}"
    )
    # `save_last` ensures a `last.ckpt` copy is always created
    save_last: bool = True
    # recommend checkpoint on some multiple of validation epoch interval. too frequent will eat up disk.
    every_n_epochs: int = 8
    # save `k` best models according to some metric
    save_top_k: int = 2
    monitor: str = "valid/codesign_bb_rmsd"
    mode: str = "min"


@dataclass
class ExperimentConfig(BaseClassConfig):
    """Training Experiment configuration."""

    seed: int = "${shared.seed}"
    # debug True for local tensorboard logger, False to enable W&B, saving outputs etc
    debug: bool = False
    # num GPU devices TODO support more than one, esp for GPUs
    num_devices: int = 1
    # checkpoint path, parent directory expected to contain `config.yaml`
    warm_start_ckpt: Optional[str] = None
    # override config with warm start config
    warm_start_cfg_override: bool = False
    # force reload module config, provide path
    raw_state_dict_reload: Optional[str] = None
    # pytorch Trainer profiler, "simple", "advanced", None
    profiler: Optional[str] = None
    # enable torch.compile(), requires no graph breaks TODO make it work
    torch_compile: bool = False
    # save a `final.ckpt` when training is complete, defaults to symlink sentinel
    save_final_ckpt: bool = True
    final_ckpt_symlink: bool = True

    # sub-modules
    training: ExperimentTrainingConfig = field(default_factory=ExperimentTrainingConfig)
    wandb: ExperimentWandbConfig = field(default_factory=ExperimentWandbConfig)
    optimizer: ExperimentOptimizerConfig = field(
        default_factory=ExperimentOptimizerConfig
    )
    trainer: ExperimentTrainerConfig = field(default_factory=ExperimentTrainerConfig)
    checkpointer: ExperimentCheckpointerConfig = field(
        default_factory=ExperimentCheckpointerConfig
    )


@dataclass
class InferenceSamplesConfig(BaseClassConfig):
    """
    Inference sampling configuration.
    Can define either `length_subset` or `min_length`, `max_length`, `length_step`.
    `length_subset` takes priority.
    """

    # Number of backbone samples per sequence length.
    samples_per_length: int = 100
    # Batch size when sampling from the model
    num_batch: int = 1
    # Subset of lengths to sample. If null, sample all targets between min_length and max_length
    length_subset: Optional[List[int]] = field(
        default_factory=lambda: [70, 100, 200, 300]
    )
    # Minimum sequence length to sample.
    min_length: int = 60
    # Maximum sequence length to sample.
    max_length: int = 256
    # gap between lengths to sample, `range(min_length, max_length, length_step)`
    length_step: int = 1
    # Multimers - set `chain_idx` for 2+ chains, where each chain must be `min_length`
    multimer_fraction: float = 0.25
    multimer_min_length: int = 100


@dataclass
class InferenceConfig(BaseClassConfig):
    task: InferenceTask = InferenceTask.unconditional

    seed: int = "${shared.seed}"
    use_gpu: bool = True
    num_gpus: int = 1
    predict_dir: str = str(PATH_PROJECT_ROOT / "inference_outputs")
    inference_subdir: str = "${shared.id}"

    # checkpoints
    saved_ckpt_dir: str = "${shared.project_root}/ckpt/${shared.id}"

    unconditional_ckpt_path: Optional[str] = str(PATH_PUBLIC_WEIGHTS / "last.ckpt")
    forward_folding_ckpt_path: Optional[str] = str(PATH_PUBLIC_WEIGHTS / "last.ckpt")
    inverse_folding_ckpt_path: Optional[str] = str(PATH_PUBLIC_WEIGHTS / "last.ckpt")
    # note inpainting not explicitly supported by public MultiFlow model
    inpainting_ckpt_path: Optional[str] = str(PATH_PUBLIC_WEIGHTS / "last.ckpt")

    interpolant: InterpolantConfig = field(default_factory=InterpolantConfig)
    samples: InferenceSamplesConfig = field(default_factory=InferenceSamplesConfig)

    # validation
    # whether to also fold the generated pmpnn seq for each structure
    also_fold_pmpnn_seq: bool = True
    # whether to also save generation trajectory artifacts when sampling
    write_sample_trajectories: bool = True
    # whether to include animations, which are slow to generate (~10-15s for 50 frames)
    write_animations: bool = True
    animation_max_frames: int = 50


@dataclass
class FoldingConfig(BaseClassConfig):
    seq_per_sample: int = 8
    folding_model: str = "af2"  # "af2" only at the moment, maybe "esm" in the future
    # dedicated device for folding. decrement other devices by 1 if True
    own_device: bool = False
    # Assume ProteinMPNN to be a sibling to project root, installed separately
    pmpnn_path: Path = PATH_PROJECT_ROOT.parent / "ProteinMPNN"
    pmpnn_seed: int = "${shared.seed}"
    pt_hub_dir: Path = PATH_PROJECT_ROOT / "cache" / "torch"
    # uses LocalColabFold for folding locally
    # https://github.com/YoshitakaMo/localcolabfold
    # which is a variant of ColabFold
    # installation: https://bcrf.biochem.wisc.edu/2023/04/27/alphafold2-on-macintosh-m1/
    colabfold_path: Path = (
        PATH_PROJECT_ROOT.parent / "localcolabfold/colabfold-conda/bin/colabfold_batch"
    )


@dataclass
class Config(BaseClassConfig):
    """
    We use dataclasses as part of hydra's structured config system, to enable type checking and default values.
    This is the base class and default config.
    """

    shared: SharedConfig = field(default_factory=SharedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    folding: FoldingConfig = field(default_factory=FoldingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    interpolant: InterpolantConfig = field(default_factory=InterpolantConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def load_dict_from_file(
        cls,
        filepath: Union[str, Path],
        remove_unknown_fields: bool = True,
    ) -> Tuple[Dict[str, Any], bool]:
        """
        Load a Config saved as yaml as a Dict, and check if it appears to be a public Multiflow checkpoint.

        Specify `remove_unknown_fields` to remove any fields not in the current schema.

        Returns a dict, rather than Config, to avoid setting inappropriate defaults,
        since it is likely to be merged after calling this function.
        """
        yml_cfg = OmegaConf.load(filepath)
        # `resolve` required for public Multiflow - they include templates in the saved cfg
        # If an interpolated cfg is saved, should be idempotent.
        yml_dict = OmegaConf.to_container(yml_cfg, resolve=True)

        # Check if it appears to be a public Multiflow checkpoint, so can handle accordingly
        is_multiflow = (
            "interpolant" in yml_dict and "twisting" in yml_dict["interpolant"]
        )

        # remove fields not in current schema to avoid problems merging, e.g. if an old key is present.
        if remove_unknown_fields:
            yml_dict = prune_unknown_dataclass_fields(cls=cls, obj=yml_dict)

        return yml_dict, is_multiflow

    def merge_checkpoint_cfg(
        self,
        ckpt_path: str,
        preserve_inference_cfg: bool = True,
        remove_unknown_fields: bool = True,
    ) -> Tuple["Config", str]:
        """
        Loads and merges a checkpoint config into the current config.

        The checkpoint config is merged into the current config,
        but inference-specific fields from the current config are preserved.

        `ckpt_path` should be the path to a checkpoint file, e.g. `last.ckpt`, with a sibling `config.yaml`.
        `remove_unknown_fields` specifies whether to remove fields not in the current schema.
        If false, and there are unknown fields, an error is raised.
        `preserve_inference_cfg` specifies whether to preserve the inference config from the current config,
        e.g. if loading a checkpoint from training and beginning inference

        We also attempt to support MultiFlow checkpoints, which are only partially compatible with all our options.
        We attempt to map the state_dict to the new module names, and save a new checkpoint, which can be loaded.
        This requires that the model architecture matches MultiFlow, i.e. the same modules and shapes, which
        will be enforced by using the `ckpt` configuration for `cfg.model`.
        However, all config options not in our current schemas are dropped, and we'll just assume our defaults are ok.
        There are some cases where that may not be the case! Use the `Config.public_multiflow()` constructor.
        """
        if not ckpt_path.endswith(".ckpt"):
            raise ValueError(
                f"Invalid checkpoint path {ckpt_path}, should end with .ckpt"
            )
        assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist."

        ckpt_dir = os.path.dirname(ckpt_path)
        ckpt_cfg_path = os.path.join(ckpt_dir, "config.yaml")
        assert os.path.exists(
            ckpt_cfg_path
        ), f"Checkpoint {ckpt_cfg_path} does not exist."
        ckpt_cfg_dict, is_multiflow = Config.load_dict_from_file(
            ckpt_cfg_path, remove_unknown_fields=remove_unknown_fields
        )

        # Hang on to the original config, we'll override some fields in the merged config
        orig_cfg = self.interpolate()

        # Merge the checkpoint config into the current config
        merged_cfg = self.merge_dict(ckpt_cfg_dict, interpolate=True)

        # For inference, overwrite certain fields from the checkpoint config.
        # We want to inherit the model specification etc. to match what was saved
        # but preserve current cfg for inference, validation, etc.
        if preserve_inference_cfg:
            # metadata (even though already interpolated)
            merged_cfg.shared.id = orig_cfg.shared.id
            # inference (and inference interpolant), folding validation take priority
            merged_cfg.inference = orig_cfg.inference
            merged_cfg.folding = orig_cfg.folding
            # datasets, if used for inference
            merged_cfg.dataset = orig_cfg.dataset
            # trainer cfg
            merged_cfg.experiment.trainer = orig_cfg.experiment.trainer
            merged_cfg.experiment.num_devices = orig_cfg.experiment.num_devices

        # Special handling for public Multiflow checkpoints.
        # If we got a config from MultiFlow, we need to map to our new module names.
        # We'll map, and save a new checkpoint, and then load that checkpoint.
        # TODO move to separate function
        if is_multiflow:
            ckpt = torch.load(
                ckpt_path, map_location=torch.device("cpu"), weights_only=False
            )

            # Define new checkpoint directory
            ckpt_dir = (
                f"{merged_cfg.inference.saved_ckpt_dir}/mapped_{merged_cfg.shared.id}"
            )
            ckpt_path = os.path.join(ckpt_dir, "mapped.ckpt")

            # Map modules in state_dict
            # Assumes that these modules are active in the network, i.e. network shape is the same as MultiFlow.
            state_dict = ckpt["state_dict"]
            new_state_dict = OrderedDict()
            replacements = {
                "model.trunk.": "model.attention_ipa_trunk.trunk.",
                "model.aatype_pred_net.": "model.aa_pred_net.aatype_pred_net.",
            }
            for key, value in state_dict.items():
                for old, new in replacements.items():
                    key = key.replace(old, new)
                new_state_dict[key] = value

            # Save new checkpoint
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt["state_dict"] = new_state_dict
            torch.save(ckpt, ckpt_path)

        return merged_cfg, ckpt_path

    @classmethod
    def test_uninterpolated(cls, tmp_path: Path) -> "Config":
        """
        Return a config set up for testing, with a tiny (useless) model.
        Requires running `interpolate()`, to allow for further modification.
        """
        raw_cfg = cls()

        # set to local mode, impacting accelerator etc.
        raw_cfg.shared.local = True

        # default to tiny model for faster model evaluations
        raw_cfg.model.hyper_params = ModelHyperParamsConfig.tiny()
        raw_cfg.model.edge_features.feat_dim = 8
        # and smaller transformers
        raw_cfg.model.ipa.no_heads = 2
        raw_cfg.model.ipa.num_blocks = 2
        raw_cfg.model.sequence_ipa_net.ipa.no_heads = 2
        raw_cfg.model.sequence_ipa_net.ipa.num_blocks = 1

        # filter to small PDBs for faster model + sampling
        raw_cfg.dataset.filter.min_num_res = 20
        raw_cfg.dataset.filter.max_num_res = 40
        # avoid synthetic + redesigned samples
        raw_cfg.dataset.use_redesigned = False
        raw_cfg.dataset.use_synthetic = False
        # small batches
        raw_cfg.data.sampler.max_batch_size = 4

        # set output directories to temp paths
        raw_cfg.experiment.checkpointer.dirpath = str(tmp_path / "ckpt")
        raw_cfg.inference.predict_dir = str(tmp_path / "inference")

        # limit number of lengths + timesteps sampled for validation / inference
        raw_cfg.interpolant.sampling.num_timesteps = 3
        raw_cfg.inference.interpolant.sampling.num_timesteps = 3
        raw_cfg.inference.samples.samples_per_length = 2
        raw_cfg.inference.samples.length_subset = [10, 30]
        raw_cfg.dataset.samples_per_eval_length = 2
        raw_cfg.dataset.num_eval_lengths = 1
        # shortest validation samples in public data are 60 residues
        raw_cfg.dataset.max_eval_length = 63

        # inpainting, always generate motifs by default
        raw_cfg.dataset.inpainting.strategy = (
            DatasetInpaintingMotifStrategy.single_motif
        )
        raw_cfg.interpolant.inpainting_unconditional_prop = 0.0
        raw_cfg.interpolant.codesign_forward_fold_prop = 0.0
        raw_cfg.interpolant.codesign_inverse_fold_prop = 0.0

        return raw_cfg

    @classmethod
    def public_multiflow(cls):
        """
        Returns a config that maintains compatibility with Public MultiFlow weights + better aligned with its defaults.
        Requires running `interpolate()`, to allow for further modification.
        """
        raw_cfg = cls()

        # Default to unconditional generation
        raw_cfg.data.task = DataTask.hallucination
        raw_cfg.inference.task = InferenceTask.unconditional

        # Model
        # Use public MultiFlow model size hyperparameters
        raw_cfg.model.hyper_params = ModelHyperParamsConfig.public_multiflow()
        # Use simple aa_pred_net from public MultiFlow
        raw_cfg.model.sequence_pred_type = ModelSequencePredictionEnum.aa_pred
        # stochastic paths not part of public MultiFlow
        raw_cfg.shared.stochastic = False
        # use simple gaussian prior for translations
        raw_cfg.interpolant.trans.noise_type = (
            InterpolantTranslationsNoiseTypeEnum.centered_gaussian
        )
        raw_cfg.inference.interpolant.trans.noise_type = (
            InterpolantTranslationsNoiseTypeEnum.centered_gaussian
        )
        # Don't predict torsion angles
        raw_cfg.model.predict_psi_torsions = False
        # positional embeddings
        raw_cfg.model.node_features.embed_chain = False
        raw_cfg.model.edge_features.embed_chain = False
        raw_cfg.model.hyper_params.pos_embed_max_len = 2056
        raw_cfg.model.hyper_params.pos_embed_method = (
            PositionalEmbeddingMethod.sine_cosine
        )
        # no filter on unknown residues
        raw_cfg.dataset.filter.max_percent_residues_unknown = 0.0
        # rotations non-expoential schedule
        raw_cfg.interpolant.provide_kappa = False
        raw_cfg.inference.interpolant.provide_kappa = False
        # linear time sampling for training
        raw_cfg.interpolant.train_time_sampling_method = (
            InterpolantTrainTimeSamplingEnum.uniform
        )

        # Weights
        # assume we have public weights available, use as default checkpoint
        raw_cfg.inference.unconditional_ckpt_path = str(
            PATH_PUBLIC_WEIGHTS / "last.ckpt"
        )
        raw_cfg.inference.forward_folding_ckpt_path = str(
            PATH_PUBLIC_WEIGHTS / "last.ckpt"
        )
        raw_cfg.inference.inverse_folding_ckpt_path = str(
            PATH_PUBLIC_WEIGHTS / "last.ckpt"
        )

        return raw_cfg


# Register the config class with Hydra
cs = ConfigStore.instance()
cs.store(name="base", node=Config)
