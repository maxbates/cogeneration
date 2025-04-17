import datetime
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from cogeneration.util.base_classes import StrEnum

"""
Structured configurations for cogeneration.

See __init__.py for extensions to OmegaConf, e.g. `ternary`.

Several parameters from Multiflow appear to be unused in the public code.
Many are marked with `# DROP`.
"""

PATH_PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
PATH_PUBLIC_WEIGHTS = PATH_PROJECT_ROOT / "multiflow_weights"

# TODO - support default configs for `forward_folding` and `inverse folding`
#   There are a handful of values across the config that need to have other defaults.
#   i.e. instead of the current `ternary` paradigm
#   to better differentate the training vs inference interpolant

# hydra does not currently support `Literal` and suggests using Enums instead.
# https://github.com/omry/omegaconf/issues/422


class DataTaskEnum(StrEnum):
    """task for training"""

    hallucination = "hallucination"
    inpainting = "inpainting"  # aka `scaffolding`


class InferenceTaskEnum(StrEnum):
    """task for inference"""

    unconditional = "unconditional"
    inpainting = "inpainting"  # aka `scaffolding`
    forward_folding = "forward_folding"
    inverse_folding = "inverse_folding"


class DatasetEnum(StrEnum):
    """dataset for training"""

    pdb = "pdb"


@dataclass
class SharedConfig:
    """
    shared config, system "flags", metadata
    """

    # `local` to train locally on a Mac. use `mps` not `gpu` with `ddp`, etc.
    local: bool = True
    # now timestamp
    now: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    # randomness / stochastic paths
    stochastic: bool = True
    seed: int = 123
    # project root path
    project_root: str = str(PATH_PROJECT_ROOT)

    # TODO - `size` argument to determine model hyperparameters
    # TODO - gpu size argument to determine batch size, etc.


@dataclass
class ModelHyperParamsConfig:
    """
    Shared hyperparameters for the model.
    Use a structured config for hyperparameters so easy to reference in templates.
    Default values match those from Multiflow. Use factory methods for different configurations.

    TODO centralize other hyperparameters here, like transformer depth, heads, etc.
    (assuming we really want to share them, i.e. different transformers should be the same size)

    TODO register hydra config group to enable easy switching
    https://hydra.cc/docs/tutorials/structured_config/defaults/
    """

    node_embed_size: int = 256
    edge_embed_size: int = 128
    pos_embed_size: int = 128
    timestep_embed_size: int = 128

    aa_num_tokens: int = 21  # number of amino acid types (if masking), 21 = mask

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
class ModelNodeFeaturesConfig:
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
    # max_num_res: maximum number of residues
    max_num_res: int = 2000
    # embed_chain: whether to embed chain index.
    embed_chain: bool = True
    # embed_aatype: whether to embed amino acid type
    embed_aatype: bool = True
    # use_mlp: whether to use MLP for embedding, otherwise linear layer
    use_mlp: bool = True


@dataclass
class ModelEdgeFeaturesConfig:
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
    # self_condition: used by interpolant config.
    self_condition: bool = True
    # embed_chain: whether to embed chain index.
    embed_chain: bool = True
    # embed_diffuse_mask: whether to embed diffuse mask. important esp. for `scaffolding` task.
    embed_diffuse_mask: bool = True


@dataclass
class ModelIPAConfig:
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
class ModelAAPredConfig:
    """
    Amino acid prediction configuration using simple linear layers.
    """

    # c_s: node embedding size (input)
    c_s: int = "${model.hyper_params.node_embed_size}"
    # aatype_pred_num_tokens: number of amino acid types => logits / rate-matrix shape
    aatype_pred_num_tokens: int = "${model.hyper_params.aa_num_tokens}"


@dataclass
class ModelSequenceIPANetConfig:
    """
    IPA style transformer, predicting logits instead of backbone update.

    TODO - consider centralizing on AAPredConfig, and just use this for IPA, without duplicate fields

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
class ModelConfig:
    symmetric: bool = False

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
class InterpolantRotationsConfig:
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
    stochastic: bool = "${shared.stochastic}"
    # sigma scaled by g * sqrt(t * (1-t))
    stochastic_noise_intensity: float = 0.1  # `g` in FoldFlow SO3SFM


class InterpolantTranslationsScheduleEnum(StrEnum):
    linear = "linear"
    # variance-preserving SDE. sampling only.
    # does not itself inject noise into path, but variance controlled to remain stable throughout process.
    # helps "smooth" intermediate samples to avoid collapse (low variance) or divergence (high variance).
    vpsde = "vpsde"


@dataclass
class InterpolantTranslationsConfig:
    # corrupt translations (unless inverse folding)
    corrupt: bool = (
        "${ternary:${equals: ${inference.task}, 'inverse_folding'}, False, True}"
    )
    # batch_ot: enable minibatch optimal transport AND importantly, centering
    batch_ot: bool = True
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
    # sigma scaled by g * sqrt(t * (1-t))
    stochastic_noise_intensity: float = 0.1  # `g` in FoldFlow SO3SFM


class InterpolantAATypesScheduleEnum(StrEnum):
    linear = "linear"
    exp = "exp"  # TODO re-introduce, 'exp' not used in public MultiFlow code


class InterpolantAATypesInterpolantTypeEnum(StrEnum):
    masking = "masking"
    uniform = "uniform"


@dataclass
class InterpolantAATypesConfig:
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
    train_extra_mask: float = 0.0  # TODO reintroduce or DROP
    # TODO - extra stochastic option, in addition to purity


@dataclass
class InterpolantSamplingConfig:
    # training takes a random t. Sampling runs over t timestemps.
    num_timesteps: int = 500


@dataclass
class InterpolantConfig:
    min_t: float = 1e-2
    # kappa allows scaling rotation t exponentially during sampling
    provide_kappa: bool = True
    hierarchical_t: bool = False
    separate_t: bool = False  # TODO drop, unused / refactor into rots/trans/aa
    # Use separate t times for rots / trans / aatypes in batch corruption
    # Effectively sets some proportion of the batch to be forward_folding / inverse_folding
    codesign_separate_t: bool = (
        "${ternary:${equals: ${inference.task}, 'unconditional'}, False, True}"
    )
    # proportion of samples allocated to forward or inverse folding, if using codesign_separate_t
    codesign_forward_fold_prop: float = 0.1
    codesign_inverse_fold_prop: float = 0.1
    # enable self-conditioning
    self_condition: bool = "${model.edge_features.self_condition}"
    self_condition_prob: float = 0.5  # 0.5 in public MultiFlow

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
class DataLoaderConfig:
    num_workers: int = 8
    prefetch_factor: int = 10


@dataclass
class DataSamplerConfig:
    # Setting for 40GB GPUs
    max_batch_size: int = 80  # 128 for 80GB GPUs
    max_num_res_squared: int = 400_000  # 1_000_000 for 80GB GPUs


@dataclass
class DataConfig:
    task: DataTaskEnum = DataTaskEnum.hallucination
    dataset: DatasetEnum = DatasetEnum.pdb
    loader: DataLoaderConfig = field(default_factory=DataLoaderConfig)
    sampler: DataSamplerConfig = field(default_factory=DataSamplerConfig)


@dataclass
class DatasetFilterConfig:
    max_num_res: int = 384
    min_num_res: int = 60
    max_coil_percent: float = 0.667  # was 0.5 in public MultiFlow
    min_num_confident_plddt: float = 40  # TODO implement
    # TODO - support filter on low pLDDT percentage
    # TODO - min/max motif percent threshold (avoid loopy things)
    rog_quantile: float = 0.96
    oligomeric: List[str] = field(
        default_factory=lambda: [
            "monomer",
            "monomeric",
            "monomeric,monomeric",
            "homomer",
        ]
    )
    num_chains: List[int] = field(default_factory=lambda: [1])


@dataclass
class DatasetInpaintingConfig:
    """
    Configuration for generating motifs / scaffolding
    """

    # % of time unconditional, i.e. not motif selected. 0% in FoldFlow.
    unconditional_percent: float = 0.2
    # number of possible motifs
    min_num_motifs: int = 1
    max_num_motifs: int = 5
    # fraction of residues to be in motif
    min_percent_motifs: float = 0.05
    max_percent_motifs: float = 0.50
    # motif length bounds
    min_motif_len: int = 8
    max_motif_len: int = 768
    # minimal spacing between motifs (i.e. min scaffold length)
    min_padding: int = 3


# dataset_metadata_dir_path is the root directory for dataset / metadata files
dataset_metadata_dir_path = PATH_PROJECT_ROOT / "cogeneration" / "datasets" / "metadata"


@dataclass
class DatasetConfig:
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
    # add gaussian noise to atom positions
    # TODO cfg to only add noise if t below some threshold (requires moving out of dataset)
    # TODO ensure noise added each time accessed and not cached
    noise_atom_positions_angstroms: float = 0.1

    # Redesigned, i.e. use ProteinMPNN to generate sequences for a structure
    use_redesigned: bool = True
    redesigned_csv_path: Optional[Path] = (
        dataset_metadata_dir_path / "pdb_redesigned.csv"
    )

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
class ExperimentTrainingConfig:
    mask_plddt: bool = True
    bb_atom_scale: float = 0.1
    trans_scale: float = 0.1
    aatypes_loss_weight: float = 0.25  # default 0.0 in multiflow
    aatypes_label_smoothing: float = 0.0
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
class ExperimentWandbConfig:
    """W&B configuration. Some properties are kwargs to logger"""

    name: str = "${data.task}_${data.dataset}_${shared.now}"
    project: str = "cogeneration"
    # offline if `local` (not copied to W&B)
    offline: bool = "${ternary:${equals: ${shared.local}, True}, True, False}"


@dataclass
class ExperimentOptimizerConfig:
    lr: float = 1e-4


@dataclass
class ExperimentTrainerConfig:
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
class ExperimentCheckpointerConfig:
    """Arguments to ModelCheckpoint()"""

    dirpath: str = (
        "ckpt/${experiment.wandb.project}/${experiment.wandb.name}/${shared.now}"
    )
    save_last: bool = True
    save_top_k: int = 3
    # recommend checkpoint on some multiple of validation epoch interval. too frequent will eat up disk.
    every_n_epochs: int = 8
    monitor: str = "valid/codesign_bb_rmsd"
    mode: str = "min"


@dataclass
class ExperimentConfig:
    """Training Experiment configuration."""

    seed: int = "${shared.seed}"
    # debug True for local tensorboard logger, False to enable W&B, saving outputs etc
    debug: bool = False
    # num GPU devices TODO support more than one, esp for GPUs
    num_devices: int = 1
    # directory, checkpoint path
    warm_start: Optional[str] = None
    # override config with warm start config
    warm_start_cfg_override: bool = False
    # force reload module config, provide path
    raw_state_dict_reload: Optional[str] = None
    # pytorch Trainer profiler, "simple", "advanced", None
    profiler: Optional[str] = None
    # enable torch.compile(), requires no graph breaks TODO make it work
    torch_compile: bool = False

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
class InferenceSamplesConfig:
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


@dataclass
class InferenceConfig:
    task: InferenceTaskEnum = InferenceTaskEnum.unconditional

    seed: int = "${shared.seed}"
    use_gpu: bool = True
    num_gpus: int = 1
    predict_dir: str = str(PATH_PROJECT_ROOT / "inference_outputs")
    inference_subdir: str = "${shared.now}"

    # checkpoints
    saved_ckpt_dir: str = "${shared.project_root}/ckpt/${experiment.wandb.project}"

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
    # whether to also save the trajectory of the generation process
    write_sample_trajectories: bool = False


@dataclass
class FoldingConfig:
    seq_per_sample: int = 8
    folding_model: str = "af2"  # "af2" only at the moment, maybe "esm" in the future
    # dedicated device for folding. decrement other devices by 1 if True
    own_device: bool = False
    # TODO update ProteinMPNN path
    pmpnn_path: Path = PATH_PROJECT_ROOT / "ProteinMPNN"
    pmpnn_seed: int = "${shared.seed}"
    pt_hub_dir: Path = PATH_PROJECT_ROOT / "cache" / "torch"
    # TODO update colabfold path
    colabfold_path: Path = PATH_PROJECT_ROOT / "colabfold/bin/colabfold_batch"


@dataclass
class Config:
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

    def interpolate(self) -> "Config":
        """
        Interpolate Config with OmegaConf, yielding an interpolated Config class

        OmegaConf supports interpolated fields like `my_field: int = "${my_other_field}"`.
        This function interpolates those fields, yielding a new Config object.

        Intermediate fields are dataclasses, not DictConfig.
        """
        # use `to_object()` so intermediate fields are dataclasses, not DictConfig
        # `create()` interpolates the fields
        # TODO consider using `hydra.instantiate` to interpolate the config?
        return OmegaConf.to_object(OmegaConf.create(self))

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

        return raw_cfg

    @classmethod
    def public_multiflow(cls):
        """
        Returns a config that maintains compatibility with Public MultiFlow weights + better aligned with its defaults.
        Requires running `interpolate()`, to allow for further modification.
        """
        raw_cfg = cls()

        # Default to unconditional generation
        raw_cfg.data.task = DataTaskEnum.hallucination
        raw_cfg.inference.task = InferenceTaskEnum.unconditional

        # Model
        # Use public MultiFlow model size hyperparameters
        raw_cfg.model.hyper_params = ModelHyperParamsConfig.public_multiflow()
        # Use simple aa_pred_net from public MultiFlow
        raw_cfg.model.sequence_pred_type = ModelSequencePredictionEnum.aa_pred
        # stochastic paths not part of public MultiFlow
        raw_cfg.shared.stochastic = False
        # Don't predict torsion angles
        raw_cfg.model.predict_psi_torsions = False
        # Other parameters as defined by public codebase
        raw_cfg.model.node_features.embed_chain = False
        raw_cfg.model.edge_features.embed_chain = False

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
