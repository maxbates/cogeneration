import datetime
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import List, Optional

import hydra
from hydra.core.config_store import ConfigStore

"""
Note:

Structured configurations for cogeneration.

Several parameters from Multiflow appear to be unused, many are marked with `# DROP`.
"""

# TODO - support default configs for `forward_folding` and `inverse folding`
#   There are a handful of values across the config that need to have other defaults.
#   i.e. instead of the current `ternary` paradigm


class ConfEnum(str, Enum):
    """
    Base class for OmegaConf/Hydra Enums.
    hydra does not currently support `Literal` and suggests using Enums instead.
    https://github.com/omry/omegaconf/issues/422
    """

    def __str__(self):
        return str(self.value)


class DataTaskEnum(ConfEnum):
    """task for training"""

    hallucination = "hallucination"


class InferenceTaskEnum(ConfEnum):
    """task for inference"""

    unconditional = "unconditional"
    forward_folding = "forward_folding"
    inverse_folding = "inverse_folding"


class DatasetEnum(ConfEnum):
    """dataset for training"""

    pdb = "pdb"


@dataclass
class MetadataConfig:
    """shared metadata"""

    now: str = field(
        default_factory=lambda: datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    )


@dataclass
class ModelHyperParamsConfig:
    """
    Shared hyperparameters for the model.
    Use a structured config for hyperparameters so easy to reference in templates.
    Default values match those from Multiflow. Use factory methods for different configurations.

    TODO centralize other hyperparameters here, like transformer depth, heads, etc.

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
            node_embed_size=64,
            edge_embed_size=32,
            pos_embed_size=32,
            timestep_embed_size=32,
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

    @classmethod
    def esmc_300m(cls):
        """Factory for configuration compatible with ESM-C 300M"""
        return cls(
            node_embed_size=960,
            edge_embed_size=256,
            pos_embed_size=128,
            timestep_embed_size=128,
        )

    @classmethod
    def esmc_600m(cls):
        """Factory for configuration compatible with ESM-C 600M"""
        return cls(
            node_embed_size=1152,
            edge_embed_size=384,
            pos_embed_size=192,
            timestep_embed_size=192,
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
    timestep_int: int = 1000  # DROP
    # embed_chain: whether to embed chain index. TODO True when support multimers
    embed_chain: bool = False
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
    relpos_k: int = (
        64  # DROP. duplicate of `feat_dim`? Appears to be carry over from AF2?
    )
    # feat_dim: partial internal dimension (actual width is multiple)
    feat_dim: int = 64
    # num_bins: number of bins for edge distrogram
    num_bins: int = 22
    # self_condition: used by interpolant config. TODO determine why defined here?
    self_condition: bool = True
    # embed_chain: whether to embed chain index. TODO True when support multimers
    embed_chain: bool = False
    # embed_diffuse_mask: whether to embed diffuse mask
    embed_diffuse_mask: bool = True


@dataclass
class ModelAAPredConfig:
    """
    Amino acid prediction configuration.
    """

    # aatype_pred: whether to predict amino acid types, i.e. enable this network
    aatype_pred: bool = True
    # c_s: node embedding size (input)
    c_s: int = "${model.hyper_params.node_embed_size}"
    # aatype_pred_num_tokens: number of amino acid types => logits / rate-matrix shape
    aatype_pred_num_tokens: int = "${model.hyper_params.aa_num_tokens}"


@dataclass
class ModelIPAConfig:
    """
    Invariant Point Attention configuration.
    Keys match ported OpenFold's IPA module.
    """

    # IPA parameters
    c_s: int = "${model.hyper_params.node_embed_size}"
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
    aa_pred: ModelAAPredConfig = field(default_factory=ModelAAPredConfig)


class InterpolantRotationsScheduleEnum(str, Enum):
    linear = "linear"
    exp = "exp"


@dataclass
class InterpolantRotationsConfig:
    corrupt: bool = True
    train_schedule: InterpolantRotationsScheduleEnum = (
        InterpolantRotationsScheduleEnum.linear
    )
    sample_schedule: InterpolantRotationsScheduleEnum = (
        InterpolantRotationsScheduleEnum.exp
    )
    exp_rate: float = 10


class InterpolantTranslationsScheduleEnum(str, Enum):
    linear = "linear"
    vpsde = "vpsde"  # variance-preserving SDE


@dataclass
class InterpolantTranslationsConfig:
    # corrupt: corrupt translations
    corrupt: bool = True
    # batch_ot: enable minibatch optimal transport
    batch_ot: bool = True
    # train_schedule: training schedule for interpolant
    train_schedule: InterpolantTranslationsScheduleEnum = (
        InterpolantTranslationsScheduleEnum.linear
    )
    # sample_schedule: sampling schedule for interpolant
    sample_schedule: InterpolantTranslationsScheduleEnum = (
        InterpolantTranslationsScheduleEnum.linear
    )
    # sample_temp: sampling temperature
    sample_temp: float = 1.0
    # vpsde_bmin: variance-preserving SDE minimum
    vpsde_bmin: float = 0.1
    # vpsde_bmax: variance-preserving SDE maximum
    vpsde_bmax: float = 20.0
    # potentials and radius of gyration not used in public multiflow code
    # potential: str = 'null'
    # potential_t_scaling: bool = False
    # rog:
    #   weight: 10.0
    #   cutoff: 5.0


class InterpolantAATypesScheduleEnum(str, Enum):
    linear = "linear"
    exp = "exp"  # TODO re-introduce, 'exp' not used in public MultiFlow code


class InterpolantAATypesInterpolantTypeEnum(str, Enum):
    masking = "masking"
    uniform = "uniform"


@dataclass
class InterpolantAATypesConfig:
    # corrupt: corrupt amino acid types
    corrupt: bool = True
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
    # noise: AA type change noise
    noise: float = (
        "${ternary:${equals: ${inference.task}, 'forward_folding'}, 0.0, 20.0}"
    )
    # do_purity: enable purity, allows for unmasking by max log probs and for re-masking by `noise`
    # purity requires masking interpolant
    do_purity: bool = (
        "${ternary:${equals: ${inference.task}, 'forward_folding'}, False, True}"
    )
    train_extra_mask: float = 0.0  # DROP


@dataclass
class InterpolantSamplingConfig:
    # training takes a random t. Sampling runs over t timestemps.
    num_timesteps: int = 500
    # SDE not in public multiflow code
    do_sde: bool = False


@dataclass
class InterpolantConfig:
    min_t: float = 1e-2
    separate_t: bool = False
    provide_kappa: bool = False
    hierarchical_t: bool = False
    codesign_separate_t: bool = (
        "${ternary:${equals: ${inference.task}, 'unconditional'}, False, True}"
    )
    codesign_forward_fold_prop: float = 0.1
    codesign_inverse_fold_prop: float = 0.1
    self_condition: bool = "${model.edge_features.self_condition}"
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
    max_coil_percent: float = 0.5
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


# metadata_dir is the root directory for dataset / metadata files
metadata_dir_path = Path(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "datasets/metadata"))
)


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
    """

    seed: int = 123
    processed_data_path: str = os.path.dirname(metadata_dir_path)
    csv_path: Path = metadata_dir_path / "pdb_metadata.csv"
    cluster_path: Optional[Path] = metadata_dir_path / "pdb.clusters"
    max_cache_size: int = 100_000
    cache_num_res: int = 0  # min size to enable caching
    inpainting_percent: float = 1.0
    add_plddt_mask: bool = False
    min_plddt_threshold: float = (
        0.0  # [0, 100]. Minimum threshold, masked if below and add_plddt_mask=True
    )

    # Redesigned, i.e. use ProteinMPNN to generate sequences for a structure
    use_redesigned: bool = True
    redesigned_csv_path: Optional[Path] = metadata_dir_path / "pdb_redesigned.csv"

    # Synthetic, e.g. AlphaFold structures?
    use_synthetic: bool = True
    synthetic_csv_path: Optional[Path] = metadata_dir_path / "distillation_metadata.csv"
    synthetic_cluster_path: Optional[Path] = metadata_dir_path / "distillation.clusters"

    # Eval parameters
    test_set_pdb_ids_path: Optional[Path] = None
    max_eval_length: int = 256
    samples_per_eval_length: int = 5
    num_eval_lengths: int = 8

    # Filtering
    filter: DatasetFilterConfig = field(default_factory=DatasetFilterConfig)

    @classmethod
    def PDBPost2021(cls):
        return cls(
            csv_path=metadata_dir_path / "test_set_metadata.csv",
            cluster_path=metadata_dir_path / "test_set_clusters.csv",
            cache_num_res=0,
            add_plddt_mask=False,
            # disable Redesigned and Synthetic for test set
            use_redesigned=False,
            redesigned_csv_path=None,
            use_synthetic=False,
            synthetic_csv_path=None,
            synthetic_cluster_path=None,
            # Eval parameters
            test_set_pdb_ids_path=metadata_dir_path / "test_set_pdb_ids.csv",
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
    t_normalize_clip: float = 0.9
    rotation_loss_weights: float = 1.0
    aux_loss_weight: float = 0.25  # default 0.0 in multiflow
    aux_loss_use_bb_loss: bool = True
    aux_loss_use_pair_loss: bool = True
    aux_loss_t_pass: float = 0.5


@dataclass
class ExperimentWandbConfig:
    """W&B configuration. Some properties are kwargs to logger"""

    name: str = "${data.task}_${data.dataset}_${metadata.now}"
    offline: bool = False
    project: str = "cogeneration"


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
    # TODO - default to GPU, with flagset for local development, and ddp
    accelerator: str = "gpu"
    # `strategy` is argument name to Trainer(), ddp = distributed data parallel
    strategy: Optional[str] = "ddp"
    overfit_batches: int = 0
    min_epochs: int = 1  # prevents early stopping
    max_epochs: int = 200
    deterministic: bool = False
    check_val_every_n_epoch: int = 4
    accumulate_grad_batches: int = 2
    # logging
    log_every_n_steps: int = 1
    # if experiment.debug, use tensorboard logger
    tensorboard_logdir: Path = Path("./lightning_logs/")

    def __post_init__(self):
        # distributed training (ddp) not currently supported with MPS
        if self.accelerator == "mps" and self.strategy is not None:
            self.strategy = "auto"


@dataclass
class ExperimentCheckpointerConfig:
    dirpath: str = (
        "ckpt/${experiment.wandb.project}/${experiment.wandb.name}/${metadata.now}"
    )
    save_last: bool = True
    save_top_k: int = 3
    every_n_epochs: int = 50
    monitor: str = "valid/codesign_bb_rmsd"
    mode: str = "min"


@dataclass
class ExperimentConfig:
    """Training Experiment configuration."""

    debug: bool = (
        False  # True for local tensorboard logger, False to enable W&B, saving outputs etc
    )
    seed: int = 123
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
    inference_dir: Optional[str] = None


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
    # Subset of lengths to sample. If null, sample all targets.
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

    seed: int = 123
    use_gpu: bool = True
    num_gpus: int = 1
    predict_dir: Path = Path("./inference_outputs/")
    inference_subdir: str = "${metadata.now}"

    # checkpoints
    saved_ckpt_dir: str = "./ckpt/${experiment.wandb.project}"
    unconditional_ckpt_path: Optional[str] = "./weights/last.ckpt"
    forward_folding_ckpt_path: Optional[str] = "./weights/last.ckpt"
    inverse_folding_ckpt_path: Optional[str] = "./weights/last.ckpt"

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
    # validate using esmf or af2
    folding_model: str = "esmf"
    # dedicated device for folding. decrement other devices by 1 if True
    own_device: bool = False
    pmpnn_path: Path = Path("./ProteinMPNN/")
    pt_hub_dir: Path = Path("./.cache/torch/")
    colabfold_path: Path = Path(
        "path/to/colabfold-conda/bin/colabfold_batch"
    )  # for AF2


@dataclass
class Config:
    """
    We use dataclasses as part of hydra's structured config system, to enable type checking and default values.
    This is the base class and default config.
    """

    data: DataConfig = field(default_factory=DataConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    folding: FoldingConfig = field(default_factory=FoldingConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    interpolant: InterpolantConfig = field(default_factory=InterpolantConfig)
    metadata: MetadataConfig = field(default_factory=MetadataConfig)  # shared
    model: ModelConfig = field(default_factory=ModelConfig)


# Register the config class with Hydra
cs = ConfigStore.instance()
cs.store(name="base", node=Config)
