import math
from dataclasses import dataclass, field
from typing import Optional

from hydra.core.config_store import ConfigStore

from cogeneration.config.base import (
    AttentionType,
    BaseClassConfig,
    DataConfig,
    DatasetConfig,
    DatasetContactConditioningConfig,
    DatasetInpaintingConfig,
    DatasetInpaintingMotifStrategy,
    ExperimentConfig,
    ExperimentWandbConfig,
    FoldingConfig,
    ModelAAPredConfig,
    ModelAttentionConfig,
    ModelAttentionTrunkConfig,
    ModelBFactorConfig,
    ModelContactConditioningConfig,
    ModelEdgeFeaturesConfig,
    ModelESMCombinerConfig,
    ModelESMKey,
    ModelHyperParamsConfig,
    ModelIPAConfig,
    ModelPLDDTConfig,
    MotifGuidanceConfig,
    MotifGuidanceVarScale,
    SharedConfig,
)
from cogeneration.type.str_enum import StrEnum


@dataclass
class VarcoTreePlanConfig(BaseClassConfig):
    """
    Configuration for TreePlan generation during training.
    These parameters control when indel events occur in training trajectories.
    """

    # Beta distribution for split times (alpha < beta = early bias).
    split_time_beta_alpha: float = 1.0
    split_time_beta_beta: float = "${interpolant.sampling.split_hazard.power}"
    # Beta distribution for delete times
    delete_time_beta_alpha: float = 1.0
    delete_time_beta_beta: float = 1.0
    # Maximum time for indel events
    max_indel_time: float = 0.85
    # Scaffold nuclei sampling range per span
    min_scaffold_nuclei: int = 1
    max_scaffold_nuclei: int = 10
    # Probability of deleting scaffold positions (Poisson rate multiplier)
    p_deletion: float = 0.20


@dataclass
class VarcoDatasetConfig(DatasetConfig):
    # debug_head_samples: int = 1000  # faster startup
    enable_cogeneration_pdb: bool = True
    enable_cogeneration_afdb: bool = True
    enable_cogeneration_redesigns: bool = True
    enable_multiflow_redesigned: bool = True
    enable_multiflow_synthetic: bool = True
    contact_conditioning: DatasetContactConditioningConfig = field(
        default_factory=lambda: DatasetContactConditioningConfig(
            conditioning_prob_disabled=0.5,
            conditioning_prob_motif_only=1.0,
            dist_noise_ang=0.25,
        )
    )
    inpainting: DatasetInpaintingConfig = field(default_factory=lambda: DatasetInpaintingConfig(
        strategy=DatasetInpaintingMotifStrategy.single_motif,
        min_percent_motifs=0.40,
        max_percent_motifs=0.80,
    ))
    tree_plan: VarcoTreePlanConfig = field(default_factory=VarcoTreePlanConfig)


# alias for ckpt, deprecated
VarcoMotifVarScale = MotifGuidanceVarScale


# deprecated. use InterpolantMotifGuidanceConfig instead
class VarcoMotifGuidanceType(StrEnum):
    """Scale function for motif guidance (deprecated, use MotifGuidanceVarScale)."""

    posterior_variance = "posterior_variance"
    linear_decay = "linear_decay"


# deprecated. use InterpolantMotifGuidanceConfig instead
@dataclass
class VarcoInterpolantMotifGuidanceConfig(MotifGuidanceConfig):
    """Varco-specific motif guidance config (extends shared MotifGuidanceConfig).

    Includes deprecated fields for backward compatibility with existing configs.
    """

    # For linear_decay scale: strength multiplier
    linear_decay_strength: float = 1.0
    # Scale function (deprecated, use var_scale_type from MotifGuidanceConfig)
    scale_type: VarcoMotifGuidanceType = VarcoMotifGuidanceType.posterior_variance
    # Per-step translation force cap (angstroms) - deprecated
    max_step_force_ang: float = 10.0
    # Per-step rotation force cap (radians) - deprecated
    max_rot_step_force_rad: float = math.pi


@dataclass
class VarcoInterpolantCouplerConfig(BaseClassConfig):
    """Shared base configuration for interpolant couplers."""

    # Sigma for stochastic bridge noise (0 for deterministic)
    noise_scale: float = 1.0
    # Noise is forced to 0 for t >= noise_end_t
    noise_end_t: float = 1.0


@dataclass
class VarcoInterpolantTransCouplerConfig(VarcoInterpolantCouplerConfig):
    """Configuration for translation coupler (Brownian bridge)."""

    # Cap per-residue translation drift step (angstroms) during sampling.
    drift_step_cap_ang: float = 5.0


@dataclass
class VarcoInterpolantAATypesCouplerConfig(VarcoInterpolantCouplerConfig):
    """Configuration for amino acid types coupler (CTMC bridge)."""

    # Total leaving rate for the CTMC
    beta: float = 2.0
    # Temperature for softmax in euler_step (lower = sharper)
    drift_temp: float = 1.0
    # Cap for the 1/(1-t) drift gain used in sampling.
    drift_gain_cap: float = 50.0
    # Exponent for uncertainty gating
    uncertainty_sharpness: float = 1.0
    # Maximum total off-diagonal probability per step
    leave_mass_cap: float = 0.2


@dataclass
class VarcoInterpolantRotationCouplerConfig(VarcoInterpolantCouplerConfig):
    """Configuration for rotation coupler (SO(3) geodesic bridge with IGSO3 noise)."""

    # IGSO3 sigma range for sampling
    igso3_sigma_min: float = 1e-4
    igso3_sigma_max: float = 1.5
    # Cap per-residue rotation drift step (radians) during sampling.
    drift_step_cap_rad: float = math.pi / 3
    # Exponential schedule rate (higher = faster settling)
    exp_rate: float = 3


class VarcoHazardKind(StrEnum):
    """Time-bias hazard CDF family for counting-process events."""

    uniform = "uniform"
    late_power = "late_power"
    early_power = "early_power"


@dataclass
class VarcoHazardConfig(BaseClassConfig):
    """Hazard CDF H(t) specified as a simple closed-form family on [0, 1]."""

    kind: VarcoHazardKind = VarcoHazardKind.uniform
    power: float = 1


@dataclass
class VarcoInterpolantSamplingConfig(BaseClassConfig):
    """Configuration for Varco sampling behavior."""

    # Maximum sequence length during sampling (to prevent GPU OOM)
    # If exceeded, further insertions are blocked
    max_length: int = 512
    # Hazard distributions controlling insertion (split) and deletion time bias during sampling.
    # TreePlan.generate() defaults: splits early (Beta(1,2)), deletions late (Beta(2,1))
    split_hazard: VarcoHazardConfig = field(
        default_factory=lambda: VarcoHazardConfig(
            kind=VarcoHazardKind.early_power, power=2.5
        )
    )
    delete_hazard: VarcoHazardConfig = field(
        default_factory=lambda: VarcoHazardConfig(
            kind=VarcoHazardKind.uniform, power=1.0
        )
    )
    # Sharpening exponent for indel probabilities. Applied as p^gamma before sampling.
    # 1.0 uses model directly. Higher values suppress low-confidence predictions.
    indel_sharpness: float = 1.0
    # Optional time-dependent sharpening schedule
    # Use this to smoothly suppress late indels.  Note that this will not shift them earlier!
    # gamma(t) = indel_sharpness + indel_sharpness_late_delta * ramp(t)^power
    # where ramp(t) = clip((t - indel_sharpness_late_start_t) / (1 - start), 0, 1).
    indel_sharpness_late_start_t: float = 0.75  # 1.0 disables schedule
    indel_sharpness_late_delta: float = 2.0  # extra sharpness by t=1
    indel_sharpness_late_power: float = 2.0  # curvature of the ramp, > 1.0


@dataclass
class VarcoInterpolantConfig(BaseClassConfig):
    """Configuration for Varco interpolant / sampling behavior."""

    sigma: float = 1.0  # 0 for deterministic bridges (legacy, prefer coupler configs)
    t_corrupt_exp: float = 1.0  # < 1.0 -> bias later times in corrupt_batch()

    trans_coupler: VarcoInterpolantTransCouplerConfig = field(
        default_factory=VarcoInterpolantTransCouplerConfig
    )
    aatypes_coupler: VarcoInterpolantAATypesCouplerConfig = field(
        default_factory=VarcoInterpolantAATypesCouplerConfig
    )
    rotation_coupler: VarcoInterpolantRotationCouplerConfig = field(
        default_factory=VarcoInterpolantRotationCouplerConfig
    )
    motif_guidance: VarcoInterpolantMotifGuidanceConfig = field(
        default_factory=VarcoInterpolantMotifGuidanceConfig
    )
    sampling: VarcoInterpolantSamplingConfig = field(
        default_factory=VarcoInterpolantSamplingConfig
    )


@dataclass
class VarcoLossConfig(BaseClassConfig):
    """Configuration for Varco loss weights and parameters."""

    # Time normalization clip (higher weight as t -> 1)
    t_normalize_clip: float = 0.8
    # Exponent to smooth time norm `(1 - t_clip) ** exp`
    # (<1 -> flatter, >1 -> t->1 weight stronger)
    t_normalize_exponent: float = 1.0
    # Local pairwise distance threshold (angstroms)
    proximity_threshold_ang: float = 7.0

    # Model predicts time-independent mass M; target is remaining_insertions / S(t).
    # This should match the split_hazard used in sampling.
    split_hazard: VarcoHazardConfig = field(
        default_factory=lambda: VarcoHazardConfig(
            kind="${interpolant.sampling.split_hazard.kind}",
            power="${interpolant.sampling.split_hazard.power}",
        )
    )

    # Loss weights
    trans_loss_weight: float = 2.0
    pairwise_dist_loss_weight: float = 0.2
    rot_vf_loss_weight: float = 1.0
    seq_loss_weight: float = 1.0
    seq_prob_loss_weight: float = 1.0  # anchor probs
    seq_token_loss_weight: float = 0.5  # sampled anchor token
    seq_ins_loss_weight: float = 0.2
    split_loss_weight: float = 0.3
    split_mass_entropy_weight: float = 0.1
    split_mass_l2_weight: float = 0.1
    split_pooled_loss_weight: float = 0.2
    del_loss_weight: float = 0.5
    bfactor_loss_weight: float = 0.02
    plddt_loss_weight: float = 0.25


class VarcoPlotColorBy(StrEnum):
    auto = "auto"
    position = "position"
    sequence = "sequence"


@dataclass
class VarcoInferencePlotConfig(BaseClassConfig):
    enabled: bool = True
    max_frames: Optional[int] = 50
    max_samples: int = 2
    max_cols: int = 2
    only_alpha_carbons: bool = True
    show_residue_letters: bool = True
    color_by: VarcoPlotColorBy = VarcoPlotColorBy.auto


@dataclass
class VarcoFoldingValidationConfig(BaseClassConfig):
    enabled: bool = False
    # Fold predicted sequence with Boltz, or designability with ProteinMPNN -> Boltz
    assess_designability: bool = False
    # max validation batches per epoch (should be reduced by eval dataset)
    max_batches: int = 50


@dataclass
class VarcoInferenceConfig(BaseClassConfig):
    predict_dir: str = "varco/outputs/${shared.id}"
    ckpt_path: Optional[str] = None
    inference_subdir: str = "predict"
    plot: VarcoInferencePlotConfig = field(default_factory=VarcoInferencePlotConfig)
    folding_validation: VarcoFoldingValidationConfig = field(
        default_factory=VarcoFoldingValidationConfig
    )


@dataclass
class VarcoExperimentWandbConfig(ExperimentWandbConfig):
    project: str = "varco"


@dataclass
class VarcoExperimentConfig(ExperimentConfig):
    wandb: ExperimentWandbConfig = field(default_factory=VarcoExperimentWandbConfig)

    # Path to warm start with cogeneration checkpoint (must be compatible shape)
    cogen_ckpt_path: Optional[str] = None

    # Exponential moving average coefficient for `L/train_ema` (higher -> smoother)
    train_loss_ema_beta: float = 0.98


@dataclass
class VarcoModelConfig(BaseClassConfig):
    hyper_params: ModelHyperParamsConfig = field(default_factory=ModelHyperParamsConfig)
    edge_features: ModelEdgeFeaturesConfig = field(
        default_factory=lambda: ModelEdgeFeaturesConfig(
            embed_self_condition=False,
            embed_chain=True,
            embed_diffuse_mask=True,  # motif mask
            contact_conditioning=ModelContactConditioningConfig(
                enabled=True,
            ),
        )
    )
    esm_combiner: ModelESMCombinerConfig = field(
        default_factory=lambda: ModelESMCombinerConfig(
            enabled=True,
            esm_model_key=ModelESMKey.esm2_t12_35M_UR50D,
            only_single=True,
        )
    )
    attention: ModelAttentionConfig = field(default_factory=ModelAttentionConfig)
    trunk: ModelAttentionTrunkConfig = field(
        default_factory=lambda: ModelAttentionTrunkConfig(
            attn_type=AttentionType.PAIRFORMER,
            num_layers="${model.hyper_params.trunk_num_layers}",
        )
    )
    ipa: ModelIPAConfig = field(default_factory=ModelIPAConfig)
    seq_trunk: ModelAttentionTrunkConfig = field(
        default_factory=lambda: ModelAttentionTrunkConfig(
            attn_type=AttentionType.PAIRFORMER,
            num_layers="${model.hyper_params.seq_trunk_num_layers}",
            # merge back in time / positional embeddings, per FoldFlow-2.
            pre_add_init_embed="${ternary:${greater_than: ${model.hyper_params.seq_trunk_num_layers}, 0}, True, False}",
            # skip layer norm and just run trunk.
            pre_node_layer_norm=False,
            pre_edge_layer_norm=False,
        )
    )
    # aa_pred config used for base and insertion logits
    aa_pred: ModelAAPredConfig = field(default_factory=ModelAAPredConfig)
    bfactor: ModelBFactorConfig = field(default_factory=ModelBFactorConfig)
    plddt: ModelPLDDTConfig = field(default_factory=ModelPLDDTConfig)


@dataclass
class VarcoConfig(BaseClassConfig):
    shared: SharedConfig = field(default_factory=SharedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataset: VarcoDatasetConfig = field(default_factory=VarcoDatasetConfig)
    experiment: VarcoExperimentConfig = field(default_factory=VarcoExperimentConfig)
    inference: VarcoInferenceConfig = field(default_factory=VarcoInferenceConfig)
    interpolant: VarcoInterpolantConfig = field(default_factory=VarcoInterpolantConfig)
    loss: VarcoLossConfig = field(default_factory=VarcoLossConfig)
    model: VarcoModelConfig = field(default_factory=VarcoModelConfig)
    folding: FoldingConfig = field(default_factory=FoldingConfig)


# Register the config class with Hydra
cs = ConfigStore.instance()
cs.store(name="varco", node=VarcoConfig)
