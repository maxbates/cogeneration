import datetime
import math
import os
import re
from collections import OrderedDict
from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf, SCMode

from cogeneration.config.base import (
    AttentionType,
    BaseClassConfig,
    DataConfig,
    DatasetConfig,
    DatasetContactConditioningConfig,
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
    SharedConfig,
)
from cogeneration.type.str_enum import StrEnum


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


class VarcoMotifGuidanceType(StrEnum):
    """scale function for motif guidance"""

    posterior_variance = "posterior_variance"
    linear_decay = "linear_decay"


class VarcoHazardKind(StrEnum):
    """Time-bias hazard CDF family for counting-process events."""

    uniform = "uniform"
    late_power = "late_power"
    early_power = "early_power"


@dataclass
class VarcoHazardConfig(BaseClassConfig):
    """Hazard CDF H(t) specified as a simple closed-form family on [0, 1]."""

    kind: VarcoHazardKind = VarcoHazardKind.uniform
    power: int = 1


@dataclass
class VarcoInterpolantMotifGuidanceConfig(BaseClassConfig):
    """Configuration for motif position guidance during sampling."""

    enabled: bool = True
    # Scale function
    scale_type: VarcoMotifGuidanceType = VarcoMotifGuidanceType.posterior_variance
    # For posterior_variance scale: max clamp value (for close to t=0)
    var_scale_cap: float = 10.0
    # For linear_decay scale: strength multiplier
    linear_decay_strength: float = 1.0
    # Per-step translation force cap (angstroms) - prevents huge single-step jumps
    max_step_force_ang: float = 5.0
    # Per-step rotation force cap (radians) - prevents huge single-step rotations
    max_rot_step_force_rad: float = math.pi / 2


@dataclass
class VarcoInterpolantCouplerConfig(BaseClassConfig):
    """Shared base configuration for interpolant couplers."""

    # Sigma for stochastic bridge noise (0 for deterministic)
    noise_scale: float = 1.0
    # Noise is forced to 0 for t >= noise_end_t in sampling steps.
    noise_end_t: float = 0.95


@dataclass
class VarcoInterpolantTransCouplerConfig(VarcoInterpolantCouplerConfig):
    """Configuration for translation coupler (Brownian bridge)."""

    # Cap per-residue translation drift step (angstroms) during sampling.
    drift_step_cap_ang: float = 5.0


@dataclass
class VarcoInterpolantAATypesCouplerConfig(VarcoInterpolantCouplerConfig):
    """Configuration for amino acid types coupler (CTMC bridge)."""

    # Total leaving rate for the CTMC
    beta: float = 3.0
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
    exp_rate: float = 1.5


@dataclass
class VarcoInterpolantSamplingConfig(BaseClassConfig):
    """Configuration for Varco sampling behavior."""

    # Maximum sequence length during sampling (to prevent GPU OOM)
    # If exceeded, further insertions are blocked
    max_length: int = 512
    # Hazard distributions controlling insertion (split) and deletion time bias during sampling.
    # Defaults match TreePlan.generate() defaults: splits early (Beta(1,2)), deletions late (Beta(2,1)).
    split_hazard: VarcoHazardConfig = field(
        default_factory=lambda: VarcoHazardConfig(
            kind=VarcoHazardKind.early_power, power=2
        )
    )
    delete_hazard: VarcoHazardConfig = field(
        default_factory=lambda: VarcoHazardConfig(
            kind=VarcoHazardKind.late_power, power=2
        )
    )


@dataclass
class VarcoInterpolantConfig(BaseClassConfig):
    """Configuration for Varco interpolant / sampling behavior."""

    sigma: float = 1.0  # 0 for deterministic bridges (legacy, prefer coupler configs)

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
    t_normalize_clip: float = 0.9
    # Local pairwise distance threshold (angstroms)
    proximity_threshold_ang: float = 7.0

    # Loss weights
    trans_loss_weight: float = 2.0
    pairwise_dist_loss_weight: float = 0.2
    rot_vf_loss_weight: float = 1.0
    seq_loss_weight: float = 1.0
    seq_prob_loss_weight: float = 1.0  # anchor probs
    seq_token_loss_weight: float = 0.5  # sampled anchor token
    seq_ins_loss_weight: float = 0.2
    split_loss_weight: float = 0.2
    split_pooled_loss_weight: float = 0.05
    del_loss_weight: float = 0.35
    bfactor_loss_weight: float = 0.02
    plddt_loss_weight: float = 0.2


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
