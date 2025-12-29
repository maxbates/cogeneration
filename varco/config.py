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
    enable_cogeneration_redesigns: bool = False
    enable_multiflow_redesigned: bool = False
    enable_multiflow_synthetic: bool = False

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
class VarcoInterpolantTransCouplerConfig(BaseClassConfig):
    """Configuration for translation coupler (Brownian bridge)."""

    # Sigma for Brownian bridge (0 for deterministic bridges)
    sigma: float = 1.0


@dataclass
class VarcoInterpolantAATypesCouplerConfig(BaseClassConfig):
    """Configuration for amino acid types coupler (CTMC bridge)."""

    # Scale for uniform noise added to step probabilities
    noise_scale: float = 1.0
    # Total leaving rate for the CTMC
    beta: float = 3.0
    # Temperature for softmax in euler_step (lower = sharper)
    drift_temp: float = 1.0
    # Exponent for uncertainty gating
    uncertainty_sharpness: float = 1.0
    # Maximum total off-diagonal probability per step
    leave_mass_cap: float = 0.25


@dataclass
class VarcoInterpolantRotationCouplerConfig(BaseClassConfig):
    """Configuration for rotation coupler (SO(3) geodesic bridge with IGSO3 noise)."""

    # Sigma for stochastic bridge noise (0 for deterministic geodesic interpolation)
    sigma: float = 1.0
    # IGSO3 sigma range for sampling
    igso3_sigma_min: float = 1e-4
    igso3_sigma_max: float = 1.5
    # Exponential schedule rate (higher = faster settling)
    exp_rate: float = 1.5


@dataclass
class VarcoInterpolantSamplingConfig(BaseClassConfig):
    """Configuration for Varco sampling behavior."""

    # Maximum sequence length during sampling (to prevent GPU OOM)
    # If exceeded, further insertions are blocked
    max_length: int = 512


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
    seq_prob_loss_weight: float = 1.0
    seq_token_loss_weight: float = 0.5
    seq_ins_loss_weight: float = 0.2
    split_loss_weight: float = 0.2
    split_pooled_loss_weight: float = 0.05
    del_loss_weight: float = 0.2
    bfactor_loss_weight: float = 0.02
    plddt_loss_weight: float = 0.02


@dataclass
class VarcoInferenceConfig(BaseClassConfig):
    predict_dir: str = "varco/outputs/${shared.id}"


@dataclass
class VarcoExperimentWandbConfig(ExperimentWandbConfig):
    project: str = "varco"


@dataclass
class VarcoExperimentConfig(ExperimentConfig):
    wandb: ExperimentWandbConfig = field(default_factory=VarcoExperimentWandbConfig)

    # Path to cogeneration checkpoint to load compatible weights from
    cogen_ckpt_path: Optional[str] = None


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


# Register the config class with Hydra
cs = ConfigStore.instance()
cs.store(name="varco", node=VarcoConfig)
