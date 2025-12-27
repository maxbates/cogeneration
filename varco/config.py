import datetime
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
    ExperimentConfig,
    ModelAAPredConfig,
    ModelAttentionConfig,
    ModelAttentionTrunkConfig,
    ModelContactConditioningConfig,
    ModelEdgeFeaturesConfig,
    ModelESMCombinerConfig,
    ModelESMKey,
    ModelHyperParamsConfig,
    SharedConfig,
)
from cogeneration.type.str_enum import StrEnum


@dataclass
class VarcoDatasetConfig(DatasetConfig):
    debug_head_samples: int = 1000  # TODO - enable everything
    enable_cogeneration_pdb: bool = True
    enable_cogeneration_afdb: bool = True
    enable_cogeneration_redesigns: bool = False
    enable_multiflow_redesigned: bool = False
    enable_multiflow_synthetic: bool = False


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
    # Per-step force cap (angstroms) - prevents huge single-step jumps
    max_step_force_ang: float = 5.0


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
    motif_guidance: VarcoInterpolantMotifGuidanceConfig = field(
        default_factory=VarcoInterpolantMotifGuidanceConfig
    )
    sampling: VarcoInterpolantSamplingConfig = field(
        default_factory=VarcoInterpolantSamplingConfig
    )


@dataclass
class VarcoInferenceConfig(BaseClassConfig):
    predict_dir: str = "varco/outputs/${shared.id}"


@dataclass
class VarcoModelConfig(BaseClassConfig):
    # TODO - moar power!
    hyper_params: ModelHyperParamsConfig = field(
        default_factory=ModelHyperParamsConfig.poc
    )
    edge_features: ModelEdgeFeaturesConfig = field(
        default_factory=lambda: ModelEdgeFeaturesConfig(
            embed_self_condition=False,
            embed_chain=True,
            embed_diffuse_mask=True,  # motif mask
            contact_conditioning=ModelContactConditioningConfig(
                enabled=False,
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
    aa_pred: ModelAAPredConfig = field(default_factory=ModelAAPredConfig)


@dataclass
class VarcoConfig(BaseClassConfig):
    shared: SharedConfig = field(default_factory=SharedConfig)
    data: DataConfig = field(default_factory=DataConfig)
    dataset: VarcoDatasetConfig = field(default_factory=VarcoDatasetConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    inference: VarcoInferenceConfig = field(default_factory=VarcoInferenceConfig)
    interpolant: VarcoInterpolantConfig = field(default_factory=VarcoInterpolantConfig)
    model: VarcoModelConfig = field(default_factory=VarcoModelConfig)


# Register the config class with Hydra
cs = ConfigStore.instance()
cs.store(name="varco", node=VarcoConfig)
