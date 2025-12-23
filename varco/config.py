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


@dataclass
class VarcoDatasetConfig(DatasetConfig):
    debug_head_samples: int = 1000  # TODO - enable everything
    enable_cogeneration_pdb: bool = True
    enable_cogeneration_afdb: bool = True
    enable_cogeneration_redesigns: bool = False
    enable_multiflow_redesigned: bool = False
    enable_multiflow_synthetic: bool = False


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
    model: VarcoModelConfig = field(default_factory=VarcoModelConfig)


# Register the config class with Hydra
cs = ConfigStore.instance()
cs.store(name="varco", node=VarcoConfig)
