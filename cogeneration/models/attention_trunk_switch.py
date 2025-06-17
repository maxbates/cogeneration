from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from cogeneration.config.base import (
    AttentionType,
    ModelAttentionPairBiasConfig,
    ModelDoubleAttentionPairConfig,
    ModelIPAConfig,
    ModelPairformerConfig,
)
from cogeneration.data.rigid import Rigid
from cogeneration.models.attention_pair_bias import AttentionPairBiasTrunk
from cogeneration.models.double_attention_pair import DoubleAttentionPairTrunk
from cogeneration.models.ipa_attention import AttentionIPATrunk
from cogeneration.models.pairformer import PairformerModule, PairformerNoSeqModule


class AttentionTrunkSwitch(nn.Module):
    """
    Switch Wrapper for trunks of various attention types supported.

    Unchanged tensors are passed through, e.g. if node_embed/edge_embed is not touched
    """

    def __init__(
        self,
        attn_type: AttentionType,
        cfg: Union[
            ModelAttentionPairBiasConfig,
            ModelDoubleAttentionPairConfig,
            ModelPairformerConfig,
            ModelIPAConfig,
        ],
        num_layers: Optional[int] = None,  # override cfg default
    ) -> None:
        super().__init__()
        self.attn_type = attn_type

        if attn_type is AttentionType.NONE:
            self.trunk = None

        elif attn_type == AttentionType.PAIR_BIAS:
            cfg: ModelAttentionPairBiasConfig = cfg.clone()
            if num_layers is not None:
                cfg.num_blocks = num_layers

            self.trunk = AttentionPairBiasTrunk(cfg=cfg)

        elif attn_type is AttentionType.DOUBLE_AXIS:
            cfg: ModelDoubleAttentionPairConfig = cfg.clone()
            if num_layers is not None:
                cfg.num_blocks = num_layers

            self.trunk = DoubleAttentionPairTrunk(cfg=cfg)

        elif attn_type is AttentionType.PAIRFORMER:
            cfg: ModelPairformerConfig = cfg.clone()
            if num_layers is not None:
                cfg.num_blocks = num_layers

            self.trunk = PairformerModule(cfg=cfg)

        elif attn_type is AttentionType.PAIRFORMER_NO_SEQ:
            cfg: ModelPairformerConfig = cfg.clone()
            if num_layers is not None:
                cfg.num_blocks = num_layers

            self.trunk = PairformerNoSeqModule(cfg=cfg)

        elif attn_type is AttentionType.IPA:
            cfg: ModelIPAConfig = cfg.clone()
            if num_layers is not None:
                cfg.num_blocks = num_layers

            # Don't support backbone updates in this wrapper - call separately for bb + torsions
            self.trunk = AttentionIPATrunk(
                cfg=cfg,
                perform_backbone_update=False,
                predict_psi_torsions=False,
                predict_all_torsions=False,
            )
        else:
            raise ValueError(f"unknown attention kind: {attn_type}")

    @property
    def enabled(self):
        # Some trunks may allow specifying 0 blocks, expect `enabled` property
        if self.trunk is None:
            return False
        if hasattr(self.trunk, "enabled"):
            return self.trunk.enabled
        return True

    def forward(
        self,
        node_embed: torch.Tensor,  # (B, N, node_dim)
        edge_embed: Optional[torch.Tensor],  # (B, N, N, edge_dim)
        node_mask: Optional[torch.Tensor],  # (B, N)
        edge_mask: Optional[torch.Tensor],  # (B, N, N)
        rigid: Optional[Rigid] = None,  # frames (IPA only)
        r3_t: Optional[torch.Tensor] = None,  # time, for time conditioning
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.enabled:
            return node_embed, edge_embed

        elif self.attn_type is AttentionType.PAIR_BIAS:
            node_embed = self.trunk(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
            )

        elif self.attn_type is AttentionType.DOUBLE_AXIS:
            edge_embed = self.trunk(
                edge_embed=edge_embed,
                edge_mask=edge_mask,
                r3_t=r3_t,
            )

        elif self.attn_type is AttentionType.PAIRFORMER:
            node_embed, edge_embed = self.trunk(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

        elif self.attn_type is AttentionType.PAIRFORMER_NO_SEQ:
            edge_embed = self.trunk(
                edge_embed=edge_embed,
                edge_mask=edge_mask,
            )

        elif self.attn_type is AttentionType.IPA:
            node_embed, edge_embed, rigid, _ = self.trunk(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                edge_mask=edge_mask,
                rigid=rigid,
            )

        else:
            raise ValueError(f"unknown attention kind: {self.attn_type}")

        return node_embed, edge_embed
