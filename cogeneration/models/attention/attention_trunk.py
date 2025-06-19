from typing import Optional, Tuple

import torch
import torch.nn as nn

from cogeneration.config.base import (
    AttentionType,
    ModelAttentionConfig,
    ModelAttentionPairBiasConfig,
    ModelAttentionTrunkConfig,
    ModelDoubleAttentionPairConfig,
    ModelIPAConfig,
    ModelPairformerConfig,
)
from cogeneration.data.rigid import Rigid
from cogeneration.models.attention.attention_pair_bias import AttentionPairBiasTrunk
from cogeneration.models.attention.double_attention_pair import DoubleAttentionPairTrunk
from cogeneration.models.attention.ipa_attention import AttentionIPATrunk
from cogeneration.models.attention.pairformer import (
    PairformerModule,
    PairformerNoSeqModule,
)


class AttentionTrunk(nn.Module):
    """
    Wrapper for trunks of various attention types supported.
    Also supports pre-post merging with initial embeddings, and layer norms.

    Unchanged tensors are passed through, e.g. if node_embed/edge_embed is not touched
    """

    def __init__(
        self,
        cfg: ModelAttentionTrunkConfig,
        attn_cfg: ModelAttentionConfig,
    ) -> None:
        super().__init__()
        self.cfg = cfg

        # pre trunk
        if self.cfg.pre_node_layer_norm:
            self.pre_node_ln = nn.LayerNorm(self.cfg.node_dim)
        if self.cfg.pre_edge_layer_norm:
            self.pre_edge_ln = nn.LayerNorm(self.cfg.edge_dim)

        # TODO(attn) - consider making num_layers an argument to each module
        #   rather than requiring it be in the config, to avoid the `merge_dict`.

        # trunk

        if self.cfg.num_layers < 1 or self.cfg.attn_type is AttentionType.NONE:
            self.trunk = None

        elif self.cfg.attn_type == AttentionType.PAIR_BIAS:
            attn_cfg: ModelAttentionPairBiasConfig = attn_cfg.pair_bias.merge_dict(
                {
                    "num_layers": self.cfg.num_layers,
                }
            )

            self.trunk = AttentionPairBiasTrunk(cfg=attn_cfg)

        elif self.cfg.attn_type is AttentionType.DOUBLE:
            attn_cfg: ModelDoubleAttentionPairConfig = (
                attn_cfg.double_attention_pair.merge_dict(
                    {
                        "num_layers": self.cfg.num_layers,
                    }
                )
            )

            self.trunk = DoubleAttentionPairTrunk(cfg=attn_cfg)

        elif self.cfg.attn_type is AttentionType.PAIRFORMER:
            attn_cfg: ModelPairformerConfig = attn_cfg.pairformer.merge_dict(
                {
                    "num_layers": self.cfg.num_layers,
                }
            )

            self.trunk = PairformerModule(cfg=attn_cfg)

        elif self.cfg.attn_type is AttentionType.PAIRFORMER_NO_SEQ:
            attn_cfg: ModelPairformerConfig = attn_cfg.pairformer.merge_dict(
                {
                    "num_layers": self.cfg.num_layers,
                }
            )

            self.trunk = PairformerNoSeqModule(cfg=attn_cfg)

        elif self.cfg.attn_type is AttentionType.IPA:
            attn_cfg: ModelIPAConfig = attn_cfg.ipa.merge_dict(
                {
                    "num_blocks": self.cfg.num_layers,
                }
            )

            # Don't support backbone updates in this wrapper - call separately for bb + torsions.
            # Assumes for representation enrichment, so enables final edge update.
            self.trunk = AttentionIPATrunk(
                cfg=attn_cfg,
                perform_backbone_update=False,
                perform_final_edge_update=True,
                predict_psi_torsions=False,
                predict_all_torsions=False,
            )
        else:
            raise ValueError(f"unknown attention kind: {self.cfg.attn_type}")

        # post trunk
        if self.cfg.post_node_layer_norm:
            self.post_node_ln = nn.LayerNorm(self.cfg.node_dim)
        if self.cfg.post_edge_layer_norm:
            self.post_edge_ln = nn.LayerNorm(self.cfg.edge_dim)

    def forward(
        self,
        init_node_embed: torch.Tensor,
        init_edge_embed: torch.Tensor,
        node_embed: torch.Tensor,  # (B, N, node_dim)
        edge_embed: torch.Tensor,  # (B, N, N, edge_dim)
        node_mask: torch.Tensor,  # (B, N)
        edge_mask: torch.Tensor,  # (B, N, N)
        rigid: Optional[Rigid],  # nm scale frames (only used by IPA)
        r3_t: Optional[torch.Tensor],  # time, if time conditioning used
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pre trunk

        if self.cfg.pre_add_init_embed:
            node_embed = 0.5 * (node_embed + init_node_embed)
            edge_embed = 0.5 * (edge_embed + init_edge_embed)

            node_embed = node_embed * node_mask[..., None]
            edge_embed = edge_embed * edge_mask[..., None]

        if self.cfg.pre_node_layer_norm:
            node_embed = self.pre_node_ln(node_embed)
        if self.cfg.pre_edge_layer_norm:
            edge_embed = self.pre_edge_ln(edge_embed)

        # trunk

        if self.trunk is None:
            pass

        elif self.cfg.attn_type is AttentionType.PAIR_BIAS:
            node_embed = self.trunk(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
            )

        elif self.cfg.attn_type is AttentionType.DOUBLE:
            edge_embed = self.trunk(
                edge_embed=edge_embed,
                edge_mask=edge_mask,
                r3_t=r3_t,
            )

        elif self.cfg.attn_type is AttentionType.PAIRFORMER:
            node_embed, edge_embed = self.trunk(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )

        elif self.cfg.attn_type is AttentionType.PAIRFORMER_NO_SEQ:
            edge_embed = self.trunk(
                edge_embed=edge_embed,
                edge_mask=edge_mask,
            )

        elif self.cfg.attn_type is AttentionType.IPA:
            node_embed, edge_embed, rigid, _ = self.trunk(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                edge_mask=edge_mask,
                diffuse_mask=node_mask,
                curr_rigids_nm=rigid,
            )

        else:
            raise ValueError(f"unknown attention kind: {self.cfg.attn_type}")

        # post trunk

        if self.cfg.post_add_init_embed:
            node_embed = 0.5 * (node_embed + init_node_embed)
            edge_embed = 0.5 * (edge_embed + init_edge_embed)

            node_embed = node_embed * node_mask[..., None]
            edge_embed = edge_embed * edge_mask[..., None]

        if self.cfg.post_node_layer_norm:
            node_embed = self.post_node_ln(node_embed)

        if self.cfg.post_edge_layer_norm:
            edge_embed = self.post_edge_ln(edge_embed)

        return node_embed, edge_embed
