# Adapted from Boltz-2

from typing import Optional, Tuple

import torch
import torch.utils.checkpoint as _ckpt
from torch import Tensor, nn

from cogeneration.config.base import ModelPairformerConfig
from cogeneration.models.attention.attention_pair_bias import AttentionPairBias
from cogeneration.models.attention.dropout import get_dropout_mask
from cogeneration.models.attention.transition import Transition
from cogeneration.models.attention.triangular_attention import (
    TriangleAttentionEndingNode,
    TriangleAttentionStartingNode,
)
from cogeneration.models.attention.triangular_mult import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)

# chunking constant
CHUNK_SIZE_THRESHOLD = 384


def _get_chunk_size(
    is_training: bool,
    edge_embed: torch.Tensor,  # (B, N, N, c_z)
) -> Optional[int]:
    if not is_training:
        if edge_embed.shape[1] > CHUNK_SIZE_THRESHOLD:
            chunk_size_tri_attn = min(128, edge_embed.shape[1] // 2)
        else:
            chunk_size_tri_attn = 512
    else:
        chunk_size_tri_attn = None

    return chunk_size_tri_attn


class PairformerLayer(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        cfg: ModelPairformerConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.pre_norm_s = nn.LayerNorm(self.cfg.node_dim)
        self.attention = AttentionPairBias(
            cfg=self.cfg.attention_pair_bias,
        )

        self.tri_mul_out = TriangleMultiplicationOutgoing(dim=self.cfg.edge_dim)
        self.tri_mul_in = TriangleMultiplicationIncoming(dim=self.cfg.edge_dim)

        self.tri_att_start = TriangleAttentionStartingNode(
            c_in=self.cfg.edge_dim,
            c_hidden=self.cfg.pairwise_head_width,
            no_heads=self.cfg.pairwise_num_heads,
            inf=1e9,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_in=self.cfg.edge_dim,
            c_hidden=self.cfg.pairwise_head_width,
            no_heads=self.cfg.pairwise_num_heads,
            inf=1e9,
        )

        self.node_transition = Transition(
            dim=self.cfg.node_dim, hidden=self.cfg.node_dim * 4
        )
        self.edge_transition = Transition(
            dim=self.cfg.edge_dim, hidden=self.cfg.edge_dim * 4
        )

        self.node_post_norm = (
            nn.LayerNorm(self.cfg.node_dim)
            if self.cfg.post_layer_norm
            else nn.Identity()
        )

    def forward(
        self,
        node_embed: Tensor,  # (B, N, node_dim)
        edge_embed: Tensor,  # (B, N, N, edge_dim)
        node_mask: Tensor,  # (B, N)
        edge_mask: Tensor,  # (B, N, N)
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> tuple[Tensor, Tensor]:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_mul_out(
            x=edge_embed, mask=edge_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_mul_in(
            x=edge_embed, mask=edge_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_att_start(
            x=edge_embed,
            mask=edge_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        dropout = get_dropout_mask(
            self.cfg.dropout, edge_embed, self.training, columnwise=True
        )
        edge_embed = edge_embed + dropout * self.tri_att_end(
            x=edge_embed,
            mask=edge_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        edge_embed = edge_embed + self.edge_transition(edge_embed)

        # Compute sequence stack
        with torch.autocast("cuda", enabled=False):
            s_normed = self.pre_norm_s(node_embed.float())
            node_embed = node_embed.float() + self.attention(
                node_embed=s_normed,
                edge_embed=edge_embed.float(),
                node_mask=node_mask.float(),
                k_in=s_normed,
            )
            node_embed = node_embed + self.node_transition(node_embed)
            node_embed = self.node_post_norm(node_embed)

        return node_embed, edge_embed


class PairformerNoSeqLayer(nn.Module):
    """Pairformer module without sequence track."""

    def __init__(
        self,
        cfg: ModelPairformerConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.tri_mul_out = TriangleMultiplicationOutgoing(dim=self.cfg.edge_dim)
        self.tri_mul_in = TriangleMultiplicationIncoming(dim=self.cfg.edge_dim)

        self.tri_att_start = TriangleAttentionStartingNode(
            c_in=self.cfg.edge_dim,
            c_hidden=self.cfg.pairwise_head_width,
            no_heads=self.cfg.pairwise_num_heads,
            inf=1e9,
        )
        self.tri_att_end = TriangleAttentionEndingNode(
            c_in=self.cfg.edge_dim,
            c_hidden=self.cfg.pairwise_head_width,
            no_heads=self.cfg.pairwise_num_heads,
            inf=1e9,
        )

        self.transition_z = Transition(
            dim=self.cfg.edge_dim, hidden=self.cfg.edge_dim * 4
        )

    def forward(
        self,
        edge_embed: Tensor,
        edge_mask: Tensor,
        chunk_size_tri_attn: Optional[int] = None,
        use_kernels: bool = False,
        use_cuequiv_mul: bool = False,
        use_cuequiv_attn: bool = False,
    ) -> Tensor:
        # Compute pairwise stack
        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_mul_out(
            x=edge_embed, mask=edge_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_mul_in(
            x=edge_embed, mask=edge_mask, use_kernels=use_cuequiv_mul or use_kernels
        )

        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_att_start(
            x=edge_embed,
            mask=edge_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        dropout = get_dropout_mask(self.cfg.dropout, edge_embed, self.training)
        edge_embed = edge_embed + dropout * self.tri_att_end(
            x=edge_embed,
            mask=edge_mask,
            chunk_size=chunk_size_tri_attn,
            use_kernels=use_cuequiv_attn or use_kernels,
        )

        edge_embed = edge_embed + self.transition_z(edge_embed)
        return edge_embed


class PairformerModule(nn.Module):
    """Pairformer module."""

    def __init__(
        self,
        cfg: ModelPairformerConfig,
        num_layers: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        # Allow overriding num_layers from config
        self.num_layers = num_layers if num_layers is not None else self.cfg.num_layers

        if self.num_layers > 0:
            self.layers = nn.ModuleList(
                [PairformerLayer(self.cfg) for _ in range(self.num_layers)]
            )

    def forward(
        self,
        node_embed: Tensor,
        edge_embed: Tensor,
        node_mask: Tensor,
        edge_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if self.num_layers < 1:
            return node_embed, edge_embed

        chunk_size_tri_attn = _get_chunk_size(
            is_training=self.training, edge_embed=edge_embed
        )

        for layer in self.layers:
            if self.cfg.checkpointing and self.training:
                # cannot use kwargs with checkpoint
                node_embed, edge_embed = torch.utils.checkpoint.checkpoint(
                    layer,
                    node_embed,
                    edge_embed,
                    node_mask,
                    edge_mask,
                    chunk_size_tri_attn,
                    self.cfg.use_kernels,
                )
            else:
                node_embed, edge_embed = layer(
                    node_embed=node_embed,
                    edge_embed=edge_embed,
                    node_mask=node_mask,
                    edge_mask=edge_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_kernels=self.cfg.use_kernels,
                )

        return node_embed, edge_embed


class PairformerNoSeqModule(nn.Module):
    """Stack of edge-only Pairformer layers."""

    def __init__(self, cfg: ModelPairformerConfig, num_layers: Optional[int] = None):
        super().__init__()
        self.cfg = cfg
        # Allow overriding num_layers from config
        self.num_layers = num_layers if num_layers is not None else cfg.num_layers

        if self.num_layers > 0:
            self.layers = nn.ModuleList(
                [PairformerNoSeqLayer(cfg) for _ in range(self.num_layers)]
            )

    def forward(
        self,
        edge_embed: Tensor,
        edge_mask: Tensor,
    ) -> torch.Tensor:
        if self.num_layers < 1:
            return edge_embed

        chunk_size_tri_attn = _get_chunk_size(
            is_training=self.training, edge_embed=edge_embed
        )

        for layer in self.layers:
            if self.cfg.checkpointing and self.training:
                # cannot use kwargs with checkpoint
                edge_embed = torch.utils.checkpoint.checkpoint(
                    layer,
                    edge_embed,
                    edge_mask,
                    chunk_size_tri_attn,
                    self.cfg.use_kernels,
                )
            else:
                edge_embed = layer(
                    edge_embed=edge_embed,
                    edge_mask=edge_mask,
                    chunk_size_tri_attn=chunk_size_tri_attn,
                    use_kernels=self.cfg.use_kernels,
                )

        return edge_embed
