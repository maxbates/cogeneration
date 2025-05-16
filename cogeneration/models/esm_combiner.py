from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from cogeneration.config.base import ModelESMCombinerCfg
from cogeneration.models.esm_frozen import FrozenEsmModel


def _mlp(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class ESMCombinerNetwork(nn.Module):
    """Run ESM model and combine single and pair representations"""

    def __init__(self, cfg: ModelESMCombinerCfg):
        super().__init__()
        self.cfg = cfg

        self.esm = FrozenEsmModel(model_key=cfg.esm_model_key, use_esm_attn_map=True)

        # take last single layer as input (B, N, C) -> (B, N, node_dim)
        self.seq_proj = _mlp(
            in_dim=self.esm.embed_dim,
            out_dim=self.cfg.esm_proj_single_dim,
            hidden_dim=self.cfg.mlp_proj_hidden_dim,
        )

        # pair reps come out flattened layers*heads (B, N, N, layers*heads) -> (B, N, N, edge_dim)
        self.pair_proj = _mlp(
            in_dim=self.esm.num_layers * self.esm.num_heads,
            out_dim=self.cfg.esm_proj_pair_dim,
            hidden_dim=self.cfg.mlp_proj_hidden_dim,
        )

        # LayerNorm after combine
        self.single_ln = nn.LayerNorm(self.cfg.node_embed_size)
        self.pair_ln = nn.LayerNorm(self.cfg.edge_embed_size)

    def forward(
        self,
        init_node_embed: Tensor,  # (B, N, node_dim)
        init_edge_embed: Tensor,  # (B, N, N, edge_dim)
        aatypes_t: Tensor,  # (B, N)
        chain_index: Tensor,  # (B, N)
        res_mask: Tensor,  # (B, N)
        diffuse_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        esm_single, esm_pair = self.esm(
            aatypes=aatypes_t,
            chain_index=chain_index,
            attn_mask=res_mask,
            seq_mask=diffuse_mask,
        )

        # embed pair and last single representations
        esm_pair = self.pair_proj(esm_pair)  # (B, N, N, C_pair) -> (B, N, N, edge_dim)
        esm_single = esm_single[..., -1, :]  # (B, N, nLayers, C) -> (B, N, C)
        esm_single = self.seq_proj(esm_single)  # (B, N, node_dim)

        # combine + LayerNorm
        node_embed = init_node_embed + esm_single  # (B, N, node_dim)
        edge_embed = init_edge_embed + esm_pair  # (B, N, N, edge_dim)
        node_embed = self.single_ln(node_embed)
        edge_embed = self.pair_ln(edge_embed)

        # TODO - support folding blocks, add back in initial representations
        # TODO - pass in and apply node_mask and edge_mask

        return node_embed, edge_embed
