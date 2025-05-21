from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import nn

from cogeneration.config.base import (
    ModelESMCombinerConfig,
    ModelESMDoubleAttentionPairConfig,
)
from cogeneration.models.esm_frozen import FrozenEsmModel


def _mlp(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class DoubleAttentionPairBlock(nn.Module):
    """
    Rough but O(N^2) substitute triangle‐attention.

    Uses a two‐step “gather & redistribute” over all pair edges:
    - Gather global context via a softmax‐weighted sum of `fP` and value projection `gP`
    - Redistribute that context back to each edge via a separate softmax on `hP`
    - Applies a channel‐wise FiLM (scale + shift) conditioned on the timestep

    Differs from standard self‐attention:
    - operates on the flattened edge map, not on feature tokens
    - mimics two‐hop (i -> k -> j) triangle interactions rather than dot‑product between queries/keys
    - complexity is O(N^2 * D^2) (where D << C) versus O(N^3 * C) for triangle attention
    """

    def __init__(self, cfg: ModelESMDoubleAttentionPairConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.edge_embed_size % cfg.bottleneck_scale == 0
        bottleneck_dim = self.cfg.edge_embed_size // cfg.bottleneck_scale

        self.gather_proj = nn.Linear(self.cfg.edge_embed_size, bottleneck_dim)
        self.value_proj = nn.Linear(self.cfg.edge_embed_size, bottleneck_dim)
        self.distribute_proj = nn.Linear(self.cfg.edge_embed_size, bottleneck_dim)
        self.output_proj = nn.Linear(bottleneck_dim, self.cfg.edge_embed_size)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, self.cfg.time_mlp_hidden_dim),  # embed scalar t
            nn.GELU(),
            nn.Linear(
                self.cfg.time_mlp_hidden_dim, self.cfg.edge_embed_size * 2
            ),  # scale+shift
        )

    def forward(self, edge_embed: torch.Tensor, r3_t: torch.Tensor) -> torch.Tensor:
        B, N, _, edge_dim = edge_embed.shape

        # flatten (B, N, N, edge_dim) → (B, N*N, edge_dim)
        P = edge_embed.view(B, N * N, edge_dim)

        # compute gather weights and values
        fP = self.gather_proj(P)  # (B, N*N, bottleneck_dim)
        gP = self.value_proj(P)  # (B, N*N, bottleneck_dim)
        weights = torch.softmax(fP, dim=1)  # norm over all edges
        # aggregate global context
        # (B, bottleneck_dim, N*N) @ (B, N*N, bottleneck_dim) -> (B, bottleneck_dim, bottleneck_dim)
        V = weights.transpose(1, 2) @ gP

        # compute distribute scores and redistribute context
        hP = self.distribute_proj(P)  # (B, N*N, bottleneck_dim)
        scores = torch.softmax(hP, dim=2)  # norm over context dims
        O = scores @ V  # (B, N*N, bottleneck_dim)
        O = self.output_proj(O).view(B, N, N, edge_dim)

        # Feature-wise Linear Modulation (FiLM) using timestep embedding
        t_emb = self.time_mlp(r3_t.view(B, 1))  # (B, 2*C)
        scale, shift = t_emb.chunk(2, dim=-1)
        O = O * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)

        # residual update
        return edge_embed + O


class ESMCombinerNetwork(nn.Module):
    """
    Runs ESM model and combines single and pair representations with node and edge embeddings.
    Optionally also runs a trunk of DoubleAttentionPairBlocks to enrich edge embeddings.
    """

    def __init__(self, cfg: ModelESMCombinerConfig):
        super().__init__()
        self.cfg = cfg

        self.esm = FrozenEsmModel(model_key=cfg.esm_model_key, use_esm_attn_map=True)

        # learn a scalar mix over all single layers
        self.esm_single_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        # take weighted sum of single layer as input (B, N, C) -> (B, N, node_dim)
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
        self.single_layer_norm = nn.LayerNorm(self.cfg.node_embed_size)
        self.pair_layer_norm = nn.LayerNorm(self.cfg.edge_embed_size)

        # DoubleAttentionPairBlock trunk
        if self.cfg.double_attention_pair_trunk.num_blocks > 0:
            self.double_attention_pair_trunk = nn.ModuleList(
                [
                    DoubleAttentionPairBlock(self.cfg.double_attention_pair_trunk)
                    for _ in range(self.cfg.double_attention_pair_trunk.num_blocks)
                ]
            )

            self.trunk_pair_layer_norm = nn.LayerNorm(self.cfg.edge_embed_size)

    def forward(
        self,
        init_node_embed: torch.Tensor,  # (B, N, node_dim)
        init_edge_embed: torch.Tensor,  # (B, N, N, edge_dim)
        aatypes_t: torch.Tensor,  # (B, N)
        chain_index: torch.Tensor,  # (B, N)
        res_mask: torch.Tensor,  # (B, N)
        r3_t: torch.Tensor,  # (B, 1)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # get ESM single and pair representations
        esm_single, esm_pair = self.esm(
            aatypes=aatypes_t,
            chain_index=chain_index,
            res_mask=res_mask,
        )
        esm_single = esm_single.detach()  # (B, N, nLayers, node_dim)
        esm_pair = esm_pair.detach()  # (B, N, N, nLayers*nHeads)

        # flatten `esm_single` using weighted sum `esm_single_combine`
        weights = self.esm_single_combine.softmax(0)
        w = weights.view(1, 1, -1, 1)  # (1, 1, nLayers+1, 1)
        esm_single = (w * esm_single).sum(dim=2)  # (B, N, node_dim)

        # project weighted single and flattened pair representations
        esm_single = self.seq_proj(esm_single)  # (B, N, node_dim)
        esm_pair = self.pair_proj(esm_pair)  # (B, N, N, edge_dim)

        # combine + LayerNorm
        node_embed = 0.5 * (esm_single + init_node_embed)  # (B, N, node_dim)
        edge_embed = 0.5 * (esm_pair + init_edge_embed)  # (B, N, N, edge_dim)
        node_embed = self.single_layer_norm(node_embed)
        edge_embed = self.pair_layer_norm(edge_embed)

        # TODO - support ESMFold style folding blocks instead.
        #    Note requires `openfold` install for Triangle Updates.
        #    Can install as local package, but brings in large dependency we've avoided so far.
        #    May wish to add in RelativePosition embedding for folding blocks.
        # TODO - alternatively, `trifast` https://github.com/latkins/trifast
        if self.cfg.double_attention_pair_trunk.num_blocks > 0:
            for block in self.double_attention_pair_trunk:
                edge_embed = block(edge_embed=edge_embed, r3_t=r3_t)

            # add back in initial representation after trunk + norm
            edge_embed = 0.5 * (edge_embed + init_edge_embed)  # (B, N, N, edge_dim)
            edge_embed = self.trunk_pair_layer_norm(edge_embed)  # (B, N, N, edge_dim)

        # mask
        edge_mask = res_mask[:, None] * res_mask[:, :, None]
        node_embed = node_embed * res_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        return node_embed, edge_embed
