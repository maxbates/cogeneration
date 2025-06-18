from typing import Optional

import torch
from torch import nn

from cogeneration.config.base import ModelDoubleAttentionPairConfig
from cogeneration.models.attention.ipa_pytorch import Linear


class DoubleAttentionPairBlock(nn.Module):
    """
    Rough but O(N^2) substitute triangle‐attention. Does not consider node features.

    Uses a two‐step “gather & redistribute” over all pair edges:
    - Gather global context via a softmax‐weighted sum of `fP` and value projection `gP`
    - Redistribute that context back to each edge via a separate softmax on `hP`
    - Optionally, applies a channel‐wise FiLM (scale + shift) conditioned on the timestep

    Differs from standard self‐attention:
    - operates on the flattened edge map, not on feature tokens
    - mimics two‐hop (i -> k -> j) triangle interactions rather than dot‑product between queries/keys
    - complexity is O(N^2 * D^2) (where D << C) versus O(N^3 * C) for triangle attention
    """

    def __init__(self, cfg: ModelDoubleAttentionPairConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.edge_embed_size % cfg.bottleneck_scale == 0
        bottleneck_dim = self.cfg.edge_embed_size // cfg.bottleneck_scale

        self.gather_proj = nn.Linear(self.cfg.edge_embed_size, bottleneck_dim)
        self.value_proj = nn.Linear(self.cfg.edge_embed_size, bottleneck_dim)
        self.distribute_proj = nn.Linear(self.cfg.edge_embed_size, bottleneck_dim)
        self.output_proj = nn.Linear(bottleneck_dim, self.cfg.edge_embed_size)

        if self.cfg.use_film:
            self.time_mlp = nn.Sequential(
                nn.Linear(1, self.cfg.time_mlp_hidden_dim),  # embed scalar t
                nn.GELU(),
                nn.Linear(
                    self.cfg.time_mlp_hidden_dim, self.cfg.edge_embed_size * 2
                ),  # scale+shift
            )

    def forward(
        self,
        edge_embed: torch.Tensor,  # (B, N, N, edge_dim)
        edge_mask: torch.Tensor,  # (B, N, N)
        r3_t: torch.Tensor,  # (B, 1)
    ) -> torch.Tensor:
        B, N, _, edge_dim = edge_embed.shape

        # flatten (B, N, N, edge_dim) → (B, N*N, edge_dim)
        P = edge_embed.view(B, N * N, edge_dim)

        # compute gather weights and values
        fP = self.gather_proj(P)  # (B, N*N, bottleneck_dim)
        gP = self.value_proj(P)  # (B, N*N, bottleneck_dim)

        # apply edge_mask to gather weights
        edge_mask_flat = edge_mask.view(B, N * N, 1)  # (B, N*N, 1)
        fP = fP.masked_fill(edge_mask_flat == 0, -1e8)
        weights = torch.softmax(fP, dim=1)  # norm over all edges
        # to be safe, zero out masked positions and renormalize
        weights = weights * edge_mask_flat
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)

        # aggregate global context
        # (B, bottleneck_dim, N*N) @ (B, N*N, bottleneck_dim) -> (B, bottleneck_dim, bottleneck_dim)
        V = weights.transpose(1, 2) @ gP

        # compute distribute scores and redistribute context
        hP = self.distribute_proj(P)  # (B, N*N, bottleneck_dim)
        scores = torch.softmax(hP, dim=2)  # norm over context dims
        O = scores @ V  # (B, N*N, bottleneck_dim)
        O = self.output_proj(O).view(B, N, N, edge_dim)

        # Feature-wise Linear Modulation (FiLM) using timestep embedding
        if self.cfg.use_film:
            t_emb = self.time_mlp(r3_t.view(B, 1))  # (B, 2*C)
            scale, shift = t_emb.chunk(2, dim=-1)
            O = O * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(
                1
            ).unsqueeze(1)

        # residual update
        return edge_embed + O


class DoubleAttentionPairTrunk(nn.Module):
    """
    Sequential trunk of `DoubleAttentionPairBlock`s plus a final LayerNorm.
    Returns (node_embed, edge_embed); only `edge_embed` is modified.
    """

    def __init__(
        self, cfg: ModelDoubleAttentionPairConfig, final_layer_norm: bool = False
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.final_layer_norm = final_layer_norm

        if self.enabled:
            self.blocks = nn.ModuleList(
                [DoubleAttentionPairBlock(cfg) for _ in range(cfg.num_layers)]
            )

            if final_layer_norm:
                self.final_layer = Linear(
                    cfg.edge_embed_size, cfg.edge_embed_size, init="final"
                )
                self.layer_norm = nn.LayerNorm(cfg.edge_embed_size)

    @property
    def enabled(self):
        return self.cfg.num_layers > 0

    def forward(
        self,
        edge_embed: torch.Tensor,  # (B, N, N, edge_dim)
        edge_mask: torch.Tensor,  # (B, N, N)
        r3_t: Optional[torch.Tensor],  # (B, 1)
    ) -> torch.Tensor:
        """
        Pass edge features through all blocks, then layer-norm.
        """
        if not self.enabled:
            return edge_embed

        for blk in self.blocks:
            edge_embed = blk(edge_embed=edge_embed, edge_mask=edge_mask, r3_t=r3_t)

        if self.final_layer_norm:
            edge_embed = self.final_layer(edge_embed)
            edge_embed = self.layer_norm(edge_embed)

        return edge_embed
