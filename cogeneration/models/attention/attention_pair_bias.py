# Adapted from Boltz-2


from typing import Optional

import torch
from einops.layers.torch import Rearrange
from torch import Tensor, nn

from cogeneration.config.base import ModelAttentionPairBiasConfig
from cogeneration.models.attention.ipa_pytorch import Linear


class AttentionPairBias(nn.Module):
    """Attention pair bias layer."""

    def __init__(
        self,
        cfg: ModelAttentionPairBiasConfig,
    ) -> None:
        """Initialize the attention pair bias layer.

        Parameters
        ----------
        node_dim : int
            The input sequence dimension.
        edge_dim : int
            The input pairwise dimension.
        num_heads : int
            The number of heads.
        inf : float, optional
            The inf value, by default 1e6

        """
        super().__init__()
        self.cfg = cfg
        self.inf = 1e6

        assert self.cfg.node_dim % self.cfg.num_heads == 0

        self.num_heads = self.cfg.num_heads
        self.head_dim = self.cfg.node_dim // self.num_heads

        self.proj_q = nn.Linear(self.cfg.node_dim, self.cfg.node_dim)
        self.proj_k = nn.Linear(self.cfg.node_dim, self.cfg.node_dim, bias=False)
        self.proj_v = nn.Linear(self.cfg.node_dim, self.cfg.node_dim, bias=False)
        self.proj_g = nn.Linear(self.cfg.node_dim, self.cfg.node_dim, bias=False)

        if self.cfg.compute_pair_bias:
            self.proj_z = nn.Sequential(
                nn.LayerNorm(self.cfg.edge_dim),
                nn.Linear(self.cfg.edge_dim, self.cfg.num_heads, bias=False),
                Rearrange("b ... h -> b h ..."),
            )
        else:
            self.proj_z = Rearrange("b ... h -> b h ...")

        self.proj_o = Linear(
            self.cfg.node_dim, self.cfg.node_dim, bias=False, init="final"
        )

    def forward(
        self,
        node_embed: Tensor,
        edge_embed: Tensor,
        node_mask: Tensor,
        k_in: Tensor,
    ) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        node_embed : torch.Tensor
            The input sequence tensor (B, N, node_dim)
        edge_embed : torch.Tensor
            The input pairwise tensor or bias (B, N, N, edge_dim)
        node_mask : torch.Tensor
            The pairwise mask tensor (B, N, N)
        k_in : torch.Tensor
            The input sequence tensor (i.e. node_embed), or a different tensor (B, N, node_dim)
        Returns
        -------
        torch.Tensor
            The output sequence tensor.

        """
        B = node_embed.shape[0]

        # Compute projections
        q = self.proj_q(node_embed).view(B, -1, self.num_heads, self.head_dim)
        k = self.proj_k(k_in).view(B, -1, self.num_heads, self.head_dim)
        v = self.proj_v(k_in).view(B, -1, self.num_heads, self.head_dim)

        bias = self.proj_z(edge_embed)

        g = self.proj_g(node_embed).sigmoid()

        with torch.autocast("cuda", enabled=False):
            # Compute attention weights
            attn = torch.einsum("bihd,bjhd->bhij", q.float(), k.float())
            attn = attn / (self.head_dim**0.5) + bias.float()
            attn = attn + (1 - node_mask[:, None, None].float()) * -self.inf
            attn = attn.softmax(dim=-1)

            # Compute output
            o = torch.einsum("bhij,bjhd->bihd", attn, v.float()).to(v.dtype)
        o = o.reshape(B, -1, self.cfg.node_dim)
        o = self.proj_o(g * o)

        return o


class AttentionPairBiasTrunk(nn.Module):
    """Stacked Attention-Pair-Bias layers acting on node embeddings."""

    def __init__(
        self,
        cfg: ModelAttentionPairBiasConfig,
        final_layer_norm: bool = False,
        num_layers: Optional[int] = None,
    ):
        super().__init__()
        self.cfg = cfg
        self.final_layer_norm = final_layer_norm
        self.num_layers = num_layers if num_layers is not None else cfg.num_layers

        self.layers = nn.ModuleList(
            [AttentionPairBias(cfg) for _ in range(self.num_layers)]
        )

        if final_layer_norm:
            self.final_layer = Linear(cfg.node_dim, cfg.node_dim, init="final")
            self.layer_norm = nn.LayerNorm(cfg.node_dim)

    def forward(
        self,
        node_embed: torch.Tensor,  # (B, N, node_dim)
        edge_embed: torch.Tensor,  # (B, N, N, edge_dim)
        node_mask: torch.Tensor,  # (B, N) or None
    ) -> torch.Tensor:
        """Apply the stack; pair features are read-only."""

        if self.num_layers < 1:
            return node_embed

        for layer in self.layers:
            node_embed = layer(
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                k_in=node_embed,
            )

        if self.final_layer_norm:
            node_embed = self.final_layer(node_embed)
            node_embed = self.layer_norm(node_embed)

        return node_embed
