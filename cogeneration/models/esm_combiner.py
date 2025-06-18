from typing import Tuple

import torch
from torch import nn

from cogeneration.config.base import ModelESMCombinerConfig
from cogeneration.models.esm_frozen import FrozenEsmModel


def _mlp(in_dim: int, out_dim: int, hidden_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.LayerNorm(in_dim),
        nn.Linear(in_dim, hidden_dim),
        nn.GELU(),
        nn.Linear(hidden_dim, out_dim),
    )


class ESMCombinerNetwork(nn.Module):
    """
    Runs ESM model and combines single and pair representations with node and edge embeddings.
    If cfg.only_single, only generates single rep and enriches node embeddings.
    """

    def __init__(self, cfg: ModelESMCombinerConfig):
        super().__init__()
        self.cfg = cfg

        self.esm = FrozenEsmModel(
            model_key=cfg.esm_model_key,
            use_esm_attn_map=not self.cfg.only_single,
        )

        # learn a scalar mix over all single layers
        self.esm_single_combine = nn.Parameter(torch.zeros(self.esm.num_layers + 1))

        # take weighted sum of single layer as input (B, N, C) -> (B, N, node_dim)
        self.seq_proj = _mlp(
            in_dim=self.esm.embed_dim,
            out_dim=self.cfg.esm_proj_single_dim,
            hidden_dim=self.cfg.mlp_proj_hidden_dim,
        )
        self.single_layer_norm = nn.LayerNorm(self.cfg.node_embed_size)

        if not self.cfg.only_single:
            # pair reps come out flattened layers*heads (B, N, N, layers*heads) -> (B, N, N, edge_dim)
            self.pair_proj = _mlp(
                in_dim=self.esm.num_layers * self.esm.num_heads,
                out_dim=self.cfg.esm_proj_pair_dim,
                hidden_dim=self.cfg.mlp_proj_hidden_dim,
            )
            self.pair_layer_norm = nn.LayerNorm(self.cfg.edge_embed_size)

    def forward(
        self,
        init_node_embed: torch.Tensor,  # (B, N, node_dim)
        init_edge_embed: torch.Tensor,  # (B, N, N, edge_dim)
        aatypes_t: torch.Tensor,  # (B, N)
        chain_index: torch.Tensor,  # (B, N)
        res_mask: torch.Tensor,  # (B, N)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_mask = res_mask[:, None] * res_mask[:, :, None]

        # get ESM single and pair representations
        esm_single, esm_pair = self.esm(
            aatypes=aatypes_t,
            chain_index=chain_index,
            res_mask=res_mask,
        )

        esm_single = esm_single.detach()  # (B, N, nLayers, node_dim)

        # flatten `esm_single` using weighted sum `esm_single_combine`
        weights = self.esm_single_combine.softmax(0)
        w = weights.view(1, 1, -1, 1)  # (1, 1, nLayers+1, 1)
        esm_single = (w * esm_single).sum(dim=2)  # (B, N, node_dim)

        # project weighted single representations
        esm_single = self.seq_proj(esm_single)  # (B, N, node_dim)

        # combine + LayerNorm
        node_embed = 0.5 * (esm_single + init_node_embed)  # (B, N, node_dim)
        node_embed = self.single_layer_norm(node_embed)

        # If generated, process pair representations, otherwise use initial edge embeddings
        if self.cfg.only_single:
            edge_embed = init_edge_embed
        else:
            esm_pair = esm_pair.detach()  # (B, N, N, nLayers*nHeads)

            # project weighted single and flattened pair representations
            esm_pair = self.pair_proj(esm_pair)  # (B, N, N, edge_dim)

            # combine + LayerNorm
            edge_embed = 0.5 * (esm_pair + init_edge_embed)  # (B, N, N, edge_dim)
            edge_embed = self.pair_layer_norm(edge_embed)

        # mask
        node_embed = node_embed * res_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        return node_embed, edge_embed
