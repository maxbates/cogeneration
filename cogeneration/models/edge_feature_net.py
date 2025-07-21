from typing import Optional

import torch
from torch import nn

from cogeneration.config.base import ModelEdgeFeaturesConfig
from cogeneration.models.contact_conditioning import ContactConditioning
from cogeneration.models.embed import calc_distogram, get_index_embedding


class EdgeFeatureNet(nn.Module):
    """
    Embed edges using distrogram, plus self-conditioned dist, chain, masks, dist constraints, etc.
    """

    def __init__(self, cfg: ModelEdgeFeaturesConfig):
        super(EdgeFeatureNet, self).__init__()
        self.cfg = cfg

        # linear layer for node features
        self.linear_s_p = nn.Linear(self.cfg.c_s, self.cfg.feat_dim)

        # linear layer for relative position
        self.linear_relpos = nn.Linear(self.cfg.feat_dim, self.cfg.feat_dim)

        total_edge_feats = self.cfg.feat_dim * 3 + self.cfg.num_bins * 2
        if self.cfg.embed_chain:
            total_edge_feats += 1
        if self.cfg.embed_diffuse_mask:
            total_edge_feats += 2

        # MLP to embed edge feats
        # (B, N, N, total_edge_feats) -> (B, N, N, c_p)
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.cfg.c_p),
            nn.ReLU(),
            nn.Linear(self.cfg.c_p, self.cfg.c_p),
            nn.ReLU(),
            nn.Linear(self.cfg.c_p, self.cfg.c_p),
            nn.LayerNorm(self.cfg.c_p),
        )

        # contact conditioning
        self.contact_conditioning = ContactConditioning(
            cfg=self.cfg.contact_conditioning
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: (B, N)
        # Output: (B, N, N)
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(
            d,
            embed_size=self.cfg.feat_dim,
            max_len=self.cfg.pos_embed_max_len,
            pos_embed_method=self.cfg.pos_embed_method,
        )
        return self.linear_relpos(pos_emb)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return (
            torch.cat(
                [
                    torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
                    torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
                ],
                dim=-1,
            )
            .float()
            .reshape([num_batch, num_res, num_res, -1])
        )

    def forward(
        self,
        node_embed: torch.Tensor,  # (B, N, c_s)
        trans: torch.Tensor,  # (B, N, 3)
        trans_sc: torch.Tensor,  # (B, N, 3)
        edge_mask: torch.Tensor,  # (B, N, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        chain_index: torch.Tensor,  # (B, N)
        contact_conditioning: Optional[torch.Tensor],  # (B, N, N)
    ):
        num_batch, num_res, _ = node_embed.shape

        p_i = self.linear_s_p(node_embed)
        cross_node_feats = self._cross_concat(
            p_i, num_batch, num_res
        )  # (B, N, N, feat_dim)

        r = (
            torch.arange(num_res, device=node_embed.device)
            .unsqueeze(0)
            .repeat(num_batch, 1)
        )
        relpos_feats = self.embed_relpos(r)  # (B, N, N, feat_dim)

        dist_feats = calc_distogram(
            trans, min_bin=1e-4, max_bin=20.0, num_bins=self.cfg.num_bins
        )  # (B, N, N, num_bins)
        sc_feats = calc_distogram(
            trans_sc, min_bin=1e-4, max_bin=20.0, num_bins=self.cfg.num_bins
        )  # (B, N, N, num_bins)

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self.cfg.embed_chain:
            rel_chain = (chain_index[:, :, None] == chain_index[:, None, :]).float()
            all_edge_feats.append(rel_chain[..., None])
        if self.cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= edge_mask.unsqueeze(-1)

        # modulate edge features with contact conditioning
        edge_feats = self.contact_conditioning(edge_feats, contact_conditioning)

        return edge_feats  # (B, N, N, c_p)
