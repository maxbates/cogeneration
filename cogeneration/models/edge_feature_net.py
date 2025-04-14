import torch
from torch import nn

from cogeneration.config.base import ModelEdgeFeaturesConfig
from cogeneration.models.embed import calc_distogram, get_index_embedding


class EdgeFeatureNet(nn.Module):
    """
    Simple network to embed edges using distrogram
    """

    def __init__(self, cfg: ModelEdgeFeaturesConfig):
        super(EdgeFeatureNet, self).__init__()
        self.cfg = cfg

        # Embed node features with linear layer
        self.linear_s_p = nn.Linear(self.cfg.c_s, self.cfg.feat_dim)

        # Embed relative position with linear layer
        self.linear_relpos = nn.Linear(self.cfg.feat_dim, self.cfg.feat_dim)

        total_edge_feats = self.cfg.feat_dim * 3 + self.cfg.num_bins * 2
        if self.cfg.embed_chain:
            total_edge_feats += 1
        if self.cfg.embed_diffuse_mask:
            total_edge_feats += 2

        # MLP to embed edge features
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.cfg.c_p),
            nn.ReLU(),
            nn.Linear(self.cfg.c_p, self.cfg.c_p),
            nn.ReLU(),
            nn.Linear(self.cfg.c_p, self.cfg.c_p),
            nn.LayerNorm(self.cfg.c_p),
        )

    def embed_relpos(self, r):
        # AlphaFold 2 Algorithm 4 & 5
        # Based on OpenFold utils/tensor_utils.py
        # Input: (B, N)
        # Output: (B, N, N)
        d = r[:, :, None] - r[:, None, :]
        pos_emb = get_index_embedding(d, self.cfg.feat_dim, max_len=2056)
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
    ):
        num_batch, num_res, _ = node_embed.shape

        # (B, N, c_p]
        p_i = self.linear_s_p(node_embed)
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)

        # (B, N)
        r = (
            torch.arange(num_res, device=node_embed.device)
            .unsqueeze(0)
            .repeat(num_batch, 1)
        )
        relpos_feats = self.embed_relpos(r)

        dist_feats = calc_distogram(
            trans, min_bin=1e-3, max_bin=20.0, num_bins=self.cfg.num_bins
        )
        sc_feats = calc_distogram(
            trans_sc, min_bin=1e-3, max_bin=20.0, num_bins=self.cfg.num_bins
        )

        all_edge_feats = [cross_node_feats, relpos_feats, dist_feats, sc_feats]
        if self.cfg.embed_chain:
            rel_chain = (chain_index[:, :, None] == chain_index[:, None, :]).float()
            all_edge_feats.append(rel_chain[..., None])
        if self.cfg.embed_diffuse_mask:
            diff_feat = self._cross_concat(diffuse_mask[..., None], num_batch, num_res)
            all_edge_feats.append(diff_feat)

        edge_feats = self.edge_embedder(torch.concat(all_edge_feats, dim=-1))
        edge_feats *= edge_mask.unsqueeze(-1)
        return edge_feats
