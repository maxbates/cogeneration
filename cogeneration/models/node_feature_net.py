import torch
from torch import nn

from cogeneration.config.base import ModelNodeFeaturesConfig
from cogeneration.models.embed import get_index_embedding, get_time_embedding
from cogeneration.type.structure import StructureExperimentalMethod


class NodeFeatureNet(nn.Module):
    """
    Simple network for initial representation of structure, sequence, masks, positional embeddings, time embeddings.
    """

    def __init__(self, cfg: ModelNodeFeaturesConfig):
        super(NodeFeatureNet, self).__init__()
        self.cfg = cfg

        # input embedding size is function of what we optionally embed
        embed_size = self.cfg.c_pos_emb + 1 + self.cfg.c_timestep_emb * 2

        if self.cfg.embed_aatype:
            # Always support 21 (20 AA + UNK)
            num_embeddings = 21
            self.aatype_embedding = nn.Embedding(num_embeddings, self.cfg.c_s)
            embed_size += (
                self.cfg.c_s + self.cfg.c_timestep_emb + self.cfg.aatype_pred_num_tokens
            )

        if self.cfg.embed_chain:
            embed_size += self.cfg.c_pos_emb

        if self.cfg.embed_torsions:
            self.torsion_embedding = nn.Linear(14, self.cfg.c_pos_emb)
            embed_size += self.cfg.c_pos_emb

        if self.cfg.embed_structural_method:
            num_methods = len(StructureExperimentalMethod)
            self.structural_method_embedding = nn.Embedding(
                num_methods, self.cfg.c_pos_emb
            )
            embed_size += self.cfg.c_pos_emb

        # TODO consider modulating node embed with hotspots (like 2d constraints) rather than concat + embedding
        if self.cfg.embed_hotspots:
            embed_size += 1  # binary hot spot indicator

        if self.cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.cfg.c_s),
                nn.ReLU(),
                nn.Linear(self.cfg.c_s, self.cfg.c_s),
                nn.ReLU(),
                nn.Linear(self.cfg.c_s, self.cfg.c_s),
                nn.LayerNorm(self.cfg.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.cfg.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps=timesteps[:, 0],
            embedding_dim=self.cfg.c_timestep_emb,
            max_positions=2056,
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
        self,
        so3_t: torch.Tensor,  # (B, 1)
        r3_t: torch.Tensor,  # (B, 1)
        cat_t: torch.Tensor,  # (B, 1)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        chain_index: torch.Tensor,  # (B, N)
        res_index: torch.Tensor,  # (B, N)
        aatypes: torch.Tensor,  # (B, N)
        aatypes_sc: torch.Tensor,  # (B, N, aatype_pred_num_tokens)
        torsions_t: torch.Tensor,  # (B, N, 7, 2)
        structure_method: torch.Tensor,  # (B, 1)
        hot_spots_mask: torch.Tensor,  # (B, N)
    ):
        pos_emb = get_index_embedding(
            res_index,
            embed_size=self.cfg.c_pos_emb,
            max_len=self.cfg.pos_embed_max_len,
            pos_embed_method=self.cfg.pos_embed_method,
        )
        pos_emb = pos_emb * res_mask.unsqueeze(-1)  # (B, N, c_pos_emb)

        input_feats = [
            pos_emb,
            diffuse_mask[..., None],  # (B, N, 1)
            self.embed_t(so3_t, res_mask),  # (B, N, c_timestep_emb)
            self.embed_t(r3_t, res_mask),  # (B, N, c_timestep_emb)
        ]

        if self.cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            input_feats.append(aatypes_sc)

        if self.cfg.embed_chain:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    embed_size=self.cfg.c_pos_emb,
                    max_len=100,  # very unlikely >= 100 chains
                    pos_embed_method=self.cfg.pos_embed_method,
                )  # (B, N, c_pos_emb)
            )

        if self.cfg.embed_torsions:
            input_feats.append(
                self.torsion_embedding(
                    torsions_t.flatten(-2)  # (B, N, 7, 2) -> (B, N, 14)
                )
            )  # (B, N, c_pos_emb)

        if self.cfg.embed_structural_method:
            B, N = res_mask.shape
            method_feature = structure_method.expand(-1, N).long()  # (B, N)
            input_feats.append(
                self.structural_method_embedding(method_feature)
            )  # (B, N, c_pos_emb)

        if self.cfg.embed_hotspots:
            input_feats.append(hot_spots_mask.float().unsqueeze(-1))  # (B, N, 1)

        return self.linear(torch.cat(input_feats, dim=-1))
