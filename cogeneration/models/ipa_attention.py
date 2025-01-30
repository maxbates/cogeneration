import torch
import torch.nn as nn

from cogeneration.config.base import ModelIPAConfig
from cogeneration.data.rigid_utils import Rigid
from cogeneration.models import ipa_pytorch


class AttentionIPATrunk(nn.Module):
    """
    IPA-based Attention Trunk

    Works in nanometer scale, not angstroms.
    """

    def __init__(
        self,
        cfg: ModelIPAConfig,
    ):
        super(AttentionIPATrunk, self).__init__()
        self.cfg = cfg

        self.trunk = nn.ModuleDict()
        for b in range(self.cfg.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self.cfg)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self.cfg.c_s)

            tfmr_in = self.cfg.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self.cfg.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=self.cfg.transformer_dropout,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(
                encoder_layer=tfmr_layer,
                num_layers=self.cfg.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self.cfg.c_s, init="final"
            )
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self.cfg.c_s
            )
            self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(
                self.cfg.c_s,
                use_rot_updates=True,
            )

            if b < self.cfg.num_blocks - 1:
                # No edge update on the last block.
                edge_in = self.cfg.c_z
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self.cfg.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self.cfg.c_z,
                )

    def forward(
        self,
        init_node_embed: torch.Tensor,
        init_edge_embed: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        curr_rigids_nm: Rigid,
    ):
        init_node_embed = init_node_embed * node_mask[..., None]
        node_embed = init_node_embed * node_mask[..., None]
        edge_embed = init_edge_embed * edge_mask[..., None]

        for b in range(self.cfg.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](
                node_embed,  # s, single repr
                edge_embed,  # z, pair repr
                curr_rigids_nm,  # r, rigid
                node_mask,
            )
            ipa_embed *= node_mask[..., None]
            node_embed = self.trunk[f"ipa_ln_{b}"](node_embed + ipa_embed)

            seq_tfmr_out = self.trunk[f"seq_tfmr_{b}"](
                node_embed, src_key_padding_mask=(1 - node_mask).to(torch.bool)
            )
            node_embed = node_embed + self.trunk[f"post_tfmr_{b}"](seq_tfmr_out)
            node_embed = self.trunk[f"node_transition_{b}"](node_embed)
            node_embed = node_embed * node_mask[..., None]

            rigid_update = self.trunk[f"bb_update_{b}"](
                node_embed * node_mask[..., None]
            )

            update_mask = (node_mask * diffuse_mask)[..., None]
            curr_rigids_nm = curr_rigids_nm.compose_q_update_vec(
                rigid_update, update_mask
            )

            if b < self.cfg.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        return node_embed, edge_embed, curr_rigids_nm
