from typing import Optional, Tuple

import torch
import torch.nn as nn

from cogeneration.config.base import ModelIPAConfig
from cogeneration.data.rigid_utils import Rigid
from cogeneration.models import ipa_pytorch
from cogeneration.models.ipa_pytorch import TorsionAngles


class AttentionIPATrunk(nn.Module):
    """
    IPA-based Attention Trunk
    Performs Invariant Point Attention, which considers trans/rots between points (residues),
    followed by a transformer block, and linear-ish layers for updating node and edge embeddings.

    The default network matches the public MultiFlow model, and AlphaFold/OpenFold models.

    Performing Backbone Update is optional but on by default.
    Specify `perform_backbone_update=False` to avoid changing backbone coordinates,
    and only update node and edge embeddings.

    Performing Final Edge Update is optional but off by default.
    Specify `perform_final_edge_update=True` to update edge embeddings in the final block.
    You may want to do this if you have downstream models that use edge embeddings.

    Works in nanometer scale, not angstroms.
    """

    def __init__(
        self,
        cfg: ModelIPAConfig,
        perform_final_edge_update: bool = False,
        perform_backbone_update: bool = True,
        predict_torsions: bool = True,
    ):
        super(AttentionIPATrunk, self).__init__()
        self.cfg = cfg
        self.perform_final_edge_update = perform_final_edge_update
        self.perform_backbone_update = perform_backbone_update
        self.predict_torsions = predict_torsions

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

            if self.perform_backbone_update:
                self.trunk[f"bb_update_{b}"] = ipa_pytorch.BackboneUpdate(
                    self.cfg.c_s,
                    use_rot_updates=True,
                )

            # No edge update on the last block, unless specified.
            if b < self.cfg.num_blocks - 1 or self.perform_final_edge_update:
                edge_in = self.cfg.c_z
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self.cfg.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self.cfg.c_z,
                )

        if self.predict_torsions:
            self.torsion_pred = TorsionAngles(self.cfg.c_s, num_torsions=1)

    def forward(
        self,
        init_node_embed: torch.Tensor,
        init_edge_embed: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        diffuse_mask: torch.Tensor,
        curr_rigids_nm: Rigid,
    ) -> Tuple[torch.Tensor, torch.Tensor, Rigid, Optional[torch.Tensor]]:
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

            if self.perform_backbone_update:
                rigid_update = self.trunk[f"bb_update_{b}"](
                    node_embed * node_mask[..., None]
                )

                update_mask = (node_mask * diffuse_mask)[..., None]
                curr_rigids_nm = curr_rigids_nm.compose_q_update_vec(
                    rigid_update, update_mask
                )

            if b < self.cfg.num_blocks - 1 or self.perform_final_edge_update:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        psi_pred = None
        if self.predict_torsions:
            _, psi_pred = self.torsion_pred(node_embed)

        return node_embed, edge_embed, curr_rigids_nm, psi_pred
