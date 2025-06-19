from typing import Optional, Tuple

import torch
import torch.nn as nn

from cogeneration.config.base import ModelIPAConfig
from cogeneration.data.rigid_utils import Rigid
from cogeneration.models.attention.ipa_flash import MaybeFlashIPA
from cogeneration.models.attention.ipa_pytorch import (
    BackboneUpdate,
    EdgeTransition,
    Linear,
    StructureModuleTransition,
    TorsionAngles,
)


class AttentionIPATrunk(nn.Module):
    """
    Invariant Point Attention Trunk
    IPA considers trans/rots between points (residues), followed by a transformer block,
    and linear-ish layers for updating node and edge embeddings.

    The default network matches the public MultiFlow model, and AlphaFold/OpenFold models.
    (Though instead of taking templates / MSA evoformer as input,
    we pass the node / edge embeddings we generate)

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
        predict_psi_torsions: bool = True,
        predict_all_torsions: bool = True,
        num_blocks: Optional[int] = None,
    ):
        super(AttentionIPATrunk, self).__init__()
        self.cfg = cfg
        self.perform_final_edge_update = perform_final_edge_update
        self.perform_backbone_update = perform_backbone_update
        self.predict_psi_torsions = predict_psi_torsions
        self.predict_all_torsions = predict_all_torsions
        self.num_blocks = num_blocks if num_blocks is not None else cfg.num_blocks

        self.trunk = nn.ModuleDict()
        for b in range(self.num_blocks):
            self.trunk[f"ipa_{b}"] = MaybeFlashIPA(self.cfg)
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
            self.trunk[f"post_tfmr_{b}"] = Linear(tfmr_in, self.cfg.c_s, init="final")
            self.trunk[f"node_transition_{b}"] = StructureModuleTransition(
                c=self.cfg.c_s
            )

            if self.perform_backbone_update:
                self.trunk[f"bb_update_{b}"] = BackboneUpdate(
                    self.cfg.c_s,
                    use_rot_updates=True,
                )

            # No edge update on the last block, unless specified.
            if b < self.num_blocks - 1 or self.perform_final_edge_update:
                edge_in = self.cfg.c_z
                self.trunk[f"edge_transition_{b}"] = EdgeTransition(
                    node_embed_size=self.cfg.c_s,
                    edge_embed_in=edge_in,
                    edge_embed_out=self.cfg.c_z,
                )

        # torsions optional
        if self.num_torsions > 0:
            self.torsion_pred = TorsionAngles(
                self.cfg.c_s, num_torsions=self.num_torsions
            )

    @property
    def num_torsions(self) -> int:
        """
        Determine number of torsions to predict.
        """
        if self.predict_all_torsions:
            return 7
        if self.predict_psi_torsions:
            return 1
        return 0

    def forward(
        self,
        node_embed: torch.Tensor,  # (B, N, c_s)
        edge_embed: torch.Tensor,  # (B, N, N, c_z)
        node_mask: torch.Tensor,  # (B, N)
        edge_mask: torch.Tensor,  # (B, N, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        curr_rigids_nm: Rigid,
    ) -> Tuple[torch.Tensor, torch.Tensor, Rigid, Optional[torch.Tensor]]:
        node_embed = node_embed * node_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        for b in range(self.num_blocks):
            ipa_embed = self.trunk[f"ipa_{b}"](
                s=node_embed,  # s = single repr
                z=edge_embed,  # z = pair repr
                r=curr_rigids_nm,  # r = rigid
                mask=node_mask,
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

                # backbone update only performed on `res_mask & diffuse_mask`
                update_mask = (node_mask * diffuse_mask)[..., None]
                curr_rigids_nm = curr_rigids_nm.compose_q_update_vec(
                    rigid_update, update_mask
                )

            if b < self.num_blocks - 1 or self.perform_final_edge_update:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        torsion_pred = None
        if self.num_torsions > 0:
            _, torsion_pred = self.torsion_pred(node_embed)  # (B, N, K * 2)
            B, N = torsion_pred.shape[:2]
            torsion_pred = torsion_pred.reshape(
                B, N, self.num_torsions, 2
            )  # (B, N, K, 2)
            torsion_pred = torsion_pred * node_mask[..., None, None]

        return node_embed, edge_embed, curr_rigids_nm, torsion_pred
