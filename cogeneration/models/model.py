from typing import Dict, Union

import torch
from torch import nn

from cogeneration.config.base import ModelConfig
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.batch_props import PredBatchProps as pbp
from cogeneration.data.const import rigids_ang_to_nm, rigids_nm_to_ang
from cogeneration.data.rigid import create_rigid
from cogeneration.models.aa_pred import AminoAcidPredictionNet
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.ipa_attention import AttentionIPATrunk
from cogeneration.models.node_feature_net import NodeFeatureNet


class FlowModel(nn.Module):
    """
    Primary model that co-generates sequence and structure.

    Takes as input:
    - a partially masked sequence
    - a protein structure (frames - translations and rotations)

    Sequence is embedding using a language model like ESM
    Structure is embedded using IPA
    Trunk of the model merges representations, runs through folding blocks

    Learns a vector field to generate both structure (frames, i.e. rotations and translations) and sequence
    """

    def __init__(self, cfg: ModelConfig):
        super(FlowModel, self).__init__()
        self.cfg = cfg

        # sub-modules
        self.node_feature_net = NodeFeatureNet(cfg.node_features)
        self.edge_feature_net = EdgeFeatureNet(cfg.edge_features)
        self.attention_ipa_trunk = AttentionIPATrunk(cfg.ipa)
        self.aa_pred_net = AminoAcidPredictionNet(cfg.aa_pred)

    def forward(self, input_feats) -> Dict[Union[bp, pbp], torch.Tensor]:
        node_mask = input_feats[bp.res_mask]
        edge_mask = node_mask[:, None] * node_mask[:, :, None]
        diffuse_mask = input_feats[bp.diffuse_mask]
        chain_index = input_feats[bp.chain_idx]
        res_index = input_feats[bp.res_idx]
        so3_t = input_feats[nbp.so3_t]
        r3_t = input_feats[nbp.r3_t]
        cat_t = input_feats[nbp.cat_t]
        trans_t = input_feats[nbp.trans_t]
        rotmats_t = input_feats[nbp.rotmats_t]
        aatypes_t = input_feats[nbp.aatypes_t].long()  # converts to int64
        trans_sc = input_feats[nbp.trans_sc]
        aatypes_sc = input_feats[nbp.aatypes_sc]

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=node_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            res_index=res_index,
            aatypes=aatypes_t,
            aatypes_sc=aatypes_sc,
        )
        init_edge_embed = self.edge_feature_net(
            node_embed=init_node_embed,
            trans=trans_t,
            trans_sc=trans_sc,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
        )

        # Initial rigids
        curr_rigids_ang = create_rigid(rotmats_t, trans_t)

        # Main trunk
        # TODO - clean up angstrom vs nm... can we just abstract it away?
        node_embed, edge_embed, curr_rigids_nm = self.attention_ipa_trunk(
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,
            curr_rigids_nm=rigids_ang_to_nm(curr_rigids_ang),
        )
        # Convert back to angstroms, get translations and rotations
        curr_rigids_ang = rigids_nm_to_ang(curr_rigids_nm)
        pred_trans = curr_rigids_ang.get_trans()
        pred_rotmats = curr_rigids_ang.get_rots().get_rot_mats()

        # Amino acid prediction
        pred_logits, pred_aatypes = self.aa_pred_net(
            node_embed=node_embed,
            aatypes_t=aatypes_t,
        )

        return {
            pbp.pred_trans: pred_trans,
            pbp.pred_rotmats: pred_rotmats,
            pbp.pred_logits: pred_logits,
            pbp.pred_aatypes: pred_aatypes,
        }
