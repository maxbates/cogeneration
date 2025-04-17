from typing import Dict, Union

import torch
from torch import nn

from cogeneration.config.base import ModelConfig, ModelSequencePredictionEnum
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.batch_props import PredBatchProps as pbp
from cogeneration.data.const import rigids_ang_to_nm, rigids_nm_to_ang
from cogeneration.data.rigid import create_rigid
from cogeneration.models.aa_pred import AminoAcidNOOPNet, AminoAcidPredictionNet
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.ipa_attention import AttentionIPATrunk
from cogeneration.models.node_feature_net import NodeFeatureNet
from cogeneration.models.sequence_ipa_net import SequenceIPANet


class FlowModel(nn.Module):
    """
    Primary model that co-generates sequence and structure.
    Learns a vector field to generate both structure (frames, i.e. rotations and translations) and sequence

    Takes as input:
    - a (partially masked) sequence
    - a protein structure, represented as frames (i.e. translations and rotations)

    Structure (and sequence) is embedded to get a linear representation (NodeFeatureNet)
    Structure residue distances (edges) are embedded to get a pair representation (EdgeFeatureNet)
    These representations + structure is embedded using IPA (AttentionIPATrunk)
        Trunk of the model merges representations, runs through folding blocks
        Returns a new node embedding, edge embedding, and updated structure

    Specifically, it predicts translations, rotations, and amino acid logits.
    Comparing the predicted translations and rotations to the ground truth structure is main loss.
    Also, the translation and rotation vector field is computed and compared to the ground truth trajectory.
    For ODE (no stochastic paths) the ground truth trajectory is (~simply)
    a linear interpolation from start (noise) to end (structure).

    Optionally we can predict psi torsions. These do not impact the input batch or output translations and rotations
    directly. Instead, the psi torsions are used to construct the atom14 and atom37 rigits, to better pack side chains.
    """

    def __init__(self, cfg: ModelConfig):
        super(FlowModel, self).__init__()
        self.cfg = cfg

        self.node_feature_net = NodeFeatureNet(cfg.node_features)
        self.edge_feature_net = EdgeFeatureNet(cfg.edge_features)

        # sequence prediction sub-module
        if self.cfg.sequence_pred_type == ModelSequencePredictionEnum.NOOP:
            self.aa_pred_net = AminoAcidNOOPNet(cfg.aa_pred)
        elif self.cfg.sequence_pred_type == ModelSequencePredictionEnum.aa_pred:
            self.aa_pred_net = AminoAcidPredictionNet(cfg.aa_pred)
        elif (
            self.cfg.sequence_pred_type == ModelSequencePredictionEnum.sequence_ipa_net
        ):
            self.aa_pred_net = SequenceIPANet(cfg.sequence_ipa_net)
        else:
            raise ValueError(
                f"Invalid sequence prediction type: {self.cfg.sequence_pred_type}"
            )

        # IPA trunk
        # Whether we perform final edge update depends on if used by aa_pred_net
        self.attention_ipa_trunk = AttentionIPATrunk(
            cfg=cfg.ipa,
            perform_final_edge_update=self.aa_pred_net.uses_edge_embed,
            predict_torsions=self.cfg.predict_psi_torsions,
        )

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

        init_rigids_ang = create_rigid(rots=rotmats_t, trans=trans_t)

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

        # Main trunk
        # Note that IPA trunk works in nm scale, rather than angstroms
        node_embed, edge_embed, curr_rigids_nm, psi_pred = self.attention_ipa_trunk(
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,
            curr_rigids_nm=rigids_ang_to_nm(init_rigids_ang),
        )
        # Convert back to angstroms, get translations and rotations
        curr_rigids_ang = rigids_nm_to_ang(curr_rigids_nm)
        pred_trans = curr_rigids_ang.get_trans()
        pred_rotmats = curr_rigids_ang.get_rots().get_rot_mats()

        # Amino acid prediction
        pred_logits, pred_aatypes = self.aa_pred_net(
            node_embed=node_embed,
            aatypes_t=aatypes_t,
            edge_embed=edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            curr_rigids_nm=curr_rigids_nm,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
        )

        return {
            pbp.pred_trans: pred_trans,
            pbp.pred_rotmats: pred_rotmats,
            pbp.pred_psi: psi_pred,
            pbp.pred_logits: pred_logits,
            pbp.pred_aatypes: pred_aatypes,
            # other model outputs
            pbp.node_embed: node_embed,
            pbp.edge_embed: edge_embed,
        }
