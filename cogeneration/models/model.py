from torch import nn

from cogeneration.config.base import ModelConfig, ModelSequencePredictionEnum
from cogeneration.data.const import rigids_ang_to_nm, rigids_nm_to_ang
from cogeneration.data.rigid import create_rigid
from cogeneration.models.aa_pred import AminoAcidNOOPNet, AminoAcidPredictionNet
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.esm_combiner import ESMCombinerNetwork
from cogeneration.models.ipa_attention import AttentionIPATrunk
from cogeneration.models.node_feature_net import NodeFeatureNet
from cogeneration.models.sequence_ipa_net import SequenceIPANet
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import ModelPrediction
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp


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

        if self.cfg.esm_combiner.enabled:
            self.esm_combiner = ESMCombinerNetwork(cfg=self.cfg.esm_combiner)

        # IPA trunk
        # Whether we perform final edge update depends on if used by aa_pred_net
        self.attention_ipa_trunk = AttentionIPATrunk(
            cfg=cfg.ipa,
            perform_final_edge_update=self.aa_pred_net.uses_edge_embed,
            predict_torsions=self.cfg.predict_psi_torsions,
        )

    def forward(self, batch: NoisyFeatures) -> ModelPrediction:
        res_mask = batch[bp.res_mask]
        node_mask = res_mask
        edge_mask = res_mask[:, None] * res_mask[:, :, None]

        diffuse_mask = batch[bp.diffuse_mask]
        motif_mask = batch.get(bp.motif_mask, None)
        # embed `1-motif_mask` instead of `diffuse_mask` if defined
        # for inpainting with guidance: `diffuse_mask == 1` for backbone update so `motif_mask` more meaningful
        embed_diffuse_mask = (
            (1 - motif_mask)
            if (motif_mask is not None and motif_mask.any())
            else diffuse_mask
        )

        chain_index = batch[bp.chain_idx]
        res_index = batch[bp.res_idx]
        so3_t = batch[nbp.so3_t]
        r3_t = batch[nbp.r3_t]
        cat_t = batch[nbp.cat_t]
        trans_t = batch[nbp.trans_t]
        rotmats_t = batch[nbp.rotmats_t]
        aatypes_t = batch[nbp.aatypes_t].long()
        trans_sc = batch[nbp.trans_sc]
        aatypes_sc = batch[nbp.aatypes_sc]

        init_rigids_ang = create_rigid(rots=rotmats_t, trans=trans_t)

        # Initialize node and edge embeddings
        init_node_embed = self.node_feature_net(
            so3_t=so3_t,
            r3_t=r3_t,
            cat_t=cat_t,
            res_mask=res_mask,
            diffuse_mask=embed_diffuse_mask,
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
            diffuse_mask=embed_diffuse_mask,
            chain_index=chain_index,
        )

        # Optionally run ESM + combine with node and edge embeddings
        if self.cfg.esm_combiner.enabled:
            node_embed, edge_embed = self.esm_combiner(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                aatypes_t=aatypes_t,
                chain_index=chain_index,
                res_mask=res_mask,
                r3_t=r3_t,
            )
        else:
            node_embed, edge_embed = init_node_embed, init_edge_embed

        # Main trunk
        # Note that IPA trunk works in nm scale, rather than angstroms
        node_embed, edge_embed, curr_rigids_nm, psi_pred = self.attention_ipa_trunk(
            node_embed=node_embed,
            edge_embed=edge_embed,
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
