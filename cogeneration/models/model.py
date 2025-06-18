from torch import nn

from cogeneration.config.base import ModelConfig, ModelSequencePredictionEnum
from cogeneration.data.const import rigids_ang_to_nm, rigids_nm_to_ang
from cogeneration.data.rigid import create_rigid
from cogeneration.models.aa_pred import AminoAcidNOOPNet, AminoAcidPredictionNet
from cogeneration.models.attention_trunk import AttentionTrunk
from cogeneration.models.bfactors import BFactorModule
from cogeneration.models.confidence import PLDDTModule
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.esm_combiner import ESMCombinerNetwork
from cogeneration.models.ipa_attention import AttentionIPATrunk
from cogeneration.models.node_feature_net import NodeFeatureNet
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import ModelPrediction
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp


class FlowModel(nn.Module):
    """
    Complete model for protein sequence-structure cogeneration with flow matching.

    Several portions of the model are optional or highly configurable.

    The model can work across domains of sequence, rotations, translations, and torsions.
    The input can be:
     - nothing (unconditional)
     - sequence (forward folding)
     - structure (inverse folding)
     - or portions of either (inpainting)
    """

    def __init__(self, cfg: ModelConfig):
        super(FlowModel, self).__init__()
        self.cfg = cfg

        # Input + conditioning embedding
        self.node_feature_net = NodeFeatureNet(cfg.node_features)
        self.edge_feature_net = EdgeFeatureNet(cfg.edge_features)

        # ESM + combiner
        if self.cfg.esm_combiner.enabled:
            self.esm_combiner = ESMCombinerNetwork(cfg=self.cfg.esm_combiner)

        # Attention Trunk
        if self.cfg.trunk.enabled:
            self.trunk = AttentionTrunk(
                cfg=self.cfg.trunk,
                attn_cfg=self.cfg.attention,
            )

        # IPA trunk
        self.ipa_trunk = AttentionIPATrunk(
            cfg=cfg.ipa,
            perform_final_edge_update=self.cfg.seq_trunk.enabled,
            predict_psi_torsions=self.cfg.predict_psi_torsions,
            predict_all_torsions=self.cfg.predict_all_torsions,
        )

        # B-factor / confidence prediction
        if self.cfg.bfactor.enabled:
            self.bfactor_net = BFactorModule(cfg=cfg.bfactor)
        if self.cfg.plddt.enabled:
            self.plddt_net = PLDDTModule(cfg=cfg.plddt)

        # Seq trunk
        if self.cfg.seq_trunk.enabled:
            self.seq_trunk = AttentionTrunk(
                cfg=self.cfg.seq_trunk,
                attn_cfg=self.cfg.attention,
            )

        # Seq prediction
        if self.cfg.sequence_pred_type == ModelSequencePredictionEnum.NOOP:
            self.aa_pred_net = AminoAcidNOOPNet(cfg.aa_pred)
        elif self.cfg.sequence_pred_type == ModelSequencePredictionEnum.aa_pred:
            self.aa_pred_net = AminoAcidPredictionNet(cfg.aa_pred)
        else:
            raise ValueError(
                f"Invalid sequence prediction type: {self.cfg.sequence_pred_type}"
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
        torsions_t = batch[nbp.torsions_t]
        aatypes_t = batch[nbp.aatypes_t].long()
        trans_sc = batch[nbp.trans_sc]
        aatypes_sc = batch[nbp.aatypes_sc]
        structure_method = batch[bp.structure_method]

        init_rigids_ang = create_rigid(rots=rotmats_t, trans=trans_t)
        init_rigids_nm = rigids_ang_to_nm(init_rigids_ang)

        # Initial node and edge embeddings
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
            torsions_t=torsions_t,
            structure_method=structure_method,
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
            )
        else:
            node_embed, edge_embed = init_node_embed, init_edge_embed

        # Trunk
        if self.cfg.trunk.enabled:
            node_embed, edge_embed = self.trunk(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                edge_mask=edge_mask,
                rigid=init_rigids_nm,
                r3_t=r3_t,
            )

        # IPA trunk - note works in nm scale, not angstroms
        node_embed, edge_embed, pred_rigids_nm, pred_torsions = self.ipa_trunk(
            node_embed=node_embed,
            edge_embed=edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,
            curr_rigids_nm=init_rigids_nm,
        )
        # Convert rigid back to angstroms
        pred_rigids_ang = rigids_nm_to_ang(pred_rigids_nm)
        pred_trans = pred_rigids_ang.get_trans()
        pred_rotmats = pred_rigids_ang.get_rots().get_rot_mats()

        # B-factor prediction
        pred_bfactor = None
        if self.cfg.bfactor.enabled:
            pred_bfactor = self.bfactor_net(node_embed=node_embed)

        # pLDDT prediction
        pred_lddt = None
        if self.cfg.plddt.enabled:
            pred_lddt = self.plddt_net(node_embed=node_embed)

        # Seq trunk
        if self.cfg.seq_trunk.enabled:
            node_embed, edge_embed = self.seq_trunk(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=node_mask,
                edge_mask=edge_mask,
                rigid=pred_rigids_nm,
                r3_t=r3_t,
            )

        # Sequence prediction
        pred_logits, pred_aatypes = self.aa_pred_net(
            node_embed=node_embed,
            aatypes_t=aatypes_t,
            edge_embed=edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            pred_rigids_nm=pred_rigids_nm,
            diffuse_mask=diffuse_mask,
            chain_index=chain_index,
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
        )

        return {
            pbp.pred_trans: pred_trans,
            pbp.pred_rotmats: pred_rotmats,
            pbp.pred_torsions: pred_torsions,
            pbp.pred_logits: pred_logits,
            pbp.pred_aatypes: pred_aatypes,
            pbp.pred_bfactor: pred_bfactor,
            pbp.pred_lddt: pred_lddt,
            # other model outputs
            pbp.node_embed: node_embed,
            pbp.edge_embed: edge_embed,
        }
