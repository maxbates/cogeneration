import torch
from torch import nn
from torch.nn import functional as F

from cogeneration.data.const import NUM_TOKENS, rigids_ang_to_nm, rigids_nm_to_ang
from cogeneration.data.rigid import create_rigid
from cogeneration.dataset.featurizer import BatchFeaturizer
from cogeneration.models.aa_pred import AminoAcidPredictionNet
from cogeneration.models.attention.attention_trunk import AttentionTrunk
from cogeneration.models.attention.ipa_attention import AttentionIPATrunk
from cogeneration.models.bfactors import BFactorModule
from cogeneration.models.confidence import PLDDTModule
from cogeneration.models.edge_feature_net import EdgeFeatureNet
from cogeneration.models.embed import get_index_embedding, get_time_embedding
from cogeneration.models.esm_combiner import ESMCombinerNetwork
from cogeneration.type.embed import PositionalEmbeddingMethod
from varco.config import VarcoModelConfig
from varco.data import DataCorrupted, ModelPrediction


class BranchFlowModel(nn.Module):
    def __init__(self, cfg: VarcoModelConfig):
        super().__init__()
        self.cfg = cfg

        self.num_aatype_tokens = NUM_TOKENS + 1  # 21: 20 amino acids + X
        self.pos_embed_dim = self.cfg.hyper_params.pos_embed_size
        self.time_embed_dim = self.cfg.hyper_params.timestep_embed_size
        self.node_dim = self.cfg.hyper_params.node_embed_size
        self.edge_dim = self.cfg.hyper_params.edge_embed_size

        self.input_dim = (
            1  # birth_time
            + 1  # motif_mask
            + 1  # chain_idx
            + self.time_embed_dim  # time_embed
            + self.pos_embed_dim  # pos_embed
            + self.num_aatype_tokens  # aatypes_onehot
        )

        # simpler MLP than NodeFeatureNet
        self.node_feature_net = nn.Sequential(
            nn.Linear(self.input_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.node_dim),
            nn.LayerNorm(self.node_dim),
        )

        self.edge_feature_net = EdgeFeatureNet(cfg=self.cfg.edge_features)

        if self.cfg.esm_combiner.enabled:
            self.esm_combiner = ESMCombinerNetwork(cfg=self.cfg.esm_combiner)

        self.trunk = AttentionTrunk(
            cfg=self.cfg.trunk,
            attn_cfg=self.cfg.attention,
        )

        # IPA trunk for structure prediction (trans + rotmats using rigids in nm)
        self.ipa_trunk = AttentionIPATrunk(
            cfg=self.cfg.ipa,
            perform_final_edge_update=self.cfg.seq_trunk.enabled,
            perform_backbone_update=True,
            predict_psi_torsions=False,
            predict_all_torsions=False,
        )

        # Seq trunk
        if self.cfg.seq_trunk.enabled:
            self.seq_trunk = AttentionTrunk(
                cfg=self.cfg.seq_trunk,
                attn_cfg=self.cfg.attention,
            )

        # Base amino acid logits
        self.aatype_pred = AminoAcidPredictionNet(cfg=self.cfg.aa_pred)

        # Insertion amino acid logits
        self.insertion_logits_pred = nn.Sequential(
            nn.Linear(self.node_dim + self.num_aatype_tokens * 2, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, self.num_aatype_tokens),
        )

        # Insertions and deletions
        self.split_rate_pred = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, 1),
        )
        self.split_pooled_log1p_rate_pred = nn.Linear(self.node_dim, 1)
        self.del_logits_pred = nn.Sequential(
            nn.Linear(self.node_dim, self.node_dim),
            nn.ReLU(),
            nn.Linear(self.node_dim, 1),
        )

        # Confidence prediction modules (from cogeneration)
        if self.cfg.bfactor.enabled:
            self.bfactor_net = BFactorModule(cfg=self.cfg.bfactor)
        if self.cfg.plddt.enabled:
            self.plddt_net = PLDDTModule(cfg=self.cfg.plddt)

    def forward(self, batch: DataCorrupted) -> ModelPrediction:
        B, P, _ = batch.trans_t.shape

        valid = batch.valid_mask.float()  # (B, P)
        edge_valid = valid[:, None, :] * valid[:, :, None]  # (B, P, P)

        res_idx = BatchFeaturizer.infer_res_index(
            chain_idx=batch.chain_idx,
            valid_mask=batch.valid_mask,
        )  # (B, P)
        pos_embed = get_index_embedding(
            res_idx,
            embed_size=self.pos_embed_dim,
            max_len=1024,
            pos_embed_method=PositionalEmbeddingMethod.rotary,
        )  # (B, P, pos_embed_dim)
        pos_embed = pos_embed * batch.valid_mask.unsqueeze(-1).float()

        time_embed = get_time_embedding(
            timesteps=batch.t,
            embedding_dim=self.time_embed_dim,
            max_positions=1024,
        )[:, None, :].repeat(
            1, P, 1
        )  # (B, P, time_embed_dim)
        time_embed = time_embed * batch.valid_mask.unsqueeze(-1).float()

        # One-hot encode aatypes_t
        aatypes_onehot = F.one_hot(
            batch.aatypes_t.long(), num_classes=self.num_aatype_tokens
        ).float()  # (B, P, 21)

        # clamp birth_time +inf padding to 1.0
        birth_time = batch.birth_time[:, :, None].float().clamp(0.0, 1.0)

        input_feats = torch.cat(
            [
                birth_time,  # (B, P, 1)
                batch.motif_mask[:, :, None].float(),  # (B, P, 1)
                batch.chain_idx[:, :, None].float(),  # (B, P, 1)
                time_embed,  # (B, P, time_embed_dim)
                pos_embed,  # (B, P, pos_embed_dim)
                aatypes_onehot,  # (B, P, 21)
            ],
            dim=-1,
        )
        node_embed = self.node_feature_net(input_feats)  # (B, P, node_dim)
        node_embed = node_embed * valid.unsqueeze(-1)

        edge_embed = self.edge_feature_net(
            node_embed=node_embed,
            trans=batch.trans_t,
            trans_sc=None,
            edge_mask=edge_valid,
            diffuse_mask=~batch.motif_mask,
            chain_index=batch.chain_idx,
            contact_conditioning=batch.contact_conditioning,  # may be None
        )  # (B, P, P, edge_dim)
        edge_embed = edge_embed * edge_valid.unsqueeze(-1)

        init_node_embed = node_embed
        init_edge_embed = edge_embed
        if self.cfg.esm_combiner.enabled:
            node_embed, edge_embed = self.esm_combiner(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                aatypes_t=batch.aatypes_t,
                chain_index=batch.chain_idx,
                res_mask=torch.ones_like(valid),
                pad_mask=valid,
            )

        # Trunk
        node_embed, edge_embed = self.trunk(
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
            node_embed=node_embed,
            edge_embed=edge_embed,
            node_mask=valid,
            edge_mask=edge_valid,
            rigid=None,
            r3_t=batch.t,
        )

        # IPA trunk predicts structure updates using rigid, nm scale internally
        init_rigids = create_rigid(rots=batch.rotmats_t, trans=batch.trans_t)
        init_rigids_nm = rigids_ang_to_nm(init_rigids)
        node_embed, edge_embed, pred_rigids_nm, _ = self.ipa_trunk(
            node_embed=node_embed,
            edge_embed=edge_embed,
            node_mask=valid,
            edge_mask=edge_valid,
            diffuse_mask=valid,
            curr_rigids_nm=init_rigids_nm,
        )

        # Convert rigid back to angstroms for output
        pred_rigids_ang = rigids_nm_to_ang(pred_rigids_nm)
        pred_trans_1 = pred_rigids_ang.get_trans()  # (B, P, 3)
        pred_rotmats_1 = pred_rigids_ang.get_rots().get_rot_mats()  # (B, P, 3, 3)

        # Seq trunk
        if self.cfg.seq_trunk.enabled:
            node_embed, edge_embed = self.seq_trunk(
                init_node_embed=init_node_embed,
                init_edge_embed=init_edge_embed,
                node_embed=node_embed,
                edge_embed=edge_embed,
                node_mask=valid,
                edge_mask=edge_valid,
                rigid=pred_rigids_nm,
                r3_t=batch.t,
            )

        # Predict amino acid logits
        pred_aatype_logits, _ = self.aatype_pred(
            node_embed=node_embed,
            aatypes_t=batch.aatypes_t,
            edge_embed=edge_embed,
            node_mask=valid,
            edge_mask=edge_valid,
            pred_rigids_nm=pred_rigids_nm,
            diffuse_mask=~batch.motif_mask,
            chain_index=batch.chain_idx,
            init_node_embed=init_node_embed,
            init_edge_embed=init_edge_embed,
        )  # (B, P, K)
        pred_aatype_logits = pred_aatype_logits * valid.unsqueeze(-1).float()

        # Predict insertion amino acid logits
        pred_insertion_logits = self.insertion_logits_pred(
            torch.cat(
                [
                    node_embed,  # (B, P, node_dim)
                    # aatypes_t one-hot (B, P, K)
                    F.one_hot(
                        batch.aatypes_t.long().clamp(0, self.num_aatype_tokens - 1),
                        num_classes=self.num_aatype_tokens,
                    ).float(),
                    # stopgrad pred logits (B, P, K)
                    pred_aatype_logits.detach(),
                ],
                dim=-1,
            )
        )  # (B, P, K)
        pred_insertion_logits = pred_insertion_logits * valid.unsqueeze(-1).float()

        # Predict nonnegative time-independent insertion mass M per token.
        # At sampling time, remaining insertions R_t = M * S(t) where S(t) = 1 - H(t).
        split_mass = F.softplus(self.split_rate_pred(node_embed)).squeeze(-1)  # (B, P)

        # Masked mean pool over alive tokens to predict total insertion mass per example
        valid_count = valid.sum(dim=1, keepdim=True).float().clamp(min=1)  # (B, 1)
        pooled = (node_embed * valid.unsqueeze(-1)).sum(
            dim=1
        ) / valid_count  # (B, model_dim)
        split_pooled_log1p_mass = self.split_pooled_log1p_rate_pred(pooled).squeeze(
            -1
        )  # (B,)

        # Predict deletion logits
        del_logits = self.del_logits_pred(node_embed).squeeze(-1)  # (B, P)

        # Confidence predictions
        pred_bfactor = None
        if self.cfg.bfactor.enabled:
            pred_bfactor = self.bfactor_net(node_embed=node_embed)  # (B, P, num_bins)

        pred_plddt = None
        if self.cfg.plddt.enabled:
            pred_plddt = self.plddt_net(node_embed=node_embed)  # (B, P, num_bins)

        return ModelPrediction(
            pred_trans_1=pred_trans_1,
            pred_rotmats_1=pred_rotmats_1,
            pred_aatype_logits=pred_aatype_logits,
            pred_insertion_logits=pred_insertion_logits,
            pred_split_mass=split_mass,
            pred_split_pooled_log1p_mass=split_pooled_log1p_mass,
            pred_del_logits=del_logits,
            pred_bfactor=pred_bfactor,
            pred_plddt=pred_plddt,
        )
