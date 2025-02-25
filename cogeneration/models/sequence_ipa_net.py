from typing import Tuple

import torch
from data.rigid_utils import Rigid
from models.aa_pred import BaseSequencePredictionNet
from torch import nn

from cogeneration.config.base import ModelSequenceIPANetConfig
from cogeneration.data.const import MASK_TOKEN_INDEX, NUM_TOKENS
from cogeneration.models import ipa_pytorch

# TODO - consider a backwards predictor too, see Discrete Flow Matching
# in practice, this will predict masks, but allow for "correction"

# consider more interesting proability paths than just masking
#   e.g. could you corrupt ranking by an independent language model’s perplexities
#   (requires that LM can predict perplexities for partially masked sequences)


class SequenceIPANet(BaseSequencePredictionNet):
    """
    IPA-Attention style transformer to predict amino acid token logits.

    Based very heavily on AttentionIPATrunk, except does not update backbone.

    Because the backbone is not updated, only the node and edge representations,
    and these representations have already gone through several IPA blocks,
    probably don't need too many blocks before predicting logits.
    """

    def __init__(self, cfg: ModelSequenceIPANetConfig):
        super(SequenceIPANet, self).__init__()
        self.cfg = cfg

        self.trunk = nn.ModuleDict()
        for b in range(self.cfg.ipa.num_blocks):
            self.trunk[f"ipa_{b}"] = ipa_pytorch.InvariantPointAttention(self.cfg.ipa)
            self.trunk[f"ipa_ln_{b}"] = nn.LayerNorm(self.cfg.ipa.c_s)

            tfmr_in = self.cfg.ipa.c_s
            tfmr_layer = torch.nn.TransformerEncoderLayer(
                d_model=tfmr_in,
                nhead=self.cfg.ipa.seq_tfmr_num_heads,
                dim_feedforward=tfmr_in,
                batch_first=True,
                dropout=self.cfg.ipa.transformer_dropout,
                norm_first=False,
            )
            self.trunk[f"seq_tfmr_{b}"] = nn.TransformerEncoder(
                encoder_layer=tfmr_layer,
                num_layers=self.cfg.ipa.seq_tfmr_num_layers,
                enable_nested_tensor=False,
            )
            self.trunk[f"post_tfmr_{b}"] = ipa_pytorch.Linear(
                tfmr_in, self.cfg.ipa.c_s, init="final"
            )
            self.trunk[f"node_transition_{b}"] = ipa_pytorch.StructureModuleTransition(
                c=self.cfg.ipa.c_s
            )

            # no backbone update

            if b < self.cfg.ipa.num_blocks - 1:
                # No edge update on the last block.
                self.trunk[f"edge_transition_{b}"] = ipa_pytorch.EdgeTransition(
                    node_embed_size=self.cfg.ipa.c_s,
                    edge_embed_in=self.cfg.ipa.c_z,
                    edge_embed_out=self.cfg.ipa.c_z,
                )

        # Use final representation to predict amino acid tokens
        self.trunk[f"logits"] = nn.Linear(
            self.cfg.ipa.c_s,
            self.cfg.aatype_pred_num_tokens,
        )

    def forward(
        self,
        node_embed: torch.Tensor,
        aatypes_t: torch.Tensor,
        edge_embed: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        curr_rigids_nm: Rigid,
        diffuse_mask: torch.Tensor,
        chain_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        node_embed = node_embed * node_mask[..., None]
        edge_embed = edge_embed * edge_mask[..., None]

        for b in range(self.cfg.ipa.num_blocks):
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

            # no backbone update

            if b < self.cfg.ipa.num_blocks - 1:
                edge_embed = self.trunk[f"edge_transition_{b}"](node_embed, edge_embed)
                edge_embed *= edge_mask[..., None]

        # predict logits from updated representation
        pred_logits = self.trunk["logits"](node_embed)

        # num_tokens can include mask or not. If it does, "mask" the mask logits
        if self.cfg.aatype_pred_num_tokens == NUM_TOKENS + 1:
            pred_logits_wo_mask = pred_logits.clone()
            pred_logits_wo_mask[:, :, MASK_TOKEN_INDEX] = -1e9
            pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = torch.argmax(pred_logits, dim=-1)

        return pred_logits, pred_aatypes
