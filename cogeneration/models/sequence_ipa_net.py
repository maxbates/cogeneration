from typing import Tuple

import torch
from torch import nn

from cogeneration.config.base import ModelSequenceIPANetConfig
from cogeneration.data.const import MASK_TOKEN_INDEX, NUM_TOKENS
from cogeneration.data.rigid_utils import Rigid
from cogeneration.models.aa_pred import BaseSequencePredictionNet
from cogeneration.models.ipa_attention import AttentionIPATrunk

# TODO(model) - consider a backwards predictor too, see Discrete Flow Matching
#   as an alternative to purity sampling
#   in practice, this will predict masks, but allow for "correction"

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

        # structure already predicted so no backbone updates or torsion predictions
        self.ipa_trunk = AttentionIPATrunk(
            cfg=cfg.ipa,
            perform_backbone_update=False,
            perform_final_edge_update=False,  # final module
            predict_psi_torsions=False,
            predict_all_torsions=False,
        )

        # Use final representation to predict amino acid tokens.
        # Note: ReLU serves to only allow flow to tokens that are more likely.
        self.aatype_pred_net = nn.Sequential(
            nn.Linear(self.cfg.ipa.c_s, self.cfg.c_s),
            nn.ReLU(),
            nn.Linear(self.cfg.c_s, self.cfg.aatype_pred_num_tokens),
        )

    @property
    def uses_edge_embed(self) -> bool:
        return True

    def forward(
        self,
        node_embed: torch.Tensor,
        aatypes_t: torch.Tensor,
        edge_embed: torch.Tensor,
        node_mask: torch.Tensor,
        edge_mask: torch.Tensor,
        pred_rigids_nm: Rigid,
        diffuse_mask: torch.Tensor,
        chain_index: torch.Tensor,
        init_node_embed: torch.Tensor,
        init_edge_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add initial embeddings, assume improve passing though of time / positional embeddings
        # Skip project/layernorm and just pass to IPA.
        if self.cfg.use_init_embed:
            node_embed = 0.5 * (node_embed + init_node_embed)
            edge_embed = 0.5 * (edge_embed + init_edge_embed)

            node_embed = node_embed * node_mask[..., None]
            edge_embed = edge_embed * edge_mask[..., None]

        # run through IPA trunk
        node_embed, edge_embed, _, _ = self.ipa_trunk(
            node_embed=node_embed,
            edge_embed=edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,  # unused; no backbone update
            curr_rigids_nm=pred_rigids_nm,  # pass predicted structure
        )

        # predict logits from updated representation
        pred_logits = self.aatype_pred_net(node_embed)

        # num_tokens can include mask or not. If it does, "mask" the mask logits
        if self.cfg.aatype_pred_num_tokens == NUM_TOKENS + 1:
            pred_logits_wo_mask = pred_logits.clone()
            pred_logits_wo_mask[:, :, MASK_TOKEN_INDEX] = -1e9
            pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = torch.argmax(pred_logits, dim=-1)

        return pred_logits, pred_aatypes
