from typing import Tuple

import torch
from torch import nn

from cogeneration.config.base import ModelSequenceIPANetConfig
from cogeneration.data.const import MASK_TOKEN_INDEX, NUM_TOKENS
from cogeneration.data.rigid_utils import Rigid
from cogeneration.models.aa_pred import BaseSequencePredictionNet
from cogeneration.models.ipa_attention import AttentionIPATrunk

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

        self.ipa_trunk = AttentionIPATrunk(
            cfg=cfg.ipa,
            perform_backbone_update=False,  # no backbone update
            perform_final_edge_update=False,  # no edge update on last block, not used further
            predict_torsions=False,  # no torsion prediction, leave to structure module
        )

        # Use final representation to predict amino acid tokens
        # TODO - consider also using edge features to predict logits.
        self.aatype_pred_net = nn.Linear(
            self.cfg.ipa.c_s,
            self.cfg.aatype_pred_num_tokens,
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
        curr_rigids_nm: Rigid,
        diffuse_mask: torch.Tensor,
        chain_index: torch.Tensor,
        init_node_embed: torch.Tensor,
        init_edge_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add initial embeddings, may improve passing though of time / positional embeddings
        if self.cfg.use_init_embed:
            node_embed = 0.5 * (node_embed + init_node_embed)
            edge_embed = 0.5 * (edge_embed + init_edge_embed)

            node_embed = node_embed * node_mask[..., None]
            edge_embed = edge_embed * edge_mask[..., None]

        # run through IPA trunk
        node_embed, edge_embed, curr_rigids_nm, _ = self.ipa_trunk(
            init_node_embed=node_embed,
            init_edge_embed=edge_embed,
            node_mask=node_mask,
            edge_mask=edge_mask,
            diffuse_mask=diffuse_mask,
            curr_rigids_nm=curr_rigids_nm,
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
