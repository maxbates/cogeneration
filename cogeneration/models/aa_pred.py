from typing import Tuple

import torch
from torch import nn

from cogeneration.config.base import ModelAAPredConfig
from cogeneration.data.const import MASK_TOKEN_INDEX, NUM_TOKENS

# TODO - consider a backwards predictor too, see Discrete Flow Matching
# in practice, this will predict masks, but allow for "correction"

# consider moreinteresting proability paths than just masking
#   e.g. could you corrupt ranking by an independent language model’s perplexities
#   (requires that LM can predict perplexities for partially masked sequences)


class AminoAcidPredictionNet(nn.Module):
    """
    Linear network to predict amino acid token logits

    Set cfg.aatype_pred = False to effectively disable the network, just use one-hot of provided aatypes
    """

    def __init__(self, cfg: ModelAAPredConfig):
        super(AminoAcidPredictionNet, self).__init__()
        self.cfg = cfg

        if self.cfg.aatype_pred:
            node_embed_size = self.cfg.c_s

            # Note: ReLU serves to only allow flow to tokens that are more likely
            self.aatype_pred_net = nn.Sequential(
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, node_embed_size),
                nn.ReLU(),
                nn.Linear(node_embed_size, self.cfg.aatype_pred_num_tokens),
            )

    def forward(
        self,
        node_embed,
        aatypes_t,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # If not predicting amino acids, return the input aatypes
        if not self.cfg.aatype_pred:
            pred_aatypes = aatypes_t
            pred_logits = nn.functional.one_hot(
                pred_aatypes, num_classes=self.cfg.aatype_pred_num_tokens
            ).float()
            return pred_logits, pred_aatypes

        # Otherwise, run through network to get logits
        pred_logits = self.aatype_pred_net(node_embed)

        # num_tokens can include mask or not. If it does, "mask" the mask logits
        if self.cfg.aatype_pred_num_tokens == NUM_TOKENS + 1:
            pred_logits_wo_mask = pred_logits.clone()
            pred_logits_wo_mask[:, :, MASK_TOKEN_INDEX] = -1e9
            pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = torch.argmax(pred_logits, dim=-1)

        return pred_logits, pred_aatypes
