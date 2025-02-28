from typing import Tuple

import torch
from torch import nn

from cogeneration.config.base import ModelAAPredConfig
from cogeneration.data.const import MASK_TOKEN_INDEX, NUM_TOKENS
from cogeneration.data.rigid_utils import Rigid


class BaseSequencePredictionNet(nn.Module):
    """
    Abstract base class for sequence prediction networks.
    Defines a common interface for different sequence prediction models.
    """

    def __init__(self):
        super(BaseSequencePredictionNet, self).__init__()

    @property
    def uses_edge_embed(self) -> bool:
        """
        Returns whether the model uses edge embeddings.
        If True, upstream IPA module should update them (i.e. not skip in final block)
        In general, used by IPA transformer, not simple MLPs.
        """
        return False

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
        """
        Forward pass to predict amino acid logits.

        Args:
            node_embed: Single residue embeddings
            aatypes_t: amino acid types at t
            edge_embed: Pairwise residue embeddings
            node_mask: Mask for valid nodes
            edge_mask: Mask for valid edges
            curr_rigids_nm: predicted Rigid body representation (following IPA backbone updates)
            diffuse_mask: Diffusion mask
            chain_index: Chain index

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Predicted logits and predicted amino acid types.
        """
        raise NotImplementedError("Subclasses must implement the forward method.")


class AminoAcidPredictionNet(BaseSequencePredictionNet):
    """
    Linear network to predict amino acid token logits

    This is the simple linear network from public MultiFlow to predict sequence.
    It's simplicity partially explains why MultiFlow was bad at inverse folding.
    See SequenceNet for a more advanced network.

    Set cfg.aatype_pred = False to effectively disable the network, just use one-hot of provided aatypes
    """

    def __init__(self, cfg: ModelAAPredConfig):
        super(AminoAcidPredictionNet, self).__init__()
        self.cfg = cfg

        node_embed_size = self.cfg.c_s

        # Note: ReLU serves to only allow flow to tokens that are more likely
        # Note: `aatype_pred_net` name must update with `replacements` to merge model params.
        self.aatype_pred_net = nn.Sequential(
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, self.cfg.aatype_pred_num_tokens),
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
        init_node_embed: torch.Tensor,
        init_edge_embed: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Run through network to get logits
        pred_logits = self.aatype_pred_net(node_embed)

        # num_tokens can include mask or not. If it does, "mask" the mask logits
        if self.cfg.aatype_pred_num_tokens == NUM_TOKENS + 1:
            pred_logits_wo_mask = pred_logits.clone()
            pred_logits_wo_mask[:, :, MASK_TOKEN_INDEX] = -1e9
            pred_aatypes = torch.argmax(pred_logits_wo_mask, dim=-1)
        else:
            pred_aatypes = torch.argmax(pred_logits, dim=-1)

        return pred_logits, pred_aatypes


class AminoAcidNOOPNet(BaseSequencePredictionNet):
    """
    Network to do nothing to amino acid tokens.
    Effectively disables the amino acid prediction network.
    This is a simple network that just returns the input aatypes.

    In public MultiFlow, this behavior is enabled by setting `cfg.aatype_pred = False`.
    """

    def __init__(self, cfg: ModelAAPredConfig):
        super(AminoAcidNOOPNet, self).__init__()
        self.cfg = cfg

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
        pred_aatypes = aatypes_t
        pred_logits = nn.functional.one_hot(
            pred_aatypes, num_classes=self.cfg.aatype_pred_num_tokens
        ).float()
        return pred_logits, pred_aatypes
