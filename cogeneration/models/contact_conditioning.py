from typing import Optional

import torch
from torch import nn

from cogeneration.config.base import ModelContactConditioningConfig
from cogeneration.models.attention.ipa_pytorch import Linear
from cogeneration.models.embed import FourierEmbedding


class ContactConditioning(nn.Module):
    """
    Modulate pair embeddings with optional contact constraints.
    Returns original edge embeddings if no contact constraints are provided.

    Unlike boltz, we don't specify the connection type, assume everything is a contact.
    So, we just embed the normalized distance thresholds.
    """

    def __init__(self, cfg: ModelContactConditioningConfig):
        super().__init__()
        self.cfg = cfg

        if self.cfg.enabled:
            self.fourier = FourierEmbedding(self.cfg.fourier_dim)
            self.encoder = Linear(
                self.cfg.fourier_dim, self.cfg.pair_dim, init="normal"
            )
            self.layer_norm = nn.LayerNorm(self.cfg.pair_dim)

    def forward(
        self,
        edge_embed: torch.Tensor,  # (B, N, N, c_p)
        contact: Optional[torch.Tensor],  # (B, N, N)
    ) -> torch.Tensor:
        if not self.cfg.enabled:
            return edge_embed
        if contact is None or not torch.any(contact > 0):
            # Keep a graph connection to this module's params so DDP doesn't treat them as unused
            # on batches without contact constraints.
            dummy = 0.0 * (
                self.encoder.weight.sum()
                + self.encoder.bias.sum()
                + self.layer_norm.weight.sum()
                + self.layer_norm.bias.sum()
            )
            return edge_embed + dummy

        # normalize distances to [0, 1]
        contacts_norm = (contact - self.cfg.dist_min) / (
            self.cfg.dist_max - self.cfg.dist_min
        )
        contacts_norm = contacts_norm.clamp(0, 1)

        emb = self.fourier(contacts_norm.flatten().flatten()).reshape(
            contacts_norm.shape + (-1,)
        )
        emb = self.encoder(emb)
        emb = self.layer_norm(emb)

        return edge_embed + emb
