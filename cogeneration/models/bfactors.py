import torch
from torch import nn

from cogeneration.config.base import ModelBFactorConfig
from cogeneration.models.attention.ipa_pytorch import Linear


class BFactorModule(nn.Module):
    """Simple B-factor prediction module."""

    def __init__(self, cfg: ModelBFactorConfig) -> None:
        super().__init__()
        self.bfactor = Linear(cfg.c_s, cfg.num_bins)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        return self.bfactor(node_embed)  # (B, N, num_bins)
