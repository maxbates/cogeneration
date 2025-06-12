import torch
from torch import nn

from cogeneration.config.base import ModelBFactorConfig


class BFactorModule(nn.Module):
    """Simple B-factor prediction module."""

    def __init__(self, cfg: ModelBFactorConfig) -> None:
        super().__init__()
        self.bfactor = nn.Linear(cfg.c_s, cfg.num_bins)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        return self.bfactor(node_embed)  # (B, N, num_bins)
