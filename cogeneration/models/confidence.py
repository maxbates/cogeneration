import torch
from torch import nn

from cogeneration.config.base import ModelPLDDTConfig


class PLDDTModule(nn.Module):
    """Simple pLDDT prediction module, predicts logits over pLDDT bins per residue"""

    def __init__(self, cfg: ModelPLDDTConfig) -> None:
        super().__init__()
        self.plddt = nn.Linear(cfg.c_s, cfg.num_bins)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        return self.plddt(node_embed)  # (B, N, num_bins)
