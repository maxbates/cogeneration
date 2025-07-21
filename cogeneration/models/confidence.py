from typing import Tuple

import torch
from torch import nn

from cogeneration.config.base import ModelPAEConfig, ModelPLDDTConfig
from cogeneration.models.attention.ipa_pytorch import Linear


def compute_aggregated_metric(logits: torch.Tensor, end: float = 1.0) -> torch.Tensor:
    """Expectation value of a binned distribution (pLDDT → end = 1, PAE → end = 32)."""
    num_bins = logits.shape[-1]
    bin_width = end / num_bins
    centers = torch.arange(0.5 * bin_width, end, bin_width, device=logits.device)
    probs = torch.softmax(logits, dim=-1)
    view_shape = (1,) * (probs.ndim - 1) + (-1,)  # broadcast centres
    return torch.sum(probs * centers.view(*view_shape), dim=-1)


def tm_function(
    d: torch.Tensor,  # (B, N, N) distance / PAE values
    n_res: torch.Tensor,  # (B,)
) -> torch.Tensor:
    """AlphaFold TM-score kernel"""
    d0 = 1.24 * (torch.clamp(n_res.float(), min=19) - 15) ** (1 / 3) - 1.8  # (B,)
    d0 = d0[:, None, None]  # (B, 1, 1)
    return 1.0 / (1.0 + (d / d0) ** 2)  # (B, N, N)


def compute_ptms(
    pae_logits: torch.Tensor,  # (B, N, N, num_bins)
    chain_idx: torch.Tensor,  # (B, N)
    mask: torch.Tensor,  # (B, N)
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pTM and iPTM from PAE logits.
    pTM = best-residue average over all residue pairs.
    iPTM = best-residue average restricted to inter-chain pairs.
    """
    num_bins = pae_logits.shape[-1]
    bin_width = 32.0 / num_bins

    n_res = mask.sum(dim=-1, keepdim=False)  # (B,)
    pae_values = (
        torch.arange(0.5 * bin_width, 32.0, bin_width, device=pae_logits.device)
        .unsqueeze(0)  # (1, num_bins)
        .expand(n_res.size(0), -1)  # (B, num_bins)
        .unsqueeze(1)  # (B, 1, num_bins)
    )

    # Expected TM-score for each pair
    tm_values = tm_function(pae_values, n_res).unsqueeze(1)  # (B, 1, 1, num_bins)
    tm_exp = torch.sum(
        torch.softmax(pae_logits, dim=-1) * tm_values, dim=-1
    )  # (B, N, N)

    pair_mask = mask.unsqueeze(2) * mask.unsqueeze(1)  # (B, N, N)

    # pTM
    per_res_ptm = (tm_exp * pair_mask).sum(-1) / (pair_mask.sum(-1) + 1e-5)  # (B, N)
    ptm = per_res_ptm.max(dim=-1).values  # (B,)

    # iPTM
    inter_mask = pair_mask * (chain_idx.unsqueeze(2) != chain_idx.unsqueeze(1))
    per_res_iptm = (tm_exp * inter_mask).sum(-1) / (inter_mask.sum(-1) + 1e-5)  # (B, N)
    iptm = per_res_iptm.max(dim=-1).values  # (B,)

    return ptm, iptm


class PLDDTModule(nn.Module):
    """Simple pLDDT prediction module, predicts logits over pLDDT bins per residue"""

    def __init__(self, cfg: ModelPLDDTConfig) -> None:
        super().__init__()
        self.plddt = Linear(cfg.c_s, cfg.num_bins)

    def forward(self, node_embed: torch.Tensor) -> torch.Tensor:
        return self.plddt(node_embed)  # (B, N, num_bins)


class PAEModule(nn.Module):
    """Predicts logits over PAE bins for each residue pair, and computes pTM and iPTM."""

    def __init__(self, cfg: ModelPAEConfig) -> None:
        super().__init__()
        self.linear = Linear(cfg.c_z, cfg.num_bins)

    def forward(
        self,
        pair_embed: torch.Tensor,
        chain_idx: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        pae_logits = self.linear(pair_embed)  # (B, N, N, num_bins)
        ptm, iptm = compute_ptms(pae_logits, chain_idx=chain_idx, mask=mask)
        return pae_logits, ptm, iptm
