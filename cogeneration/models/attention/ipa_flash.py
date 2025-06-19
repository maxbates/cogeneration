from typing import Optional

import torch
from torch import nn

from cogeneration.config.base import ModelIPAConfig
from cogeneration.data.rigid import Rigid
from cogeneration.models.attention.ipa_pytorch import InvariantPointAttention

# Try to import FlashIPA
try:
    from flash_ipa.edge_embedder import EdgeEmbedder, EdgeEmbedderConfig
    from flash_ipa.factorizer import LinearFactorizer
    from flash_ipa.ipa import InvariantPointAttention as FastIPA
except (ImportError, OSError):
    FastIPA = None
    EdgeEmbedder = None
    EdgeEmbedderConfig = None
    LinearFactorizer = None


class _EdgeToZFactor(nn.Module):
    """Convert dense edge bias (B,L,L,Cz) to two rank-R factors (B,L,R,Cz)"""

    def __init__(self, cz: int, rank: int):
        super().__init__()
        self.factorizer = LinearFactorizer(
            in_L=1,  # factoriser treats each row independently
            in_D=cz,
            target_rank=rank,
            target_inner_dim=cz,
        )

    def forward(
        self,
        edge: torch.Tensor,  # (B,L,L,Cz)
    ):
        z1, z2 = self.factorizer(
            edge.to(self.factorizer.weight.dtype)
        )  # (B*D,L,R) each
        B, L, R, Cz = z1.shape
        z1 = z1.view(B, L, R, Cz)
        z2 = z2.view(B, L, R, Cz)
        return z1, z2


class MaybeFlashIPA(nn.Module):
    """
    Wrapper that uses FlashIPA when it’s installed and
    cfg.use_flash_attn is True, otherwise falls back.
    """

    def __init__(self, cfg: ModelIPAConfig):
        super().__init__()
        self.cfg = cfg

        use_flash_ipa = (
            torch.cuda.is_available()
            and getattr(cfg, "use_flash_attn", False)
            and FastIPA is not None
        )
        self.is_flash = bool(use_flash_ipa)
        self.impl = FastIPA(cfg) if self.is_flash else InvariantPointAttention(cfg)

        if self.is_flash:
            if self.cfg.z_factor_rank <= 0:
                raise ValueError("z_factor_rank must be > 0 when use_flash_attn=True")
            self._edge_to_z = _EdgeToZFactor(self.cfg.c_z, self.cfg.z_factor_rank)

    def forward(
        self,
        s: torch.Tensor,  # single repr, [*, N_res, C_s]
        z: Optional[torch.Tensor],  # pair repr, [*, N_res, N_res, C_z]
        r: Rigid,
        mask: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ):
        if self.is_flash:
            z1, z2 = self._edge_to_z(z)  # (B,N_res,R,C_z)
            edge_arg = None  # FlashIPA ignores dense bias

            return self.impl(
                s,  # (B,N_res,Cs)
                edge_arg,  # dense edge bias or None
                z1,
                z2,  # FlashIPA factors or None
                r,
                mask.bool(),  # (B,N_res)
            )
        else:
            return self.impl(
                s,  # (B,N_res,Cs)
                z,  # (B,N_res,N_res,Cz) or None
                r,
                mask,  # (B,N_res)
                attention_mask,  # (B,N_res,N_res) or None
            )
