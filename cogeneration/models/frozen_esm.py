from dataclasses import dataclass, field
from functools import lru_cache
from typing import Callable, Dict, Optional, Tuple, Any

import esm
import torch
import tree
from esm.data import Alphabet
from torch import nn

from cogeneration.data.residue_constants import restypes_with_x


@dataclass
class EsmRegistry:
    """Singleton holding factories that return `(model, alphabet)` pairs."""

    registry: Dict[str, Callable[[], Tuple[nn.Module, any]]] = field(
        default_factory=dict
    )

    def register(self, key: str, factory: Callable[[], Tuple[nn.Module, any]]) -> None:
        self.registry[key] = factory

    def load(self, key: str) -> Tuple[nn.Module, any]:
        if key not in self.registry:
            raise KeyError(f"Model key '{key}' is not registered in EsmRegistry")
        return self.registry[key]()

    # helper to register a dummy model for testing
    def register_dummy(
        self,
        key: str = "dummy",
        embedding_size: int = 4,
        n_layers: int = 1,
        n_heads: int = 1,
    ) -> None:
        """Registers a MinimalRandomESM under `key`."""

        def factory():
            model = _MinimalRandomESM(embedding_size, n_layers, n_heads)
            # use the small 8M model's alphabet
            _, alphabet = esm.pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
            return model, alphabet

        self.register(key, factory)


class _MinimalRandomESM(nn.Module):
    """Dummy ESM-like model that returns random representations/attentions"""

    def __init__(self, embed_dim: int, n_layers: int, n_heads: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = n_layers
        self.num_heads = n_heads

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,  # (B, L)
        repr_layers: Any, # ignored, use num_layers
        need_head_weights: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        B, L = tokens.shape
        reps: Dict[int, torch.Tensor] = {}
        for layer in tuple(range(self.num_layers + 1)):
            if layer in repr_layers:
                reps[layer] = torch.randn(B, L, self.embed_dim, device=tokens.device)
        attn = None
        if need_head_weights:
            attn = torch.randn(
                B, self.num_layers, self.num_heads, L, L, device=tokens.device
            )
        return {"representations": reps, "attentions": attn}


ESM_REGISTRY = EsmRegistry(
    registry={
        "esm2_8M_270K": esm.pretrained.esm2_t6_8M_UR50D,
        "esm2_35M_270K": esm.pretrained.esm2_t12_35M_UR50D,
        "esm2_650M": esm.pretrained.esm2_t33_650M_UR50D,
        "esm2_3B": esm.pretrained.esm2_t36_3B_UR50D,
        "esm2_15B": esm.pretrained.esm2_t48_15B_UR50D,
    }
)


@dataclass
class SequenceData:
    """Container for an ESM‑formatted sequence and a mask for real residues."""

    aa_sequence: torch.Tensor  # (B, L_total) - ESM tokens with BOS/EOS/linkers/pad
    non_linker_mask: torch.Tensor  # (B, L_total) – mask for real residues (sum == L)
    orig_len: int  # := L - original input length

    @classmethod
    def from_af2(
        cls,
        aatypes: torch.Tensor,  # (B, L) AF2 indices (0‑20, 0==A)
        chain_idx: torch.Tensor,  # (B, L) chain identifiers
        esm_dict: Alphabet,  # alphabet/dictionary from ESM model
        attn_mask: Optional[torch.Tensor] = None,  # (B, L) 1 for valid residues
        seq_mask: Optional[torch.Tensor] = None,  # (B, L) 1 for masked residues
    ) -> "SequenceData":
        """End‑to‑end conversion: AF2‑indices → ESM tokens with BOS/EOS/linkers."""
        B, L = aatypes.shape

        if attn_mask is None:
            attn_mask = torch.ones_like(aatypes, dtype=torch.bool)

        # convert AF2 indices -> ESM indices (with padding/masking)
        conv_table = cls._make_af2_to_esm_lookup(esm_dict).to(aatypes.device)

        # shift by +1 so 0‑based AA becomes 1..20 and we reserve 0 for PAD
        seq = (aatypes + 1).masked_fill(~attn_mask.bool(), 0)
        seq = conv_table[seq]

        if seq_mask is not None:
            seq = seq.masked_fill(seq_mask.bool(), esm_dict.mask_idx)

        # add BOS / EOS / linker tokens between chains
        bos, eos, pad = esm_dict.cls_idx, esm_dict.eos_idx, esm_dict.padding_idx
        linker = eos  # use <eos> as separator (ESM authors' recommendation)

        device, dtype = seq.device, seq.dtype
        sequences, residue_masks = [], []

        for b in range(B):
            breaks = torch.nonzero(
                chain_idx[b][1:] != chain_idx[b][:-1], as_tuple=False
            ).flatten()
            tokens, mask_bits = [bos], [False]
            last = 0
            for brk in breaks:
                idx = brk.item() + 1
                tokens.extend(seq[b, last:idx].tolist())
                mask_bits.extend([True] * (idx - last))
                tokens.append(linker)
                mask_bits.append(False)
                last = idx
            # tail + EOS
            tokens.extend(seq[b, last:].tolist())
            mask_bits.extend([True] * (L - last))
            tokens.append(eos)
            mask_bits.append(False)

            sequences.append(torch.tensor(tokens, device=device, dtype=dtype))
            residue_masks.append(
                torch.tensor(mask_bits, device=device, dtype=torch.bool)
            )

        # pad sequences to the same length, in case linkers were added
        max_len = max(t.size(0) for t in sequences)
        batch_seq = seq.new_full((B, max_len), pad)
        batch_mask = torch.zeros((B, max_len), dtype=torch.bool, device=device)

        for b in range(B):
            l_total = sequences[b].size(0)
            batch_seq[b, :l_total] = sequences[b]
            batch_mask[b, :l_total] = residue_masks[b]

        return cls(aa_sequence=batch_seq, non_linker_mask=batch_mask, orig_len=L)

    @staticmethod
    @lru_cache(maxsize=1)  # expect single esm_dict
    def _make_af2_to_esm_lookup(esm_dict: Alphabet) -> torch.Tensor:
        """
        Creates a `(22,)` tensor mapping AF2 indices (‑1..20) → ESM tokens.

        AF2: 0‑19 canonical aa, 20 == "X" (unknown).
        store `padding_idx` at 0 so can vectorise lookup with `lut[seq+1]`.
        """
        order = [esm_dict.padding_idx] + [esm_dict.get_idx(v) for v in restypes_with_x]
        return torch.tensor(order, dtype=torch.long)

    @classmethod
    def from_single_chain(
        cls, aatypes: torch.Tensor, esm_dict: Alphabet
    ) -> "SequenceData":
        B, L = aatypes.shape
        dummy_chain = torch.ones((B, L), dtype=torch.long, device=aatypes.device)
        return cls.from_af2(aatypes, dummy_chain, esm_dict)


class FrozenEsmModel(nn.Module):
    def __init__(
        self,
        model_key: str,
        use_esm_attn_map: bool = True,  # return pair repr
        caching: bool = True,  # enable caching the last call
    ):
        super().__init__()
        self.esm, self.esm_dict = ESM_REGISTRY.load(model_key)
        self.use_esm_attn_map = use_esm_attn_map
        self.caching = caching
        self._previous_call = None
        self.repr_layers = tuple(range(self.esm.num_layers + 1))

    @torch.no_grad()
    def forward(
        self,
        aa_sequence: torch.Tensor,
        chain_idx: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sequence_data = SequenceData.from_af2(
            aatypes=aa_sequence,
            chain_idx=chain_idx,
            esm_dict=self.esm_dict,
            attn_mask=attn_mask,
            seq_mask=seq_mask,
        )

        if (
            self.caching
            and self._previous_call is not None
            and self._previous_call["inputs"].shape == sequence_data.aa_sequence.shape
            and torch.all(self._previous_call["inputs"] == sequence_data.aa_sequence)
        ):
            return self._previous_call["outputs"]

        padding_mask = sequence_data.aa_sequence == self.esm_dict.padding_idx
        res = self.esm(
            sequence_data.aa_sequence,
            repr_layers=self.repr_layers,
            need_head_weights=self.use_esm_attn_map,
            return_contacts=False,
            padding_mask=padding_mask,
        )

        # postprocess
        # for single chains, we could just take `[1:-1]`
        # but to support chain breaks / linkers use `non_linker_mask`

        residue_mask = sequence_data.non_linker_mask  # (B, L_total)
        L = sequence_data.orig_len  # original residue count, target length

        reps = torch.stack(
            [v for _, v in sorted(res["representations"].items())], dim=2
        )
        B, L_total, nLayers, C = reps.shape

        single_repns = torch.empty(
            (B, L, nLayers, C), device=reps.device, dtype=reps.dtype
        )
        for b in range(B):
            single_repns[b] = reps[b][residue_mask[b]]  # -> (L, nLayers, C)

        if self.use_esm_attn_map:
            attn = res["attentions"].permute(
                0, 3, 4, 1, 2
            )  # B, L_total, L_total, nLayers, nHeads
            attn = attn.flatten(3, 4)  # B, L_total, L_total, nLayers*nHeads
            pair_repns = torch.empty(
                (B, L, L, attn.shape[-1]), device=attn.device, dtype=attn.dtype
            )
            for b in range(B):
                idx = residue_mask[b]
                pair_repns[b] = attn[b][idx][:, idx]
        else:
            pair_repns = None

        if self.caching:
            self._previous_call = {
                "inputs": sequence_data.aa_sequence.clone().detach(),
                "outputs": tree.map_structure(
                    lambda x: x.clone().detach(), (single_repns, pair_repns)
                ),
            }

        return single_repns, pair_repns
