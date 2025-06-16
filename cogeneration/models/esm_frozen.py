from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import esm
import torch
import tree
from esm.data import Alphabet
from faesm.esm import FAEsmForMaskedLM
from torch import nn

from cogeneration.config.base import ModelESMKey
from cogeneration.data.residue_constants import restypes_with_x

"""
Note on Flash attention etc: 
Flash Attention does not expose the attention weights. 
Nor does SDPE (scaled dot product attention) from PyTorch.

If want pair representation out of ESM, then will fall back to standard attention.

However, if we only cared about enriching the single representation, or using for logits, 
and not the pair representation, we use these faster variants using FAPLM.
"""


# registry keys are string or ModelESMKey
ESMRegistryKey = Union[str, ModelESMKey]

# registry returns nn.Module or FAEsmForMaskedLM
ESMRegistryReturn = Union[nn.Module, FAEsmForMaskedLM]

# registry factory method takes bool for whether to use flash attention
ESMRegistryFactory = Callable[[bool], ESMRegistryReturn]


@dataclass
class EsmRegistry:
    """Singleton holding factories that return ESMRegistryReturn=(model, alphabet)"""

    registry: Dict[ESMRegistryKey, ESMRegistryFactory] = field(default_factory=dict)

    def register(self, key: ESMRegistryKey, factory: ESMRegistryFactory) -> None:
        self.registry[key] = factory

    def load_model(self, key: ESMRegistryKey) -> ESMRegistryReturn:
        if key not in self.registry:
            raise KeyError(
                f"Model key '{key}' is not registered in EsmRegistry, have: {list(self.registry.keys())}"
            )

        use_flash_attention = torch.cuda.is_available()

        return self.registry[key](use_flash_attention)

    def load_alphabet(self) -> Alphabet:
        _, alphabet = esm.pretrained.load_model_and_alphabet(
            ModelESMKey.esm2_t6_8M_UR50D
        )
        return alphabet

    # helper to register a dummy model for testing
    def register_dummy(
        self,
        key: ESMRegistryKey = ModelESMKey.DUMMY,
        embedding_size: int = 4,
        n_layers: int = 1,
        n_heads: int = 1,
    ):
        """Registers a MinimalRandomESM under `key`."""

        def factory(use_fa: bool = False) -> _DummyRandomFAPLM:
            return _DummyRandomFAPLM(embedding_size, n_layers, n_heads)

        self.register(key, factory)


class _DummyRandomESM(nn.Module):
    """Dummy FAIR ESM-like model for testing that returns random representations/attentions"""

    def __init__(self, embed_dim: int, n_layers: int, n_heads: int):
        super().__init__()

        # mirror FAIR ESM attributes
        self.embed_dim = embed_dim
        self.num_layers = n_layers
        self.attention_heads = n_heads

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,  # (B, L)
        repr_layers: Any,  # ignored, use num_layers
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
                B, self.num_layers, self.attention_heads, L, L, device=tokens.device
            )
        return {"representations": reps, "attentions": attn}


class _DummyRandomFAPLM(nn.Module):
    """Dummy FAPLM-like model for testing that returns random representations/attentions"""

    def __init__(self, embed_dim: int, n_layers: int, n_heads: int):
        super().__init__()

        # hack - use fair-ESM style, since FAPLM uses hidden struct class
        self.embed_dim = embed_dim
        self.num_layers = n_layers
        self.attention_heads = n_heads

    @torch.no_grad()
    def forward(
        self,
        tokens: torch.Tensor,  # (B, L)
        output_attentions: bool = True,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        B, L = tokens.shape

        reps: List[torch.Tensor] = []
        for _ in tuple(range(self.num_layers + 1)):
            # (unsqueeze(0), B, L, <num_layers>, embed_dim)
            reps.append(torch.randn(1, B, L, self.embed_dim, device=tokens.device))

        attn = None
        if output_attentions:
            attn = []
            for _ in range(self.num_layers):
                # (unsqueeze(0), B, <num_layers>, attention_heads, L, L)
                attn.append(
                    torch.randn(
                        (1, B, self.attention_heads, L, L), device=tokens.device
                    )
                )

        return {"hidden_states": reps, "attentions": attn}


# ESM registry singleton
ESM_REGISTRY = EsmRegistry(
    registry={
        ModelESMKey.esm2_t6_8M_UR50D: lambda use_fa: FAEsmForMaskedLM.from_pretrained(
            "facebook/esm2_t6_8M_UR50D", use_fa=use_fa
        ),
        ModelESMKey.esm2_t12_35M_UR50D: lambda use_fa: FAEsmForMaskedLM.from_pretrained(
            "facebook/esm2_t12_35M_UR50D", use_fa=use_fa
        ),
        ModelESMKey.esm2_t30_150M_UR50D: lambda use_fa: FAEsmForMaskedLM.from_pretrained(
            "facebook/esm2_t30_150M_UR50D", use_fa=use_fa
        ),
        ModelESMKey.esm2_t33_650M_UR50D: lambda use_fa: FAEsmForMaskedLM.from_pretrained(
            "facebook/esm2_t33_650M_UR50D", use_fa=use_fa
        ),
        ModelESMKey.esm2_t36_3B_UR50D: lambda use_fa: FAEsmForMaskedLM.from_pretrained(
            "facebook/esm2_t36_3B_UR50D", use_fa=use_fa
        ),
        ModelESMKey.esm2_t48_15B_UR50D: lambda use_fa: FAEsmForMaskedLM.from_pretrained(
            "facebook/esm2_t48_15B_UR50D", use_fa=use_fa
        ),
    }
)
ESM_REGISTRY.register_dummy()


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
        res_mask: torch.Tensor,  # (B, L) 1 for valid residues
    ) -> "SequenceData":
        """
        End‑to‑end conversion: AF2‑indices → ESM tokens with BOS/EOS/linkers.
        adapted from https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py
        """
        B, L = aatypes.shape

        # convert AF2 indices -> ESM indices (with padding/masking)
        conv_table = cls._make_af2_to_esm_lookup(esm_dict).to(aatypes.device)
        bos, eos, pad = esm_dict.cls_idx, esm_dict.eos_idx, esm_dict.padding_idx

        # shift by +1 so 0‑based AA becomes 1..20
        seq = aatypes + 1
        seq = conv_table[seq]
        # set non-residue positions to <pad> to ignore (<unk> rarely observed so less stable)
        # TODO(model) - ideally differentiate non-canonical residues from non-residues (e.g. metal ion)
        #   So that residues in the chain are set to X. We don't differentiate in res_mask.
        seq = seq.masked_fill(~res_mask.bool(), pad)

        # add BOS / EOS / linker tokens between chains
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
        cls,
        aatypes: torch.Tensor,
        esm_dict: Alphabet,
        res_mask: torch.Tensor,
    ) -> "SequenceData":
        B, L = aatypes.shape
        dummy_chain = torch.ones((B, L), dtype=torch.long, device=aatypes.device)
        return cls.from_af2(
            aatypes, chain_idx=dummy_chain, esm_dict=esm_dict, res_mask=res_mask
        )


class FrozenEsmModel(nn.Module):
    def __init__(
        self,
        model_key: str,
        use_esm_attn_map: bool = True,  # return pair repr
        caching: bool = True,  # enable caching the last call
    ):
        super().__init__()
        self.esm = ESM_REGISTRY.load_model(model_key)
        self.esm_dict = ESM_REGISTRY.load_alphabet()
        self.use_esm_attn_map = use_esm_attn_map
        self.caching = caching
        self._previous_call = None
        self.repr_layers = tuple(range(self.num_layers + 1))

        # freeze ESM
        for param in self.esm.parameters():
            param.requires_grad = False
        self.esm.eval()
        self.esm.to(torch.float16)  # half precision

    def state_dict(self, *args, **kwargs):
        # contibute nothing to state_dict (save space in checkpoints)
        return {}

    def load_state_dict(self, state_dict, strict, assign):
        # ignore anything coming from a checkpoint, will load ESM model separately
        return {}

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        # ignore parent module `load_state_dict` nested calls
        return

    @property
    def embed_dim(self):
        try:
            return self.esm.embed_dim  # FAIR ESM style
        except AttributeError:
            return self.esm.config.hidden_size  # FAPLM style

    @property
    def num_layers(self):
        try:
            return self.esm.num_layers  # FAIR ESM style
        except AttributeError:
            return self.esm.config.num_hidden_layers  # FAPLM style

    @property
    def num_heads(self):
        try:
            return self.esm.attention_heads  # FAIR ESM style
        except AttributeError:
            return self.esm.config.num_attention_heads  # FAPLM style

    @torch.no_grad()
    def forward(
        self,
        aatypes: torch.Tensor,
        chain_index: torch.Tensor,
        res_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        sequence_data = SequenceData.from_af2(
            aatypes=aatypes,
            chain_idx=chain_index,
            esm_dict=self.esm_dict,
            res_mask=res_mask,
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
            # FAPLM style arguments
            output_attentions=self.use_esm_attn_map,
            output_hidden_states=True,
            # FAIR ESM kwargs not supported in FAEsmForMaskedLM
            # repr_layers=self.repr_layers,
            # return_contacts=False,
            # need_head_weights=self.use_esm_attn_map,
            # padding_mask=padding_mask,
        )

        # postprocess
        # for single chains, we could just take `[1:-1]`
        # but to support chain breaks / linkers use `non_linker_mask`

        residue_mask = sequence_data.non_linker_mask  # (B, L_total)
        N = sequence_data.orig_len  # original residue count, target length

        reps = torch.stack(
            # [v for _, v in sorted(res["representations"].items())], dim=2  # FAIR esm style
            [v.squeeze(0) for v in res["hidden_states"]],
            dim=2,  # FAPLM style
        )
        B, L_total, nLayers, C = reps.shape

        single_repns = torch.empty(
            (B, N, nLayers, C), device=reps.device, dtype=reps.dtype
        )
        for b in range(B):
            single_repns[b] = reps[b][residue_mask[b]]

        if self.use_esm_attn_map:
            # attn = res["attentions"]  # FAIR ESM style
            attn = torch.stack(
                [a.squeeze(0) for a in res["attentions"]], dim=1
            )  # FAPLM style
            attn = attn.permute(0, 3, 4, 1, 2)  # B, L_total, L_total, nLayers, nHeads
            attn = attn.flatten(3, 4)  # B, L_total, L_total, nLayers*nHeads
            pair_repns = torch.empty(
                (B, N, N, attn.shape[-1]), device=attn.device, dtype=attn.dtype
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

        # (B, N, nLayers, C), (B, N, N, nLayers*nHeads)
        return single_repns, pair_repns
