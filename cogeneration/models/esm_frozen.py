import logging
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
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.residue_constants import restypes, restypes_with_x
from cogeneration.models.utils import get_model_size_str
from cogeneration.util.log import rank_zero_logger

"""
Note on Flash attention etc: 
Flash Attention does not expose the attention weights. 
Nor does SDPE (scaled dot product attention) from PyTorch.

If want pair representation out of ESM, then will fall back to standard attention.

However, if we only cared about enriching the single representation, or using for logits, 
and not the pair representation, we use these faster variants using FAPLM.
"""

logger = rank_zero_logger(__name__)


# registry keys are string or ModelESMKey
ESMRegistryKey = Union[str, ModelESMKey]

# registry returns nn.Module or FAEsmForMaskedLM
ESMRegistryReturn = Union[nn.Module, FAEsmForMaskedLM]

# registry factory method takes bool for whether to use flash attention
ESMRegistryFactory = Callable[[bool], ESMRegistryReturn]


@dataclass
class EsmRegistry:
    """
    Singleton holding factories that return ESMRegistryReturn=(model, alphabet)

    And a cache of loaded models, so callers can share the same underlying model.
    """

    registry: Dict[ESMRegistryKey, ESMRegistryFactory] = field(default_factory=dict)
    # Cache created model instances per key so all callers share the same underlying model
    _loaded_models: Dict[ESMRegistryKey, ESMRegistryReturn] = field(
        default_factory=dict, init=False, repr=False
    )
    # Cache alphabet since it's independent of model size
    _alphabet: Optional[Alphabet] = field(default=None, init=False, repr=False)

    def register(self, key: ESMRegistryKey, factory: ESMRegistryFactory) -> None:
        self.registry[key] = factory

    def load_model(self, key: ESMRegistryKey) -> ESMRegistryReturn:
        if key not in self.registry:
            raise KeyError(
                f"Model key '{key}' is not registered in EsmRegistry, have: {list(self.registry.keys())}"
            )

        # Return cached instance if available so multiple wrappers share weights and device
        if key in self._loaded_models:
            return self._loaded_models[key]

        use_flash_attention = torch.cuda.is_available()
        model = self.registry[key](use_flash_attention)
        self._loaded_models[key] = model
        return model

    def load_alphabet(self) -> Alphabet:
        if self._alphabet is None:
            _, alphabet = esm.pretrained.load_model_and_alphabet(
                ModelESMKey.esm2_t6_8M_UR50D
            )
            self._alphabet = alphabet
        return self._alphabet

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

        # use fair-ESM style, since FAPLM uses hidden struct class
        self.embed_dim = embed_dim
        self.num_layers = n_layers
        self.attention_heads = n_heads
        self.vocab_size = len(_ESM_REGISTRY.load_alphabet())

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

        # random logits over a ESM alphabet
        logits = torch.randn(B, L, self.vocab_size, device=tokens.device)

        return {"hidden_states": reps, "attentions": attn, "logits": logits}


# ESM registry singleton, factory for actual ESM models
_ESM_REGISTRY = EsmRegistry(
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
_ESM_REGISTRY.register_dummy()


@dataclass
class SequenceData:
    """Container for an ESM-formatted sequence and a mask for real residues."""

    aa_sequence: torch.Tensor  # (B, L) - ESM tokens with BOS/EOS/linkers/pad
    non_linker_mask: (
        torch.Tensor
    )  # (B, L) – mask for real residues including UNK (sum == N)
    orig_len: int  # := N - original input length

    @classmethod
    def from_af2(
        cls,
        aatypes: torch.Tensor,  # (B, N) AF2 indices (0-20, 0==A)
        chain_idx: torch.Tensor,  # (B, N) chain identifiers
        esm_dict: Alphabet,  # alphabet/dictionary from ESM model
        res_mask: torch.Tensor,  # (B, N) 1 for valid residues
    ) -> "SequenceData":
        """
        End-to-end conversion: AF2-indices → ESM tokens with BOS/EOS/linkers.
        adapted from https://github.com/facebookresearch/esm/blob/main/esm/esmfold/v1/esmfold.py
        """
        B, N = aatypes.shape

        bos = esm_dict.cls_idx
        eos = esm_dict.eos_idx
        pad = esm_dict.padding_idx
        mask = esm_dict.mask_idx
        X = esm_dict.get_idx("X")
        linker = eos  # use <eos> as linker (ESM authors' recommendation)

        # Convert AF2 indices to ESM indices
        # Note X is converted to [MASK], since we are likely using the masking interpolant (TODO check)
        # And below non-residues are set to X.
        conv_table = cls._make_af2_to_esm_lookup(esm_dict).to(aatypes.device)
        # shift by +1 so 0-based AA becomes 1..20 (0 is padding)
        seq = conv_table[aatypes + 1]

        # Set positions not in res_mask to X, i.e. assume they are non-canonical or non-residues.
        # with chain-independent trimming, this is probably a reasonable assumption.
        # (also, avoid using <pad> for simplicity with flash attention, which drops it.)
        seq = seq.masked_fill(~res_mask.bool(), X)

        device, dtype = seq.device, seq.dtype
        sequences, residue_masks = [], []

        # add BOS / EOS / linker tokens between chains
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
            mask_bits.extend([True] * (N - last))
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
            l_seq = sequences[b].size(0)
            batch_seq[b, :l_seq] = sequences[b]
            batch_mask[b, :l_seq] = residue_masks[b]

        return cls(aa_sequence=batch_seq, non_linker_mask=batch_mask, orig_len=N)

    @staticmethod
    @lru_cache(maxsize=1)  # expect single esm_dict
    def _make_af2_to_esm_lookup(esm_dict: Alphabet) -> torch.Tensor:
        """
        Creates a `(22,)` tensor mapping AF2 indices (-1..20) → ESM tokens.
        store `padding_idx` at 0 so can vectorise lookup with `lut[seq+1]`.

        AF2: 0-19 canonical aa, 20 == "X" (used as both unknown and mask).

        We convert X to [MASK].
        Usually X means masked sequence, though it does also mean unknown residue.
        After translating from AF2 to ESM tokens, positions not in `res_mask` are set to `X`.
        """
        order = (
            [esm_dict.padding_idx]
            + [esm_dict.get_idx(v) for v in restypes]
            + [esm_dict.mask_idx]
        )
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


# Shared cache across all FrozenEsmModel instances.
# Keyed only by model_key.
# Matches on exact `aatypes` tensor on CPU.
# If attn_map required but not in cache, trigger recompute.
# If attns present and not needed, still a cache hit.
_SHARED_ESM_CACHE: Dict[str, Dict[str, Any]] = {}


class FrozenEsmModel(nn.Module):
    def __init__(
        self,
        model_key: str,
        use_esm_attn_map: bool = True,  # return pair repr
        caching: bool = True,  # enable caching the last call
        precision: str = "32",  # model precision (32, 16, bf16, bf16-mixed, etc.)
    ):
        """
        FrozenEsmModel wraps ESM, sets to eval, supports caching,
        and moves inputs/cached outputs to the input aatypes device

        Use `get_frozen_esm` to get a singleton instance.
        Construct explicitly if want isolated instance.
        """
        super().__init__()

        logger.info(f"Loading ESM model: {model_key}...")

        # May see a warning about some parameters not being initialized,
        # this is fine, because we aren't using the MaskedLM head.
        self._model_key = model_key
        self.esm = _ESM_REGISTRY.load_model(model_key)
        self.esm_dict = _ESM_REGISTRY.load_alphabet()
        self.use_esm_attn_map = use_esm_attn_map
        self.caching = caching
        self.repr_layers = tuple(range(self.num_layers + 1))

        # freeze ESM
        for param in self.esm.parameters():
            param.requires_grad = False
        # Set to eval mode
        self.esm.eval()

        # Set precision for inference (bf16/fp16 supported by flash attention)
        self._set_precision(precision)

        logger.info(
            f"✅ ESM {model_key} loaded (precision: {precision}, use_esm_attn_map: {use_esm_attn_map})"
        )

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

    def _set_precision(self, precision: str) -> None:
        """Convert ESM model to specified precision for inference."""
        if precision == "32":
            self.esm = self.esm.float()
        elif precision in ("16", "fp16"):
            self.esm = self.esm.half()
        elif precision in ("bf16", "bf16-mixed"):
            self.esm = self.esm.to(dtype=torch.bfloat16)
        elif precision == "64":
            self.esm = self.esm.double()
        else:
            logger.warning(
                f"Unknown precision '{precision}' for ESM, keeping default precision"
            )

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

    @staticmethod
    def _unpack_flash_attention_output(
        packed_output: torch.Tensor,  # (1, num_valid_tokens, C) or (1, B*L, C)
        attention_mask: torch.Tensor,  # (B, L)
        target_shape: Tuple[int, int, int],  # (B, L, C)
    ) -> torch.Tensor:
        """
        Unpack flash attention outputs that drop padding tokens.

        Args:
            packed_output: (1, num_valid_tokens, C) or (1, B*L, C)
            attention_mask: (B, L) boolean mask for valid tokens
            target_shape: (B, L, C) desired output shape

        Returns:
            unpacked: (B, L, C) tensor with zeros at padding positions
        """
        B, L, C = target_shape
        squeezed = packed_output.squeeze(0)  # (num_valid_tokens, C) or (B*L, C)

        # Flatten attention mask for 1D indexing
        attention_mask_flat = attention_mask.flatten()  # (B*L,)
        num_valid_tokens = attention_mask_flat.sum().item()

        # Check if output is packed (only valid tokens) or full
        if squeezed.shape[0] == num_valid_tokens:
            # Packed format - reconstruct with zeros at padding positions
            full_flat = squeezed.new_zeros(B * L, C)
            full_flat[attention_mask_flat] = squeezed
            return full_flat.view(B, L, C)
        else:
            # Already full format - just reshape
            return squeezed.view(B, L, C)

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

        # Shared last-call cache across instances
        if self.caching:
            shared_key = str(self._model_key)
            prev = _SHARED_ESM_CACHE.get(shared_key)
            if prev is not None:
                prev_inputs = prev.get("aatypes")
                if (
                    prev_inputs is not None
                    and tuple(prev.get("shape", ())) == tuple(aatypes.shape)
                    and torch.equal(prev_inputs, aatypes.detach().cpu())
                ):
                    cached_single, cached_pair, cached_logits = prev["outputs"]
                    # If pair reps requested but not present, fall through to recompute with attentions
                    # If pair reps is present but not requested, drop it
                    if cached_pair is None and self.use_esm_attn_map:
                        pass
                    else:
                        pair_out = cached_pair if self.use_esm_attn_map else None
                        return tuple(
                            x.to(aatypes.device) if x is not None else None
                            for x in (cached_single, pair_out, cached_logits)
                        )

        residue_mask = sequence_data.non_linker_mask  # (B, L)
        padding_mask = sequence_data.aa_sequence == self.esm_dict.padding_idx  # (B, L)
        attention_mask = ~padding_mask  # (B, L) True for valid tokens

        # track sizes
        B, N = aatypes.shape
        L = residue_mask.shape[1]  # L > N; includes BOS, EOS, maybe linkers

        res = self.esm(
            sequence_data.aa_sequence,
            attention_mask=attention_mask,
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
        # We extract single reps (per layer), pair reps (per layer), and logits (final layer)
        # for single chains, we could just take `[1:-1]` (remove BOS, EOS)
        # but to support chain breaks / linkers use `non_linker_mask`

        # # FAIR esm style
        # reps = torch.stack(
        #     [v for _, v in sorted(res["representations"].items())],
        #     dim=2,
        # )  # (B, L, nLayers, C)

        # Flash attention may pack outputs (dropping padding tokens), so we unpack them.
        # get embedding dimension from first hidden state
        C = res["hidden_states"][0].shape[-1]
        # unpack all hidden state layers: (B, L, nLayers, C)
        reps = torch.stack(
            [
                self._unpack_flash_attention_output(h, attention_mask, (B, L, C))
                for h in res["hidden_states"]
            ],
            dim=2,
        )

        # keep only the non-linker / non-special residues
        single_repns = reps[residue_mask].view(
            B, N, *reps.shape[2:]
        )  # (B, N, nLayers, C)

        if self.use_esm_attn_map:
            # FAIR esm style
            # attn = res["attentions"]

            # FAPLM style
            # Note: When output_attentions=True, FAEsm falls back to standard attention
            # (not flash attention), so outputs are in standard format, not packed.
            flat_attn = []
            for a in res["attentions"]:
                # handle flash attention "layer batch".
                # If batch size is actually 1, handled by view() below
                if a.shape[0] == 1:  # (1, h, B·L, B·L)
                    a = a.squeeze(0)  # (h, B·L, B·L)
                    a = a.view(B, -1, L, L)  # (B, h, L, L)
                # else: no FA, already (B, h, L, L)
                flat_attn.append(a)
            # stack over layers: (B, nLayers, nHeads, L, L)
            attn = torch.stack(flat_attn, dim=1)

            attn = attn.permute(0, 3, 4, 1, 2)  # B, L, L, nLayers, nHeads
            attn = attn.flatten(3, 4)  # B, L, L, nLayers*nHeads

            # limit to actual residues
            pair_repns = torch.empty(
                (B, N, N, attn.shape[-1]), device=attn.device, dtype=attn.dtype
            )
            for b in range(B):
                idx = residue_mask[b]
                pair_repns[b] = attn[b][idx][:, idx]
        else:
            pair_repns = None

        # Extract and unpack logits: (B, L, vocab_size)
        logits = res["logits"]
        vocab_size = logits.shape[-1]
        logits = self._unpack_flash_attention_output(
            logits, attention_mask, (B, L, vocab_size)
        )
        # Convert to AF2 ordering, use 'X' token for AF2 'X' class
        lut_logits = torch.tensor(
            [self.esm_dict.get_idx(a) for a in restypes] + [self.esm_dict.get_idx("X")],
            device=logits.device,
            dtype=torch.long,
        )
        logits = logits.index_select(dim=-1, index=lut_logits)  # (B, L, 21)
        # Pull out residues, drop linkers, pad, etc.
        logits = logits[residue_mask].view(B, N, logits.shape[-1])  # (B, N, 21)

        if self.caching:
            shared_key = str(self._model_key)
            _SHARED_ESM_CACHE[shared_key] = {
                "shape": tuple(aatypes.shape),
                "aatypes": aatypes.detach().cpu().clone(),
                "outputs": tree.map_structure(
                    lambda x: x.clone().detach() if x is not None else None,
                    (single_repns, pair_repns, logits),
                ),
            }

        # (B, N, nLayers, C), (B, N, N, nLayers*nHeads), (B, N, 21)
        return single_repns, pair_repns, logits


# Singleton accessor so different callers share the same wrapper and underlying model
_FROZEN_ESM_SINGLETONS: Dict[Tuple[str, bool, bool, str], FrozenEsmModel] = {}


def get_frozen_esm(
    model_key: str,
    use_esm_attn_map: bool,
    caching: bool = True,
    precision: str = "32",
) -> FrozenEsmModel:
    """
    Factory for singleton (of same model key + outputs) FrozenEsmModel instances.

    For example, if a model's module and interpolant potential share the same instances,
    only one set of weights are loaded, and caching is shared between them.
    (Note the model device will be set by the containing model,
    and potential will just run on that device.
    FrozenEsmModel will move the inputs / cached outputs to the input device.)
    """
    key = (str(model_key), bool(use_esm_attn_map), bool(caching), str(precision))
    if key not in _FROZEN_ESM_SINGLETONS:
        _FROZEN_ESM_SINGLETONS[key] = FrozenEsmModel(
            model_key=model_key,
            use_esm_attn_map=use_esm_attn_map,
            caching=caching,
            precision=precision,
        )
    return _FROZEN_ESM_SINGLETONS[key]
