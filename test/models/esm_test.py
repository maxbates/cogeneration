import math

import pytest
import torch
from esm.pretrained import esm2_t6_8M_UR50D

from cogeneration.config.base import ModelESMKey
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.potentials import ESMLogitsPotential
from cogeneration.data.residue_constants import restypes, restypes_with_x
from cogeneration.data.trajectory import SamplingStep
from cogeneration.dataset.test_utils import create_single_item_batch
from cogeneration.models.esm_frozen import (
    _ESM_REGISTRY,
    FrozenEsmModel,
    SequenceData,
    get_frozen_esm,
)
from cogeneration.type.batch import BatchProp as bp


@pytest.fixture(scope="module")
def esm2_alphabet():
    _, alphabet = esm2_t6_8M_UR50D()
    return alphabet


class TestSequenceData:
    def test_sequence_data_from_af2(self, esm2_alphabet):
        # AF2 indices 0..20 (restypes + X)
        aatypes = torch.arange(0, 21).unsqueeze(0)  # (1, 21)
        chain_idx = torch.ones_like(aatypes)
        res_mask = torch.ones_like(aatypes)
        seq_data = SequenceData.from_af2(
            aatypes, chain_idx, esm_dict=esm2_alphabet, res_mask=res_mask
        )
        assert seq_data.orig_len == 21

        # Expected: [CLS] + 20 canonical residues + [MASK for AF2 X] + [EOS]
        expected = torch.tensor(
            [
                [esm2_alphabet.cls_idx]
                + [esm2_alphabet.get_idx(aa) for aa in restypes]
                + [esm2_alphabet.mask_idx]
                + [esm2_alphabet.eos_idx]
            ]
        )
        assert torch.equal(seq_data.aa_sequence, expected)

        expected_mask = torch.tensor([[False] + [True] * 21 + [False]])
        assert torch.equal(seq_data.non_linker_mask, expected_mask)

    def test_sequence_data_with_mask(self, esm2_alphabet):
        # AF2 indices 0..20 (restypes + X)
        aatypes = torch.arange(0, 21).unsqueeze(0)  # (1, 21)
        chain_idx = torch.ones_like(aatypes)

        # mask the 5th residue
        res_mask = torch.ones_like(aatypes).int()
        res_mask[0, 4] = 0
        seq_data_masked = SequenceData.from_af2(
            aatypes, chain_idx, esm2_alphabet, res_mask=res_mask
        )
        # Build expected: index 4 is outside res_mask -> 'X'; index 20 (AF2 X) within mask -> [MASK]
        tokens = [esm2_alphabet.get_idx(aa) for aa in restypes]
        tokens[4] = esm2_alphabet.get_idx("X")
        tokens.append(esm2_alphabet.mask_idx)
        expected = torch.tensor(
            [[esm2_alphabet.cls_idx] + tokens + [esm2_alphabet.eos_idx]]
        )
        assert torch.equal(seq_data_masked.aa_sequence, expected)


class TestFrozenEsmModel:
    @pytest.mark.parametrize("batch, length", [(1, 12), (2, 8)])
    @pytest.mark.parametrize("get_attn_map", [True, False])
    @pytest.mark.parametrize(
        "model_key", [ModelESMKey.esm2_t6_8M_UR50D, ModelESMKey.DUMMY]
    )
    def test_forward_pass(self, model_key, get_attn_map, batch, length):
        # load a real model (8M) from the registry
        model = FrozenEsmModel(model_key, use_esm_attn_map=get_attn_map, caching=False)

        # dummy AF2-style inputs
        aatypes = torch.randint(0, 21, (batch, length))  # residue indices
        chain_idx = torch.ones_like(aatypes)  # single-chain
        res_mask = torch.ones_like(aatypes, dtype=torch.bool)  # all valid

        token_reps, pair_reps, logits = model(aatypes, chain_idx, res_mask)

        # token-level reps: (B, L, nLayers+1, embed_dim)
        assert token_reps.shape == (
            batch,
            length,
            model.num_layers + 1,
            model.embed_dim,
        )

        if get_attn_map:
            # pair-level reps: (B, L, L, nLayers * nHeads)
            assert pair_reps is not None
            assert pair_reps.shape == (
                batch,
                length,
                length,
                model.num_layers * model.num_heads,
            )
        else:
            assert pair_reps is None

        # logits: (B, L, 21)
        assert logits.shape == (batch, length, 21)

    def test_logits_conversion_matches_esm(self):
        # Use real ESM for realistic + deterministic logits
        model = FrozenEsmModel(
            ModelESMKey.esm2_t6_8M_UR50D, use_esm_attn_map=False, caching=False
        )

        # create small random AF2-style inputs
        B, N = 1, 10
        aatypes = torch.randint(0, 21, (B, N))
        chain_idx = torch.ones_like(aatypes)
        res_mask = torch.ones_like(aatypes, dtype=torch.bool)

        # Build tokens exactly as the model will
        seq_data = SequenceData.from_af2(
            aatypes=aatypes,
            chain_idx=chain_idx,
            esm_dict=model.esm_dict,
            res_mask=res_mask,
        )

        # Run the underlying ESM to get raw logits in ESM vocab
        esm_out = model.esm(
            seq_data.aa_sequence,
            output_attentions=False,
            output_hidden_states=True,
        )
        logits_raw = esm_out["logits"]
        if logits_raw.dim() == 3 and logits_raw.shape[0] == 1:
            logits_raw = logits_raw.squeeze(0).view(
                B, seq_data.non_linker_mask.shape[1], -1
            )

        # Expected AF2 logits by applying the same mapping as in the model
        lut_indices = torch.tensor(
            [model.esm_dict.get_idx(aa) for aa in restypes]
            + [model.esm_dict.get_idx("X")],
            device=logits_raw.device,
            dtype=torch.long,
        )
        logits_expected = logits_raw.index_select(dim=-1, index=lut_indices)
        logits_expected = logits_expected[seq_data.non_linker_mask].view(B, N, 21)

        # Model output logits should match exactly
        _, _, logits_out = model(aatypes, chain_idx, res_mask)
        assert torch.allclose(logits_out, logits_expected)

    def test_masked_token_likelihoods(self, pdb_2qlw_processed_feats):
        batch = create_single_item_batch(pdb_2qlw_processed_feats)

        aatypes = batch[bp.aatypes_1].clone().long()
        chain_idx = batch[bp.chain_idx].clone().long()
        res_mask = batch[bp.res_mask].clone().bool()

        B, N = aatypes.shape
        assert B == 1

        model = FrozenEsmModel(
            ModelESMKey.esm2_t6_8M_UR50D, use_esm_attn_map=False, caching=False
        )

        candidates = [i for i in range(N) if (res_mask[0, i] and aatypes[0, i] != 20)]
        assert len(candidates) > 0
        k = min(50, len(candidates))

        successes_uniform = 0
        successes_top5 = 0
        for idx in candidates[:k]:
            true_aa = int(aatypes[0, idx].item())

            masked_seq = aatypes.clone()
            masked_seq[0, idx] = MASK_TOKEN_INDEX

            _, _, logits = model(masked_seq, chain_idx, res_mask)
            probs = torch.softmax(logits[0, idx], dim=-1)
            p_true = float(probs[true_aa].item())
            top5 = torch.topk(probs, k=5).indices.tolist()

            if true_aa in top5:
                successes_top5 += 1

            if p_true > 1.0 / 21.0:
                successes_uniform += 1

        # Expect most positions better than uniform
        assert successes_uniform >= math.floor(k * 0.9)

        # Expect a strong fraction of positions to have the true AA in top-5
        assert successes_top5 >= math.floor(k * 0.8)

    @pytest.mark.parametrize("get_attn_map", [True, False])
    @pytest.mark.parametrize(
        "model_key", [ModelESMKey.esm2_t6_8M_UR50D, ModelESMKey.DUMMY]
    )
    def test_forward_pass_with_variable_length_batch(self, model_key, get_attn_map):
        """Test that variable-length sequences in a batch (requiring padding) work correctly.

        This is the scenario that triggered the flash attention packing bug where
        hidden states would be (1, total_valid_tokens, C) instead of (1, B*L, C).
        """
        model = FrozenEsmModel(model_key, use_esm_attn_map=get_attn_map, caching=False)

        B = 3
        max_length = 15

        # Create variable-length sequences: lengths [15, 10, 7]
        aatypes = torch.randint(0, 21, (B, max_length))
        chain_idx = torch.ones_like(aatypes)

        # Create masks with different valid lengths per batch element
        res_mask = torch.zeros_like(aatypes, dtype=torch.bool)
        lengths = [15, 10, 7]
        for b, length in enumerate(lengths):
            res_mask[b, :length] = True

        token_reps, pair_reps, logits = model(aatypes, chain_idx, res_mask)

        # token-level reps: (B, L, nLayers+1, embed_dim)
        # Note: L here is the actual valid length per batch element after packing/unpacking
        assert token_reps.shape[0] == B
        assert token_reps.shape[2] == model.num_layers + 1
        assert token_reps.shape[3] == model.embed_dim

        # Each batch element should have output for its valid length
        for b, length in enumerate(lengths):
            # Check that we get valid outputs for each sequence
            batch_reps = token_reps[b, :length]
            assert batch_reps.shape == (length, model.num_layers + 1, model.embed_dim)
            # Check that representations are not all zeros (i.e., were actually computed)
            assert torch.any(batch_reps != 0)

        if get_attn_map:
            assert pair_reps is not None
            assert pair_reps.shape[0] == B
            assert pair_reps.shape[3] == model.num_layers * model.num_heads

            # Check each batch element's pair reps
            for b, length in enumerate(lengths):
                batch_pair = pair_reps[b, :length, :length]
                assert batch_pair.shape == (
                    length,
                    length,
                    model.num_layers * model.num_heads,
                )
                # Check that attention is not all zeros
                assert torch.any(batch_pair != 0)
        else:
            assert pair_reps is None

        # Check logits for each sequence
        assert logits.shape[0] == B
        assert logits.shape[2] == 21
        for b, length in enumerate(lengths):
            batch_logits = logits[b, :length]
            assert batch_logits.shape == (length, 21)
            assert torch.any(batch_logits != 0)

    def test_caching_across_instances(self, monkeypatch):
        """Cache should avoid calling underlying ESM forward twice for identical inputs, including across instances with same model key."""
        B, N = 1, 8
        aatypes = torch.randint(0, 21, (B, N))
        chain_idx = torch.ones_like(aatypes)
        res_mask = torch.ones_like(aatypes, dtype=torch.bool)

        model1 = get_frozen_esm(
            model_key=ModelESMKey.DUMMY, use_esm_attn_map=False, caching=True
        )

        # Patch the underlying singleton ESM model; all wrappers should share this instance
        call_counter = {"n": 0}
        orig_forward_singleton = model1.esm.forward

        def wrapped_forward_singleton(*args, **kwargs):
            call_counter["n"] += 1
            return orig_forward_singleton(*args, **kwargs)

        monkeypatch.setattr(
            model1.esm, "forward", wrapped_forward_singleton, raising=True
        )

        _ = model1(aatypes, chain_idx, res_mask)
        assert call_counter["n"] == 1

        _ = model1(aatypes, chain_idx, res_mask)
        assert call_counter["n"] == 1

        # should be the same instance and cached
        model2 = get_frozen_esm(ModelESMKey.DUMMY, use_esm_attn_map=False, caching=True)
        _ = model2(aatypes, chain_idx, res_mask)
        assert call_counter["n"] == 1

        # Also ensure the FK potential that uses a singleton FrozenESM reuses the shared cache
        potential = ESMLogitsPotential(
            esm_model_key=ModelESMKey.DUMMY,
            guidance_scale=1.0,
            esm_logits_temperature=1.0,
            esm_logits_cap=100.0,
        )

        # ensure have same underlying instance
        assert potential.esm is model1

        # minimal batch and step using the same inputs
        batch = {
            bp.res_mask: res_mask.clone().int(),
            bp.diffuse_mask: torch.ones_like(aatypes).int(),
            bp.chain_idx: chain_idx.clone().long(),
        }
        zeros_trans = torch.zeros(B, N, 3)
        eye_rot = torch.eye(3).view(1, 1, 3, 3).repeat(B, N, 1, 1)
        step = SamplingStep(
            res_mask=res_mask.clone(),
            trans=zeros_trans,
            rotmats=eye_rot,
            aatypes=aatypes.clone().long(),
            torsions=None,
            logits=None,
        )

        _E, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )
        # should not invoke the underlying esm.forward again
        assert call_counter["n"] == 1
