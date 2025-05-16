import pytest
import torch
from esm.pretrained import esm2_t6_8M_UR50D

from cogeneration.data.residue_constants import restypes, restypes_with_x
from cogeneration.models.esm_frozen import SequenceData


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

        # Expected: [CLS] + all 21 residues + [EOS]
        expected = torch.tensor(
            [
                [esm2_alphabet.cls_idx]
                + [esm2_alphabet.get_idx(aa) for aa in restypes]
                + [esm2_alphabet.get_idx("X")]  # explicit check X is not <unk>
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
        expected = torch.tensor(
            [
                [esm2_alphabet.cls_idx]
                + [esm2_alphabet.get_idx(aa) for aa in restypes_with_x[:4]]
                + [esm2_alphabet.padding_idx]
                + [esm2_alphabet.get_idx(aa) for aa in restypes_with_x[5:]]
                + [esm2_alphabet.eos_idx]
            ]
        )
        assert torch.equal(seq_data_masked.aa_sequence, expected)
