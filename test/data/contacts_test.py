import torch

from cogeneration.dataset.contacts import get_contact_conditioning_matrix


def make_multimer_coords(n_per_chain=15, separation=10.0):
    """
    Create coordinates for a simple two-chain multimer.
    Chain 1 at origin, Chain 2 separated by `separation` distance in x-direction.
    """
    chain1_coords = torch.zeros(n_per_chain, 3)
    chain2_coords = torch.zeros(n_per_chain, 3)
    chain2_coords[:, 0] = separation

    trans = torch.cat([chain1_coords, chain2_coords], dim=0)
    chain_idx = torch.cat([torch.zeros(n_per_chain), torch.ones(n_per_chain)]).long()
    res_mask = torch.ones(2 * n_per_chain, dtype=torch.bool)

    return trans, res_mask, chain_idx


class TestGetContactConditioningMatrix:
    def test_basic_functionality(self):
        """Test that the function generates contact matrices correctly."""
        trans, res_mask, chain_idx = make_multimer_coords(
            n_per_chain=10, separation=8.0
        )

        # Basic contact matrix generation
        matrix = get_contact_conditioning_matrix(
            trans=trans,
            res_mask=res_mask,
            chain_idx=chain_idx,
            motif_mask=None,
            include_inter_chain=True,
            downsample_inter_chain=False,
            min_res_gap=3,
            min_dist=2.0,
            max_dist=15.0,
        )

        assert matrix.shape == (20, 20), f"Expected (20, 20) matrix, got {matrix.shape}"
        assert matrix.dtype == torch.float32, f"Expected float32, got {matrix.dtype}"

        # Should have some inter-chain contacts since chains are 8.0 apart (within max_dist=15.0)
        inter_chain_mask = chain_idx.unsqueeze(-1) != chain_idx.unsqueeze(-2)
        inter_chain_contacts = ((matrix > 0) & inter_chain_mask).sum()
        assert inter_chain_contacts > 0, "Should have some inter-chain contacts"

    def test_downsampling_reduces_contacts(self):
        """Test that downsampling reduces the number of inter-chain contacts."""
        trans, res_mask, chain_idx = make_multimer_coords(
            n_per_chain=15, separation=6.0
        )

        # Get full contacts
        matrix_full = get_contact_conditioning_matrix(
            trans=trans,
            res_mask=res_mask,
            chain_idx=chain_idx,
            motif_mask=None,
            include_inter_chain=True,
            downsample_inter_chain=False,
            min_res_gap=3,
            min_dist=2.0,
            max_dist=15.0,
        )

        # Get downsampled contacts
        matrix_downsampled = get_contact_conditioning_matrix(
            trans=trans,
            res_mask=res_mask,
            chain_idx=chain_idx,
            motif_mask=None,
            include_inter_chain=True,
            downsample_inter_chain=True,
            downsample_inter_chain_min_contacts=2,
            downsample_inter_chain_max_contacts=4,
            min_res_gap=3,
            min_dist=2.0,
            max_dist=15.0,
        )

        inter_chain_mask = chain_idx.unsqueeze(-1) != chain_idx.unsqueeze(-2)

        full_contacts = ((matrix_full > 0) & inter_chain_mask).sum()
        downsampled_contacts = ((matrix_downsampled > 0) & inter_chain_mask).sum()

        assert full_contacts > 0, "Should have some contacts before downsampling"
        assert (
            downsampled_contacts <= full_contacts
        ), "Downsampling should reduce contacts"
        assert (
            2 * 2 <= downsampled_contacts <= 4 * 2
        ), f"Downsampled contacts should be in range [2, 4], got {downsampled_contacts}"

    def test_no_inter_chain_when_disabled(self):
        """Test that no inter-chain contacts are included when include_inter_chain=False."""
        trans, res_mask, chain_idx = make_multimer_coords(
            n_per_chain=10, separation=8.0
        )

        matrix = get_contact_conditioning_matrix(
            trans=trans,
            res_mask=res_mask,
            chain_idx=chain_idx,
            motif_mask=None,
            include_inter_chain=False,  # Disabled
            downsample_inter_chain=False,
            min_res_gap=3,
            min_dist=2.0,
            max_dist=15.0,
        )

        inter_chain_mask = chain_idx.unsqueeze(-1) != chain_idx.unsqueeze(-2)
        inter_chain_contacts = ((matrix > 0) & inter_chain_mask).sum()

        assert (
            inter_chain_contacts == 0
        ), f"Should have no inter-chain contacts when disabled, got {inter_chain_contacts}"
