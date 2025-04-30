import numpy as np
import torch

from cogeneration.data.noise_mask import centered_gaussian, centered_harmonic


def make_chain_idx(batch_size, N, split):
    """
    Utility to create a (batch_size, N) tensor where
    the first `split` residues are chain 0 and the rest chain 1.
    """
    chains = []
    for _ in range(batch_size):
        idx = torch.zeros(N, dtype=torch.long)
        idx[split:] = 1
        chains.append(idx)
    return torch.stack(chains, dim=0)


class TestCenteredHarmonic:
    def test_zero_centered(self):
        # noise should have global mean ~0
        chain_idx = make_chain_idx(batch_size=4, N=30, split=10)
        noise = centered_harmonic(chain_idx, sigma=1.5, device=torch.device("cpu"))
        mean = noise.mean(dim=(0, 1))
        assert torch.allclose(
            mean, torch.zeros_like(mean), atol=1e-6
        ), f"Global mean not zero: {mean.tolist()}"

    def test_per_chain_center_of_mass(self):
        # each chain block should have its own center of mass
        chain_idx = make_chain_idx(batch_size=4, N=30, split=10)
        noise = centered_harmonic(chain_idx, sigma=1.5, device=torch.device("cpu"))
        for b in range(noise.size(0)):
            chain_coms = {}
            for i, c in enumerate(torch.unique(chain_idx[b])):
                mask = chain_idx[b] == c
                com = noise[b][mask].mean(dim=0)
                chain_coms[i] = com

            # check COMs are unique within batch
            assert len(chain_coms) == 2, "Expected  2 unique chain centers of mass"
            assert not torch.allclose(
                chain_coms[0], chain_coms[1]
            ), "Chain COMs are not unique"

    def test_neighbor_distance_smaller_than_gaussian(self):
        # harmonic prior should keep neighbors closer on average
        chain_idx = make_chain_idx(batch_size=4, N=30, split=10)
        gaussian = centered_gaussian(
            num_batch=1, num_res=30, n_bb_atoms=3, device=torch.device("cpu")
        ).view(30, 3)
        harmonic = centered_harmonic(
            chain_idx[:1], sigma=1.5, device=torch.device("cpu")
        )[0]
        # calc mean neighbor distances
        dist_gaussian = []
        dist_harmonic = []
        idx = chain_idx[0]
        for i in range(idx.size(0) - 1):
            if idx[i] == idx[i + 1]:
                dist_gaussian.append((gaussian[i] - gaussian[i + 1]).norm().item())
                dist_harmonic.append((harmonic[i] - harmonic[i + 1]).norm().item())
        mean_gauss = np.mean(dist_gaussian)
        mean_harm = np.mean(dist_harmonic)
        assert (
            mean_harm < mean_gauss
        ), f"Expected harmonic neighbor distance < gaussian, got {mean_harm:.3f} >= {mean_gauss:.3f}"
