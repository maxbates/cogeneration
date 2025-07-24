from typing import Optional

import torch
from torch.distributions import VonMises

from cogeneration.data.const import MASK_TOKEN_INDEX


def centered_gaussian(
    num_batch, num_res, device: torch.device, n_bb_atoms: int = 3
) -> torch.Tensor:
    """
    Generates a tensor of shape (num_batch, num_res, n_bb_atoms=3)
    with values sampled from a centered Gaussian distribution.
    e.g. t=0 translations, in nanometer scale.
    """
    noise = torch.randn(num_batch, num_res, n_bb_atoms, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def centered_harmonic(
    chain_idx: torch.Tensor,
    device: torch.device,
    n_bb_atoms: int = 3,
    sigma: float = 2.08,
) -> torch.Tensor:
    """
    Creates a simple harmonic prior for protein chains, where internal residues have 2 neighbors and the ends have 1.
    `chain_idx` (B, N) is used for determining multimer chain breaks.

    Returns (B, N, n_bb_atoms=3) zero‐mean harmonic noise, in nanometer scale.

    Based on HarmonicFlow implementation, for which small molecule application probably more important.
    However, has the advantage for protein noodles that neighboring residues are placed closer together in space,
    compared to a simple guassian cloud.

    Choose sigma ~2.08 so that the marginal variance per coordinate of the harmonic prior
    matches the unit variance of torch.randn().
    For J = L + (1/σ²)·I with L the nearest-neighbor Laplacian, the average variance
        (1/N)·tr(J⁻¹) = (1/N)∑_i 1/(λ_i(L) + 1/σ²) should equal 1.
    Solving for σ gives ~2.08
    """
    num_batch, num_res = chain_idx.shape
    lam = 1.0 / sigma**2  # diagonal loading (precision)
    a = 1.0  # nearest‐neighbor spring constant

    out = torch.zeros(num_batch, num_res, n_bb_atoms, device=device)
    for b in range(num_batch):
        # find where residue i and i+1 share the same chain
        same = chain_idx[b][:-1] == chain_idx[b][1:]
        pos = torch.nonzero(same, as_tuple=True)[0]

        # build precision matrix J = L + λI
        J = torch.zeros(num_res, num_res, device=device)
        if pos.numel():
            i = pos
            j = pos + 1
            J[i, i] += a
            J[j, j] += a  # degree contributions
            J[i, j] = -a
            J[j, i] = -a  # off‐diagonals
        J += lam * torch.eye(num_res, device=device)

        # eigendecompose J
        D, P = torch.linalg.eigh(J)

        # sample standard normal noise z ∈ ℝ^(N×n_bb_atoms)
        z = torch.randn(num_res, n_bb_atoms, device=device)
        # harmonic prior: x = P @ (z / sqrt(D))
        x = P @ (z / torch.sqrt(D)[:, None])
        # center to zero mean
        out[b] = x - x.mean(dim=0, keepdim=True)

    return out


def random_rotation_matrix(device):
    """Generate a random rotation matrix using PyTorch."""
    q = torch.randn(4, device=device)
    q = q / q.norm()  # Normalize the quaternion
    q0, q1, q2, q3 = q
    return torch.tensor(
        [
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        ],
        dtype=torch.float32,
        device=device,
    )


def random_rotation_matrices(num_matrices: int, device):
    """Generate multiple random rotation matrices using PyTorch."""
    return torch.stack(
        [random_rotation_matrix(device=device) for _ in range(num_matrices)]
    )


def uniform_so3(num_batch: int, num_res: int, device) -> torch.Tensor:
    """
    Generates a tensor of shape (num_batch, num_res, 3, 3) with values sampled from a uniform SO(3) distribution.
    e.g. t=0 rotation matrices
    """
    return random_rotation_matrices(num_batch * num_res, device=device).reshape(
        num_batch, num_res, 3, 3
    )


def masked_categorical(num_batch, num_res, device) -> torch.Tensor:
    """
    Returns a mask tensor of shape (num_batch, num_res) with all values set to MASK_TOKEN_INDEX.
    e.g. t=0 aa types, masking interpolation
    """
    return torch.ones(size=(num_batch, num_res), device=device) * MASK_TOKEN_INDEX


def uniform_categorical(num_batch, num_res, num_tokens, device) -> torch.Tensor:
    """
    Returns uniform random samples from the range [0, num_tokens) of shape (num_batch, num_res).
    e.g. t=0 aa types, uniform interpolation
    """
    return torch.randint(
        size=(num_batch, num_res), low=0, high=num_tokens, device=device
    )


def torsions_empty(
    num_batch: int, num_res: int, device, num_angles: int = 7
) -> torch.Tensor:
    """Returns 0° torsions. Every slot defaults to the identity rotation (sin=0, cos=1)"""
    torsions = torch.zeros((num_batch, num_res, num_angles, 2), device=device)
    torsions[..., 1] = 1.0  # cos(0) = 1
    return torsions


def fill_torsions(
    shape: torch.Size,
    device: torch.device,
    torsions: Optional[torch.Tensor] = None,  # (B, N, K, 2)  K={1,5,7}
):
    """
    Generate full set of torsions (sin, cos) of shape (B, N, 7, 2) from partial set of torsions.
    The partial set may be undefined, or have 1 (psi), 5 (psi+chi) or 7 (all) angles.
    """
    B, N, K, _ = shape
    full = torsions_empty(
        num_batch=B, num_res=N, num_angles=7, device=device
    )  # (B, N, 7, 2)
    if torsions is None:
        return full

    K = torsions.shape[2]
    I = 0 if K == 7 else 2
    full[..., I : I + K, :] = torsions

    return full


def angles_noise(
    sigma: torch.Tensor,  # (B,)
    num_samples: int,  # N, e.g. num_res
    num_angles: int = 7,  # K
    kappa_max: float = 1e4,
) -> torch.Tensor:
    """
    Sample torsion angles from a Von Mises whose concentration kappa = 1 / sigma^2,
    clamped to [0, kappa_max], and return angles of shape (B, N, K).
    """
    num_batch = sigma.shape[0]

    # Compute kappa from sigma^2, clamped for stability
    # For large concentration, a Von Mises behaves like a Gaussian with variance ~1/kappa.
    # Setting kappa = 1 / sigma^2 makes its circular spread match a Normal(theta, sigma^2).
    kappa = 1.0 / (sigma.square() + 1e-6)  # (B,)
    kappa = torch.clamp(kappa, max=kappa_max)

    # to support MPS (no float64) need to build VonMises on CPU
    vm_device = sigma.device if "mps" not in str(sigma.device) else torch.device("cpu")
    vm = VonMises(
        loc=torch.zeros((num_batch,), device=vm_device),
        concentration=kappa.to(vm_device),
    )
    # sample raw angles in (−π, π]
    angles = vm.sample((num_samples, num_angles))  # (N, K, B)
    angles = angles.permute(2, 0, 1).to(sigma.device).float()  # (B, N, K)
    return angles


def torsions_noise(
    sigma: torch.Tensor,  # (B,)
    num_samples: int,  # N, e.g. num_res
    num_angles: int = 7,  # K
    kappa_max: float = 1e4,
) -> torch.Tensor:
    """
    Sample torsion (sin, cos) from a Von Mises whose concentration kappa = 1 / sigma^2,
    clamped to [0, kappa_max], and return (sin, cos) of shape (B, N, K, 2).
    """
    angles = angles_noise(
        sigma=sigma, num_samples=num_samples, num_angles=num_angles, kappa_max=kappa_max
    )
    return torch.stack((angles.sin(), angles.cos()), dim=-1)  # (B, N, K, 2)


def mask_blend_1d(
    aatypes_t: torch.Tensor, aatypes_1: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mask 1D tensor aatypes_t with 1D mask tensor mask, and set masked values to aatypes_1.
    Appropriate for e.g. amino acid types with diffuse_mask
    """
    return aatypes_t * mask + aatypes_1 * (1 - mask)


def mask_blend_2d(
    trans_t: torch.Tensor, trans_1: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mask 2D tensor trans_t with 2D mask tensor mask, and set masked values to trans_1.
    Appropriate for e.g. translations or logits with diffuse_mask
    """
    return trans_t * mask[..., None] + trans_1 * (1 - mask[..., None])


def mask_blend_3d(
    rotmats_t: torch.Tensor, rotmats_1: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    Mask 3D tensor rotmats_t with 2D mask tensor mask, and set masked values to rotmats_1.
    Appropriate for e.g. rotation matrices with diffuse_mask
    """
    return rotmats_t * mask[..., None, None] + rotmats_1 * (1 - mask[..., None, None])
