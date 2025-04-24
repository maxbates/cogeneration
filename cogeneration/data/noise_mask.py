import torch

from cogeneration.data.const import MASK_TOKEN_INDEX

# TODO - move masks to a different file?


def centered_gaussian(num_batch, num_res, device: torch.device, n_bb_atoms: int = 3):
    """
    Generates a tensor of shape (num_batch, num_res, 3) with values sampled from a centered Gaussian distribution.
    e.g. t=0 translations, in nanometer scale.
    """
    noise = torch.randn(num_batch, num_res, n_bb_atoms, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


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
