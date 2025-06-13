import math

import torch
from torch.nn import functional as F

from cogeneration.type.embed import PositionalEmbeddingMethod


def get_rotary_embedding(indices, embed_size, max_len: int):
    """
    Creates rotary positional embeddings from prespecified indices.

    Args:
        indices: offsets of size [..., N] of type integer
        embed_size: dimension of the embeddings to create (must be even)
        max_len: maximum length, used as theta base

    Returns:
        rotary positional embedding of shape [..., N, embed_size]
    """
    if embed_size % 2 != 0:
        raise ValueError("embed_size must be even for rotary embeddings.")
    theta = max_len ** (
        -torch.arange(0, embed_size, 2, device=indices.device).float() / embed_size
    )
    indices = indices.float()[..., None]
    angles = indices * theta
    emb = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
    return emb


def get_sine_cosine_embedding(indices, embed_size, max_len: int):
    """
    Creates sine / cosine positional embeddings from prespecified indices.
    """
    K = torch.arange(embed_size // 2, device=indices.device)
    power = torch.pow(max_len, 2 * K[None] / embed_size)
    pos_embedding_sin = torch.sin(indices[..., None] * math.pi / power).to(
        indices.device
    )
    pos_embedding_cos = torch.cos(indices[..., None] * math.pi / power).to(
        indices.device
    )
    pos_embedding = torch.cat([pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


def get_index_embedding(
    indices,
    embed_size,
    max_len: int,
    pos_embed_method: PositionalEmbeddingMethod,
):
    """
    Creates positional embeddings from prespecified indices using the specified method.

    TODO(model) - consider adding chain gaps to res_idx, which currently is 1-indexed per chain.
       Currently we embed chain_idx separately, but could be worth comparing approaches.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embed_size: dimension of the embeddings to create
        pos_embed_method: method to use for positional embedding

    Returns:
        positional embedding of shape [N, embed_size]
    """
    if pos_embed_method == PositionalEmbeddingMethod.rotary:
        return get_rotary_embedding(indices, embed_size, max_len)
    elif pos_embed_method == PositionalEmbeddingMethod.sine_cosine:
        return get_sine_cosine_embedding(indices, embed_size, max_len)
    else:
        raise ValueError(f"Unknown positional embedding method: {pos_embed_method}")


def get_time_embedding(timesteps, embedding_dim, max_positions=2000):
    """
    Adapted from https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/nn.py
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert (
        len(timesteps.shape) == 1
    ), f"Expected 1D tensor, got {timesteps.shape}"  # and timesteps.dtype == tf.int32
    timesteps = timesteps * max_positions

    half_dim = embedding_dim // 2
    emb = math.log(max_positions) / (half_dim - 1)
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=timesteps.device) * -emb
    )
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1), mode="constant")
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def sinusoidal_encoding(v, N, D):
    """Taken from GENIE.

    Args:

    """
    # v: [*]

    # [D]
    k = torch.arange(1, D + 1).to(v.device)

    # [*, D]
    sin_div_term = N ** (2 * k / D)
    sin_div_term = sin_div_term.view(*((1,) * len(v.shape) + (len(sin_div_term),)))
    sin_enc = torch.sin(v.unsqueeze(-1) * math.pi / sin_div_term)

    # [*, D]
    cos_div_term = N ** (2 * (k - 1) / D)
    cos_div_term = cos_div_term.view(*((1,) * len(v.shape) + (len(cos_div_term),)))
    cos_enc = torch.cos(v.unsqueeze(-1) * math.pi / cos_div_term)

    # [*, D]
    enc = torch.zeros_like(sin_enc).to(v.device)
    enc[..., 0::2] = cos_enc[..., 0::2]
    enc[..., 1::2] = sin_enc[..., 1::2]

    return enc.to(v.dtype)


def calc_distogram(pos, min_bin, max_bin, num_bins):
    """
    Calculate 2D distance histogram, binning pairwise distances between pos.
    takes `pos` (B, N, 3) and returns (B, N, N, num_bins)
    """
    dists_2d = torch.linalg.norm(pos[:, :, None, :] - pos[:, None, :, :], axis=-1)[
        ..., None
    ]
    lower = torch.linspace(min_bin, max_bin, num_bins, device=pos.device)
    upper = torch.cat([lower[1:], lower.new_tensor([1e8])], dim=-1)
    dgram = ((dists_2d > lower) * (dists_2d < upper)).type(pos.dtype)
    return dgram
