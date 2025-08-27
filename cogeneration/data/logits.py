from typing import Literal, Optional, Sequence

import torch


def combine_logits(
    logits: Sequence[torch.Tensor],
    center: Literal["logZ", "mean", "none"] = "logZ",
    mask_col: Optional[int] = None,  # e.g., MASK_TOKEN_INDEX
) -> torch.Tensor:
    """
    Combine N already-scaled logit tensors via product-of-experts-style addition.

    - All tensors in `logits_list` must have identical shape (..., S), dtype, and device.
    - Each source should have already been pre-multiplied by its own weight and divided by its own temperature.
    - `center="logZ"` treats each source as (rowwise) log-probs; `mean` subtracts rowwise mean; `none` uses raw values.
    - Optionally invalidate a column (e.g., MASK) or a set of columns by setting them to -inf in the result.

    Returns:
        combined_logits: torch.Tensor with same shape as inputs.
    """
    assert len(logits) > 0, "logits_list must be non-empty"
    ref = logits[0]
    ref_shape, ref_dtype, ref_dev = ref.shape, ref.dtype, ref.device

    # ensure all shapes/devices/dtypes match
    for i, L in enumerate(logits):
        assert (
            L.shape == ref_shape
        ), f"shape mismatch at index {i}: {L.shape} != {ref_shape}"
        assert (
            L.device == ref_dev
        ), f"device mismatch at index {i}: {L.device} != {ref_dev}"
        assert (
            L.dtype == ref_dtype
        ), f"dtype mismatch at index {i}: {L.dtype} != {ref_dtype}"

    # optionally center each source before summing (shift-invariant w.r.t. final softmax)
    # row-wise log-softmax
    centered = []
    if center == "logZ":
        for L in logits:
            centered.append(L - torch.logsumexp(L, dim=-1, keepdim=True))
    # rowwise mean-centering
    elif center == "mean":
        for L in logits:
            centered.append(L - L.mean(dim=-1, keepdim=True))
    elif center == "none":
        centered = list(logits)
    else:
        raise ValueError(f"Unknown centering mode: {center}")

    # sum all centered sources
    combined = torch.zeros_like(ref)
    for L in centered:
        combined = combined + L

    # if mask col, set to ~ -inf
    if mask_col is not None:
        combined = combined.clone()
        combined[..., mask_col] = -1e9

    return combined
