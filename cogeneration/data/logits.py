from typing import Literal, Optional, Sequence

import torch

from cogeneration.data.const import MASK_TOKEN_INDEX


def center_logits(
    logits: torch.Tensor,
    center: Literal["logZ", "mean", "none"] = "logZ",
    ignore_mask: bool = True,
) -> torch.Tensor:
    """
    Center logits by subtracting the log-sum-exp or mean of each row.

    If `ignore_mask` is True, and there are 21 logits, the last logit is ignored (treated as mask) when centering
    """
    if center == "none":
        return logits

    S = logits.shape[-1]

    if S == 20 or not ignore_mask:
        if center == "logZ":
            return logits - torch.logsumexp(logits, dim=-1, keepdim=True)
        elif center == "mean":
            return logits - logits.mean(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown centering mode: {center}")

    assert S == 21, "Masking is only supported for 21 logits"
    neg_inf = torch.finfo(logits.dtype).min
    masked = logits.masked_fill(
        torch.nn.functional.one_hot(
            torch.full(logits.shape[:-1], MASK_TOKEN_INDEX, device=logits.device),
            num_classes=logits.shape[-1],
        ).bool(),
        neg_inf,
    )
    if center == "logZ":
        logZ = torch.logsumexp(masked, dim=-1, keepdim=True)
        return logits - logZ
    elif center == "mean":
        return logits - masked.mean(dim=-1, keepdim=True)
    else:
        raise ValueError(f"Unknown centering mode: {center}")


def clamp_logits(
    logits: torch.Tensor,
    abs_cap: Optional[float] = None,
    max_margin: Optional[float] = None,
    ignore_mask: bool = True,
) -> torch.Tensor:
    """
    Clamp logits by combining a per-row margin cap and/or an absolute value cap.

    `abs_cap` caps the maximum absolute value of each row.
    `max_margin` caps the difference between the top two logits of each row.
    If `ignore_mask` is True, and there are 21 logits, the last logit is ignored (treated as mask) when computing stats.
    """
    S = logits.shape[-1]
    eps = 1e-6
    orig_dtype = logits.dtype
    x = logits.float()  # use fp32 for stability (for numerical stability)

    # track if mask present
    mask_idx = int(MASK_TOKEN_INDEX) if ignore_mask and (S == 21) else None

    # views for stats that ignore MASK (do not mutate x)
    if mask_idx is not None:
        x_topk = x.clone()
        x_topk[..., mask_idx] = -torch.inf  # keep MASK out of top-k competition
        abs_vals = x.abs()
        abs_vals[..., mask_idx] = 0.0  # keep MASK out of |.| max
    else:
        x_topk = x
        abs_vals = x.abs()

    # base neutral scale
    scales = x.new_ones(x.shape[:-1] + (1,))

    # absolute cap: scale down rows whose max |logit| exceeds abs_cap
    if abs_cap is not None:
        max_abs = abs_vals.amax(dim=-1, keepdim=True)
        scale_abs = (float(abs_cap) / (max_abs + eps)).clamp_max(1.0)
        scales = torch.minimum(scales, scale_abs)

    # margin clamp: scale down rows whose (top1 - top2) exceeds max_margin
    if max_margin is not None:
        assert S > 2, "margin clamp requires at least 2 non-mask logits"
        top2 = torch.topk(x_topk, k=2, dim=-1).values
        margin = (top2[..., 0] - top2[..., 1]).unsqueeze(-1).clamp_min(eps)
        scale_margin = (float(max_margin) / margin).clamp_max(1.0)
        scales = torch.minimum(scales, scale_margin)

    # apply the stricter combined scale once
    y = x * scales

    # final hard clamp to guarantee bounds even with numerical noise
    if abs_cap is not None:
        y = y.clamp(min=-float(abs_cap), max=float(abs_cap))

    return y.to(orig_dtype)


def combine_logits(
    logits: Sequence[torch.Tensor],
    center: Literal["logZ", "mean", "none"] = "none",
    ignore_mask: bool = True,
) -> torch.Tensor:
    """
    Combine N already-scaled logit tensors via product-of-experts-style addition.

    To maintain exactly associative combination, each logits should be centered independently,
    and combined without any additional centering.

    - All tensors in `logits_list` must have identical shape (..., S), dtype, and device.
    - Each source should have already been pre-multiplied by its own weight and divided by its own temperature.
    - `center="logZ"` treats each source as (rowwise) log-probs; `mean` subtracts rowwise mean; `none` uses raw values.

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

    # center each source before summing (shift-invariant w.r.t. final softmax)
    centered = [
        center_logits(L, center=center, ignore_mask=ignore_mask) for L in logits
    ]

    # sum all centered sources
    combined = torch.zeros_like(ref)
    for L in centered:
        combined = combined + L

    return combined
