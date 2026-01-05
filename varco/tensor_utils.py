import math
from typing import List, Optional, Union

import torch


class SeededRNG:
    """Seeded random number generator wrapping torch.Generator for reproducibility."""

    def __init__(self, seed: Optional[int] = None, device: str = "cpu"):
        self.rng = torch.Generator(device=device)
        if seed is None:
            seed = int(torch.seed() % (2**31 - 1))
        self.rng.manual_seed(int(seed))

    def rand_int(self, high: int) -> int:
        """Sample uniform integer from [0, high)."""
        return int(torch.randint(0, high, (1,), generator=self.rng).item())

    def rand_float(self) -> float:
        """Sample uniform float from [0, 1)."""
        return float(torch.rand(1, generator=self.rng).squeeze().item())

    def rand(self, size: int, device: str = "cpu") -> torch.Tensor:
        """Sample uniform floats from [0, 1) as a tensor"""
        return torch.rand(size, generator=self.rng, device=device)

    def sample_exp1(self) -> float:
        """Sample from Exp(1) distribution via inverse CDF."""
        u = float(torch.rand((), generator=self.rng).clamp_min(1e-12).item())
        return -math.log(u)

    def sample_poisson(self, lam: float) -> int:
        """Sample from Poisson(lam) distribution using Knuth's algorithm."""
        if lam <= 0.0:
            return 0
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while True:
            k += 1
            p *= self.rand_float()
            if p <= L:
                return k - 1

    def sample_beta(
        self, size: int, alpha: float = 1.0, beta: float = 1.0, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Sample from Beta(alpha, beta) distribution.

        Uses the ratio of gamma variates method: if X ~ Gamma(alpha, 1) and Y ~ Gamma(beta, 1),
        then X / (X + Y) ~ Beta(alpha, beta).

        For alpha < beta, the distribution is biased toward 0 (earlier times).
        For alpha > beta, the distribution is biased toward 1 (later times).
        For alpha = beta = 1, this is uniform on [0, 1].

        Args:
            size: Number of samples to generate
            alpha: First shape parameter (controls left tail)
            beta: Second shape parameter (controls right tail)
            device: Device for output tensor

        Returns:
            Tensor of shape (size,) with samples in [0, 1]
        """
        if size == 0:
            return torch.empty(0, device=device)
        if alpha <= 0 or beta <= 0:
            raise ValueError("alpha and beta must be positive")

        # Sample from Gamma distributions using the Marsaglia and Tsang method
        def sample_gamma(a: float, n: int) -> torch.Tensor:
            if a < 1:
                # For a < 1, use: Gamma(a) = Gamma(a+1) * U^(1/a)
                g = sample_gamma(a + 1, n)
                u = torch.rand(n, generator=self.rng, device="cpu")
                return g * (u ** (1.0 / a))

            d = a - 1.0 / 3.0
            c = 1.0 / math.sqrt(9.0 * d)

            samples = []
            while len(samples) < n:
                batch_size = max(n - len(samples), 64)
                z = torch.randn(batch_size, generator=self.rng, device="cpu")
                u = torch.rand(batch_size, generator=self.rng, device="cpu")

                v = (1.0 + c * z) ** 3
                valid = (z > -1.0 / c) & (
                    torch.log(u) < 0.5 * z**2 + d * (1.0 - v + torch.log(v))
                )
                samples.extend((d * v)[valid].tolist())

            return torch.tensor(samples[:n], device="cpu")

        x = sample_gamma(alpha, size)
        y = sample_gamma(beta, size)
        return (x / (x + y)).to(device=device)


def clone_detach(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    """Clone and detach an Optional tensor."""
    if x is None:
        return None
    return x.detach().clone()


def to_device(
    x: Optional[torch.Tensor], device: torch.device
) -> Optional[torch.Tensor]:
    """Move an Optional tensor to a device."""
    if x is None:
        return None
    return x.to(device=device)


def gather_and_pad(
    source: Optional[torch.Tensor],  # (B, N, ...) or (B, N, N) if is_2d
    index: torch.Tensor,  # (B, P)
    mask: torch.Tensor,  # (B, P)
    fill_value: Union[float, torch.Tensor] = 0.0,
    is_2d: bool = False,
) -> Optional[torch.Tensor]:  # (B, P, ...) or (B, P, P) if is_2d
    """
    Gather from source along dim=1 using index, then fill padding positions with fill_value.

    Handles arbitrary trailing dimensions by expanding index and mask.
    For positions where mask is False, the result is set to fill_value.

    Args:
        source: (B, N, ...) tensor to gather from, or (B, N, N) if is_2d=True
        index: (B, P) indices into dim=1 of source (must be in [0, N-1])
        mask: (B, P) boolean mask; True for valid positions, False for padding
        fill_value: value to fill where mask is False. Can be a scalar float or
                    a tensor with shape matching source trailing dimensions (...).
        is_2d: if True, source is (B, N, N) and we gather along both dim=1 and dim=2
               to produce (B, P, P). Used for contact_conditioning matrices.

    Returns:
        (B, P, ...) tensor with gathered values where mask is True, fill_value otherwise
        If is_2d=True, returns (B, P, P) tensor.
    """
    if source is None:
        return None

    B, P = index.shape

    if is_2d:
        # source is (B, N, N), gather along both dimensions to get (B, P, P)
        # First gather rows: (B, N, N) -> (B, P, N)
        idx_row = index.unsqueeze(-1).expand(-1, -1, source.shape[2])  # (B, P, N)
        gathered_rows = source.gather(1, idx_row)  # (B, P, N)
        # Then gather columns: (B, P, N) -> (B, P, P)
        idx_col = index.unsqueeze(1).expand(-1, P, -1)  # (B, P, P)
        gathered = gathered_rows.gather(2, idx_col)  # (B, P, P)

        # Fill value for 2D case
        if isinstance(fill_value, torch.Tensor):
            fill = fill_value.unsqueeze(0).unsqueeze(0).expand(B, P, P)
            fill = fill.to(device=gathered.device, dtype=gathered.dtype)
        else:
            fill = torch.full_like(gathered, fill_value)

        # 2D mask: both row and column must be valid
        mask_2d = mask.unsqueeze(2) & mask.unsqueeze(1)  # (B, P, P)
        return torch.where(mask_2d, gathered, fill)

    # Standard 1D case
    trailing_shape = source.shape[2:]
    idx = index
    for _ in trailing_shape:
        idx = idx.unsqueeze(-1)
    idx = idx.expand(-1, -1, *trailing_shape)  # (B, P, ...)

    gathered = source.gather(1, idx)  # (B, P, ...)

    # Handle fill value - either scalar or tensor
    if isinstance(fill_value, torch.Tensor):
        # Tensor fill value: broadcast to (B, P, ...)
        fill = fill_value.unsqueeze(0).unsqueeze(0).expand(B, P, *trailing_shape)
        fill = fill.to(device=gathered.device, dtype=gathered.dtype)
    else:
        # Scalar fill value
        fill = torch.full_like(gathered, fill_value)

    if not trailing_shape:
        return torch.where(mask, gathered, fill)
    else:
        m = mask
        for _ in trailing_shape:
            m = m.unsqueeze(-1)
        m = m.expand_as(gathered)
        return torch.where(m, gathered, fill)


def pad_and_stack(
    tensors: List[torch.Tensor],  # (B, P, ...)
    max_len: int,
    fill_value: float = 0.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:  # (B, P_max, ...)
    """
    Pad a list of 1D or 2D tensors to max_len and stack into a batch.

    Args:
        tensors: list of (P, ...) tensors with varying P
        max_len: target length to pad to
        fill_value: value to fill padding positions
        dtype: output dtype (defaults to first tensor's dtype)

    Returns:
        (B, P_max, ...) tensor
    """
    B = len(tensors)
    if B == 0:
        raise ValueError("Empty tensor list")

    first = tensors[0]
    dtype = dtype or first.dtype
    trailing_shape = first.shape[1:]  # () for 1D, (D,) for 2D

    out = torch.full((B, max_len, *trailing_shape), fill_value, dtype=dtype)
    for b, t in enumerate(tensors):
        L = t.shape[0]
        out[b, :L] = t.to(dtype)
    return out
