# Adapted from Boltz-2
# Note that outgoing and incoming triangle mults only really differ in `einsum()`

import torch
from cuequivariance_torch.primitives.triangle import triangle_multiplicative_update
from torch import Tensor, nn

from cogeneration.models.attention.ipa_pytorch import LayerNorm, Linear


@torch.compiler.disable
def kernel_triangular_mult(
    x,
    direction,
    mask,
    norm_in_weight,
    norm_in_bias,
    p_in_weight,
    g_in_weight,
    norm_out_weight,
    norm_out_bias,
    p_out_weight,
    g_out_weight,
    eps,
):
    # NVidia cuEquivariance kernel
    return triangle_multiplicative_update(
        x,
        direction=direction,
        mask=mask,
        norm_in_weight=norm_in_weight,
        norm_in_bias=norm_in_bias,
        p_in_weight=p_in_weight,
        g_in_weight=g_in_weight,
        norm_out_weight=norm_out_weight,
        norm_out_bias=norm_out_bias,
        p_out_weight=p_out_weight,
        g_out_weight=g_out_weight,
        eps=eps,
    )


class TriangleMultiplicationOutgoing(nn.Module):
    """TriangleMultiplicationOutgoing."""

    def __init__(self, dim: int) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input

        """
        super().__init__()

        self.norm_in = LayerNorm(dim)
        self.p_in = Linear(dim, 2 * dim, bias=False, init="lecun")
        self.g_in = Linear(dim, 2 * dim, bias=False, init="gating")

        self.norm_out = LayerNorm(dim)
        self.p_out = Linear(dim, dim, bias=False, init="final")
        self.g_out = Linear(dim, dim, bias=False, init="gating")

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        use_kernels: bool
            Whether to use the kernel

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return kernel_triangular_mult(
                x,
                direction="outgoing",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bikd,bjkd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x


class TriangleMultiplicationIncoming(nn.Module):
    """TriangleMultiplicationIncoming."""

    def __init__(self, dim: int = 128) -> None:
        """Initialize the TriangularUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128

        """
        super().__init__()

        self.norm_in = LayerNorm(dim, eps=1e-5)
        self.p_in = Linear(dim, 2 * dim, bias=False, init="lecun")
        self.g_in = Linear(dim, 2 * dim, bias=False, init="gating")

        self.norm_out = LayerNorm(dim)
        self.p_out = Linear(dim, dim, bias=False, init="final")
        self.g_out = Linear(dim, dim, bias=False, init="gating")

    def forward(self, x: Tensor, mask: Tensor, use_kernels: bool = False) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (B, N, N, D)
        mask: torch.Tensor
            The input mask of shape (B, N, N)
        use_kernels: bool
            Whether to use the kernel

        Returns
        -------
        x: torch.Tensor
            The output data of shape (B, N, N, D)

        """
        if use_kernels:
            return kernel_triangular_mult(
                x,
                direction="incoming",
                mask=mask,
                norm_in_weight=self.norm_in.weight,
                norm_in_bias=self.norm_in.bias,
                p_in_weight=self.p_in.weight,
                g_in_weight=self.g_in.weight,
                norm_out_weight=self.norm_out.weight,
                norm_out_bias=self.norm_out.bias,
                p_out_weight=self.p_out.weight,
                g_out_weight=self.g_out.weight,
                eps=1e-5,
            )

        # Input gating: D -> D
        x = self.norm_in(x)
        x_in = x
        x = self.p_in(x) * self.g_in(x).sigmoid()

        # Apply mask
        x = x * mask.unsqueeze(-1)

        # Split input and cast to float
        a, b = torch.chunk(x.float(), 2, dim=-1)

        # Triangular projection
        x = torch.einsum("bkid,bkjd->bijd", a, b)

        # Output gating
        x = self.p_out(self.norm_out(x)) * self.g_out(x_in).sigmoid()

        return x
