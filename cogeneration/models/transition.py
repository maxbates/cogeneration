# Adapted from Boltz-2

from typing import Optional

from torch import Tensor, nn

from cogeneration.models.ipa_pytorch import LayerNorm, Linear


class Transition(nn.Module):
    """Perform a two-layer MLP."""

    def __init__(
        self,
        dim: int = 128,
        hidden: int = 512,
        out_dim: Optional[int] = None,
    ) -> None:
        """Initialize the TransitionUpdate module.

        Parameters
        ----------
        dim: int
            The dimension of the input, default 128
        hidden: int
            The dimension of the hidden, default 512
        out_dim: Optional[int]
            The dimension of the output, default None

        """
        super().__init__()
        if out_dim is None:
            out_dim = dim

        self.norm = LayerNorm(dim)
        self.fc1 = Linear(dim, hidden, bias=False, init="lecun")
        self.fc2 = Linear(dim, hidden, bias=False, init="lecun")
        self.fc3 = Linear(hidden, out_dim, bias=False, init="final")
        self.silu = nn.SiLU()
        self.hidden = hidden

    def forward(self, x: Tensor, chunk_size: int = None) -> Tensor:
        """Perform a forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input data of shape (..., D)

        Returns
        -------
        x: torch.Tensor
            The output data of shape (..., D)

        """
        x = self.norm(x)

        if chunk_size is None or self.training:
            x = self.silu(self.fc1(x)) * self.fc2(x)
            x = self.fc3(x)
            return x

        # Compute in chunks
        for i in range(0, self.hidden, chunk_size):
            fc1_slice = self.fc1.weight[i : i + chunk_size, :]
            fc2_slice = self.fc2.weight[i : i + chunk_size, :]
            fc3_slice = self.fc3.weight[:, i : i + chunk_size]
            x_chunk = self.silu((x @ fc1_slice.T)) * (x @ fc2_slice.T)
            if i == 0:
                x_out = x_chunk @ fc3_slice.T
            else:
                x_out = x_out + x_chunk @ fc3_slice.T
        return x_out
