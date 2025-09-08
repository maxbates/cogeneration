from abc import ABC, abstractmethod
from typing import Any

import torch


class FlowMatcher(ABC):
    """
    Abstract base class for domain-specific flow matchers.

    Implementations handle one domain (e.g., translations, rotations, torsions, sequence)
    and provide methods for:
    - sampling from the base distribution at t=0 (sample_base)
    - corrupting ground-truth values from t=1 to intermediate t (corrupt)
    - performing an Euler step toward t=1 using a vector field/logits (euler_step)
    """

    @abstractmethod
    def set_device(self, device: torch.device):
        """Set the device for the flow matcher."""
        raise NotImplementedError

    @staticmethod
    def _compute_sigma_t(
        t: torch.Tensor, scale: float = 1.0, min_sigma: float = 0.0
    ) -> torch.Tensor:
        """
        Compute the instantaneous standard deviation of the noise at time t.

        The standard deviation follows a sqrt-parabolic schedule that is zero at t=0 and t=1:
            sigma(t) = sqrt(scale^2 * t * (1 - t) + min_sigma^2)
        """
        return torch.sqrt(scale**2 * t * (1 - t) + min_sigma**2)

    def time_training(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map raw t in [0,1] to domain-specific training schedule time tau for corruption.
        Base implementation is identity.
        """
        return t

    def time_sampling(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map raw t in [0,1] to domain-specific sampling schedule time tau for drift/noise usage.
        Base implementation is identity.
        """
        return t

    @abstractmethod
    def sample_base(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Sample from the domain's base distribution (t=0)."""
        raise NotImplementedError

    @abstractmethod
    def corrupt(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Corrupt t=1 values to time t according to the domain's interpolation/noise."""
        raise NotImplementedError

    @abstractmethod
    def euler_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        """Perform a single Euler step update for the domain."""
        raise NotImplementedError
