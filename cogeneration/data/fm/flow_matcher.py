from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

from cogeneration.config.base import InterpolantDomainConfig


@dataclass
class FlowMatcher(ABC):
    """
    Abstract base class for domain-specific flow matchers.

    Implementations handle one domain (e.g., translations, rotations, torsions, sequence)
    and provide methods for:
    - sampling from the base distribution at t=0 (sample_base)
    - corrupting ground-truth values from t=1 to intermediate t (corrupt)
    - performing an Euler step toward t=1 using a vector field/logits (euler_step)
    """

    cfg: InterpolantDomainConfig

    def __post_init__(self):
        # define in post_init to allow required fields in subclasses
        self._device = torch.device("cpu")

    def set_device(self, device: torch.device):
        self._device = device

    def effective_stochastic_scale(
        self,
        t: torch.Tensor,  # (B,)
        stochastic_scale: torch.Tensor,  # (B,)  batch[bp.stochastic_scale]
    ) -> torch.Tensor:
        """
        Compute the effective per-batch stochasticity scale at time t (B,).

        Default: if stochastic disabled or intensity==0, returns zeros; else returns
        cfg.stochastic_noise_intensity * batch[bp.stochastic_scale].
        """
        if not self.cfg.stochastic or self.cfg.stochastic_noise_intensity == 0.0:
            return torch.zeros_like(t)

        tau = self.time_sampling(t)
        return self.cfg.stochastic_noise_intensity * stochastic_scale.to(
            tau.device
        ).view(-1)

    @staticmethod
    def _compute_sigma_t(
        t: torch.Tensor,  # (B,)
        scale: torch.Tensor,  # (B,) per-domain scale * per-batch-item stochasticity scale
        min_sigma: float = 0.0,
        end_t: float = 1.0,  # time at which sigma should reach zero
    ) -> torch.Tensor:
        """
        Compute the instantaneous standard deviation of the noise at time t.

        The standard deviation follows a sqrt-parabolic schedule that is zero at t=0 and t=end_t:
            sigma(t) = sqrt(scale^2 * t' * (1 - t') + min_sigma^2)

        where t' = t / end_t (clamped to [0, 1]).
        """
        if end_t >= 1.0:
            return torch.sqrt(scale**2 * t * (1 - t) + min_sigma**2)

        # Compressed parabolic: zero at t=0 and t=end_t
        t_normalized = (t / end_t).clamp(max=1.0)
        return torch.sqrt(scale**2 * t_normalized * (1 - t_normalized) + min_sigma**2)

    def time_training(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map raw t (B,) in [0,1] to domain-specific training schedule time tau for corruption.
        Base implementation is identity.
        """
        return t

    def time_sampling(self, t: torch.Tensor) -> torch.Tensor:
        """
        Map raw t (B,) in [0,1] to domain-specific sampling schedule time tau for drift/noise usage.
        Base implementation is identity.
        """
        return t

    def stabilized_endpoint(
        self,
        t: torch.Tensor,
        target_1: torch.Tensor,
        fix_t: float,
        change_cap: float,
        change_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> torch.Tensor:
        """
        Optionally stabilize endpoint predictions after fix_t by caching the last stable target.
        Returns the target_1 tensor that should be used for the Euler step.
        """
        if change_cap <= 0.0 or fix_t >= 1.0:
            return target_1

        if t.max().item() <= fix_t:
            self._cached_target_1 = None
            return target_1

        prev_target = getattr(self, "_cached_target_1", None)
        if prev_target is None:
            self._cached_target_1 = target_1.detach().clone()
            return target_1

        delta = change_fn(prev_target, target_1)
        max_change = delta.max().item()
        if max_change > change_cap:
            return prev_target

        self._cached_target_1 = target_1.detach().clone()
        return target_1

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
