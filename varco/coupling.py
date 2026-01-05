from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Optional, Tuple, TypeVar

import torch

from varco.tree_plan import BatchedTreePlan


@dataclass
class Coupling(ABC):
    """
    Coupling struct captures domain-specific creation / anchor values for corruption tree plan.
    """

    tree: BatchedTreePlan
    anchors: torch.Tensor  # (B, A, ...)
    creation_state: torch.Tensor  # (B, A, ...)

    def validate(self) -> None:
        return


CouplingT = TypeVar("CouplingT")


class Coupler(ABC, Generic[CouplingT]):
    """Abstract interface for a domain coupler.

    A coupler owns:
      - how to sample the base distribution for its domain (`sample_base`)
      - how to corrupt a domain endpoint `x1` to time `t` using a shared tree plan (`corrupt`)

    The coupler returns the corrupted feature aligned to the tree capacity A, plus a domain-specific
    coupling/targets object used for supervision.
    """

    @abstractmethod
    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool - True for motif positions
        x1: torch.Tensor,  # (B, N, ...) endpoint values at t=1
        device: torch.device,
    ) -> torch.Tensor:
        """Sample base distribution values at t=0.

        Args:
            motif_mask: (B, N) bool mask where True indicates motif positions
            x1: (B, N, ...) endpoint values at t=1 (for motifs to preserve)
            device: device to create tensors on

        Returns:
            x0: (B, N, ...) base values at t=0
        """
        raise NotImplementedError

    @abstractmethod
    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, ...)
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Combine two child anchor values into parent anchor.

        Args:
            child_anchors: (N_valid, 2, ...) anchor values for the two children
            child_weights: (N_valid, 2) total_leaves weights for the two children

        Returns:
            parent_anchors: (N_valid, ...) combined anchor for parent
        """
        raise NotImplementedError

    @abstractmethod
    def sample_bridge(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        s: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample bridge from x_start at t0 to x_end at t=1, evaluated at time s.

        Args:
            x_start: Start values at time t0
            x_end: End values at time t=1
            s: Time to evaluate bridge
            t0: Birth time

        Returns:
            Sampled values at time s
        """
        raise NotImplementedError

    @abstractmethod
    def post_process(
        self,
        x_t: torch.Tensor,
        present_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Apply domain-specific masking for present_mask and motif_mask.

        Args:
            x_t: (B, A, ...) values at time t before masking
            present_mask: (B, A) True for nodes present at time t
            motif_mask: (B, A) True for motif nodes
            anchors: (B, A, ...) anchor values

        Returns:
            x_t: (B, A, ...) masked values
        """
        raise NotImplementedError

    def build_anchors(
        self,
        x1: torch.Tensor,
        tree: BatchedTreePlan,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Generic anchor building using depth-based traversal.

        Args:
            x1: (B, N, ...) endpoint values at t=1
            tree: Batched tree plan
            fill_value: Value to use for padding/internal nodes initially

        Returns:
            anchors: (B, A, ...) anchor values for all nodes
        """
        anchors = tree.broadcast_to_leaves(x1, fill_value=fill_value)

        def combine_fn(batch_idx, node_idx, children, child_weights, node_values):
            child_anchors = torch.stack(
                [
                    node_values[batch_idx, children[:, 0]],
                    node_values[batch_idx, children[:, 1]],
                ],
                dim=1,
            )
            return self.combine_anchors(
                child_anchors=child_anchors, child_weights=child_weights
            )

        return tree.traverse_bottom_up(anchors, combine_fn)

    @staticmethod
    def _compute_sigma_t(
        t: torch.Tensor,  # (B,)
        scale: torch.Tensor,  # (B,) per-domain scale
        min_sigma: float = 0.0,
        noise_end_t: float = 0.95,
    ) -> torch.Tensor:
        """
        Compute the instantaneous standard deviation of the noise at time t.

        Uses a sqrt-parabolic schedule with boundary conditions sigma(0)=sigma(1)=min_sigma:
            sigma(t) = sqrt(scale^2 * t * (1 - t) + min_sigma^2)

        Additionally, to make the final sampling steps noise-free, time is warped so that
        sigma(t) becomes 0 (up to min_sigma) for t >= noise_end_t.
        """
        if t.ndim != 1:
            raise ValueError(f"Expected t to have shape (B,); got {tuple(t.shape)}")
        if scale.shape != t.shape:
            raise ValueError(
                f"Expected scale to have shape (B,); got {tuple(scale.shape)} vs {tuple(t.shape)}"
            )
        if not (0.0 < float(noise_end_t) <= 1.0):
            raise ValueError(f"Expected noise_end_t in (0, 1]; got {noise_end_t}")

        t_eff = (t / float(noise_end_t)).clamp(0.0, 1.0)
        return torch.sqrt(
            scale.square() * t_eff * (1.0 - t_eff) + float(min_sigma) ** 2
        )

    def bridge_step(
        self,
        coupling: CouplingT,
        x_prev: torch.Tensor,  # (B, A, ...)
        t_prev: float,
        t_next: float,
    ) -> torch.Tensor:
        """Advance a single coupled bridge path from t_prev -> t_next.

        This is intended for time-coupled visualization / debugging: it reuses a fixed
        sampled coupling (anchors + creation_state) and produces a single stochastic
        trajectory instead of independent marginal samples at each timepoint.

        Notes:
        - Per-position start time is clamped as t0 = max(t_prev, birth_time).
        - For positions not yet born at t_prev, we start from coupling.creation_state.
        - post_process() is applied at t_next (masking, motif fixing, etc.).
        """
        if float(t_next) < float(t_prev):
            raise ValueError(f"Expected t_next >= t_prev; got {t_next} < {t_prev}")

        tree = coupling.tree
        birth_time = tree.birth_time  # (B, A)

        # validate x_prev is in aligned space, matches coupling states
        if (
            x_prev.shape != coupling.creation_state.shape
            or x_prev.shape != coupling.anchors.shape
        ):
            raise ValueError(
                "x_prev, coupling.creation_state, and coupling.anchors must have the same shape; "
                f"got x_prev={tuple(x_prev.shape)}, creation_state={tuple(coupling.creation_state.shape)}, "
                f"anchors={tuple(coupling.anchors.shape)}"
            )

        t_prev_full = torch.full_like(birth_time, float(t_prev))  # (B, A)
        t_next_full = torch.full_like(birth_time, float(t_next))  # (B, A)

        # start time is min_clamped by birth_time
        t0 = torch.maximum(t_prev_full, birth_time)

        # start state is creation state if t_prev < birth_time, otherwise provided x_prev
        use_prev = birth_time <= float(t_prev)  # (B, A)
        use_prev_expanded = use_prev
        while use_prev_expanded.ndim < x_prev.ndim:
            use_prev_expanded = use_prev_expanded.unsqueeze(-1)
        x_start = torch.where(use_prev_expanded, x_prev, coupling.creation_state)

        # sample bridge from x_prev (x_start) -> t_next (anchors)
        x_next = self.sample_bridge(
            x_start=x_start,  # (B, A, ...)
            x_end=coupling.anchors,  # (B, A, ...)
            s=t_next_full,
            t0=t0,
        )

        # Apply domain-specific masking / constraints at t_next
        present_mask = tree.present_mask(
            t=torch.full(
                (birth_time.shape[0],), float(t_next), device=birth_time.device
            )
        )
        x_next = self.post_process(
            x_t=x_next,
            present_mask=present_mask,
            motif_mask=tree.motif_mask,
            anchors=coupling.anchors,
        )

        return x_next

    def corrupt(
        self,
        tree: BatchedTreePlan,
        t: torch.Tensor,  # (B,)
        x1: torch.Tensor,  # (B, N, ...)
        x0: Optional[torch.Tensor] = None,  # (B, A, ...)
    ) -> Tuple[torch.Tensor, Coupling]:
        """Generic corruption using depth-based traversal.

        Args:
            tree: Batched tree plan
            t: (B,) corruption time for each example
            x1: (B, N, ...) endpoint values at t=1
            x0: (B, A, ...) optional base values at t=0 (aligned to tree)

        Returns:
            x_t: (B, A, ...) corrupted values at time t
            coupling: Domain-specific coupling object
        """
        device = x1.device
        B = x1.shape[0]
        A = tree.parent_idx.shape[1]
        t_expanded = t.unsqueeze(1).expand(B, A)

        if x0 is None:
            # Broadcast x1 to aligned space (copies leaves, fills internal nodes)
            x1_aligned = tree.broadcast_to_leaves(x1, fill_value=0)
            # Sample base distribution for all positions
            x0 = self.sample_base(
                motif_mask=tree.motif_mask,
                x1=x1_aligned,
                device=device,
            )

        # traverse bottom up to get anchors (which aggregate leaves).
        # for trans and rots, anchors are deterministic "centers" of leaves
        # for aatypes, anchor probs are deterministic but we *sample* an anchor token
        anchors = self.build_anchors(x1=x1, tree=tree)

        def split_fn(node_creation, node_target, node_t0, node_st):
            return self.sample_bridge(
                x_start=node_creation,
                x_end=node_target,
                s=node_st,
                t0=node_t0,
            )

        # traverse top-down from roots -> anchors -> leaves to get creation states
        # motifs + roots will use t=0 state. anchors states determined by bridges.
        creation_state = x0.clone()
        creation_state = tree.traverse_top_down(
            creation_state=creation_state,
            target_state=anchors,
            split_fn=split_fn,
            max_split_time=t_expanded,
        )

        # track domain's coupling: creation state + anchors (target states)
        coupling = self._make_coupling(
            tree=tree, anchors=anchors, creation_state=creation_state
        )

        # sample bridge from creation state -> anchor/terminal state to get state at t
        x_t = self.sample_bridge(
            x_start=creation_state,
            x_end=anchors,
            s=t_expanded,
            t0=tree.birth_time,
        )

        if x_t.dtype.is_floating_point and not torch.isfinite(x_t).all():
            raise RuntimeError(
                "sample_bridge produced non-finite values. "
                "Domain couplers must clamp time deltas for unborn nodes."
            )

        # domain specific post-processing, mostly null out non-present nodes
        x_t = self.post_process(
            x_t=x_t,
            present_mask=tree.present_mask(t=t),
            motif_mask=tree.motif_mask,
            anchors=anchors,
        )

        return x_t, coupling

    @abstractmethod
    def _make_coupling(
        self,
        tree: BatchedTreePlan,
        anchors: torch.Tensor,  # (B, A, ...)
        creation_state: torch.Tensor,  # (B, A, ...)
    ) -> Coupling:
        """Create domain-specific coupling object."""
        raise NotImplementedError

    @abstractmethod
    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P, d)
        x1_pred: torch.Tensor,  # (B, P, d)
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, ...)
    ) -> torch.Tensor:
        """Single Euler(-Maruyama) step for sampling.

        Noise is controlled by the coupler instance (e.g. sigma); if sigma is None/0, this is deterministic.
        Guidance potential can be provided to pull toward targets.
        """
        raise NotImplementedError
