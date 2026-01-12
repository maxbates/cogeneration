from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.nn import functional as F

from cogeneration.data.const import MASK_TOKEN_INDEX, NUM_TOKENS
from cogeneration.data.noise_mask import uniform_categorical
from varco.config import VarcoInterpolantAATypesCouplerConfig
from varco.coupling import Coupler, Coupling
from varco.tree_plan import BatchedTreePlan

# TODO - support aatypes guidance potential, support particle-free ESM potential


@dataclass
class AATypesCoupling(Coupling):
    """Coupling for amino acid types using CTMC bridge."""

    # Anchor probability distributions (before sampling)
    anchor_probs: torch.Tensor  # (B, A, K)


class AATypesCoupler(Coupler[AATypesCoupling]):
    """
    Coupler for amino acid types using a continuous-time Markov chain (CTMC) bridge.

    Uses a uniform substitution CTMC (complete graph) with closed-form transition probabilities.
    The bridge maintains creation_token at birth_time and anchor token at t=1 for each node.
    """

    K: int = NUM_TOKENS + 1  # 21 tokens (20 amino acids + mask/X token at index 20)

    def __init__(
        self,
        cfg: VarcoInterpolantAATypesCouplerConfig,
    ):
        self.cfg = cfg

    def _aa_only_renormalize(
        self,
        probs: torch.Tensor,  # (..., K)
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Renormalize probabilities over amino-acid tokens only (exclude MASK_TOKEN_INDEX).

        If a row has (near-)zero AA mass, falls back to uniform over AA tokens.
        """
        if probs.shape[-1] != self.K:
            raise ValueError(
                f"Expected probs last dim K={self.K}; got shape {tuple(probs.shape)}"
            )

        out = probs.clone()
        out[..., MASK_TOKEN_INDEX] = 0.0

        row_sum = out.sum(dim=-1, keepdim=True)
        has_mass = row_sum > eps

        uniform_aa = torch.ones_like(out)
        uniform_aa[..., MASK_TOKEN_INDEX] = 0.0
        uniform_aa = uniform_aa / uniform_aa.sum(dim=-1, keepdim=True).clamp_min(eps)

        out = torch.where(has_mass, out / row_sum.clamp_min(eps), uniform_aa)
        return out

    def sample_base(
        self,
        motif_mask: torch.Tensor,  # (B, N) bool
        x1: torch.Tensor,  # (B, N) amino acid indices
        device: torch.device,
    ) -> torch.Tensor:
        """Sample base distribution amino acid types at t=0.

        Preserves motif sequences and samples random sequences for scaffolds.
        """
        B, N = motif_mask.shape
        # Sample uniform random amino acids for all positions (exclude MASK token).
        x0 = uniform_categorical(B, N, num_tokens=NUM_TOKENS, device=device)
        # Preserve motif sequences from x1
        x0 = torch.where(motif_mask, x1, x0)
        return x0

    def combine_anchors(
        self,
        child_anchors: torch.Tensor,  # (N_valid, 2, K) probability distributions
        child_weights: torch.Tensor,  # (N_valid, 2)
    ) -> torch.Tensor:
        """Weighted mixture of child probability distributions."""
        wsum = child_weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        weights = (child_weights.float() / wsum).unsqueeze(-1)  # (N_valid, 2, 1)
        return (child_anchors * weights).sum(dim=1)  # (N_valid, K)

    def _transition_prob(
        self,
        delta: torch.Tensor,  # (...) time delta
        same: bool = True,  # True if staying prob, False if transition prob
    ) -> torch.Tensor:
        """
        Compute CTMC transition probability for uniform substitution model.

        For a uniform substitution CTMC with leaving rate β:
        - P_ii(Δ) = 1/K + (1 - 1/K) * exp(-λΔ)
        - P_ij(Δ) = 1/K - (1/K) * exp(-λΔ)  for i ≠ j

        where λ = β * K / (K - 1)
        """
        delta = delta.clamp_min(0.0)
        lam = self.cfg.beta * self.K / (self.K - 1)
        exp_term = torch.exp(-lam * delta)
        inv_K = 1.0 / self.K

        if same:
            return inv_K + (1.0 - inv_K) * exp_term
        else:
            return inv_K - inv_K * exp_term

    def _bridge_marginal(
        self,
        token_start: torch.Tensor,  # (M,) long, token at t0
        token_end: torch.Tensor,  # (M,) long, token at t=1
        s: torch.Tensor,  # (M,) time to sample at
        t0: torch.Tensor,  # (M,) birth time
    ) -> torch.Tensor:
        """
        Sample from CTMC bridge marginal at time s conditioned on endpoints.

        For a CTMC bridge from X_{t0}=i to X_1=j, the marginal at time s is:
        P(X_s = k | X_{t0} = i, X_1 = j) ∝ P(s - t0)_{i,k} * P(1 - s)_{k,j}
        """
        M = token_start.shape[0]
        device = token_start.device

        # Time intervals
        delta_start = (s - t0).clamp_min(0.0)  # (M,)
        delta_end = (1.0 - s).clamp_min(0.0)  # (M,)

        # Build transition probability matrix rows
        # P(s - t0)_{i,k} for all k
        # P(1 - s)_{k,j} for all k
        K = self.K

        # Compute probabilities for each candidate token k
        # P(s - t0)_{i,k}: probability of going from i to k in time (s - t0)
        # P(1 - s)_{k,j}: probability of going from k to j in time (1 - s)

        # For efficiency, compute staying vs leaving probabilities
        p_stay_start = self._transition_prob(delta_start, same=True)  # (M,)
        p_jump_start = self._transition_prob(delta_start, same=False)  # (M,)
        p_stay_end = self._transition_prob(delta_end, same=True)  # (M,)
        p_jump_end = self._transition_prob(delta_end, same=False)  # (M,)

        # Build full probability vector for each token k
        # P_{i,k}(Δ1) = p_stay_start if k == i else p_jump_start
        # P_{k,j}(Δ2) = p_stay_end if k == j else p_jump_end

        # Create one-hot indicators
        k_range = torch.arange(K, device=device).unsqueeze(0)  # (1, K)
        is_start = (k_range == token_start.unsqueeze(1)).float()  # (M, K)
        is_end = (k_range == token_end.unsqueeze(1)).float()  # (M, K)

        # Transition probs from start to each k
        p_start_to_k = is_start * p_stay_start.unsqueeze(1) + (
            1.0 - is_start
        ) * p_jump_start.unsqueeze(
            1
        )  # (M, K)

        # Transition probs from each k to end
        p_k_to_end = is_end * p_stay_end.unsqueeze(1) + (
            1.0 - is_end
        ) * p_jump_end.unsqueeze(
            1
        )  # (M, K)

        # Bridge marginal: P(X_s = k) ∝ P_{i,k}(Δ1) * P_{k,j}(Δ2)
        probs = p_start_to_k * p_k_to_end  # (M, K)

        # Normalize, handling zero-sum rows (invalid/padding nodes)
        row_sums = probs.sum(dim=-1, keepdim=True)  # (M, 1)
        has_mass = row_sums > 1e-12  # (M, 1)

        # For rows with no mass (padding), use uniform distribution to allow multinomial
        uniform_fallback = torch.ones(M, K, device=device) / K
        probs = torch.where(
            has_mass, probs / row_sums.clamp_min(1e-12), uniform_fallback
        )

        # Sample
        sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)  # (M,)

        # For invalid rows, return the start token as fallback
        sampled = torch.where(has_mass.squeeze(-1), sampled, token_start)

        return sampled

    def build_anchor_alignment(
        self,
        aatypes_1: torch.Tensor,
        tree: BatchedTreePlan,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build anchor tokens and probability distributions bottom-up.

        Args:
            aatypes_1: (B, N) amino acid types at t=1
            tree: Batched tree plan

        Returns:
            anchor_tokens: (B, A) long, sampled anchor tokens for all nodes
            anchor_probs: (B, A, K) probability distributions for all nodes
        """
        if aatypes_1.ndim != 2:
            raise ValueError(f"Expected aatypes_1 shape (B, N); got {aatypes_1.shape}")

        device = aatypes_1.device
        B, N = aatypes_1.shape
        A = tree.parent_idx.shape[1]
        K = self.K

        leaf_tokens = tree.broadcast_to_leaves(
            aatypes_1.to(torch.long),
            fill_value=MASK_TOKEN_INDEX,
        )

        anchor_probs = torch.zeros(B, A, K, device=device)

        leaf_mask = tree.leaf_mask
        leaf_idx = leaf_tokens.clamp(0, K - 1)
        anchor_probs.scatter_(
            -1,
            leaf_idx.unsqueeze(-1),
            leaf_mask.unsqueeze(-1).float(),
        )

        def combine_fn(batch_idx, node_idx, children, child_weights, node_values):
            child_probs = torch.stack(
                [
                    node_values[batch_idx, children[:, 0]],
                    node_values[batch_idx, children[:, 1]],
                ],
                dim=1,
            )
            return self.combine_anchors(
                child_anchors=child_probs,
                child_weights=child_weights,
            )

        anchor_probs = tree.traverse_bottom_up(anchor_probs, combine_fn)

        # uniform fallback where no mass for multinomial()
        row_sums = anchor_probs.sum(dim=-1, keepdim=True)
        has_mass = row_sums > 1e-12
        uniform_fallback = torch.ones(B, A, K, device=device) / K
        anchor_probs = torch.where(
            has_mass,
            anchor_probs / row_sums.clamp_min(1e-12),
            uniform_fallback,
        )

        # Mix noise into anchor_probs to sample anchor tokens
        noisy_anchor_probs = anchor_probs
        if self.cfg.noise_scale > 0:
            noise_weight = min(self.cfg.noise_scale * 0.15, 0.2)
            noisy_anchor_probs = (
                1.0 - noise_weight
            ) * noisy_anchor_probs + noise_weight * uniform_fallback
            # disallow mask token for anchor tokens
            noisy_anchor_probs[:, :, MASK_TOKEN_INDEX] = 0.0
            noisy_anchor_probs = noisy_anchor_probs / noisy_anchor_probs.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)

        anchor_probs_flat = noisy_anchor_probs.view(-1, K)
        anchor_tokens = torch.multinomial(
            anchor_probs_flat,
            num_samples=1,
        ).squeeze(-1)
        anchor_tokens = anchor_tokens.view(B, A)

        anchor_tokens = torch.where(leaf_mask, leaf_tokens, anchor_tokens)

        anchor_tokens = torch.where(
            has_mass.squeeze(-1),
            anchor_tokens,
            torch.full_like(anchor_tokens, MASK_TOKEN_INDEX),
        )

        return anchor_tokens, anchor_probs

    def build_anchors(
        self,
        x1: torch.Tensor,
        tree: BatchedTreePlan,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        """Override to call build_anchor_alignment and return tokens only."""
        anchor_tokens, anchor_probs = self.build_anchor_alignment(
            aatypes_1=x1,
            tree=tree,
        )
        self._last_anchor_probs = anchor_probs
        return anchor_tokens

    def sample_bridge(
        self,
        x_start: torch.Tensor,
        x_end: torch.Tensor,
        s: torch.Tensor,
        t0: torch.Tensor,
    ) -> torch.Tensor:
        """Sample CTMC bridge marginal at time s."""
        original_shape = x_start.shape
        return self._bridge_marginal(
            token_start=x_start.reshape(-1),
            token_end=x_end.reshape(-1),
            s=s.reshape(-1),
            t0=t0.reshape(-1),
        ).view(original_shape)

    def post_process(
        self,
        x_t: torch.Tensor,
        present_mask: torch.Tensor,
        motif_mask: torch.Tensor,
        anchors: torch.Tensor,
    ) -> torch.Tensor:
        """Mask non-present nodes and fix motif sequence."""
        x_t = torch.where(
            present_mask,
            x_t,
            torch.full_like(x_t, MASK_TOKEN_INDEX),
        )
        x_t = torch.where(motif_mask, anchors, x_t)
        return x_t

    def _make_coupling(
        self,
        tree: BatchedTreePlan,
        anchors: torch.Tensor,  # (B, A)
        creation_state: torch.Tensor,  # (B, A)
    ) -> AATypesCoupling:
        return AATypesCoupling(
            tree=tree,
            anchors=anchors,
            creation_state=creation_state,
            anchor_probs=self._last_anchor_probs,  # (B, A, K)
        )

    def _uncertainty_gate(
        self,
        x_t: torch.Tensor,  # (B, P) current tokens
        probs: torch.Tensor,  # (B, P, K) predicted probabilities
    ) -> torch.Tensor:
        """
        Compute uncertainty gate: 1 - p(current token), raised to sharpness.
        Reduces jump probability when the model is confident about the current token.
        """
        K = self.K
        # Get probability of current token
        p_current = probs.gather(-1, x_t.long().clamp(0, K - 1).unsqueeze(-1)).squeeze(
            -1
        )
        p_current = p_current.clamp(0.0, 1.0)  # (B, P)

        # Uncertainty = (1 - p_current) ^ sharpness
        uncertainty = (1.0 - p_current) ** self.cfg.uncertainty_sharpness
        return uncertainty  # (B, P)

    def _compute_step_probs(
        self,
        logits: torch.Tensor,  # (B, P, K) predicted logits for t=1
        x_t: torch.Tensor,  # (B, P) current tokens
        t: torch.Tensor,  # (B,) current time
        dt: float,
        valid_mask: torch.Tensor,  # (B, P) positions that are valid (born)
    ) -> torch.Tensor:
        """
        Compute regularized step probabilities using the legacy heuristic drift sampler (not CTMC).
        This treats the model logits as a drift target with a hand-designed 1/(1-t) schedule.
        Applies temperature, uncertainty gate, noise, leave mass cap, and regularization.
        """
        B, P = x_t.shape
        K = self.K
        device = x_t.device

        # Softmax with temperature
        probs = F.softmax(logits / self.cfg.drift_temp, dim=-1)  # (B, P, K)
        probs = self._aa_only_renormalize(probs)

        # Uncertainty gating
        uncertainty = self._uncertainty_gate(x_t, probs)  # (B, P)

        # Drift gain: 1 / (1 - t), clamped
        t_clamped = t.clamp(0.0, 0.99)
        drift_gain = 1.0 / (1.0 - t_clamped)
        drift_gain = drift_gain.clamp(max=float(self.cfg.drift_gain_cap)).view(
            B, 1, 1
        )  # (B, 1, 1)

        # Compute off-diagonal drift mass
        step_probs = dt * drift_gain * probs * uncertainty.unsqueeze(-1)  # (B, P, K)

        # Zero out current token (off-diagonal only)
        current_onehot = F.one_hot(x_t.long().clamp(0, K - 1), num_classes=K).float()
        step_probs = step_probs * (1.0 - current_onehot)

        # Noise injection: add uniform mass to off-diagonal, scaled by sigma_t
        if self.cfg.noise_scale > 0:
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=torch.ones_like(t),
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            ).view(B, 1, 1)
            noise_weight = float(self.cfg.noise_scale) * dt * sigma_t.square()

            # Uniform over non-current AA tokens (exclude MASK).
            uniform_noise = (1.0 - current_onehot) / max(K - 1, 1)
            uniform_noise[..., MASK_TOKEN_INDEX] = 0.0
            uniform_noise = uniform_noise / uniform_noise.sum(
                dim=-1, keepdim=True
            ).clamp_min(1e-12)
            step_probs = step_probs + noise_weight * uniform_noise

        # Cap leave mass
        if self.cfg.leave_mass_cap is not None and self.cfg.leave_mass_cap > 0:
            row_sum = step_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            shrink = (self.cfg.leave_mass_cap / row_sum).clamp_max(1.0)
            step_probs = step_probs * shrink

        # Regularize: set diagonal to 1 - sum(off-diagonal)
        row_sum = step_probs.sum(dim=-1, keepdim=True)
        step_probs = step_probs + current_onehot * (1.0 - row_sum)

        # Clamp to valid range
        step_probs = step_probs.clamp(min=0.0, max=1.0)

        # For invalid positions: set to "stay" distribution (100% on current token)
        # This avoids wasted compute and makes debugging clearer
        stay_dist = current_onehot  # (B, P, K)
        step_probs = torch.where(valid_mask.unsqueeze(-1), step_probs, stay_dist)

        # For valid positions, MASK is not a valid residue state.
        step_probs = torch.where(
            valid_mask.unsqueeze(-1),
            self._aa_only_renormalize(step_probs),
            step_probs,
        )
        return step_probs

    def _ctmc_step_probs(
        self,
        logits: torch.Tensor,  # (B, P, K) predicted logits for endpoint distribution
        x_t: torch.Tensor,  # (B, P) current tokens
        t: torch.Tensor,  # (B,) current time
        dt: float,
        valid_mask: torch.Tensor,  # (B, P) positions that are valid (born)
        eps: float = 1e-8,
    ) -> torch.Tensor:
        """
        Compute step probabilities using a CTMC(beta) Doob h-transform (matches the CTMC corruption model).
        This conditions a uniform-substitution CTMC on the model's predicted endpoint distribution; we also
        apply an uncertainty gate to reduce substitutions when the model is confident about the current token.
        """
        B, P = x_t.shape
        K = self.K

        # Endpoint distribution q_end from logits (optionally tempered).
        q_end = F.softmax(logits / float(self.cfg.drift_temp), dim=-1)  # (B, P, K)
        q_end = self._aa_only_renormalize(q_end, eps=eps)

        # Uncertainty gating: suppress substitutions when q_end is confident about the current token.
        # This is heuristic (not part of the base CTMC model), but tends to stabilize sampling.
        uncertainty = self._uncertainty_gate(x_t=x_t, probs=q_end)  # (B, P)

        # Optional exploration noise: mix q_end with uniform mass early/mid-trajectory.
        if float(self.cfg.noise_scale) > 0.0 and float(dt) > 0.0:
            sigma_t = self._compute_sigma_t(
                t=t,
                scale=torch.ones_like(t),
                min_sigma=0.0,
                noise_end_t=float(self.cfg.noise_end_t),
            )  # (B,)
            mix = (
                float(self.cfg.noise_scale)
                * float(dt)
                * sigma_t.square().clamp_min(0.0)
            ).clamp(
                0.0, 0.25
            )  # (B,)
            uniform = torch.ones_like(q_end) / K
            uniform = self._aa_only_renormalize(uniform, eps=eps)
            q_end = (1.0 - mix.view(B, 1, 1)) * q_end + mix.view(B, 1, 1) * uniform
            q_end = self._aa_only_renormalize(q_end, eps=eps)

        # h_t(x) = sum_y P_{x,y}(1-t) q_end(y), computed in closed form for uniform substitution.
        delta_end = (1.0 - t).clamp_min(0.0).view(B, 1, 1)  # (B, 1, 1)
        p_stay_end = self._transition_prob(delta_end, same=True)  # (B, 1, 1)
        p_jump_end = self._transition_prob(delta_end, same=False)  # (B, 1, 1)
        h_all = p_jump_end + q_end * (p_stay_end - p_jump_end)  # (B, P, K)

        # h_t(i) for current token i.
        h_i = h_all.gather(-1, x_t.long().clamp(0, K - 1).unsqueeze(-1)).squeeze(-1)
        h_i = h_i.clamp_min(eps)  # (B, P)

        # Uniform substitution generator: Q(i->j)=beta/(K-1) for j!=i.
        q_rate = float(self.cfg.beta) / max(K - 1, 1)
        ratios = (h_all / h_i.unsqueeze(-1)).clamp_min(0.0).clamp_max(1e6)  # (B, P, K)

        current_onehot = F.one_hot(x_t.long().clamp(0, K - 1), num_classes=K).float()
        off_rates = (
            q_rate * ratios * (1.0 - current_onehot) * uncertainty.unsqueeze(-1)
        )  # (B, P, K)

        # First-order interval probabilities p(i->j) ~= rate * dt.
        step_off = (off_rates * float(dt)).clamp_min(0.0)  # (B, P, K)

        # Cap total leave mass for numerical stability.
        if self.cfg.leave_mass_cap is not None and float(self.cfg.leave_mass_cap) > 0.0:
            row_sum = step_off.sum(dim=-1, keepdim=True).clamp_min(eps)
            shrink = (float(self.cfg.leave_mass_cap) / row_sum).clamp_max(1.0)
            step_off = step_off * shrink

        row_sum = step_off.sum(dim=-1, keepdim=True).clamp_min(0.0)
        step_probs = step_off + current_onehot * (1.0 - row_sum)  # (B, P, K)
        step_probs = step_probs / step_probs.sum(dim=-1, keepdim=True).clamp_min(eps)

        # For invalid positions: stay put.
        stay_dist = current_onehot
        step_probs = torch.where(valid_mask.unsqueeze(-1), step_probs, stay_dist)

        # For valid positions, MASK is not a valid residue state (prevents "all-mask" degeneration).
        step_probs = torch.where(
            valid_mask.unsqueeze(-1),
            self._aa_only_renormalize(step_probs, eps=eps),
            step_probs,
        )

        # If a position is valid but currently MASK, force an immediate unmask to AA-only.
        is_mask = x_t == MASK_TOKEN_INDEX  # (B, P)
        force_unmask = valid_mask & is_mask  # (B, P)
        if force_unmask.any():
            q_end_aa = q_end  # already AA-only
            step_probs = torch.where(
                force_unmask.unsqueeze(-1),
                q_end_aa,
                step_probs,
            )
        return step_probs

    def euler_step(
        self,
        x_t: torch.Tensor,  # (B, P) current tokens
        x1_pred: torch.Tensor,  # (B, P, K) predicted logits for t=1
        t: torch.Tensor,  # (B,)
        dt: float,
        birth_time: torch.Tensor,  # (B, P)
        motif_mask: torch.Tensor,  # (B, P)
        potential: Optional[torch.Tensor] = None,  # (B, P, K) guidance logits
    ) -> torch.Tensor:
        """
        Single sampling step for discrete amino acids using a CTMC (uniform substitution).

        Training corruption for aatypes uses a CTMC bridge (see sample_bridge / _bridge_marginal). For inference,
        we don't have the true endpoint token X_1, so we form a "soft" bridge using the model's predicted endpoint
        distribution q_end (from logits), and sample the time-inhomogeneous conditioned CTMC:

            rate(i -> j | t) = Q(i -> j) * h_t(j) / h_t(i),
            h_t(x) = P(X_1 ~ q_end | X_t = x) = sum_y P_{x,y}(1-t) q_end(y),

        where Q is the uniform substitution generator with leaving rate beta.
        """
        B, P = x_t.shape
        K = self.K

        if x1_pred.shape != (B, P, K):
            raise ValueError(f"Expected x1_pred shape (B, P, K); got {x1_pred.shape}")

        assert potential is None, "potential not yet supported"

        # Valid positions are those born before (or at) current time.
        valid_mask = birth_time <= t[:, None]  # (B, P)
        step_probs = self._ctmc_step_probs(
            logits=x1_pred,
            x_t=x_t,
            t=t,
            dt=dt,
            valid_mask=valid_mask,
        )

        x_next = torch.multinomial(step_probs.view(-1, K), num_samples=1).squeeze(-1)
        x_next = x_next.view(B, P)

        # Keep motif positions fixed.
        x_next = torch.where(motif_mask, x_t, x_next)
        return x_next
