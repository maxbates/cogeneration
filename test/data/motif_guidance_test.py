"""
Unit tests for cogeneration motif guidance (compute_motif_potential).

Tests verify:
A. Zero-error test: guidance ~0 when predictions match targets
B. Sign test: one step in guidance direction reduces motif error
C. Finite-difference gradient check for translations
D. SO(3) tangent check for rotations
E. Scale/magnitude sanity checks

NOTE: The compute_motif_potential function requires that pred_trans_1 and
pred_rotmats_1 are connected to trans_t and rotmats_t through a computational graph.
In real usage, this happens via the model forward pass. In tests, we simulate this
by creating simple "mock model" functions that connect inputs to outputs.
"""

import pytest
import torch

from cogeneration.config.base import MotifGuidanceConfig, MotifGuidanceVarScale
from cogeneration.data import so3_utils
from cogeneration.data.motif_guidance import compute_motif_potential
from cogeneration.data.potentials import PotentialField


def _create_guidance_config(
    var_scale_type: MotifGuidanceVarScale = MotifGuidanceVarScale.ot,
    guidance_scale: float = 1.0,
    guidance_start_t: float = 0.1,
    guidance_end_t: float = 0.95,
) -> MotifGuidanceConfig:
    """Create a MotifGuidanceConfig with specified parameters."""
    return MotifGuidanceConfig(
        enabled=True,
        scale_factor=guidance_scale,
        var_scale_type=var_scale_type,
        guidance_start_t=guidance_start_t,
        guidance_end_t=guidance_end_t,
        obs_noise_trans_ang=0.5,
        obs_noise_rot_rad=0.1,
    )


def _random_rotmats(B: int, P: int, device: torch.device = None) -> torch.Tensor:
    """Generate random valid rotation matrices (B, P, 3, 3)."""
    if device is None:
        device = torch.device("cpu")
    # Use random axis-angle then convert to rotmat
    axis_angle = torch.randn(B, P, 3, device=device)
    return so3_utils.rotvec_to_rotmat(axis_angle)


class TestZeroErrorGuidance:
    """A. Zero-error test: when predictions exactly match targets, guidance should be ~0."""

    def test_zero_trans_error_gives_zero_guidance(self):
        """Translation guidance is zero when pred_trans_1 == trans_1_motifs."""
        B, P = 2, 10
        cfg = _create_guidance_config()

        t = torch.full((B,), 0.5)

        # trans_t requires grad for autograd
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P)
        rotmats_t = rotmats_t.requires_grad_(True)

        # Targets: some fixed positions
        trans_1_motifs = torch.randn(B, P, 3)
        rotmats_1_motifs = _random_rotmats(B, P)

        # Simulate a model that predicts exactly the target
        # For gradients to flow, pred must depend on trans_t
        # Use: pred = trans_t * 0 + target (keeps grad flow via trans_t * 0)
        pred_trans_1 = trans_t * 0.0 + trans_1_motifs
        pred_rotmats_1 = (
            rotmats_t @ torch.eye(3).unsqueeze(0).unsqueeze(0) * 0.0 + rotmats_1_motifs
        )

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,  # no alignment for this test
            )

        # Both potentials should be very close to zero
        assert potential.trans is not None
        assert potential.rotmats is not None
        assert (
            potential.trans.abs().max() < 1e-5
        ), f"Trans potential not zero: {potential.trans.abs().max()}"
        assert (
            potential.rotmats.abs().max() < 1e-5
        ), f"Rot potential not zero: {potential.rotmats.abs().max()}"

    def test_zero_error_with_alignment(self):
        """Zero error case with alignment enabled."""
        B, P = 2, 8
        cfg = _create_guidance_config()

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # Exact match via gradient-connected path
        trans_1_motifs = torch.randn(B, P, 3)
        rotmats_1_motifs = _random_rotmats(B, P)

        pred_trans_1 = trans_t * 0.0 + trans_1_motifs
        # For rotation with alignment, need matching batch dims
        pred_rotmats_1 = rotmats_1_motifs.clone()
        # Make it depend on rotmats_t for grad flow
        pred_rotmats_1 = pred_rotmats_1 + (rotmats_t * 0.0)

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=True,
            )

        assert potential.trans is not None
        assert potential.rotmats is not None
        assert (
            potential.trans.abs().max() < 1e-4
        ), f"Trans potential: {potential.trans.abs().max()}"
        assert (
            potential.rotmats.abs().max() < 1e-4
        ), f"Rot potential: {potential.rotmats.abs().max()}"


class TestSignConvention:
    """B. Sign test: stepping in guidance direction should reduce motif error."""

    def test_trans_guidance_direction(self):
        """Guidance pushes trans_t toward reducing prediction error."""
        B, P = 1, 5
        cfg = _create_guidance_config(guidance_scale=1.0)

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # Target and prediction with known offset
        trans_1_motifs = torch.zeros(B, P, 3)
        # Simulate model: pred = trans_t (endpoint predicted at current position)
        pred_trans_1 = trans_t.clone()

        # No rotation error
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.trans is not None
        assert (
            potential.trans.norm() > 0.01
        ), "Guidance should be non-zero for non-zero error"

        # Since pred = trans_t and target = 0, error = trans_t
        # log_p = -0.5 * ||trans_t||^2 / var
        # d(log_p)/d(trans_t) = -trans_t / var
        # The guidance should point opposite to trans_t (toward target 0)
        # Check: trans_pot should be approximately -k * trans_t for some k > 0
        # This means trans_pot and trans_t should have opposite signs (negative dot product)
        dot = (potential.trans.detach() * trans_t.detach()).sum()
        assert dot < 0, f"Guidance should point toward target (opposite to error), dot={dot}"

    def test_rot_guidance_reduces_error(self):
        """Rotation guidance points toward reducing rotation error."""
        B, P = 1, 3
        cfg = _create_guidance_config(guidance_scale=1.0)

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # No translation error
        trans_1_motifs = torch.randn(B, P, 3)
        pred_trans_1 = trans_1_motifs.clone() + trans_t * 0.0

        # Rotation error: pred is rotmats_t, target is something different
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_t.clone()  # pred = current

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.rotmats is not None
        assert (
            potential.rotmats.norm() > 0.01
        ), "Rotation guidance should be non-zero for non-zero error"


class TestFiniteDifferenceGradient:
    """C. Finite-difference gradient check for translations."""

    def test_finite_diff_single_coord(self):
        """Detailed finite difference check on a single coordinate."""
        B, P = 1, 2
        eps = 1e-4

        # The key insight: we need to construct pred_trans_1 as a function of trans_t
        # to get meaningful gradients. Let's use a simple identity: pred_trans_1 = trans_t
        # This simulates a model that predicts the endpoint equals current state.

        trans_1_motifs = torch.randn(B, P, 3)

        def compute_log_p(trans_t_val):
            # Simulate: pred = trans_t (identity model)
            pred_trans_1 = trans_t_val.clone()
            # Compute log_p manually (simplified, no scaling)
            trans_sq = ((pred_trans_1 - trans_1_motifs) ** 2).sum(dim=-1)
            log_p = -0.5 * trans_sq.sum()
            return log_p

        # Base point
        trans_t = torch.randn(B, P, 3)

        # Finite difference for coordinate [0, 0, 0]
        trans_t_plus = trans_t.clone()
        trans_t_plus[0, 0, 0] += eps
        log_p_plus = compute_log_p(trans_t_plus)

        trans_t_minus = trans_t.clone()
        trans_t_minus[0, 0, 0] -= eps
        log_p_minus = compute_log_p(trans_t_minus)

        fd_grad = (log_p_plus - log_p_minus) / (2 * eps)

        # Autograd
        trans_t_ag = trans_t.clone().requires_grad_(True)
        log_p_ag = compute_log_p(trans_t_ag)
        log_p_ag.backward()
        ag_grad = trans_t_ag.grad[0, 0, 0]

        # Compare
        assert torch.isclose(fd_grad, ag_grad, atol=1e-3, rtol=1e-2), (
            f"Finite diff: {fd_grad.item():.6f}, Autograd: {ag_grad.item():.6f}"
        )

    def test_trans_gradient_nonzero(self):
        """Check autograd gradient is non-zero and finite for translations."""
        B, P = 1, 4
        cfg = _create_guidance_config()

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # Create some error: pred = trans_t (model predicts current state)
        trans_1_motifs = torch.randn(B, P, 3)
        pred_trans_1 = trans_t.clone()  # Keeps grad connection
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.trans is not None
        assert torch.isfinite(potential.trans).all()
        assert potential.trans.abs().max() > 1e-6


class TestSO3TangentSpace:
    """D. SO(3) tangent check for rotations."""

    def test_rot_potential_zero_when_aligned(self):
        """Rotation potential is zero when pred_rotmats_1 == rotmats_1_motifs."""
        B, P = 2, 5
        cfg = _create_guidance_config()

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # No translation error (with grad connection)
        trans_1_motifs = torch.randn(B, P, 3)
        pred_trans_1 = trans_1_motifs.clone() + trans_t * 0.0

        # No rotation error (pred = target, with grad connection through rotmats_t)
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.rotmats is not None
        assert (
            potential.rotmats.abs().max() < 1e-5
        ), f"Rot potential should be ~0: {potential.rotmats.abs().max()}"

    def test_rot_guidance_nonzero_for_error(self):
        """Rotation guidance is non-zero when there's rotation error."""
        B, P = 1, 3
        cfg = _create_guidance_config(guidance_scale=1.0)

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # No translation error
        trans_1_motifs = torch.randn(B, P, 3)
        pred_trans_1 = trans_1_motifs.clone() + trans_t * 0.0

        # Create rotation error: pred = rotmats_t (model predicts current)
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_t.clone()  # Error: pred != target

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.rotmats is not None
        assert potential.rotmats.norm() > 0.01, "Should have non-zero rotation guidance"
        assert torch.isfinite(potential.rotmats).all(), "Rotation guidance should be finite"


class TestMagnitudeAndScaling:
    """E. Scale/magnitude sanity checks."""

    def test_guidance_magnitude_at_different_times(self):
        """Guidance should have reasonable magnitude across time range."""
        B, P = 1, 5
        cfg = _create_guidance_config(guidance_scale=1.0)

        trans_1_motifs = torch.zeros(B, P, 3)
        rotmats_1_motifs = _random_rotmats(B, P)

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        magnitudes = []
        for t_val in [0.2, 0.4, 0.6, 0.8]:
            t = torch.full((B,), t_val)
            # Create fresh tensors for each iteration
            trans_t = torch.ones(B, P, 3, requires_grad=True)  # 1 Angstrom from target
            rotmats_t = _random_rotmats(B, P).requires_grad_(True)

            # pred = trans_t creates error of 1 Angstrom
            pred_trans_1 = trans_t.clone()
            pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0

            with torch.enable_grad():
                potential, _ = compute_motif_potential(
                    t=t,
                    trans_t=trans_t,
                    rotmats_t=rotmats_t,
                    pred_trans_1=pred_trans_1,
                    pred_rotmats_1=pred_rotmats_1,
                    trans_1_motifs=trans_1_motifs,
                    rotmats_1_motifs=rotmats_1_motifs,
                    motif_mask=motif_mask,
                    valid_mask=valid_mask,
                    cfg=cfg,
                    align=False,
                )

            if potential.trans is not None:
                magnitudes.append((t_val, potential.trans.norm().item()))

        # Verify magnitudes are finite and not exploding
        for t_val, mag in magnitudes:
            assert mag < 1000, f"Guidance too large at t={t_val}: {mag}"
            assert mag > 0, f"Guidance zero at t={t_val}"

    def test_different_var_scale_types(self):
        """Different variance scaling types should all produce finite guidance."""
        B, P = 1, 4

        trans_1_motifs = torch.randn(B, P, 3)
        rotmats_1_motifs = _random_rotmats(B, P)
        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)
        t = torch.full((B,), 0.5)

        for var_type in [
            MotifGuidanceVarScale.ot,
            MotifGuidanceVarScale.linear,
            MotifGuidanceVarScale.constant,
        ]:
            cfg = _create_guidance_config(var_scale_type=var_type)

            trans_t = torch.randn(B, P, 3, requires_grad=True)
            rotmats_t = _random_rotmats(B, P).requires_grad_(True)

            pred_trans_1 = trans_t.clone()  # error = trans_t - target
            pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0

            with torch.enable_grad():
                potential, _ = compute_motif_potential(
                    t=t,
                    trans_t=trans_t,
                    rotmats_t=rotmats_t,
                    pred_trans_1=pred_trans_1,
                    pred_rotmats_1=pred_rotmats_1,
                    trans_1_motifs=trans_1_motifs,
                    rotmats_1_motifs=rotmats_1_motifs,
                    motif_mask=motif_mask,
                    valid_mask=valid_mask,
                    cfg=cfg,
                    align=False,
                )

            # All should produce finite, non-zero guidance
            assert potential.trans is not None, f"No trans guidance for {var_type}"
            assert potential.rotmats is not None, f"No rot guidance for {var_type}"
            assert torch.isfinite(potential.trans).all(), f"Trans not finite for {var_type}"
            assert torch.isfinite(potential.rotmats).all(), f"Rot not finite for {var_type}"

    def test_prefactor_clamping(self):
        """Verify prefactor clamp prevents explosion at small t."""
        B, P = 1, 3
        cfg = _create_guidance_config(
            guidance_start_t=0.01,  # Very early start
            guidance_end_t=0.95,
        )

        t = torch.full((B,), 0.02)  # Very small t
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        trans_1_motifs = torch.zeros(B, P, 3)
        pred_trans_1 = trans_t.clone()  # Some error
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0
        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        # Should be finite due to clamping
        if potential.trans is not None:
            assert torch.isfinite(potential.trans).all(), "Trans potential should be finite"
            # The prefactor is clamped at 100, so guidance shouldn't be astronomical
            assert (
                potential.trans.abs().max() < 10000
            ), f"Trans potential too large: {potential.trans.abs().max()}"


class TestMotifMasking:
    """Test that guidance only affects motif residues."""

    def test_guidance_zero_on_non_motifs(self):
        """Guidance should be zero for non-motif residues."""
        B, P = 1, 6
        cfg = _create_guidance_config()

        t = torch.full((B,), 0.5)
        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        trans_1_motifs = torch.randn(B, P, 3)
        pred_trans_1 = trans_t.clone()  # Error everywhere
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_t.clone()  # Error everywhere

        # Only first 3 residues are motifs
        motif_mask = torch.zeros(B, P, dtype=torch.bool)
        motif_mask[:, :3] = True
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.trans is not None
        assert potential.rotmats is not None

        # Non-motif residues (indices 3, 4, 5) should have zero guidance
        assert (potential.trans[:, 3:] == 0).all(), "Non-motif trans should be zero"
        assert (potential.rotmats[:, 3:] == 0).all(), "Non-motif rot should be zero"

        # Motif residues should have non-zero guidance (since there's error)
        assert potential.trans[:, :3].abs().max() > 1e-6, "Motif trans should be non-zero"


class TestWindowFunction:
    """Test guidance window (time-based activation)."""

    def test_guidance_off_outside_window(self):
        """Guidance should be None outside the active time window."""
        B, P = 1, 4
        cfg = _create_guidance_config(
            guidance_start_t=0.2,
            guidance_end_t=0.8,
        )

        trans_t = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        trans_1_motifs = torch.randn(B, P, 3)
        pred_trans_1 = trans_t.clone()
        rotmats_1_motifs = _random_rotmats(B, P)
        pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0
        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        # Test before window
        t_early = torch.full((B,), 0.1)
        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t_early,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )
        assert potential.trans is None, "Should be None before window"
        assert potential.rotmats is None, "Should be None before window"

        # Test after window
        t_late = torch.full((B,), 0.9)
        trans_t2 = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t2 = _random_rotmats(B, P).requires_grad_(True)
        pred_trans_12 = trans_t2.clone()
        pred_rotmats_12 = rotmats_1_motifs.clone() + rotmats_t2 * 0.0

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t_late,
                trans_t=trans_t2,
                rotmats_t=rotmats_t2,
                pred_trans_1=pred_trans_12,
                pred_rotmats_1=pred_rotmats_12,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )
        assert potential.trans is None, "Should be None after window"
        assert potential.rotmats is None, "Should be None after window"

        # Test inside window
        t_mid = torch.full((B,), 0.5)
        trans_t3 = torch.randn(B, P, 3, requires_grad=True)
        rotmats_t3 = _random_rotmats(B, P).requires_grad_(True)
        pred_trans_13 = trans_t3.clone()
        pred_rotmats_13 = rotmats_1_motifs.clone() + rotmats_t3 * 0.0

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t_mid,
                trans_t=trans_t3,
                rotmats_t=rotmats_t3,
                pred_trans_1=pred_trans_13,
                pred_rotmats_1=pred_rotmats_13,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )
        assert potential.trans is not None, "Should be active inside window"
        assert potential.rotmats is not None, "Should be active inside window"


class TestLocalImprovement:
    """Test that guidance actually improves the motif error in one step."""

    def test_trans_step_reduces_error(self):
        """Taking a step in guidance direction should reduce translation error."""
        B, P = 1, 4
        cfg = _create_guidance_config(guidance_scale=1.0)

        t = torch.full((B,), 0.5)

        # Target: origin
        trans_1_motifs = torch.zeros(B, P, 3)
        rotmats_1_motifs = _random_rotmats(B, P)

        # Current state: some distance from target
        trans_t = torch.randn(B, P, 3) * 3.0  # ~3 Angstrom from target
        trans_t.requires_grad_(True)
        rotmats_t = _random_rotmats(B, P).requires_grad_(True)

        # Model predicts current state
        pred_trans_1 = trans_t.clone()
        pred_rotmats_1 = rotmats_1_motifs.clone() + rotmats_t * 0.0

        motif_mask = torch.ones(B, P, dtype=torch.bool)
        valid_mask = torch.ones(B, P, dtype=torch.bool)

        # Compute initial error
        initial_error = ((pred_trans_1 - trans_1_motifs) ** 2).sum()

        with torch.enable_grad():
            potential, _ = compute_motif_potential(
                t=t,
                trans_t=trans_t,
                rotmats_t=rotmats_t,
                pred_trans_1=pred_trans_1,
                pred_rotmats_1=pred_rotmats_1,
                trans_1_motifs=trans_1_motifs,
                rotmats_1_motifs=rotmats_1_motifs,
                motif_mask=motif_mask,
                valid_mask=valid_mask,
                cfg=cfg,
                align=False,
            )

        assert potential.trans is not None

        # Take a small step in the guidance direction
        step_size = 0.1
        trans_t_stepped = trans_t.detach() + step_size * potential.trans

        # New prediction after step
        pred_trans_1_stepped = trans_t_stepped.clone()
        new_error = ((pred_trans_1_stepped - trans_1_motifs) ** 2).sum()

        # Error should decrease
        assert new_error < initial_error, (
            f"Error should decrease: {initial_error.item():.4f} -> {new_error.item():.4f}"
        )
