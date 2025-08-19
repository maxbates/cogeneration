# tests/test_fk_steering.py
from typing import Tuple

import pytest
import torch

from cogeneration.config.base import (
    InterpolantSteeringConfig,
    ModelESMKey,
    ProteinMPNNRunnerConfig,
)
from cogeneration.data.data_transforms import make_one_hot
from cogeneration.data.potentials import (
    ChainBreakPotential,
    ContactConditioningPotential,
    ESMLogitsPotential,
    FKSteeringCalculator,
    FKSteeringResampler,
    HotSpotPotential,
    InverseFoldPotential,
)
from cogeneration.data.trajectory import SamplingStep
from cogeneration.dataset.test_utils import MockDataloader
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures, empty_feats
from cogeneration.type.task import DataTask


def _make_batch_and_step(
    B: int = 2,
    N: int = 30,
    multimer: bool = False,
    with_break: bool = True,  # iff single chain
    hot_spots_in_contact: bool = True,  # iff multimer
) -> Tuple[NoisyFeatures, SamplingStep]:
    """
    Create test batch with either single chain (with optional backbone break) or multimer (with hot spots).

    For single chain (multimer=False):
    - Creates straight line on z axis, residues 3.8 angstroms apart
    - If with_break=True, second half shifted by +10 at residue 15

    For multimer (multimer=True):
    - Creates 2 chains with hot spots at first residue of each chain
    - N should be even (split between 2 chains)
    - If hot_spots_in_contact=True, chains are close; if False, far apart
    """
    assert N % 2 == 0, "N must be even"  # for multimer split
    assert N >= 20, "N must be at least 20"  # for chain breaks

    res_dist = 3.8

    dataloader = MockDataloader(
        sample_lengths=[N] * B,
        task=DataTask.inpainting,
        corrupt=True,
        multimer=multimer,
        batch_size=B,
    )
    batch = next(iter(dataloader))

    if multimer:
        # Find where second chain starts (first position where chain_idx changes from 1 to 2)
        chain_idx = batch[bp.chain_idx]  # (B, N)
        chain_1_start = None
        for i in range(N):
            if chain_idx[0, i] == 2:  # Chain 2 (1-indexed)
                chain_1_start = i
                break
        assert chain_1_start is not None, "Could not find chain 1 start position"

        # Set up hot spots: first residue of each chain
        hot_spots = torch.zeros(B, N)
        hot_spots[:, 0] = 1  # First residue of chain 0
        hot_spots[:, chain_1_start] = 1  # First residue of chain 1
        batch[bp.hot_spots] = hot_spots

        # Set up translations for multimer
        if hot_spots_in_contact:
            # Place hot spots close together (within contact distance)
            trans = torch.zeros(B, N, 3)
            # Chain 0 residues
            trans[:, :chain_1_start, 0] = (
                torch.arange(chain_1_start).float() * res_dist
            )  # Normal spacing
            # Chain 1 residues - start close to chain 0's first residue
            trans[:, chain_1_start:, 0] = (
                torch.arange(N - chain_1_start).float() * res_dist + 5.0
            )  # Close contact
            trans[:, chain_1_start:, 1] = 2.0  # Slight y offset
        else:
            # Place hot spots far apart (beyond contact distance)
            trans = torch.zeros(B, N, 3)
            # Chain 0 residues
            trans[:, :chain_1_start, 0] = torch.arange(chain_1_start).float() * res_dist
            # Chain 1 residues - start far from chain 0
            trans[:, chain_1_start:, 0] = (
                torch.arange(N - chain_1_start).float() * res_dist + 20.0
            )  # Far apart
            trans[:, chain_1_start:, 1] = 10.0  # Large y offset

        batch[nbp.trans_t] = trans
    else:
        # Set up translations for single chain with optional break
        # translations are lines on z axis, res are 3.8 angstroms apart:
        # if `with_break` second half shifted by +10
        line1 = torch.cat(
            [torch.zeros(N, 2), torch.arange(N).unsqueeze(1).float() * res_dist], dim=-1
        )  # (N, 3)
        trans = line1.clone()
        if with_break:
            trans[15:, 2] += 10.0  # break
        batch[nbp.trans_t] = trans.unsqueeze(0).repeat(B, 1, 1)  # (B, N, 3)

    assert batch[nbp.trans_t].shape == (B, N, 3)

    step = SamplingStep(
        res_mask=batch[bp.res_mask],
        trans=batch[nbp.trans_t],
        rotmats=batch[nbp.rotmats_t],
        aatypes=batch[nbp.aatypes_t],
        torsions=None,
        logits=make_one_hot(batch[nbp.aatypes_t].long(), 21),
    )

    return batch, step


class TestChainBreakPotential:
    @pytest.mark.parametrize("with_break", [True, False])
    def test_chain_break_energy(self, with_break, mock_cfg):
        batch, step = _make_batch_and_step(
            B=2, N=30, multimer=False, with_break=with_break
        )
        potential = ChainBreakPotential()

        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        if with_break:
            # expect 1 break
            assert torch.allclose(E, 1.0 - torch.exp(torch.tensor(-1.0)))
        else:
            # no breaks, energy should be 0
            assert torch.allclose(E, torch.tensor(0.0))


class TestHotSpotPotential:
    @pytest.mark.parametrize("hot_spots_in_contact", [True, False])
    def test_hot_spot_energy(self, hot_spots_in_contact):
        batch, step = _make_batch_and_step(
            B=1, N=30, multimer=True, hot_spots_in_contact=hot_spots_in_contact
        )
        potential = HotSpotPotential()

        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        if hot_spots_in_contact:
            # Hot spots are in contact, energy should be low (close to 0)
            assert E.item() < 0.01, f"Expected low energy for contacts, got {E.item()}"
        else:
            # Hot spots are far apart, energy should be high (> 0)
            assert (
                E.item() > 0.5
            ), f"Expected high energy for distant hot spots, got {E.item()}"

    def test_no_hot_spots(self):
        """Test that potential returns zero energy when no hot spots are defined."""
        batch, step = _make_batch_and_step(
            B=1, N=30, multimer=True, hot_spots_in_contact=True
        )

        # Remove hot spots
        batch[bp.hot_spots] = torch.zeros_like(batch[bp.hot_spots])

        potential = HotSpotPotential()
        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        assert (
            E.item() == 0.0
        ), f"Expected zero energy with no hot spots, got {E.item()}"

    def test_single_chain(self):
        """Test that potential returns zero energy for single chain structures."""
        batch, step = _make_batch_and_step(
            B=1, N=30, multimer=False, with_break=False
        )  # Single chain

        # Add hot spots to single chain, should be ignored
        hot_spots = torch.zeros(1, 8, dtype=torch.bool)
        hot_spots[:, 0] = True
        hot_spots[:, 3] = True
        batch[bp.hot_spots] = hot_spots

        potential = HotSpotPotential()
        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        assert E.item() == 0.0, f"Expected zero energy for single chain, got {E.item()}"


class TestContactConditioningPotential:
    def _make_simple_batch_and_step(self, B: int = 1, N: int = 6):
        # Reuse helper but trim to a small N to isolate pair interactions
        batch, step = _make_batch_and_step(
            B=B, N=max(20, N), multimer=False, with_break=False
        )
        for k in [bp.res_mask, bp.diffuse_mask, bp.chain_idx, bp.res_idx]:
            batch[k] = batch[k][:, :N]
        # Remove motif mask to avoid interfering with guidance tests
        if bp.motif_mask in batch:
            del batch[bp.motif_mask]
        step = SamplingStep(
            res_mask=batch[bp.res_mask],
            trans=step.trans[:, :N, :],
            rotmats=step.rotmats[:, :N, :, :],
            aatypes=step.aatypes[:, :N],
            torsions=None,
            logits=make_one_hot(step.aatypes[:, :N].long(), 21),
        )
        return batch, step

    def test_no_contacts_defined(self):
        """Test that potential returns zero energy when no contacts are defined."""
        batch, step = _make_batch_and_step(B=1, N=30, multimer=False)

        # No contact conditioning defined (default is zeros)
        batch[bp.contact_conditioning] = torch.zeros((1, 30, 30))

        potential = ContactConditioningPotential()
        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        assert E.item() == 0.0, f"Expected zero energy with no contacts, got {E.item()}"

    def test_contacts_satisfied(self):
        """Test that potential returns low energy when contacts are satisfied."""
        batch, step = _make_batch_and_step(B=1, N=30, multimer=False)

        # Define contact conditioning matrix with tight constraints that should be satisfied
        # (since structure is linear with 3.8 spacing)
        contact_conditioning = torch.zeros(1, 30, 30)
        contact_conditioning[0, 0, 1] = 3.8  # expect ~3.8A contact
        contact_conditioning[0, 1, 2] = 3.8  # expect ~3.8A contact
        batch[bp.contact_conditioning] = contact_conditioning

        potential = ContactConditioningPotential()
        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        # Should be very low energy since contacts are satisfied
        assert (
            E.item() < 0.1
        ), f"Expected low energy with satisfied contacts, got {E.item()}"

    @pytest.mark.parametrize("dist", [2.0, 3.8, 10.0])
    def test_contacts_violated(self, dist):
        """Test that potential returns high energy when contacts are violated."""
        batch, step = _make_batch_and_step(B=1, N=30, multimer=False)

        # force no motif mask, so all residues considered for loss
        if bp.motif_mask in batch:
            del batch[bp.motif_mask]

        # Define contact conditioning matrix with tight constraints that will be violated
        contact_conditioning = torch.zeros(1, 30, 30)
        contact_conditioning[0, 0, 1] = dist  # expect dist but actual is ~3.8A
        contact_conditioning[0, 1, 0] = dist  # make symmetric
        contact_conditioning[0, 1, 2] = dist  # expect dist but actual is ~3.8A
        contact_conditioning[0, 2, 1] = dist  # make symmetric
        batch[bp.contact_conditioning] = contact_conditioning

        potential = ContactConditioningPotential()
        E, _ = potential.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        # Should be high energy since contacts are violated
        if dist == 3.8:
            assert (
                E.item() == 0.0
            ), f"Expected zero energy with violated contacts, got {E.item()}"
        else:
            assert (
                E.item() > 0.05
            ), f"Expected high energy with violated contacts, got {E.item()} @ dist={dist}"

    def test_inter_chain_weighting(self):
        """Test that inter-chain contacts are upweighted."""
        batch, step = _make_batch_and_step(B=1, N=30, multimer=True)

        # Remove motif mask to avoid interference
        if bp.motif_mask in batch:
            del batch[bp.motif_mask]

        # Find where chain 2 starts
        chain_idx = batch[bp.chain_idx]  # (B, N)
        chain_2_start = None
        for i in range(30):
            if chain_idx[0, i] == 2:  # Chain 2 (1-indexed)
                chain_2_start = i
                break
        assert chain_2_start is not None, "Could not find chain 2 start position"

        # Define contact conditioning matrix with inter-chain contact
        contact_conditioning = torch.zeros(1, 30, 30)

        # Use residue 0 (chain 1) and chain_2_start (chain 2) for inter-chain contact
        actual_dist = torch.norm(
            step.trans[0, 0, :] - step.trans[0, chain_2_start, :]
        ).item()
        target_dist = actual_dist - 2.0  # Set target to be 2.0A less than actual

        # Residue 0 (chain 1) to chain_2_start (chain 2) - should be upweighted
        contact_conditioning[0, 0, chain_2_start] = (
            target_dist  # moderate constraint that will be violated
        )
        contact_conditioning[0, chain_2_start, 0] = target_dist  # make matrix symmetric
        batch[bp.contact_conditioning] = contact_conditioning

        # Test with default inter-chain weight
        potential_weighted = ContactConditioningPotential(inter_chain_weight=2.0)
        E_weighted, _ = potential_weighted.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        # Test with no inter-chain weight
        potential_unweighted = ContactConditioningPotential(inter_chain_weight=1.0)
        E_unweighted, _ = potential_unweighted.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        # Both should have non-zero energy, and weighted should be higher
        assert (
            E_unweighted.item() > 0
        ), f"Expected non-zero energy for unweighted, got {E_unweighted.item()}"
        assert (
            E_weighted.item() > E_unweighted.item()
        ), f"Expected higher energy with inter-chain weighting: {E_weighted.item()} vs {E_unweighted.item()}"

    def test_guidance_pull_toward_target_when_too_far(self):
        """If current distance > target + tol, guidance should pull residues together (decrease distance)."""
        B, N = 1, 6
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        # Actual ~3.8A; set target much shorter to trigger "too_far"
        target = 2.0
        cc = torch.zeros(B, N, N)
        cc[0, 0, 1] = target
        cc[0, 1, 0] = target
        batch[bp.contact_conditioning] = cc

        potential = ContactConditioningPotential(guidance_scale=1.0)
        _, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert guidance is not None and guidance.trans is not None
        v01 = step.trans[0, 1] - step.trans[0, 0]
        v10 = step.trans[0, 0] - step.trans[0, 1]
        # Move residue 0 toward residue 1
        assert (guidance.trans[0, 0] * v01).sum().item() > 0
        # Move residue 1 toward residue 0
        assert (guidance.trans[0, 1] * v10).sum().item() > 0

    def test_guidance_push_away_when_too_close(self):
        """If current distance < target - tol, guidance should push residues apart (increase distance)."""
        B, N = 1, 6
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        # Use a spaced-out pair (0 and 2) with actual ~7.6A; set target slightly longer
        i, j = 0, 2
        actual = torch.norm(step.trans[0, j] - step.trans[0, i]).item()
        target = actual + 1.0  # > 3.8 and > actual + tolerance
        cc = torch.zeros(B, N, N)
        cc[0, i, j] = target
        cc[0, j, i] = target
        batch[bp.contact_conditioning] = cc

        potential = ContactConditioningPotential(guidance_scale=1.0)
        _, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert guidance is not None and guidance.trans is not None
        v_ij = step.trans[0, j] - step.trans[0, i]
        v_ji = step.trans[0, i] - step.trans[0, j]
        # Move residue 0 away from residue 1
        assert (guidance.trans[0, i] * v_ij).sum().item() < 0
        # Move residue 1 away from residue 0
        assert (guidance.trans[0, j] * v_ji).sum().item() < 0

    def test_guidance_zero_on_motif_pairs(self):
        """When both residues are motifs, guidance should not be applied to them (scaffold-only)."""
        B, N = 1, 6
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        # Define a contact between residues 0 and 1
        target = 2.0
        cc = torch.zeros(B, N, N)
        cc[0, 0, 1] = target
        cc[0, 1, 0] = target
        batch[bp.contact_conditioning] = cc

        # Mark both residues as motifs
        motif_mask = torch.zeros(B, N)
        motif_mask[:, 0] = 1
        motif_mask[:, 1] = 1
        batch[bp.motif_mask] = motif_mask

        potential = ContactConditioningPotential(guidance_scale=1.0)
        _, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert guidance is not None and guidance.trans is not None
        assert torch.all(guidance.trans[0, 0] == 0)
        assert torch.all(guidance.trans[0, 1] == 0)


class DummyMPNNResult:
    def __init__(self, logits: torch.Tensor):
        self.averaged_logits = logits


class DummyProteinMPNNRunner:
    """
    Dummy runner that returns provided target logits.
    Expects logits shaped (B, N, 21).
    """

    def __init__(self, logits: torch.Tensor):
        self._logits = logits

    def run_batch(
        self,
        trans,
        rotmats,
        aatypes,
        res_mask,
        diffuse_mask,
        chain_idx,
        torsions=None,
        num_passes: int = 1,
        sequences_per_pass: int = 1,
        **kwargs,
    ):
        return DummyMPNNResult(self._logits)


class TestInverseFoldPotential:
    def _make_simple_batch_and_step(self, B: int = 1, N: int = 8):
        # Reuse helper but override aatypes for determinism
        batch, step = _make_batch_and_step(
            B=B, N=max(20, N), multimer=False, with_break=False
        )
        # Trim to N if needed (helper enforces N>=20); keep masks aligned
        for k in [bp.res_mask, bp.diffuse_mask, bp.chain_idx, bp.res_idx]:
            batch[k] = batch[k][:, :N]
        aatypes = torch.randint(0, 20, (B, N))
        step = SamplingStep(
            res_mask=batch[bp.res_mask],
            trans=step.trans[:, :N, :],
            rotmats=step.rotmats[:, :N, :, :],
            aatypes=aatypes,
            torsions=None,
            logits=make_one_hot(aatypes.long(), 21),
        )
        return batch, step

    def test_low_energy_when_logits_match_targets(self):
        B, N = 2, 10
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        # Make all residues valid/designed
        batch[bp.res_mask] = torch.ones(B, N, dtype=torch.int)
        batch[bp.diffuse_mask] = torch.ones(B, N, dtype=torch.int)

        # Build logits favoring the correct class moderately
        logits = torch.zeros(B, N, 21, device=step.aatypes.device)
        for b in range(B):
            for i in range(N):
                tgt = int(step.aatypes[b, i].item())
                logits[b, i, tgt] = 4.0

        potential = InverseFoldPotential(
            energy_scale=1.0,
            protein_mpnn_cfg=ProteinMPNNRunnerConfig(),
        )
        potential.protein_mpnn_runner = DummyProteinMPNNRunner(logits)

        E, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert E.shape == (B,)
        # Energy should be low but not ~0 with moderate confidence logits
        assert torch.all(E > 0.01)
        assert torch.all(E < 0.2)
        # Guidance should be returned with correct shape and match provided logits (no temp/scale effects)
        assert guidance is not None and guidance.logits is not None
        assert guidance.logits.shape == (B, N, 21)
        assert torch.allclose(guidance.logits, logits)

    def test_masks_are_respected(self):
        B, N = 1, 12
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        batch[bp.res_mask] = torch.ones(B, N, dtype=torch.int)
        batch[bp.diffuse_mask] = torch.ones(B, N, dtype=torch.int)
        batch[bp.diffuse_mask][:, ::2] = 0

        # Build logits: correct on designed positions, wrong elsewhere
        logits = torch.zeros(B, N, 21, device=step.aatypes.device)
        for i in range(N):
            tgt = int(step.aatypes[0, i].item())
            if batch[bp.diffuse_mask][0, i] > 0:
                logits[0, i, tgt] = 20.0
            else:
                logits[0, i, (tgt + 1) % 20] = 20.0

        potential = InverseFoldPotential(
            energy_scale=1.0,
            protein_mpnn_cfg=ProteinMPNNRunnerConfig(),
        )
        potential.protein_mpnn_runner = DummyProteinMPNNRunner(logits)

        E, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert E.shape == (B,)
        assert E.item() < 1e-3
        # Guidance present and correctly shaped (masking is applied later by calculator)
        assert guidance is not None and guidance.logits is not None
        assert guidance.logits.shape == (B, N, 21)

    def test_high_energy_when_logits_wrong(self):
        B, N = 1, 10
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        batch[bp.res_mask] = torch.ones(B, N, dtype=torch.int)
        batch[bp.diffuse_mask] = torch.ones(B, N, dtype=torch.int)

        # Build logits with high confidence on wrong class
        logits = torch.zeros(B, N, 21, device=step.aatypes.device)
        for i in range(N):
            tgt = int(step.aatypes[0, i].item())
            logits[0, i, (tgt + 1) % 20] = 10.0

        potential = InverseFoldPotential(
            energy_scale=1.0,
            protein_mpnn_cfg=ProteinMPNNRunnerConfig(),
        )
        potential.protein_mpnn_runner = DummyProteinMPNNRunner(logits)

        E, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert E.shape == (B,)
        # Wrong-high logits should yield very large normalized energy
        assert E.item() > 1.0
        # Guidance should be capped to 4.0 uniformly per-position
        assert guidance is not None and guidance.logits is not None
        assert guidance.logits.shape == (B, N, 21)
        max_abs = guidance.logits.abs().amax(dim=-1)
        assert torch.allclose(max_abs, torch.full_like(max_abs, 4.0))
        # Since only the wrong class had magnitude, it should be exactly 4.0 after capping; others 0
        for i in range(N):
            tgt = int(step.aatypes[0, i].item())
            wrong = (tgt + 1) % 20
            assert torch.allclose(
                guidance.logits[0, i, wrong],
                torch.tensor(4.0, device=guidance.logits.device),
            )
            # all other 20 classes (including mask class) should be 0
            mask = torch.ones(21, dtype=torch.bool, device=guidance.logits.device)
            mask[wrong] = False
            assert torch.all(guidance.logits[0, i][mask] == 0)

    def test_guidance_logit_padding_and_shape(self):
        B, N = 1, 7
        batch, step = self._make_simple_batch_and_step(B=B, N=N)

        # Make all residues valid/designed
        batch[bp.res_mask] = torch.ones(B, N, dtype=torch.int)
        batch[bp.diffuse_mask] = torch.ones(B, N, dtype=torch.int)

        # Dummy MPNN returns 20-way logits; potential should pad to 21 to match model_pred
        logits20 = torch.zeros(B, N, 20, device=step.aatypes.device)
        logits20[:, :, 3] = 1.5

        potential = InverseFoldPotential(
            energy_scale=1.0,
            protein_mpnn_cfg=ProteinMPNNRunnerConfig(),
            guidance_scale=1.0,
            inverse_fold_logits_temperature=1.0,
            inverse_fold_logits_cap=100.0,
        )
        potential.protein_mpnn_runner = DummyProteinMPNNRunner(logits20)

        E, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert guidance is not None
        assert guidance.logits is not None
        assert guidance.logits.shape == step.logits.shape == (B, N, 21)
        # Expect padding in the last class and other classes unchanged
        assert torch.allclose(guidance.logits[:, :, :20], logits20)
        assert torch.all(guidance.logits[:, :, 20] == 0)


class TestESMLogitsPotential:
    def test_returns_logits(self):
        B, N = 2, 20
        batch, step = _make_batch_and_step(B=B, N=N, multimer=False, with_break=False)

        potential = ESMLogitsPotential(
            esm_model_key=ModelESMKey.DUMMY,
            guidance_scale=1.0,
            esm_logits_temperature=1.0,
            esm_logits_cap=100.0,
        )

        E, guidance = potential.compute(
            batch=batch, model_pred=step, protein_pred=step, protein_state=step
        )

        assert E.shape == (B,)
        assert torch.all(E >= 0.0) and torch.all(E <= 1.0)

        assert guidance is not None and guidance.logits is not None
        assert guidance.logits.shape == (B, N, 21)


class TestFKSteeringCalculator:
    def test_default_compute(self, mock_cfg):
        batch, step = _make_batch_and_step(B=2, N=30, multimer=False, with_break=True)
        calc = FKSteeringCalculator(cfg=mock_cfg.inference.interpolant.steering)

        E, guidance = calc.compute(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        assert torch.all(E > 0)
        assert guidance is not None


class TestFKSteeringResampler:
    @pytest.mark.slow
    def test_resample_collapses_particles(self, mock_cfg):
        batch, step = _make_batch_and_step(B=2, N=30, multimer=False)

        orig_batch_size = batch[bp.res_mask].shape[0]
        assert orig_batch_size == 2

        steering_cfg = mock_cfg.inference.interpolant.steering
        steering_cfg.num_particles = 3
        steering_cfg.resampling_interval = 1
        steering_cfg.fk_lambda = 1.0
        steering_cfg.energy_weight_difference = 1.0
        steering_cfg.energy_weight_absolute = 0.0

        resampler = FKSteeringResampler(cfg=steering_cfg)
        assert resampler.enabled

        batch = resampler.init_particles(batch)

        # should have expanded to B * K
        steer_batch_size = orig_batch_size * steering_cfg.num_particles
        assert batch[bp.res_mask].shape[0] == steer_batch_size

        # expand the step to match the batch size
        step = step.select_batch_idx(idx=torch.tensor([0, 0, 0, 1, 1, 1]))
        assert step.trans.shape[0] == steer_batch_size

        batch, idx, metric, guidance = resampler.on_step(
            step_idx=0,
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )
        assert idx is not None
        assert batch[bp.res_mask].shape[0] == steer_batch_size

        # collapse particles
        batch, best_idx = resampler.best_particle_in_batch(batch)
        assert best_idx is not None
        assert best_idx.shape[0] == orig_batch_size
        assert batch[bp.res_mask].shape[0] == orig_batch_size
