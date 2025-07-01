# tests/test_fk_steering.py
from typing import Tuple

import pytest
import torch

from cogeneration.config.base import InterpolantSteeringConfig
from cogeneration.data.potentials import (
    ChainBreakPotential,
    FKSteeringCalculator,
    FKSteeringResampler,
    HotSpotPotential,
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
                torch.arange(chain_1_start).float() * 3.8
            )  # Normal spacing
            # Chain 1 residues - start close to chain 0's first residue
            trans[:, chain_1_start:, 0] = (
                torch.arange(N - chain_1_start).float() * 3.8 + 5.0
            )  # Close contact
            trans[:, chain_1_start:, 1] = 2.0  # Slight y offset
        else:
            # Place hot spots far apart (beyond contact distance)
            trans = torch.zeros(B, N, 3)
            # Chain 0 residues
            trans[:, :chain_1_start, 0] = torch.arange(chain_1_start).float() * 3.8
            # Chain 1 residues - start far from chain 0
            trans[:, chain_1_start:, 0] = (
                torch.arange(N - chain_1_start).float() * 3.8 + 20.0
            )  # Far apart
            trans[:, chain_1_start:, 1] = 10.0  # Large y offset

        batch[nbp.trans_t] = trans
    else:
        # Set up translations for single chain with optional break
        # translations are lines on z axis, res are 3.8 angstroms apart:
        # if `with_break` second half shifted by +10
        line1 = torch.cat(
            [torch.zeros(N, 2), torch.arange(N).unsqueeze(1).float() * 3.8], dim=-1
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
        logits=None,
    )

    return batch, step


class TestChainBreakPotential:
    @pytest.mark.parametrize("with_break", [True, False])
    def test_chain_break_energy(self, with_break, mock_cfg):
        batch, step = _make_batch_and_step(
            B=2, N=30, multimer=False, with_break=with_break
        )
        potential = ChainBreakPotential()

        E = potential.compute_energy(
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

        E = potential.compute_energy(
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
        E = potential.compute_energy(
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
        E = potential.compute_energy(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        assert E.item() == 0.0, f"Expected zero energy for single chain, got {E.item()}"


class TestFKSteeringCalculator:
    def test_chain_break_energy(self, mock_cfg):
        batch, step = _make_batch_and_step(B=2, N=30, multimer=False, with_break=True)
        calc = FKSteeringCalculator(cfg=mock_cfg.inference.interpolant.steering)

        E = calc.compute_energy(
            batch=batch,
            model_pred=step,
            protein_pred=step,
            protein_state=step,
        )

        assert torch.all(E > 0)
        assert torch.all(E <= 1)


class TestFKSteeringResampler:
    def test_resample_collapses_particles(self):
        batch, step = _make_batch_and_step(B=2, N=30, multimer=False)

        orig_batch_size = batch[bp.res_mask].shape[0]
        assert orig_batch_size == 2

        cfg = InterpolantSteeringConfig(
            num_particles=3,
            resampling_interval=1,
            fk_lambda=1.0,
            energy_weight_difference=1.0,
            energy_weight_absolute=0.0,
        )
        resampler = FKSteeringResampler(cfg=cfg)
        assert resampler.enabled

        batch = resampler.init_particles(batch)

        # should have expanded to B * K
        steer_batch_size = orig_batch_size * cfg.num_particles
        assert batch[bp.res_mask].shape[0] == steer_batch_size

        # expand the step to match the batch size
        step = step.select_batch_idx(idx=torch.tensor([0, 0, 0, 1, 1, 1]))
        assert step.trans.shape[0] == steer_batch_size

        batch, idx, metric = resampler.on_step(
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
