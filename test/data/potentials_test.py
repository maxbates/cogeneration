# tests/test_fk_steering.py
from typing import Tuple

import pytest
import torch

from cogeneration.config.base import InterpolantSteeringConfig
from cogeneration.data.potentials import (
    ChainBreakPotential,
    FKSteeringCalculator,
    FKSteeringResampler,
)
from cogeneration.data.trajectory import SamplingStep
from cogeneration.dataset.test_utils import MockDataloader
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures, empty_feats
from cogeneration.type.task import DataTask


def _make_batch_and_step(
    B: int = 2, N: int = 30, with_break: bool = True
) -> Tuple[NoisyFeatures, SamplingStep]:
    """
    2 straight chains with 1 optional backbone break at residue 15.
    """
    dataloader = MockDataloader(
        sample_lengths=[N] * B,
        task=DataTask.inpainting,
        corrupt=True,
        multimer=False,
        batch_size=B,
    )
    batch = next(iter(dataloader))

    # translations are two lines on z axis, 3.5 angstroms apart:
    # if `with_break` second half shifted by +10
    line1 = torch.cat(
        [torch.zeros(N, 2), torch.arange(N).unsqueeze(1).float() * 3.5], dim=-1
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
        batch, step = _make_batch_and_step(B=2, N=30, with_break=with_break)
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


class TestFKSteeringCalculator:
    def test_chain_break_energy(self, mock_cfg):
        batch, step = _make_batch_and_step(B=2, N=30, with_break=True)
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
        batch, step = _make_batch_and_step(B=2, N=30)

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
