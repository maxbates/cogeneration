import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # noqa
from torch import nn
from tqdm.auto import tqdm

from cogeneration.config.base import (
    InterpolantAATypesInterpolantTypeEnum,
    InterpolantAATypesScheduleEnum,
    InterpolantConfig,
    InterpolantRotationsScheduleEnum,
    InterpolantTrainTimeSamplingEnum,
    InterpolantTranslationsNoiseTypeEnum,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX, NM_TO_ANG_SCALE
from cogeneration.data.logits import combine_logits
from cogeneration.data.noise_mask import (
    angles_noise,
    centered_gaussian,
    centered_harmonic,
    fill_torsions,
    mask_blend_1d,
    mask_blend_2d,
    mask_blend_3d,
    masked_categorical,
    torsions_empty,
    torsions_noise,
    uniform_categorical,
    uniform_so3,
)
from cogeneration.data.potentials import (
    FKSteeringResampler,
    FKSteeringTrajectory,
    FKStepMetric,
    PotentialField,
)
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass
from cogeneration.data.trajectory import SamplingStep, SamplingTrajectory
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import ModelPrediction
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProp as pbp
from cogeneration.type.structure import StructureExperimentalMethod
from cogeneration.type.task import DataTask, InferenceTask


@dataclass
class BatchTrueFeatures:
    """
    Struct for time=1 (true/target distribution) features of a batch.
    Used during sampling for conditional generation (e.g. if a domain is fixed, motifs in inpainting)
    """

    trans: torch.Tensor  # (B, N, 3)
    rotmats: torch.Tensor  # (B, N, 3, 3)
    torsions: torch.Tensor  # (B, N, 7, 2)
    aatypes: torch.Tensor  # (B, N)
    logits: torch.Tensor  # (B, N, S) where S=21 if masking else S=20

    @classmethod
    def from_optional(
        cls,
        res_mask: torch.Tensor,
        num_tokens: int,
        trans: Optional[torch.Tensor] = None,
        rotmats: Optional[torch.Tensor] = None,
        torsions: Optional[torch.Tensor] = None,
        aatypes: Optional[torch.Tensor] = None,
    ):
        num_batch, num_res = res_mask.shape

        if trans is None:
            trans = torch.zeros(num_batch, num_res, 3, device=res_mask.device)
        if rotmats is None:
            rotmats = torch.eye(3, device=res_mask.device)[None, None].repeat(
                num_batch, num_res, 1, 1
            )
        if torsions is None:
            torsions = torsions_empty(num_batch, num_res, device=res_mask.device)
        if aatypes is None:
            aatypes = torch.zeros(num_batch, num_res, device=res_mask.device).long()

        logits_1 = torch.nn.functional.one_hot(aatypes, num_classes=num_tokens).float()

        return cls(
            trans=trans,
            rotmats=rotmats,
            torsions=torsions,
            aatypes=aatypes,
            logits=logits_1,
        )


class Interpolant:
    """
    Interpolant is responsible for generating noise, corrupting samples, and sampling from learned vector fields.

    It has two almost-but-not-quite separate roles:
    (1) corrupt batches with noise, generating intermediate samples at some time `t`
    (2) generates samples, interpolating each modality using the learned vector fields over t=[0, 1] from noise to sample.

    Works across multiple domains: translations and rotations (i.e. backbone frames), torsion angles, and amino acid types (i.e. sequence).

    (1) Translations
    - in R^3 and are Euclidean
    - Noise is Gaussian, or uses harmonic prior
    - Supports minibatch optimal transport (OT)
    - Supports stochastic paths
    - Supports Euler sampling (standard flow matching, i.e. Euclidean)
    (2) Rotations
    - in SO(3) and are Riemannian
    - Initial random rotation matrices are generated from uniform SO(3) distribution
    - Noise is IGSO(3)
    - Supports stochastic paths
    - Supports Euler sampling using geodesic (Riemannian flow matching)
    (3) Torsion angles
    - Optional, and only sort of. Not a model input, only an output.
    - Noise sampled from VonMises distribution
    - Interpolate in angle space
    - Supports stochastic paths
    - Supports Euler sampling
    (4) Amino acid types
    - are discrete
    - over either uniform distribution (n=20) or with masks (n=21)
    - supports Euler steps using learned rate matrix (discrete flow matching)
    - supports "purity" sampling, i.e. masking by max log probs and re-masking noise
    - supports stochastic paths using CTMC jump

    (Frames a.k.a. Rigids are therefore in SE(3) = R^3 x SO(3))
    torsion angles are not really considered by the interpolant, except when output by the model to generate Rigids.
    """

    def __init__(self, cfg: InterpolantConfig):
        self.cfg = cfg

        # TODO - consider building igso3 on instantiation, to simplify device management
        #   (will increase instantiation time slightly)
        self._igso3 = None
        self._device = None

        is_masking_interpolant = (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        )
        self.num_tokens = 21 if is_masking_interpolant else 20

    @property
    def igso3(self):
        if self._igso3 is None:
            # Build in float64 for accuracy at small sigma, then cast for runtime.
            steps = 1000
            sigma_grid = torch.logspace(
                math.log10(self.cfg.rots.igso3_sigma_min),
                math.log10(self.cfg.rots.igso3_sigma),
                steps=steps,
                dtype=torch.float64,
            )
            self._igso3 = so3_utils.SampleIGSO3(steps, sigma_grid, cache_dir=".cache")

            # on CUDA, move to GPU, so can be broadcasted. on MPS, leave on CPU.
            if self._device is not None and self._device.type != "mps":
                self._igso3.to(self._device)

            # store tables in fp32
            self._igso3 = self._igso3.float()

        return self._igso3

    def set_device(self, device):
        self._device = device

        # on CUDA, move to GPU, so can be broadcasted. on MPS, leave on CPU.
        if not device.type == "mps" and self._igso3 is not None:
            self._igso3.to(device)

    def sample_t(self, num_batch):
        """
        Take a random t in the range [min_t, 1-min_t], for corrupting / training a batch.
        """
        if (
            self.cfg.train_time_sampling_method
            == InterpolantTrainTimeSamplingEnum.uniform
        ):
            # sample from uniform [0, 1]
            t = torch.rand(num_batch, device=self._device)
        elif (
            self.cfg.train_time_sampling_method
            == InterpolantTrainTimeSamplingEnum.late_biased
        ):
            # late-biased is composed of three different distributions; we sample one to sample from.
            #    |
            #    |                ________
            #    |           ___/│       │
            #    |       ___/    │       │
            #    |   ___/ ramp   │ late  │
            #    |__/____________│_______│
            #    |          uniform      │
            #    +---------------+-------+-> t
            #    0             t_late    1
            t_late = 0.80

            # pre-sample all three schedules
            # 0 = uniform
            t0 = torch.rand(num_batch, device=self._device)
            # 1 = ramp
            t1 = torch.sqrt(torch.rand(num_batch, device=self._device)) * t_late
            # 2 = late uniform
            t2 = t_late + torch.rand(num_batch, device=self._device) * (1 - t_late)

            # pick one of {0=uniform,1=ramp,2=late} per sample
            choice = torch.tensor(
                np.random.choice([0, 1, 2], size=num_batch, p=[0.5, 0.3, 0.2]),
                device=self._device,
            )
            t = torch.where(choice == 0, t0, torch.where(choice == 1, t1, t2)).to(
                self._device
            )
        else:
            raise ValueError(
                f"Unknown train time sampling method: {self.cfg.train_time_sampling_method}"
            )

        # scale to [min_t, 1-min_t]
        min_t = self.cfg.min_t
        return t * (1 - 2 * min_t) + min_t

    def _compute_sigma_t(self, t: torch.Tensor, scale: float = 1.0, min_sigma: float = 0.0):
        """
        Compute the instantaneous standard deviation of the noise at time `t` (B,)
        The standard deviation is a sqrt parabolic function of t, and is zero at t=0 and t=1.
        """
        return torch.sqrt(scale**2 * t * (1 - t) + min_sigma**2)  # (B,)

    def _trans_noise(
        self, chain_idx: torch.Tensor, is_intermediate: bool
    ) -> torch.Tensor:
        """
        sample t=0 noise for translations, scaled to angstroms
        """
        # Determine noise type, depending on whether sampling from t=0 base distribution
        # or adding noise to intermediate sample
        if is_intermediate:
            noise_type = self.cfg.trans.intermediate_noise_type
        else:
            noise_type = self.cfg.trans.initial_noise_type

        if noise_type == InterpolantTranslationsNoiseTypeEnum.centered_gaussian:
            return (
                centered_gaussian(
                    *chain_idx.shape,
                    device=self._device,
                )
                * NM_TO_ANG_SCALE
            )
        elif noise_type == InterpolantTranslationsNoiseTypeEnum.centered_harmonic:
            # For harmonic noise, each Brownian increment has covariance J^{-1}
            # (the harmonic prior) instead of identity.
            return (
                centered_harmonic(
                    chain_idx=chain_idx,
                    device=self._device,
                )
                * NM_TO_ANG_SCALE
            )
        else:
            raise ValueError(f"Unknown translation noise type {noise_type}")

    def _rotmats_noise(self, res_mask: torch.Tensor):
        """
        Generate t=0 SO(3) noise rotation matrices
        """
        num_batch, num_res = res_mask.shape
        return uniform_so3(num_batch, num_res, device=self._device)

    def _torsions_noise(self, res_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate t=0 torsion angles noise
        """
        num_batch, num_res = res_mask.shape
        return torsions_noise(
            sigma=torch.ones((num_batch,), device=self._device),
            num_samples=num_res,
            num_angles=7,
        )

    def _aatypes_noise(self, res_mask: torch.Tensor) -> torch.Tensor:
        """
        Generate t=0 amino acid types based on the interpolant type
        """
        num_batch, num_res = res_mask.shape

        if (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            # all masks
            return masked_categorical(num_batch, num_res, device=self._device)
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            # random AA
            return uniform_categorical(
                num_batch, num_res, num_tokens=self.num_tokens, device=self._device
            )
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

    def _batch_ot(self, trans_0, trans_1, res_mask, center: bool = False):
        """
        Compute optimal transport between two batches of translations.
        returns OT mapping of trans_0 structures to trans_1 structures.
        Will force translations are centered if `center==True`.
        Does not re-order the translations within a structure.

        For multimers, ignore chain_idx for now.
        If we use a harmonic prior, with a gaussian per chain, probably the samples will match back up.
        If we use a gaussian prior... just assume it will work out for now at least.
        """
        num_batch, num_res = trans_0.shape[:2]

        # note the indexing here `(num_batch, num_batch)` creates all pairs for calculating minibatch OT cost matrix
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_0 = trans_0[noise_idx]
        batch_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]

        # center and align the structures within each pairing
        batch_0, batch_1, _ = batch_align_structures(
            batch_0, batch_1, mask=batch_mask, center=center
        )
        batch_0 = batch_0.reshape(num_batch, num_batch, num_res, 3)  # (B, B, N, 3)
        batch_1 = batch_1.reshape(num_batch, num_batch, num_res, 3)  # (B, B, N, 3)
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)  # (B, B, N)

        # Compute cost matrix between all pairings
        distances = torch.linalg.norm(batch_0 - batch_1, dim=-1)  # (B, B, N)
        cost_matrix = torch.sum(distances, dim=-1) / torch.sum(batch_mask, dim=-1)

        # Find optimal matching between noisy and ground truth structures
        # Return the reordered noisy structures -> ground truth pairing
        noise_perm, gt_perm = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return batch_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_trans(
        self,
        trans_1,
        t,
        res_mask,
        diffuse_mask,
        chain_idx,
        stochasticity_scale: float = 1.0,
    ):
        """
        Corrupt translations from t=1 to t using Gaussian noise.
        """
        trans_0 = self._trans_noise(
            chain_idx=chain_idx, is_intermediate=False
        )  # (B, N, 3)

        # compute batch OT, or aligning, to enable learning straighter paths.
        # Expect no need to re-center as noise and t=1 should already be.
        # Also, we center below after adding noise / motif-masking.
        if self.cfg.trans.batch_ot:
            trans_0 = self._batch_ot(
                trans_0,
                trans_1,
                res_mask=res_mask,
                center=False,
            )
        elif self.cfg.trans.batch_align:
            trans_0, _, _ = batch_align_structures(
                pos_1=trans_0,
                pos_2=trans_1,
                mask=res_mask,
                center=False,
            )

        # compute trans_t
        if self.cfg.trans.train_schedule == InterpolantTranslationsScheduleEnum.linear:
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(f"Unknown trans schedule {self.cfg.trans.train_schedule}")

        # stochastic paths
        if (
            self.cfg.trans.stochastic
            and self.cfg.trans.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # guassian noise added is markovian; just sample from gaussian, scaled by sigma_t, and add.
            # sigma_t is ~parabolic (and ~0 at t=0 and t=1) so corrupted sample reflects marginal distribution at t.
            sigma_t = self._compute_sigma_t(
                t.squeeze(1),  # (B,)
                scale=self.cfg.trans.stochastic_noise_intensity * stochasticity_scale,
            )
            intermediate_noise = self._trans_noise(
                chain_idx=chain_idx, is_intermediate=True
            )  # (B, N, 3)
            intermediate_noise = intermediate_noise * sigma_t[..., None, None]
            trans_t += intermediate_noise

        # Fix non-diffused residues to t=1
        trans_t = mask_blend_2d(trans_t, trans_1, diffuse_mask)

        # Center residues at origin
        trans_t -= batch_center_of_mass(trans_t, mask=res_mask)[:, None]

        return trans_t * res_mask[..., None]

    def _corrupt_rotmats(
        self, rotmats_1, t, res_mask, diffuse_mask, stochasticity_scale: float = 1.0
    ):
        """
        Corrupt rotations from t=1 to t using IGSO3.
        """
        num_batch, num_res = res_mask.shape

        # sample IGSO(3)
        sigma = torch.tensor(
            [self.cfg.rots.igso3_sigma], device=self.igso3.sigma_grid.device
        )
        noisy_rotmats = self.igso3.sample(sigma, num_batch * num_res).to(self._device)
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)

        # multiple rotations by noise to get t=0 noisy rotations
        # applying noise as composition ensures noise is relative to reference frame + stay on SO(3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        so3_schedule = self.cfg.rots.train_schedule
        if so3_schedule == InterpolantRotationsScheduleEnum.linear:
            so3_t = t
        elif so3_schedule == InterpolantRotationsScheduleEnum.exp:
            # Normalize schedule anchored at t=0 -> 0 and t=1 -> 1
            rate = self.cfg.rots.exp_rate
            so3_t = (1 - torch.exp(-t * rate)) / (1 - math.exp(-rate))
        else:
            raise ValueError(f"Invalid schedule: {so3_schedule}")

        # interpolate on geodesic between rotmats_0 and rotmats_1 to get rotmats_t
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)

        # stochastic paths
        if (
            self.cfg.rots.stochastic
            and self.cfg.rots.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # gaussian noise added is markovian; we are sampling intermediate point directly from marginal
            # so we just need to compute sigma_t for variance of IGSO(3) noise.
            # compute noise std deviation (mean is just rotmats_t)
            sigma_t = self._compute_sigma_t(
                so3_t.squeeze(1),  # (B,)
                scale=self.cfg.rots.stochastic_noise_intensity * stochasticity_scale,
            )

            # sample IGSO(3) noise
            sigma_t = sigma_t.to(self.igso3.sigma_grid.device)
            intermediate_noise = self.igso3.sample(sigma_t, num_res).to(self._device)
            intermediate_noise = intermediate_noise.reshape(num_batch, num_res, 3, 3)

            rotmats_t = torch.einsum(
                "...ij,...jk->...ik", rotmats_t, intermediate_noise
            )

        # set residues not in `res_mask` to identity
        identity = torch.eye(3, device=self._device)
        rotmats_t = mask_blend_3d(rotmats_t, identity[None, None], res_mask)

        # only corrupt residues in `diffuse_mask`
        return mask_blend_3d(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_torsions(
        self, torsions_1, t, res_mask, diffuse_mask, stochasticity_scale: float = 1.0
    ):
        """
        Corrupt torsions from t=1 to t using noise.
        """
        num_batch, num_res = res_mask.shape
        torsions_0 = self._torsions_noise(res_mask=res_mask)

        # interpolate in angle space using linear schedule
        angles_1 = torch.atan2(torsions_1[..., 0], torsions_1[..., 1])  # (B, N, 7)
        angles_0 = torch.atan2(torsions_0[..., 0], torsions_0[..., 1])  # (B, N, 7)
        t_broadcast = t.view(num_batch, 1, 1)  # (B, 1, 1)
        angles_t = (1.0 - t_broadcast) * angles_0 + t_broadcast * angles_1

        if (
            self.cfg.torsions.stochastic
            and self.cfg.torsions.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            sigma_t = self._compute_sigma_t(
                t.squeeze(1),  # (B,)
                scale=self.cfg.torsions.stochastic_noise_intensity
                * stochasticity_scale,
            )
            angles_t += angles_noise(sigma=sigma_t, num_samples=num_res, num_angles=7)

        # wrap to keep angles in (-π,π]
        angles_t = (angles_t + math.pi) % (2.0 * math.pi) - math.pi

        # angles -> (sin, cos)
        torsions_t = torch.stack((torch.sin(angles_t), torch.cos(angles_t)), dim=-1)

        # Fix non-diffused residues to t=1
        torsions_t = mask_blend_3d(torsions_t, torsions_1, diffuse_mask)

        return torsions_t * res_mask[..., None, None]

    def _corrupt_aatypes_old(
        self, aatypes_1, t, res_mask, diffuse_mask, stochasticity_scale: float = 1.0
    ):
        """
        Corrupt AA residues from t=1 to t, using masking or uniform sampling.

        If `self.cfg.aatypes.stochastic` is True, use a uniform CTMC forward process to swap some residues.
        (Uniform because we can't use logits to sample more likely residues in a meaningful way without simulation).
        """
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        # aatypes_t = aatypes_1 with masked fraction of residues based on t
        u = torch.rand(num_batch, num_res, device=self._device)
        corruption_mask = (u < (1 - t)).int()
        aatypes_noise = self._aatypes_noise(res_mask=res_mask)
        aatypes_t = mask_blend_1d(aatypes_noise, aatypes_1, corruption_mask)

        if (
            self.cfg.aatypes.stochastic
            and self.cfg.aatypes.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            sigma_t = self._compute_sigma_t(
                t.squeeze(1),  # (B,)
                scale=self.cfg.aatypes.stochastic_noise_intensity * stochasticity_scale,
            ).unsqueeze(
                1
            )  # (B, 1)

            # probability a residue jumps
            p_jump = sigma_t.clamp(max=1.0)  # (B, 1)
            jump_mask = torch.rand(num_batch, num_res, device=self._device) < p_jump

            # TODO - include non-mask AA in "noise" for stochasticity
            if jump_mask.any():
                jump_noise = self._aatypes_noise(res_mask=res_mask)
                aatypes_t = mask_blend_1d(jump_noise, aatypes_t, jump_mask)

        # residues outside `res_mask` are set to mask regardless of aatype noise strategy.
        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        # only corrupt residues in `diffuse_mask`
        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _corrupt_aatypes(
        self,
        aatypes_1: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # (B, 1)
        res_mask: torch.Tensor,  # (B, N)
        diffuse_mask: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ):
        """
        Corrupt AA residues from t=1 to t, using masking or uniform sampling.

        If `self.cfg.aatypes.stochastic` is True, we also corrupt with unmasks/remasks/changes.
        AA -> AA' vs mask (change vs remask) proportion uses fixed split.
        """
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        # aatypes_t = aatypes_1 with masked fraction of residues based on t
        _, tau = self._aatypes_schedule(t=t)
        u = torch.rand(num_batch, num_res, device=self._device)
        corruption_mask = (u < (1.0 - tau)).int()
        aatypes_noise = self._aatypes_noise(res_mask=res_mask)
        aatypes_t = mask_blend_1d(aatypes_noise, aatypes_1, corruption_mask)

        if (
            self.cfg.aatypes.stochastic
            and self.cfg.aatypes.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # For stochasticity, instead of specifying rates and using CTMC jump,
            # we simply introduce changes, dependent on t and interpolant type.

            sigma_t = self._compute_sigma_t(
                t.squeeze(1),  # (B,)
                scale=self.cfg.aatypes.stochastic_noise_intensity * stochasticity_scale,
            )

            # probability a residue jumps
            p_jump = (
                sigma_t.unsqueeze(1).expand(num_batch, num_res).clamp(max=1.0)
            )  # (B, N)
            jump_mask = torch.rand(num_batch, num_res, device=self._device) < p_jump

            if jump_mask.any():
                # For uniform interpolant, convert AA -> AA' (not allowing self)
                if (
                    self.cfg.aatypes.interpolant_type
                    == InterpolantAATypesInterpolantTypeEnum.uniform
                ):
                    assert self.num_tokens == 20
                    K = jump_mask.sum().item()
                    # disallow self and normalize probs
                    probs = torch.ones(K, self.num_tokens, device=self._device)
                    probs.scatter_(1, aatypes_t.long()[jump_mask].view(-1, 1), 0.0)
                    probs = probs / probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
                    # pick and set new aatypes
                    new_aatypes = torch.multinomial(probs, num_samples=1).squeeze(-1)
                    aatypes_t[jump_mask] = new_aatypes.to(aatypes_t.dtype)

                # For masking interpolant, additional unmask / remask / change, weighted by component scales
                elif (
                    self.cfg.aatypes.interpolant_type
                    == InterpolantAATypesInterpolantTypeEnum.masking
                ):
                    # unmask unneeded - any mask that is jumping will unmask
                    change_scale, unmask_scale, remask_scale = (
                        self._aatypes_component_scales(t=t.squeeze(1))
                    )
                    change_scale = change_scale.view(num_batch)
                    remask_scale = remask_scale.view(num_batch)

                    is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
                    is_aa = ~is_mask
                    aa_col_mask = torch.ones(
                        self.num_tokens, dtype=torch.bool, device=self._device
                    )
                    aa_col_mask[MASK_TOKEN_INDEX] = False

                    # mask -> AA uniformly
                    select_mask = is_mask & jump_mask
                    if select_mask.any():
                        K = select_mask.sum().item()
                        probs = torch.zeros(K, self.num_tokens, device=self._device)
                        probs[:, aa_col_mask] = 1.0 / aa_col_mask.sum()  # 1/20
                        new_states = torch.multinomial(probs, num_samples=1).squeeze(-1)
                        aatypes_t[select_mask] = new_states.to(aatypes_t.dtype)

                    # AA -> AA' or mask (depending on interpolant type)
                    select_aa = is_aa & jump_mask
                    if select_aa.any():
                        # relative rates of change vs remasking are independent of drift ratio
                        # the propertion could be a function of t, but it can also just be a constant
                        prop_change = 0.5
                        select_change = (
                            torch.rand(num_batch, num_res, device=self._device)
                            < prop_change
                        ) & select_aa
                        select_remask = select_aa & (~select_change)

                        # AA->AA' uniformly (exclude self and MASK)
                        if select_change.any():
                            K = select_change.sum().item()
                            probs = torch.ones(K, self.num_tokens, device=self._device)
                            probs.scatter_(
                                1, aatypes_t.long()[select_change].view(-1, 1), 0.0
                            )
                            probs[:, MASK_TOKEN_INDEX] = 0.0
                            probs = probs / probs.sum(dim=-1, keepdim=True).clamp_min(
                                1e-8
                            )
                            new_states = torch.multinomial(
                                probs, num_samples=1
                            ).squeeze(-1)
                            aatypes_t[select_change] = new_states.to(aatypes_t.dtype)

                        # AA->MASK (remask)
                        if select_remask.any():
                            aatypes_t[select_remask] = torch.as_tensor(
                                MASK_TOKEN_INDEX,
                                device=self._device,
                                dtype=aatypes_t.dtype,
                            )

                else:
                    raise ValueError(
                        f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
                    )

        # residues outside `res_mask` are set to mask regardless of aatype noise/interpolant strategy.
        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        # only corrupt residues in `diffuse_mask`
        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _set_corruption_times(
        self,
        noisy_batch: NoisyFeatures,
        task: DataTask,
    ):
        res_mask = noisy_batch[bp.res_mask]  # (B, N)
        num_batch, num_res = res_mask.shape

        if self.cfg.codesign_separate_t:
            u = torch.rand((num_batch,), device=self._device)

            # Sample t values for normal structure and categorical corruption.
            # In `codesign_separate_t` each has its own `t`, separate from fixed domains set to t=1.
            normal_structure_t = self.sample_t(num_batch)
            normal_cat_t = self.sample_t(num_batch)
            ones_t = torch.ones((num_batch,), device=self._device)

            # determine multi-task allocations. Each DataTask supports different multi-tasks.
            if task == DataTask.inpainting:
                # In the case of inpainting, there are 2 ways to allocate sub-tasks:
                #  1) drop motif mask -> MultiFlow style training
                #     i.e. unconditional, forward folding, inverse folding without motifs
                #  2) proportion of sub-tasks -> motif-style forward folding / inverse folding
                #     i.e. conditional sequence infilling / folding
                #
                # For unconditional, we don't attempt to construct some notion with motifs, and always drop them.
                #
                # But notice that forward folding and inverse folding can be run both with motifs or without.
                # Despite some of the sequence being fixed, we still set intermediate `t` without
                # taking the motif sequence into account. The fixed domain is set to t=1.
                #
                # |       task      |     motifs    |    no motifs    |
                # |-----------------|---------------|-----------------|
                # |    [codesign]   | [inpainting]  | [unconditional] |
                # |                 |    (1-unc)    |       unc       |
                # |-----------------|---------------|-----------------|
                # | forward_folding | fwd * (1-unc) |    fwd * unc    |
                # | inverse_folding | inv * (1-unc) |    inv * unc    |
                # |-----------------|---------------|-----------------|

                assert (
                    bp.motif_mask in noisy_batch
                    and noisy_batch[bp.motif_mask] is not None
                ), "Motif mask is required for inpainting task, but not found in batch."

                # forward_folding and inverse_folding set fixed domain to t=1
                # and use sampled t for the modeled domains, regardless of motif % defined.
                fwd = self.cfg.codesign_forward_fold_prop
                inv = self.cfg.codesign_inverse_fold_prop
                forward_fold_mask = (u < fwd).float()
                inverse_fold_mask = ((u >= fwd) & (u < fwd + inv)).float()

                # some portion of examples are made unconditional, and these examples may overlap with
                # forward folding and inverse folding masks above so they lack motifs.
                v = torch.rand((num_batch,), device=self._device)
                unc = self.cfg.inpainting_unconditional_prop
                unconditional_mask = (v < unc).bool()

                # for `inpainting -> unconditional` examples, reset `motif_mask` and `diffuse_mask`
                noisy_batch[bp.diffuse_mask] = torch.where(
                    unconditional_mask[:, None],
                    torch.ones_like(noisy_batch[bp.diffuse_mask]),
                    noisy_batch[bp.diffuse_mask],
                ).float()
                noisy_batch[bp.motif_mask] = torch.where(
                    unconditional_mask[:, None],
                    torch.zeros_like(noisy_batch[bp.motif_mask]),
                    noisy_batch[bp.motif_mask],
                ).float()
                # Re-center only the unconditional structures, in case they were centered using the motifs.
                unconditional_com = batch_center_of_mass(
                    pos=noisy_batch[bp.trans_1][unconditional_mask],
                    mask=noisy_batch[bp.res_mask][unconditional_mask],
                )[:, None]
                noisy_batch[bp.trans_1][unconditional_mask] -= unconditional_com

            elif task == DataTask.hallucination:
                # proportions: forward, inverse, rest = normal unconditional
                fwd = self.cfg.codesign_forward_fold_prop
                inv = self.cfg.codesign_inverse_fold_prop
                forward_fold_mask = (u < fwd).float()
                inverse_fold_mask = ((u >= fwd) & (u < fwd + inv)).float()

            else:
                raise ValueError(f"Unknown task {task}")

            # If we are forward folding, then cat_t should be 1 (i.e. data, sequence)
            # If we are inverse folding or codesign then cat_t should be uniform
            cat_t = forward_fold_mask * ones_t + (1 - forward_fold_mask) * normal_cat_t

            # If we are inverse folding, then structure_t should be 1 (i.e. data, structure)
            # If we are forward folding or codesign then structure_t should be uniform
            structure_t = (
                inverse_fold_mask * ones_t
                + (1 - inverse_fold_mask) * normal_structure_t
            )

            so3_t = structure_t[:, None]
            r3_t = structure_t[:, None]
            cat_t = cat_t[:, None]

        # Default: single `t` time is shared by each domain
        # Note: for inpainting, it could make sense to pick an intermediate t depending on how many residues are defined
        else:
            t = self.sample_t(num_batch)[:, None]
            so3_t = t
            r3_t = t
            cat_t = t

        noisy_batch[nbp.so3_t] = so3_t  # [B, 1]
        noisy_batch[nbp.r3_t] = r3_t  # [B, 1]
        noisy_batch[nbp.cat_t] = cat_t  # [B, 1]

    def corrupt_batch(
        self,
        batch: BatchFeatures,
        task: DataTask,
    ) -> NoisyFeatures:
        """
        Corrupt `t=1` data into a noisy batch at sampled `t`.

        Supports within-batch multi-task learning if `codesign_separate_t == True`.
        This allows a wider set of tasks during inference.

        Some examples in the batch are assigned t=1 values (sequence for forward_folding, structure for inverse_folding),
        or modify the `diffuse_mask` (inpainting -> unconditional).

        Fixing at t=1 has the effect of not corrupting the domain.
        """
        noisy_batch: NoisyFeatures = copy.deepcopy(batch)

        # set t values for each domain, i.e. `bp.so3_t`, `bp.r3_t`, `bp.cat_t`, and potentially modify masks.
        self._set_corruption_times(
            noisy_batch=noisy_batch,
            task=task,
        )

        trans_1 = batch[bp.trans_1]  # (B, N, 3) in Angstroms
        rotmats_1 = batch[bp.rotmats_1]  # (B, N, 3, 3)
        torsions_1 = batch[bp.torsions_1]  # (B, N, 7, 2)
        aatypes_1 = batch[bp.aatypes_1]  # (B, N)
        res_mask = batch[bp.res_mask]  # (B, N)
        diffuse_mask = batch[bp.diffuse_mask]  # (B, N)
        motif_mask = batch.get(bp.motif_mask, None)  # (B, N)
        chain_idx = noisy_batch[bp.chain_idx]  # (B, N)
        r3_t = noisy_batch[nbp.r3_t]  # (B, 1)
        so3_t = noisy_batch[nbp.so3_t]  # (B, 1)
        cat_t = noisy_batch[nbp.cat_t]  # (B, 1)

        # Determine sequence and structure corruption masks.
        # Inpainting:
        #   The motif sequence is fixed. However, the structure is not, and t should be in sync across domains.
        #   With guidance: The motifs are explicitly interpolated over time, so we corrupt the entire structure.
        #   Fixed motifs: The diffuse_mask is only for the scaffolds and the motifs are fixed at t=1
        # For other tasks, everything is corrupted i.e. `(diffuse_mask == 1.0).all()`
        #   Though values at t=1 effectively won't be corrupted.
        scaffold_mask = (1 - motif_mask) if motif_mask is not None else diffuse_mask

        # Stochastic dropout, if enabled, disables stochasticity for some proportion of batches.
        # It is at the batch level, rather than sample level, largely out of expedience
        # (sigma_t == 0 does not imply no noise is added), but could be changed to sample level.
        if (
            self.cfg.stochastic_dropout_prop > 0.0
            and torch.rand(1).item() < self.cfg.stochastic_dropout_prop
        ):
            stochasticity_scale = 0.0
        else:
            stochasticity_scale = 1.0

        # Apply corruptions

        if self.cfg.trans.corrupt:
            trans_t = self._corrupt_trans(
                trans_1,
                t=r3_t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                chain_idx=chain_idx,
                stochasticity_scale=stochasticity_scale,
            )
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        noisy_batch[nbp.trans_t] = trans_t

        if self.cfg.rots.corrupt:
            rotmats_t = self._corrupt_rotmats(
                rotmats_1,
                t=so3_t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                stochasticity_scale=stochasticity_scale,
            )
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch[nbp.rotmats_t] = rotmats_t

        if self.cfg.torsions.corrupt:
            torsions_t = self._corrupt_torsions(
                torsions_1,
                t=r3_t,
                res_mask=res_mask,
                diffuse_mask=diffuse_mask,
                stochasticity_scale=stochasticity_scale,
            )
        else:
            torsions_t = torsions_1
        if torch.any(torch.isnan(torsions_t)):
            raise ValueError("NaN in torsions_t during corruption")
        noisy_batch[nbp.torsions_t] = torsions_t

        if self.cfg.aatypes.corrupt:
            aatypes_t = self._corrupt_aatypes(
                aatypes_1,
                t=cat_t,
                res_mask=res_mask,
                diffuse_mask=scaffold_mask,
                stochasticity_scale=stochasticity_scale,
            )
        else:
            aatypes_t = aatypes_1
        noisy_batch[nbp.aatypes_t] = aatypes_t

        # zeroed out self-conditioned values. may be defined by module.
        noisy_batch[nbp.trans_sc] = torch.zeros_like(trans_1)
        noisy_batch[nbp.aatypes_sc] = torch.zeros_like(aatypes_1)[..., None].repeat(
            1, 1, self.num_tokens
        )

        return noisy_batch

    def _trans_vector_field(
        self,
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        trans_1: torch.Tensor,  # (B, N, 3)
        trans_t: torch.Tensor,  # (B, N, 3)
    ):
        if self.cfg.trans.sample_schedule == InterpolantTranslationsScheduleEnum.linear:
            trans_vf = (trans_1 - trans_t) / (1 - t)
        elif (
            self.cfg.trans.sample_schedule == InterpolantTranslationsScheduleEnum.vpsde
        ):
            bmin = self.cfg.trans.vpsde_bmin
            bmax = self.cfg.trans.vpsde_bmax
            bt = bmin + (bmax - bmin) * (1 - t)  # scalar
            alpha_t = torch.exp(
                -bmin * (1 - t) - 0.5 * (1 - t) ** 2 * (bmax - bmin)
            )  # scalar
            trans_vf = 0.5 * bt * trans_t + 0.5 * bt * (
                torch.sqrt(alpha_t) * trans_1 - trans_t
            ) / (1 - alpha_t)
        else:
            raise ValueError(
                f"Invalid sample schedule: {self.cfg.trans.sample_schedule}"
            )
        return trans_vf

    def _trans_euler_step(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        trans_1: torch.Tensor,  # (B, N, 3)
        trans_t: torch.Tensor,  # (B, N, 3)
        chain_idx: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, 3) vector field
    ) -> torch.Tensor:
        # unconditional vector field
        trans_vf = self._trans_vector_field(t=t, trans_1=trans_1, trans_t=trans_t)

        # noise for stochastic paths
        if (
            self.cfg.trans.stochastic
            and self.cfg.trans.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # Sample from intermediate noise type (either Gaussian or Harmonic prior)
            intermediate_noise = self._trans_noise(
                chain_idx=chain_idx,
                is_intermediate=True,
            )  # (B, N, 3) in Angstroms
            sigma_t = self._compute_sigma_t(
                torch.ones(trans_1.shape[0], device=self._device)
                * t.to(self._device),  # (B)
                scale=self.cfg.trans.stochastic_noise_intensity * stochasticity_scale,
            )
            sigma_t = sigma_t.to(trans_t.device)
            intermediate_noise = (
                intermediate_noise * math.sqrt(d_t) * sigma_t[..., None, None]
            )
        else:
            intermediate_noise = torch.zeros_like(trans_t)

        # potential, if provided, modifies the vector field
        if potential is not None:
            assert (
                potential.shape == trans_vf.shape
            ), f"potential {potential.shape} != trans_vf {trans_vf.shape}"
            trans_vf += potential

        trans_next = trans_t + trans_vf * d_t + intermediate_noise

        return trans_next

    def _rots_vf_scaling(self, t: torch.Tensor):
        """Calculate rotmats scaling factor of `d_t` step given `t`, depending on the schedule."""
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            return (1 / (1 - t)).clamp(min=1e-4)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            # Ensure tensor math even if t is a float
            t = (
                t.to(self._device)
                if torch.is_tensor(t)
                else torch.tensor(
                    t, dtype=torch.get_default_dtype(), device=self._device
                )
            )

            # exp_rate is expected > 0
            # Exact complementary scaling for exp τ(t):
            # τ(t) = (1 - exp(-r t)) / (1 - exp(-r))
            # so s(t) = (dτ/dt) / (1 - τ)
            # => s(t) = r / (1 - exp(-r * (1 - t)))
            r = torch.tensor(
                self.cfg.rots.exp_rate, dtype=torch.get_default_dtype(), device=t.device
            )

            # 1 - e^{-r(1-t)}, avoid div-by-zero at t≈1, match linear clamp min
            denom = 1.0 - torch.exp(-r * (1.0 - t))
            denom = torch.clamp(denom, min=1e-8)
            scale = r / denom
            return scale.clamp(min=1e-4)
        else:
            raise ValueError(f"Unknown sample schedule {self.cfg.rots.sample_schedule}")

    def _rots_euler_step(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        rotmats_1: torch.Tensor,  # (B, N, 3, 3)
        rotmats_t: torch.Tensor,  # (B, N, 3, 3)
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, 3) tangent to rotmats_t
    ) -> torch.Tensor:
        scaling = self._rots_vf_scaling(t)

        rot_vf = so3_utils.calc_rot_vf(mat_t=rotmats_t, mat_1=rotmats_1)

        if potential is not None:
            assert (
                potential.shape == rot_vf.shape
            ), f"potential {potential.shape} != rot_vf {rot_vf.shape}"
            rot_vf += potential

        rotmats_next = so3_utils.geodesic_t(
            t=scaling * d_t,  # scaled time along geodesic
            mat=rotmats_1,
            base_mat=rotmats_t,
            rot_vf=rot_vf,
        )

        if (
            self.cfg.rots.stochastic
            and self.cfg.rots.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # Brownian increment: sigma_t * sqrt(dt), with sigma_t = intensity * sqrt(t(1–t))
            # Sample IGSO(3) noise with a time-independent sigma_t, scaled by sqrt(dt)
            # Add IGSO(3) noise to stay on SO(3).
            num_batch, num_res, _, _ = rotmats_t.shape

            sigma_t = self._compute_sigma_t(
                torch.ones(num_batch, device=rotmats_t.device)
                * t.to(rotmats_t.device),  # (B)
                scale=self.cfg.rots.stochastic_noise_intensity * stochasticity_scale,
            ) * math.sqrt(
                d_t
            )  # (B,)
            sigma_t = sigma_t.to(self.igso3.sigma_grid.device)

            # check sigma_t within IGSO3 grid spec
            if sigma_t.min() < self.igso3.sigma_grid.min():
                print(
                    f"WARNING: rots sigma_t < igso3 grid min, noise will be larger than desired. Lower igso3_sigma_min."
                )

            intermediate_noise = self.igso3.sample(sigma_t, num_res)
            intermediate_noise = intermediate_noise.to(rotmats_t.device)
            intermediate_noise = intermediate_noise.reshape(num_batch, num_res, 3, 3)
            rotmats_next = torch.einsum(
                "...ij,...jk->...ik", rotmats_next, intermediate_noise
            )

        return rotmats_next

    def _torsions_euler_step(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        torsions_1: torch.Tensor,  # (B, N, 7, 2)
        torsions_t: torch.Tensor,  # (B, N, K, 2)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:  # (B, N, 7, 2)
        """
        Perform an Euler step in angle space to update torsion angles.

        `K` can be 1 or 7, depending on the number of torsions predicted by the model.
        However, torsions_1 is always 7, and torsions_t is filled to 7 if K < 7.
        """
        B, N, K = torsions_1.shape[:3]

        # (B, N, K, 2) -> (B, N, 7, 2)
        torsions_t = fill_torsions(
            shape=torsions_1.shape,
            torsions=torsions_t,
            device=torsions_1.device,
        )

        angles_1 = torch.atan2(torsions_1[..., 0], torsions_1[..., 1])  # (B, N, 7)
        angles_t = torch.atan2(torsions_t[..., 0], torsions_t[..., 1])  # (B, N, 7)

        t = t.to(self._device)
        angles_vf = (angles_1 - angles_t) / (1 - t)
        angles_next = angles_t + angles_vf * d_t

        if (
            self.cfg.torsions.stochastic
            and self.cfg.torsions.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            sigma_t = self._compute_sigma_t(
                torch.full((B,), t, device=angles_next.device),
                scale=self.cfg.torsions.stochastic_noise_intensity
                * stochasticity_scale,
            ) * math.sqrt(d_t)
            # add sampled noisy angles
            angles_next += angles_noise(sigma=sigma_t, num_samples=N, num_angles=K)

        # wrap back into (-π, π]
        angles_next = (angles_next + np.pi) % (2 * np.pi) - np.pi

        return torch.stack(
            [angles_next.sin(), angles_next.cos()], dim=-1
        )  # (B, N, 7, 2)

    def _regularize_step_probs(
        self,
        probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:  # (B, N, S)
        """
        Regularize the step probabilities to build a per-step probability row, where rows sum to 1,
        to be used for euler discrete sampling.
        """
        num_batch, num_res, S = probs.shape
        device = probs.device
        assert aatypes_t.shape == (num_batch, num_res)

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        probs = torch.clamp(probs, min=0.0, max=1.0)

        batch_idx = torch.arange(num_batch, device=device).repeat_interleave(num_res)
        residue_idx = torch.arange(num_res, device=device).repeat(num_batch)
        curr_states = aatypes_t.long().flatten()

        # set the probabilities corresponding to the current amino acid types to 0.0
        probs[batch_idx, residue_idx, curr_states] = 0.0

        # adjust the probabilities at the current positions to be 1 - sum of all other values in the row
        row_sums = torch.sum(probs, dim=-1).flatten()
        probs[batch_idx, residue_idx, curr_states] = 1.0 - row_sums

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        # in case negative or out-of-bound values appear after the diagonal assignment.
        probs = torch.clamp(probs, min=0.0, max=1.0)

        return probs

    def _aatypes_euler_step_uniform(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        logits_1: torch.Tensor,  # (B, N, S=20)
        aatypes_t: torch.Tensor,  # (B, N)
    ):
        num_batch, num_res, num_states = logits_1.shape
        assert aatypes_t.shape == (num_batch, num_res)
        assert num_states == 20
        assert (
            aatypes_t.max() < 20
        ), "No UNK tokens allowed in the uniform sampling step!"

        device = logits_1.device
        temp = self.cfg.aatypes.drift_temp
        noise = self.cfg.aatypes.stochastic_noise_intensity

        # convert logits to probabilities
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

        # probability of x1 matching xt exactly
        pt_x1_eq_xt_prob = torch.gather(
            pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)
        )  # (B, N, 1)
        assert pt_x1_eq_xt_prob.shape == (num_batch, num_res, 1)

        # compute step probabilities (scaled by d_t), with noise and time factoring.
        # encourages transitions with an additional uniform 'noise' term for the matching residue.
        step_probs = d_t * (
            pt_x1_probs * ((1.0 + noise + noise * (num_states - 1) * t) / (1.0 - t))
            + noise * pt_x1_eq_xt_prob
        )

        # force valid rate matrix
        step_probs = self._regularize_step_probs(step_probs, aatypes_t)

        # sample new residues from step_probs
        new_aatypes = torch.multinomial(step_probs.view(-1, num_states), num_samples=1)
        return new_aatypes.view(num_batch, num_res)

    def _aatypes_euler_step_masking(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
    ):
        num_batch, num_res, num_states = logits_1.shape
        assert num_states == 21
        assert aatypes_t.shape == (num_batch, num_res)

        device = logits_1.device
        temp = self.cfg.aatypes.drift_temp
        noise = (
            self.cfg.aatypes.stochastic_noise_intensity
        )  # used to be its own value, ~20.0

        # set mask to small negative so won't be picked in softmax
        logits_1[:, :, MASK_TOKEN_INDEX] = -1e9

        # convert logits to probabilities
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

        # prepare a (0,0,...1) mask vector to help add masking transitions.
        mask_one_hot = torch.zeros((num_states,), device=device)
        mask_one_hot[MASK_TOKEN_INDEX] = 1.0

        # identify which positions are currently mask
        aatypes_t_is_mask = (
            (aatypes_t == MASK_TOKEN_INDEX).view(num_batch, num_res, 1).float()
        )

        # compute step probabilities (scaled by d_t), with noise and time factoring
        step_probs = d_t * pt_x1_probs * ((1.0 + noise * t) / (1.0 - t))  # (B, N, S)
        # add transitions from non-mask to mask as noise
        step_probs += (
            d_t * (1.0 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * noise
        )

        # force valid rate matrix
        step_probs = self._regularize_step_probs(step_probs, aatypes_t)

        # sample new residues from step_probs
        new_aatypes = torch.multinomial(step_probs.view(-1, num_states), num_samples=1)
        return new_aatypes.view(num_batch, num_res)

    def _aatypes_euler_step_purity(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,)
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
    ):
        """
        Perform an Euler step with purity sampling scheme, which decides which tokens to unmask
        based on their log-probabilities.

        This function identifies masked residues, ranks them by max log prob,
        picks some number to unmask, and re-masks some positions proportional to (d_t, noise, t).

        On average more dimensions are unmasked at each step, with the probability of ultimately
        selecting the correct token increasing as noise increases.

        This function is designed specifically for the "masking" interpolant type, where S = 21.
        """
        num_batch, num_res, num_states = logits_1.shape

        assert aatypes_t.shape == (num_batch, num_res)
        assert (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        )
        assert (
            num_states == 21
        ), "Purity-based unmasking only works with masking interpolant type"

        device = logits_1.device
        temp = self.cfg.aatypes.drift_temp
        noise = self.cfg.aatypes.stochastic_noise_intensity

        # remove mask dimension to handle only the 20 valid aa states. mask is 21.
        logits_1_wo_mask = logits_1[:, :, : (num_states - 1)]  # (B, N, S-1)
        # convert logits to probabilities for the non-mask states
        pt_x1_probs = F.softmax(logits_1_wo_mask / temp, dim=-1)  # (B, N, S-1)
        # find maximum log prob among valid states
        max_logprob = torch.log(pt_x1_probs).max(dim=-1)[0]  # (B, N)
        # bias so only currently masked are favored for unmasking
        max_logprob -= (aatypes_t != MASK_TOKEN_INDEX).float() * 1e9
        # sort positions by descending maximum log-prob (highest to lowest)
        sorted_max_logprob_indices = torch.argsort(
            max_logprob, dim=-1, descending=True
        )  # (B, N)

        # determine how many masked residues to unmask
        # probability of unmasking is scaled by d_t, noise, and time
        unmask_probs = (
            (d_t * ((1.0 + noise * t) / (1.0 - t))).clamp(max=1.0).to(device)
        )  # scalar
        masked_counts = (aatypes_t == MASK_TOKEN_INDEX).sum(dim=-1).float()  # (B,)
        number_to_unmask = torch.binomial(
            count=masked_counts, prob=unmask_probs
        )  # (B,)

        # sample new residues for unmasked positions from pt_x1_probs
        # we'll assign these new residues to top 'number_to_unmask' positions
        unmasked_samples = torch.multinomial(
            pt_x1_probs.view(-1, num_states - 1), num_samples=1
        )
        unmasked_samples = unmasked_samples.view(num_batch, num_res)

        # Next lines are vectorized version of:
        # for b in range(B):
        #     for n in range(N):
        #         if n < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprob_indices[b, n]] = unmasked_samples[b, sorted_max_logprob_indices[b, d]]

        # create a mask that indicates top positions to unmask in each batch
        D_grid = (
            torch.arange(num_res, device=device).unsqueeze(0).expand(num_batch, -1)
        )  # (B, N)
        # 'mask1' is 1 where d < number_to_unmask[b], 0 otherwise
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()

        # if 0 to unmask, we set sorted_max_logprob_indices[:, 0] for all positions
        fallback = sorted_max_logprob_indices[:, 0].view(-1, 1).repeat(1, num_res)
        selected_indices = (mask1 * sorted_max_logprob_indices) + (
            (1.0 - mask1) * fallback
        )
        selected_indices = selected_indices.long()

        # 'mask2' indicates precisely which positions in aatypes_t we replace
        mask2 = torch.zeros((num_batch, num_res), device=device)
        mask2.scatter_(
            dim=1,
            index=selected_indices,
            src=torch.ones((num_batch, num_res), device=device),
        )
        # if zero are unmasked, we skip altogether
        none_unmasked_mask = (number_to_unmask == 0).unsqueeze(-1).float()
        mask2 *= 1.0 - none_unmasked_mask

        # update unmasked positions in aatypes_t
        aatypes_t = aatypes_t * (1.0 - mask2) + unmasked_samples * mask2

        # re-mask some positions as noise with probability (d_t * noise)
        re_mask_prob = d_t * noise
        rand_vals = torch.rand(num_batch, num_res, device=device)
        re_mask_mask = (rand_vals < re_mask_prob).float()
        aatypes_t = aatypes_t * (1.0 - re_mask_mask) + (MASK_TOKEN_INDEX * re_mask_mask)

        return aatypes_t

    def _aatype_jump_step(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,) in [0,1]
        logits_1: Optional[torch.Tensor],  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        CTMC jump for aatypes. pass no logits for stochastic jump.

        Uses logits -> rate matrix to allow sequnce to explore neighboring states
        in proportion to the rates the network thinks are possible.
        So notice that unlike in training, where jumps are to a uniform-random sampled state,
        here the jumps are to states the network thinks are plausible.
        The thinking is that during training, the model must learn a broad "restoring drift" and uniform sampling is class balanced.
        (We can't use logits during training without simulation, they are just a one-hot of the sequence.)
        However, this training may slow convergence, or the model may only learn to fix very wrong residues.

        This is different than the `cfg.aatypes.noise` term,
        which adds noise to the rate matrix in determinsitic interpolation.
        """
        B, N, S = logits_1.shape
        assert aatypes_t.shape == (B, N)

        device = logits_1.device

        # logits -> probabilities
        prob_rows = F.softmax(
            logits_1 / self.cfg.aatypes.drift_temp, dim=-1
        )  # (B, N, S)
        prob_rows = prob_rows.clamp(min=1e-8)  # avoid zeros

        # Build rate matrix (positive exits, negative stays)
        current_idx = aatypes_t.unsqueeze(-1).long()  # (B,N,1)
        exit_rates = prob_rows.scatter_(-1, current_idx, 0.0)
        exit_sums = exit_rates.sum(-1, keepdim=True)
        step_rates = exit_rates.clone()
        step_rates.scatter_(-1, current_idx, -exit_sums)  # current = neg sum of others

        # Multiply by sigma_t so amount of noise mirrors other domains.
        # Multiplying the whole rate matrix by the same value preserves valid rate matrix.
        sigma_t = self._compute_sigma_t(
            torch.ones(aatypes_t.shape[0], device=device) * t.to(device),
            scale=self.cfg.aatypes.stochastic_noise_intensity * stochasticity_scale,
        )
        step_rates = step_rates * sigma_t[..., None, None]  # (B, N, S)

        # decide whether each residue jumps during d_t
        # total exit rate λ_i = − q_{i, current_state}
        lambda_step = -step_rates.gather(-1, current_idx).squeeze(-1)  # (B, N), >0
        p_jump = 1.0 - torch.exp(-lambda_step * d_t)  # (B, N)
        jump_mask = torch.rand_like(p_jump) < p_jump  # bool (B, N)

        if not jump_mask.any():
            # no jumps this step
            return aatypes_t

        # sample new aa for jumped residues
        jump_aa_probs = step_rates.clone()  # (B, N, S)
        jump_aa_probs.scatter_(-1, current_idx, 0.0)  # zero out current col
        jump_aa_probs = jump_aa_probs / lambda_step.clamp_min(1e-8).unsqueeze(
            -1
        )  # normalise rows

        # sample new aa only for residues that jump
        jumped_states = torch.multinomial(
            jump_aa_probs[jump_mask].reshape(-1, S), 1
        ).squeeze(-1)

        jumped_aatypes_t = aatypes_t.clone().float()
        jumped_aatypes_t[jump_mask] = jumped_states.float()
        return jumped_aatypes_t

    def _aatypes_euler_step_old(
        self,
        d_t: torch.Tensor,  # scalar Tensor (0-d)
        t: torch.Tensor,  # scalar Tensor (0-d) or (B,) in [0,1]
        logits_1: torch.Tensor,  # (B, N, S) unscaled probabilities, S={20, 21}
        aatypes_t: torch.Tensor,  # (B, N) current amino acid types
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,  # (B, N, S)
    ):
        """
        Perform an Euler step to update amino acid types based on the provided logits and interpolation settings.

        This function handles two interpolation strategies:
        1. "masking": Updates the amino acid types by masking certain positions and sampling new types
           based on modified probabilities. Assumes S = 21 to include a special MASK token.
        2. "uniform": Samples new amino acid types uniformly, with the assumption that no MASK tokens are involved.
           Assumes S = 20.
        """
        # incorporate guidance logits if provided ahead of aatypes euler step method
        if potential is not None:
            assert (
                potential.shape == logits_1.shape
            ), f"Guidance logits shape {potential.shape} does not match logits_1 shape {logits_1.shape}"
            logits_1 = logits_1 + potential

        if self.cfg.aatypes.purity_selection:
            aatypes_t = self._aatypes_euler_step_purity(
                d_t=d_t,
                t=t,
                logits_1=logits_1,
                aatypes_t=aatypes_t,
            )
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            aatypes_t = self._aatypes_euler_step_masking(
                d_t=d_t,
                t=t,
                logits_1=logits_1,
                aatypes_t=aatypes_t,
            )
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            aatypes_t = self._aatypes_euler_step_uniform(
                d_t=d_t,
                t=t,
                logits_1=logits_1,
                aatypes_t=aatypes_t,
            )
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

        if (
            self.cfg.aatypes.stochastic
            and self.cfg.aatypes.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # Additional stochastic CTMC jump step
            aatypes_t = self._aatype_jump_step(
                d_t,
                t=t,
                logits_1=logits_1,
                aatypes_t=aatypes_t,
                stochasticity_scale=stochasticity_scale,
            )

        return aatypes_t

    def _aatypes_schedule(
        self, t: torch.Tensor, kappa: float = 1.0, eps: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Schedule for aatypes drift rate per-component scales

        Returns `sched` and `tau` (mapped t -> y),
        where `sched` is (1 + kappa * tau + eps) / (1 - tau + eps)

        eps smooths blowup at t=1
        kappa > 0 adds some mass early on
        """
        if eps is None:
            eps = self.cfg.min_t

        if self.cfg.aatypes.schedule == InterpolantAATypesScheduleEnum.linear:
            y = t
        elif self.cfg.aatypes.schedule == InterpolantAATypesScheduleEnum.exp:
            # map t -> y in [0,1): y = (1 - e^{-rt}) / (1 - e^{-r})
            r = float(abs(self.cfg.aatypes.schedule_exp_rate))
            y = (1.0 - torch.exp(-r * t)) / (1.0 - math.e ** (-r))
        else:
            raise ValueError(f"Unknown aatypes schedule {self.cfg.aatypes.schedule}")

        sched = (1.0 + kappa * y + eps) / (1.0 - y + eps)
        sched = sched.clamp_min(0.0)

        return sched, y

    def _aatypes_component_scales(
        self,
        t: torch.Tensor,  # scalar or (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Time-only aatypes drift rate per-component scales

        Returns:
        change_scale (AA->AA')  : (B, 1, 1)  ~ cfg.change_rate * s_sched(t)
        unmask_scale (mask->AA) : (B, 1, 1)  ~ cfg.unmask_rate * s_sched(t)
        remask_scale (AA->MASK) : (B, 1, 1)  ~ cfg.remask_rate * 1/s_sched(t) - eps [->0 @ t=1]
        """
        t = t.to(self._device)
        if t.dim() == 0:
            t_b = t.expand(1)  # (B=1,)
        else:
            t_b = t.view(-1)  # (B,)

        # aggressively smooth out near t=1
        eps = 0.25
        sched, _ = self._aatypes_schedule(t=t_b, eps=eps)
        sched = sched.view(-1, 1, 1)
        inv_sched = (1.0 / sched).view(-1, 1, 1)

        # multiply base component weight * schedule
        change_scale = self.cfg.aatypes.change_rate * sched
        unmask_scale = self.cfg.aatypes.unmask_rate * sched
        remask_scale = self.cfg.aatypes.remask_rate * inv_sched - eps

        return (
            change_scale.clamp_min(0.0),
            unmask_scale.clamp_min(0.0),
            remask_scale.clamp_min(0.0),
        )

    def _aatypes_build_rates_drift(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        logits_1: torch.Tensor,  # (B, N, S)
        t: torch.Tensor,  # scalar or (B,)
        potential: Optional[torch.Tensor] = None,  # (B, N, S)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate drift rate vector from current state (B, N, S) and change probabilities (B, N, S).

        The probabilities:
        - are per-site perference over targets (no "stay" entry)
        - sum to 1
        - can be converted to Multiflow-style rate matrix by _regularize_step_probs() for categorical / euler discrete sampling

        The rates:
        - represent per-site off-diagonal exit rates with units 1/time
        - is a rate vector that can be used directly by CTMC jump sampling
        - are scaled per `t` schedule, and change rates specified in config
        - the "stay" rate is the negative sum of the other rates; stay is implicit

        In contrast to the Multiflow-style rate matrix construction, which regularizes the step change probabilities
        by setting the current state to the negative sum of the other rates (i.e. `d_t * rate`), CTMC jump uses one-jump
        hazard rate `1 - exp(-lambda * d_t)`, where `lambda` is the exit rate per site.
        They are approximately equal on the first order for low lambda and low `d_t`.
        One benefit, since we support stochastic paths, is that the rate matrices can be added, to perform a single CTMC jump.

        Purity gating should be applied to the returned rates (after adding noise).
        """
        B, N, S = logits_1.shape
        dev = logits_1.device
        t_b = t.to(dev) if t.dim() == 0 else t.to(dev).view(-1)  # (B,)
        interpolant_type = self.cfg.aatypes.interpolant_type

        # combine model logits and potential logits
        if potential is not None:
            # Note that the potential logits is already the combination of all the potentials,
            # and has been centered. Combining the potential logits with the predicted logits
            # yields the same softmax probs as if they had been added all at once, but the logits
            # themselves differ by a row-wise constant. So, if in the future we are interested in
            # some logits-loss, the logits should be normalized, or use probabilities instead.
            drift_logits = combine_logits(logits=[logits_1, potential])
        else:
            drift_logits = logits_1

        # in masking, forbid mask as drift target
        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            assert S == 21
            drift_logits = drift_logits.clone()
            drift_logits[..., MASK_TOKEN_INDEX] = -1e9
        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            assert S == 20
            assert aatypes_t.lt(S).all()
        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        # compute probabilities from logits, scaled by temperature
        drift_temp = self.cfg.aatypes.drift_temp
        drift_probs = torch.softmax(drift_logits / drift_temp, dim=-1)  # (B, N, S)

        # per-component scales as some function of t
        change_scale, unmask_scale, remask_scale = self._aatypes_component_scales(t_b)

        # masks for mask vs AA, depending on interpolant type
        aa_cols = torch.ones(S, dtype=torch.bool, device=dev)
        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            aa_cols[MASK_TOKEN_INDEX] = False
        is_mask = aatypes_t == MASK_TOKEN_INDEX
        is_aa = ~is_mask

        # uncertainty gating: for AA rows use 1 - p(current), for MASK rows set to 1
        # TODO consider scaling (1 - p(current)) ** 1/gamma (configurable) to sharpen
        if self.cfg.aatypes.uncertainty_gating:
            p_current = (
                drift_probs.gather(-1, aatypes_t.long().unsqueeze(-1))
                .squeeze(-1)
                .clamp(0.0, 1.0)
            )  # (B, N)
            # (1 - p_current) ** gamma
            uncert = (1.0 - p_current) ** self.cfg.aatypes.uncertainty_gating_gamma
            # mask to AA rows
            uncert = uncert * is_aa.float() + (1.0 * is_mask.float()) # (B, N)
        else:
            uncert = torch.ones_like(is_aa, dtype=torch.float, device=dev)  # (B, N)

        # off-diagonal step mass defined below depending on interpolant type + rates
        rates_off = torch.zeros_like(drift_probs)

        # for masked positions, all mass goes to unmasking
        # (implicitly, this must be masking interpolant)
        if is_mask.any():
            # disallow mask -> mask
            probs_aa = drift_probs.masked_fill(~aa_cols.view(1, 1, -1), 0.0)
            # normalize probs over AA columns only
            z = probs_aa.sum(-1, keepdim=True).clamp_min(1e-8)
            probs_aa = probs_aa / z
            # put unmask mass on AA columns
            rates_unmask = (
                unmask_scale * uncert.unsqueeze(-1) * probs_aa
            ) * is_mask.unsqueeze(-1).float()
            rates_off += rates_unmask

        # for AA positions, mass is split between change and remask
        if is_aa.any():
            # disallow -> self
            change_probs = drift_probs.clone()
            change_probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
            # disallow -> mask here (remasking handled separately below)
            if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
                change_probs[..., MASK_TOKEN_INDEX] = 0.0
            z = change_probs.sum(-1, keepdim=True).clamp_min(1e-8)
            change_probs = change_probs / z
            rates_change = (
                change_scale
                * uncert.unsqueeze(-1)
                * change_probs
                * is_aa.unsqueeze(-1).float()
            )  # (B, N, S)
            rates_off += rates_change

            # remasking has single mask target
            if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
                rates_remask = (remask_scale * uncert.unsqueeze(-1)).squeeze(
                    -1
                )  # (B, N)
                rates_off[..., MASK_TOKEN_INDEX] += rates_remask * is_aa.float()

        # convert to off-diagonal rates r_off for jump sampling
        # defensively ensure self column is zero
        rates_off.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)

        return rates_off, drift_probs

    def _aatypes_base_rates_noise(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """
        Build logits-free corruption-matching noise rate vector (B, N, S).
        If aatypes are provided, no hazard rate for current state.
        """
        B, N = aatypes_t.shape
        device = aatypes_t.device

        interpolant_type = self.cfg.aatypes.interpolant_type

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            S = 21
            aa_cols = torch.ones(S, dtype=torch.bool, device=device)
            aa_cols[MASK_TOKEN_INDEX] = False
            aa_count = int(aa_cols.sum().item())  # 20
        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            S = 20
            assert aatypes_t.lt(S).all(), "unexpected mask token"
        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        # Initialize rates
        rates_noise = torch.zeros(B, N, S, device=device)

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            # AA -> AA' uniformly over S-1 (excludes self)
            rates_noise += 1.0 / (S - 1)  # (B, N, S)
            rates_noise.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)

        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
            is_aa = ~is_mask

            # mask rows: split uniformly across AA columns
            if is_mask.any():
                masked_view = rates_noise[is_mask]
                masked_view[..., aa_cols] = 1.0 / aa_count
                rates_noise[is_mask] = masked_view

            # AA rows: split mass between AA->AA' and AA->mask
            if is_aa.any():
                prop_change = self.cfg.aatypes.noise_prop_change
                prop_remask = 1.0 - prop_change

                # AA -> mask (remask)
                aa_view = rates_noise[is_aa]
                aa_view[..., MASK_TOKEN_INDEX] = prop_remask
                rates_noise[is_aa] = aa_view

                # AA -> AA' uniformly (exclude self and mask)
                change_probs = torch.zeros(B, N, S, device=device)  # (B, N, S)
                change_probs[
                    is_aa.unsqueeze(-1).expand(-1, -1, S) & aa_cols.view(1, 1, -1)
                ] = 1.0
                change_probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
                z = change_probs.sum(-1, keepdim=True).clamp_min(1e-8)
                change_probs = change_probs / z
                rates_noise += (prop_change * is_aa.float()).unsqueeze(
                    -1
                ) * change_probs

        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        return rates_noise

    def _aatypes_build_rates_noise(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # scalar or (B,)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Build logits-free corruption-matching noise rate vector (B, N, S).

        - uniform: AA -> AA' uniformly over S-1 targets (excludes self)
        - masking: mask -> AA uniformly over 20 AAs; AA -> mask or AA' (excludes self) with fixed split

        per-row noise mass is sigma_t * stochastic_noise_intensity * stochasticity_scale

        Returns rates (in units 1/time) independent of d_t. Caller adds to drift rates and does a single CTMC jump.
        """
        B, N = aatypes_t.shape
        device = aatypes_t.device

        interpolant_type = self.cfg.aatypes.interpolant_type

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            S = 21
            aa_cols = torch.ones(S, dtype=torch.bool, device=device)
            aa_cols[MASK_TOKEN_INDEX] = False
            aa_count = int(aa_cols.sum().item())  # 20
        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            S = 20
            assert aatypes_t.lt(S).all(), "unexpected mask token"
        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        # Initialize rates
        rates_noise = torch.zeros(B, N, S, device=device)

        if stochasticity_scale == 0.0:
            return rates_noise

        # sigma_t per batch (B,) scaled by config + stochasticity_scale
        if isinstance(t, torch.Tensor) and t.dim() > 0:
            t_b = t.to(device).view(-1)  # (B,)
        else:
            t_b = torch.full((B,), float(t), device=device)  # (B,)
        sigma = self._compute_sigma_t(
            t=t_b,
            scale=self.cfg.aatypes.stochastic_noise_intensity * stochasticity_scale,
        )
        sigma = sigma.clamp_min(0.0)  # (B,)

        # Broadcasts
        sigma_BN = sigma.view(B, 1).expand(B, N)  # (B, N)
        sigma_BN1 = sigma_BN.unsqueeze(-1)  # (B, N, 1)

        if interpolant_type == InterpolantAATypesInterpolantTypeEnum.uniform:
            # AA -> AA' uniformly over S-1 (excludes self)
            rates_noise += sigma_BN1 / (S - 1)  # (B, N, S)
            rates_noise.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)

        elif interpolant_type == InterpolantAATypesInterpolantTypeEnum.masking:
            is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
            is_aa = ~is_mask

            # MASK rows: split sigma_t uniformly across AA columns
            if is_mask.any():
                masked_view = rates_noise[is_mask]
                masked_view[..., aa_cols] = (sigma_BN[is_mask] / aa_count).unsqueeze(-1)
                rates_noise[is_mask] = masked_view

            # AA rows: split sigma between AA->AA' and AA->MASK
            if is_aa.any():
                prop_change = 0.5
                prop_remask = 1.0 - prop_change

                # AA -> mask (remask)
                aa_view = rates_noise[is_aa]
                aa_view[..., MASK_TOKEN_INDEX] = sigma_BN[is_aa] * prop_remask
                rates_noise[is_aa] = aa_view

                # AA -> AA' uniformly (exclude self and mask)
                change_probs = torch.zeros(B, N, S, device=device)  # (B, N, S)
                change_probs[
                    is_aa.unsqueeze(-1).expand(-1, -1, S) & aa_cols.view(1, 1, -1)
                ] = 1.0
                change_probs.scatter_(-1, aatypes_t.long().unsqueeze(-1), 0.0)
                zc = change_probs.sum(-1, keepdim=True).clamp_min(1e-8)
                change_probs = change_probs / zc
                # lambda_change is per-row mass
                lambda_change = sigma_BN * prop_change * is_aa.float()  # (B, N)
                rates_noise += lambda_change.unsqueeze(-1) * change_probs

        else:
            raise ValueError(f"Unknown aatypes interpolant type {interpolant_type}")

        return rates_noise

    def _aatypes_purity_gate(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        rates: torch.Tensor,  # (B, N, S) drift rates
        drift_probs: torch.Tensor,  # (B, N, S) drift softmax, used for scoring
        t: torch.Tensor,  # scalar or (B,)
        d_t: torch.Tensor,  # scalar
    ) -> torch.Tensor:
        """
        Build a purity gate (B, N, S) for the masking interpolant that prevents unmasking on
        masked rows not selected by purity in this step. This mask never touches AA rows
        or the mask column; it only zeros mask -> AA transitions for non-selected masked rows.

        The gate is built by:
        - score masked rows by confidence (max over AA probs)
        - compute per-row jump probabilities from drift
        - sample Bernoulli for masked rows, then sum -> K_b (Poisson–binomial)
        - select top-K masked rows by confidence; other masked rows are gated

        Use the returned gate to zero out combined (drift + noise)rates:
        ```
        purity_gate = self._aatypes_purity_gate(...)
        rates = rates.masked_fill(purity_gate, 0.0)
        ```
        """
        B, N, S = rates.shape
        device = rates.device
        assert S == 21, "Purity is only defined for masking (21 tokens)."

        # rows that are currently MASK
        is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
        masked_counts = is_mask.sum(dim=1)  # (B,)
        if masked_counts.max() == 0:
            return torch.zeros_like(rates, dtype=torch.bool)

        # mask for mask vs AA
        aa_cols = torch.ones(S, dtype=torch.bool, device=device)
        aa_cols[MASK_TOKEN_INDEX] = False

        # calculate exit prob per masked row this step to determine number to unmask
        # per-row confidence = max over AA columns of predicted probabilities
        # p_row: (B, N) with p_bi = 1 - exp(-lambda_bi * d_t)
        lam = rates.sum(dim=-1)  # (B, N)
        if d_t.dim() == 0:
            p_row = 1.0 - torch.exp(-lam * d_t.to(device))  # (B, N)
        else:
            p_row = 1.0 - torch.exp(-lam * d_t.to(device).view(-1, 1))  # (B, N)
        p_row = p_row.clamp(0.0, 1.0)
        # sample independent Bernoulli for masked rows, then sum -> K_b (Poisson–binomial)
        u = torch.rand_like(p_row)
        unmask_draw = (u < p_row) & is_mask  # (B, N) 1 if that row would unmask
        number_to_unmask = unmask_draw.sum(dim=1).to(torch.long)  # (B,)

        # build gate where True = allowed to unmask
        # select top-K masked rows per batch (vectorized via ranks)
        # ranks: 0 for largest score, 1 for second, ...
        aa_max = drift_probs[..., aa_cols].amax(dim=-1)  # (B, N)
        neg_inf = torch.tensor(float("-inf"), device=device, dtype=aa_max.dtype)
        masked_scores = torch.where(
            is_mask, aa_max, neg_inf
        )  # (B, N), non-masked = -inf
        sort_idx = torch.argsort(masked_scores, dim=1, descending=True)  # (B, N)
        ranks = torch.empty_like(sort_idx)
        D_grid = torch.arange(N, device=device).unsqueeze(0).expand(B, -1)
        ranks.scatter_(1, sort_idx, D_grid)
        gate = (ranks < number_to_unmask.view(-1, 1)) & is_mask  # (B, N)

        # zero mask->AA rates in rows not selected
        to_zero = ((~gate) & is_mask).unsqueeze(-1) & aa_cols.view(
            1, 1, -1
        )  # (B, N, S)

        return to_zero

    def _aatypes_regularize_rates(
        self,
        rates: torch.Tensor,  # (B, N, S)
        drift_probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        t: torch.Tensor,  # scalar or (B,)
        d_t: torch.Tensor,  # scalar or (B,)
    ) -> torch.Tensor:
        """
        Regularize rates by capping them.

        Computes a per-row jump probability cap `pmax` and converts to a smooth, row-wise squash.
        Small rows are ~unchanged, large rows are soft-capped so still allow confident changes.
        """
        B, N, S = rates.shape
        device = rates.device
        t_b = t.to(device) if t.dim() > 0 else t.to(device).expand(B)  # (B,)
        dt_b = d_t.to(device) if d_t.dim() > 0 else d_t.to(device).expand(B)  # (B,)

        # uncertainty only for AA rows
        is_mask = aatypes_t == MASK_TOKEN_INDEX  # (B, N)
        p_current = (
            drift_probs.gather(-1, aatypes_t.long().unsqueeze(-1))
            .squeeze(-1)
            .clamp(0.0, 1.0)
        )
        uncert = (1.0 - p_current) * (~is_mask).float()  # (B, N)

        # simple per-step probability caps
        # mask: allow aggressive late unmasking (ramps to 0.6)
        pmax_mask = (
            torch.clamp(0.1 + 0.6 * t_b, 0.05, 0.6).view(B, 1).expand(B, N)
        )  # (B, N)
        # AA: allow flips mostly when uncertain & mid-trajectory
        pmax_aa = torch.clamp(
            0.4 * (t_b * (1 - t_b)).view(B, 1) * uncert, 0.0, 0.25
        )  # (B, N)

        # pmax depends on whether currently masked or AA
        # avoid 1.0 (would give inf rate)
        pmax = torch.where(is_mask, pmax_mask, pmax_aa).clamp(0.0, 0.999)

        # convert probability to rate cap
        # CTMC relation: p = 1 - exp(-lambda * d_t) -> lambda = -log(1-p) / d_t
        lambda_cap = -torch.log1p(-pmax) / dt_b.view(B, 1)

        # soft-cap λ_eff = λ_cap * (1 - exp(-λ_raw / λ_cap))
        # monotone, identity for small rows, bounded for large rows
        # total row hazard λ_raw = sum_j r_ij (off-diagonal rates)
        lambda_raw = rates.sum(dim=-1)
        # λ_eff = λ_cap * (1 - exp(-λ_raw / λ_cap))
        lambda_eff = lambda_cap * (1.0 - torch.exp(-lambda_raw / (lambda_cap + 1e-6)))

        # scale = λ_eff / λ_raw preserves destination mix but limits movement per step
        scale = (lambda_eff / (lambda_raw + 1e-6)).unsqueeze(-1)  # (B, N, 1)

        return rates * scale

    def _aatypes_ctmc_jump(
        self,
        aatypes_t: torch.Tensor,  # (B, N)
        rates: torch.Tensor,  # (B, N, S)
        d_t: torch.Tensor,  # scalar or (B,)
    ) -> torch.Tensor:
        """
        Single CTMC jump step using rate matrix (i.e. rates from current state), scaled by d_t.

        rates (hazard densities) are functions of t with units 1/time.
        per-step jump probabilities are the rates multiplied by d_t:
        p_jump = 1.0 - exp(-lambda_step * d_t)
        where lambda_step = sum rates at that position
        (and so sum rates -> >1 => p_jump -> >d_t)
        """
        B, N = aatypes_t.shape
        S = rates.shape[-1]
        device = rates.device

        # ensure d_t is on the right device/shape
        d_t_b = d_t.to(device)
        if d_t_b.dim() == 0:
            d_t_b = d_t_b.expand(B)  # (B,)
        elif d_t_b.dim() == 1:
            assert d_t_b.shape[0] == B, "d_t shape must match batch"
        else:
            raise ValueError("d_t must be scalar or (B,)")

        # decide whether each residue jumps during d_t
        lambda_step = rates.sum(dim=-1)  # exit rate per site (B, N)
        p_jump = 1.0 - torch.exp(-lambda_step * d_t_b.view(-1, 1))  # (B, N)
        jump_mask = torch.rand_like(p_jump) < p_jump  # (B, N)
        if not jump_mask.any():
            return aatypes_t

        # sample new aa for jumped residues
        jump_rates = rates.clone()
        current_aatypes_idx = aatypes_t.long().unsqueeze(-1)
        jump_rates.scatter_(-1, current_aatypes_idx, 0.0)  # zero out current col
        # normalize, conditional over targets
        jump_probs = jump_rates / lambda_step.clamp_min(1e-8).unsqueeze(-1)

        # sample new aa only for residues that jump
        new_states = torch.multinomial(jump_probs[jump_mask].reshape(-1, S), 1).squeeze(
            -1
        )
        new_aatypes = aatypes_t.clone()
        new_aatypes[jump_mask] = new_states.to(aatypes_t.dtype)
        return new_aatypes

    def _aatypes_euler_step(
        self,
        d_t: torch.Tensor,  # scalar
        t: torch.Tensor,  # scalar or (B,)
        logits_1: torch.Tensor,  # (B,N,S)
        aatypes_t: torch.Tensor,  # (B,N)
        stochasticity_scale: float = 1.0,
        potential: Optional[torch.Tensor] = None,
    ):
        # get drift rates + probs
        rates_drift, drift_probs = self._aatypes_build_rates_drift(
            aatypes_t=aatypes_t, logits_1=logits_1, t=t, potential=potential
        )

        # If purity sampling is enabled, not only do we avoid AA -> AA' transitions,
        # but also limit rates to top-K masked sites by ranked confidence
        # TODO - match Multiflow behavior of guaranteeing jump?
        if (
            self.cfg.aatypes.purity_selection
            and self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            # Purity is determined by the drift rates
            purity_gate = self._aatypes_purity_gate(
                aatypes_t=aatypes_t,
                rates=rates_drift,
                drift_probs=drift_probs,
                t=t,
                d_t=d_t,
            )
            rates = rates.masked_fill(purity_gate, 0.0)

        # soft-cap regularize rates to prevent blowups as t->1
        rates = self._aatypes_regularize_rates(
            rates=rates, drift_probs=drift_probs, aatypes_t=aatypes_t, t=t, d_t=d_t
        )

        # optionally add logits-free noise
        rates_noise = torch.zeros_like(rates_drift)
        if (
            self.cfg.aatypes.stochastic
            and self.cfg.aatypes.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            rates_noise = self._aatypes_build_rates_noise(
                aatypes_t=aatypes_t, t=t, stochasticity_scale=stochasticity_scale
            )

        rates = rates_drift + rates_noise

        # single CTMC jump using combined rates
        aatypes_next = self._aatypes_ctmc_jump(
            aatypes_t=aatypes_t, rates=rates, d_t=d_t
        )

        return aatypes_next

    def motif_potentials(
        self,
        t: float,  # scalar Tensor
        noisy_batch: NoisyFeatures,
        pred: ModelPrediction,
        true_feats: BatchTrueFeatures,
        motif_mask: Optional[torch.Tensor],  # (B, N) bool
    ) -> PotentialField:
        """
        Generate potentials for motif guidance, returning translation and rotation vector fields.
        `aatypes` are fixed in motifs, so no guidance for them.

        Model predicts an unconditional drift, and this VF (masked to motifs) is passed to
        euler steps and modifies those unconditional velocities.

        This is similar to how FrameFlow does it, or twisted SMC,
        but only considering a single trajectory and without gradients.

        Cf. substituting the motifs into the prediction and interpolating towards them,
        prefer using an additional velocity because:
        - keep ODE smooth
        - reduce scaffold jitter
        - softens as t-> 1 (but > 0) so motifs are not completely locked in
        - preserves equivariance

        See FrameFlow paper for details about motif guidance (sec 3.2):
        https://arxiv.org/pdf/2401.04082
        """
        if motif_mask is None or not motif_mask.any():
            return PotentialField()

        trans_t = noisy_batch[nbp.trans_t]
        rotmats_t = noisy_batch[nbp.rotmats_t]
        pred_trans_1 = pred[pbp.pred_trans]
        pred_rotmats_1 = pred[pbp.pred_rotmats]
        motif_sel = motif_mask.bool()

        eps = 1e-3
        t = torch.clamp(t, min=eps, max=1.0 - eps)

        # derive scale = 1/2 g(t)² / ω², starting with g(t) (guidance scale) and ω² (posterior scale)
        # ω² values from FrameFlow eq 14, 15 assume linear interpolation
        # ω²(t) for linear path κ(t)=1−t is (1−t)² / (t² + (1−t)²)
        # generalized form: ω²(t) = κ(t)² / (t² + κ(t)²)
        # similarly, g(t) in FrameFlow is (1-t)/t because linear and so because it matches
        # diffusion coefficient for diffusion SDE that matches marginals of flow SDE.
        # g(t) can be generalized to κ(t) / t
        # However, we support non-linear schedules for interpolation.
        # Additionally, translations and rotations may have different schedules.
        # Note on signs -- the scale is positive, and the drift is (true - pred).

        # translations
        if self.cfg.trans.sample_schedule == InterpolantTranslationsScheduleEnum.linear:
            kappa_trans = 1.0 - t
        elif (
            self.cfg.trans.sample_schedule == InterpolantTranslationsScheduleEnum.vpsde
        ):
            bmin, bmax = self.cfg.trans.vpsde_bmin, self.cfg.trans.vpsde_bmax
            # ᾱ(t) = exp(-∫₀ᵗ β(s)ds), β(s)=bmax-(bmax-bmin)s  ⇒ ∫= bmax t - 0.5(bmax-bmin)t²
            alphabar_t = torch.exp(-(bmax * t - 0.5 * (bmax - bmin) * t * t))
            # κ_T(t) = sqrt(ᾱ(t)) makes κ(0)=1, κ(1)≈0 (noise→data)
            kappa_trans = torch.sqrt(alphabar_t)
        else:
            raise ValueError("Unknown translation schedule")
        g_trans = kappa_trans / t
        omega2_trans = kappa_trans**2 / (t**2 + kappa_trans**2)
        scale_trans = 0.5 * g_trans * g_trans / omega2_trans

        # rotations
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            kappa_rotmats = 1.0 - t
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            # exp(-c•t) for c>0 => fast early rotation "lock-in" and vanishing near t->1
            kappa_rotmats = torch.exp(-self.cfg.rots.exp_rate * t)
        else:
            raise ValueError("Unknown rotation schedule")
        g_rotmats = kappa_rotmats / t
        omega2_rotmats = kappa_rotmats**2 / (t**2 + kappa_rotmats**2)
        scale_rotmats = 0.5 * g_rotmats * g_rotmats / omega2_rotmats

        # clamp the scales, which can be very large (>100) near t=0 and lead to overshooting.
        # TODO make configurable, scale depending on noise strength, maybe proportional to uncond VF
        scale_trans = torch.clamp(scale_trans, min=0.0, max=50.0).to(
            pred_trans_1.device
        )
        scale_rotmats = torch.clamp(scale_rotmats, min=0.0, max=50.0).to(
            pred_rotmats_1.device
        )

        # trans_vf = 1/2 g(t)² ∇x / ω²(t) = scale_trans • ∇x
        # diff = (true - pred) so push towards target.
        trans_vf = torch.zeros_like(pred_trans_1)  # (B, N, 3)
        trans_vf[motif_sel] = (true_feats.trans - pred_trans_1)[motif_sel]
        trans_vf *= scale_trans

        # rotmats_vf = 1/2 g(t)² ∇r / ω²(t) = scale_rotmats • ∇r
        # compute tangent from current -> true and current -> pred
        # and their difference is ~ the gradient (all in tangent space of current)
        rotmats_t_to_true = so3_utils.calc_rot_vf(
            mat_t=rotmats_t,
            mat_1=true_feats.rotmats,
        )
        rotmats_t_to_pred = so3_utils.calc_rot_vf(mat_t=rotmats_t, mat_1=pred_rotmats_1)
        rotmats_vf = torch.zeros_like(rotmats_t_to_true)
        rotmats_vf[motif_sel] = (rotmats_t_to_true - rotmats_t_to_pred)[motif_sel]
        rotmats_vf *= scale_rotmats

        return PotentialField(trans=trans_vf, rotmats=rotmats_vf)

    def _rot_sample_kappa(self, t: torch.Tensor):
        """kappa to scale rotation `so3_t` times, so that rotations can settle more quickly"""
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            return 1 - torch.exp(-t * self.cfg.rots.exp_rate)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            return t
        else:
            raise ValueError(f"Invalid schedule: {self.cfg.rots.sample_schedule}")

    def sample_single_step(
        self,
        noisy_batch: NoisyFeatures,
        true_feats: BatchTrueFeatures,
        model,
        task: InferenceTask,
        resampler: FKSteeringResampler,
        step_idx: int,
        t_1: torch.Tensor,  # scalar Tensor (0-d)
        t_2: Optional[torch.Tensor],  # scalar Tensor (0-d) or None if final step
        stochasticity_scale: float = 1.0,
    ) -> Tuple[NoisyFeatures, SamplingStep, SamplingStep, Optional[FKStepMetric]]:
        """
        Perform a single step of sampling, integrating from `t_1` toward `t_2`.
        Batch `_t` properties should be at `t_1`.

        Returns a batch with noisy features updated to t_2, and model + protein intermediate states.

        Note on torsions:
        `torsions_1` are always defined.
        `torsions_t` are always defined by `corrupt_batch`.
        `torsions_t` will always be defined during sampling, but may not be used.
        However, torsions are only optionally predicted by the model, and are used to generate
        the predicted structure if the model predicts them.
        If any torsions are predicted, `fill_torsions` fills them (B, N, K, 2) to (B, N, 7, 2).
        """

        is_final_step = t_2 is None

        # Pull out masks + idx
        res_mask = noisy_batch[bp.res_mask]
        diffuse_mask = noisy_batch[bp.diffuse_mask]
        chain_idx = noisy_batch[bp.chain_idx]
        motif_mask = noisy_batch.get(bp.motif_mask, None)
        # for inpainting, define `scaffold_mask` for sequence corruption using `1-motif_mask`.
        # we use `scaffold_mask` for sequence interpolation, and `diffuse_mask` for structure.
        # if `motif_mask` is missing or empty, effectively uses `diffuse_mask`.
        scaffold_mask = (1 - motif_mask) if motif_mask is not None else diffuse_mask

        # Pull out t_1 values
        trans_t_1 = noisy_batch[nbp.trans_t]
        rotmats_t_1 = noisy_batch[nbp.rotmats_t]
        aatypes_t_1 = noisy_batch[nbp.aatypes_t]
        torsions_t_1 = noisy_batch[nbp.torsions_t]

        # Gather protein state before interpolating (we already saved after previous step)
        prev_protein_state = SamplingStep.from_batch(
            batch=noisy_batch,
        )

        # Get model output given batch at time `t`
        with torch.no_grad():
            model_out = model(noisy_batch)
        pred_trans_1 = model_out[pbp.pred_trans]
        pred_rotmats_1 = model_out[pbp.pred_rotmats]
        pred_torsions_1 = model_out[pbp.pred_torsions]  # may be None
        pred_aatypes_1 = model_out[pbp.pred_aatypes]
        pred_logits_1 = model_out[pbp.pred_logits]

        model_pred = SamplingStep.from_model_prediction(
            pred=model_out,
            res_mask=res_mask,
        )

        # Mask fixed values to prepare for integration and update the batch for next step.
        if task == InferenceTask.unconditional:
            pass
        elif task == InferenceTask.inpainting:
            # Fix the logits / sequence outside of scaffold_mask / diffuse_mask.
            # i.e. if we have motifs, fix the sequence in the motifs.
            pred_logits_1 = mask_blend_2d(
                pred_logits_1, true_feats.logits, scaffold_mask
            )
            pred_aatypes_1 = mask_blend_1d(
                pred_aatypes_1, true_feats.aatypes, scaffold_mask
            )

            # Leave the torsions alone, let the model predict them,
            # since they mostly depend on the frames.
            if pred_torsions_1 is not None:
                pass

        elif task == InferenceTask.forward_folding:
            # scale logits during integration, assumes will `softmax`
            pred_logits_1 = 100.0 * true_feats.logits
            pred_aatypes_1 = true_feats.aatypes
        elif task == InferenceTask.inverse_folding:
            pred_trans_1 = true_feats.trans
            pred_rotmats_1 = true_feats.rotmats
            if pred_torsions_1 is not None:
                pred_torsions_1 = true_feats.torsions
        else:
            raise ValueError(f"Unknown task {task}")

        # Save the modified preditions
        protein_pred = SamplingStep.from_values(
            res_mask=res_mask,
            trans=pred_trans_1,
            rotmats=pred_rotmats_1,
            aatypes=pred_aatypes_1,
            torsions=pred_torsions_1,
            logits=None,
        )

        # For inpainting, compute a potential (scaled by `t=t_1`) to pull the motifs
        # (trans + rotmats) towards their known positions.
        # If not inpainting, motif_mask is None, and no guidance.
        motif_guidance = self.motif_potentials(
            t=t_1,
            noisy_batch=noisy_batch,
            pred=model_out,
            true_feats=true_feats,
            motif_mask=motif_mask,
        )

        # Feynman-Kac steering, if enabled.
        noisy_batch, resample_idx, step_metrics, potential_guidance = resampler.on_step(
            step_idx=step_idx,
            batch=noisy_batch,
            protein_state=prev_protein_state,
            protein_pred=protein_pred,
            model_pred=model_pred,
        )

        # add motif and potential guidance vector fields
        guidance = motif_guidance + potential_guidance

        # During sampling, update the self-conditioned values and take Euler steps for each domain.
        # On the final step, just use the cleaned up model predictions.
        if is_final_step:
            # Note we deviate from the convention in FrameFlow which does not mask with true values,
            # as we do above. Forcing them, like `forward_folding` or `inverse_folding`
            # seems to make sense if they are used for guidance and were fixed throughout.
            trans_t_2 = pred_trans_1
            rotmats_t_2 = pred_rotmats_1
            aatypes_t_2 = pred_aatypes_1
            torsions_t_2 = pred_torsions_1
        else:
            # Update self-conditioning values using model outputs.
            # Fixed domains depending on task have already been updated above.
            if self.cfg.self_condition:
                noisy_batch[nbp.trans_sc] = mask_blend_2d(
                    pred_trans_1, true_feats.trans, diffuse_mask
                )
                noisy_batch[nbp.aatypes_sc] = mask_blend_2d(
                    pred_logits_1, true_feats.logits, scaffold_mask
                )

            # Take next step, size `d_t` from `t_1` (current value) to `t_2` (toward predicted value)
            # We are at `t_1` with state `{domain}_t_1`. The model predicted `pred_{domain}_1`.
            # We use a shared `d_t` across domains, even though each may have its own `t` in the batch.
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t=d_t,
                t=t_1,
                trans_1=pred_trans_1,
                trans_t=trans_t_1,
                chain_idx=chain_idx,
                stochasticity_scale=stochasticity_scale,
                potential=guidance.trans,
            )
            rotmats_t_2 = self._rots_euler_step(
                d_t=d_t,
                t=t_1,
                rotmats_1=pred_rotmats_1,
                rotmats_t=rotmats_t_1,
                stochasticity_scale=stochasticity_scale,
                potential=guidance.rotmats,
            )
            aatypes_t_2 = self._aatypes_euler_step(
                d_t=d_t,
                t=t_1,
                logits_1=pred_logits_1,
                aatypes_t=aatypes_t_1,
                stochasticity_scale=stochasticity_scale,
                potential=guidance.logits,
            )
            torsions_t_2 = (
                self._torsions_euler_step(
                    d_t=d_t,
                    t=t_1,
                    torsions_1=pred_torsions_1,
                    torsions_t=torsions_t_1,
                    stochasticity_scale=stochasticity_scale,
                )
                if pred_torsions_1 is not None
                else None
            )

        # for inpainting, at all steps (including final step), motif sequence is fixed
        if task == InferenceTask.inpainting:
            # TODO(inpainting-fixed) to support fixed motifs, fix the t_2 structure motifs.
            # Fix the sequence in the motifs
            aatypes_t_2 = mask_blend_1d(aatypes_t_2, true_feats.aatypes, scaffold_mask)

        # Center diffused residues to maintain translation invariance
        # Definitely should center if inpainting or stochastic, but might as well if unconditional too
        # TODO(inpainting-fixed) keep fixed motifs centered, so condition remains the same (learn scaffold drift)
        #   The motifs vs structure @ t will have different centers of mass.
        trans_t_2 -= batch_center_of_mass(trans_t_2, mask=res_mask)[:, None]

        # update batch to t_2
        noisy_batch[nbp.trans_t] = (
            trans_t_2 if self.cfg.trans.corrupt else true_feats.trans
        )
        noisy_batch[nbp.rotmats_t] = (
            rotmats_t_2 if self.cfg.rots.corrupt else true_feats.rotmats
        )
        noisy_batch[nbp.aatypes_t] = (
            aatypes_t_2 if self.cfg.aatypes.corrupt else true_feats.aatypes
        )
        if torsions_t_2 is not None:
            noisy_batch[nbp.torsions_t] = torsions_t_2

        # Save protein state after interpolating
        protein_state = SamplingStep.from_batch(
            batch=noisy_batch,
        )

        # FK Steering particle selection, if enabled
        if resample_idx is not None:
            model_pred = model_pred.select_batch_idx(resample_idx)
            protein_state = protein_state.select_batch_idx(resample_idx)

        return noisy_batch, model_pred, protein_state, step_metrics

    def sample(
        self,
        num_batch: int,
        num_res: int,
        model,
        task: InferenceTask,
        diffuse_mask: torch.Tensor,
        motif_mask: Optional[torch.Tensor],
        chain_idx: torch.Tensor,
        res_idx: torch.Tensor,
        trans_0: Optional[torch.Tensor] = None,
        rotmats_0: Optional[torch.Tensor] = None,
        torsions_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
        trans_1: Optional[torch.Tensor] = None,
        rotmats_1: Optional[torch.Tensor] = None,
        torsions_1: Optional[torch.Tensor] = None,
        aatypes_1: Optional[torch.Tensor] = None,
        # t_nn is model/function that generates explicit time steps (r3, so3, cat) given t
        t_nn: Union[Callable[[torch.Tensor], torch.Tensor], Any] = None,
        # scale euler step stochasticity, 0 to disable for this `sample()`.
        # stochasticity for each domain must be enabled in cfg; this scales it those values
        stochasticity_scale: float = 1.0,
        structure_method: StructureExperimentalMethod = StructureExperimentalMethod.XRAY_DIFFRACTION,
        hot_spots: Optional[torch.Tensor] = None,
        contact_conditioning: Optional[torch.Tensor] = None,
        # Optional override for FK steering particles; set to 1 to disable
        num_particles: Optional[int] = None,
        # progress bar
        show_progress: bool = False,
    ) -> Tuple[SamplingTrajectory, SamplingTrajectory, FKSteeringTrajectory]:
        """
        Generate samples by interpolating towards model predictions.

        In theory, sampling is fairly straight-forward: integrate over a bunch of time steps.
        However, there is a lot of special handling to:
        - generate t=0 values if not provided
        - handling each task, setting up t=0 and t=1 values, and during interpolating
        - handling `codesign_separate_t` for fixed domains
        - supporting self-conditioning
        - saving each step of the trajectory (direct model output and integrated output)
        - special handling for the final timestep

        Returns three trajectories:
        1) sample_trajectory / predicted states - sampled trajectory.
            Intermediate steps resulting from integration over vector fields. Fixed values set by mask.
            No logits, just the amino acids (integration with rate matrix yields discrete sequence).
        2) model_trajectory / clean states - model predictions
            Without masking or added noise. Does not involve integrating. Includes logits.
        3) fk_trajectory - Feynman-Kac steering metrics trajectory
            FK steering metrics for each resampling step, tracking energy, weights, and effective sample size.

        Note that while sampling operates on backbones, the emitted structures are all-atom (i.e. atom37).

        The general process is:
        - generate the initial noisy batch, using optional inputs if provided
        - sample over time-steps
            - updating translations, rotations, and amino acid types at each time step
            - generate rigids from the updated frames
        - perform the final step after the loop
        """
        # task-specific input checks
        if task == InferenceTask.unconditional:
            assert self.cfg.trans.corrupt
            assert self.cfg.rots.corrupt
            assert self.cfg.aatypes.corrupt
            # no inputs required
        elif task == InferenceTask.inpainting:
            assert self.cfg.trans.corrupt
            assert self.cfg.rots.corrupt
            assert self.cfg.aatypes.corrupt
            # inputs
            assert trans_1 is not None
            assert rotmats_1 is not None
            assert torsions_1 is not None
            assert aatypes_1 is not None
            assert diffuse_mask is not None
            assert motif_mask is not None
        elif task == InferenceTask.forward_folding:
            assert self.cfg.trans.corrupt
            assert self.cfg.rots.corrupt
            assert not self.cfg.aatypes.corrupt
            # inputs
            assert aatypes_1 is not None
        elif task == InferenceTask.inverse_folding:
            assert not self.cfg.trans.corrupt
            assert not self.cfg.rots.corrupt
            assert self.cfg.aatypes.corrupt
            # inputs
            assert trans_1 is not None
            assert rotmats_1 is not None
            assert torsions_1 is not None
        else:
            raise ValueError(f"Unknown task {task}")

        # Set up Feynman-Kac steering + resampling for this sampling run
        resampler = FKSteeringResampler(
            cfg=self.cfg.steering, num_particles=num_particles
        )
        if resampler.enabled:
            assert (
                self.cfg.sampling.num_timesteps % resampler.cfg.resampling_interval == 0
            ), f"Number of sampling timesteps ({self.cfg.sampling.num_timesteps}) must be divisible by the steering resampling interval ({resampler.cfg.resampling_interval})"

        # fk_trajectory tracks FK steering metrics over time
        fk_trajectory = FKSteeringTrajectory(
            num_batch=num_batch,
            num_particles=resampler.num_particles,
        )

        # model_trajectory tracks model outputs
        model_trajectory = SamplingTrajectory(
            num_batch=num_batch,
            num_res=num_res,
            num_tokens=self.num_tokens,
        )
        # sample_trajectory tracks predicted intermediate states integrating from t=0 to t=1
        sample_trajectory = SamplingTrajectory(
            num_batch=num_batch,
            num_res=num_res,
            num_tokens=self.num_tokens,
        )

        # for inference, all residues under consideration
        res_mask = torch.ones(num_batch, num_res, device=self._device)
        # for inpainting, define `scaffold_mask` for sequence corruption using `1-motif_mask`
        scaffold_mask = (1 - motif_mask) if motif_mask is not None else diffuse_mask

        # set up empty self-conditioning values, to be filled during sampling.
        # if self-conditioning disabled these values will not be updated
        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device
        )

        # set up additional default values
        structure_method_tensor = (
            StructureExperimentalMethod.to_tensor(structure_method)
            .to(self._device)
            .expand(num_batch, 1)
        )
        if hot_spots is None:
            hot_spots = torch.zeros(num_batch, num_res, device=self._device)
        if contact_conditioning is None:
            contact_conditioning = torch.zeros(
                num_batch, num_res, num_res, device=self._device
            )

        # Set up batch. This batch object will be modified and be re-used for each time step.
        batch = {
            bp.res_mask: res_mask,
            bp.diffuse_mask: diffuse_mask,
            bp.chain_idx: chain_idx,
            bp.res_idx: res_idx,
            bp.structure_method: structure_method_tensor,
            bp.hot_spots: hot_spots,
            bp.contact_conditioning: contact_conditioning,
            nbp.trans_sc: trans_sc,
            nbp.aatypes_sc: aatypes_sc,
        }
        if motif_mask is not None:
            batch[bp.motif_mask] = motif_mask

        # Capture t=1 values for translations, rotations, torsions, and aatypes
        # Set to empty/identity values if they are not provided.
        true_feats = BatchTrueFeatures.from_optional(
            res_mask=res_mask,
            num_tokens=self.num_tokens,
            trans=trans_1,
            rotmats=rotmats_1,
            torsions=torsions_1,
            aatypes=aatypes_1,
        )

        # Initialize t=0 prior samples i.e. noise (technically `t=cfg.min_t`)

        if trans_0 is None:
            # Generate centered Gaussian noise / Harmonic prior for translations (B, N, 3)
            trans_0 = self._trans_noise(chain_idx=chain_idx, is_intermediate=False)
        if rotmats_0 is None:
            # Generate uniform SO(3) rotation matrices (B, N, 3, 3)
            rotmats_0 = self._rotmats_noise(res_mask=res_mask)
        if aatypes_0 is None:
            # Generate mask / random aa_types (B, N)
            aatypes_0 = self._aatypes_noise(res_mask=res_mask)
        if torsions_0 is None:
            # Generate initial torsion angles
            torsions_0 = self._torsions_noise(res_mask=res_mask)  # (B, N, 7, 2)

        # For inpainting, adjust motifs.
        if task == InferenceTask.inpainting:
            # The sequence is fixed to for partial sequence conditioning.
            aatypes_0 = mask_blend_1d(aatypes_0, true_feats.aatypes, scaffold_mask)

            # Translations and rotations will be interpolated, by passing `diffuse_mask == 1` to the network
            # (because `diffuse_mask == 0` residues are not updated by backbone update).

            # TODO(inpainting-fixed) depending on guidance type, set _1 values to t=1 using mask
            #    Enabling inpainting-fixed is mostly as simple as fixing the motifs and not diffusing the motifs.
            #    The main difficulty is centering and avoiding biasing the model as to where the scaffolds go.
            # trans_0 = mask_blend_2d(trans_0, true_feats.trans, diffuse_mask)
            # rotmats_0 = mask_blend_3d(rotmats_0, true_feats.rotmats, diffuse_mask)
            # torsions_0 = mask_blend_3d(torsions_0, true_feats.torsions, diffuse_mask)

        # check for NaNs, should only happen for invalid data or diffuse_mask
        if torch.isnan(trans_0).any():
            raise ValueError("NaN in trans_0")

        # Handle `codesign_separate_t` so some domains are fixed to t=1.
        if self.cfg.codesign_separate_t:
            if task == InferenceTask.unconditional:
                pass
            elif task == InferenceTask.inpainting:
                pass
            elif task == InferenceTask.forward_folding:
                aatypes_0 = true_feats.aatypes
            elif task == InferenceTask.inverse_folding:
                trans_0 = true_feats.trans
                rotmats_0 = true_feats.rotmats
                torsions_0 = true_feats.torsions
            else:
                raise ValueError(f"Unknown task {task}")

        # Set up initial t=0 values in batch
        batch[nbp.trans_t] = trans_0
        batch[nbp.rotmats_t] = rotmats_0
        batch[nbp.torsions_t] = torsions_0
        batch[nbp.aatypes_t] = aatypes_0

        # Set up Feynman-Kac resampling, which expands each batch member to K particles.
        # Idempotent if not enabled or number of particles is 1.
        batch = resampler.init_particles(batch=batch)  # (B, ...) -> (B * K, ...)
        num_batch = batch[bp.res_mask].shape[0]  # may have changed

        # save t=0 state
        sample_trajectory.append(SamplingStep.from_batch(batch=batch))

        # Set-up time steps
        ts = torch.linspace(
            self.cfg.min_t, 1.0, self.cfg.sampling.num_timesteps, device=self._device
        )
        step_idx = 0

        # We will integrate in a loop over ts, handling the last step after the loop.
        # t_1 is the current time (handle updating ourselves at end of loop).
        # t_2 is the next time.
        t_1 = ts[0]
        for t_2 in tqdm(
            ts[1:], desc="Sampling timestep", disable=not show_progress, leave=False
        ):
            # Determine time for each domain
            t = torch.ones((num_batch, 1), device=self._device) * t_1.to(self._device)
            if t_nn is not None:
                (
                    batch[nbp.r3_t],
                    batch[nbp.so3_t],
                    batch[nbp.cat_t],
                ) = torch.split(t_nn(t), -1)
            else:
                if self.cfg.provide_kappa:
                    batch[nbp.so3_t] = self._rot_sample_kappa(t)
                else:
                    batch[nbp.so3_t] = t
                batch[nbp.r3_t] = t
                batch[nbp.cat_t] = t

            # If `codesign_separate_t`, fixed domains set to t=1-min_t (penultimate step, take final step later)
            if self.cfg.codesign_separate_t:
                # TODO consider setting to t=1 to match how `codesign_separate_t` sets values in `corrupt_batch()`
                t_minus_1 = (1 - self.cfg.min_t) * torch.ones(
                    (num_batch, 1), device=self._device
                )
                if task == InferenceTask.unconditional:
                    pass
                elif task == InferenceTask.inpainting:
                    pass
                elif task == InferenceTask.forward_folding:
                    batch[nbp.cat_t] = t_minus_1
                elif task == InferenceTask.inverse_folding:
                    batch[nbp.r3_t] = t_minus_1
                    batch[nbp.so3_t] = t_minus_1
                else:
                    raise ValueError(f"Unknown task {task}")

            # Take a single step, updating the batch in place
            batch, model_step, sample_step, step_metrics = self.sample_single_step(
                noisy_batch=batch,
                true_feats=true_feats,
                model=model,
                task=task,
                resampler=resampler,
                step_idx=step_idx,
                t_1=t_1,
                t_2=t_2,
                stochasticity_scale=stochasticity_scale,
            )

            model_trajectory.append(model_step)
            sample_trajectory.append(sample_step)
            fk_trajectory.append(step_metrics)

            # Update t_1 to t_2 for the next step
            t_1 = t_2
            step_idx += 1

        # We only integrated to ts[-1], so need to make a final step

        # update times to final `t`
        assert t_1 == ts[-1]
        t = torch.ones((num_batch, 1), device=self._device) * t_1
        batch[nbp.so3_t] = t
        batch[nbp.r3_t] = t
        batch[nbp.cat_t] = t

        batch, model_step, sample_step, step_metrics = self.sample_single_step(
            noisy_batch=batch,
            true_feats=true_feats,
            model=model,
            task=task,
            resampler=resampler,
            step_idx=self.cfg.sampling.num_timesteps,  # final step
            t_1=t_1,
            t_2=None,  # final step
            stochasticity_scale=stochasticity_scale,
        )

        model_trajectory.append(model_step)
        sample_trajectory.append(sample_step)
        fk_trajectory.append(step_metrics)

        # If FK steering is enabled, pick the best particle per sample
        # TODO(fksteering) - allow passing through all the particles.
        #   Ensure each is handled properly.
        #   Right now we make some assumptions and build directory structure before calling sample().
        _, best_idx = resampler.best_particle_in_batch(batch=batch)
        if best_idx is not None:
            # Select the best particle for each sample
            model_trajectory = model_trajectory.select_batch_idx(best_idx)
            sample_trajectory = sample_trajectory.select_batch_idx(best_idx)

        return sample_trajectory, model_trajectory, fk_trajectory
