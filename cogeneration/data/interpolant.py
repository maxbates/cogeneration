import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # noqa
from torch import nn

from cogeneration.config.base import (
    InterpolantAATypesInterpolantTypeEnum,
    InterpolantConfig,
    InterpolantRotationsScheduleEnum,
    InterpolantTrainTimeSamplingEnum,
    InterpolantTranslationsNoiseTypeEnum,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX, NM_TO_ANG_SCALE
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
)
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass
from cogeneration.data.trajectory import SamplingStep, SamplingTrajectory
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProp as bp
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


class Interpolant(nn.Module):
    """
    Interpolant is responsible for generating noise, corrupting samples, and sampling from learned vector fields.

    It has two almost-but-not-quite separate roles:
    (1) corrupt batches with noise, generating intermediate samples at some time `t`
    (2) generates samples, interpolating each modality using the learned vector fields over t=[0, 1] from noise to sample.

    Works across multiple domains: translations and rotations (i.e. backbone frames), torsion angles, and amino acid types (i.e. sequence).

    (1) Translations
    - in R^3 and are Euclidean
    - Noise is Gaussian
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
        super().__init__()
        self.cfg = cfg

        self.resampler = FKSteeringResampler(cfg=self.cfg.steering)
        if self.resampler.enabled:
            assert (
                self.cfg.sampling.num_timesteps % self.resampler.cfg.resampling_interval
                == 0
            ), f"Number of sampling timesteps ({self.cfg.sampling.num_timesteps}) must be divisible by the steering resampling interval ({self.resampler.cfg.resampling_interval})"

        self._igso3 = None
        self._device = None

        is_masking_interpolant = (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        )
        self.num_tokens = 21 if is_masking_interpolant else 20

    @property
    def igso3(self):
        # On CPU
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, self.cfg.rots.igso3_sigma, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

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
                np.random.choice([0, 1, 2], size=num_batch, p=[0.3, 0.45, 0.25]),
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

    def _compute_sigma_t(self, t: torch.Tensor, scale: float, min_sigma: float = 0.0):
        """
        Compute the standard deviation of the noise at time `t` (B,)
        The standard deviation is a parabolic function of t, and is zero at t=0 and t=1.
        """
        return torch.sqrt(scale**2 * t * (1 - t) + min_sigma**2)  # (B,)

    def _trans_noise(self, chain_idx: torch.Tensor) -> torch.Tensor:
        """
        sample t=0 noise for translations, scaled to angstroms
        """
        if (
            self.cfg.trans.noise_type
            == InterpolantTranslationsNoiseTypeEnum.centered_gaussian
        ):
            return (
                centered_gaussian(
                    *chain_idx.shape,
                    device=self._device,
                )
                * NM_TO_ANG_SCALE
            )
        elif (
            self.cfg.trans.noise_type
            == InterpolantTranslationsNoiseTypeEnum.centered_harmonic
        ):
            return (
                centered_harmonic(
                    chain_idx=chain_idx,
                    device=self._device,
                )
                * NM_TO_ANG_SCALE
            )
        else:
            raise ValueError(
                f"Unknown translation noise type {self.cfg.trans.noise_type}"
            )

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
            return masked_categorical(num_batch, num_res, device=self._device)
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
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
        trans_0 = self._trans_noise(chain_idx=chain_idx)  # (B, N, 3)

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
            intermediate_noise = self._trans_noise(chain_idx=chain_idx)  # (B, N, 3)
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
        sigma = torch.tensor([self.cfg.rots.igso3_sigma])
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
            sigma_t = sigma_t.cpu()  # ensure on cpu for igso3 calculation
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

    def _corrupt_aatypes(
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
            jump_mask = (
                torch.rand(num_batch, num_res, device=self._device) < p_jump
            ).int()

            if jump_mask.any():
                jump_noise = self._aatypes_noise(res_mask=res_mask)
                aatypes_t = mask_blend_1d(jump_noise, aatypes_t, jump_mask)

        # residues outside `res_mask` are set to mask regardless of aatype noise strategy.
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

    def _rot_sample_kappa(self, t: torch.Tensor):
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            return 1 - torch.exp(-t * self.cfg.rots.exp_rate)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            return t
        else:
            raise ValueError(f"Invalid schedule: {self.cfg.rots.sample_schedule}")

    def _trans_vector_field(
        self, t: float, trans_1: torch.Tensor, trans_t: torch.Tensor
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
        d_t: float,
        t: float,
        trans_1: torch.Tensor,  # (B, N, 3)
        trans_t: torch.Tensor,  # (B, N, 3)
        chain_idx: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        trans_next = trans_t + trans_vf * d_t

        if (
            self.cfg.trans.stochastic
            and self.cfg.trans.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # Sample from either Gaussian or Harmonic prior (zero‐mean, correct covariance)
            # For harmonic noise, each Brownian increment has covariance J^{-1} (the harmonic prior) instead of identity.
            base_noise = self._trans_noise(
                chain_idx=chain_idx
            )  # (B, N, 3) in Angstroms
            sigma_t = self._compute_sigma_t(
                torch.ones(trans_1.shape[0]) * t,  # (B)
                scale=self.cfg.trans.stochastic_noise_intensity * stochasticity_scale,
            )
            sigma_t = sigma_t.to(trans_next.device)
            trans_next += base_noise * math.sqrt(d_t) * sigma_t[..., None, None]

        return trans_next

    def _rots_euler_step(
        self,
        d_t: float,
        t: float,
        rotmats_1: torch.Tensor,  # (B, N, 3, 3)
        rotmats_t: torch.Tensor,  # (B, N, 3, 3)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            scaling = 1 / (1 - t)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            scaling = self.cfg.rots.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self.cfg.rots.sample_schedule}")

        rotmats_next = so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

        if (
            self.cfg.rots.stochastic
            and self.cfg.rots.stochastic_noise_intensity > 0.0
            and stochasticity_scale > 0.0
        ):
            # Brownian increment: σ(t) · √dt with σ(t)=intensity·√(t(1–t))
            # Sample IGSO(3) noise with a time-independent sigma_t, scaled by sqrt(dt)
            # Add IGSO(3) noise to stay on SO(3).
            num_batch, num_res, _, _ = rotmats_t.shape

            sigma_t = self._compute_sigma_t(
                torch.ones(num_batch) * t,  # (B)
                scale=self.cfg.rots.stochastic_noise_intensity * stochasticity_scale,
            ) * math.sqrt(
                d_t
            )  # (B,)
            sigma_t = sigma_t.cpu()  # ensure on cpu for igso3 calculation
            intermediate_noise = self.igso3.sample(sigma_t, num_res)
            intermediate_noise = intermediate_noise.to(rotmats_t.device)
            intermediate_noise = intermediate_noise.reshape(num_batch, num_res, 3, 3)
            rotmats_next = torch.einsum(
                "...ij,...jk->...ik", rotmats_next, intermediate_noise
            )

        return rotmats_next

    def _torsions_euler_step(
        self,
        d_t: float,
        t: float,
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
        step_probs: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
    ):
        """
        Regularize the step probabilities to ensure they conform to the requirements of a rate matrix.
        The rate matrix we learn at `t` needs each row to sum to zero.
        """
        num_batch, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (num_batch, num_res)

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        batch_idx = torch.arange(num_batch, device=device).repeat_interleave(num_res)
        residue_idx = torch.arange(num_res, device=device).repeat(num_batch)
        curr_states = aatypes_t.long().flatten()

        # set the probabilities corresponding to the current amino acid types to 0.0
        step_probs[batch_idx, residue_idx, curr_states] = 0.0

        # adjust the probabilities at the current positions to be the negative sum of all other values in the row
        row_sums = torch.sum(step_probs, dim=-1).flatten()
        step_probs[batch_idx, residue_idx, curr_states] = 1.0 - row_sums

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        # in case negative or out-of-bound values appear after the diagonal assignment.
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        return step_probs

    def _aatypes_euler_step_uniform(
        self,
        d_t: float,
        t: float,
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
        temp = self.cfg.aatypes.temp
        noise = self.cfg.aatypes.noise

        # convert logits to probabilities
        pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

        # probability of x1 matching xt exactly
        pt_x1_eq_xt_prob = torch.gather(
            pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)
        )  # (B, D, 1)
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
        d_t: float,
        t: float,
        logits_1: torch.Tensor,  # (B, N, S=21)
        aatypes_t: torch.Tensor,  # (B, N)
    ):
        num_batch, num_res, num_states = logits_1.shape
        assert num_states == 21
        assert aatypes_t.shape == (num_batch, num_res)

        device = logits_1.device
        temp = self.cfg.aatypes.temp
        noise = self.cfg.aatypes.noise

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
        d_t: float,
        t: float,
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
        temp = self.cfg.aatypes.temp
        noise = self.cfg.aatypes.noise

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
        unmask_probs = (d_t * ((1.0 + noise * t) / (1.0 - t))).clamp(max=1.0)  # scalar
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
        d_t: float,
        t: float,  # t in [0,1]
        logits_1: torch.Tensor,  # (B, N, S)
        aatypes_t: torch.Tensor,  # (B, N)
        stochasticity_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Stochastic CTMC jump for aatypes.

        Uses rate matrix to allow sequnce to explore neighboring states
        in proportion to the rates the network thinks are possible.
        So notice that unlike in training, where jumps are to a uniform-random sampled state,
        here the jumps are to states the network thinks are plausible.
        The thinking is that during training, the model must learn a broad "restoring drift" and uniform sampling is class balanced.
        (We can't use logits during training without simulation, they are just a one-hot of the sequence.)
        However, this training may slow convergence, or the model may only learn to fix very wrong residues.

        This is different than the `noise` term, which adds noise to the rate matrix in determinsitic interpolation.
        """
        B, N, S = logits_1.shape
        assert aatypes_t.shape == (B, N)

        device = logits_1.device

        # logits -> probabilities
        prob_rows = F.softmax(
            logits_1 / self.cfg.aatypes.stochastic_temp, dim=-1
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
            torch.ones(aatypes_t.shape[0], device=device) * t,
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

    def _aatypes_euler_step(
        self,
        d_t: float,
        t: float,  # t in [0,1]
        logits_1: torch.Tensor,  # (B, N, S) unscaled probabilities, S={20, 21}
        aatypes_t: torch.Tensor,  # (B, N) current amino acid types
        stochasticity_scale: float = 1.0,
    ):
        """
        Perform an Euler step to update amino acid types based on the provided logits and interpolation settings.

        This function handles two interpolation strategies:
        1. "masking": Updates the amino acid types by masking certain positions and sampling new types
           based on modified probabilities. Assumes S = 21 to include a special MASK token.
        2. "uniform": Samples new amino acid types uniformly, with the assumption that no MASK tokens are involved.
           Assumes S = 20.
        """
        if self.cfg.aatypes.do_purity:
            aatypes_t = self._aatypes_euler_step_purity(d_t, t, logits_1, aatypes_t)
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            aatypes_t = self._aatypes_euler_step_masking(
                d_t=d_t, t=t, logits_1=logits_1, aatypes_t=aatypes_t
            )
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            aatypes_t = self._aatypes_euler_step_uniform(
                d_t=d_t, t=t, logits_1=logits_1, aatypes_t=aatypes_t
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

    def sample_single_step(
        self,
        noisy_batch: NoisyFeatures,
        true_feats: BatchTrueFeatures,
        model,
        task: InferenceTask,
        step_idx: int,
        t_1: float,  # [scalar Tensor]
        t_2: Optional[float],  # [scalar Tensor] None if final step
        stochasticity_scale: float = 1.0,
    ) -> Tuple[NoisyFeatures, SamplingStep, SamplingStep, Optional[FKStepMetric]]:
        """
        Perform a single step of sampling, integrating from `t_1` toward `t_2`.
        Batch `_t` properties should be at `t_1`.

        Returns a batch with noisy features updated to t_2, and model + protein intermediate states.

        Note on torsions:
        Currently, torsions are not an input to the model.
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
        # for inpainting, define `scaffold_mask` for sequence corruption using `1-motif_mask`
        motif_mask = noisy_batch.get(bp.motif_mask, None)
        scaffold_mask = (1 - motif_mask) if motif_mask is not None else diffuse_mask

        # Pull out t_1 values
        trans_t_1 = noisy_batch[nbp.trans_t]
        rotmats_t_1 = noisy_batch[nbp.rotmats_t]
        aatypes_t_1 = noisy_batch[nbp.aatypes_t]
        torsions_t_1 = noisy_batch[nbp.torsions_t]

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
            # fix the logits / sequence
            pred_logits_1 = mask_blend_2d(
                pred_logits_1, true_feats.logits, scaffold_mask
            )
            pred_aatypes_1 = mask_blend_1d(
                pred_aatypes_1, true_feats.aatypes, scaffold_mask
            )

            # TODO(inpainting) - consider whether we should first align (i.e. rotate)
            #   the pred structure to known motifs, rather than simply substituting them in.
            #   I don't think FrameFlow guidance did this, but the twisting variant sort of does?

            # For inpainting with guidance, fix known motifs of predicted structure
            # so that we interpolate toward the known motifs. Also for self-conditioning.
            pred_trans_1 = mask_blend_2d(pred_trans_1, true_feats.trans, scaffold_mask)
            pred_rotmats_1 = mask_blend_3d(
                pred_rotmats_1, true_feats.rotmats, scaffold_mask
            )
            if pred_torsions_1 is not None:
                pred_torsions_1 = mask_blend_3d(
                    pred_torsions_1, true_feats.torsions, scaffold_mask
                )
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
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(
                d_t=d_t,
                t=t_1,
                trans_1=pred_trans_1,
                trans_t=trans_t_1,
                chain_idx=chain_idx,
                stochasticity_scale=stochasticity_scale,
            )
            rotmats_t_2 = self._rots_euler_step(
                d_t=d_t,
                t=t_1,
                rotmats_1=pred_rotmats_1,
                rotmats_t=rotmats_t_1,
                stochasticity_scale=stochasticity_scale,
            )
            aatypes_t_2 = self._aatypes_euler_step(
                d_t=d_t,
                t=t_1,
                logits_1=pred_logits_1,
                aatypes_t=aatypes_t_1,
                stochasticity_scale=stochasticity_scale,
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

            if task == InferenceTask.inpainting:
                # Because we already set the motif positions in pred_{trans/rots}_1,
                # we have already interpolated towards them as guidance.
                # TODO(inpainting-fixed) to support fixed motifs, fix the t_2 structure motifs.

                # Sequence is fixed for motifs at t=1
                aatypes_t_2 = mask_blend_1d(
                    aatypes_t_2, true_feats.aatypes, scaffold_mask
                )

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

        # Feynman-Kac steering, if enabled.
        # TODO(FK) better expose metrics, to track in trajectory over time
        # TODO(FK) Determine if we should expose model states for un-selected particles
        noisy_batch, resample_idx, step_metrics = self.resampler.on_step(
            step_idx=step_idx,
            batch=noisy_batch,
            protein_pred=protein_pred,
            model_pred=model_pred,
            protein_state=protein_state,
        )
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
        # stochasticity must be enabled in cfg, this only scales it per domain
        stochasticity_scale: float = 1.0,
        structure_method: StructureExperimentalMethod = StructureExperimentalMethod.XRAY_DIFFRACTION,
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
            assert (
                self.cfg.aatypes.noise == 0.0
            )  # cfg check unnecessary if not corrupting?
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
        # fk_trajectory tracks Feynman-Kac steering metrics over time
        fk_trajectory = FKSteeringTrajectory(
            num_batch=num_batch,
            num_particles=self.cfg.steering.num_particles,
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

        # Set up batch. This batch object will be modified and be re-used for each time step.
        batch = {
            bp.res_mask: res_mask,
            bp.diffuse_mask: diffuse_mask,
            bp.chain_idx: chain_idx,
            bp.res_idx: res_idx,
            bp.structure_method: StructureExperimentalMethod.to_tensor(structure_method)
            .to(self._device)
            .expand(num_batch, 1),
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
            # Generate centered Gaussian noise for translations (B, N, 3)
            trans_0 = self._trans_noise(chain_idx=chain_idx)
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
        batch = self.resampler.init_particles(batch=batch)  # (B, ...) -> (B * K, ...)
        num_batch = batch[bp.res_mask].shape[0]  # may have changed

        # save t=0 state
        sample_trajectory.append(SamplingStep.from_batch(batch=batch))

        # Set-up time steps
        ts = torch.linspace(self.cfg.min_t, 1.0, self.cfg.sampling.num_timesteps)
        step_idx = 0

        # We will integrate in a loop over ts, handling the last step after the loop.
        # t_1 is the current time (handle updating ourselves at end of loop).
        # t_2 is the next time.
        t_1 = ts[0]
        for t_2 in ts[1:]:
            # Determine time for each domain
            t = torch.ones((num_batch, 1), device=self._device) * t_1
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
        _, best_idx = self.resampler.best_particle_in_batch(batch=batch)
        if best_idx is not None:
            # Select the best particle for each sample
            model_trajectory = model_trajectory.select_batch_idx(best_idx)
            sample_trajectory = sample_trajectory.select_batch_idx(best_idx)

        return sample_trajectory, model_trajectory, fk_trajectory
