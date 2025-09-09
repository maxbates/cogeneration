import copy
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from tqdm.auto import tqdm

from cogeneration.config.base import (
    InterpolantAATypesInterpolantTypeEnum,
    InterpolantAATypesScheduleEnum,
    InterpolantConfig,
    InterpolantRotationsScheduleEnum,
    InterpolantTrainTimeSamplingEnum,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX
from cogeneration.data.fm.aatypes import (
    FlowMatcherAATypes,
    FlowMatcherAATypesMasking,
    FlowMatcherAATypesUniform,
)
from cogeneration.data.fm.rotations import FlowMatcherRotations
from cogeneration.data.fm.torsions import FlowMatcherTorsions
from cogeneration.data.fm.translations import FlowMatcherTrans
from cogeneration.data.logits import combine_logits
from cogeneration.data.noise_mask import (
    mask_blend_1d,
    mask_blend_2d,
    mask_blend_3d,
    masked_categorical,
    torsions_empty,
    uniform_categorical,
    uniform_so3,
)
from cogeneration.data.potentials import (
    FKSteeringResampler,
    FKSteeringTrajectory,
    FKStepMetric,
    PotentialField,
)
from cogeneration.data.rigid import batch_center_of_mass
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
    Interpolant is responsible for coordinating:
    - generating noise
    - corrupting samples
    - sampling from learned vector fields

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

        self._device = torch.device("cpu")

        # Flow matchers per domain
        self.trans_fm = FlowMatcherTrans(cfg=self.cfg.trans)
        self.torsions_fm = FlowMatcherTorsions(cfg=self.cfg.torsions)
        self.rots_fm = FlowMatcherRotations(cfg=self.cfg.rots)

        # Instantiate AAtypes flow matcher according to interpolant type
        if (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            self.aatypes_fm = FlowMatcherAATypesMasking(cfg=self.cfg.aatypes)
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            self.aatypes_fm = FlowMatcherAATypesUniform(cfg=self.cfg.aatypes)
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

    def set_device(self, device: torch.device):
        self._device = device

        # propagate device to flow matchers
        self.trans_fm.set_device(device)
        self.torsions_fm.set_device(device)
        self.rots_fm.set_device(device)
        self.aatypes_fm.set_device(device)

    @property
    def num_tokens(self):
        return self.aatypes_fm.num_tokens

    def sample_t(self, num_batch: int) -> torch.Tensor:
        """
        Sample `t` in the range [min_t, 1-min_t], for corrupting / training a batch.
        returns (B,) tensor
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
                np.random.choice([0, 1, 2], size=num_batch, p=[0.6, 0.25, 0.15]),
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
        return t * (1 - 2 * min_t) + min_t  # (B,)

    def _set_corruption_times(
        self,
        noisy_batch: NoisyFeatures,
        task: DataTask,
    ):
        """
        Sets `t` times (B,) for each domain, i.e. `bp.so3_t`, `bp.r3_t`, `bp.cat_t`, and potentially modify masks.
        """
        res_mask = noisy_batch[bp.res_mask]  # (B, N)
        num_batch, num_res = res_mask.shape

        if self.cfg.codesign_separate_t:
            u = torch.rand((num_batch,), device=self._device)

            # Sample t values for normal structure and categorical corruption.
            # In `codesign_separate_t` each has its own `t`, separate from fixed domains set to t=1.
            normal_structure_t = self.sample_t(num_batch)  # (B,)
            normal_cat_t = self.sample_t(num_batch)  # (B,)
            ones_t = torch.ones((num_batch,), device=self._device)  # (B,)

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

            so3_t = structure_t
            r3_t = structure_t
            cat_t = cat_t

        # Default: single `t` time is shared by each domain
        # Note: for inpainting, it could make sense to pick an intermediate t depending on how many residues are defined
        else:
            t = self.sample_t(num_batch)
            so3_t = t
            r3_t = t
            cat_t = t

        noisy_batch[nbp.so3_t] = so3_t  # (B,)
        noisy_batch[nbp.r3_t] = r3_t  # (B,)
        noisy_batch[nbp.cat_t] = cat_t  # (B,)

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
        r3_t = noisy_batch[nbp.r3_t]  # (B,)
        so3_t = noisy_batch[nbp.so3_t]  # (B,)
        cat_t = noisy_batch[nbp.cat_t]  # (B,)
        stochasticity_scale = batch.get(
            bp.stochastic_scale, torch.ones(res_mask.shape[0], device=trans_1.device)
        )  # (B,)

        # Determine sequence and structure corruption masks.
        # Inpainting:
        #   The motif sequence is fixed. However, the structure is not, and t should be in sync across domains.
        #   With guidance: The motifs are explicitly interpolated over time, so we corrupt the entire structure.
        #   Fixed motifs: The diffuse_mask is only for the scaffolds and the motifs are fixed at t=1
        # For other tasks, everything is corrupted i.e. `(diffuse_mask == 1.0).all()`
        #   Though values at t=1 effectively won't be corrupted.
        scaffold_mask = (1 - motif_mask) if motif_mask is not None else diffuse_mask

        # Apply corruptions

        if self.cfg.trans.corrupt:
            trans_t = self.trans_fm.corrupt(
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
            rotmats_t = self.rots_fm.corrupt(
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
            torsions_t = self.torsions_fm.corrupt(
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
            aatypes_t = self.aatypes_fm.corrupt(
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

    def motif_potentials(
        self,
        t: torch.Tensor,  # (B,)
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
        scale_trans = 0.5 * g_trans * g_trans / omega2_trans  # (B,)

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
        scale_rotmats = 0.5 * g_rotmats * g_rotmats / omega2_rotmats  # (B,)

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
        trans_vf *= scale_trans.view(-1, 1, 1)

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
        rotmats_vf *= scale_rotmats.view(-1, 1, 1)

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
        t_1: torch.Tensor,  # (B,)
        d_t: Optional[torch.Tensor],  # scalar or None if final step
        stochasticity_scale: torch.Tensor,  # (B,)
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

        is_final_step = d_t is None

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

            # Take next step of size `d_t` from `t_1` toward the next time `t_2` for each domain
            trans_t_2 = self.trans_fm.euler_step(
                d_t=d_t,
                t=t_1,
                trans_1=pred_trans_1,
                trans_t=trans_t_1,
                chain_idx=chain_idx,
                stochasticity_scale=stochasticity_scale,
                potential=guidance.trans,
            )
            rotmats_t_2 = self.rots_fm.euler_step(
                d_t=d_t,
                t=t_1,
                rotmats_1=pred_rotmats_1,
                rotmats_t=rotmats_t_1,
                stochasticity_scale=stochasticity_scale,
                potential=guidance.rotmats,
            )
            aatypes_t_2 = self.aatypes_fm.euler_step(
                d_t=d_t,
                t=t_1,
                logits_1=pred_logits_1,
                aatypes_t=aatypes_t_1,
                stochasticity_scale=stochasticity_scale,
                potential=guidance.logits,
            )
            torsions_t_2 = (
                self.torsions_fm.euler_step(
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

        # Build per-sample stochasticity tensor (B,)
        if isinstance(stochasticity_scale, torch.Tensor):
            stochastic_scale_tensor = stochasticity_scale.to(self._device).view(-1)
            if stochastic_scale_tensor.shape[0] != num_batch:
                stochastic_scale_tensor = stochastic_scale_tensor.expand(num_batch)
        else:
            stochastic_scale_tensor = (
                torch.tensor([float(stochasticity_scale)], device=self._device)
                .expand(num_batch)
                .float()
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
            bp.stochastic_scale: stochastic_scale_tensor,
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
            trans_0 = self.trans_fm.sample_base(
                chain_idx=chain_idx, is_intermediate=False
            )
        if rotmats_0 is None:
            # Generate uniform SO(3) rotation matrices (B, N, 3, 3)
            rotmats_0 = self.rots_fm.sample_base(res_mask=res_mask)
        if aatypes_0 is None:
            # Generate mask / random aa_types (B, N)
            aatypes_0 = self.aatypes_fm.sample_base(res_mask=res_mask)
        if torsions_0 is None:
            # Generate initial torsion angles
            torsions_0 = self.torsions_fm.sample_base(res_mask=res_mask)  # (B, N, 7, 2)

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

        # t_1 (scalar) is the current time (handle updating ourselves at end of loop).
        # t_2 (scalar) is the next time.
        # d_t (scalar) = t_2 - t_1  is the delta.
        t_1 = ts[0]
        for t_2 in tqdm(
            ts[1:], desc="Sampling timestep", disable=not show_progress, leave=False
        ):
            # Determine time for each domain
            t_1_b = torch.ones((num_batch,), device=self._device) * t_1  # (B,)
            if t_nn is not None:
                (
                    batch[nbp.r3_t],
                    batch[nbp.so3_t],
                    batch[nbp.cat_t],
                ) = torch.split(t_nn(t_1_b), -1)
            else:
                if self.cfg.provide_kappa:
                    batch[nbp.so3_t] = self._rot_sample_kappa(t_1_b)
                else:
                    batch[nbp.so3_t] = t_1_b
                batch[nbp.r3_t] = t_1_b
                batch[nbp.cat_t] = t_1_b

            # If `codesign_separate_t`, fixed domains set to t=1-min_t (penultimate step, take final step later)
            if self.cfg.codesign_separate_t:
                # TODO consider setting to t=1 to match how `codesign_separate_t` sets values in `corrupt_batch()`
                t_minus_1 = torch.ones((num_batch,), device=self._device) * (
                    1 - self.cfg.min_t
                )  # (B,)
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
                t_1=t_1_b,
                d_t=(t_2 - t_1),
                stochasticity_scale=stochastic_scale_tensor,
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
        t_1_b = torch.ones((num_batch,), device=self._device) * t_1
        batch[nbp.so3_t] = t_1_b
        batch[nbp.r3_t] = t_1_b
        batch[nbp.cat_t] = t_1_b

        batch, model_step, sample_step, step_metrics = self.sample_single_step(
            noisy_batch=batch,
            true_feats=true_feats,
            model=model,
            task=task,
            resampler=resampler,
            step_idx=self.cfg.sampling.num_timesteps,  # final step
            t_1=t_1_b,
            d_t=None,  # final step
            stochasticity_scale=stochastic_scale_tensor,
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
