import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from random import random
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment  # noqa

from cogeneration.config.base import (
    InterpolantAATypesInterpolantTypeEnum,
    InterpolantConfig,
    InterpolantRotationsScheduleEnum,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data import all_atom, so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX, NM_TO_ANG_SCALE, NUM_TOKENS
from cogeneration.data.noise_mask import (
    centered_gaussian,
    mask_blend_1d,
    mask_blend_2d,
    mask_blend_3d,
    masked_categorical,
    uniform_categorical,
    uniform_so3,
)
from cogeneration.data.rigid import batch_align_structures, batch_center_of_mass
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProps as bp
from cogeneration.type.batch import NoisyBatchProps as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.batch import PredBatchProps as pbp
from cogeneration.type.task import DataTaskEnum, InferenceTaskEnum


def to_cpu(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    return x.detach().cpu()


@dataclass
class SamplingStep:
    """
    A single step in the sampling trajectory.

    tensors should be detached and moved to CPU.
    """

    structure: torch.Tensor  # (batch_size, num_res, 37, 3)  i.e. atom37 representation
    amino_acids: torch.Tensor  # (batch_size, num_res)
    logits: Optional[torch.Tensor] = None


@dataclass
class SamplingTrajectory:
    """
    A trajectory of inference sampling steps.
    """

    num_batch: int
    num_res: int
    num_tokens: int
    steps: List[SamplingStep] = field(default_factory=list)
    check_dimensions: bool = True

    def __len__(self):
        return len(self.steps)

    def __getitem__(self, index: int) -> SamplingStep:
        return self.steps[index]

    def append(self, step: SamplingStep):
        self.steps.append(step)

    @property
    def num_steps(self):
        return len(self.steps)

    @property
    def structure(self) -> torch.Tensor:
        """
        Returns structure / backbone tensor [num_batch, traj_length, sample_length, 37, 3]
        """
        t = torch.stack([step.structure for step in self.steps], dim=0).transpose(0, 1)
        if self.check_dimensions:
            expected_shape = (self.num_batch, self.num_steps, self.num_res, 37, 3)
            assert (
                t.shape == expected_shape
            ), f"Unexpected structure shape {t.shape}, expected {expected_shape}"
        return t

    @property
    def amino_acids(self) -> torch.Tensor:
        """
        Returns amino acid types tensor [num_batch, traj_length, sample_length]
        """
        t = (
            torch.stack([step.amino_acids for step in self.steps], dim=0)
            .transpose(0, 1)
            .long()
        )
        if self.check_dimensions:
            expected_shape = (self.num_batch, self.num_steps, self.num_res)
            assert (
                t.shape == expected_shape
            ), f"Unexpected amino_acids shape {t.shape}, expected {expected_shape}"
        return t

    @property
    def logits(self) -> Optional[torch.Tensor]:
        """
        Returns logits tensor if available [num_batch, traj_length, sample_length, num_tokens]
        Currently only available for model steps, not protein steps, since we don't flow for logits.
        """
        if self.steps[0].logits is None:
            return None
        t = torch.stack([step.logits for step in self.steps], dim=0).transpose(0, 1)
        if self.check_dimensions:
            expected_shape = (
                self.num_batch,
                self.num_steps,
                self.num_res,
                self.num_tokens,
            )
            assert (
                t.shape == expected_shape
            ), f"Unexpected logits shape {t.shape}, expected {expected_shape}"
        return t


class Interpolant:
    """
    Interpolant is responsible for generating noise, corrupting samples, and sampling from learned vector fields.

    It has two almost-but-not-quite separate roles:
    (1) corrupt batches with noise, generating intermediate samples at some time `t`
    (2) generates samples, interpolating each modality using the learned vector fields over t=[0, 1] from noise to sample.

    Works across 3 modalties: translations and rotations (i.e. backbone frames), and amino acid types (i.e. sequence).

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
    (3) Amino acid types
    - are discrete
    - over either uniform distribution (n=20) or with masks (n=21)
    - supports Euler steps using learned rate matrix (discrete flow matching)
    - supports "purity" sampling, i.e. masking by max log probs and re-masking noise

    (Frames a.k.a. Rigids are therefore in SE(3) = R^3 x SO(3))
    psi torsion angles are not really considered by the interpolant, except when output by the model to generate Rigids.
    """

    def __init__(self, cfg: InterpolantConfig):
        self.cfg = cfg
        self._igso3 = None
        self._device = None

        is_masking_interpolant = (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        )
        self.num_tokens = 21 if is_masking_interpolant else 20

    @property
    def igso3(self):
        # On CPU. TODO consider moving to `self._device`.
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, self.cfg.rots.igso3_sigma, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """Take a random t in the range [min_t, 1-min_t], for corrupting / training a batch."""
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self.cfg.min_t) + self.cfg.min_t

    def _compute_sigma_t(self, t: torch.Tensor, scale: float, min_sigma: float = 0.01):
        """
        Compute the standard deviation of the IGSO(3) noise at time t.
        The standard deviation is a parabolic function of t, with a minimum at t=0 and t=1.
        """
        sigma_t = scale * torch.sqrt(t * (1 - t) + 1e-4)
        return torch.clamp(sigma_t, min=min_sigma)

    def _batch_ot(self, trans_0, trans_1, res_mask, center: bool = False):
        """
        Compute optimal transport between two batches of translations.
        returns OT mapping of trans_0 structures to trans_1 structures.
        Will force translations are centered if `center==True`.
        Does not re-order the translations within a structure.
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
        noise_perm, gt_perm = linear_sum_assignment(to_cpu(cost_matrix).numpy())
        return batch_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        """
        Corrupt translations from t=1 to t using Gaussian noise.
        """
        trans_nm_0 = centered_gaussian(*res_mask.shape, device=self._device)
        trans_0 = trans_nm_0 * NM_TO_ANG_SCALE

        # compute batch OT. Expect no need to center, noise and t=1 should already be.
        if self.cfg.trans.batch_ot:
            trans_0 = self._batch_ot(
                trans_0,
                trans_1,
                res_mask=res_mask,
            )

        # compute trans_t
        if self.cfg.trans.train_schedule == InterpolantTranslationsScheduleEnum.linear:
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(f"Unknown trans schedule {self.cfg.trans.train_schedule}")

        # stochastic paths
        if self.cfg.trans.stochastic:
            # guassian noise added is markovian; just sample from gaussian, scaled by sigma_t, and add.
            # sigma_t is ~parabolic (and ~0 at t=0 and t=1) so corrupted sample reflects marginal distribution at t.
            sigma_t = self._compute_sigma_t(
                t.squeeze(1),  # t is (B, 1), we need (B,)
                scale=self.cfg.trans.stochastic_noise_intensity,
            )
            intermediate_noise = centered_gaussian(
                *res_mask.shape,
                device=self._device,
            )  # (B, N, 3)
            intermediate_noise = intermediate_noise * sigma_t[..., None, None]
            intermediate_noise = intermediate_noise * NM_TO_ANG_SCALE
            trans_t = trans_t + intermediate_noise

        trans_t = mask_blend_2d(trans_t, trans_1, diffuse_mask)

        # Center residues at origin
        trans_t -= batch_center_of_mass(trans_t, mask=res_mask)[:, None]

        return trans_t * res_mask[..., None]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
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
            so3_t = 1 - torch.exp(-t * self.cfg.rots.exp_rate)
        else:
            raise ValueError(f"Invalid schedule: {so3_schedule}")

        # interpolate on geodesic between rotmats_0 and rotmats_1 to get rotmats_t
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)

        # stochastic paths
        if self.cfg.rots.stochastic:
            # gaussian noise added is markovian; we are sampling intermediate point directly from marginal
            # so we just need to compute sigma_t for variance of IGSO(3) noise.
            # compute noise std deviation (mean is just rotmats_t)
            sigma_t = self._compute_sigma_t(
                so3_t.squeeze(1),  # t is (B, 1), we need (B,)
                scale=self.cfg.rots.stochastic_noise_intensity,
            )
            sigma_t = sigma_t.cpu()  # ensure on cpu for igso3 calculation
            # multipy rotmats_t by IGSO(3) noise
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

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        """Corrupt AA residues from t=1 to t, using masking or uniform sampling."""
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        # mask fraction of residues based on t
        u = torch.rand(num_batch, num_res, device=self._device)
        aatypes_t = aatypes_1.clone()
        corruption_mask = u < (1 - t)  # (B, N)

        if (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            # For masking interpolant, corrupted residues are set to mask
            aatypes_t[corruption_mask] = MASK_TOKEN_INDEX

        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            # For uniform interpolant, corrupted residues are set to random logits
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

        # TODO - determine how to add additional noise
        #   We only have access to the discrete types here, not the logits / rate matrix
        #   But we want to be able to add additional noise to the rate matrix on trajectory
        #   Do we just add additional noise now?

        # residues outside `res_mask` are set to mask.
        aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        # only corrupt residues in `diffuse_mask`
        return mask_blend_1d(aatypes_t, aatypes_1, diffuse_mask)

    def _set_corruption_times(
        self,
        noisy_batch: NoisyFeatures,
        task: DataTaskEnum,
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

            # determine multi-task allocations. Each DataTaskEnum supports different multi-tasks.
            if task == DataTaskEnum.inpainting:
                # proportions: unconditional, forward, inverse, rest = normal inpainting
                unc = self.cfg.inpainting_unconditional_prop
                fwd = self.cfg.codesign_forward_fold_prop
                inv = self.cfg.codesign_inverse_fold_prop
                unconditional_mask = (u < unc).bool()
                forward_fold_mask = ((u >= unc) & (u < unc + fwd)).float()
                inverse_fold_mask = ((u >= unc + fwd) & (u < unc + fwd + inv)).float()

                # for `inpainting -> unconditional` examples, override `diffuse_mask` to 1.0
                orig_diffuse = noisy_batch[bp.diffuse_mask]
                noisy_batch[bp.diffuse_mask] = torch.where(
                    unconditional_mask[:, None],
                    torch.ones_like(orig_diffuse),
                    orig_diffuse,
                ).float()

            elif task == DataTaskEnum.hallucination:
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

            # TODO(inpainting-fixed) support fixed motifs
            #   Requires we have a sane way to center the structures, and tension of doing here vs in dataset.
            #   See `scaffolding.md` for discussion.

            so3_t = structure_t[:, None]
            r3_t = structure_t[:, None]
            cat_t = cat_t[:, None]

        # Default: single `t` time is shared by each domain
        # TODO(inpainting) up for debate what the appropriate `t` is for the sequence.
        #   Maybe `t + (1-t) * (1-diffuse_mask.mean())` could make sense...
        else:
            t = self.sample_t(num_batch)[:, None]
            so3_t = t
            r3_t = t
            cat_t = t

        noisy_batch[nbp.so3_t] = so3_t  # [B, 1]
        noisy_batch[nbp.r3_t] = r3_t  # [B, 1]
        noisy_batch[nbp.cat_t] = cat_t  # [B, 1]

    def corrupt_batch(self, batch: BatchFeatures, task: DataTaskEnum) -> NoisyFeatures:
        """
        Corrupt `t=1` data into a noisy batch at sampled `t`.

        Supports within-batch multi-task learning if `codesign_separate_t == True`.
        This allows a wider set of tasks during inference.

        Some examples in the batch are assigned t=1 values (sequence for forward_folding, structure for inverse_folding),
        or modify the `diffuse_mask` (inpainting -> unconditional).

        Fixing at t=1 has the effect of not corrupting the domain.
        """
        noisy_batch: NoisyFeatures = copy.deepcopy(batch)

        # set t values for each domain, i.e. `bp.so3_t`, `bp.r3_t`, `bp.cat_t`, and potentially modify `diffuse_mask`.
        self._set_corruption_times(
            noisy_batch=noisy_batch,
            task=task,
        )

        trans_1 = batch[bp.trans_1]  # (B, N, 3) in Angstroms
        rotmats_1 = batch[bp.rotmats_1]  # (B, N, 3, 3)
        aatypes_1 = batch[bp.aatypes_1]  # (B, N)
        res_mask = batch[bp.res_mask]  # (B, N)
        diffuse_mask = batch[bp.diffuse_mask]  # (B, N)
        r3_t = noisy_batch[nbp.r3_t]  # (B, 1)
        so3_t = noisy_batch[nbp.so3_t]  # (B, 1)
        cat_t = noisy_batch[nbp.cat_t]  # (B, 1)

        # Determine sequence and structure corruption masks.
        # Inpainting:
        #   The motifs are interpolated over time, so we corrupt the entire structure. We explicitly fix the motif.
        #   The motif sequence is fixed. However, the rest is not, and t should be in sync across domains.
        # For other tasks, everything is corrupted i.e. `(diffuse_mask == 1.0).all()`
        #   Though values at t=1 effectively won't be corrupted.
        corruption_mask_structure = diffuse_mask.clone()
        corruption_mask_sequence = diffuse_mask
        if task == DataTaskEnum.inpainting:
            # any rows with a fixed motif have a `diffuse_mask.mean() < 1.0`
            is_inpainting_mask = diffuse_mask.float().mean(dim=1) < 1.0
            corruption_mask_structure[is_inpainting_mask] = 1.0

        # Apply corruptions

        if self.cfg.trans.corrupt:
            trans_t = self._corrupt_trans(
                trans_1,
                t=r3_t,
                res_mask=res_mask,
                diffuse_mask=corruption_mask_structure,
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
                diffuse_mask=corruption_mask_structure,
            )
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch[nbp.rotmats_t] = rotmats_t

        if self.cfg.aatypes.corrupt:
            aatypes_t = self._corrupt_aatypes(
                aatypes_1,
                t=cat_t,
                res_mask=res_mask,
                diffuse_mask=corruption_mask_sequence,
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

    def rot_sample_kappa(self, t: torch.Tensor):
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            return 1 - torch.exp(-t * self.cfg.rots.exp_rate)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            return t
        else:
            raise ValueError(f"Invalid schedule: {self.cfg.rots.sample_schedule}")

    def _trans_vector_field(
        self, t: torch.Tensor, trans_1: torch.Tensor, trans_t: torch.Tensor
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
        self, d_t: float, t: torch.Tensor, trans_1: torch.Tensor, trans_t: torch.Tensor
    ) -> torch.Tensor:
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)

        trans_next = trans_t + trans_vf * d_t

        if self.cfg.trans.stochastic:
            # add gaussian noise scaled by sqrt(dt)
            trans_next += (
                torch.randn_like(trans_next)
                * np.sqrt(d_t)
                * self.cfg.trans.stochastic_noise_intensity
            )

        return trans_next

    def _rots_euler_step(
        self,
        d_t: float,
        t: torch.Tensor,
        rotmats_1: torch.Tensor,
        rotmats_t: torch.Tensor,
    ) -> torch.Tensor:
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            scaling = 1 / (1 - t)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            scaling = self.cfg.rots.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self.cfg.rots.sample_schedule}")

        rotmats_next = so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

        if self.cfg.rots.stochastic:
            # Sample IGSO(3) noise with a time-independent sigma_t, scaled by sqrt(dt)
            # Add IGSO(3) noise to stay on SO(3).
            # Also scale by `scaling` so the ratio of drift and diffusion is consistent.
            num_batch, num_res, _, _ = rotmats_t.shape
            # TODO(stochastic) - determine approprite value for sigma
            sigma = torch.tensor([1]) * (
                scaling * self.cfg.rots.stochastic_noise_intensity * np.sqrt(d_t)
            )
            intermediate_noise = self.igso3.sample(sigma, num_batch * num_res).to(
                rotmats_t.device
            )
            intermediate_noise = intermediate_noise.reshape(num_batch, num_res, 3, 3)
            rotmats_next = torch.einsum(
                "...ij,...jk->...ik", rotmats_next, intermediate_noise
            )

        return rotmats_next

    def _psi_euler_step(
        self, d_t: float, t: torch.Tensor, psi_1: torch.Tensor, psi_t: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform an Euler step to update psi angles.
        Note that only the model predicts psi angles, but they are not an input to the model.
        Primary use is for inpainting to step towards known psi angles in motifs.
        """
        psi_vf = (psi_1 - psi_t) / (1 - t)
        psi_next = psi_t + psi_vf * d_t

        # HACK piggyback on `trans.stochastic`
        if self.cfg.trans.stochastic:
            # add gaussian noise scaled by sqrt(dt)
            psi_next += (
                torch.randn_like(psi_next)
                * np.sqrt(d_t)
                * self.cfg.trans.stochastic_noise_intensity
            )

        return psi_next

    def _regularize_step_probs(self, step_probs, aatypes_t):
        """
        Regularize the step probabilities to ensure they conform to the requirements of a rate matrix.
        The rate matrix we learn at `t` needs each row to sum to zero.
        """
        batch_size, num_res, S = step_probs.shape
        device = step_probs.device
        assert aatypes_t.shape == (batch_size, num_res)

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        # set the probabilities corresponding to the current amino acid types to 0.0
        batch_idx = torch.arange(batch_size, device=device).repeat_interleave(num_res)
        residue_idx = torch.arange(num_res, device=device).repeat(batch_size)
        curr_states = aatypes_t.long().flatten()
        step_probs[batch_idx, residue_idx, curr_states] = 0.0

        # adjust the probabilities at the current positions to be the negative sum of all other values in the row
        row_sums = torch.sum(step_probs, dim=-1).flatten()
        step_probs[batch_idx, residue_idx, curr_states] = 1.0 - row_sums

        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
        # in case negative or out-of-bound values appear after the diagonal assignment.
        step_probs = torch.clamp(step_probs, min=0.0, max=1.0)

        return step_probs

    def _aatypes_euler_step(self, d_t, t, logits_1, aatypes_t):
        """
        Perform an Euler step to update amino acid types based on the provided logits and interpolation settings.

        This function handles two interpolation strategies:
        1. "masking": Updates the amino acid types by masking certain positions and sampling new types
           based on modified probabilities. Assumes S = 21 to include a special MASK token.
        2. "uniform": Samples new amino acid types uniformly, with the assumption that no MASK tokens are involved.
           Assumes S = 20.
        """
        # d_t:       float timestep delta
        # t:         current interpolation time in [0,1]
        # logits_1:  (B, N, S) unscaled probabilities for each aa
        # aatypes_t: (B, N) current aa types

        # If `purity` enabled, use purity-based Euler step function.
        if self.cfg.aatypes.do_purity:
            return self._aatypes_euler_step_purity(d_t, t, logits_1, aatypes_t)

        batch_size, num_res, num_states = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        device = logits_1.device

        temp = self.cfg.aatypes.temp
        noise = self.cfg.aatypes.noise

        if (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            assert num_states == 21

            # set mask to small negative so won't be picked in softmax
            logits_1[:, :, MASK_TOKEN_INDEX] = -1e9

            # convert logits to probabilities
            pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

            # prepare a (0,0,...1) mask vector to help add masking transitions.
            mask_one_hot = torch.zeros((num_states,), device=device)
            mask_one_hot[MASK_TOKEN_INDEX] = 1.0

            # identify which positions are currently mask
            aatypes_t_is_mask = (
                (aatypes_t == MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
            )

            # compute step probabilities (scaled by d_t), with noise and time factoring
            step_probs = (
                d_t * pt_x1_probs * ((1.0 + noise * t) / (1.0 - t))
            )  # (B, N, S)
            # add transitions from non-mask to mask as noise
            step_probs += (
                d_t * (1.0 - aatypes_t_is_mask) * mask_one_hot.view(1, 1, -1) * noise
            )

            # force valid rate matrix
            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            # sample new residues from step_probs
            new_aatypes = torch.multinomial(
                step_probs.view(-1, num_states), num_samples=1
            )
            return new_aatypes.view(batch_size, num_res)
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            assert num_states == 20
            assert (
                aatypes_t.max() < 20
            ), "No UNK tokens allowed in the uniform sampling step!"

            # convert logits to probabilities
            pt_x1_probs = F.softmax(logits_1 / temp, dim=-1)  # (B, N, S)

            # probability of x1 matching xt exactly
            pt_x1_eq_xt_prob = torch.gather(
                pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)
            )  # (B, D, 1)
            assert pt_x1_eq_xt_prob.shape == (batch_size, num_res, 1)

            # compute step probabilities (scaled by d_t), with noise and time factoring.
            # encourages transitions with an additional uniform 'noise' term for the matching residue.
            step_probs = d_t * (
                pt_x1_probs * ((1.0 + noise + noise * (num_states - 1) * t) / (1.0 - t))
                + noise * pt_x1_eq_xt_prob
            )

            # force valid rate matrix
            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            # sample new residues from step_probs
            new_aatypes = torch.multinomial(
                step_probs.view(-1, num_states), num_samples=1
            )
            return new_aatypes.view(batch_size, num_res)
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        """
        Perform an Euler step with a focus on maintaining "purity" by selectively unmasking and updating
        amino acid types. This function is designed specifically for the "masking" interpolant type, where S = 21.

        This function identifies masked residues, then picks some number to unmask based on their maximum
        log-probabilities, then re-masks some positions proportional to noise.
        """
        batch_size, num_res, num_states = logits_1.shape
        device = logits_1.device

        assert aatypes_t.shape == (batch_size, num_res)
        assert (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        )
        assert (
            num_states == 21
        ), "Purity-based unmasking only works with masking interpolant type"

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
        unmasked_samples = unmasked_samples.view(batch_size, num_res)

        # Next lines are vectorized version of:
        # for b in range(B):
        #     for n in range(N):
        #         if n < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprob_indices[b, n]] = unmasked_samples[b, sorted_max_logprob_indices[b, d]]

        # create a mask that indicates top positions to unmask in each batch
        D_grid = (
            torch.arange(num_res, device=device).unsqueeze(0).expand(batch_size, -1)
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
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(
            dim=1,
            index=selected_indices,
            src=torch.ones((batch_size, num_res), device=device),
        )
        # if zero are unmasked, we skip altogether
        none_unmasked_mask = (number_to_unmask == 0).unsqueeze(-1).float()
        mask2 *= 1.0 - none_unmasked_mask

        # update unmasked positions in aatypes_t
        aatypes_t = aatypes_t * (1.0 - mask2) + unmasked_samples * mask2

        # re-mask some positions as noise with probability (d_t * noise)
        re_mask_prob = d_t * noise
        rand_vals = torch.rand(batch_size, num_res, device=device)
        re_mask_mask = (rand_vals < re_mask_prob).float()
        aatypes_t = aatypes_t * (1.0 - re_mask_mask) + (MASK_TOKEN_INDEX * re_mask_mask)

        return aatypes_t

    def sample(
        self,
        num_batch: int,
        num_res: int,
        model,
        task: InferenceTaskEnum,
        diffuse_mask: torch.Tensor,
        chain_idx: torch.Tensor,
        res_idx: torch.Tensor,
        trans_0: Optional[torch.Tensor] = None,
        rotmats_0: Optional[torch.Tensor] = None,
        psis_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
        trans_1: Optional[torch.Tensor] = None,
        rotmats_1: Optional[torch.Tensor] = None,
        psis_1: Optional[torch.Tensor] = None,
        aatypes_1: Optional[torch.Tensor] = None,
        # t_nn is model/function that generates explicit time steps (r3, so3, cat) given t
        t_nn: Union[Callable[[torch.Tensor], torch.Tensor], Any] = None,
    ) -> Tuple[SamplingTrajectory, SamplingTrajectory]:
        """
        Generate samples using the learned vector fields.

        In theory, sampling is fairly straight-forward: integrate over a bunch of time steps.
        However, there is a lot of special handling to:
        - generate t=0 values if not provided
        - handling each task, setting up t=0 and t=1 values, and during interpolating
        - handling `codesign_separate_t` for fixed domains
        - supporting self-conditioning
        - saving each step of the trajectory (direct model output and integrated output)
        - special handling for the final timestep

        Returns two trajectories:
        1) protein_trajectory / predicted states - sampled trajectory.
            Intermediate steps resulting from integration over vector fields. Fixed values set by mask.
            No logits, just the amino acids (integration with rate matrix yields discrete sequence).
        2) model_trajectory / clean states - direct model output.
            Without masking or added noise. Does not involve integrating. Includes logits.

        Note that while sampling operates on backbones, the emitted structures are all-atom (i.e. atom37).

        The general process is:
        - generate the initial noisy batch, using optional inputs if provided
        - sample over time-steps
            - updating translations, rotations, and amino acid types at each time step
            - generate rigids from the updated frames
        - perform the final step after the loop
        """
        # task-specific input checks
        if task == InferenceTaskEnum.unconditional:
            assert self.cfg.trans.corrupt
            assert self.cfg.rots.corrupt
            assert self.cfg.aatypes.corrupt
            # no inputs required
        elif task == InferenceTaskEnum.inpainting:
            assert self.cfg.trans.corrupt
            assert self.cfg.rots.corrupt
            assert self.cfg.aatypes.corrupt
            # inputs
            assert trans_1 is not None
            assert rotmats_1 is not None
            assert psis_1 is not None
            assert aatypes_1 is not None
            assert diffuse_mask is not None
        elif task == InferenceTaskEnum.forward_folding:
            assert self.cfg.trans.corrupt
            assert self.cfg.rots.corrupt
            assert not self.cfg.aatypes.corrupt
            # inputs
            assert aatypes_1 is not None
            assert (
                self.cfg.aatypes.noise == 0.0
            )  # cfg check unnecessary if not corrupting?
        elif task == InferenceTaskEnum.inverse_folding:
            assert not self.cfg.trans.corrupt
            assert not self.cfg.rots.corrupt
            assert self.cfg.aatypes.corrupt
            # inputs
            assert trans_1 is not None
            assert rotmats_1 is not None
            assert psis_1 is not None
        else:
            raise ValueError(f"Unknown task {task}")

        # for inference, all residues under consideration
        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Initialize t=1 values for translations, rotations, psi, and aatypes, if not defined

        if trans_1 is None:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(
                num_batch, num_res, 1, 1
            )
        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()
        if psis_1 is None:
            psis_1 = torch.zeros((num_batch, num_res, 2), device=self._device)
        logits_1 = torch.nn.functional.one_hot(
            aatypes_1, num_classes=self.num_tokens
        ).float()

        # Initialize t=0 prior samples i.e. noise (technically `t=cfg.min_t`)

        if trans_0 is None:
            # Generate centered Gaussian noise for translations (shape: [num_batch, num_res, 3])
            trans_0 = (
                centered_gaussian(num_batch, num_res, device=self._device)
                * NM_TO_ANG_SCALE
            )
        if rotmats_0 is None:
            # Generate uniform SO(3) rotation matrices (shape: [num_batch, num_res, 3, 3])
            rotmats_0 = uniform_so3(num_batch, num_res, device=self._device)
        if aatypes_0 is None:
            # Generate initial amino acid types based on the interpolant type
            if (
                self.cfg.aatypes.interpolant_type
                == InterpolantAATypesInterpolantTypeEnum.masking
            ):
                aatypes_0 = masked_categorical(num_batch, num_res, device=self._device)
            elif (
                self.cfg.aatypes.interpolant_type
                == InterpolantAATypesInterpolantTypeEnum.uniform
            ):
                aatypes_0 = uniform_categorical(
                    num_batch, num_res, num_tokens=self.num_tokens, device=self._device
                )
            else:
                raise ValueError(
                    f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
                )
        if psis_0 is None:
            # Generate initial psi angles
            # TODO a better prior, reflecting these are angles
            psis_0 = torch.randn((num_batch, num_res, 2), device=self._device)

        # For inpainting, fix motif sequence. Translations and rotations will be sampled.
        # TODO(inpainting-fixed) support fixed motifs
        if task == InferenceTaskEnum.inpainting:
            # trans_0 = mask_blend_2d(trans_0, trans_1, diffuse_mask)
            # rotmats_0 = mask_blend_3d(rotmats_0, rotmats_1, diffuse_mask)
            aatypes_0 = mask_blend_1d(aatypes_0, aatypes_1, diffuse_mask)

        # Handle `codesign_separate_t`
        # t=0 values will be fixed to t=1 values throughout where appropriate
        if self.cfg.codesign_separate_t:
            if task == InferenceTaskEnum.unconditional:
                pass
            elif task == InferenceTaskEnum.inpainting:
                pass
            elif task == InferenceTaskEnum.forward_folding:
                aatypes_0 = aatypes_1
            elif task == InferenceTaskEnum.inverse_folding:
                trans_0 = trans_1
                rotmats_0 = rotmats_1
                psis_0 = psis_1
            else:
                raise ValueError(f"Unknown task {task}")

        # no self-conditioning during first step sampling, or if self-conditioning disabled
        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device
        )

        # Set up batch. This batch object will be modified and used for each time step.
        batch = {
            bp.res_mask: res_mask,
            bp.diffuse_mask: diffuse_mask,
            bp.chain_idx: chain_idx,
            bp.res_idx: res_idx,
            nbp.trans_sc: trans_sc,
            nbp.aatypes_sc: aatypes_sc,
        }

        # Helper function to get atom37 structure from residue frames, only for `res_mask`
        frames_to_atom37 = lambda trans, rots, psis: to_cpu(
            all_atom.atom37_from_trans_rot(
                trans=trans,
                rots=rots,
                psi_torsions=psis,
                res_mask=res_mask,
            )
        )

        # model_trajectory tracks model outputs
        model_trajectory = SamplingTrajectory(
            num_batch=num_batch,
            num_res=num_res,
            num_tokens=self.num_tokens,
        )
        # protein_trajectory tracks predicted intermediate states integrating from t=0 to t=1
        protein_trajectory = SamplingTrajectory(
            num_batch=num_batch,
            num_res=num_res,
            num_tokens=self.num_tokens,
        )
        protein_trajectory.append(
            SamplingStep(
                structure=frames_to_atom37(trans_0, rotmats_0, psis_0),
                amino_acids=to_cpu(aatypes_0),
            )
        )

        # Set-up time
        ts = torch.linspace(self.cfg.min_t, 1.0, self.cfg.sampling.num_timesteps)  # cpu
        t_1 = ts[0]

        # We will integrate in a loop over ts (handling the last step after the loop)
        # t_1 is the current time, t_2 is the next time
        # t_1 starts at t=0
        trans_t_1 = trans_0
        rotmats_t_1 = rotmats_0
        psis_t_1 = psis_0
        aatypes_t_1 = aatypes_0

        for t_2 in ts[1:]:
            # update batch values
            batch[nbp.trans_t] = trans_t_1 if self.cfg.trans.corrupt else trans_1
            batch[nbp.rotmats_t] = rotmats_t_1 if self.cfg.rots.corrupt else rotmats_1
            batch[nbp.aatypes_t] = (
                aatypes_t_1 if self.cfg.aatypes.corrupt else aatypes_1
            )

            # determine time for each domain
            t = torch.ones((num_batch, 1), device=self._device) * t_1
            if t_nn is not None:
                (
                    batch[nbp.r3_t],
                    batch[nbp.so3_t],
                    batch[nbp.cat_t],
                ) = torch.split(t_nn(t), -1)
            else:
                if self.cfg.provide_kappa:
                    batch[nbp.so3_t] = self.rot_sample_kappa(t)
                else:
                    batch[nbp.so3_t] = t
                batch[nbp.r3_t] = t
                batch[nbp.cat_t] = t

            # If `codesign_separate_t`, fixed domains set to ~t=1 (penultimate step, take final step later)
            if self.cfg.codesign_separate_t:
                # TODO consider setting to t=1 to match how `codesign_separate_t` sets values in `corrupt_batch()`
                t_minus_1 = (1 - self.cfg.min_t) * torch.ones(
                    (num_batch, 1), device=self._device
                )
                if task == InferenceTaskEnum.unconditional:
                    pass
                elif task == InferenceTaskEnum.inpainting:
                    pass
                elif task == InferenceTaskEnum.forward_folding:
                    batch[nbp.cat_t] = t_minus_1
                elif task == InferenceTaskEnum.inverse_folding:
                    batch[nbp.r3_t] = t_minus_1
                    batch[nbp.so3_t] = t_minus_1
                else:
                    raise ValueError(f"Unknown task {task}")

            # Get model output at translations/rotations/aatypes respective `t`
            with torch.no_grad():
                model_out = model(batch)
            pred_trans_1 = model_out[pbp.pred_trans]
            pred_rotmats_1 = model_out[pbp.pred_rotmats]
            pred_psis_1 = model_out[pbp.pred_psi]  # may be None, if not predicting
            pred_aatypes_1 = model_out[pbp.pred_aatypes]
            pred_logits_1 = model_out[pbp.pred_logits]

            model_trajectory.append(
                SamplingStep(
                    structure=frames_to_atom37(
                        pred_trans_1, pred_rotmats_1, pred_psis_1
                    ),
                    amino_acids=to_cpu(pred_aatypes_1),
                    logits=to_cpu(pred_logits_1),
                )
            )

            # Mask fixed values to prepare for integration and update the batch for next step.
            if task == InferenceTaskEnum.unconditional:
                pass
            elif task == InferenceTaskEnum.inpainting:
                # TODO(inpainting-fixed) depending on guidance type, set _1 values to t=1 using mask
                # pred_trans_1 = mask_blend_2d(pred_trans_1, trans_1, diffuse_mask)
                # pred_rotmats_1 = mask_blend_3d(pred_rotmats_1, rotmats_1, diffuse_mask)
                # pred_psis_1 = mask_blend_2d(pred_psis_1, psis_1, diffuse_mask)
                pred_logits_1 = mask_blend_2d(pred_logits_1, logits_1, diffuse_mask)
                pred_aatypes_1 = mask_blend_1d(pred_aatypes_1, aatypes_1, diffuse_mask)
            elif task == InferenceTaskEnum.forward_folding:
                # scale logits during integration, assumes will `softmax`
                pred_logits_1 = 100.0 * logits_1
                pred_aatypes_1 = aatypes_1
            elif task == InferenceTaskEnum.inverse_folding:
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1
                pred_psis_1 = psis_1
            else:
                raise ValueError(f"Unknown task {task}")

            # Update self-conditioning values
            if self.cfg.self_condition:
                if task == InferenceTaskEnum.unconditional:
                    batch[nbp.trans_sc] = mask_blend_2d(
                        pred_trans_1, trans_1, diffuse_mask
                    )
                    batch[nbp.aatypes_sc] = mask_blend_2d(
                        pred_logits_1, logits_1, diffuse_mask
                    )
                elif task == InferenceTaskEnum.inpainting:
                    # TODO(inpainting-fixed) set _1 values to t=1 using mask
                    # batch[nbp.trans_sc] = mask_blend_2d(pred_trans_1, trans_1, diffuse_mask)
                    batch[nbp.trans_sc] = pred_trans_1  # interpolate all residues
                    batch[nbp.aatypes_sc] = mask_blend_2d(
                        pred_logits_1, logits_1, diffuse_mask
                    )
                elif task == InferenceTaskEnum.forward_folding:
                    batch[nbp.trans_sc] = mask_blend_2d(
                        pred_trans_1, trans_1, diffuse_mask
                    )
                    # sequence fixed for self-conditioning
                    batch[nbp.aatypes_sc] = logits_1
                elif task == InferenceTaskEnum.inverse_folding:
                    # matching Multiflow, `trans_sc` not fixed
                    batch[nbp.trans_sc] = mask_blend_2d(
                        pred_trans_1, trans_1, diffuse_mask
                    )
                    batch[nbp.aatypes_sc] = mask_blend_2d(
                        pred_logits_1, logits_1, diffuse_mask
                    )
                else:
                    raise ValueError(f"Unknown task {task}")

            # Take next step, size `d_t` from `t_1` (current value) to `t_2` (toward predicted value)
            # We are at `t_1` with state `{domain}_t_1`. The model predicted `pred_{domain}_1`.
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)
            aatypes_t_2 = self._aatypes_euler_step(d_t, t_1, pred_logits_1, aatypes_t_1)
            psis_t_2 = (
                self._psi_euler_step(d_t, t_1, pred_psis_1, psis_t_1)
                if pred_psis_1 is not None
                else None
            )

            # For inpainting, fix motif sequence; set motif structure by interpolating motif toward known t=1.
            # TODO(inpainting-fixed) support fixed motifs
            if task == InferenceTaskEnum.inpainting:
                # Sequence is fixed for motifs at t=1
                aatypes_t_2 = mask_blend_1d(aatypes_t_2, aatypes_1, diffuse_mask)

                # Get interpolated values for motif translations and rotations
                trans_t_2_motif = self._trans_euler_step(d_t, t_1, trans_1, trans_t_1)
                rotmats_t_2_motif = self._rots_euler_step(
                    d_t, t_1, rotmats_1, rotmats_t_1
                )
                # Set motif positions to interpolated values
                trans_t_2 = mask_blend_2d(trans_t_2, trans_t_2_motif, diffuse_mask)
                rotmats_t_2 = mask_blend_3d(
                    rotmats_t_2, rotmats_t_2_motif, diffuse_mask
                )

                # For psi angles, take an Euler step toward known values
                if psis_t_2 is not None:
                    psi_t_2_motif = self._psi_euler_step(d_t, t_1, psis_1, pred_psis_1)
                    psis_t_2 = mask_blend_2d(psis_t_2, psi_t_2_motif, diffuse_mask)

            # Center diffused residues to maintain translation invariance
            # Definitely should center if inpainting or stochastic, but might as well if unconditional too
            # TODO(inpainting-fixed) keep fixed motifs centered, so condition remains the same (learn scaffold drift)
            trans_t_2 -= batch_center_of_mass(trans_t_2, mask=res_mask)[:, None]

            # Add to trajectory
            protein_trajectory.append(
                SamplingStep(
                    structure=frames_to_atom37(trans_t_2, rotmats_t_2, psis_t_2),
                    amino_acids=to_cpu(aatypes_t_2),
                )
            )

            # Get ready for the next step
            trans_t_1 = trans_t_2
            rotmats_t_1 = rotmats_t_2
            aatypes_t_1 = aatypes_t_2
            psis_t_1 = psis_t_2
            t_1 = t_2

        # We only integrated to 1-min_t, so need to make a final step
        t = torch.ones((num_batch, 1), device=self._device) * ts[-1]
        batch[nbp.so3_t] = t
        batch[nbp.r3_t] = t
        batch[nbp.cat_t] = t

        batch[nbp.trans_t] = trans_t_1 if self.cfg.trans.corrupt else trans_1
        batch[nbp.rotmats_t] = rotmats_t_1 if self.cfg.rots.corrupt else rotmats_1
        # Note - assigned to 'aatype_t' not 'aatypes_t' in public MultiFlow code, assume a bug.
        # Perhaps impacted inverse folding performance, was timestep behind (but also network was small).
        batch[nbp.aatypes_t] = aatypes_t_1 if self.cfg.aatypes.corrupt else aatypes_1

        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out[pbp.pred_trans]
        pred_rotmats_1 = model_out[pbp.pred_rotmats]
        pred_psis_1 = model_out[pbp.pred_psi]  # may be None, if not predicting
        pred_aatypes_1 = model_out[pbp.pred_aatypes]
        pred_logits_1 = model_out[pbp.pred_logits]

        model_trajectory.append(
            SamplingStep(
                structure=frames_to_atom37(pred_trans_1, pred_rotmats_1, pred_psis_1),
                amino_acids=to_cpu(pred_aatypes_1),
                logits=to_cpu(pred_logits_1),
            )
        )

        # Clean up the final outputs
        if task == InferenceTaskEnum.unconditional:
            pass
        elif task == InferenceTaskEnum.inpainting:
            # Here, deviate from the convention in FrameFlow which leaves these alone.
            # Forcing them, like `forward_folding` or `inverse_folding` seems to make sense
            # if they are used for guidance.
            pred_logits_1 = mask_blend_2d(pred_logits_1, logits_1, diffuse_mask)
            pred_aatypes_1 = mask_blend_1d(pred_aatypes_1, aatypes_1, diffuse_mask)
            pred_trans_1 = mask_blend_2d(pred_trans_1, trans_1, diffuse_mask)
            pred_rotmats_1 = mask_blend_3d(pred_rotmats_1, rotmats_1, diffuse_mask)
            pred_psis_1 = (
                mask_blend_2d(pred_psis_1, psis_1, diffuse_mask)
                if pred_psis_1 is not None
                else None
            )
        elif task == InferenceTaskEnum.forward_folding:
            pred_logits_1 = logits_1
            pred_aatypes_1 = aatypes_1
        elif task == InferenceTaskEnum.inverse_folding:
            pred_trans_1 = trans_1
            pred_rotmats_1 = rotmats_1
            pred_psis_1 = psis_1
        else:
            raise ValueError(f"Unknown task {task}")

        protein_trajectory.append(
            SamplingStep(
                structure=frames_to_atom37(pred_trans_1, pred_rotmats_1, pred_psis_1),
                amino_acids=to_cpu(pred_aatypes_1),
            )
        )

        return protein_trajectory, model_trajectory
