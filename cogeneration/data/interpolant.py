import copy
from collections import defaultdict

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from cogeneration.config.base import (
    InterpolantAATypesInterpolantTypeEnum,
    InterpolantConfig,
    InterpolantRotationsScheduleEnum,
    InterpolantTranslationsScheduleEnum,
)
from cogeneration.data import all_atom, so3_utils
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.batch_props import PredBatchProps as pbp
from cogeneration.data.const import MASK_TOKEN_INDEX, NM_TO_ANG_SCALE, NUM_TOKENS
from cogeneration.data.rigid import batch_align_structures


def _centered_gaussian(num_batch, num_res, device):
    """
    Generates a tensor of shape (num_batch, num_res, 3) with values sampled from a centered Gaussian distribution.
    e.g. t=0 translations
    """
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def random_rotation_matrix(device):
    """Generate a random rotation matrix using PyTorch."""
    q = torch.randn(4, device=device)
    q = q / q.norm()  # Normalize the quaternion
    q0, q1, q2, q3 = q
    return torch.tensor(
        [
            [1 - 2 * (q2**2 + q3**2), 2 * (q1 * q2 - q0 * q3), 2 * (q1 * q3 + q0 * q2)],
            [2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1**2 + q3**2), 2 * (q2 * q3 - q0 * q1)],
            [2 * (q1 * q3 - q0 * q2), 2 * (q2 * q3 + q0 * q1), 1 - 2 * (q1**2 + q2**2)],
        ],
        dtype=torch.float32,
        device=device,
    )


def random_rotation_matrices(num_matrices: int, device):
    """Generate multiple random rotation matrices using PyTorch."""
    return torch.stack(
        [random_rotation_matrix(device=device) for _ in range(num_matrices)]
    )


def _uniform_so3(num_batch: int, num_res: int, device):
    """
    Generates a tensor of shape (num_batch, num_res, 3, 3) with values sampled from a uniform SO(3) distribution.
    e.g. t=0 rotation matrices
    """
    return random_rotation_matrices(num_batch * num_res, device=device).reshape(
        num_batch, num_res, 3, 3
    )


def _masked_categorical(num_batch, num_res, device):
    """
    Returns a mask tensor of shape (num_batch, num_res) with all values set to MASK_TOKEN_INDEX.
    e.g. t=0 aa types, masking interpolation
    """
    return torch.ones(size=(num_batch, num_res), device=device) * MASK_TOKEN_INDEX


def _uniform_categorical(num_batch, num_res, num_tokens, device):
    """
    Returns uniform random samples from the range [0, num_tokens) of shape (num_batch, num_res).
    e.g. t=0 aa types, uniform interpolation
    """
    return torch.randint(
        size=(num_batch, num_res), low=0, high=num_tokens, device=device
    )


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    """
    Mask 2D tensor with 1D mask tensor.
    """
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask):
    """
    Mask 3D tensor with 1D mask tensor.
    """
    return rotmats_t * diffuse_mask[..., None, None] + rotmats_1 * (
        1 - diffuse_mask[..., None, None]
    )


def _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask):
    """
    Mask 1D tensor with 1D mask tensor.
    """
    return aatypes_t * diffuse_mask + aatypes_1 * (1 - diffuse_mask)


class Interpolant:
    """
    Interpolant corrupts batches with noise, and generates samples using the learned vector fields.

    It generates noise for rotations (IGSO3), translations (Gaussian), and amino acid types (masking/uniform),
    and supports corrupts the input batch with the generated noise.

    It implements sampling and Euler steps for rotations, translations, and amino acid types.
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
        if self._igso3 is None:
            sigma_grid = torch.linspace(0.1, 1.5, 1000)
            self._igso3 = so3_utils.SampleIGSO3(1000, sigma_grid, cache_dir=".cache")
        return self._igso3

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        """Take a random t in the range [min_t, 1-min_t], for corrupting / training a batch."""
        t = torch.rand(num_batch, device=self._device)
        return t * (1 - 2 * self.cfg.min_t) + self.cfg.min_t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        """Corrupt translations from t=1 to t using Gaussian noise."""
        trans_nm_0 = _centered_gaussian(*res_mask.shape, self._device)
        trans_0 = trans_nm_0 * NM_TO_ANG_SCALE
        if self.cfg.trans.batch_ot:
            trans_0 = self._batch_ot(trans_0, trans_1, diffuse_mask)
        if self.cfg.trans.train_schedule == InterpolantTranslationsScheduleEnum.linear:
            trans_t = (1 - t[..., None]) * trans_0 + t[..., None] * trans_1
        else:
            raise ValueError(f"Unknown trans schedule {self.cfg.trans.train_schedule}")
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]

    def _batch_ot(self, trans_0, trans_1, res_mask):
        """Compute optimal transport between two batches of translations."""
        num_batch, num_res = trans_0.shape[:2]
        noise_idx, gt_idx = torch.where(torch.ones(num_batch, num_batch))
        batch_nm_0 = trans_0[noise_idx]
        batch_nm_1 = trans_1[gt_idx]
        batch_mask = res_mask[gt_idx]
        aligned_nm_0, aligned_nm_1, _ = batch_align_structures(
            batch_nm_0, batch_nm_1, mask=batch_mask
        )
        aligned_nm_0 = aligned_nm_0.reshape(num_batch, num_batch, num_res, 3)
        aligned_nm_1 = aligned_nm_1.reshape(num_batch, num_batch, num_res, 3)

        # Compute cost matrix of aligned noise to ground truth
        batch_mask = batch_mask.reshape(num_batch, num_batch, num_res)
        cost_matrix = torch.sum(
            torch.linalg.norm(aligned_nm_0 - aligned_nm_1, dim=-1), dim=-1
        ) / torch.sum(batch_mask, dim=-1)
        noise_perm, gt_perm = linear_sum_assignment(cost_matrix.detach().cpu().numpy())
        return aligned_nm_0[(tuple(gt_perm), tuple(noise_perm))]

    def _corrupt_rotmats(self, rotmats_1, t, res_mask, diffuse_mask):
        """Corrupt rotations from t=1 to t using IGSO3."""
        num_batch, num_res = res_mask.shape
        noisy_rotmats = self.igso3.sample(torch.tensor([1.5]), num_batch * num_res).to(
            self._device
        )
        noisy_rotmats = noisy_rotmats.reshape(num_batch, num_res, 3, 3)
        rotmats_0 = torch.einsum("...ij,...jk->...ik", rotmats_1, noisy_rotmats)

        so3_schedule = self.cfg.rots.train_schedule
        if so3_schedule == InterpolantRotationsScheduleEnum.exp:
            so3_t = 1 - torch.exp(-t * self.cfg.rots.exp_rate)
        elif so3_schedule == InterpolantRotationsScheduleEnum.linear:
            so3_t = t
        else:
            raise ValueError(f"Invalid schedule: {so3_schedule}")
        rotmats_t = so3_utils.geodesic_t(so3_t[..., None], rotmats_1, rotmats_0)
        identity = torch.eye(3, device=self._device)
        rotmats_t = rotmats_t * res_mask[..., None, None] + identity[None, None] * (
            1 - res_mask[..., None, None]
        )
        return _rots_diffuse_mask(rotmats_t, rotmats_1, diffuse_mask)

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        """Corrupt AA residues from t=1 to t, using masking or uniform sampling."""
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)

        if (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t)  # (B, N)

            aatypes_t[corruption_mask] = MASK_TOKEN_INDEX

            aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)

        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            u = torch.rand(num_batch, num_res, device=self._device)
            aatypes_t = aatypes_1.clone()
            corruption_mask = u < (1 - t)  # (B, N)
            uniform_sample = torch.randint_like(aatypes_t, low=0, high=NUM_TOKENS)
            aatypes_t[corruption_mask] = uniform_sample[corruption_mask]

            aatypes_t = aatypes_t * res_mask + MASK_TOKEN_INDEX * (1 - res_mask)
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

        return _aatypes_diffuse_mask(aatypes_t, aatypes_1, diffuse_mask)

    def corrupt_batch(self, batch):
        """
        Sample t to generate a noisy batch from the input batch.
        """
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch[bp.trans_1]  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch[bp.rotmats_1]

        # [B, N]
        aatypes_1 = batch[bp.aatypes_1]

        # [B, N]
        res_mask = batch[bp.res_mask]
        diffuse_mask = batch[bp.diffuse_mask]
        num_batch, num_res = diffuse_mask.shape

        # [B, 1]
        if self.cfg.codesign_separate_t:
            # Generate random values `u` to determine the type of corruption
            u = torch.rand((num_batch,), device=self._device)
            # Assign the type of corruption based on `u` and proportion for forward/inverse folding
            forward_fold_mask = (u < self.cfg.codesign_forward_fold_prop).float()
            inverse_fold_mask = (
                u
                < self.cfg.codesign_inverse_fold_prop
                + self.cfg.codesign_forward_fold_prop
            ).float() * (u >= self.cfg.codesign_forward_fold_prop).float()

            # Sample t values for normal structure and categorical corruption
            normal_structure_t = self.sample_t(num_batch)
            inverse_fold_structure_t = torch.ones((num_batch,), device=self._device)
            normal_cat_t = self.sample_t(num_batch)
            forward_fold_cat_t = torch.ones((num_batch,), device=self._device)

            # If we are forward folding, then cat_t should be 1 (i.e. data, sequence)
            # If we are inverse folding or codesign then cat_t should be uniform
            cat_t = (
                forward_fold_mask * forward_fold_cat_t
                + (1 - forward_fold_mask) * normal_cat_t
            )

            # If we are inverse folding, then structure_t should be 1 (i.e. data, structure)
            # If we are forward folding or codesign then structure_t should be uniform
            structure_t = (
                inverse_fold_mask * inverse_fold_structure_t
                + (1 - inverse_fold_mask) * normal_structure_t
            )

            so3_t = structure_t[:, None]
            r3_t = structure_t[:, None]
            cat_t = cat_t[:, None]

        else:
            t = self.sample_t(num_batch)[:, None]
            so3_t = t
            r3_t = t
            cat_t = t
        noisy_batch[nbp.so3_t] = so3_t
        noisy_batch[nbp.r3_t] = r3_t
        noisy_batch[nbp.cat_t] = cat_t

        # Apply corruptions

        if self.cfg.trans.corrupt:
            trans_t = self._corrupt_trans(trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError("NaN in trans_t during corruption")
        noisy_batch[nbp.trans_t] = trans_t

        if self.cfg.rots.corrupt:
            rotmats_t = self._corrupt_rotmats(rotmats_1, so3_t, res_mask, diffuse_mask)
        else:
            rotmats_t = rotmats_1
        if torch.any(torch.isnan(rotmats_t)):
            raise ValueError("NaN in rotmats_t during corruption")
        noisy_batch[nbp.rotmats_t] = rotmats_t

        if self.cfg.aatypes.corrupt:
            aatypes_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask, diffuse_mask)
        else:
            aatypes_t = aatypes_1
        noisy_batch[nbp.aatypes_t] = aatypes_t
        noisy_batch[nbp.trans_sc] = torch.zeros_like(trans_1)
        noisy_batch[nbp.aatypes_sc] = torch.zeros_like(aatypes_1)[..., None].repeat(
            1, 1, self.num_tokens
        )
        return noisy_batch

    def rot_sample_kappa(self, t):
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            return 1 - torch.exp(-t * self.cfg.rots.exp_rate)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            return t
        else:
            raise ValueError(f"Invalid schedule: {self.cfg.rots.sample_schedule}")

    def _trans_vector_field(self, t, trans_1, trans_t):
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

    def _trans_euler_step(self, d_t, t, trans_1, trans_t):
        assert d_t >= 0
        trans_vf = self._trans_vector_field(t, trans_1, trans_t)
        return trans_t + trans_vf * d_t

    def _rots_euler_step(self, d_t, t, rotmats_1, rotmats_t):
        if self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.linear:
            scaling = 1 / (1 - t)
        elif self.cfg.rots.sample_schedule == InterpolantRotationsScheduleEnum.exp:
            scaling = self.cfg.rots.exp_rate
        else:
            raise ValueError(f"Unknown sample schedule {self.cfg.rots.sample_schedule}")
        # TODO: Add in SDE.
        return so3_utils.geodesic_t(scaling * d_t, rotmats_1, rotmats_t)

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
        # TODO replace with torch._scatter
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten(),
        ] = 0.0
        # adjust the probabilities at the current positions to be the negative sum of all other values in the row
        step_probs[
            torch.arange(batch_size, device=device).repeat_interleave(num_res),
            torch.arange(num_res, device=device).repeat(batch_size),
            aatypes_t.long().flatten(),
        ] = (
            1.0 - torch.sum(step_probs, dim=-1).flatten()
        )
        # clamp the probabilities in `step_probs` to the range [0.0, 1.0] to ensure valid probability values.
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
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        device = logits_1.device

        if (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        ):
            assert S == 21

            mask_one_hot = torch.zeros((S,), device=device)
            mask_one_hot[MASK_TOKEN_INDEX] = 1.0

            logits_1[:, :, MASK_TOKEN_INDEX] = -1e9

            pt_x1_probs = F.softmax(
                logits_1 / self.cfg.aatypes.temp, dim=-1
            )  # (B, D, S)

            aatypes_t_is_mask = (
                (aatypes_t == MASK_TOKEN_INDEX).view(batch_size, num_res, 1).float()
            )
            step_probs = (
                d_t * pt_x1_probs * ((1 + self.cfg.aatypes.noise * t) / ((1 - t)))
            )  # (B, D, S)
            step_probs += (
                d_t
                * (1 - aatypes_t_is_mask)
                * mask_one_hot.view(1, 1, -1)
                * self.cfg.aatypes.noise
            )

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(
                batch_size, num_res
            )
        elif (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.uniform
        ):
            assert S == 20
            assert (
                aatypes_t.max() < 20
            ), "No UNK tokens allowed in the uniform sampling step!"

            pt_x1_probs = F.softmax(
                logits_1 / self.cfg.aatypes.temp, dim=-1
            )  # (B, D, S)

            pt_x1_eq_xt_prob = torch.gather(
                pt_x1_probs, dim=-1, index=aatypes_t.long().unsqueeze(-1)
            )  # (B, D, 1)
            assert pt_x1_eq_xt_prob.shape == (batch_size, num_res, 1)

            N = self.cfg.aatypes.noise
            step_probs = d_t * (
                pt_x1_probs * ((1 + N + N * (S - 1) * t) / (1 - t))
                + N * pt_x1_eq_xt_prob
            )

            step_probs = self._regularize_step_probs(step_probs, aatypes_t)

            return torch.multinomial(step_probs.view(-1, S), num_samples=1).view(
                batch_size, num_res
            )
        else:
            raise ValueError(
                f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
            )

    def _aatypes_euler_step_purity(self, d_t, t, logits_1, aatypes_t):
        """
        Perform an Euler step with a focus on maintaining "purity" by selectively unmasking and updating
        amino acid types.

        This function is designed specifically for the "masking" interpolant type, where S = 21. It prioritizes
        unmasking residues based on their maximum log-probabilities and re-masks some positions with a probability
        proportional to the noise term.
        """
        batch_size, num_res, S = logits_1.shape
        assert aatypes_t.shape == (batch_size, num_res)
        assert S == 21
        assert (
            self.cfg.aatypes.interpolant_type
            == InterpolantAATypesInterpolantTypeEnum.masking
        )
        device = logits_1.device

        logits_1_wo_mask = logits_1[:, :, 0:-1]  # (B, D, S-1)
        pt_x1_probs = F.softmax(
            logits_1_wo_mask / self.cfg.aatypes.temp, dim=-1
        )  # (B, D, S-1)
        # step_probs = (d_t * pt_x1_probs * (1/(1-t))).clamp(max=1) # (B, D, S-1)
        max_logprob = torch.max(torch.log(pt_x1_probs), dim=-1)[0]  # (B, D)
        # bias so that only currently masked positions get chosen to be unmasked
        max_logprob = max_logprob - (aatypes_t != MASK_TOKEN_INDEX).float() * 1e9
        sorted_max_logprobs_idcs = torch.argsort(
            max_logprob, dim=-1, descending=True
        )  # (B, D)

        unmask_probs = (
            d_t * ((1 + self.cfg.aatypes.noise * t) / (1 - t)).to(device)
        ).clamp(
            max=1
        )  # scalar

        number_to_unmask = torch.binomial(
            count=torch.count_nonzero(aatypes_t == MASK_TOKEN_INDEX, dim=-1).float(),
            prob=unmask_probs,
        )
        unmasked_samples = torch.multinomial(
            pt_x1_probs.view(-1, S - 1), num_samples=1
        ).view(batch_size, num_res)

        # Vectorized version of:
        # for b in range(B):
        #     for d in range(D):
        #         if d < number_to_unmask[b]:
        #             aatypes_t[b, sorted_max_logprobs_idcs[b, d]] = unmasked_samples[b, sorted_max_logprobs_idcs[b, d]]

        D_grid = torch.arange(num_res, device=device).view(1, -1).repeat(batch_size, 1)
        mask1 = (D_grid < number_to_unmask.view(-1, 1)).float()
        inital_val_max_logprob_idcs = (
            sorted_max_logprobs_idcs[:, 0].view(-1, 1).repeat(1, num_res)
        )
        masked_sorted_max_logprobs_idcs = (
            mask1 * sorted_max_logprobs_idcs + (1 - mask1) * inital_val_max_logprob_idcs
        ).long()
        mask2 = torch.zeros((batch_size, num_res), device=device)
        mask2.scatter_(
            dim=1,
            index=masked_sorted_max_logprobs_idcs,
            src=torch.ones((batch_size, num_res), device=device),
        )
        unmask_zero_row = (number_to_unmask == 0).view(-1, 1).repeat(1, num_res).float()
        mask2 = mask2 * (1 - unmask_zero_row)
        aatypes_t = aatypes_t * (1 - mask2) + unmasked_samples * mask2

        # re-mask
        u = torch.rand(batch_size, num_res, device=self._device)
        re_mask_mask = (u < d_t * self.cfg.aatypes.noise).float()
        aatypes_t = aatypes_t * (1 - re_mask_mask) + MASK_TOKEN_INDEX * re_mask_mask

        return aatypes_t

    def sample(
        self,
        num_batch: int,
        num_res: int,
        model,
        num_timesteps=None,
        trans_0=None,
        rotmats_0=None,
        aatypes_0=None,
        trans_1=None,
        rotmats_1=None,
        aatypes_1=None,
        diffuse_mask=None,
        chain_idx=None,
        res_idx=None,
        t_nn=None,
        forward_folding: bool = False,
        inverse_folding: bool = False,
        separate_t: bool = False,
    ):
        """
        Generate samples using the learned vector fields.
        Generates initialy noise batch, samples using Euler steps for rotations, translations, and amino acid types.
        Special handling for forward folding, reverse folding, and separate t.
        """

        res_mask = torch.ones(num_batch, num_res, device=self._device)

        # Set-up initial prior samples (noise)

        if trans_0 is None:
            # Generate centered Gaussian noise for translations (shape: [num_batch, num_res, 3])
            trans_0 = (
                _centered_gaussian(num_batch, num_res, self._device) * NM_TO_ANG_SCALE
            )
        if rotmats_0 is None:
            # Generate uniform SO(3) rotation matrices (shape: [num_batch, num_res, 3, 3])
            rotmats_0 = _uniform_so3(num_batch, num_res, self._device)
        if aatypes_0 is None:
            # Generate initial amino acid types based on the interpolant type
            if (
                self.cfg.aatypes.interpolant_type
                == InterpolantAATypesInterpolantTypeEnum.masking
            ):
                aatypes_0 = _masked_categorical(num_batch, num_res, self._device)
            elif (
                self.cfg.aatypes.interpolant_type
                == InterpolantAATypesInterpolantTypeEnum.uniform
            ):
                aatypes_0 = _uniform_categorical(
                    num_batch, num_res, self.num_tokens, self._device
                )
            else:
                raise ValueError(
                    f"Unknown aatypes interpolant type {self.cfg.aatypes.interpolant_type}"
                )
        if res_idx is None:
            # Generate residue indices (shape: [num_batch, num_res])
            # Each generated sample will be the same length
            res_idx = torch.arange(num_res, device=self._device, dtype=torch.float32)[
                None
            ].repeat(num_batch, 1)

        if chain_idx is None:
            # Default: assumes monomer
            chain_idx = res_mask

        if diffuse_mask is None:
            # Default: diffuse all residues
            diffuse_mask = res_mask

        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device
        )

        batch = {
            bp.res_mask: res_mask,
            bp.diffuse_mask: diffuse_mask,
            bp.chain_idx: chain_idx,
            bp.res_idx: res_idx,
            nbp.trans_sc: trans_sc,
            nbp.aatypes_sc: aatypes_sc,
        }

        # Set-up time
        if num_timesteps is None:
            num_timesteps = self.cfg.sampling.num_timesteps
        ts = torch.linspace(self.cfg.min_t, 1.0, num_timesteps)
        t_1 = ts[0]

        # Initialize t_1 values for translations, rotations, and aatypes
        if trans_1 is None:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(
                num_batch, num_res, 1, 1
            )
        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()

        logits_1 = torch.nn.functional.one_hot(
            aatypes_1, num_classes=self.num_tokens
        ).float()

        if forward_folding:
            assert aatypes_1 is not None
            assert self.cfg.aatypes.noise == 0.0  # cfg sanity check
        if forward_folding and separate_t:
            aatypes_0 = aatypes_1
        if inverse_folding:
            assert trans_1 is not None
            assert rotmats_1 is not None
        if inverse_folding and separate_t:
            trans_0 = trans_1
            rotmats_0 = rotmats_1

        # Helper function to get structure from residue frames
        frames_to_atom37 = (
            lambda trans, rots: all_atom.atom37_from_trans_rot(
                trans=trans,
                rots=rots,
                res_mask=res_mask,
            )
            .detach()
            .cpu()
        )

        # We will integrate in a loop over ts (handling the last step after the loop)
        # t_1 is the current time, t_2 is the next time

        trans_t_1, rotmats_t_1, aatypes_t_1 = trans_0, rotmats_0, aatypes_0
        # prot_traj tracks predicted intermediate states integrating from t=0 to t=1
        prot_traj = [
            (frames_to_atom37(trans_t_1, rotmats_t_1), aatypes_0.detach().cpu())
        ]
        # clean_traj tracks states predicted by model without noise / processing
        clean_traj = []
        for t_2 in ts[1:]:
            if self.cfg.trans.corrupt:
                batch[nbp.trans_t] = trans_t_1
            else:
                if trans_1 is None:
                    raise ValueError("Must provide trans_1 if not corrupting.")
                batch[nbp.trans_t] = trans_1

            if self.cfg.rots.corrupt:
                batch[nbp.rotmats_t] = rotmats_t_1
            else:
                if rotmats_1 is None:
                    raise ValueError("Must provide rotmats_1 if not corrupting.")
                batch[nbp.rotmats_t] = rotmats_1

            if self.cfg.aatypes.corrupt:
                batch[nbp.aatypes_t] = aatypes_t_1
            else:
                if aatypes_1 is None:
                    raise ValueError("Must provide aatype if not corrupting.")
                batch[nbp.aatypes_t] = aatypes_1

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

            if forward_folding and separate_t:
                batch[nbp.cat_t] = (1 - self.cfg.min_t) * torch.ones_like(
                    batch[nbp.cat_t]
                )
            if inverse_folding and separate_t:
                batch[nbp.r3_t] = (1 - self.cfg.min_t) * torch.ones_like(
                    batch[nbp.r3_t]
                )
                batch[nbp.so3_t] = (1 - self.cfg.min_t) * torch.ones_like(
                    batch[nbp.so3_t]
                )

            # Get model output at translations/rotations/aatypes respective `t`
            with torch.no_grad():
                model_out = model(batch)

            # Process model output, getting predicted structure/sequence
            pred_trans_1 = model_out[pbp.pred_trans]
            pred_rotmats_1 = model_out[pbp.pred_rotmats]
            pred_aatypes_1 = model_out[pbp.pred_aatypes]
            pred_logits_1 = model_out[pbp.pred_logits]
            clean_traj.append(
                (
                    frames_to_atom37(pred_trans_1, pred_rotmats_1),
                    pred_aatypes_1.detach().cpu(),
                )
            )
            if forward_folding:
                pred_logits_1 = 100.0 * logits_1
            if inverse_folding:
                pred_trans_1 = trans_1
                pred_rotmats_1 = rotmats_1

            if self.cfg.self_condition:
                batch[nbp.trans_sc] = _trans_diffuse_mask(
                    pred_trans_1, trans_1, diffuse_mask
                )
                if forward_folding:
                    batch[nbp.aatypes_sc] = logits_1
                else:
                    # TODO - bug? use _aatypes_diffuse_mask? Or just using because 2D?
                    batch[nbp.aatypes_sc] = _trans_diffuse_mask(
                        pred_logits_1, logits_1, diffuse_mask
                    )

            # Take reverse step
            # Move from current structure/sequence at t_1, taking d_t Euler step toward t_2
            d_t = t_2 - t_1
            trans_t_2 = self._trans_euler_step(d_t, t_1, pred_trans_1, trans_t_1)
            rotmats_t_2 = self._rots_euler_step(d_t, t_1, pred_rotmats_1, rotmats_t_1)

            # Update amino acid types, allowing for "purity", i.e. unmasking and re-masking
            if self.cfg.aatypes.do_purity:
                aatypes_t_2 = self._aatypes_euler_step_purity(
                    d_t, t_1, pred_logits_1, aatypes_t_1
                )
            else:
                aatypes_t_2 = self._aatypes_euler_step(
                    d_t, t_1, pred_logits_1, aatypes_t_1
                )

            # Only update the masked residues
            trans_t_2 = _trans_diffuse_mask(trans_t_2, trans_1, diffuse_mask)
            rotmats_t_2 = _rots_diffuse_mask(rotmats_t_2, rotmats_1, diffuse_mask)
            aatypes_t_2 = _aatypes_diffuse_mask(aatypes_t_2, aatypes_1, diffuse_mask)

            # Add to trajectory
            prot_traj.append(
                (frames_to_atom37(trans_t_2, rotmats_t_2), aatypes_t_2.cpu().detach())
            )

            # Get ready for the next step
            trans_t_1, rotmats_t_1, aatypes_t_1 = trans_t_2, rotmats_t_2, aatypes_t_2
            t_1 = t_2

        # We only integrated to min_t, so need to make a final step
        # and save the trajectory + final structure

        t_1 = ts[-1]

        if self.cfg.trans.corrupt:
            batch[nbp.trans_t] = trans_t_1
        else:
            if trans_1 is None:
                raise ValueError("Must provide trans_1 if not corrupting.")
            batch[nbp.trans_t] = trans_1

        if self.cfg.rots.corrupt:
            batch[nbp.rotmats_t] = rotmats_t_1
        else:
            if rotmats_1 is None:
                raise ValueError("Must provide rotmats_1 if not corrupting.")
            batch[nbp.rotmats_t] = rotmats_1

        if self.cfg.aatypes.corrupt:
            # Note - was 'aatype_t' in multiflow code, assume a bug
            batch[nbp.aatypes_t] = aatypes_t_1
        else:
            if aatypes_1 is None:
                raise ValueError("Must provide aatype if not corrupting.")
            # Note - was 'aatype_t' in multiflow code, assume a bug
            batch[nbp.aatypes_t] = aatypes_1

        with torch.no_grad():
            model_out = model(batch)
        pred_trans_1 = model_out[pbp.pred_trans]
        pred_rotmats_1 = model_out[pbp.pred_rotmats]
        pred_aatypes_1 = model_out[pbp.pred_aatypes]

        if forward_folding:
            pred_aatypes_1 = aatypes_1
        if inverse_folding:
            pred_trans_1 = trans_1
            pred_rotmats_1 = rotmats_1

        pred_atom37 = frames_to_atom37(pred_trans_1, pred_rotmats_1)
        clean_traj.append((pred_atom37, pred_aatypes_1.detach().cpu()))
        prot_traj.append((pred_atom37, pred_aatypes_1.detach().cpu()))

        return prot_traj, clean_traj
