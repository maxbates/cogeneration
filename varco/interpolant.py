import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from cogeneration.data import so3_utils
from cogeneration.data.const import MASK_TOKEN_INDEX, NM_TO_ANG_SCALE
from cogeneration.data.rigid import batch_center_of_mass
from varco.config import (
    VarcoHazardConfig,
    VarcoHazardKind,
    VarcoInterpolantConfig,
    VarcoMotifGuidanceType,
)
from varco.coupling import Coupler
from varco.coupling_aatypes import AATypesCoupler, AATypesCoupling
from varco.coupling_rots import RotationCoupler, RotationCoupling
from varco.coupling_trans import TranslationCoupler, TranslationCoupling
from varco.data import (
    DataBatch,
    DataBridged,
    DataCorrupted,
    SampleTrajectory,
    Trajectory,
)
from varco.model import BranchFlowModel
from varco.tensor_utils import SeededRNG, gather_and_pad, to_device


@dataclass
class TreeCouplings:
    """Container for all domain couplings from a single corruption call."""

    translation: TranslationCoupling
    aatypes: AATypesCoupling
    rotation: RotationCoupling


@dataclass
class TreeInterpolant:
    cfg: VarcoInterpolantConfig
    device: torch.device = torch.device("cpu")
    min_t: float = 0.01
    translation_coupler: Coupler[TranslationCoupling] = field(init=False)
    aatypes_coupler: Coupler[AATypesCoupling] = field(init=False)
    rotation_coupler: RotationCoupler = field(init=False)

    def __post_init__(self):
        self.translation_coupler = TranslationCoupler(cfg=self.cfg.trans_coupler)
        self.aatypes_coupler = AATypesCoupler(cfg=self.cfg.aatypes_coupler)
        self.rotation_coupler = RotationCoupler(cfg=self.cfg.rotation_coupler)

    def set_device(self, device: torch.device):
        self.device = device
        self.rotation_coupler.set_device(device)  # for IGSO3 device

    def seed_all(self, seed: int):
        if seed is None:
            return
        torch.manual_seed(int(seed))
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))

    def compute_motif_guidance_vf(
        self,
        t: torch.Tensor,  # (B,)
        pred_trans_1: torch.Tensor,  # (B, P, 3)
        trans_1_motifs: torch.Tensor,  # (B, P, 3) true motif positions
        pred_rotmats_1: torch.Tensor,  # (B, P, 3, 3)
        rotmats_t: torch.Tensor,  # (B, P, 3, 3)
        rotmats_1_motifs: torch.Tensor,  # (B, P, 3, 3) true motif rotations
        motif_mask: torch.Tensor,  # (B, P)
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Compute guidance velocity fields pulling motif positions toward their targets.

        Returns (trans_guidance_vf, rotmats_guidance_vf), each (B, P, 3) or None.
        Returns (None, None) if guidance is disabled or no motifs present.
        """
        guidance_cfg = self.cfg.motif_guidance
        if not guidance_cfg.enabled or not motif_mask.any():
            return None, None

        B, P = motif_mask.shape
        t_clamped = t.clamp(min=1e-3, max=1.0 - 1e-3)

        # Compute scale based on config
        if guidance_cfg.scale_type == VarcoMotifGuidanceType.posterior_variance:
            # scale = 0.5 * g² / ω² where g = κ/t, ω² = κ²/(t² + κ²), κ = 1-t
            # see cogeneration interpolant for details
            kappa = 1.0 - t_clamped
            g = kappa / t_clamped
            omega2 = kappa**2 / (t_clamped**2 + kappa**2)
            scale = 0.5 * g * g / omega2
            scale = scale.clamp(min=0.0, max=guidance_cfg.var_scale_cap)

            if guidance_cfg.var_decay:
                # Decay to 0 as t approaches guidance_end_t
                decay = (1.0 - t_clamped / guidance_cfg.guidance_end_t).clamp(min=0.0)
                scale = scale * decay
        elif guidance_cfg.scale_type == VarcoMotifGuidanceType.linear_decay:
            scale = guidance_cfg.linear_decay_strength * (1.0 - t_clamped)
        else:
            raise ValueError(f"Unknown scale_type: {guidance_cfg.scale_type}")

        # --- Translation guidance ---
        trans_guidance_vf = (trans_1_motifs - pred_trans_1) * scale.view(B, 1, 1)
        trans_guidance_vf = trans_guidance_vf * motif_mask.unsqueeze(-1).float()

        # Cap per-residue magnitude
        if guidance_cfg.max_step_force_ang > 0:
            norm = trans_guidance_vf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            trans_guidance_vf = trans_guidance_vf * (
                guidance_cfg.max_step_force_ang / norm
            ).clamp(max=1.0)

        # --- Rotation guidance ---
        # Compute rotation vector fields in tangent space
        rot_vf_to_target = so3_utils.calc_rot_vf(
            mat_t=rotmats_t, mat_1=rotmats_1_motifs
        )
        rot_vf_to_pred = so3_utils.calc_rot_vf(mat_t=rotmats_t, mat_1=pred_rotmats_1)

        # Guidance = scale * (target_vf - pred_vf)
        rotmats_guidance_vf = (rot_vf_to_target - rot_vf_to_pred) * scale.view(B, 1, 1)
        rotmats_guidance_vf = rotmats_guidance_vf * motif_mask.unsqueeze(-1).float()

        # Cap per-residue rotation magnitude (in radians)
        if guidance_cfg.max_rot_step_force_rad > 0:
            norm = rotmats_guidance_vf.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            rotmats_guidance_vf = rotmats_guidance_vf * (
                guidance_cfg.max_rot_step_force_rad / norm
            ).clamp(max=1.0)

        return trans_guidance_vf, rotmats_guidance_vf

    def pack_bridged_states(
        self,
        batch: DataBatch,
        t: torch.Tensor,  # (B,)
        trans_t: torch.Tensor,  # (B, A, 3)
        rotmats_t: torch.Tensor,  # (B, A, 3, 3)
        aatypes_t: torch.Tensor,  # (B, A)
    ) -> DataBridged:
        """
        Pack static batch fields + per-domain tree-aligned states into a DataBridged.
        """
        tree = batch.tree.to(self.device)
        trans_1 = batch.trans_1.to(self.device)
        rotmats_1 = batch.rotmats_1.to(self.device)
        present_mask = tree.present_mask(t=t)

        res_mask = tree.broadcast_to_leaves(
            x=batch.res_mask.to(self.device), fill_value=0
        )
        chain_idx = tree.broadcast_to_leaves(
            x=batch.chain_idx.to(self.device), fill_value=0
        )
        res_bfactor = tree.broadcast_to_leaves(
            x=batch.res_bfactor.to(self.device), fill_value=0.0
        )
        res_plddt = tree.broadcast_to_leaves(
            x=batch.res_plddt.to(self.device), fill_value=0.0
        )
        contact_conditioning = tree.broadcast_to_leaves(
            x=batch.contact_conditioning.to(self.device), fill_value=0.0, is_2d=True
        )

        trans_1_motifs = tree.broadcast_to_leaves(x=trans_1, fill_value=0.0)
        trans_1_motifs = trans_1_motifs * tree.motif_mask.unsqueeze(-1).float()

        identity = torch.eye(3, device=self.device, dtype=rotmats_1.dtype)
        rotmats_1_motifs = tree.broadcast_to_leaves(x=rotmats_1, fill_value=identity)
        rotmats_1_motifs = torch.where(
            tree.motif_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
            rotmats_1_motifs,
            identity.unsqueeze(0).unsqueeze(0).expand_as(rotmats_1_motifs),
        )

        # Recenter translations to maintain translation invariance
        trans_t = trans_t - batch_center_of_mass(trans_t, mask=present_mask)[:, None]

        bridged = DataBridged(
            t=t,
            present_mask=present_mask,
            motif_mask=tree.motif_mask,
            birth_time=tree.birth_time,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_t=trans_t,
            rotmats_t=rotmats_t,
            aatypes_t=aatypes_t,
            trans_1_motifs=trans_1_motifs,
            rotmats_1_motifs=rotmats_1_motifs,
            remaining_insertions=tree.remaining_insertions_t(t=t),
            deleted=tree.leaf_deleted,
            planar_position=tree.planar_position,
            res_bfactor=res_bfactor,
            res_plddt=res_plddt,
            contact_conditioning=contact_conditioning,
        )
        bridged.validate()
        return bridged

    def corrupt_to(
        self,
        batch: DataBatch,
        t: torch.Tensor,  # (B,)
        trans_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
        rotmats_0: Optional[torch.Tensor] = None,
    ) -> Tuple[DataBridged, TreeCouplings]:
        """Corrupt a batch to time t."""
        tree = batch.tree.to(self.device)
        t = t.to(self.device)

        # Corrupt domains to time t

        trans_t, trans_coupling = self.translation_coupler.corrupt(
            tree=tree,
            t=t,
            x1=batch.trans_1.to(self.device),
            x0=to_device(trans_0, self.device),
        )
        trans_coupling.validate()

        rotmats_t, rotation_coupling = self.rotation_coupler.corrupt(
            tree=tree,
            t=t,
            x1=batch.rotmats_1.to(self.device),
            x0=to_device(rotmats_0, self.device),
        )
        rotation_coupling.validate()

        aatypes_t, aatypes_coupling = self.aatypes_coupler.corrupt(
            tree=tree,
            t=t,
            x1=batch.aatypes_1.to(self.device),
            x0=to_device(aatypes_0, self.device),
        )
        aatypes_coupling.validate()

        couplings = TreeCouplings(
            translation=trans_coupling,
            aatypes=aatypes_coupling,
            rotation=rotation_coupling,
        )

        bridged = self.pack_bridged_states(
            batch=batch,
            t=t,
            trans_t=trans_t,
            rotmats_t=rotmats_t,
            aatypes_t=aatypes_t,
        )

        return bridged, couplings

    def corrupt_batch(self, batch: DataBatch) -> Tuple[DataBridged, TreeCouplings]:
        """
        Corrupt a batch to a shared time. Bias later using `t_corrupt_exp < 1.0`.
        Pick a single time to share across the batch,
        simply so they have a similar number of insertion/deletions to simulate
        since corruption is run across the batch.
        """
        shared_t = torch.rand(1, device=self.device) ** self.cfg.t_corrupt_exp
        shared_t = shared_t.clamp(min=self.min_t, max=1.0 - self.min_t)
        t = torch.ones(batch.trans_1.shape[0], device=self.device) * shared_t  # (B,)
        return self.corrupt_to(batch=batch, t=t)

    def corrupt_trajectory(
        self,
        batch: DataBatch,
        times: Optional[List[float]] = None,
        seed: Optional[int] = None,
        trans_0: Optional[torch.Tensor] = None,
        aatypes_0: Optional[torch.Tensor] = None,
        rotmats_0: Optional[torch.Tensor] = None,
    ) -> Tuple[Trajectory, TreeCouplings]:
        """Generate a time-coupled corruption trajectory"""
        self.set_device(batch.trans_1.device)
        self.seed_all(seed)

        B = batch.trans_1.shape[0]
        tree = batch.tree.to(self.device)
        trans_1 = batch.trans_1.to(self.device)
        rotmats_1 = batch.rotmats_1.to(self.device)
        aatypes_1 = batch.aatypes_1.to(self.device)

        if times is None:
            times = list(np.linspace(0.0, 1.0, 50))
        if len(times) == 0:
            raise ValueError("times must be non-empty")
        times = [float(np.clip(t, self.min_t, 1.0 - self.min_t)) for t in times]

        # corrupt to t_build to get couplings for trajectory
        t_build = float(times[-1])
        t_build_tensor = torch.ones(B, device=self.device) * t_build

        # Define consistent base samples for the whole trajectory (in aligned space)
        if trans_0 is None:
            trans_0 = self.translation_coupler.sample_base(
                motif_mask=tree.motif_mask,
                x1=tree.broadcast_to_leaves(trans_1, fill_value=0),
                device=self.device,
            )
        if rotmats_0 is None:
            rotmats_0 = self.rotation_coupler.sample_base(
                motif_mask=tree.motif_mask,
                x1=tree.broadcast_to_leaves(
                    rotmats_1, fill_value=torch.eye(3, device=self.device)
                ),
                device=self.device,
            )
        if aatypes_0 is None:
            aatypes_0 = self.aatypes_coupler.sample_base(
                motif_mask=tree.motif_mask,
                x1=tree.broadcast_to_leaves(aatypes_1, fill_value=MASK_TOKEN_INDEX),
                device=self.device,
            )

        # Build couplings once (anchors + creation states)
        _, trans_coupling = self.translation_coupler.corrupt(
            tree=tree,
            t=t_build_tensor,
            x1=trans_1,
            x0=to_device(trans_0, self.device),
        )
        _, rotmats_coupling = self.rotation_coupler.corrupt(
            tree=tree,
            t=t_build_tensor,
            x1=rotmats_1,
            x0=to_device(rotmats_0, self.device),
        )
        _, aatypes_coupling = self.aatypes_coupler.corrupt(
            tree=tree,
            t=t_build_tensor,
            x1=aatypes_1,
            x0=to_device(aatypes_0, self.device),
        )
        couplings = TreeCouplings(
            translation=trans_coupling,
            aatypes=aatypes_coupling,
            rotation=rotmats_coupling,
        )

        # Start from creation states (defined at birth_time for each node), and step forward.
        trans_cur = trans_coupling.creation_state
        rotmats_cur = rotmats_coupling.creation_state
        aatypes_cur = aatypes_coupling.creation_state
        t_prev = 0.0

        # Iterate through time, bridging current state to next timepoint
        samples: List[DataCorrupted] = []
        for t_val in tqdm(times, desc="corrupt_trajectory()", leave=False):
            trans_cur = self.translation_coupler.bridge_step(
                coupling=trans_coupling,
                x_prev=trans_cur,
                t_prev=t_prev,
                t_next=t_val,
            )
            rotmats_cur = self.rotation_coupler.bridge_step(
                coupling=rotmats_coupling,
                x_prev=rotmats_cur,
                t_prev=t_prev,
                t_next=t_val,
            )
            aatypes_cur = self.aatypes_coupler.bridge_step(
                coupling=aatypes_coupling,
                x_prev=aatypes_cur,
                t_prev=t_prev,
                t_next=t_val,
            )

            t_tensor = torch.ones(B, device=self.device) * float(t_val)
            bridged = self.pack_bridged_states(
                batch=batch,
                t=t_tensor,
                trans_t=trans_cur,
                rotmats_t=rotmats_cur,
                aatypes_t=aatypes_cur,
            )
            samples.append(bridged.pack_present())
            t_prev = float(t_val)

        return Trajectory(samples=samples), couplings

    @staticmethod
    def _sample_initial_positions(
        motif_mask: torch.Tensor,  # (B, N)
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample initial positions for branching flow sampling.

        For each batch element:
        - All motif positions (motif_mask == True) are included
        - Each scaffold span (contiguous motif_mask == False) contributes K sampled roots,
          where K is sampled uniformly from [min_scaffold_nuclei, min(max_scaffold_nuclei, span_len)]

        Returns:
            init_length: (B,) long - number of initial positions per batch element
            motif_idx: (B, P_max) long - source index in original (N,) data for motif positions,
                       -1 for scaffold root positions. Padding is implicit: positions >= init_length.
        """
        device = motif_mask.device
        B, N = motif_mask.shape
        motif_mask = motif_mask.bool()

        # Use SeededRNG for determinism
        rng = SeededRNG(seed=seed, device="cpu")

        # For each batch element, compute indices and whether they map to original data
        # motif_idx_val >= 0 means motif (value is source index), -1 means scaffold root
        init_indices_list: List[List[int]] = []

        for b in range(B):
            mask_b = motif_mask[b]  # (N,)
            # indices stores: source index for motifs, -1 for scaffold roots
            indices: List[int] = []

            i = 0
            while i < N:
                if mask_b[i].item():
                    # Motif position - include with source index
                    indices.append(i)
                    i += 1
                else:
                    # Scaffold span - find extent
                    span_start = i
                    while i < N and not mask_b[i].item():
                        i += 1
                    span_len = i - span_start

                    # Sample K roots for this span
                    k_hi = min(max_scaffold_nuclei, span_len)
                    k_lo = min(min_scaffold_nuclei, k_hi)
                    target_k = (
                        k_lo if k_lo == k_hi else (k_lo + rng.rand_int(k_hi - k_lo + 1))
                    )

                    # Add -1 for each scaffold root (they don't map to original data)
                    for _ in range(target_k):
                        indices.append(-1)

            init_indices_list.append(indices)

        # Compute init_length and P_max
        init_lengths = [len(indices) for indices in init_indices_list]
        P_max = max(init_lengths) if init_lengths else 0

        # Build output tensors
        init_length = torch.tensor(
            init_lengths, dtype=torch.long, device=device
        )  # (B,)
        # Padding value is -1, same as scaffold roots, but distinguished by position >= init_length
        motif_idx = torch.full((B, P_max), -1, dtype=torch.long, device=device)

        for b in range(B):
            L = init_lengths[b]
            motif_idx[b, :L] = torch.tensor(
                init_indices_list[b], dtype=torch.long, device=device
            )

        return init_length, motif_idx

    def _init_sampling_batch(
        self,
        data: DataBatch,
        min_scaffold_nuclei: int = 1,
        max_scaffold_nuclei: int = 10,
        seed: Optional[int] = None,
    ) -> DataCorrupted:
        """
        Initialize a batch of samples for sampling.

        Uses _sample_initial_positions to determine which positions to include at t=0:
        - Motif positions are gathered from data (res_mask, chain_idx)
        - Scaffold roots get fresh samples from base prior
        """
        device = self.device
        B, N = data.motif_mask.shape

        # Get initial position layout
        init_length, motif_idx = self._sample_initial_positions(
            motif_mask=data.motif_mask.to(device),
            min_scaffold_nuclei=min_scaffold_nuclei,
            max_scaffold_nuclei=max_scaffold_nuclei,
            seed=seed,
        )

        # Set up masks / indices
        P_max = motif_idx.shape[1]
        pos_idx = torch.arange(P_max, device=device).unsqueeze(0)  # (1, P_max)
        valid_mask = pos_idx < init_length.unsqueeze(1)  # (B, P_max)
        is_motif = (motif_idx >= 0) & valid_mask  # (B, P_max)

        motif_mask = is_motif

        birth_time = torch.full((B, P_max), float("inf"), device=device)
        birth_time[valid_mask] = 0.0

        # Gather some features from data using motif_idx (clamp to 0 for valid gather idx)
        # For scaffold roots (motif_idx=-1) and padding, fill with 0
        gather_idx = motif_idx.clamp(min=0)  # (B, P_max)
        res_mask = gather_and_pad(
            data.res_mask.to(device), gather_idx, is_motif, fill_value=0
        )
        chain_idx = gather_and_pad(
            data.chain_idx.to(device), gather_idx, is_motif, fill_value=0
        )

        # contact_conditioning: (B, N, N) -> (B, P_max, P_max), zeros for scaffolds
        contact_conditioning = gather_and_pad(
            data.contact_conditioning.to(device),
            gather_idx,
            is_motif,
            fill_value=0.0,
            is_2d=True,
        )
        # confident confidence!
        res_plddt = gather_and_pad(
            data.res_plddt.to(device), gather_idx, is_motif, fill_value=90.0
        )
        res_bfactor = None

        # Gather x1 values from data for motif positions (fill scaffolds with placeholder)
        trans_1_gathered = gather_and_pad(
            data.trans_1.to(device), gather_idx, is_motif, fill_value=0.0
        )
        rotmats_1_gathered = gather_and_pad(
            data.rotmats_1.to(device),
            gather_idx,
            is_motif,
            fill_value=torch.eye(3, device=device, dtype=data.rotmats_1.dtype),
        )
        aatypes_1_gathered = gather_and_pad(
            data.aatypes_1.to(device), gather_idx, is_motif, fill_value=MASK_TOKEN_INDEX
        )

        # Sample base distributions using coupler interfaces
        trans_0 = self.translation_coupler.sample_base(
            motif_mask=is_motif, x1=trans_1_gathered, device=device
        )
        rotmats_0 = self.rotation_coupler.sample_base(
            motif_mask=is_motif, x1=rotmats_1_gathered, device=device
        )
        aatypes_0 = self.aatypes_coupler.sample_base(
            motif_mask=is_motif, x1=aatypes_1_gathered, device=device
        )

        # init batch at t=0
        t = torch.zeros((B,), dtype=torch.float32, device=device)

        return DataCorrupted(
            t=t,
            motif_mask=motif_mask,
            birth_time=birth_time,
            res_mask=res_mask,
            chain_idx=chain_idx,
            trans_t=trans_0,
            rotmats_t=rotmats_0,
            aatypes_t=aatypes_0,
            trans_1_motifs=trans_1_gathered,
            rotmats_1_motifs=rotmats_1_gathered,
            contact_conditioning=contact_conditioning,
            res_bfactor=res_bfactor,
            res_plddt=res_plddt,
        )

    @staticmethod
    def _sample_insert_delete_substitute(
        split_rate: torch.Tensor,  # (B, P)
        del_logits: torch.Tensor,  # (B, P)
        is_root: torch.Tensor,  # (B, P) bool
        valid_mask: torch.Tensor,  # (B, P) bool
        t_val: float,  # current time
        dt: float,
        split_hazard: VarcoHazardConfig,
        delete_hazard: VarcoHazardConfig,
        pred_split_pooled_log1p_rate: Optional[torch.Tensor] = None,  # (B,)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample insert/delete/substitute events for present positions in a batch.

        Uses exact CTMC interval probabilities instead of g(t)*dt approximation:
        - Insertions: p = 1 - exp(-R * I) where I = integral_t^{t+dt} g(u) du
        - Deletions: p = 1 - (1 - p_final*H(t+dt)) / (1 - p_final*H(t))

        For insertions (counting process with K remaining events):
        - split_rate predicts E[remaining insertions]
        - optionally calibrated using the pooled prediction
        - exact integral I of g(t) = h(t)/S(t) over [t, t+dt]

        For deletions (binary event: will/won't be deleted by t=1):
        - del_logits predicts P(deleted by t=1) via sigmoid
        - exact conditional interval probability from CTMC theory
        - this ensures the cumulative deletion probability by t=1 equals p_final
        """
        eps = 1e-6
        t = max(eps, min(1.0 - eps, float(t_val)))
        t_next = min(1.0 - eps, t + float(dt))

        # Calibrate per-token split_rate using the pooled prediction
        split_rate = split_rate.clamp_min(0.0)  # (B, P)
        if pred_split_pooled_log1p_rate is not None:
            token_sum = (split_rate * valid_mask.float()).sum(dim=1)  # (B,)
            # pooled prediction is in log1p space: log(1 + total_remaining)
            pooled_total = torch.expm1(pred_split_pooled_log1p_rate).clamp_min(
                0.0
            )  # (B,)
            # if per-token sum exceeds pooled, scale down
            scale = pooled_total / token_sum.clamp_min(0.1)  # (B,)
            # allow up to 10% over-prediction before scaling kicks in
            # and cap at 10x scale down
            scale = scale.clamp(min=0.1, max=1.1)
            split_rate = split_rate * scale.unsqueeze(1)  # (B, P)

        # I_split = integral_t^{t_next} g(u) du, closed form for each hazard
        p_split = float(max(1, split_hazard.power))

        if split_hazard.kind == VarcoHazardKind.uniform:
            # g(u) = 1/(1-u), integral = -log(1-u)
            I_split = math.log((1.0 - t) / max(1e-12, 1.0 - t_next))
        elif split_hazard.kind == VarcoHazardKind.early_power:
            # g(u) = p/(1-u), integral = -p*log(1-u)
            I_split = p_split * math.log((1.0 - t) / max(1e-12, 1.0 - t_next))
        elif split_hazard.kind == VarcoHazardKind.late_power:
            # H(u) = u^p, g(u) = p*u^(p-1)/(1-u^p), integral = -log(1-u^p)
            tp = t**p_split
            tnp = t_next**p_split
            I_split = math.log((1.0 - tp) / max(1e-12, 1.0 - tnp))
        else:
            raise ValueError(f"Unknown split hazard kind: {split_hazard.kind!r}")

        # CTMC-matching per-step event probability: 1 - exp(-R * I_split)
        # technically there could be multiple events, we check for 1.
        lam_ins = split_rate * float(I_split)
        p_ins = (1.0 - torch.exp(-lam_ins.clamp_max(10.0))).clamp(0.0, 0.95)

        # --- Deletions (binary eventual event) ---
        # p_final = sigmoid(del_logits) is P(deleted by t=1)
        #
        # For a binary deletion modulated by hazard H(t):
        #   lambda(t) = p_final * h(t) / (1 - p_final * H(t))
        #
        # Exact conditional interval probability:
        #   p_del(t -> t+dt) = 1 - (1 - p_final*H(t_next)) / (1 - p_final*H(t))

        p_del_final = torch.sigmoid(del_logits).clamp(eps, 1.0 - eps)
        p_del_pow = float(max(1, delete_hazard.power))

        # H(t) for the chosen hazard family
        if delete_hazard.kind == VarcoHazardKind.uniform:
            H_t = t
            H_tn = t_next
        elif delete_hazard.kind == VarcoHazardKind.early_power:
            H_t = 1.0 - (1.0 - t) ** p_del_pow
            H_tn = 1.0 - (1.0 - t_next) ** p_del_pow
        elif delete_hazard.kind == VarcoHazardKind.late_power:
            H_t = t**p_del_pow
            H_tn = t_next**p_del_pow
        else:
            raise ValueError(f"Unknown delete hazard kind: {delete_hazard.kind!r}")

        # Exact conditional interval probability
        denom = (1.0 - p_del_final * H_t).clamp_min(eps)
        numer = (1.0 - p_del_final * H_tn).clamp_min(0.0)
        p_del = (1.0 - (numer / denom)).clamp(0.0, 0.95)

        insertions = torch.rand_like(p_ins) < p_ins
        deletions = torch.rand_like(p_del) < p_del
        insertions = insertions & valid_mask
        deletions = deletions & valid_mask

        # Resolve conflicts: if both sampled, mark substitution instead
        substitutions = insertions & deletions
        insertions = insertions & ~substitutions
        deletions = deletions & ~substitutions

        return insertions, deletions, substitutions

    def sample(
        self,
        model: BranchFlowModel,
        data: DataBatch,
        num_steps: int = 200,
        traj_frames: Optional[int] = None,
    ) -> SampleTrajectory:
        device = self.device

        # Create initial batch, which we edit in-place through the trajectory
        num_batch, _ = data.motif_mask.shape
        batch = self._init_sampling_batch(data=data)

        traj = SampleTrajectory()
        traj.samples.append(batch.detach_clone(device=torch.device("cpu")))

        t_end = float(1.0 - self.min_t)
        model.eval()
        with torch.no_grad():
            t_grid = torch.linspace(0.0, t_end, steps=num_steps + 1, device=device)

            pbar = tqdm(range(num_steps), total=num_steps, desc="Sampling", leave=False)
            for step_num in pbar:
                t_val = float(t_grid[step_num].item())
                t_next = float(t_grid[step_num + 1].item())
                dt = float(max(1e-6, t_next - t_val))

                # Set current time and predict
                batch.t = torch.full(
                    (num_batch,), t_val, dtype=torch.float32, device=device
                )
                pred = model.forward(batch)

                if traj_frames is None or step_num % traj_frames == 0:
                    traj.pred.append(pred.detach_clone(device=torch.device("cpu")))

                # Compute motif guidance VFs
                trans_guidance_vf, rotmats_guidance_vf = self.compute_motif_guidance_vf(
                    t=batch.t,
                    pred_trans_1=pred.pred_trans_1,
                    trans_1_motifs=batch.trans_1_motifs,
                    pred_rotmats_1=pred.pred_rotmats_1,
                    rotmats_t=batch.rotmats_t,
                    rotmats_1_motifs=batch.rotmats_1_motifs,
                    motif_mask=batch.motif_mask,
                )

                # Euler steps for domains

                trans_next = self.translation_coupler.euler_step(
                    x_t=batch.trans_t,
                    x1_pred=pred.pred_trans_1,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                    potential=trans_guidance_vf,
                )
                batch.trans_t = trans_next

                rotmats_next = self.rotation_coupler.euler_step(
                    x_t=batch.rotmats_t,
                    x1_pred=pred.pred_rotmats_1,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                    potential=rotmats_guidance_vf,
                )
                batch.rotmats_t = rotmats_next

                aatypes_next = self.aatypes_coupler.euler_step(
                    x_t=batch.aatypes_t,
                    x1_pred=pred.pred_aatype_logits,
                    t=batch.t,
                    dt=dt,
                    birth_time=batch.birth_time,
                    motif_mask=batch.motif_mask,
                )
                batch.aatypes_t = aatypes_next

                # Disallowed in motifs
                scaffold_mask = batch.valid_mask & ~batch.motif_mask  # (B, P)
                is_root = batch.birth_time <= 0.0  # (B, P)

                # sample indels
                insertions, deletions, _ = self._sample_insert_delete_substitute(
                    split_rate=pred.pred_split_rate,
                    del_logits=pred.pred_del_logits,
                    is_root=is_root,
                    valid_mask=scaffold_mask,
                    t_val=t_val,
                    dt=dt,
                    split_hazard=self.cfg.sampling.split_hazard,
                    delete_hazard=self.cfg.sampling.delete_hazard,
                    pred_split_pooled_log1p_rate=pred.pred_split_pooled_log1p_rate,
                )

                # Enforce max_length: block insertions once we're at the limit
                max_len = self.cfg.sampling.max_length
                cur_lens = batch.valid_mask.sum(dim=1)  # (B,)
                at_limit = cur_lens >= max_len  # (B,)
                if at_limit.any():
                    insertions = insertions & ~at_limit.unsqueeze(1)

                batch, insert_mask, gather_idx = batch.apply_insertions_deletions(
                    insertions=insertions,
                    deletions=deletions,
                    t_birth=t_next,  # born at t_next since after euler step
                )

                # Domain-specific initialization for newly inserted tokens.
                # TODO - use couplings for domain-specific corruptions
                if insert_mask.any():
                    # Add small isotropic perturbation to inserted translations
                    trans_noise = torch.randn_like(batch.trans_t) * (
                        0.1 * NM_TO_ANG_SCALE
                    )
                    batch.trans_t = (
                        batch.trans_t + insert_mask.unsqueeze(-1).float() * trans_noise
                    )

                    # Add small IGSO3 perturbation to inserted rotations
                    # Inserted positions inherit parent's rotation from apply_insertions_deletions
                    # Add noise to break symmetry
                    B_new, P_new = batch.rotmats_t.shape[:2]
                    self.rotation_coupler._ensure_igso3_device(device)
                    # Use small sigma for perturbation
                    sigma_insert = torch.full(
                        (B_new,),
                        0.1,
                        device=self.rotation_coupler.igso3.sigma_grid.device,
                    )
                    insert_noise = self.rotation_coupler.igso3.sample(
                        sigma_insert, P_new
                    ).to(
                        device
                    )  # (B, P, 3, 3)
                    # Apply noise to inserted positions only
                    rotmats_with_noise = torch.einsum(
                        "...ij,...jk->...ik", batch.rotmats_t, insert_noise
                    )
                    batch.rotmats_t = torch.where(
                        insert_mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3),
                        rotmats_with_noise,
                        batch.rotmats_t,
                    )

                    # Sample amino acids for inserted positions from parent's insertion logits
                    K = pred.pred_insertion_logits.shape[-1]
                    P_old = pred.pred_insertion_logits.shape[1]

                    # Gather parent's insertion logits for new positions
                    insertion_logits_gathered = pred.pred_insertion_logits.gather(
                        1,
                        gather_idx.unsqueeze(-1).expand(-1, -1, K).clamp(0, P_old - 1),
                    )  # (B, P_new, K)

                    # Convert to probs and mix in uniform noise (like _compute_step_probs)
                    probs = F.softmax(
                        insertion_logits_gathered, dim=-1
                    )  # (B, P_new, K)
                    uniform_dist = torch.ones_like(probs) / K
                    probs = (
                        1.0 - self.cfg.aatypes_coupler.noise_scale
                    ) * probs + self.cfg.aatypes_coupler.noise_scale * uniform_dist

                    # Sample from noisy distribution
                    sampled_tokens = torch.multinomial(
                        probs.view(-1, K), num_samples=1
                    ).view(B_new, P_new)
                    batch.aatypes_t = torch.where(
                        insert_mask, sampled_tokens, batch.aatypes_t
                    )

                # Recenter translations to maintain translation invariance
                # Everything is "present" in sampling,so use valid_mask
                com = batch_center_of_mass(batch.trans_t, mask=batch.valid_mask)
                batch.trans_t = batch.trans_t - com[:, None, :]

                # Save
                if traj_frames is None or step_num % traj_frames == 0:
                    traj.samples.append(batch.detach_clone(device=torch.device("cpu")))

                # Update progress bar with batch dimensions
                B, P = batch.trans_t.shape[:2]
                pbar.set_postfix_str(f"B={B} P={P}")

                # Cleanup
                if step_num % 10 == 0:
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()

            pbar.close()

            # Final endpoint step: take the model's endpoint prediction after integrating to t=1-min_t.
            # No motif guidance or insertions/deletions are applied in this step.
            t_val = float(t_grid[-1].item())
            batch.t = torch.full(
                (num_batch,), t_val, dtype=torch.float32, device=device
            )
            pred = model.forward(batch)
            traj.pred.append(pred.detach_clone(device=torch.device("cpu")))

            # take predicted translations and rotations, without motif guidance
            batch.trans_t = pred.pred_trans_1
            batch.rotmats_t = pred.pred_rotmats_1

            # Sample final-interval (t_end -> 1.0) deletions, not insertions.
            _, final_deletions, _ = self._sample_insert_delete_substitute(
                split_rate=torch.zeros_like(pred.pred_split_rate),
                del_logits=pred.pred_del_logits,
                is_root=batch.birth_time <= 0.0,
                valid_mask=batch.valid_mask & ~batch.motif_mask,
                t_val=t_val,
                dt=1.0 - t_val,
                split_hazard=self.cfg.sampling.split_hazard,
                delete_hazard=self.cfg.sampling.delete_hazard,
                pred_split_pooled_log1p_rate=None,
            )
            if final_deletions.any():
                zeros = torch.zeros_like(final_deletions)
                batch, _, _ = batch.apply_insertions_deletions(
                    insertions=zeros,
                    deletions=final_deletions,
                    t_birth=t_end,
                )

            # recenter
            batch.t = torch.ones((num_batch,), dtype=torch.float32, device=device)
            com = batch_center_of_mass(batch.trans_t, mask=batch.valid_mask)
            batch.trans_t = batch.trans_t - com[:, None, :]

            # save final sample
            traj.samples.append(batch.detach_clone(device=torch.device("cpu")))

        return traj
