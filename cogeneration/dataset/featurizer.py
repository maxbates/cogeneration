import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from cogeneration.config.base import DatasetConfig
from cogeneration.data import data_transforms, rigid_utils
from cogeneration.data.const import seq_to_aatype
from cogeneration.data.noise_mask import mask_blend_2d
from cogeneration.data.protein import chain_str_to_int
from cogeneration.dataset.contacts import get_contact_conditioning_matrix
from cogeneration.dataset.interaction import MultimerInteractions
from cogeneration.dataset.motif_factory import (
    ChainBreak,
    Motif,
    MotifFactory,
    Scaffold,
    Segment,
)
from cogeneration.type.batch import METADATA_BATCH_PROPS, BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import empty_feats
from cogeneration.type.dataset import DatasetCSVRow
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import DatasetTransformColumn as dtc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataCSVRow, ProcessedFile, RedesignColumn
from cogeneration.type.structure import StructureExperimentalMethod
from cogeneration.type.task import DataTask


@dataclass
class BatchFeaturizer:
    """
    Helper for generating BatchFeatures from a ProcessedFile.
    Requires as input an already-loaded `ProcessedFile` and already-merged DatasetCSVRow.
    """

    cfg: DatasetConfig
    task: DataTask
    eval: bool
    motif_factory: Optional[MotifFactory] = None

    def __post_init__(self):
        self._rng = np.random.default_rng(seed=self.cfg.seed)
        self._rng_fixed = np.random.default_rng(seed=42069)

        if self.motif_factory is None:
            self.motif_factory = MotifFactory(
                cfg=self.cfg.inpainting,
                rng=self._rng if self.is_training else self._rng_fixed,
            )

    @property
    def is_training(self) -> bool:
        """
        Returns whether the current batch is for training.
        """
        return not self.eval

    @staticmethod
    def randomize_chains(feats: BatchFeatures):
        """
        Randomly permute chain indices; new chain IDs start at 1.
        """
        chain_idx = feats[bp.chain_idx]

        # identify unique original chain IDs (sorted)
        all_chain_ids = torch.unique(chain_idx).tolist()
        # sample a random permutation
        permuted = random.sample(all_chain_ids, len(all_chain_ids))
        # remap to new IDs starting at 1
        # Note that public MultiFlow simply started at 1 but retained gaps in IDs. We just re-index.
        new_chain_idx = torch.zeros_like(chain_idx)
        for new_id, orig_id in enumerate(permuted, start=1):
            new_chain_idx[chain_idx == orig_id] = new_id

        feats[bp.chain_idx] = new_chain_idx.long()

    @staticmethod
    def reset_residue_index(feats: BatchFeatures):
        """
        Re-number `res_index`, assuming `feats` is flat i.e. a single sample.
        - starting from 1 within each chain
        - strictly increasing within a chain (occasionally, PDBs have duplicate residue index)
        - preserve chain gaps from input PDB
        """
        chain_idx = feats[bp.chain_idx]  # (N,)
        res_idx = feats[bp.res_idx]  # (N,)
        new_res_idx = torch.zeros_like(res_idx)

        for cid in torch.unique(chain_idx):
            chain_mask = chain_idx == cid
            chain_res = res_idx[chain_mask]
            # 1 for duplicates
            dup_flags = torch.diff(chain_res, prepend=chain_res[:1]).le(0).long()
            # running count of duplicates
            dup_offset = dup_flags.cumsum(0)
            # shift everything forward by duplicates
            fixed_idx = chain_res + dup_offset
            # start at 1 by substracting minimum value in chain
            fixed_idx -= fixed_idx.min() - 1
            new_res_idx[chain_mask] = fixed_idx

        feats[bp.res_idx] = new_res_idx

    @staticmethod
    def recenter_structure(
        feats: BatchFeatures,
        mask: Optional[torch.Tensor] = None,
    ):
        """
        Centers the structure to be translation invariant.

        `mask` defaults to `feats[bp.res_mask]` if not provided.
        For inpainting, may want `motif_mask`.

        Note, complete structure should already be centered on load by `parse_pdb_feats`.
        """
        if mask is None:
            mask = feats[bp.res_mask]
            trans_1 = feats[bp.trans_1]
        else:
            trans_1 = feats[bp.trans_1] * mask[:, None]

        center_of_mass = torch.sum(trans_1, dim=0) / torch.sum(mask + 1e-5)

        # modify in place
        feats[bp.trans_1] -= center_of_mass[None, :]

    @staticmethod
    def segment_features(
        feats: BatchFeatures,
        segments: List[Segment],
    ) -> BatchFeatures:
        """
        Apply `Motif` and `Scaffold` and `ChainBreak` (`Segment` instances) to `feats`.

        Preserves the feats in motifs, and masks them in scaffolds, and extends scaffolds to target length.
        i.e. has the effect of masking `trans_1`, `rotmats_1`, `aatypes_1` etc. in scaffolds.

        Resets `chain_idx` to match Segments provided.
        Expected to reset indices after calling this function, though they should be valid.
        """
        N_old = feats[bp.res_mask].shape[0]
        device = feats[bp.res_mask].device
        N_new = sum([seg.length for seg in segments])

        new_feats = empty_feats(N=N_new, task=DataTask.inpainting)

        # copy over 0D metadata features from original
        for prop in METADATA_BATCH_PROPS:
            if prop in feats:
                new_feats[prop] = feats[prop]

        # There are several indices we need to track:
        # 1) indices in original `feats`
        # 2) indices in `new_feats`, because scaffolds may have different lengths
        # 3) updating `chain_idx` and `res_idx`.
        #    `Motif` specify a `chain` or `chain_id`, but only as the source of their information.
        #       In the context of mapping features, all chains are flat features, so just use start/end positions.
        #    `Scaffold` just continue the current chain
        #    `ChainBreak` specify the beginning of a new chain, i.e. `chain_idx+1`
        #   `res_idx` != (1) or (2) if there is a chain break.
        #
        # We'll iterate through the segments, and build up a mapping from old to new indices.
        # Some fields we'll set explicitly while looping through segments, like `diffuse_mask` / `motif_mask`.
        # We'll manually track `chain_idx` / `res_idx`, and increment / reset when we encounter `ChainBreak`.

        # track old -> new indices, for re-mapping 1D and 2D features
        # -1 for scaffold / dropped positions
        old_to_new = torch.full(
            (N_old,), fill_value=-1, device=device, dtype=torch.long
        )
        # track residues in motifs
        motif_old_idx = []
        # track cursor position in `new_feats`
        new_feats_start = 0
        # track current chain (1-indexed)
        current_chain_idx = 1
        # track res_idx (1-indexed per chain)
        current_res_idx = 1

        for segment in segments:
            # chain break only bumps `current_chain_idx`.
            # it should not modify new_feats or `feats` / `new_feats` indices
            if isinstance(segment, ChainBreak):
                current_chain_idx += 1  # increment
                current_res_idx = 1  # reset
                continue

            # for motifs / scaffolds, compute original and new indices
            os, oe, l = segment.start, segment.start + segment.length, segment.length
            ns, ne = new_feats_start, new_feats_start + l

            if isinstance(segment, Motif):
                # Motifs mostly preserve data; track indices.
                motif_range = torch.arange(os, oe, device=device)
                motif_old_idx.append(motif_range)
                old_to_new[motif_range] = torch.arange(ns, ne, device=device)

                # mark motif
                new_feats[bp.motif_mask][ns:ne] = 1
                # mark diffusable for inpainting with guidance
                # TODO(inpainting-fixed) mark fixed (and ensure translations etc. fixed)
                new_feats[bp.diffuse_mask][ns:ne] = 1
                # enforce idx
                new_feats[bp.chain_idx][ns:ne] = current_chain_idx
                new_feats[bp.res_idx][ns:ne] = torch.arange(
                    current_res_idx, current_res_idx + l
                )

            elif isinstance(segment, Scaffold):
                # Scaffolds mostly retain default `empty_feats` values; don't track.

                # mark scaffold
                new_feats[bp.motif_mask][ns:ne] = 0
                # ensure marked to be diffused
                new_feats[bp.diffuse_mask][ns:ne] = 1
                # enforce idx
                new_feats[bp.chain_idx][ns:ne] = current_chain_idx
                new_feats[bp.res_idx][ns:ne] = torch.arange(
                    current_res_idx, current_res_idx + l
                )

            else:
                raise ValueError(f"Unknown segment type: {type(segment)}")

            # advance cursors by segment length
            new_feats_start += segment.length
            current_res_idx += segment.length

        # No motifs, nothing to map
        if len(motif_old_idx) == 0:
            return new_feats

        # Flatten motif mappings
        motif_old_idx = torch.cat(motif_old_idx)
        motif_new_idx = old_to_new[motif_old_idx]

        # Map features using new indices
        for prop in list(new_feats.keys()):
            # skip metadata / undefined features
            if prop in METADATA_BATCH_PROPS or prop not in feats:
                continue

            # skip features we have already handled
            if (
                prop == bp.motif_mask
                or prop == bp.diffuse_mask
                or prop == bp.chain_idx
                or prop == bp.res_idx
            ):
                continue

            if feats[prop] is None:
                raise ValueError(f"Feature {prop} is None")
            if prop not in new_feats:
                raise ValueError(f"Feature {prop} not in new_feats")

            # Handle motif 2D features specially, i.e. pairwise contact conditioning features
            if prop == bp.contact_conditioning:
                cc_motifs = feats[bp.contact_conditioning][
                    motif_old_idx[:, None], motif_old_idx[None, :]
                ]
                new_feats[bp.contact_conditioning][
                    motif_new_idx[:, None], motif_new_idx[None, :]
                ] = cc_motifs
                continue

            # Otherwise, 1D feature
            try:
                new_feats[prop][motif_new_idx] = feats[prop][
                    motif_old_idx
                ]  # scatter motif values
            except Exception as e:
                print(
                    f"Error scattering {prop} from {motif_old_idx} to {motif_new_idx}, {feats[prop].shape} to {new_feats[prop].shape}, {feats[prop].dtype} to {new_feats[prop].dtype}, {feats[prop].device} to {new_feats[prop].device}, {e}"
                )
                raise e

        return new_feats

    def hot_spots_define(
        self,
        feats: BatchFeatures,
        hot_spots_str: Optional[str],
    ) -> torch.Tensor:
        """
        Define hot spots mask from metadata and current feature indices.

        Args:
            feats: BatchFeatures containing chain_idx and res_idx
            hot_spots_str: string defining hot spots, in format "<chain_id>:<res_index>:<num_interactions>,..." e.g. "A:10:2,"

        Returns:
            Hot spots mask tensor
        """
        hot_spots_mask = torch.zeros_like(feats[bp.res_mask]).int()

        # Disabled some fraction of the time during training
        if (
            self.is_training
            and random.random() < self.cfg.hotspots.hotspots_prob_disabled
        ):
            return torch.zeros_like(feats[bp.res_mask]).int()

        # Disabled if not multimer
        if feats[bp.chain_idx].unique().numel() <= 1:
            return torch.zeros_like(feats[bp.res_mask]).int()

        # if hot spots not defined in metadata, calculate them
        if hot_spots_str is None or hot_spots_str == "":
            multimer_interactions = MultimerInteractions.from_batch_feats(feats=feats)
            hot_spots_str = multimer_interactions.serialize_hot_spots()

        hot_spots = []
        for item in hot_spots_str.split(","):
            # our format is "<chain_id><res_index>:<num_interactions>,..."
            # but also support  "<chain_id><res_index>" format
            if ":" in item:
                item, num_interactions = item.split(":")
            else:
                num_interactions = 1
            hot_spots.append(
                (chain_str_to_int(item[0]), int(item[1:]), int(num_interactions))
            )

        # Map hot spots to current feature indices
        for hot_chain_id, hot_res_index, num_interactions in hot_spots:
            mask = (feats[bp.chain_idx] == hot_chain_id) & (
                feats[bp.res_idx] == hot_res_index
            )
            hot_spots_mask[mask] = 1

        # Ignore hot spots not in res_mask
        hot_spots_mask = hot_spots_mask * feats[bp.res_mask]

        # Apply subsampling during training if requested
        if (
            self.is_training
            and len(hot_spots) >= self.cfg.hotspots.min_hotspots_threshold
        ):
            # Subsample hot spots
            num_hot_spots = hot_spots_mask.sum().item()
            if num_hot_spots > 0:
                target_fraction = random.uniform(
                    self.cfg.hotspots.min_hotspot_fraction,
                    self.cfg.hotspots.max_hotspot_fraction,
                )
                target_count = max(1, int(num_hot_spots * target_fraction))

                # Get indices of hot spots
                hot_indices = torch.where(hot_spots_mask == 1)[0]

                # Randomly sample subset
                if len(hot_indices) > target_count:
                    sampled_indices = hot_indices[
                        torch.randperm(len(hot_indices))[:target_count]
                    ]
                    new_mask = torch.zeros_like(feats[bp.res_mask]).int()
                    new_mask[sampled_indices] = 1
                    hot_spots_mask = new_mask

        return hot_spots_mask

    def contact_conditioning_define(
        self,
        feats: BatchFeatures,
        task: DataTask,
    ) -> torch.Tensor:
        """
        Define contact conditioning matrix from features, for a single item.
        """
        N = feats[bp.trans_1].shape[0]
        empty_conditioning = torch.zeros(N, N, device=feats[bp.trans_1].device)

        # Return empty matrix during training some proportion of the time
        if random.random() < self.cfg.contact_conditioning.conditioning_prob_disabled:
            return empty_conditioning

        # Include inter-chain conditioning some proportion of the time in training
        include_inter_chain = (
            random.random() < self.cfg.contact_conditioning.include_inter_chain_prob
        )
        # Downsample inter-chain contacts to just a few
        downsample_inter_chain = (
            random.random() < self.cfg.contact_conditioning.downsample_inter_chain_prob
        )

        # Limit conditioning to motifs some proportion of the time in training
        motif_only = (
            task == DataTask.inpainting
            and random.random()
            < self.cfg.contact_conditioning.conditioning_prob_motif_only
        )

        return get_contact_conditioning_matrix(
            trans=feats[bp.trans_1],
            res_mask=feats[bp.res_mask],
            motif_mask=feats.get(bp.motif_mask, None),
            chain_idx=feats[bp.chain_idx],
            include_inter_chain=include_inter_chain,
            downsample_inter_chain=downsample_inter_chain,
            downsample_inter_chain_min_contacts=self.cfg.contact_conditioning.downsample_inter_chain_min_contacts,
            downsample_inter_chain_max_contacts=self.cfg.contact_conditioning.downsample_inter_chain_max_contacts,
            motif_only=motif_only,
            min_res_gap=self.cfg.contact_conditioning.min_res_gap,
            min_dist=self.cfg.contact_conditioning.dist_minimum_ang,
            max_dist=self.cfg.contact_conditioning.dist_maximum_ang,
            dist_noise_ang=self.cfg.contact_conditioning.dist_noise_ang,
        )

    @staticmethod
    def batch_features_from_processed_file(
        processed_file: ProcessedFile,
        csv_row: MetadataCSVRow,
        cfg: DatasetConfig,
        processed_file_path: str,
    ) -> BatchFeatures:
        """
        Direct conversion of a ProcessedFile (numpy) into BatchFeatures (tensors, frames).
        Keep as a static method so easy to construct direct batch features from ProcessedFile.

        Does not support inpainting and motif mask generation.
        Defaults to `diffuse_mask` of all ones, no conditioning, etc.
        Does not center, re-index, randomize, etc.

        Note that actual batch items are further processed in `featurize_processed_file`.
        This function is static; that function is stochastic.
        """

        # Check the sequence
        aatypes_1 = processed_file[dpc.aatype]
        num_unique_aatypes = len(set(aatypes_1))
        if num_unique_aatypes <= 1:
            raise ValueError(
                f"Example @ {processed_file_path} has only {num_unique_aatypes} unique amino acids."
            )
        aatypes_1 = torch.tensor(aatypes_1).long()

        # Construct an intermediate `chain_feats` for OpenFold pipeline
        chain_feats = {
            "aatype": aatypes_1,
            "all_atom_positions": torch.tensor(
                processed_file[dpc.atom_positions]
            ).double(),
            "all_atom_mask": torch.tensor(processed_file[dpc.atom_mask]).double(),
        }

        # Add noise to atom position tensor.
        # Features are in angstrom-scale.
        add_noise = cfg.noise_atom_positions_angstroms > 0
        if add_noise:
            atom_position_noise = (
                torch.randn_like(chain_feats["all_atom_positions"])
                * cfg.noise_atom_positions_angstroms
            )
            chain_feats["all_atom_positions"] += atom_position_noise

        # Run through OpenFold data transforms.
        # Convert atomic representation to frames
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        # calculate torsion angles, in case predicting torsion angles
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # Extract rigids (translations + rotations), check for poorly processed
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats[dtc.rigidgroups_gt_frames]
        )[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        torsions_1 = chain_feats[dtc.torsion_angles_sin_cos].float()
        if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
            raise ValueError(f"Found NaNs in {processed_file_path}")

        # res_mask for residues under consideration uses `dpc.bb_mask`
        res_mask = torch.tensor(processed_file[dpc.bb_mask]).int()

        # pull out res_idx and chain_idx
        res_idx = torch.tensor(processed_file[dpc.residue_index])
        chain_idx = torch.tensor(processed_file[dpc.chain_index])

        # Extract method and b factors / pLDDTs (depending on the method)
        bfactors = torch.tensor(processed_file[dpc.b_factors][:, 1]).float()
        method = StructureExperimentalMethod.from_value(
            csv_row.get(
                mc.structure_method, StructureExperimentalMethod.XRAY_DIFFRACTION
            )
        )
        method_feature = StructureExperimentalMethod.to_tensor(method)
        is_experimental = StructureExperimentalMethod.is_experimental(method)
        if is_experimental:
            res_bfactors = bfactors
            # pLDDT 100.0 for true structures
            res_plddt = torch.full_like(bfactors, fill_value=100.0)
        else:
            res_bfactors = torch.zeros_like(bfactors)
            res_plddt = bfactors

        # mask low pLDDT residues for synthetic data
        plddt_mask = torch.ones_like(res_mask)
        if cfg.add_plddt_mask:
            plddt_mask = (res_plddt > cfg.min_plddt_threshold).int()

        # Define some default values, which may be overridden when featurizing the processed file.
        diffuse_mask = torch.ones_like(res_mask).int()
        hot_spots_mask = torch.zeros_like(res_mask).int()
        contact_conditioning = torch.zeros(res_mask.shape[0], res_mask.shape[0])

        feats: BatchFeatures = {
            bp.res_mask: res_mask,
            bp.aatypes_1: aatypes_1,
            bp.trans_1: trans_1,
            bp.rotmats_1: rotmats_1,
            bp.torsions_1: torsions_1,
            bp.chain_idx: chain_idx,
            bp.res_idx: res_idx,
            bp.res_bfactor: res_bfactors,
            bp.res_plddt: res_plddt,
            bp.structure_method: method_feature,
            bp.diffuse_mask: diffuse_mask,
            bp.plddt_mask: plddt_mask,
            bp.hot_spots: hot_spots_mask,
            bp.contact_conditioning: contact_conditioning,
            # bp.motif_mask: None, # inpainting only, defined when featurizing
            # Pass through relevant data from CSV
            bp.pdb_name: csv_row.get(mc.pdb_name, ""),
        }

        return feats

    def featurize_processed_file(
        self,
        processed_file: ProcessedFile,
        csv_row: DatasetCSVRow,
    ) -> BatchFeatures:
        """
        Processes a pickled features from single structure + metadata.
        Converts numpy feats to tensor feats.
        Responsible for masking, noising, centering, re-indexing, etc. to yield a data sample.

        This function should be called as examples are needed, i.e. in `__get_item__`,
        because it adds noise to atom positions, picks motif positions, etc. as defined by cfg.
        """
        # Redesigned sequences can be used to substitute the original sequence during training.
        if self.is_training and RedesignColumn.best_seq in csv_row:
            best_seq = csv_row[RedesignColumn.best_seq]
            if not isinstance(best_seq, str):
                raise ValueError(f"Unexpected value best_seq: {best_seq}")
            best_aatype = np.array(seq_to_aatype(best_seq))
            assert (
                processed_file[dpc.aatype].shape == best_aatype.shape
            ), f"best_seq different length: {processed_file[dpc.aatype].shape} vs {best_aatype.shape}"
            processed_file[dpc.aatype] = best_aatype

        # Construct feats from processed_file.
        feats = BatchFeaturizer.batch_features_from_processed_file(
            processed_file=processed_file,
            csv_row=csv_row,
            cfg=self.cfg,
            processed_file_path=csv_row[mc.processed_path],
        )

        # Update `diffuse_mask` and `motif_mask` depending on task
        if self.task == DataTask.hallucination:
            # everything is diffused, use default `bp.diffuse_mask`
            # feats[bp.motif_mask] = None  # avoid defining motif_mask batch prop
            pass
        elif self.task == DataTask.inpainting:
            # Generate motif_mask from MotifFactory
            motif_mask = self.motif_factory.generate_motif_mask(
                res_mask=feats[bp.res_mask],
                plddt_mask=feats[bp.plddt_mask],
                chain_idx=feats[bp.chain_idx],
                res_idx=feats[bp.res_idx],
                trans_1=feats[bp.trans_1],
                rotmats_1=feats[bp.rotmats_1],
                aatypes_1=feats[bp.aatypes_1],
            )
            feats[bp.motif_mask] = motif_mask.int()

            # `diffuse_mask` for inpainting with guidance is all ones; whole structure is modeled.
            # TODO(inpainting-fixed) `diffuse_mask` is complement to `motif_mask`
            feats[bp.diffuse_mask] = torch.ones_like(feats[bp.res_mask])
        else:
            raise ValueError(f"Unknown task {self.task}")

        # Define subsampled hot spots for multimers
        # TODO(conditioning) - consider how to define hot spots with chain down selection, e.g. drop interacting chain...
        #   currently, should be dropped in scaffolds, but we may specify fewer than we want.
        #   May be easiest to prioritize defining in motif_mask, if defined?
        feats[bp.hot_spots] = self.hot_spots_define(
            feats=feats,
            hot_spots_str=csv_row.get(mc.hot_spots, None),
        )

        # Contact conditioning matrix
        feats[bp.contact_conditioning] = self.contact_conditioning_define(
            feats=feats,
            task=self.task,
        )

        # Ensure have a valid `diffuse_mask` for modeling
        if torch.sum(feats[bp.diffuse_mask]) == 0:
            feats[bp.diffuse_mask] = torch.ones_like(feats[bp.res_mask]).int()

        # For inpainting evaluation (i.e. not training), vary scaffold lengths
        # Using the `diffuse_mask`, modify the scaffold region lengths, and mask out the scaffolds
        # i.e. {trans,rots,aatypes}_1 only defined for motif positions
        if (
            not self.is_training
            and self.task == DataTask.inpainting
            and (feats[bp.motif_mask] == 1).any()
        ):
            segments = self.motif_factory.generate_segments_from_motif_mask(
                motif_mask=feats[bp.motif_mask],
                chain_idx=feats[bp.chain_idx],
            )
            # apply segmenting (note, updates masks and generates new chain_idx)
            feats = BatchFeaturizer.segment_features(feats=feats, segments=segments)

        # Centering
        if bp.motif_mask in feats and (feats[bp.motif_mask] == 1).any():
            # Center the motifs using motif_mask,
            # to prevent biasing how scaffolds are placed relative to motifs.
            BatchFeaturizer.recenter_structure(feats, mask=feats[bp.motif_mask])

            # For training, we need the scaffolds positions for losses.
            # For eval, zero out the segmented scaffolds after re-centering.
            if not self.is_training:
                feats[bp.trans_1] = mask_blend_2d(
                    feats[bp.trans_1],
                    torch.zeros_like(feats[bp.trans_1]),
                    mask=feats[bp.motif_mask],
                )
        else:
            # Center the whole structure
            BatchFeaturizer.recenter_structure(feats)

        # Randomize chains and reset residue positions
        BatchFeaturizer.randomize_chains(feats)
        BatchFeaturizer.reset_residue_index(feats)

        return feats
