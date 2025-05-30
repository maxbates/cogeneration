import logging
import os
import random
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cogeneration.config.base import (
    Config,
    DatasetConfig,
    DatasetTrimMethod,
    InferenceSamplesConfig,
)
from cogeneration.data import data_transforms, rigid_utils
from cogeneration.data.const import seq_to_aatype
from cogeneration.dataset.filterer import DatasetFilterer
from cogeneration.dataset.motif_factory import (
    ChainBreak,
    Motif,
    MotifFactory,
    Scaffold,
    Segment,
)
from cogeneration.dataset.process_pdb import read_processed_file
from cogeneration.type.batch import METADATA_BATCH_PROPS, BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import InferenceFeatures, empty_feats
from cogeneration.type.dataset import DatasetColumn, DatasetCSVRow, DatasetDataFrame
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import DatasetTransformColumn as dtc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import (
    MetadataCSVRow,
    MetadataDataFrame,
    ProcessedFile,
    RedesignColumn,
)
from cogeneration.type.task import DataTask


@dataclass
class BatchFeaturizer:
    """
    Helper for generating BatchFeatures from a ProcessedFile.
    Requires as input an already-loaded `ProcessedFile` and already-merged DatasetCSVRow.
    """

    cfg: DatasetConfig
    task: DataTask
    is_training: bool
    motif_factory: Optional[MotifFactory] = None

    def __post_init__(self):
        self._rng = np.random.default_rng(seed=self.cfg.seed)
        self._rng_fixed = np.random.default_rng(seed=42069)

        if self.motif_factory is None:
            self.motif_factory = MotifFactory(
                cfg=self.cfg.inpainting,
                rng=self._rng if self.is_training else self._rng_fixed,
            )

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

        Resets `chain_idx` to match Segments provided. Expected to reset indices after calling this function.
        """
        new_total_length = sum([seg.length for seg in segments])
        new_feats = empty_feats(N=new_total_length, task=DataTask.inpainting)

        # copy over metadata features from original
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
        # We'll manually track `chain_idx` and increment when we encounter `ChainBreak`.
        # After building up the features, can re-index which will use the new `chain_idx`.

        # track current chain (1-indexed)
        current_chain_idx = 1
        # track res_idx (1-indexed per chain)
        current_res_idx = 1
        # track position in `new_feats`
        new_feats_start = 0

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
                # Motifs mostly preserve data
                for prop in list(new_feats.keys()):
                    if prop in METADATA_BATCH_PROPS or prop not in feats:
                        continue
                    new_feats[prop][ns:ne] = feats[prop][os:oe]

                # mark motif
                new_feats[bp.motif_mask][ns:ne] = 1.0
                # mark diffusable for inpainting with guidance
                # TODO(inpainting-fixed) mark fixed
                new_feats[bp.diffuse_mask][ns:ne] = 1.0
                # enforce idx
                new_feats[bp.chain_idx][ns:ne] = current_chain_idx
                new_feats[bp.res_idx][ns:ne] = torch.arange(
                    current_res_idx, current_res_idx + l
                )

            elif isinstance(segment, Scaffold):
                # Scaffolds mostly retain default `empty_feats` values.

                # mark scaffold
                new_feats[bp.motif_mask][ns:ne] = 0.0
                # ensure marked to be diffused
                new_feats[bp.diffuse_mask][ns:ne] = 1.0
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

        return new_feats

    @staticmethod
    def batch_features_from_processed_file(
        processed_file: ProcessedFile,
        cfg: DatasetConfig,
        processed_file_path: str,
    ) -> BatchFeatures:
        """
        Direct conversion of a ProcessedFile (numpy) into BatchFeatures (tensors, frames).
        Keep as a static method so easy to construct direct batch features from ProcessedFile.

        Does not support inpainting and motif mask generation.
        Defaults to `diffuse_mask` of all ones.
        Does not center, re-index, randomize, etc.
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
        if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
            raise ValueError(f"Found NaNs in {processed_file_path}")

        # res_mask for residues under consideration uses `dpc.bb_mask`
        res_mask = torch.tensor(processed_file[dpc.bb_mask]).int()

        # pull out res_idx and chain_idx
        res_idx = torch.tensor(processed_file[dpc.residue_index])
        chain_idx = torch.tensor(processed_file[dpc.chain_index])

        # Mask low pLDDT residues
        res_plddt = torch.tensor(processed_file[dpc.b_factors][:, 1])
        plddt_mask = torch.ones_like(res_mask)
        if cfg.add_plddt_mask:
            plddt_mask = (res_plddt > cfg.min_plddt_threshold).int()

        # Default diffuse mask is all ones
        diffuse_mask = torch.ones_like(res_mask).int()

        feats: BatchFeatures = {
            bp.res_mask: res_mask,
            bp.aatypes_1: aatypes_1,
            bp.trans_1: trans_1,
            bp.rotmats_1: rotmats_1,
            bp.torsions_1: chain_feats[dtc.torsion_angles_sin_cos],
            bp.chain_idx: chain_idx,
            bp.res_idx: res_idx,
            bp.res_plddt: res_plddt,
            bp.diffuse_mask: diffuse_mask,
            bp.plddt_mask: plddt_mask,
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

        This function should be called as examples are needed, i.e. in `__get_item__`,
        because it adds noise to atom positions, picks motif positions, etc. as defined by cfg.
        """
        # Redesigned sequences can be used to substitute the original sequence during training.
        if self.is_training and self.cfg.use_redesigned:
            assert (
                RedesignColumn.best_seq in csv_row
            ), f"cfg.use_redesigned but redesign column {RedesignColumn.best_seq} not in CSV row {csv_row}"
            best_seq = csv_row[RedesignColumn.best_seq]
            if not isinstance(best_seq, str):
                raise ValueError(f"Unexpected value best_seq: {best_seq}")

            best_aatype = np.array(seq_to_aatype(best_seq))
            processed_file[dpc.aatype] = best_aatype

            assert (
                processed_file[dpc.aatype].shape == best_aatype.shape
            ), f"best_seq different length: {processed_file[dpc.aatype].shape} vs {best_aatype.shape}"

        # Construct feats from processed_file.
        feats = BatchFeaturizer.batch_features_from_processed_file(
            processed_file=processed_file,
            cfg=self.cfg,
            processed_file_path=csv_row[mc.processed_path],
        )

        # Pass through relevant data from CSV
        feats[bp.pdb_name] = csv_row[mc.pdb_name]

        # Update `diffuse_mask` and `motif_mask` depending on task
        if self.task == DataTask.hallucination:
            # everything is diffused
            feats[bp.diffuse_mask] = torch.ones_like(feats[bp.res_mask])
            # feats[bp.motif_mask] = None  # skip defining motif_mask batch prop
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

            # For inpainting evaluation (i.e. not training), vary scaffold lengths
            # Using the `diffuse_mask`, modify the scaffold region lengths, and mask out the scaffolds
            # i.e. {trans,rots,aatypes}_1 only defined for motif positions
            if not self.is_training and (feats[bp.motif_mask] == 1).any():
                segments = self.motif_factory.generate_segments_from_motif_mask(
                    motif_mask=feats[bp.motif_mask],
                    chain_idx=feats[bp.chain_idx],
                )
                # apply segmenting (note, updates masks and generates new chain_idx)
                feats = BatchFeaturizer.segment_features(feats=feats, segments=segments)
        else:
            raise ValueError(f"Unknown task {self.task}")

        # Ensure have a valid `diffuse_mask` for modeling
        if torch.sum(feats[bp.diffuse_mask]) == 0:
            feats[bp.diffuse_mask] = torch.ones_like(feats[bp.res_mask]).int()

        # Centering
        if bp.motif_mask in feats and (feats[bp.motif_mask] == 1).any():
            # Center the motifs using motif_mask
            BatchFeaturizer.recenter_structure(feats, mask=feats[bp.motif_mask])
        else:
            # Center the whole structure
            BatchFeaturizer.recenter_structure(feats)

        # Randomize chains and reset residue positions
        BatchFeaturizer.randomize_chains(feats)
        BatchFeaturizer.reset_residue_index(feats)

        return feats


def _read_clusters(cluster_path: Union[Path, str], synthetic=False) -> Dict[str, Any]:
    pdb_to_cluster = {}
    with open(cluster_path, "r") as f:
        for i, line in enumerate(f):
            for chain in line.split(" "):
                if not synthetic:
                    pdb = chain.split("_")[0].strip()
                else:
                    pdb = chain.strip()
                pdb_to_cluster[pdb.upper()] = i
    return pdb_to_cluster


def read_metadata_file(
    metadata_path: Union[Path, str],
    max_rows: Optional[int] = None,
) -> MetadataDataFrame:
    """
    Reads a metadata CSV file and returns a DataFrame.
    """
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Metadata file {metadata_path} does not exist")
    metadata_df = pd.read_csv(metadata_path, nrows=max_rows)
    return metadata_df


class BaseDataset(Dataset):
    def __init__(
        self,
        *,
        cfg: DatasetConfig,
        is_training: bool,
        task: DataTask,
    ):
        """
        BaseDataset collects all dataset files into a single Dataset, and yields samples.
        Metadata CSV defines the dataset, and `ProcessedFile` samples are read when yielded,
        and featurized by `BatchFeaturizer` into `BatchFeatures`.

        MetadataDataFrame (collection of MetadataCSVRow) is loaded from `cfg.csv_path`.
        The dataset is filtered by `DatasetFilterer` according to `cfg.filter`.

        If `cfg.use_redesigned`, redesigned sequences (e.g. from ProteinMPNN)
        can substitute the original sequence during training.

        if `cfg.use_synthetic`, synthetic data (e.g. from AlphaFold)
        can be used to augment the dataset.

        For inpainting, `MotifFactory` generates motifs and scaffolds.
        This happens during training (scaffolds are masked) *and* eval (scaffold lengths vary).
        """
        self.cfg = cfg
        self.task = task
        self._is_training = is_training

        self._log = logging.getLogger(__name__)
        self._cache = {}

        self.featurizer = BatchFeaturizer(
            cfg=cfg,
            task=task,
            is_training=self.is_training,
        )

        # Process structures and clusters
        self.raw_csv = read_metadata_file(
            metadata_path=self.cfg.csv_path,
            max_rows=cfg.debug_head_samples,
        )
        self._log.debug(f"Loaded {len(self.raw_csv)} examples from {self.cfg.csv_path}")

        # Initial filtering
        dataset_filterer = DatasetFilterer(
            cfg=cfg.filter,
            modeled_trim_method=cfg.modeled_trim_method,
        )
        metadata_csv = dataset_filterer.filter_metadata(self.raw_csv)

        # Concat redesigned data, if provided
        if self.cfg.use_redesigned:
            assert os.path.exists(
                self.cfg.redesigned_csv_path
            ), f"Redesigned CSV path {self.cfg.redesigned_csv_path} does not exist"

            self.redesigned_csv = pd.read_csv(self.cfg.redesigned_csv_path)
            metadata_csv = metadata_csv.merge(
                self.redesigned_csv,
                left_on=mc.pdb_name,
                right_on=RedesignColumn.example,
            )
            # Filter out examples with high RMSD
            metadata_csv = metadata_csv[
                metadata_csv[RedesignColumn.best_rmsd]
                < self.cfg.redesigned_rmsd_threshold
            ]

        # Add cluster information
        if self.cfg.cluster_path is not None:
            assert os.path.exists(
                self.cfg.cluster_path
            ), f"Cluster path {self.cfg.cluster_path} does not exist"

            self._pdb_to_cluster = _read_clusters(
                self.cfg.cluster_path, synthetic=False
            )
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._missing_pdbs = 0

            def cluster_lookup(pdb_name):
                pdb_name = pdb_name.upper()
                if pdb_name not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb_name] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb_name]

            metadata_csv[DatasetColumn.cluster] = metadata_csv[mc.pdb_name].map(
                cluster_lookup
            )
            self._log.debug(
                f"Assigned {self._max_cluster} clusters. {self._missing_pdbs} of {len(metadata_csv)} PDBs were missing from the cluster file."
            )

        # Add synthetic data if provided, and offset cluster numbers
        if self.cfg.use_synthetic:
            assert os.path.exists(
                self.cfg.synthetic_csv_path
            ), f"Synthetic CSV path {self.cfg.redesigned_csv_path} does not exist"
            self.synthetic_csv = pd.read_csv(self.cfg.synthetic_csv_path)

            assert os.path.exists(
                self.cfg.synthetic_cluster_path
            ), f"Synthetic cluster path {self.cfg.synthetic_cluster_path} does not exist"
            self._synthetic_pdb_to_cluster = _read_clusters(
                self.cfg.synthetic_cluster_path, synthetic=True
            )

            # Clusters simply must be defined.
            # The actual number is incremented by the number of real clusters.

            # Offset all the cluster numbers by the number of real data clusters
            num_real_clusters = metadata_csv[DatasetColumn.cluster].max() + 1

            def synthetic_cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._synthetic_pdb_to_cluster:
                    raise ValueError(
                        f"Synthetic example {pdb} not in synthetic cluster file!"
                    )
                return self._synthetic_pdb_to_cluster[pdb] + num_real_clusters

            self.synthetic_csv[DatasetColumn.cluster] = self.synthetic_csv[
                mc.pdb_name
            ].map(synthetic_cluster_lookup)

            # concat synthetic data to metadata_csv
            metadata_csv = pd.concat([metadata_csv, self.synthetic_csv])

        # We just use the synthetic and re-designed data as is without additional filtering.
        # Some filtering would be difficult, e.g. because designed structures/sequences may have
        # values like pLDDT assigned by the tool that generated them.
        # Also, presumably the synthetic data is generated from already-filtered structures/sequences.

        # TODO support augmenting dataset:
        #   multimers: chain selections
        #     based on MetadataColumn.chain_interactions
        #   too long: crop to target length
        #     for single chains, simply crop
        #     for multiple chains, select interacting chains and trim non-interacting positions
        #
        #   Likely makes sense to augment MotifFactory to support these sorts of things,
        #     and reuse some of the motif selection logic.

        self._create_split(metadata_csv)

        # If test set IDs defined, remove from training set
        if self.cfg.test_set_pdb_ids_path is not None:
            test_set_df = pd.read_csv(cfg.test_set_pdb_ids_path)
            self.csv = self.csv[
                self.csv[mc.pdb_name].isin(test_set_df[mc.pdb_name].values)
            ]

    @property
    def is_training(self):
        return self._is_training

    @property
    def modeled_length_col(self) -> mc:
        return self.cfg.modeled_trim_method.to_dataset_column()

    def _create_split(self, data_csv: DatasetDataFrame):
        # Training or validation specific logic.
        # TODO actually split - this isn't really splitting... it's ~ all the samples in both cases
        if self.is_training:
            self.csv = data_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            # pick all samples in data_csv with valid eval length
            eval_lengths = data_csv[self.modeled_length_col]
            if self.cfg.max_eval_length is not None:
                eval_lengths = eval_lengths[eval_lengths <= self.cfg.max_eval_length]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.cfg.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv[self.modeled_length_col].isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby(self.modeled_length_col).sample(
                self.cfg.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values(self.modeled_length_col, ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )

        # reset index
        self.csv[DatasetColumn.index] = list(range(len(self.csv)))

    def load_processed_file_with_caching(self, csv_row: DatasetCSVRow) -> ProcessedFile:
        """
        Loads a single structure + metadata pickled at `processed_file_path`, with caching.
        """
        processed_file_path = os.path.join(
            self.cfg.processed_data_path, csv_row[mc.processed_path]
        )
        seq_len = csv_row[self.modeled_length_col]
        use_cache = seq_len > self.cfg.cache_num_res

        if use_cache and processed_file_path in self._cache:
            return self._cache[processed_file_path]

        trim_chains_independently = (
            self.cfg.modeled_trim_method == DatasetTrimMethod.chains_independently
        )
        processed_feats = read_processed_file(
            processed_file_path,
            trim_chains_independently=trim_chains_independently,
        )

        if use_cache:
            self._cache[processed_file_path] = processed_feats

        return processed_feats

    def featurize_processed_file(
        self, processed_file: ProcessedFile, csv_row: DatasetCSVRow
    ) -> BatchFeatures:
        return self.featurizer.featurize_processed_file(
            processed_file=processed_file,
            csv_row=csv_row,
        )

    def process_csv_row(self, csv_row: DatasetCSVRow) -> BatchFeatures:
        """
        Process a single row of the CSV file.
        File loading is cached.
        Addition of noise, determination of masks etc. is not cached.
        """
        processed_file = self.load_processed_file_with_caching(
            csv_row=csv_row,
        )
        processed_row = self.featurize_processed_file(
            processed_file=processed_file,
            csv_row=csv_row,
        )
        return processed_row

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, row_idx) -> BatchFeatures:
        csv_row: DatasetCSVRow = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)

        # Storing the csv index is helpful for debugging.
        # Assign here to ensure we have the correct index.
        feats[bp.csv_idx] = torch.ones(1, dtype=torch.long) * row_idx

        return feats


class PdbDataset(BaseDataset):
    """
    PDB Dataset is a dataset of protein structures from the PDB.

    Note that in the original MultiFlow, this was the only implementation of the base class BaseDataset.
    I have merged them together, given there is a single dataset config and single dataset.
    It may make sense to separate them in the future.
    """

    def __init__(self, *, cfg: DatasetConfig, is_training: bool, task: DataTask):
        assert (
            cfg.cluster_path is not None
        ), "Cluster path must be provided for PDB dataset"
        assert os.path.exists(
            cfg.cluster_path
        ), f"Cluster path {cfg.cluster_path} does not exist"
        super().__init__(
            cfg=cfg,
            is_training=is_training,
            task=task,
        )


class LengthSamplingDataset(Dataset):
    """
    During predictions/inference, dataset to generate (e.g. unconditional) samples across lengths
    Each item comprises `num_res` (int) and `sample_id` (torch.Tensor, int or ints or strings)
    """

    def __init__(self, cfg: InferenceSamplesConfig):
        self.cfg = cfg

        # determine lengths to sample
        if cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in cfg.length_subset]
        else:
            all_sample_lengths = range(
                self.cfg.min_length, self.cfg.max_length + 1, self.cfg.length_step
            )

        all_sample_ids = []
        num_batch = self.cfg.num_batch
        assert self.cfg.samples_per_length % num_batch == 0
        self.n_samples = self.cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(self.n_samples):
                sample_ids = torch.tensor(
                    [num_batch * sample_id + i for i in range(num_batch)]
                )
                all_sample_ids.append((length, sample_ids))
        self._all_sample_ids = all_sample_ids

    def __len__(self):
        return len(self._all_sample_ids)

    def generate_idx(self, num_res: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generates chain_idx and res_idx, potentially multimeric"""
        # Generate monomer if...
        if (
            # fraction is 0 or random() < fraction
            random.random() >= self.cfg.multimer_fraction
            or
            # too short for multimers
            num_res < self.cfg.multimer_min_length * 2
        ):
            return torch.ones(num_res, dtype=torch.long), torch.arange(num_res)

        # Otherwise, sample number of chains and chain lengths for a multimer

        # sample number of chains, skewed toward fewer
        max_chains = int(floor(num_res / self.cfg.multimer_min_length))
        chain_possibilities = np.arange(2, max_chains + 1)
        num_chains = np.random.choice(
            a=chain_possibilities,
            p=1 / chain_possibilities / np.sum(1 / chain_possibilities),
        )

        # build chain_idx and res_idx by choosing chain lengths and filling in
        chain_idx = torch.zeros(num_res, dtype=torch.long)
        res_idx = torch.zeros(num_res, dtype=torch.long)

        # determine an "extension" length to add to `multimer_min_length` by breaking up remaining_length
        remaining_length = num_res - num_chains * self.cfg.multimer_min_length
        if remaining_length == 0:
            # nothing leftover, all chains the same length
            chain_extensions = np.zeros(num_chains, dtype=int)
        else:
            # pick break points along remaining_length
            chain_extensions = np.random.choice(
                remaining_length, num_chains, replace=False
            )

        # update idx
        last_extension_size = 0
        last_end_idx = 0
        for chain_id, extension_size in enumerate(list(sorted(chain_extensions))):
            chain_length = (
                self.cfg.multimer_min_length + extension_size - last_extension_size
            )
            chain_end = last_end_idx + chain_length

            # chain_idx is 1-indexed
            chain_idx[last_end_idx:chain_end] = chain_id + 1
            # res_idx is 1-indexed and per-chain
            res_idx[last_end_idx:chain_end] = torch.arange(
                1, chain_length + 1, dtype=torch.long
            )

            last_extension_size = extension_size
            last_end_idx = chain_end

        return chain_idx, res_idx

    def __getitem__(self, idx) -> InferenceFeatures:
        num_res, sample_id = self._all_sample_ids[idx]
        chain_idx, res_idx = self.generate_idx(num_res)

        return {
            bp.sample_id: sample_id,
            bp.res_mask: torch.ones(num_res),
            bp.diffuse_mask: torch.ones(num_res),
            bp.res_idx: res_idx,
            bp.chain_idx: chain_idx,
        }


DatasetClassT = TypeVar("DatasetClassT")


@dataclass
class DatasetConstructor:
    dataset_class: DatasetClassT
    cfg: DatasetConfig
    task: DataTask

    def create_datasets(self) -> Tuple[DatasetClassT, Optional[DatasetClassT]]:
        """generate dataset, and possibly validation dataset"""
        train_dataset = self.dataset_class(
            cfg=self.cfg,
            task=self.task,
            is_training=True,
        )

        eval_dataset = self.dataset_class(
            cfg=self.cfg,
            task=self.task,
            is_training=False,
        )

        return train_dataset, eval_dataset

    @classmethod
    def pdb_dataset(
        cls, dataset_cfg: DatasetConfig, task: DataTask = DataTask.hallucination
    ):
        """Generates default training and evaluation datasets"""
        return cls(
            dataset_class=PdbDataset,
            cfg=dataset_cfg,
            task=task,
        )

    @classmethod
    def from_cfg(
        cls,
        cfg: Config,
    ):
        if cfg.data.dataset == "pdb":
            return cls(
                dataset_class=PdbDataset,
                cfg=cfg.dataset,
                task=cfg.data.task,
            )
        else:
            raise ValueError(f"Unrecognized dataset {cfg.data.dataset}")
