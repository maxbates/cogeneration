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

from cogeneration.config.base import Config, DatasetConfig, InferenceSamplesConfig
from cogeneration.data import data_transforms, rigid_utils
from cogeneration.data.const import seq_to_aatype
from cogeneration.dataset.filterer import DatasetFilterer
from cogeneration.dataset.motif_factory import Motif, MotifFactory, Scaffold, Segment
from cogeneration.dataset.process_pdb import read_processed_file
from cogeneration.type.batch import METADATA_BATCH_PROPS, BatchFeatures
from cogeneration.type.batch import BatchProps as bp
from cogeneration.type.batch import InferenceFeatures, empty_feats
from cogeneration.type.dataset import DatasetColumns as dc
from cogeneration.type.dataset import DatasetProteinColumns as dpc
from cogeneration.type.dataset import DatasetTransformColumns as dtc
from cogeneration.type.dataset import MetadataCSVRow, MetadataDataFrame, ProcessedFile
from cogeneration.type.task import DataTaskEnum


def batch_features_from_processed_file(
    processed_file: ProcessedFile,
    cfg: DatasetConfig,
    processed_file_path: str,
) -> BatchFeatures:
    """
    Converts a ProcessedFile (numpy) into BatchFeatures (tensors, frames).
    Defaults to `diffuse_mask` of all ones. Can be modified by caller.
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
        "all_atom_positions": torch.tensor(processed_file[dpc.atom_positions]).double(),
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
    # calculate torsion angles, in case predicting psi angles
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
    diffuse_mask = torch.ones_like(res_mask)

    feats: BatchFeatures = {
        bp.res_mask: res_mask,
        bp.aatypes_1: aatypes_1,
        bp.trans_1: trans_1,
        bp.rotmats_1: rotmats_1,
        bp.torsion_angles_sin_cos_1: chain_feats[dtc.torsion_angles_sin_cos],
        bp.chain_idx: chain_idx,
        bp.res_idx: res_idx,
        bp.res_plddt: res_plddt,
        bp.diffuse_mask: diffuse_mask,
        bp.plddt_mask: plddt_mask,
    }

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


class BaseDataset(Dataset):
    def __init__(
        self,
        *,
        dataset_cfg: DatasetConfig,
        is_training: bool,
        task: DataTaskEnum,
    ):
        self._log = logging.getLogger(__name__)
        self._is_training = is_training
        self.dataset_cfg = dataset_cfg
        self.task = task

        self._cache = {}
        self._rng = np.random.default_rng(seed=self.dataset_cfg.seed)
        self._rng_fixed = np.random.default_rng(seed=42069)

        self.motif_factory = MotifFactory(
            cfg=self.dataset_cfg.inpainting,
            rng=self._rng if self.is_training else self._rng_fixed,
        )

        # Process structures and clusters
        assert os.path.exists(
            self.dataset_cfg.csv_path
        ), f"CSV path {self.dataset_cfg.csv_path} does not exist"
        self.raw_csv = pd.read_csv(self.dataset_cfg.csv_path)
        self._log.debug(
            f"Loaded {len(self.raw_csv)} examples from {self.dataset_cfg.csv_path}"
        )

        # Initial filtering
        dataset_filterer = DatasetFilterer(dataset_cfg)
        metadata_csv = dataset_filterer.filter_metadata(self.raw_csv)
        metadata_csv = metadata_csv.sort_values(dc.modeled_seq_len, ascending=False)

        # Concat redesigned data, if provided
        if self.dataset_cfg.use_redesigned:
            assert os.path.exists(
                self.dataset_cfg.redesigned_csv_path
            ), f"Redesigned CSV path {self.dataset_cfg.redesigned_csv_path} does not exist"

            self.redesigned_csv = pd.read_csv(self.dataset_cfg.redesigned_csv_path)
            metadata_csv = metadata_csv.merge(
                self.redesigned_csv, left_on=dc.pdb_name, right_on=dc.example
            )
            # Filter out examples with high RMSD
            # TODO - make configurable
            metadata_csv = metadata_csv[metadata_csv[dc.best_rmsd] < 2.0]

        # Add cluster information
        if self.dataset_cfg.cluster_path is not None:
            assert os.path.exists(
                self.dataset_cfg.cluster_path
            ), f"Cluster path {self.dataset_cfg.cluster_path} does not exist"

            self._pdb_to_cluster = _read_clusters(
                self.dataset_cfg.cluster_path, synthetic=False
            )
            self._max_cluster = max(self._pdb_to_cluster.values())
            self._missing_pdbs = 0

            def cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._pdb_to_cluster:
                    self._pdb_to_cluster[pdb] = self._max_cluster + 1
                    self._max_cluster += 1
                    self._missing_pdbs += 1
                return self._pdb_to_cluster[pdb]

            metadata_csv[dc.cluster] = metadata_csv[dc.pdb_name].map(cluster_lookup)
            self._log.debug(
                f"Assigned {self._max_cluster} clusters. {self._missing_pdbs} of {len(metadata_csv)} PDBs were missing from the cluster file."
            )

        # Add synthetic data if provided, and offset cluster numbers
        if self.dataset_cfg.use_synthetic:
            assert os.path.exists(
                self.dataset_cfg.synthetic_csv_path
            ), f"Synthetic CSV path {self.dataset_cfg.redesigned_csv_path} does not exist"
            self.synthetic_csv = pd.read_csv(self.dataset_cfg.synthetic_csv_path)

            assert os.path.exists(
                self.dataset_cfg.synthetic_cluster_path
            ), f"Synthetic cluster path {self.dataset_cfg.synthetic_cluster_path} does not exist"
            self._synthetic_pdb_to_cluster = _read_clusters(
                self.dataset_cfg.synthetic_cluster_path, synthetic=True
            )

            # Clusters simply must be defined.
            # The actual number is incremented by the number of real clusters.

            # Offset all the cluster numbers by the number of real data clusters
            num_real_clusters = metadata_csv[dc.cluster].max() + 1

            def synthetic_cluster_lookup(pdb):
                pdb = pdb.upper()
                if pdb not in self._synthetic_pdb_to_cluster:
                    raise ValueError(
                        f"Synthetic example {pdb} not in synthetic cluster file!"
                    )
                return self._synthetic_pdb_to_cluster[pdb] + num_real_clusters

            self.synthetic_csv[dc.cluster] = self.synthetic_csv[dc.pdb_name].map(
                synthetic_cluster_lookup
            )

            # concat synthetic data to metadata_csv
            metadata_csv = pd.concat([metadata_csv, self.synthetic_csv])

        # TODO - consider filtering synthetic and redesigned data, not just initial data

        self._create_split(metadata_csv)

        # If test set IDs defined, remove from training set
        if self.dataset_cfg.test_set_pdb_ids_path is not None:
            test_set_df = pd.read_csv(dataset_cfg.test_set_pdb_ids_path)
            self.csv = self.csv[
                self.csv[dc.pdb_name].isin(test_set_df[dc.pdb_name].values)
            ]

    @property
    def is_training(self):
        return self._is_training

    def __len__(self):
        return len(self.csv)

    def _create_split(self, data_csv: MetadataDataFrame):
        # Training or validation specific logic.
        # TODO actually split - this isn't really splitting... it's ~ all the samples in both cases
        if self.is_training:
            self.csv = data_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            # pick all samples in data_csv with valid eval length
            eval_lengths = data_csv[dc.modeled_seq_len]
            if self.dataset_cfg.max_eval_length is not None:
                eval_lengths = eval_lengths[
                    eval_lengths <= self.dataset_cfg.max_eval_length
                ]
            all_lengths = np.sort(eval_lengths.unique())
            length_indices = (len(all_lengths) - 1) * np.linspace(
                0.0, 1.0, self.dataset_cfg.num_eval_lengths
            )
            length_indices = length_indices.astype(int)
            eval_lengths = all_lengths[length_indices]
            eval_csv = data_csv[data_csv[dc.modeled_seq_len].isin(eval_lengths)]

            # Fix a random seed to get the same split each time.
            eval_csv = eval_csv.groupby(dc.modeled_seq_len).sample(
                self.dataset_cfg.samples_per_eval_length, replace=True, random_state=123
            )
            eval_csv = eval_csv.sort_values(dc.modeled_seq_len, ascending=False)
            self.csv = eval_csv
            self._log.info(
                f"Validation: {len(self.csv)} examples with lengths {eval_lengths}"
            )

        # reset index
        self.csv[dc.index] = list(range(len(self.csv)))

    def load_processed_path_with_caching(
        self, csv_row: MetadataCSVRow
    ) -> ProcessedFile:
        """
        Loads a single structure + metadata pickled at `processed_file_path`, with caching.
        """
        processed_file_path = os.path.join(
            self.dataset_cfg.processed_data_path, csv_row[dc.processed_path]
        )
        seq_len = csv_row[dc.modeled_seq_len]
        use_cache = seq_len > self.dataset_cfg.cache_num_res

        if use_cache and processed_file_path in self._cache:
            return self._cache[processed_file_path]

        processed_feats = read_processed_file(processed_file_path)

        if use_cache:
            self._cache[processed_file_path] = processed_feats

        return processed_feats

    @staticmethod
    def reset_residues_and_randomize_chains(cfg: DatasetConfig, feats: BatchFeatures):
        """
        Randomize chain indices, and re-number residue indices for each chain to start from 1.
        Modifies the input features in place.
        """
        chain_idx = feats[bp.chain_idx]
        res_idx = feats[bp.res_idx]

        new_res_idx = torch.zeros_like(res_idx)
        new_chain_idx = torch.zeros_like(res_idx)
        all_chain_idx = torch.unique(chain_idx).tolist()
        shuffled_chain_idx = (
            torch.tensor(
                random.sample(all_chain_idx, len(all_chain_idx)), dtype=torch.long
            )
            - torch.min(chain_idx)
            + 1
        )
        for i, chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).long()
            chain_min_idx = torch.min(
                res_idx + (1 - chain_mask) * cfg.chain_gap_dist
            ).long()
            new_res_idx += (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx += replacement_chain_id * chain_mask

        feats[bp.chain_idx] = new_chain_idx
        feats[bp.res_idx] = new_res_idx

    @staticmethod
    def recenter_structure(
        feats: BatchFeatures,
    ):
        """
        Centers the structure to be translation invariant.
        Note, structure should already be centered on load by `parse_pdb_feats`.
        """
        center_of_mass = torch.sum(feats[bp.trans_1], dim=0) / torch.sum(
            feats[bp.res_mask] + 1e-5
        )

        # # TODO(inpainting-fixed) explicit flag to only consider `diffuse_mask` for centering
        # if (feats[bp.diffuse_mask] == 0).any():
        #     # Re-center the structure based on non-diffused (motif) locations. Motifs should not move.
        #     motif_mask = 1 - feats[bp.diffuse_mask]
        #     motif_1 = feats[bp.trans_1] * motif_mask[:, None]
        #     center_of_mass = torch.sum(motif_1, dim=0) / (torch.sum(motif_mask) + 1e-5)
        # else:
        #     center_of_mass = torch.sum(feats[bp.trans_1], dim=0) / torch.sum(
        #         feats[bp.res_mask] + 1e-5
        #     )

        # modify in place
        feats[bp.trans_1] -= center_of_mass[None, :]

    @staticmethod
    def segment_features(
        feats: BatchFeatures,
        segments: List[Segment],
    ):
        """
        Apply `Motif` and `Scaffold` `Segments` to `feats`.
        Preserves the feats in motifs, and masks them in scaffolds, and extends scaffolds to target length.
        i.e. has the effect of masking `trans_1`, `rotmats_1`, `aatypes_1` etc. in scaffolds.

        Does not manage `res_idx` and `chain_idx` - assumes will be shuffled after calling
        TODO - support specifying chain breaks some how... don't break when shuffle
        """
        new_total_length = sum([seg.length for seg in segments])
        new_feats = empty_feats(N=new_total_length)

        # copy over metadata features from original
        for prop in METADATA_BATCH_PROPS:
            if prop in feats:
                new_feats[prop] = feats[prop]

        # copy features in motif ranges.
        # scaffold lengths may have changed, so use updated indices in new_feats
        running_segment_start = 0
        for segment in segments:
            # get positions. note `end` positions are exclusive here
            # original positions (i.e. in `feats`)
            os, oe, l = segment.start, segment.start + segment.length, segment.length
            # new positions (i.e. in `new_feats`)
            ns, ne = running_segment_start, running_segment_start + l

            if isinstance(segment, Motif):
                # copy over relevant motif features
                for prop, value in new_feats.items():
                    if prop in METADATA_BATCH_PROPS:
                        continue
                    new_feats[prop][ns:ne] = feats[prop][os:oe]

                # set diffuse_mask to 0.0 in motif positions
                new_feats[bp.diffuse_mask][ns:ne] = 0.0
            elif isinstance(segment, Scaffold):
                # set diffuse_mask to 1.0 in scaffold positions
                new_feats[bp.diffuse_mask][ns:ne] = 1.0

            # for any segment update running length
            running_segment_start += segment.length

        return new_feats

    @staticmethod
    def _featurize_processed_file(
        cfg: DatasetConfig,
        task: DataTaskEnum,
        is_training: bool,
        processed_file: ProcessedFile,
        csv_row: MetadataCSVRow,
        motif_factory: MotifFactory,
    ) -> BatchFeatures:
        """
        Processes a pickled features from single structure + metadata.
        Converts numpy feats to tensor feats.

        This function should be called as examples are needed, i.e. in `__get_item__`,
        because it adds noise to atom positions, picks motif positions, etc. as defined by cfg.
        """
        # Redesigned sequences can be used to substitute the original sequence.
        if cfg.use_redesigned:
            best_seq = csv_row[dc.best_seq]
            if not isinstance(best_seq, str):
                raise ValueError(f"Unexpected value best_seq: {best_seq}")

            best_aatype = np.array(seq_to_aatype(best_seq))
            processed_file[dpc.aatype] = best_aatype

            assert (
                processed_file[dpc.aatype].shape == best_aatype.shape
            ), f"best_seq different length: {processed_file[dpc.aatype].shape} vs {best_aatype.shape}"

        feats = batch_features_from_processed_file(
            processed_file=processed_file,
            cfg=cfg,
            processed_file_path=csv_row[dc.processed_path],
        )

        # Pass through relevant data from CSV
        feats[bp.pdb_name] = csv_row[dc.pdb_name]

        # Update `diffuse_mask` depending on task
        if task == DataTaskEnum.hallucination:
            # diffuse_mask = torch.ones()
            pass
        elif task == DataTaskEnum.inpainting:
            diffuse_mask = motif_factory.generate_diffuse_mask(
                res_mask=feats[bp.res_mask],
                plddt_mask=feats[bp.plddt_mask],
                chain_idx=feats[bp.chain_idx],
                res_idx=feats[bp.res_idx],
                trans_1=feats[bp.trans_1],
                rotmats_1=feats[bp.rotmats_1],
                aatypes_1=feats[bp.aatypes_1],
            )
            feats[bp.diffuse_mask] = diffuse_mask
        else:
            raise ValueError(f"Unknown task {task}")

        # Ensure have a valid `diffuse_mask` for modeling
        if torch.sum(feats[bp.diffuse_mask]) == 0:
            feats[bp.diffuse_mask] = torch.ones_like(feats[bp.res_mask])
        feats[bp.diffuse_mask] = feats[bp.diffuse_mask].float()

        # For inpainting evaluation (i.e. not training), vary scaffold lengths
        # Using the `diffuse_mask`, modify the scaffold region lengths, and mask out the scaffolds
        # i.e. {trans,rots,aatypes}_1 only defined for motif positions
        if (
            task == DataTaskEnum.inpainting
            and not is_training
            and (feats[bp.diffuse_mask] == 0).any()
        ):
            segments = motif_factory.generate_segments_from_diffuse_mask(
                diffuse_mask=feats[bp.diffuse_mask],
                chain_idx=feats[bp.chain_idx],
            )
            BaseDataset.segment_features(feats=feats, segments=segments)

        # TODO(inpainting-fixed) If motif positions are fixed, those motifs need to be re-centered
        # to avoid hinting / memorizing where to place scaffolds.
        # Need to update `recenter_structure` method to use `diffuse_mask` appropriately.
        # if add_noise or (diffuse_mask == 0).any()
        #    BaseDataset.recenter_structure(feats=feats)

        # Randomize chains and reset residue positions - after motif-selection!
        # The motifs refer to chains, so don't want to disrupt that mapping.
        BaseDataset.reset_residues_and_randomize_chains(cfg=cfg, feats=feats)

        return feats

    def process_processed_file(
        self, processed_file: ProcessedFile, csv_row: MetadataCSVRow
    ) -> BatchFeatures:
        return self._featurize_processed_file(
            cfg=self.dataset_cfg,
            task=self.task,
            is_training=self.is_training,
            processed_file=processed_file,
            csv_row=csv_row,
            motif_factory=self.motif_factory,
        )

    def process_csv_row(self, csv_row: MetadataCSVRow) -> BatchFeatures:
        """
        Process a single row of the CSV file.
        File loading is cached.
        Addition of noise, determination of masks etc. is not cached.
        """
        processed_file = self.load_processed_path_with_caching(
            csv_row=csv_row,
        )
        processed_row = self.process_processed_file(
            processed_file=processed_file,
            csv_row=csv_row,
        )
        return processed_row

    def __getitem__(self, row_idx) -> BatchFeatures:
        csv_row = self.csv.iloc[row_idx]
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

    def __init__(
        self, *, dataset_cfg: DatasetConfig, is_training: bool, task: DataTaskEnum
    ):
        assert (
            dataset_cfg.cluster_path is not None
        ), "Cluster path must be provided for PDB dataset"
        assert os.path.exists(
            dataset_cfg.cluster_path
        ), f"Cluster path {dataset_cfg.cluster_path} does not exist"
        super().__init__(
            dataset_cfg=dataset_cfg,
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
            chain_extensions = list(
                sorted(np.random.choice(remaining_length, num_chains, replace=False))
            )

        # update idx
        last_extension_size = 0
        last_end_idx = 0
        for chain_id, extension_size in enumerate(chain_extensions):
            chain_length = (
                self.cfg.multimer_min_length + extension_size - last_extension_size
            )
            last_extension_size = extension_size

            chain_end = last_end_idx + chain_length
            chain_idx[last_end_idx:chain_end] = chain_id + 1
            res_idx[last_end_idx:chain_end] = (
                chain_id * self.cfg.chain_gap_dist
            ) + torch.arange(last_end_idx, chain_end, dtype=torch.long)
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
    task: DataTaskEnum

    def create_datasets(self) -> Tuple[DatasetClassT, Optional[DatasetClassT]]:
        """generate dataset, and possibly validation dataset"""
        train_dataset = self.dataset_class(
            dataset_cfg=self.cfg,
            task=self.task,
            is_training=True,
        )

        eval_dataset = self.dataset_class(
            dataset_cfg=self.cfg,
            task=self.task,
            is_training=False,
        )

        return train_dataset, eval_dataset

    @classmethod
    def pdb_dataset(
        cls, dataset_cfg: DatasetConfig, task: DataTaskEnum = DataTaskEnum.hallucination
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
