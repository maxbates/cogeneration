import logging
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd
import torch
import tree
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from torch.utils.data import Dataset

from cogeneration.config.base import (
    Config,
    DatasetConfig,
    DataTaskEnum,
    InferenceSamplesConfig,
)
from cogeneration.data import data_transforms, rigid_utils
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.const import seq_to_aatype
from cogeneration.data.enum import DatasetColumns as dc
from cogeneration.data.enum import DatasetProteinColumns as dpc
from cogeneration.data.enum import DatasetTransformColumns as dtc
from cogeneration.data.io import read_pkl
from cogeneration.dataset.data_utils import parse_chain_feats


def _read_clusters(cluster_path, synthetic=False):
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


class DatasetFilterer:
    def __init__(self, dataset_cfg: DatasetConfig):
        self.dataset_cfg = dataset_cfg
        self._log = logging.getLogger("DatasetFilterer")

    def _rog_filter(self, data_csv: pd.DataFrame) -> pd.DataFrame:
        """
        Filter by radius of gyration.
        """
        y_quant = pd.pivot_table(
            data_csv,
            values=dc.radius_gyration,
            index=dc.modeled_seq_len,
            aggfunc=lambda x: np.quantile(x, self.dataset_cfg.filter.rog_quantile),
        )
        x_quant = y_quant.index.to_numpy()
        y_quant = y_quant.radius_gyration.to_numpy()

        # Fit polynomial regressor
        poly = PolynomialFeatures(degree=4, include_bias=True)
        poly_features = poly.fit_transform(x_quant[:, None])
        poly_reg_model = LinearRegression()
        poly_reg_model.fit(poly_features, y_quant)

        # Calculate cutoff for all sequence lengths
        max_len = data_csv[dc.modeled_seq_len].max()
        pred_poly_features = poly.fit_transform(np.arange(max_len)[:, None])
        # Add a little more.
        pred_y = poly_reg_model.predict(pred_poly_features) + 0.1

        row_rog_cutoffs = data_csv[dc.modeled_seq_len].map(lambda x: pred_y[x - 1])
        return data_csv[data_csv[dc.radius_gyration] < row_rog_cutoffs]

    def _length_filter(self, data_csv: pd.DataFrame) -> pd.DataFrame:
        """Filter by sequence length."""
        return data_csv[
            (data_csv[dc.modeled_seq_len] >= self.dataset_cfg.filter.min_num_res)
            & (data_csv[dc.modeled_seq_len] <= self.dataset_cfg.filter.max_num_res)
        ]

    def _plddt_filter(self, data_csv: pd.DataFrame) -> pd.DataFrame:
        """Filter proteins which do not have the required minimum pLDDT."""
        # not used in the public multiflow codebase
        # TODO - pull out pLDDTs from structure, not available in current CSV
        # return data_csv[
        #     dc.num_confident_plddt
        #     > self.dataset_cfg.filter.min_num_confident_plddt
        # ]
        return data_csv

    def _max_coil_filter(self, data_csv: pd.DataFrame) -> pd.DataFrame:
        """Filter proteins which exceed max_coil_percent."""
        return data_csv[
            data_csv[dc.coil_percent] <= self.dataset_cfg.filter.max_coil_percent
        ]

    def filter_metadata(self, raw_csv: pd.DataFrame) -> pd.DataFrame:
        """
        Initial filtering of dataset.
        Does not filter redesigned / synthetic datasets.
        """
        filter_cfg = self.dataset_cfg.filter
        running_length = len(raw_csv)
        data_csv = raw_csv.copy()

        # monomer / oligomer
        data_csv = data_csv[data_csv[dc.oligomeric_detail].isin(filter_cfg.oligomeric)]
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} -> {len(data_csv)} examples after oligomeric filter"
            )
            running_length = len(data_csv)

        # number of chains
        data_csv = data_csv[data_csv[dc.num_chains].isin(filter_cfg.num_chains)]
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after num_chains filter"
            )
            running_length = len(data_csv)

        # length
        data_csv = self._length_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after length filter"
            )
            running_length = len(data_csv)

        # max coil percent
        data_csv = self._max_coil_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} ->{len(data_csv)} examples after max coil filter"
            )
            running_length = len(data_csv)

        # radius of gyration
        data_csv = self._rog_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} -> {len(data_csv)} examples after rog filter"
            )
            running_length = len(data_csv)

        # pLDDT
        data_csv = self._plddt_filter(data_csv=data_csv)
        if len(data_csv) < running_length:
            self._log.debug(
                f"{running_length} -> {len(data_csv)} examples after pLDDT filter"
            )
            running_length = len(data_csv)

        return data_csv


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
        self._rng = np.random.default_rng(seed=self.dataset_cfg.seed)
        self._cache = {}

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

    def _create_split(self, data_csv):
        # Training or validation specific logic.
        if self.is_training:
            self.csv = data_csv
            self._log.info(f"Training: {len(self.csv)} examples")
        else:
            if self.dataset_cfg.max_eval_length is None:
                eval_lengths = data_csv[dc.modeled_seq_len]
            else:
                eval_lengths = data_csv[dc.modeled_seq_len][
                    data_csv[dc.modeled_seq_len] <= self.dataset_cfg.max_eval_length
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

    def process_processed_path(self, processed_file_path: str) -> Dict[str, Any]:
        """
        Processes a single structure + metadata pickled at `processed_file_path`.
        This file is written by `parse_pdb_files.py`.
        """
        processed_feats = read_pkl(processed_file_path)
        processed_feats = parse_chain_feats(processed_feats)

        # Only take modeled residues.
        modeled_idx = processed_feats[dpc.modeled_idx]
        min_idx = np.min(modeled_idx)
        max_idx = np.max(modeled_idx)
        del processed_feats[dpc.modeled_idx]
        processed_feats = tree.map_structure(
            lambda x: x[min_idx : (max_idx + 1)], processed_feats
        )

        # Construct an intermediate `chain_feats` for OpenFold pipeline
        chain_feats = {
            "aatype": torch.tensor(processed_feats[dpc.aatype]).long(),
            "all_atom_positions": torch.tensor(
                processed_feats[dpc.atom_positions]
            ).double(),
            "all_atom_mask": torch.tensor(processed_feats[dpc.atom_mask]).double(),
        }

        # Add noise to atom positions.
        # Features are in angstrom-scale.
        if self.dataset_cfg.noise_atom_positions_angstroms > 0:
            atom_position_noise = (
                torch.randn_like(chain_feats["all_atom_positions"])
                * self.dataset_cfg.noise_atom_positions_angstroms
            )
            chain_feats["all_atom_positions"] += atom_position_noise

        # Run through OpenFold data transforms.
        # Convert atomic representation to frames
        chain_feats = data_transforms.atom37_to_frames(chain_feats)
        # calculate torsion angles
        chain_feats = data_transforms.atom37_to_torsion_angles()(chain_feats)

        # Re-number residue indices for each chain such that it starts from 1.
        # Randomize chain indices.
        chain_idx = processed_feats[dpc.chain_index]
        res_idx = processed_feats[dpc.residue_index]
        new_res_idx = np.zeros_like(res_idx)
        new_chain_idx = np.zeros_like(res_idx)
        all_chain_idx = np.unique(chain_idx).tolist()
        shuffled_chain_idx = (
            np.array(random.sample(all_chain_idx, len(all_chain_idx)))
            - np.min(all_chain_idx)
            + 1
        )
        for i, chain_id in enumerate(all_chain_idx):
            chain_mask = (chain_idx == chain_id).astype(int)
            chain_min_idx = np.min(res_idx + (1 - chain_mask) * 1e3).astype(int)
            new_res_idx = new_res_idx + (res_idx - chain_min_idx + 1) * chain_mask

            # Shuffle chain_index
            replacement_chain_id = shuffled_chain_idx[i]
            new_chain_idx = new_chain_idx + replacement_chain_id * chain_mask

        # Extract rigits (translations + rotations), check for poorly processed
        rigids_1 = rigid_utils.Rigid.from_tensor_4x4(
            chain_feats[dtc.rigidgroups_gt_frames]
        )[:, 0]
        rotmats_1 = rigids_1.get_rots().get_rot_mats()
        trans_1 = rigids_1.get_trans()
        if torch.isnan(trans_1).any() or torch.isnan(rotmats_1).any():
            raise ValueError(f"Found NaNs in {processed_file_path}")

        # Mask low pLDDT residues
        res_plddt = processed_feats[dpc.b_factors][:, 1]
        res_mask = torch.tensor(processed_feats[dpc.bb_mask]).int()
        plddt_mask = torch.ones_like(res_mask)
        if self.dataset_cfg.add_plddt_mask:
            plddt_mask = torch.tensor(
                res_plddt > self.dataset_cfg.min_plddt_threshold
            ).int()

        # Take only necessary features
        return {
            bp.res_plddt: res_plddt,
            bp.aatypes_1: chain_feats[dpc.aatype],
            bp.rotmats_1: rotmats_1,
            bp.trans_1: trans_1,
            bp.torsion_angles_sin_cos_1: chain_feats[dtc.torsion_angles_sin_cos],
            bp.res_mask: res_mask,
            bp.plddt_mask: plddt_mask,
            bp.chain_idx: new_chain_idx,
            bp.res_idx: new_res_idx,
        }

    def process_csv_row(self, csv_row):
        """
        Process a single row of the CSV file, with caching.
        """
        path = os.path.join(
            self.dataset_cfg.processed_data_path, csv_row[dc.processed_path]
        )
        seq_len = csv_row[dc.modeled_seq_len]

        # Large protein files are slow to read. Cache them.
        use_cache = seq_len > self.dataset_cfg.cache_num_res
        if use_cache and path in self._cache:
            return self._cache[path]

        processed_row = self.process_processed_path(path)
        processed_row[dc.pdb_name] = csv_row[dc.pdb_name]

        if self.dataset_cfg.use_redesigned:
            best_seq = csv_row[dc.best_seq]
            if not isinstance(best_seq, float):
                best_aatype = torch.tensor(seq_to_aatype(best_seq)).long()
                assert processed_row[bp.aatypes_1].shape == best_aatype.shape
                processed_row[bp.aatypes_1] = best_aatype

        aatypes_1 = processed_row[bp.aatypes_1].detach().cpu().numpy()

        if len(set(aatypes_1)) == 1:
            raise ValueError(f"Example {path} has only one amino acid.")
        if use_cache:
            self._cache[path] = processed_row

        return processed_row

    def __getitem__(self, row_idx):
        # Process data example.
        csv_row = self.csv.iloc[row_idx]
        feats = self.process_csv_row(csv_row)  # process on the fly (with caching)

        if self.task == DataTaskEnum.hallucination:
            feats[bp.diffuse_mask] = torch.ones_like(feats[bp.res_mask]).bool()
        if self.task == DataTaskEnum.inpainting:
            # TODO(inpainting) - specify more interesting + random patches. Determined by config.
            feats[bp.diffuse_mask] = (torch.rand_like(feats[bp.res_mask]) > 0.5).bool()
        else:
            raise ValueError(f"Unknown task {self.task}")
        feats[bp.diffuse_mask] = feats[bp.diffuse_mask].int()

        # Storing the csv index is helpful for debugging.
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


class LengthSamplingDataset(torch.utils.data.Dataset):
    """
    During predictions/inference, dataset to generate (e.g. unconditional) samples across lengths
    Each item comprises `num_res` (int) and `sample_id` (torch.Tensor, int or ints or strings)
    """

    def __init__(self, inference_samples_cfg: InferenceSamplesConfig):
        self.cfg = inference_samples_cfg

        # determine lengths to sample
        if inference_samples_cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in inference_samples_cfg.length_subset]
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

    def __getitem__(self, idx):
        num_res, sample_id = self._all_sample_ids[idx]
        item = {
            bp.res_mask: torch.ones(num_res),
            bp.diffuse_mask: torch.ones(num_res).bool(),
            bp.num_res: num_res,
            bp.sample_id: sample_id,
        }
        return item


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
    def pdb_train_validation(cls, dataset_cfg: DatasetConfig):
        """Generates default training and evaluation datasets"""
        return cls(
            dataset_class=PdbDataset,
            cfg=dataset_cfg,
            task=DataTaskEnum.hallucination,
        )

    @classmethod
    def pdb_test(cls, dataset_cfg: DatasetConfig):
        """Generates default eval dataset"""
        return cls(
            dataset_class=PdbDataset,
            cfg=dataset_cfg,
            task=DataTaskEnum.hallucination,
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
