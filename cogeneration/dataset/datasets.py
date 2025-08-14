import logging
import os
import random
from dataclasses import dataclass
from math import floor
from pathlib import Path
from typing import List, Optional, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cogeneration.config.base import (
    Config,
    DatasetConfig,
    DatasetFilterConfig,
    DatasetTrimMethod,
    InferenceSamplesConfig,
)
from cogeneration.dataset.featurizer import BatchFeaturizer
from cogeneration.dataset.filterer import DatasetFilterer
from cogeneration.dataset.process_pdb import read_processed_file
from cogeneration.dataset.spec import DatasetSpec
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import InferenceFeatures
from cogeneration.type.dataset import (
    DATASET_KEY,
    DatasetColumn,
    DatasetCSVRow,
    DatasetDataFrame,
)
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataDataFrame, ProcessedFile
from cogeneration.type.dataset import RedesignColumn as rc
from cogeneration.type.structure import StructureExperimentalMethod
from cogeneration.type.task import DataTask, InferenceTask


class BaseDataset(Dataset):
    def __init__(
        self,
        *,
        cfg: DatasetConfig,
        task: DataTask,
        eval: bool,  # process for evaluation
        use_test: bool,  # use date-based split if True
    ):
        """
        BaseDataset collects all dataset files into a single Dataset, and yields samples.
        Metadata CSV defines the dataset, and `ProcessedFile` samples are read when yielded,
        and featurized by `BatchFeaturizer` into `BatchFeatures`.

        MetadataDataFrame (collection of MetadataCSVRow) concats datasets specified by cfg.datasets
        The dataset is filtered by `DatasetFilterer` according to `cfg.filter`.

        In `eval` mode, the dataset is processed for evaluation:
        A length subset is selected for evaluation, as defined by `cfg.dataset`.
        Samples are featurized differently, e.g. for inpainting scaffold lengths are varied.

        For inpainting, `MotifFactory` generates motifs and scaffolds.
        This happens during training (scaffolds are masked) *and* eval (scaffold lengths vary).
        """
        self.cfg = cfg
        self.task = task
        self.is_eval = eval
        self.use_test = use_test

        self._log = logging.getLogger(__name__)
        self._cache = {}

        self.featurizer = BatchFeaturizer(cfg=cfg, task=task, eval=self.is_eval)

        # Load specs as metadata DF
        metadata = BaseDataset.load_datasets(
            dataset_cfg=cfg,
            logger=self._log,
        )

        # Optionally dedupe by sequence hash, before applying date-based split
        if self.cfg.dedupe_by_sequence_hash:
            metadata = BaseDataset._dedupe_by_sequence_hash(metadata, logger=self._log)

        # Apply date-based train/test split before other filters
        metadata = self._apply_test_split(metadata)

        # Filter all structures
        dataset_filterer = DatasetFilterer(
            cfg=cfg.filter,
            modeled_trim_method=cfg.modeled_trim_method,
        )
        metadata = dataset_filterer.filter_metadata(metadata)

        # If this is an eval / validation dataset, limit to length subset
        if self.is_eval:
            metadata = BaseDataset.filter_to_eval_lengths(
                metadata=metadata,
                cfg=cfg,
            )
            self._log.info(
                f"Filtered to {len(metadata)} eval rows, lengths: {metadata[self.modeled_length_col].value_counts().to_dict()}"
            )

        # reset index
        metadata[DatasetColumn.index] = list(range(len(metadata)))
        self.csv = metadata

    @property
    def is_training(self) -> bool:
        """
        Returns True if this is an evaluation dataset.
        """
        return not self.is_eval

    @property
    def modeled_length_col(self) -> mc:
        return self.cfg.modeled_trim_method.to_dataset_column()

    @staticmethod
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

    @staticmethod
    def assign_clusters(
        cluster_path: Optional[Path],
        metadata: MetadataDataFrame,
        synthetic: bool = False,
    ):
        """
        Assign cluster numbers to structures in the metadata CSV.
        `synthetic` influences naming expectations (i.e. there is _ in name)
        """
        if cluster_path is None:
            metadata[DatasetColumn.cluster] = list(range(len(metadata)))
            return

        assert os.path.exists(
            cluster_path
        ), f"Cluster path {cluster_path} does not exist"

        # determine max cluster number, checking if some already defined
        max_cluster = 0
        if DatasetColumn.cluster in metadata.columns:
            max_cluster = metadata[DatasetColumn.cluster].max()

        # Read clusters
        pdb_to_cluster = {}
        with open(cluster_path, "r") as f:
            for i, line in enumerate(f):
                for chain in line.split(" "):
                    if not synthetic:
                        pdb = chain.split("_")[0].strip()
                    else:
                        pdb = chain.strip()
                    pdb_to_cluster[pdb.upper()] = i

        # Helper to assign cluster, or max_cluster + 1 if not found
        num_missing = 0

        def cluster_lookup(pdb_name):
            nonlocal max_cluster
            nonlocal num_missing

            pdb_name = pdb_name.upper()
            if pdb_name not in pdb_to_cluster:
                pdb_to_cluster[pdb_name] = max_cluster + 1
                max_cluster += 1
                num_missing += 1
            return pdb_to_cluster[pdb_name]

        metadata[DatasetColumn.cluster] = metadata[mc.pdb_name].map(cluster_lookup)
        print(
            f"Assigned {len(metadata)} clusters, {num_missing} missing, max cluster ID: {max_cluster}"
        )

    @staticmethod
    def load_dataset_spec_metadata(
        spec: DatasetSpec,
        cfg: DatasetConfig,
    ):
        """Loads dataset defined by spec"""
        # Process metadata
        metadata = BaseDataset.read_metadata_file(
            metadata_path=spec.metadata_path,
            max_rows=cfg.debug_head_samples,
        )

        # Processed paths should be complete paths (they may already be absolute)
        base_processed_path = Path(spec.processed_root_path)
        metadata[mc.processed_path] = metadata[mc.processed_path].apply(
            lambda rel_path: base_processed_path / rel_path
        )
        # Spot check a single path
        check_path = metadata[mc.processed_path].iloc[0]
        assert Path(
            check_path
        ).is_file(), f"Processed file {check_path} does not exist in root {spec.processed_root_path}"

        # Assign clusters, if provided
        BaseDataset.assign_clusters(
            cluster_path=spec.cluster_path,
            metadata=metadata,
        )

        return metadata

    @staticmethod
    def load_datasets(
        dataset_cfg: DatasetConfig, logger: logging.Logger
    ) -> MetadataDataFrame:
        """Load datasets specified by cfg.datasets, and concat them into a single DataFrame."""
        dfs = []
        for spec in dataset_cfg.datasets:
            if not spec.is_enabled():
                continue

            metadata_csv = BaseDataset.load_dataset_spec_metadata(
                spec=spec,
                cfg=dataset_cfg,
            )

            metadata_csv[DATASET_KEY] = spec.name

            dfs.append(metadata_csv)
            logger.info(f"Loaded dataset {spec.name} with {len(metadata_csv)} rows")

        metadata = pd.concat(dfs, ignore_index=True)
        return metadata

    @staticmethod
    def filter_to_eval_lengths(
        metadata: DatasetDataFrame,
        cfg: DatasetConfig,
    ):
        """
        Limit validation set to `cfg.num_eval_lengths` length subset
        """
        # length (post trim) is specified by the modeled length column
        length_column = cfg.modeled_trim_method.to_dataset_column()
        # Fix a random seed to get the same split each time.
        fixed_seed = 123

        assert (
            length_column in metadata.columns
        ), f"Length column {length_column} not found in metadata columns {metadata.columns}"
        assert len(metadata) > 0, "No data remains for validation"

        eval_lengths = metadata[length_column]
        if cfg.max_eval_length is not None:
            min_length = metadata[length_column].dropna().min()
            assert min_length <= cfg.max_eval_length, (
                f"Minimum length {min_length} is greater than max eval length "
                f"{cfg.max_eval_length}, no data remains for validation"
            )
            eval_lengths = eval_lengths[eval_lengths <= cfg.max_eval_length]

        # Pick up to `cfg.num_eval_lengths` lengths
        num_eval_lengths = min(cfg.num_eval_lengths, len(eval_lengths))
        all_lengths = np.sort(eval_lengths.unique())
        length_indices = (len(all_lengths) - 1) * np.linspace(
            0.0, 1.0, num_eval_lengths
        )
        length_indices = length_indices.astype(int)
        eval_lengths = all_lengths[length_indices]
        eval_csv = metadata[metadata[length_column].isin(eval_lengths)]

        # Pick subset per length
        eval_csv = eval_csv.groupby(length_column).sample(
            cfg.samples_per_eval_length, replace=True, random_state=fixed_seed
        )
        eval_csv = eval_csv.sort_values(length_column, ascending=False)

        return eval_csv

    @staticmethod
    def _dedupe_by_sequence_hash(
        metadata: MetadataDataFrame,
        logger: logging.Logger = None,
    ) -> MetadataDataFrame:
        """
        Drop exact duplicate sequences, keeping the first, using sequence hash columns.
        Redesign datasets should not have the same sequence, and will not be filtered unless they do.
        TODO - handle same sequence, different structure conformations (e.g. trajectory).
        - If `seq_hash_indep` exists, dedupe by it; else if `seq_hash` exists, use it.
        - Rows missing the hash are treated as unique and will not be dropped as duplicates.
        """
        # pick hash column
        hash_col: Optional[mc] = None
        if mc.seq_hash_indep in metadata.columns:
            hash_col = mc.seq_hash_indep
        elif mc.seq_hash in metadata.columns:
            hash_col = mc.seq_hash

        if hash_col is None:
            logger.info(
                "No sequence hash column found; skipping sequence deduplication"
            )
            return metadata

        len_before = len(metadata)

        # Create a dedupe key where NaNs are replaced with per-row unique identifiers
        # so that rows without a hash are never considered duplicates
        dedupe_key = metadata[hash_col].astype(object).copy()
        na_mask = pd.isna(dedupe_key)
        if na_mask.any():
            dedupe_key.loc[na_mask] = "NA_" + metadata.index[na_mask].astype(str)

        tmp_key = "__dedupe_key__"
        df_with_key = metadata.assign(**{tmp_key: dedupe_key})
        duplicates_mask = df_with_key.duplicated(subset=[tmp_key], keep="first")
        deduped = df_with_key.drop_duplicates(subset=[tmp_key], keep="first").drop(
            columns=[tmp_key]
        )

        if len(deduped) != len_before:
            logger.info(
                f"Dedupe by {hash_col} {len_before} -> {len(deduped)} (dropped {len_before - len(deduped)} duplicates; missing-hash rows kept: {int(na_mask.sum())})"
            )
            # Log per-dataset duplicate counts
            per_dataset_counts = (
                df_with_key.loc[duplicates_mask, DATASET_KEY].value_counts().to_dict()
            )
            # Only log if there were any duplicates by dataset
            if len(per_dataset_counts) > 0:
                logger.info(f"Duplicates dropped per dataset: {per_dataset_counts}")

        return deduped

    def _apply_test_split(self, metadata: MetadataDataFrame) -> MetadataDataFrame:
        """
        Split rows by date using cfg.test_date_cutoff and self.use_test.
        - Train: rows with date < cutoff
        - Test: rows with date >= cutoff
        If date column is missing or nan, assigned to training.

        All redesigns can be assigned to training by specifying `cfg.test_ignore_redesigns`.
        """
        if mc.date not in metadata.columns:
            # Assign all rows to training if date column is missing
            # For test split requests, return empty DataFrame
            if self.use_test:
                self._log.error(
                    "Date column missing; assigning all rows to training and none to test"
                )
                return metadata.iloc[0:0]
            return metadata

        cutoff = pd.to_datetime(self.cfg.test_date_cutoff, errors="coerce")
        dates = pd.to_datetime(metadata[mc.date].fillna("1970-01-01"), errors="coerce")

        # Determine which rows are redesigns, if redesign metadata exists
        handle_redesigns = (
            self.cfg.test_ignore_redesigns and rc.example in metadata.columns
        )
        is_redesign = metadata[rc.example].notna() if handle_redesigns else None

        if self.use_test:
            mask = (dates >= cutoff) & (~dates.isna())
            if is_redesign is not None:
                mask = mask & (~is_redesign)
            self._log.info(
                f"Test date cutoff {self.cfg.test_date_cutoff} split {len(metadata)} -> {mask.sum()} examples"
            )
        else:
            mask = (dates < cutoff) | (dates.isna())
            if is_redesign is not None:
                mask = mask | is_redesign
            self._log.info(
                f"Train date cutoff {self.cfg.test_date_cutoff} split {len(metadata)} -> {mask.sum()} examples"
            )

        return metadata[mask]

    def load_processed_file_with_caching(self, csv_row: DatasetCSVRow) -> ProcessedFile:
        """
        Loads a single structure + metadata pickled at `processed_file_path`, with caching.
        """
        processed_file_path = csv_row[mc.processed_path]
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


@dataclass
class DatasetConstructor:
    """Helper to create train + eval BaseDatasets from cfg.dataset"""

    cfg: DatasetConfig
    task: DataTask
    use_test: bool

    def create_datasets(self) -> Tuple[BaseDataset, BaseDataset]:
        """generate dataset, and possibly validation dataset"""
        train_dataset = BaseDataset(
            cfg=self.cfg,
            task=self.task,
            eval=False,
            use_test=self.use_test,
        )

        eval_dataset = BaseDataset(
            cfg=self.cfg,
            task=self.task,
            eval=True,
            use_test=self.use_test,
        )

        return train_dataset, eval_dataset

    @classmethod
    def from_cfg(cls, cfg: Config, use_test: bool = False):
        return cls(cfg=cfg.dataset, task=cfg.data.task, use_test=use_test)


@dataclass
class LengthSamplingDataset(Dataset):
    """
    During unconditional predictions/inference, dataset to generate samples across lengths
    """

    cfg: InferenceSamplesConfig

    def __post_init__(self):
        self._all_sample_ids = self.create_sample_lengths()

    def __len__(self):
        return len(self._all_sample_ids)

    def create_sample_lengths(self) -> List[Tuple[int, torch.Tensor]]:
        # determine lengths to sample
        if self.cfg.length_subset is not None:
            all_sample_lengths = [int(x) for x in self.cfg.length_subset]
        else:
            all_sample_lengths = range(
                self.cfg.min_length, self.cfg.max_length + 1, self.cfg.length_step
            )

        all_sample_ids = []
        num_batch = self.cfg.num_batch
        assert self.cfg.samples_per_length % num_batch == 0
        num_samples_per_batch = self.cfg.samples_per_length // num_batch

        for length in all_sample_lengths:
            for sample_id in range(num_samples_per_batch):
                sample_ids = torch.tensor(
                    [num_batch * sample_id + i for i in range(num_batch)]
                )
                all_sample_ids.append((length, sample_ids))
        return all_sample_ids

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
            bp.structure_method: StructureExperimentalMethod.default_tensor_feat(),
        }


@dataclass
class EvalDatasetConstructor:
    """
    Helper to create eval dataset
    may be unconditional (length sampling) or conditional (partial structures for inpainting)
    """

    cfg: InferenceSamplesConfig
    task: InferenceTask
    # dataset config required for inpainting, folding tasks
    dataset_cfg: Optional[DatasetConfig] = None
    # use test dataset for tasks using PDBs (inpainting, folding)
    use_test: bool = False

    def create_dataset(self) -> Union[LengthSamplingDataset, BaseDataset]:
        """generate eval dataset"""
        if self.task == InferenceTask.unconditional:
            return LengthSamplingDataset(cfg=self.cfg)
        elif (
            self.task == InferenceTask.inpainting
            or self.task == InferenceTask.forward_folding
            or self.task == InferenceTask.inverse_folding
        ):
            return BaseDataset(
                cfg=self._get_dataset_cfg(),
                task=InferenceTask.to_data_task(self.task),
                eval=True,
                use_test=self.use_test,
            )
        else:
            raise ValueError(f"Unsupported inference task {self.task}")

    def create_dataloader(
        self,
        batch_size: Optional[int] = None,
        shuffle: bool = False,
    ) -> torch.utils.data.DataLoader:
        """
        Create a DataLoader for the dataset.
        """
        if batch_size is None:
            batch_size = self.cfg.num_batch

        dataset = self.create_dataset()

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=False,
        )

    def _get_dataset_cfg(self) -> DatasetConfig:
        """Patch DatasetConfig for inference tasks"""
        assert (
            self.dataset_cfg is not None
        ), f"DatasetConfig required for task {self.task}"

        # Patch dataset config with relevant inference configuration
        # TODO - should not have to re-specify inference config in dataset config
        dataset_cfg = self.dataset_cfg.clone()
        filter_cfg = dataset_cfg.filter

        # Pass through relevant parameters
        dataset_cfg.samples_per_eval_length = self.cfg.samples_per_length
        filter_cfg.max_num_res = self.cfg.max_length
        filter_cfg.min_num_res = self.cfg.min_length

        # handle specifying length_subset vs min, max, step
        if self.cfg.length_subset is not None:
            dataset_cfg.num_eval_lengths = len(self.cfg.length_subset)
            dataset_cfg.max_eval_length = max(self.cfg.length_subset)
            dataset_cfg.filter.min_num_res = min(self.cfg.length_subset)
            dataset_cfg.filter.max_num_res = max(self.cfg.length_subset)
        else:
            dataset_cfg.max_eval_length = self.cfg.max_length
            dataset_cfg.num_eval_lengths = (
                (self.cfg.max_length - self.cfg.min_length) // self.cfg.length_step
            ) + 1
            dataset_cfg.filter.min_num_res = self.cfg.min_length
            dataset_cfg.filter.max_num_res = self.cfg.max_length

        # hacky handle multimer by update dataset filter
        # doesn't handle fraction correctly - that depends on data subset picked...
        if self.cfg.multimer_fraction == 0.0:
            dataset_cfg.filter.num_chains = [1]
        elif self.cfg.multimer_fraction == 1.0:
            dataset_cfg.filter.num_chains = [2, 3, 4]
            dataset_cfg.filter.oligomeric = None
        else:
            dataset_cfg.filter.num_chains = [1, 2, 3, 4]
            dataset_cfg.filter.oligomeric = None

        return dataset_cfg

    @classmethod
    def from_cfg(cls, cfg: Config):
        return cls(
            cfg=cfg.inference.samples,
            task=cfg.inference.task,
        )
