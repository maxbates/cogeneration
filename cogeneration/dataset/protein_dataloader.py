import logging
import math

import pandas as pd
import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler, dist

from cogeneration.config.base import DataConfig, DataSamplerConfig, DatasetConfig
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.type.dataset import MetadataColumn


class ProteinData(LightningDataModule):

    def __init__(
        self,
        *,
        data_cfg: DataConfig,
        train_dataset: BaseDataset,
        valid_dataset: BaseDataset,
        dataset_cfg: DatasetConfig,
        predict_dataset=None,
    ):
        super().__init__()
        self.data_cfg = data_cfg
        self.dataset_cfg = dataset_cfg
        self._train_dataset = train_dataset
        self._valid_dataset = valid_dataset
        self._predict_dataset = predict_dataset

    def train_dataloader(self, rank=None, num_replicas=None) -> DataLoader:
        num_workers = self.data_cfg.loader.num_workers

        batch_sampler = LengthBatcher(
            sampler_cfg=self.data_cfg.sampler,
            metadata_csv=self._train_dataset.csv,
            modeled_length_col=self.dataset_cfg.modeled_trim_method.to_dataset_column(),
            rank=rank,
            num_replicas=num_replicas,
        )

        return DataLoader(
            self._train_dataset,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            prefetch_factor=(
                None if num_workers == 0 else self.data_cfg.loader.prefetch_factor
            ),
            pin_memory=False,
            persistent_workers=True if num_workers > 0 else False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._valid_dataset,
            sampler=DistributedSampler(self._valid_dataset, shuffle=False),
            num_workers=4,
            prefetch_factor=2,
            pin_memory=False,
            persistent_workers=True,
        )

    def predict_dataloader(self) -> DataLoader:
        num_workers = self.data_cfg.loader.num_workers
        return DataLoader(
            self._predict_dataset,
            sampler=DistributedSampler(self._predict_dataset, shuffle=False),
            num_workers=num_workers,
            prefetch_factor=(
                None if num_workers == 0 else self.data_cfg.loader.prefetch_factor
            ),
            pin_memory=False,
            persistent_workers=True,
        )


class LengthBatcher:
    """
    Sampler that creates batches of proteins of the same length.
    Samplers return indices in the dataset.
    This sampler supports multiple replicas. The dataset is broken across `rank` replicas.
    """

    def __init__(
        self,
        *,
        sampler_cfg: DataSamplerConfig,
        metadata_csv: pd.DataFrame,
        modeled_length_col: MetadataColumn,
        seed=123,
        shuffle=True,
        num_replicas=None,
        rank=None,
    ):
        super().__init__()
        self._log = logging.getLogger(__name__)
        if num_replicas is None:
            self.num_replicas = dist.get_world_size()
        else:
            self.num_replicas = num_replicas
        if rank is None:
            self.rank = dist.get_rank()
        else:
            self.rank = rank

        self._sampler_cfg = sampler_cfg
        self._data_csv = metadata_csv
        self.modeled_length_col = modeled_length_col

        # Each replica needs the same number of batches. We set the number
        # of batches to arbitrarily be the number of examples per replica.
        if (
            "cluster" in self._data_csv.columns
            and not (self._data_csv.cluster.isna()).all()
        ):
            num_batches = self._data_csv["cluster"].nunique()
        else:
            num_batches = len(self._data_csv)

        self.overall_num_batches = num_batches
        self._num_batches = math.ceil(self.overall_num_batches / self.num_replicas)
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.max_batch_size = self._sampler_cfg.max_batch_size
        self._log.info(
            f"Created {self.__class__.__name__} dataloader rank {self.rank+1} out of {self.num_replicas}"
        )

    def _sample_indices(self):
        if "cluster" in self._data_csv and not (self._data_csv.cluster.isna()).all():
            cluster_sample = self._data_csv.groupby("cluster").sample(
                1, random_state=self.seed + self.epoch
            )
            return cluster_sample["index"].tolist()
        else:
            return self._data_csv["index"].tolist()

    def _replica_epoch_batches(self):
        # Make sure all replicas share the same seed on each epoch.
        rng = torch.Generator()
        rng.manual_seed(self.seed + self.epoch)
        indices = self._sample_indices()
        if self.shuffle:
            new_order = torch.randperm(len(indices), generator=rng).numpy().tolist()
            indices = [indices[i] for i in new_order]

        if len(self._data_csv) > self.num_replicas:
            replica_csv = self._data_csv.iloc[indices[self.rank :: self.num_replicas]]
        else:
            replica_csv = self._data_csv

        # Each batch contains multiple proteins of the same length,
        # and limit batch sizes to avoid OOM.
        sample_order = []
        for seq_len, len_df in replica_csv.groupby(self.modeled_length_col):
            max_batch_size = min(
                self.max_batch_size,
                self._sampler_cfg.max_num_res_squared // int(seq_len) ** 2 + 1,
            )
            num_batches = math.ceil(len(len_df) / max_batch_size)
            for i in range(num_batches):
                batch_df = len_df.iloc[i * max_batch_size : (i + 1) * max_batch_size]
                batch_indices = batch_df["index"].tolist()
                batch_repeats = math.floor(max_batch_size / len(batch_indices))
                sample_order.append(batch_indices * batch_repeats)

        # Shuffle to remove any length bias.
        if self.shuffle:
            new_order = (
                torch.randperm(len(sample_order), generator=rng).numpy().tolist()
            )
            return [sample_order[i] for i in new_order]
        return sample_order

    def _create_batches(self):
        # Make sure all replicas have the same number of batches Otherwise leads to bugs.
        # See bugs with shuffling https://github.com/Lightning-AI/lightning/issues/10947
        all_batches = []
        num_augments = -1
        while len(all_batches) < self._num_batches:
            all_batches.extend(self._replica_epoch_batches())
            num_augments += 1
            if num_augments > 1000:
                raise ValueError("Exceeded number of augmentations.")
        if len(all_batches) >= self._num_batches:
            all_batches = all_batches[: self._num_batches]
        self.sample_order = all_batches

    def __iter__(self):
        self._create_batches()
        self.epoch += 1
        return iter(self.sample_order)

    def __len__(self):
        return self._num_batches
