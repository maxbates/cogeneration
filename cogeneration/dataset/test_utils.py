from typing import List, Optional

from torch.utils.data import DataLoader, Dataset

from cogeneration.config.base import Config
from cogeneration.data.interpolant import Interpolant
from cogeneration.dataset.datasets import DatasetConstructor
from cogeneration.dataset.protein_dataloader import LengthBatcher
from cogeneration.dataset.util import mock_noisy_feats


class MockDataset(Dataset):
    """
    Creates mock dataset.
    Note that batches must be the same length, so if batch_size > 1, create sets of samples with the same length
    """

    def __init__(self, sample_lengths: Optional[List[int]] = None):
        if sample_lengths is None:
            sample_lengths = [10]
        assert len(sample_lengths) > 0
        self.sample_lengths = sample_lengths

        self.data = self._create_mock_data()

    def _create_mock_data(self):
        all_items = []

        for i, N in enumerate(self.sample_lengths):
            all_items.append(mock_noisy_feats(N=N, idx=i))

        return all_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_pdb_batch(cfg: Config):
    """
    Creates a single training batch of a PDB dataset given `cfg`
    """
    dataset_constructor = DatasetConstructor.pdb_dataset(
        dataset_cfg=cfg.dataset,
    )

    train_dataset, eval_dataset = dataset_constructor.create_datasets()

    # batch sampler required to sample batch size > 1
    # we borrow convention from MultiFlow to batch by length rather than pad
    # modify sampler cfg to specify `max_batch_size`.
    batch_sampler = LengthBatcher(
        sampler_cfg=cfg.data.sampler,
        metadata_csv=train_dataset.csv,
        rank=0,
        num_replicas=1,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=0,
    )

    raw_feats = next(iter(dataloader))

    return raw_feats


def create_pdb_noisy_batch(cfg: Config):
    """
    Creates a single corrupted batch of a PDB dataset given `cfg`
    """
    raw_feats = create_pdb_batch(cfg)
    interpolant = Interpolant(cfg.interpolant)
    noisy_feats = interpolant.corrupt_batch(raw_feats, task=cfg.data.task)
    return noisy_feats
