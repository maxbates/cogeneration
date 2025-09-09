from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from cogeneration.config.base import Config
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.noise_mask import uniform_so3
from cogeneration.dataset.contacts import get_contact_conditioning_matrix
from cogeneration.dataset.datasets import DatasetConstructor
from cogeneration.dataset.protein_dataloader import LengthBatcher
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.batch import NoisyBatchProp as nbp
from cogeneration.type.batch import NoisyFeatures
from cogeneration.type.structure import StructureExperimentalMethod
from cogeneration.type.task import DataTask


def mock_feats(
    N: int, idx: int, task: DataTask = DataTask.hallucination, multimer: bool = False
) -> BatchFeatures:
    """
    Creates random features for a protein of length N
    Assumes `task=unconditional`, i.e. `(diffuse_mask == 1.0).all()`
    """
    feats: BatchFeatures = {}

    # N residue protein, random frames
    feats[bp.res_mask] = torch.ones(N).int()
    feats[bp.diffuse_mask] = torch.ones(N).int()

    feats[bp.aatypes_1] = torch.randint(0, 20, (N,))  # may contain UNK (20)
    feats[bp.trans_1] = torch.rand(N, 3) * 10.0
    feats[bp.rotmats_1] = uniform_so3(1, N, device=torch.device("cpu")).squeeze(0)
    feats[bp.torsions_1] = torch.rand(N, 7, 2)
    feats[bp.res_bfactor] = (torch.rand(N) * 50.0).float()
    feats[bp.structure_method] = StructureExperimentalMethod.default_tensor_feat()
    feats[bp.res_plddt] = ((torch.rand(N) + 0.3) * 100.0).clamp(0, 100).float()
    feats[bp.plddt_mask] = feats[bp.res_bfactor] > 60.0
    feats[bp.hot_spots] = torch.zeros(N).int()
    feats[bp.stochastic_scale] = torch.tensor([1.0]).float()

    # index
    feats[bp.chain_idx] = torch.ones(N)  # 1-indexed
    feats[bp.res_idx] = torch.arange(1, N + 1)  # 1-indexed
    if multimer:
        assert N > 7
        chain_break = np.random.randint(3, N - 3)
        feats[bp.chain_idx][chain_break:] = 2
        feats[bp.res_idx][chain_break:] = feats[bp.res_idx][chain_break:] - chain_break

    if task == DataTask.inpainting:
        # set a motif_mask where some middle portion of the protein is scaffolded
        feats[bp.motif_mask] = torch.ones(N).int()
        feats[bp.motif_mask][slice(*sorted(torch.randint(0, N, (2,))))] = 0

        feats[bp.contact_conditioning] = get_contact_conditioning_matrix(
            feats[bp.trans_1],
            res_mask=feats[bp.res_mask],
            chain_idx=feats[bp.chain_idx],
            motif_mask=feats.get(bp.motif_mask, None),
        )

    # metadata
    feats[bp.pdb_name] = f"test_{idx}"
    feats[bp.csv_idx] = torch.tensor([0])

    # inference-only feats  # TODO(mock) remove, dedicated mock function
    feats[bp.sample_id] = f"test_{idx}"

    return feats


def mock_noisy_feats(
    N: int, idx: int, task: DataTask = DataTask.hallucination, multimer: bool = False
) -> NoisyFeatures:
    """
    Create random and corrupted features for a protein of length N.
    Assumes `task=unconditional` generation - the entire sequence + structure is corrupted.
    For other tasks, use `Interpolant.corrupt_batch` to corrupt the features.
    """
    feats: NoisyFeatures = mock_feats(N=N, idx=idx, task=task, multimer=multimer)

    # generate corrupted noisy values for input_feats
    t = torch.rand(())  # scalar so batches collate to shape (B,)
    feats[nbp.so3_t] = t
    feats[nbp.r3_t] = t
    feats[nbp.cat_t] = t
    feats[nbp.trans_t] = torch.rand(N, 3)
    feats[nbp.rotmats_t] = torch.rand(N, 3, 3)
    feats[nbp.torsions_t] = torch.rand(N, 7, 2)
    feats[nbp.aatypes_t] = torch.rand(N) * 20  # AA seq as floats
    feats[nbp.trans_sc] = torch.rand(N, 3)
    feats[nbp.aatypes_sc] = torch.rand(N, 21)  # include mask token

    return feats


class MockDataset(Dataset):
    """
    Creates mock dataset with `len(sample_lengths)` samples.

    If `corrupt`, the samples are mock-corrupted into `NoisyFeatures` as in unconditional generation,
    else they are `BatchFeatures`.
    """

    def __init__(
        self,
        sample_lengths: Optional[List[int]] = None,
        task: DataTask = DataTask.hallucination,
        corrupt: bool = False,
        multimer: bool = False,
    ):
        if sample_lengths is None:
            sample_lengths = [20]
        assert len(sample_lengths) > 0
        self.sample_lengths = sample_lengths

        self.task = task
        self.corrupt = corrupt
        self.multimer = multimer

        self.data = self._create_mock_data()

    def _create_mock_data(self):
        all_items = []

        for i, N in enumerate(self.sample_lengths):
            if self.corrupt:
                all_items.append(
                    mock_noisy_feats(N=N, idx=i, task=self.task, multimer=self.multimer)
                )
            else:
                all_items.append(
                    mock_feats(N=N, idx=i, task=self.task, multimer=self.multimer)
                )

        return all_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class MockDataloader(DataLoader):
    """
    Creates `task=unconditional` mock dataloader with `len(sample_lengths)` samples, batch size `batch_size`.

    Note that batches must be the same length, so if batch_size > 1,
    in dataloader, create sets of samples with the same length
    """

    def __init__(
        self,
        batch_size: int = 1,
        sample_lengths: Optional[List[int]] = None,
        task: DataTask = DataTask.hallucination,
        corrupt: bool = False,
        multimer: bool = False,
    ):
        dataset = MockDataset(
            sample_lengths=sample_lengths, task=task, corrupt=corrupt, multimer=multimer
        )
        super().__init__(dataset, batch_size=batch_size)


def create_pdb_dataloader(
    cfg: Config,
    task: Optional[DataTask] = None,
    training: bool = True,
    eval_batch_size: int = 1,
) -> DataLoader:
    """
    Creates a Dataloader for PDB dataset given `cfg`
    Returns `train_dataset` if `training` is True, else `eval_dataset`.
    """
    if task is None:
        task = cfg.data.task

    dataset_constructor = DatasetConstructor(
        cfg=cfg.dataset,
        task=task,
        use_test=False,
    )

    train_dataset, eval_dataset = dataset_constructor.create_datasets()

    if training:
        # batch sampler required to sample batch size > 1
        # we borrow convention from MultiFlow to batch by length rather than pad
        # modify sampler cfg to specify `max_batch_size`.
        batch_sampler = LengthBatcher(
            sampler_cfg=cfg.data.sampler,
            metadata_csv=train_dataset.csv,
            modeled_length_col=cfg.dataset.modeled_trim_method.to_dataset_column(),
            rank=0,
            num_replicas=1,
        )

        dataloader = DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            num_workers=0,
        )
    else:
        dataloader = DataLoader(
            eval_dataset,
            batch_size=eval_batch_size,
            num_workers=0,
        )

    return dataloader


def create_pdb_batch(
    cfg: Config, training: bool = True, eval_batch_size: int = 1
) -> BatchFeatures:
    """
    Creates a single training batch of a PDB dataset given `cfg`
    """
    dataloader = create_pdb_dataloader(
        cfg=cfg, task=cfg.data.task, training=training, eval_batch_size=eval_batch_size
    )

    raw_feats = next(iter(dataloader))

    return raw_feats


def create_pdb_noisy_batch(
    cfg: Config, training: bool = True, eval_batch_size: int = 1
) -> NoisyFeatures:
    """
    Creates a single corrupted batch of a PDB dataset given `cfg`
    """
    raw_feats = create_pdb_batch(
        cfg=cfg, training=training, eval_batch_size=eval_batch_size
    )
    interpolant = Interpolant(cfg=cfg.interpolant)
    noisy_feats = interpolant.corrupt_batch(raw_feats, task=cfg.data.task)
    return noisy_feats


def create_single_item_batch(batch_item: BatchFeatures) -> BatchFeatures:
    """
    Creates a single-item batch from a single batch item.

    Args:
        batch_item: Single batch item (BatchFeatures)

    Returns:
        Single-item batch (BatchFeatures) with batch dimension added
    """

    # Create a simple dataset that returns the single item
    class SingleItemDataset(Dataset):
        def __init__(self, item):
            self.item = item

        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return self.item

    # Create dataloader and get the batch
    dataset = SingleItemDataset(batch_item)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
    batch = next(iter(dataloader))

    return batch
