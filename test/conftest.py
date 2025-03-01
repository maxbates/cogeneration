import logging

import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

from cogeneration.config.base import (
    Config,
    InferenceSamplesConfig,
    ModelHyperParamsConfig,
)
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.interpolant import Interpolant
from cogeneration.dataset.datasets import (
    DatasetConstructor,
    LengthSamplingDataset,
    PdbDataset,
)

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def mock_cfg():
    """mock_cfg fixture defines default nested config"""
    raw_cfg = Config()

    # default to tiny model
    raw_cfg.model.hyper_params = ModelHyperParamsConfig.tiny()

    # interpolate etc. using OmegaConf
    # use to_object() so that intermediate structs are dataclasses, not DictConfig
    cfg: Config = OmegaConf.to_object(OmegaConf.create(raw_cfg))

    return cfg


@pytest.fixture
def pdb_noisy_batch(mock_cfg):
    dataset_constructor = DatasetConstructor.pdb_train_validation(
        dataset_cfg=mock_cfg.dataset,
    )
    train_dataset, valid_dataset = dataset_constructor.create_datasets()

    dataloader = DataLoader(train_dataset, batch_size=1)
    interpolant = Interpolant(mock_cfg.interpolant)

    raw_feats = next(iter(dataloader))
    input_feats = interpolant.corrupt_batch(raw_feats)

    return input_feats


class MockDataset(Dataset):
    # TODO - extend to support multiple samples, different lengths
    def __init__(self):
        self.data = self._create_mock_data()

    def _create_mock_data(self):
        input_feats = {}

        # N residue protein, random frames
        N = 10
        input_feats[bp.num_res] = torch.tensor([N])
        input_feats[bp.res_mask] = torch.ones(N)
        input_feats[bp.aatypes_1] = torch.randint(0, 20, (N,))  # AA seq as ints
        input_feats[bp.trans_1] = torch.rand(N, 3)
        input_feats[bp.rotmats_1] = torch.rand(N, 3, 3)
        input_feats[bp.torsion_angles_sin_cos_1] = torch.rand(N, 7, 2)
        input_feats[bp.chain_idx] = torch.zeros(N)
        input_feats[bp.res_idx] = torch.arange(N)
        input_feats[bp.pdb_name] = "test"
        input_feats[bp.res_plddt] = torch.floor(torch.rand(N) + 0.5)
        input_feats[bp.plddt_mask] = input_feats[bp.res_plddt] > 0.6
        input_feats[bp.diffuse_mask] = torch.ones(N)
        input_feats[bp.csv_idx] = torch.tensor([0])

        # generate corrupted noisy values for input_feats
        t = torch.rand(1)  # use same value but really they are independent
        input_feats[nbp.so3_t] = t
        input_feats[nbp.r3_t] = t
        input_feats[nbp.cat_t] = t
        input_feats[nbp.trans_t] = torch.rand(N, 3)
        input_feats[nbp.rotmats_t] = torch.rand(N, 3, 3)
        input_feats[nbp.aatypes_t] = torch.rand(N) * 20  # amino acid sequence as floats
        input_feats[nbp.trans_sc] = torch.rand(N, 3)
        input_feats[nbp.aatypes_sc] = torch.rand(N, 21)  # include mask token

        return input_feats

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data


@pytest.fixture
def mock_dataloader():
    dataset = MockDataset()
    dataloader = DataLoader(dataset, batch_size=1)
    return dataloader


@pytest.fixture
def mock_pred_unconditional_dataloader():
    length_sampling_dataset = LengthSamplingDataset(
        InferenceSamplesConfig(
            samples_per_length=1,
            num_batch=1,
            length_subset=[10],
        )
    )
    dataloader = DataLoader(length_sampling_dataset, batch_size=1)
    return dataloader


@pytest.fixture
def mock_pred_conditional_dataloader(mock_cfg):
    dataset_constructor = DatasetConstructor.pdb_test(
        dataset_cfg=mock_cfg.dataset,
    )
    eval_dataset, _ = dataset_constructor.create_datasets()

    dataloader = DataLoader(eval_dataset, batch_size=1)
    return dataloader
