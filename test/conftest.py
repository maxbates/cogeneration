import logging
import os
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Dataset

from cogeneration.config.base import (
    Config,
    InferenceSamplesConfig,
    InferenceTaskEnum,
    ModelHyperParamsConfig,
)
from cogeneration.data import all_atom
from cogeneration.data.batch_props import BatchProps as bp
from cogeneration.data.batch_props import NoisyBatchProps as nbp
from cogeneration.data.enum import MetricName
from cogeneration.data.interpolant import Interpolant
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restypes_with_x
from cogeneration.dataset.datasets import DatasetConstructor, LengthSamplingDataset
from cogeneration.dataset.protein_dataloader import ProteinData
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def public_weights_path() -> Path:
    """
    returns Path to public weights directory, containing a `config.yaml` and `last.ckpt`

    These weights must be downloaded separately.
    """
    # check paths
    public_weights_path = (Path(__file__).parent / "../multiflow_weights").resolve()
    assert os.path.exists(
        public_weights_path
    ), f"""Public weights not found at {public_weights_path}
    Public weights must be downloaded for some tests to work.
    
    See https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A
    """
    assert os.path.exists(
        public_weights_path / "config.yaml"
    ), f"Public config not found at {public_weights_path}"
    assert os.path.exists(
        public_weights_path / "last.ckpt"
    ), f"Public ckpt not found at {public_weights_path}"

    return public_weights_path


@pytest.fixture
def mock_cfg(tmp_path) -> Config:
    """mock_cfg fixture defines default nested config"""
    raw_cfg = Config()

    # set to local mode, impacting accelerator etc.
    raw_cfg.shared.local = True

    # default to tiny model for faster model evaluations
    raw_cfg.model.hyper_params = ModelHyperParamsConfig.tiny()

    # filter to small PDBs for faster model + sampling
    raw_cfg.dataset.filter.min_num_res = 20
    raw_cfg.dataset.filter.max_num_res = 40

    # set output directories to temp paths
    raw_cfg.experiment.checkpointer.dirpath = str(tmp_path / "ckpt")
    raw_cfg.inference.predict_dir = str(tmp_path / "inference")

    # limit number of lengths sampled for validation / inference
    raw_cfg.inference.interpolant.sampling.num_timesteps = 10
    raw_cfg.inference.samples.samples_per_length = 2
    raw_cfg.inference.samples.length_subset = [10, 30]
    raw_cfg.interpolant.sampling.num_timesteps = 10
    raw_cfg.dataset.samples_per_eval_length = 2
    raw_cfg.dataset.num_eval_lengths = 1
    # shortest validation samples in public data are 60 residues
    raw_cfg.dataset.max_eval_length = 63

    return raw_cfg.interpolate()


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


@pytest.fixture
def mock_folding_validation(tmp_path):
    with patch(
        "cogeneration.data.folding_validation.FoldingValidator._run_protein_mpnn"
    ) as mock_run_protein_mpnn, patch(
        "cogeneration.data.folding_validation.FoldingValidator._run_alphafold2"
    ) as mock_run_alphafold2:

        def setup_mocks(batch, cfg: Config, n_inverse_folds: int):
            assert batch is not None, "batch is required for folding validation mock"
            assert cfg is not None, "cfg is required for folding validation mock"

            if cfg.inference.task != InferenceTaskEnum.unconditional:
                print(
                    "WARNING. mocks currently assume unconditional generation. May impact outputs."
                )

            # determine size of batch and residues, handling conditional and unconditional batches
            if bp.res_mask in batch:
                batch_size, num_res = batch[bp.res_mask].shape
            else:
                batch_size = batch[bp.num_res].shape[0]
                num_res = batch[bp.num_res][0].item()
            assert batch_size == 1, "Test expects batch size of 1"

            # generate N random sequences for inverse folding
            mock_seqs: List[Tuple[str, str]] = [
                (
                    f"pmpnn_seq_{i}",
                    "".join(
                        [restypes_with_x[x] for x in np.random.randint(0, 20, num_res)]
                    ),
                )
                for i in range(n_inverse_folds)
            ]

            # determine true aa and true bb, using batch if available
            if bp.aatypes_1 in batch:
                true_aa = batch[bp.aatypes_1][0].cpu().detach().numpy()
            else:
                # mock it, e.g. for unconditional batches
                true_aa = np.random.randint(0, 20, num_res)
            true_sequence = "".join([restypes_with_x[x] for x in true_aa])

            if bp.trans_1 in batch and bp.rotmats_1 in batch:
                true_bb_pos = (
                    all_atom.atom37_from_trans_rot(
                        trans=batch[bp.trans_1],
                        rots=batch[bp.rotmats_1],
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )
            else:
                # mock it, e.g. for unconditional batches
                true_bb_pos = (
                    all_atom.atom37_from_trans_rot(
                        trans=torch.rand(1, num_res, 3),
                        rots=torch.rand(1, num_res, 3, 3),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

            # mock ProteinMPNN call
            mock_mpnn_fasta_path = str(tmp_path / "mpnn.fasta")
            mock_run_protein_mpnn.return_value = mock_mpnn_fasta_path
            with open(mock_mpnn_fasta_path, "w") as f:
                for mock_seq_name, mock_inverse_fold_seq in mock_seqs:
                    f.write(f">{mock_seq_name}\n")
                    f.write(mock_inverse_fold_seq)

            # mock AlphaFold2 call
            mock_af2_pdb_path = str(tmp_path / "model_4.pdb")
            af2_fasta_path = write_prot_to_pdb(
                prot_pos=true_bb_pos[0],
                file_path=mock_af2_pdb_path,
                aatype=true_aa,
            )
            mock_run_alphafold2.return_value = pd.DataFrame(
                [
                    {
                        MetricName.header: mock_seq_name,
                        MetricName.folded_pdb_path: af2_fasta_path,
                        MetricName.plddt_mean: np.random.rand() * 100,
                    }
                ]
            )

            return mock_run_protein_mpnn, mock_run_alphafold2

        yield setup_mocks


@pytest.fixture
def mock_checkpoint(mock_folding_validation):
    """
    Save a dummy checkpoint and config that we can load into EvalRunner

    TODO - memoize, maybe if path is not provided?
    #   Need to confirm cfg is equivalent enough to use as checkpoint? or overwrite when changes made?
    #   Maybe we can hash the config (ignoring fields like `now`)
    """

    def create_mock_checkpoint(cfg: Config, path: Path) -> Tuple[str, str]:
        ckpt_cfg_path = str(path / "config.yaml")
        ckpt_path = str(path / "last.ckpt")

        # save config
        with open(ckpt_cfg_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)

        # saving a checkpoint with pytorch lightning is annoying.
        # We need to use a lightning `Trainer`, and call `fit()` on it, which requires a datamodule.

        # Our datamodule requires DDP for now, so set it up
        setup_ddp(
            trainer_strategy=cfg.experiment.trainer.strategy,
            accelerator=cfg.experiment.trainer.accelerator,
            rank=str(DDPInfo.from_env().rank),
            world_size=str(cfg.experiment.num_devices),
        )

        # set up datamodule
        dataset_constructor = DatasetConstructor.from_cfg(cfg)
        train_dataset, valid_dataset = dataset_constructor.create_datasets()
        datamodule = ProteinData(
            data_cfg=cfg.data,
            dataset_cfg=cfg.dataset,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
        )

        # Patch config to only allow one training epoch.
        cfg.experiment.trainer.max_epochs = 1
        cfg.experiment.trainer.min_epochs = 0

        # mock validation step before calling `fit()`
        val_batch = next(iter(datamodule.val_dataloader()))
        mock_folding_validation(
            batch=val_batch,
            cfg=cfg,
            n_inverse_folds=1,  # validation
        )

        # set up trainer
        module = FlowModule(cfg)
        trainer = Trainer(**asdict(cfg.experiment.trainer))

        # call `fit()` to set the model
        trainer.fit(module, datamodule=datamodule)

        # finally, save checkpoint
        trainer.save_checkpoint(ckpt_path)

        return ckpt_cfg_path, ckpt_path

    return create_mock_checkpoint
