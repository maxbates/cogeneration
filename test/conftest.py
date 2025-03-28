import logging
import os
from dataclasses import asdict, dataclass
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
from cogeneration.dataset.protein_dataloader import LengthBatcher, ProteinData
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.config.base import PATH_PUBLIC_WEIGHTS

logging.basicConfig(level=logging.DEBUG)


@pytest.fixture
def public_weights_path() -> Path:
    """
    returns Path to public weights directory, containing a `config.yaml` and `last.ckpt`

    These weights must be downloaded separately.
    """
    # check paths
    assert os.path.exists(
        PATH_PUBLIC_WEIGHTS
    ), f"""Public weights not found at {PATH_PUBLIC_WEIGHTS}
    Public weights must be downloaded for some tests to work.
    
    See https://zenodo.org/records/10714631?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjJjMTk2YjlmLTM4OTUtNGVhYi1hODcxLWE1ZjExOTczY2IzZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDY5ZDUzYjVjMTNhNDllMDYxNmI3Yjc2NjcwYjYxZiJ9.C2eZZmRu-nu7H330G-DkV5kttfjYB3ANozdOMNm19uPahvtLrDRvd_4Eqlyb7lp24m06e4OHhHQ4zlj68S1O_A
    """
    assert os.path.exists(
        PATH_PUBLIC_WEIGHTS / "config.yaml"
    ), f"Public config not found at {public_weights_path}"
    assert os.path.exists(
        PATH_PUBLIC_WEIGHTS / "last.ckpt"
    ), f"Public ckpt not found at {public_weights_path}"

    return PATH_PUBLIC_WEIGHTS


@pytest.fixture
def mock_cfg_uninterpolated(tmp_path) -> Config:
    """
    mock_cfg_uninterpolated fixture defines default test config.
    It is not yet interpolated, so parameters like those in `shared` can be modified before interpolation.
    """
    return Config.test_uninterpolated(tmp_path=tmp_path)


@pytest.fixture
def mock_cfg(mock_cfg_uninterpolated) -> Config:
    """
    mock_cfg fixture defines default test config.
    """
    return mock_cfg_uninterpolated.interpolate()


def create_pdb_noisy_batch(cfg: Config):
    dataset_constructor = DatasetConstructor.pdb_train_validation(
        dataset_cfg=cfg.dataset,
    )

    train_dataset, _ = dataset_constructor.create_datasets()

    # batch sampler required to sample batch size > 1
    # we borrow convention from MultiFlow to batch by length rather than pad
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

    interpolant = Interpolant(cfg.interpolant)

    raw_feats = next(iter(dataloader))
    input_feats = interpolant.corrupt_batch(raw_feats)

    return input_feats


@pytest.fixture
def pdb_noisy_batch(mock_cfg):
    return create_pdb_noisy_batch(mock_cfg)


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
            input_feats = {}

            # N residue protein, random frames
            input_feats[bp.num_res] = torch.tensor([N])
            input_feats[bp.res_mask] = torch.ones(N)
            input_feats[bp.aatypes_1] = torch.randint(0, 20, (N,))  # AA seq as ints
            input_feats[bp.trans_1] = torch.rand(N, 3)
            input_feats[bp.rotmats_1] = torch.rand(N, 3, 3)
            input_feats[bp.torsion_angles_sin_cos_1] = torch.rand(N, 7, 2)
            input_feats[bp.chain_idx] = torch.zeros(N)
            input_feats[bp.res_idx] = torch.arange(N)
            input_feats[bp.pdb_name] = f"test_{i}"
            input_feats[bp.res_plddt] = torch.floor(torch.rand(N) + 0.5)
            input_feats[bp.plddt_mask] = input_feats[bp.res_plddt] > 0.6
            input_feats[bp.diffuse_mask] = torch.ones(N)
            input_feats[bp.csv_idx] = torch.tensor([0])

            # generate corrupted noisy values for input_feats
            t = torch.rand(1)  # use same value as with unconditional + not separate_t
            input_feats[nbp.so3_t] = t
            input_feats[nbp.r3_t] = t
            input_feats[nbp.cat_t] = t
            input_feats[nbp.trans_t] = torch.rand(N, 3)
            input_feats[nbp.rotmats_t] = torch.rand(N, 3, 3)
            input_feats[nbp.aatypes_t] = (
                torch.rand(N) * 20
            )  # amino acid sequence as floats
            input_feats[nbp.trans_sc] = torch.rand(N, 3)
            input_feats[nbp.aatypes_sc] = torch.rand(N, 21)  # include mask token

            all_items.append(input_feats)

        return all_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


@pytest.fixture
def mock_dataloader(request):
    batch_size = request.param.get("batch_size", 1)
    sample_lengths = request.param.get("sample_lengths", None)

    dataset = MockDataset(sample_lengths=sample_lengths)
    dataloader = DataLoader(dataset, batch_size=batch_size)
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


@dataclass
class FoldingValidationMockValue:
    """
    Mock values for folding validation.
    First seq is used for AF2
    """

    seqs: List[Tuple[str, str]]
    true_aa: np.ndarray
    true_bb_pos: np.ndarray
    mpnn_fasta_path: str
    af2_pdb_path: str
    af2_df: pd.DataFrame


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
                    f"WARNING. mocks currently assume unconditional generation. May impact outputs. Got {cfg.inference.task}"
                )

            # determine size of batch and residues, handling conditional and unconditional batches
            if bp.res_mask in batch:
                batch_size, num_res = batch[bp.res_mask].shape
            else:
                batch_size = batch[bp.num_res].shape[0]
                num_res = batch[bp.num_res][0].item()

            # Prep return values
            mock_run_protein_mpnn_calls = []
            mock_run_alphafold2_calls = []

            # Generate mock values for each item in the batch
            mock_values: List[FoldingValidationMockValue] = []
            for i in range(batch_size):
                # generate N random sequences for inverse folding
                mock_seqs: List[Tuple[str, str]] = [
                    (
                        f"{batch[bp.pdb_name][i] if (bp.pdb_name in batch) else ''}pmpnn_seq_{n}",
                        "".join(
                            [
                                restypes_with_x[x]
                                for x in np.random.randint(0, 20, num_res)
                            ]
                        ),
                    )
                    for n in range(n_inverse_folds)
                ]

                # determine true aa and true bb, using batch if available
                if bp.aatypes_1 in batch:
                    true_aa = batch[bp.aatypes_1][0].cpu().detach().numpy()
                else:
                    # mock it, e.g. for unconditional batches
                    true_aa = np.random.randint(0, 20, num_res)

                if bp.trans_1 in batch and bp.rotmats_1 in batch:
                    true_bb_pos = (
                        all_atom.atom37_from_trans_rot(
                            trans=batch[bp.trans_1],
                            rots=batch[bp.rotmats_1],
                            psi_torsions=batch[bp.torsion_angles_sin_cos_1][..., 2, :],
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                else:
                    # mock it, e.g. for unconditional batches
                    true_bb_pos = (
                        all_atom.atom37_from_trans_rot(
                            trans=torch.rand(batch_size, num_res, 3),
                            rots=torch.rand(batch_size, num_res, 3, 3),
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )

                # mock inverse folding: ProteinMPNN fasta
                mock_mpnn_fasta_path = str(tmp_path / "mpnn.fasta")
                mock_run_protein_mpnn_calls.append(mock_mpnn_fasta_path)
                with open(mock_mpnn_fasta_path, "w") as f:
                    for mock_seq_name, mock_inverse_fold_seq in mock_seqs:
                        f.write(f">{mock_seq_name}\n")
                        f.write(mock_inverse_fold_seq)

                # mock folding: AlphaFold2 PDB and DataFrame
                # Currently, only run and care about model_4
                # Actual af2_pdb_path may include indexing
                af2_pdb_path = write_prot_to_pdb(
                    prot_pos=true_bb_pos[0],
                    file_path=str(tmp_path / "model_4.pdb"),
                    aatype=true_aa,
                )

                # Helper to create DataFrame for AF2 return. For now just reuse PDB across multiple seqs.
                def create_af2_df(seq_names: List[str], af2_pdb_path: str):
                    return pd.DataFrame.from_records(
                        [
                            {
                                MetricName.header: header,
                                MetricName.folded_pdb_path: af2_pdb_path,
                                MetricName.plddt_mean: np.random.rand() * 100,
                            }
                            for header in seq_names
                        ]
                    )

                # Mock folding the designed seq
                # we don't have easy access to sample_name / sample_id here, but it doesn't really matter
                af2_df = create_af2_df(
                    seq_names=[mock_seqs[0][0]], af2_pdb_path=af2_pdb_path
                )
                mock_run_alphafold2_calls.append(af2_df)

                # For designability, we also fold the inverse folded sequences.
                # Always done for validation. Usually done for prediction...
                # This means folding a single fasta with multiple sequences.
                # So additional call to mock_run_alphafold2
                mock_run_alphafold2_calls.append(
                    create_af2_df(
                        seq_names=[seq[0] for seq in mock_seqs],
                        af2_pdb_path=af2_pdb_path,
                    )
                )

                mock_value = FoldingValidationMockValue(
                    seqs=mock_seqs,
                    true_aa=true_aa,
                    true_bb_pos=true_bb_pos,
                    mpnn_fasta_path=mock_mpnn_fasta_path,
                    af2_pdb_path=af2_pdb_path,
                    af2_df=af2_df,
                )
                mock_values.append(mock_value)

            # assign the returns to the mocked methods
            mock_run_protein_mpnn.side_effect = mock_run_protein_mpnn_calls
            mock_run_alphafold2.side_effect = mock_run_alphafold2_calls

            return mock_values

        yield setup_mocks


@pytest.fixture
def mock_checkpoint(mock_folding_validation):
    """
    Save a dummy checkpoint and config that we can load into EvalRunner
    Returns updated cfg, with the checkpoint path set, and the path to ckpt

    Note this is sort of slow, because we actually have to call `Trainer.fit()`
    to save the checkpoint, though we only train for one step.

    TODO - memoize, maybe if path is not provided?
    #   Need to confirm cfg is equivalent enough to use as checkpoint? or overwrite when changes made?
    #   Maybe we can hash the config (ignoring fields like `now`)
    """

    def create_mock_checkpoint(cfg: Config, path: Path) -> Tuple[Config, str]:
        ckpt_cfg_path = str(path / "config.yaml")
        ckpt_path = str(path / "last.ckpt")

        # update config with the checkpoint
        assert cfg.inference.task == InferenceTaskEnum.unconditional
        cfg.inference.unconditional_ckpt_path = str(ckpt_path)
        cfg.inference.forward_folding_ckpt_path = str(ckpt_path)
        cfg.inference.inverse_folding_ckpt_path = str(ckpt_path)

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
        trainer_cfg = {
            "max_steps": 1,  # curtail actual training
            **asdict(cfg.experiment.trainer),
        }
        trainer = Trainer(**trainer_cfg)

        # call `fit()` to set the model
        trainer.fit(module, datamodule=datamodule)

        # finally, save checkpoint
        trainer.save_checkpoint(ckpt_path)

        return cfg, ckpt_path

    return create_mock_checkpoint
