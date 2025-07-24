import hashlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import patch

import numpy as np
import numpy.typing as npt
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from cogeneration.config.base import PATH_PUBLIC_WEIGHTS, Config, InferenceSamplesConfig
from cogeneration.data import all_atom
from cogeneration.data.const import aatype_to_seq, seq_to_aatype
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.residue_constants import restype_order_with_x, restypes_with_x
from cogeneration.dataset.datasets import (
    BatchFeaturizer,
    DatasetConstructor,
    LengthSamplingDataset,
)
from cogeneration.dataset.process_pdb import process_chain_feats, process_pdb_file
from cogeneration.dataset.protein_dataloader import ProteinData
from cogeneration.dataset.test_utils import (
    MockDataloader,
    MockDataset,
    create_pdb_dataloader,
    create_pdb_noisy_batch,
)
from cogeneration.models.module import FlowModule
from cogeneration.scripts.utils_ddp import DDPInfo, setup_ddp
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.dataset import MetadataColumn
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import DataTask, InferenceTask

logging.basicConfig(level=logging.DEBUG)


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


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


@pytest.fixture
def pdb_noisy_batch(mock_cfg):
    return create_pdb_noisy_batch(mock_cfg)


@pytest.fixture
def mock_pred_unconditional_dataloader(request):
    # TODO(test) - increase default batch size > 1
    batch_size = getattr(request, "param", {}).get("batch_size", 1)

    length_sampling_dataset = LengthSamplingDataset(
        InferenceSamplesConfig(
            samples_per_length=1,
            num_batch=1,
            length_subset=[10],
        )
    )
    dataloader = DataLoader(length_sampling_dataset, batch_size=batch_size)
    return dataloader


@pytest.fixture
def mock_pred_conditional_dataloader(request, mock_cfg):
    """For `forward_folding` or `inverse_folding` tasks"""
    # TODO(test) - increase default batch size > 1
    batch_size = getattr(request, "param", {}).get("batch_size", 1)

    return create_pdb_dataloader(
        cfg=mock_cfg,
        task=DataTask.hallucination,
        training=False,
        eval_batch_size=batch_size,
    )


@pytest.fixture
def mock_pred_inpainting_dataloader(request, mock_cfg):
    # TODO(test) - increase default batch size > 1
    batch_size = getattr(request, "param", {}).get("batch_size", 1)

    return create_pdb_dataloader(
        cfg=mock_cfg,
        task=DataTask.inpainting,
        training=False,
        eval_batch_size=batch_size,
    )


@dataclass
class FoldingValidationMockValue:
    """
    Mock values for folding validation.
    First seq is used for AF2
    """

    seqs: List[Tuple[str, str]]
    true_aa: np.ndarray
    true_bb_pos: np.ndarray
    af2_pdb_path: str
    af2_df: pd.DataFrame  # folding of prediction
    mpnn_fasta_path: str  # inverse folded sequences
    designability_af2_df: pd.DataFrame  # folding of inverse folded sequences


@pytest.fixture
def mock_folding_validation(tmp_path):
    with patch(
        "cogeneration.data.folding_validation.FoldingValidator.inverse_fold_pdb"
    ) as mock_inverse_fold_pdb, patch(
        "cogeneration.data.folding_validation.FoldingValidator.fold_fasta"
    ) as mock_fold_fasta:

        def setup_mocks(batch, cfg: Config, n_inverse_folds: int):
            assert batch is not None, "batch is required for folding validation mock"
            assert cfg is not None, "cfg is required for folding validation mock"

            # TODO(inpainting) improve mock for inpainting
            if cfg.inference.task != InferenceTask.unconditional:
                print(
                    f"WARNING. mocks currently assume unconditional generation. May impact outputs. Got {cfg.inference.task}"
                )
            if batch[bp.res_mask].shape[0] != 1:
                print(
                    f"WARNING. mocks currently assume single sequence generation. May impact outputs. Got {batch[bp.res_mask].shape[0]} sequences"
                )

            # determine size of batch and residues, handling conditional and unconditional batches
            batch_size, num_res = batch[bp.res_mask].shape
            # use consistent chain_idx for all sequences (assume single sequence)
            chain_idx = batch[bp.chain_idx][0].cpu().detach().numpy()

            # Prep return values
            mock_run_protein_mpnn_calls = []
            mock_run_alphafold2_calls = []

            # Helper to create atom37 structure for AF2 predictions
            def dummy_structure(batch_size: int, num_res: int) -> npt.NDArray:
                return (
                    all_atom.atom37_from_trans_rot(
                        trans=torch.rand(batch_size, num_res, 3),
                        rots=torch.rand(batch_size, num_res, 3, 3),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                )

            # Helper to create DataFrame for folding.
            def create_af2_df(
                seq_names: List[str], seqs: List[str], af2_pdb_paths: List[str]
            ):
                return pd.DataFrame.from_records(
                    [
                        {
                            MetricName.header: header,
                            MetricName.sequence: sequence,
                            MetricName.folded_pdb_path: af2_pdb_path,
                            MetricName.plddt_mean: np.random.rand() * 100,
                        }
                        for (header, sequence, af2_pdb_path) in zip(
                            seq_names, seqs, af2_pdb_paths
                        )
                    ]
                )

            # Generate mock values for each item in the batch
            mock_values: List[FoldingValidationMockValue] = []
            for i in range(batch_size):
                # generate N random sequences for inverse folding
                mock_seqs: List[Tuple[str, str]] = [
                    (
                        f"{batch[bp.pdb_name][i] if (bp.pdb_name in batch) else ''}pmpnn_seq_{n}",
                        aatype_to_seq(
                            np.random.randint(0, 20, num_res), chain_idx=chain_idx
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
                            torsions=batch[bp.torsions_1],
                            aatype=batch[bp.aatypes_1],
                            unknown_to_alanine=True,
                        )
                        .cpu()
                        .detach()
                        .numpy()
                    )
                else:
                    # mock it, e.g. for unconditional batches
                    true_bb_pos = dummy_structure(
                        batch_size=batch_size, num_res=num_res
                    )

                # mock folding: AlphaFold2 PDB and DataFrame
                # Currently, only run and care about model_4
                # Actual af2_pdb_path may include indexing
                af2_pdb_path = write_prot_to_pdb(
                    prot_pos=true_bb_pos[0],
                    file_path=str(tmp_path / "model_4.pdb"),
                    aatype=true_aa,
                    chain_idx=chain_idx,
                )

                # Mock folding the designed seq
                # we don't have easy access to sample_name / sample_id here, but it doesn't really matter
                af2_df = create_af2_df(
                    seq_names=[
                        (
                            batch[bp.pdb_name][0]
                            if (bp.pdb_name in batch)
                            else "mocked_true"
                        )
                    ],
                    seqs=[aatype_to_seq(true_aa, chain_idx=chain_idx)],
                    af2_pdb_paths=[af2_pdb_path],
                )
                mock_run_alphafold2_calls.append(af2_df)

                # mock inverse folding: ProteinMPNN fasta
                mock_mpnn_fasta_path = str(tmp_path / "mpnn.fasta")
                mock_run_protein_mpnn_calls.append(mock_mpnn_fasta_path)
                with open(mock_mpnn_fasta_path, "w") as f:
                    for mock_seq_name, mock_inverse_fold_seq in mock_seqs:
                        f.write(f">{mock_seq_name}\n")
                        f.write(mock_inverse_fold_seq)

                # For designability, we also fold the inverse folded sequences.
                # Always done for validation. Usually done for prediction...
                # This means folding a single fasta with multiple sequences.

                # First we need to write a PDB for each inverse folded sequence
                # (so the correct sequence is in the PDB)
                af2_inverse_fold_paths = [
                    write_prot_to_pdb(
                        prot_pos=dummy_structure(
                            batch_size=1, num_res=len(mock_inverse_fold_seq)
                        )[0],
                        file_path=str(tmp_path / f"inv_fold_{i}_model_4.pdb"),
                        aatype=np.array(seq_to_aatype(mock_inverse_fold_seq)),
                        chain_idx=chain_idx,
                    )
                    for (i, (_, mock_inverse_fold_seq)) in enumerate(mock_seqs)
                ]

                # Mock second round of folding: designability
                designability_af2_df = create_af2_df(
                    seq_names=[seq[0] for seq in mock_seqs],
                    seqs=[seq[1] for seq in mock_seqs],
                    af2_pdb_paths=af2_inverse_fold_paths,
                )
                mock_run_alphafold2_calls.append(designability_af2_df)

                mock_value = FoldingValidationMockValue(
                    seqs=mock_seqs,
                    true_aa=true_aa,
                    true_bb_pos=true_bb_pos,
                    af2_pdb_path=af2_pdb_path,
                    af2_df=af2_df,
                    mpnn_fasta_path=mock_mpnn_fasta_path,
                    designability_af2_df=designability_af2_df,
                )
                mock_values.append(mock_value)

            # assign the returns to the mocked methods
            mock_inverse_fold_pdb.side_effect = mock_run_protein_mpnn_calls
            mock_fold_fasta.side_effect = mock_run_alphafold2_calls

            return mock_values

        yield setup_mocks


@pytest.fixture
def mock_checkpoint(mock_folding_validation):
    """
    Save a dummy checkpoint and config that we can load into EvalRunner
    Returns updated cfg, with the checkpoint path set, and the path to ckpt.
    Provide `path` to save the checkpoint if improperly set in `cfg`, defaults to `cfg.experiment.checkpointer.dirpath`

    Note this is sort of slow, because we actually have to call `Trainer.fit()`
    to save the checkpoint, though we only train for one step, which takes ~30-60s on a Mac with MPS.
    """

    def create_mock_checkpoint(
        cfg: Config, path: Optional[Path] = None
    ) -> Tuple[Config, str]:
        ckpt_dir = (
            path if path is not None else Path(cfg.experiment.checkpointer.dirpath)
        )
        ckpt_cfg_path = str(ckpt_dir / "config.yaml")
        ckpt_path = str(ckpt_dir / "last.ckpt")
        final_ckpt_path = str(ckpt_dir / "final.ckpt")

        # update config with the checkpoint
        cfg.inference.unconditional_ckpt_path = str(ckpt_path)
        cfg.inference.inpainting_ckpt_path = str(ckpt_path)
        cfg.inference.forward_folding_ckpt_path = str(ckpt_path)
        cfg.inference.inverse_folding_ckpt_path = str(ckpt_path)

        # save config
        os.makedirs(ckpt_dir, exist_ok=True)
        with open(ckpt_cfg_path, "w") as f:
            OmegaConf.save(config=cfg, f=f)

        # We cache a checkpoint across test runs, hashing on `cfg.model` to ensure compatibility
        cache_dir = os.path.join(os.path.dirname(__file__), ".cache", "model_ckpt")
        # use md5 because built-in hash() uses new seed each run
        model_hash = hashlib.md5(str(cfg.model).encode()).hexdigest()
        cache_ckpt_path = os.path.join(cache_dir, f"{model_hash}.ckpt")
        if os.path.exists(cache_ckpt_path):
            print(f"Using cached model checkpoint at {cache_ckpt_path}")
            # copy the cached checkpoint to the new location
            os.link(cache_ckpt_path, ckpt_path)
            os.link(cache_ckpt_path, final_ckpt_path)
            return cfg, ckpt_path

        print(
            f"Creating new model checkpoint at {ckpt_path} -> cache {cache_ckpt_path}"
        )

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
            **cfg.experiment.trainer.asdict(),
        }
        trainer = Trainer(**trainer_cfg)

        # call `fit()` to set the model
        trainer.fit(module, datamodule=datamodule)

        # finally, save checkpoint
        trainer.save_checkpoint(ckpt_path)

        # save final checkpoint sentinel, just a symlink
        if cfg.experiment.save_final_ckpt:
            os.link(ckpt_path, final_ckpt_path)

        # and save to the ckpt cache
        os.makedirs(os.path.dirname(cache_ckpt_path), exist_ok=True)
        trainer.save_checkpoint(cache_ckpt_path)

        return cfg, ckpt_path

    return create_mock_checkpoint


@pytest.fixture
def pdb_2qlw_path():
    """
    https://www2.rcsb.org/structure/2QLW
    This protein has a fair amount of weird stuff going on to make it a good test case:
    - it is a dimer
    - has small molcules after chains that should be trimmed
    - has small molecules between chains (MG and FMT)
    - has non-residue atom types (MG atoms)
    - has some unknown residues (like seleniummethionine)
    - has preceding sequence without atoms
    - glycine starts at position -1 in each chain
    """
    return Path(__file__).parent / "dataset" / "2qlw.pdb"


@pytest.fixture
def pdb_2pdz_path():
    """
    https://www2.rcsb.org/structure/2PDZ
    Trajectory with 15 models
    """
    return Path(__file__).parent / "dataset" / "2pdz.pdb"


@pytest.fixture
def pdb_8y2h_path():
    """
    https://www2.rcsb.org/structure/8Y2H
    Big tetramer, with 2 little hooks
    """
    return Path(__file__).parent / "dataset" / "8y2h.pdb"


@pytest.fixture
def pdb_2qlw_processed_feats(pdb_2qlw_path, mock_cfg):
    """Generate features from the 2QLW PDB file. Note includes centering, chain randomization, etc."""
    processed_file = process_pdb_file(str(pdb_2qlw_path), "2qlw")
    processed_file = process_chain_feats(
        processed_file,
        center=True,
        trim_to_modeled_residues=True,
        trim_chains_independently=True,
    )

    featurizer = BatchFeaturizer(
        cfg=mock_cfg.dataset,
        task=DataTask.hallucination,
        is_training=False,
    )
    features = featurizer.featurize_processed_file(
        processed_file=processed_file,
        csv_row={
            MetadataColumn.pdb_name: "2qlw",
            MetadataColumn.processed_path: "",
        },
    )

    return features
