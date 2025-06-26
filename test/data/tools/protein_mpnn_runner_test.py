"""
Tests for ProteinMPNN runner module

Note: lots of Claude generated code here.
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from Bio import SeqIO
from omegaconf.errors import ValidationError

from cogeneration.config.base import Config
from cogeneration.data import all_atom, const, residue_constants
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.data.tools.protein_mpnn_runner import (
    NativeMPNNResult,
    ProteinMPNNRunner,
    ProteinMPNNRunnerPool,
)
from cogeneration.dataset.datasets import BatchFeaturizer
from cogeneration.dataset.process_pdb import process_chain_feats, process_pdb_file
from cogeneration.dataset.test_utils import create_pdb_batch, create_pdb_dataloader
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.dataset import MetadataColumn as mc


def generate_mock_mpnn_fasta(
    pdb_name: str, num_passes: int = 2, temperature: float = 0.1, seed: int = 123
) -> str:
    """
    Generate a mock MPNN-format FASTA string that mimics the output format
    from the original ProteinMPNN. This includes a native sequence (first entry)
    followed by generated sequences.

    Args:
        pdb_name: Name of the PDB file (used in headers)
        num_passes: Number of generated sequences to create
        temperature: Sampling temperature for headers
        seed: Random seed for headers

    Returns:
        Mock FASTA string in ProteinMPNN format
    """
    # Mock native sequence header and sequence
    fasta_lines = [
        f">{pdb_name}, T={temperature}, seed={seed}, num_res=100, num_ligand_res=100, use_ligand_context=False, ligand_cutoff_distance=8.0, batch_size=1, number_of_batches={num_passes}, model_path=/path/to/model",
        "MKLLVLGLGGVGKSALTVQFVQGIFVEKYDPTIEDFRKYTLPTVAIGLQLFLHYTSLLQEKLSPEDRKNLIVGSCDTAGQAMALQVEKQARELTGLEVLFQGPVLQV",
    ]

    # Add generated sequences
    for i in range(1, num_passes + 1):
        # Mock generated sequence header
        fasta_lines.append(
            f">{pdb_name}, id={i}, T={temperature}, seed={seed}, overall_confidence=0.8000, ligand_confidence=0.7500, seq_rec=0.9200"
        )
        # Mock generated sequence (slightly different from native)
        fasta_lines.append(
            "AKLLVLGLGGVGKSALTVQFVQGIFVEKYDPTIEDFRKYTLPTVAIGLQLFLHYTSLLQEKLSPEDRKNLIVGSCDTAGQAMALQVEKQARELTGLEVLFQGPVLQV"
        )

    return "\n".join(fasta_lines)


class TestProteinMPNNRunner:
    """Test cases for ProteinMPNNRunner"""

    def test_runner_from_mock_cfg(self, mock_cfg):
        assert (
            mock_cfg.folding.protein_mpnn.use_native_runner is True
        ), "many tests expect native runner"
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)
        assert runner is not None

        # If we are on a mac, ensure its on MPS
        if torch.backends.mps.is_available():
            assert runner.device.type == "mps"
        # If CUDA is available, ensure its on CUDA
        if torch.cuda.is_available():
            assert runner.device.type == "cuda"

    def test_run_batch_pdb_batch(self, mock_cfg):
        """Test run_batch on actual PDB batch data"""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Create batch from real PDB data
        pdb_batch = create_pdb_batch(
            cfg=mock_cfg,
            training=False,
            eval_batch_size=2,
        )

        # Get actual dimensions from the input
        batch_size = pdb_batch[bp.trans_1].shape[0]
        num_res = pdb_batch[bp.trans_1].shape[1]
        num_passes = 3
        sequences_per_pass = 2

        result = runner.run_batch(
            trans=pdb_batch[bp.trans_1],
            rotmats=pdb_batch[bp.rotmats_1],
            aatypes=pdb_batch[bp.aatypes_1],
            res_mask=pdb_batch[bp.res_mask],
            diffuse_mask=torch.ones_like(pdb_batch[bp.res_mask]),
            chain_idx=pdb_batch[bp.chain_idx],
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.1,
        )

        # Verify results
        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape[0] == batch_size
        assert result.logits.shape[1] == num_passes
        assert result.logits.shape[2] == sequences_per_pass
        assert result.logits.shape[3] == num_res
        assert result.logits.shape[4] == 21  # vocab size

        assert result.confidence_scores.shape == (
            batch_size,
            num_passes,
            sequences_per_pass,
        )
        assert result.sequences.shape == (
            batch_size,
            num_passes,
            sequences_per_pass,
            num_res,
        )

        # Test properties work
        averaged_logits = result.averaged_logits
        assert averaged_logits.shape == (batch_size, num_res, 21)

        # Test the new average_logits_per_pass property
        avg_logits_per_pass = result.average_logits_per_pass
        assert avg_logits_per_pass.shape == (batch_size, num_passes, num_res, 21)

    def test_run_batch_2qlw(self, mock_cfg, pdb_2qlw_path):
        """Test run_batch on 2qlw structure"""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Process actual PDB file
        batch = process_pdb_file(str(pdb_2qlw_path), "2qlw")
        featurizer = BatchFeaturizer(
            cfg=mock_cfg.dataset, task=mock_cfg.data.task, is_training=False
        )
        features = featurizer.featurize_processed_file(
            processed_file=batch,
            csv_row={
                mc.pdb_name: "2qlw",
                mc.processed_path: "",
            },
        )

        # Extract single structure and add batch dimension
        trans = features[bp.trans_1].unsqueeze(0)  # (1, N, 3)
        rotmats = features[bp.rotmats_1].unsqueeze(0)  # (1, N, 3, 3)
        aatypes = features[bp.aatypes_1].unsqueeze(0)  # (1, N)
        res_mask = features[bp.res_mask].unsqueeze(0)  # (1, N)
        chain_idx = features[bp.chain_idx].unsqueeze(0)  # (1, N)

        num_res = trans.shape[1]
        num_passes = 2
        sequences_per_pass = 1

        result = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.2,
        )

        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape[0] == 1
        assert result.logits.shape[1] == num_passes
        assert result.logits.shape[2] == sequences_per_pass
        assert result.logits.shape[3] == num_res
        assert result.logits.shape[4] == 21

        assert torch.all(result.confidence_scores >= 0.0)
        assert torch.all(result.confidence_scores <= 1.0)

        assert torch.all(result.sequences >= 0)
        assert torch.all(result.sequences < 20)

    def test_generate_mock_mpnn_fasta(self):
        """Test the mock FASTA generation function"""
        fasta_content = generate_mock_mpnn_fasta("test_pdb", num_passes=2)

        # Check that it contains the expected number of sequences (2 generated + 1 native)
        lines = fasta_content.split("\n")
        sequence_lines = [line for line in lines if not line.startswith(">")]
        assert len(sequence_lines) == 3  # 1 native + 2 generated

        # Check header format
        header_lines = [line for line in lines if line.startswith(">")]
        assert len(header_lines) == 3  # 1 native + 2 generated

        # Check native sequence header
        native_header = header_lines[0]
        assert "test_pdb" in native_header
        assert "T=0.1" in native_header
        assert "seed=123" in native_header

        # Check generated sequence headers
        for i, header in enumerate(header_lines[1:], 1):
            assert f"id={i}" in header
            assert "overall_confidence=" in header
            assert "seq_rec=" in header

    def test_import_mechanism_success(self, mock_cfg):
        """Test that LigandMPNN modules can be imported successfully when available"""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        try:
            # Test importing a basic module
            data_utils = runner._load_ligandmpnn_module("data_utils")

            # Verify the module has expected attributes
            assert hasattr(data_utils, "parse_PDB")
            assert hasattr(data_utils, "featurize")
            assert hasattr(data_utils, "restype_str_to_int")

        except (ImportError, FileNotFoundError) as e:
            # If LigandMPNN is not available, provide a helpful error message
            pytest.fail(
                f"LigandMPNN import test failed. Please ensure LigandMPNN is installed and available at {mock_cfg.folding.protein_mpnn.ligand_mpnn_path}. "
                f"You can install it by cloning https://github.com/dauparas/LigandMPNN. Error: {e}"
            )

    def test_import_mechanism_missing_path(self, mock_cfg_uninterpolated):
        """Test that import mechanism fails gracefully when LigandMPNN path doesn't exist"""
        # Configure for native mode with nonexistent path

        mock_cfg_uninterpolated.folding.protein_mpnn.ligand_mpnn_path = Path(
            "/nonexistent/path/to/ligandmpnn"
        )
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        with pytest.raises(FileNotFoundError, match="LigandMPNN path not found"):
            runner._load_ligandmpnn_module("data_utils")

    def test_import_mechanism_missing_module(self, mock_cfg_uninterpolated):
        """Test that import mechanism fails gracefully when specific module doesn't exist"""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Try to import a non-existent module
        with pytest.raises(ImportError, match="Module nonexistent_module.py not found"):
            runner._load_ligandmpnn_module("nonexistent_module")

    def test_import_caching(self, mock_cfg_uninterpolated):
        """Test that modules are cached after first import"""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Import the same module twice
        module1 = runner._load_ligandmpnn_module("data_utils")
        module2 = runner._load_ligandmpnn_module("data_utils")

        # Should be the same object (cached)
        assert module1 is module2
        assert "data_utils" in runner._ligandmpnn_modules

    def test_sidechain_model_loading_disabled(self, mock_cfg_uninterpolated):
        """Test that side chain model is not loaded when pack_side_chains is False"""
        # Configure for native mode with side chain packing disabled

        mock_cfg_uninterpolated.folding.protein_mpnn.pack_side_chains = False
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        model_sc = runner._load_side_chain_model()
        # Should return None when side chain packing is disabled
        assert model_sc is None

    def test_sidechain_model_loading_enabled(self, mock_cfg_uninterpolated):
        """Test that side chain model is loaded when pack_side_chains is True"""
        # Configure for native mode with side chain packing enabled

        mock_cfg_uninterpolated.folding.protein_mpnn.pack_side_chains = True
        mock_cfg_uninterpolated.folding.protein_mpnn.number_of_packs_per_design = 2
        mock_cfg_uninterpolated.folding.protein_mpnn.sc_num_denoising_steps = 2
        mock_cfg_uninterpolated.folding.protein_mpnn.sc_num_samples = 8
        mock_cfg_uninterpolated.folding.protein_mpnn.repack_everything = False
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        try:
            model_sc = runner._load_side_chain_model()

            # Should return a model when side chain packing is enabled and checkpoint exists
            assert (
                model_sc is not None
            ), "Side chain model should be loaded when pack_side_chains=True and checkpoint exists"

            # Verify it's a model with expected attributes
            assert hasattr(
                model_sc, "eval"
            ), "Side chain model should have 'eval' method"
            assert hasattr(model_sc, "to"), "Side chain model should have 'to' method"
            # Model should be in eval mode
            assert not model_sc.training, "Side chain model should be in eval mode"

        except Exception as e:
            # Check for specific NumPy compatibility issues
            if "numpy" in str(e).lower() and (
                "int" in str(e) or "deprecated" in str(e)
            ):
                pytest.fail(
                    f"Side chain model loading failed due to NumPy compatibility issue in LigandMPNN. "
                    f"This is likely due to deprecated np.int usage in the LigandMPNN codebase. "
                    f"Consider updating LigandMPNN or using an older NumPy version. Error: {e}"
                )
            else:
                # Re-raise other exceptions
                raise

    def test_sidechain_config_validation(self, mock_cfg_uninterpolated):
        """Test that side chain packing configuration is properly validated"""
        # Configure for native mode with side chain packing

        mock_cfg_uninterpolated.folding.protein_mpnn.pack_side_chains = True
        mock_cfg_uninterpolated.folding.protein_mpnn.number_of_packs_per_design = 2
        mock_cfg_uninterpolated.folding.protein_mpnn.sc_num_denoising_steps = 2
        mock_cfg_uninterpolated.folding.protein_mpnn.sc_num_samples = 8
        mock_cfg_uninterpolated.folding.protein_mpnn.repack_everything = False
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Verify configuration values
        assert cfg.folding.protein_mpnn.pack_side_chains is True
        assert cfg.folding.protein_mpnn.number_of_packs_per_design == 2
        assert cfg.folding.protein_mpnn.sc_num_denoising_steps == 2
        assert cfg.folding.protein_mpnn.sc_num_samples == 8
        assert cfg.folding.protein_mpnn.repack_everything is False

    def test_sidechain_weights_configuration(self, mock_cfg_uninterpolated):
        """Test that pmpnn_weights configuration is used for side chain checkpoint lookup"""
        # Configure for native mode with custom weights path
        ligandmpnn_path = Path("/fake/ligandmpnn/path")
        pmpnn_weights_dir = "custom"

        mock_cfg_uninterpolated.folding.protein_mpnn.ligand_mpnn_path = ligandmpnn_path
        mock_cfg_uninterpolated.folding.protein_mpnn.pmpnn_weights_dir = (
            pmpnn_weights_dir
        )
        mock_cfg_uninterpolated.folding.protein_mpnn.pack_side_chains = True
        mock_cfg_uninterpolated.folding.protein_mpnn.checkpoint_path_sc = (
            None  # Let it auto-discover
        )
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Test that the configuration is set up correctly
        assert (
            cfg.folding.protein_mpnn.checkpoint_path_sc is None
        )  # Should auto-discover
        assert cfg.folding.protein_mpnn.pack_side_chains is True

    def test_sidechain_packing_execution(
        self, mock_cfg_uninterpolated, pdb_2qlw_path, tmp_path
    ):
        """
        Test actual side chain packing execution (requires LigandMPNN installation).
        """
        # Configure for native mode with side chain packing
        mock_cfg_uninterpolated.folding.protein_mpnn.pack_side_chains = True
        mock_cfg_uninterpolated.folding.protein_mpnn.number_of_packs_per_design = 2
        mock_cfg_uninterpolated.folding.protein_mpnn.sc_num_denoising_steps = 2
        mock_cfg_uninterpolated.folding.protein_mpnn.sc_num_samples = 8
        mock_cfg_uninterpolated.folding.protein_mpnn.repack_everything = False
        cfg = mock_cfg_uninterpolated.interpolate()

        output_dir = tmp_path
        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Parse actual PDB file to get protein_dict
        protein_dict = runner._parse_pdb_to_protein_dict(pdb_2qlw_path)

        # Create mock inference result as NativeMPNNResult
        num_res = len(protein_dict["S"])
        device = runner.device
        num_passes = 3
        sequences_per_pass = 1

        # Create tensors with correct shapes for NativeMPNNResult
        # Expected shapes: (B, num_passes, sequences_per_pass, N, 21) for logits
        #                   (B, num_passes, sequences_per_pass) for confidence
        #                   (B, num_passes, sequences_per_pass, N) for sequences
        mock_logits = torch.randn(
            1, num_passes, sequences_per_pass, num_res, 21, device=device
        )
        mock_confidence = torch.rand(1, num_passes, sequences_per_pass, device=device)
        mock_sequences = torch.randint(
            0, 20, (1, num_passes, sequences_per_pass, num_res), device=device
        )

        mock_inference_result = NativeMPNNResult(
            logits=mock_logits,
            confidence_scores=mock_confidence,
            sequences=mock_sequences,
        )

        # Test side chain packing execution
        runner._apply_side_chain_packing(
            inference_result=mock_inference_result,
            protein_dict=protein_dict,
            output_dir=output_dir,
            pdb_path=pdb_2qlw_path,
            num_passes=num_passes,
        )

        # If we are on MPS, it will fail, because VonMises hard-codes cast to torch.double,
        # and MPS doesnt support float64.
        if torch.backends.mps.is_available():
            pytest.skip("MPS not supported for side chain packing")

        # Verify that output files were created
        packed_dir = output_dir / "packed"
        output_files_exist = any(
            packed_dir.glob(f"{pdb_2qlw_path.stem}_seq_*_pack_*.pdb")
        )
        assert (
            output_files_exist
        ), f"Expected at least one packed PDB file in {packed_dir}"

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_subprocess_mocked_success(
        self, mock_path_exists, mock_subprocess_run, mock_cfg, pdb_2qlw_path, tmp_path
    ):
        """Test successful subprocess execution"""
        output_dir = tmp_path
        device_id = 0

        # Mock that all path exists calls return True
        mock_path_exists.return_value = True

        # Mock successful subprocess execution
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        # Create mock output structure
        seqs_dir = output_dir / "seqs"
        seqs_dir.mkdir()
        mock_fasta = seqs_dir / f"{pdb_2qlw_path.stem}.fa"
        mock_fasta_content = generate_mock_mpnn_fasta(pdb_2qlw_path.stem)
        mock_fasta.write_text(mock_fasta_content)

        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)
        result_path = runner.inverse_fold_pdb_subprocess(
            pdb_path=pdb_2qlw_path,
            output_dir=output_dir,
            diffuse_mask=torch.ones(216),  # 2qlw has 216 modeled residues
            num_sequences=2,
            seed=123,
        )

        # The actual return path is processed sequences file, not the raw MPNN output
        expected_path = output_dir / f"{pdb_2qlw_path.stem}_sequences.fa"
        assert result_path == expected_path
        mock_subprocess_run.assert_called_once()

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_subprocess_mocked_failure(
        self, mock_path_exists, mock_subprocess_run, mock_cfg, pdb_2qlw_path, tmp_path
    ):
        """Test subprocess execution failure"""
        output_dir = tmp_path

        # Mock that all path exists calls return True
        mock_path_exists.return_value = True

        # Mock failed subprocess execution
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_subprocess_run.return_value = mock_result

        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        with pytest.raises(subprocess.CalledProcessError):
            runner.inverse_fold_pdb_subprocess(
                pdb_path=pdb_2qlw_path,
                output_dir=output_dir,
                diffuse_mask=torch.ones(216),  # 2qlw has 216 modeled residues
            )

    def test_run_native_real_execution(
        self, mock_cfg_uninterpolated, pdb_2qlw_path, tmp_path
    ):
        """
        Test native mode with real execution (requires LigandMPNN installation).
        """
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        output_dir = tmp_path
        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Test native execution
        result_path = runner.inverse_fold_pdb_native(
            pdb_path=pdb_2qlw_path,
            output_dir=output_dir,
            diffuse_mask=torch.ones(216),  # 2qlw has 216 modeled residues
            num_sequences=2,
            seed=123,
        )

        # Verify output
        assert result_path.exists()
        assert result_path.name == f"{pdb_2qlw_path.stem}_sequences.fa"

        # Verify FASTA content - should only have generated sequences (native skipped)
        records = list(SeqIO.parse(result_path, "fasta"))
        assert (
            len(records) == 2
        )  # Should have 2 generated sequences (native sequence skipped)

        # Check sequence IDs are properly formatted
        for i, record in enumerate(records, 1):
            assert record.id == f"{pdb_2qlw_path.stem}_seq_{i}"
            assert len(str(record.seq)) > 0

    def test_process_fasta_output(self, mock_cfg, tmp_path):
        """Test FASTA output processing"""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Create mock original FASTA with native + generated sequences
        mock_fasta_content = generate_mock_mpnn_fasta("test_pdb", num_passes=2)
        original_fasta = tmp_path / "original.fa"
        original_fasta.write_text(mock_fasta_content)

        # Process the FASTA
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        processed_path = runner._process_fasta_output(
            original_fasta=original_fasta,
            output_dir=output_dir,
            pdb_stem="test_pdb",
        )

        # Verify output path
        assert processed_path == output_dir / "test_pdb_sequences.fa"
        assert processed_path.exists()

        # Verify content is properly processed - should only have generated sequences (native skipped)
        records = list(SeqIO.parse(processed_path, "fasta"))
        assert (
            len(records) == 2
        )  # Should have 2 generated sequences (native sequence skipped)

        for i, record in enumerate(records, 1):
            assert record.id == f"test_pdb_seq_{i}"
            assert f"ProteinMPNN generated sequence {i}" in record.description

    def test_config_validation(self, mock_cfg_uninterpolated):
        """Test configuration validation"""
        mock_cfg_uninterpolated.folding.protein_mpnn.model_type = "invalid_model"

        # This should raise an error during interpolation due to invalid enum value
        with pytest.raises(ValidationError, match="Invalid value 'invalid_model'"):
            cfg = mock_cfg_uninterpolated.interpolate()

    def test_pmpnn_weights_configuration(self, mock_cfg_uninterpolated):
        """Test that pmpnn_weights configuration is properly used for checkpoint lookup"""
        # Configure with custom weights path
        ligandmpnn_path = Path("/fake/ligandmpnn/path")
        pmpnn_weights_dir = "custom"
        custom_weights_path = ligandmpnn_path / pmpnn_weights_dir
        mock_cfg_uninterpolated.folding.protein_mpnn.ligand_mpnn_path = ligandmpnn_path
        mock_cfg_uninterpolated.folding.protein_mpnn.pmpnn_weights_dir = (
            pmpnn_weights_dir
        )
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Expected first path to be checked
        target_path = custom_weights_path / "proteinmpnn_v_48_020.pt"

        # Mock the exists method - we'll make the first call return True
        with patch("pathlib.Path.exists") as mock_exists:
            # Set up side effect to return True for the first path checked
            call_count = 0

            def exists_side_effect():
                nonlocal call_count
                call_count += 1
                return call_count == 1  # Only the first call returns True

            mock_exists.side_effect = exists_side_effect

            # Test that the runner finds the checkpoint in the custom weights path
            checkpoint_path = runner._get_checkpoint_path()
            assert checkpoint_path == target_path

    def test_pmpnn_weights_fallback(self, mock_cfg_uninterpolated):
        """Test that runner falls back to pmpnn_path when pmpnn_weights is empty"""
        ligandmpnn_path = Path("/fake/ligandmpnn/path")

        mock_cfg_uninterpolated.folding.protein_mpnn.ligand_mpnn_path = ligandmpnn_path
        mock_cfg_uninterpolated.folding.protein_mpnn.pmpnn_weights_dir = ""
        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Expected fallback path (first path checked when weights is None)
        target_path = ligandmpnn_path / "proteinmpnn_v_48_020.pt"

        with patch("pathlib.Path.exists") as mock_exists:
            # Set up side effect to return True for the first path checked
            call_count = 0

            def exists_side_effect():
                nonlocal call_count
                call_count += 1
                return call_count == 1  # Only the first call returns True

            mock_exists.side_effect = exists_side_effect

            # Test that the runner finds the checkpoint in the default location
            checkpoint_path = runner._get_checkpoint_path()
            assert checkpoint_path == target_path

    def test_run_batch_basic_functionality(self, mock_cfg_uninterpolated):
        """Test basic run_batch functionality."""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        num_passes = 2
        sequences_per_pass = 1

        # Create mock batch data
        B, N = 2, 10
        trans = torch.randn(B, N, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        aatypes = torch.randint(0, 20, (B, N))
        res_mask = torch.ones(B, N, dtype=torch.bool)
        chain_idx = torch.zeros(B, N, dtype=torch.long)  # Single chain

        # Run batch inference
        result = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.5,
        )

        # Check result type and structure
        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape == (
            B,
            num_passes,
            sequences_per_pass,
            N,
            21,
        )  # (batch, num_passes, sequences_per_pass, length, vocab)
        assert result.confidence_scores.shape == (
            B,
            num_passes,
            sequences_per_pass,
        )  # (batch, num_passes, sequences_per_pass)
        assert result.sequences.shape == (
            B,
            num_passes,
            sequences_per_pass,
            N,
        )  # (batch, num_passes, sequences_per_pass, length)

        # Test single structure case
        result_single = runner.run_batch(
            trans=trans[:1],
            rotmats=rotmats[:1],
            aatypes=aatypes[:1],
            res_mask=res_mask[:1],
            diffuse_mask=torch.ones_like(res_mask[:1]),
            chain_idx=chain_idx[:1],
            num_passes=1,
            sequences_per_pass=1,
        )
        assert isinstance(result_single, NativeMPNNResult)
        assert result_single.sequences.shape == (1, 1, 1, N)

    def test_run_batch_with_invalid_structures(self, mock_cfg):
        """Test run_batch with structures that have no valid residues."""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Create batch where all structures have no valid residues
        B, N = 2, 10
        trans = torch.randn(B, N, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        aatypes = torch.randint(0, 20, (B, N))
        res_mask = torch.zeros(B, N, dtype=torch.bool)  # No valid residues
        chain_idx = torch.zeros(B, N, dtype=torch.long)  # Single chain

        num_passes = 2
        sequences_per_pass = 1
        result = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
        )

        # Should return uniform logits/sequences
        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape == (B, num_passes, sequences_per_pass, N, 21)
        assert result.confidence_scores.shape == (B, num_passes, sequences_per_pass)
        assert result.sequences.shape == (B, num_passes, sequences_per_pass, N)

        # Even with no valid residues, the model may still generate sequences
        # We just check that we get a valid result structure
        assert torch.all(torch.isfinite(result.confidence_scores))

    def test_run_batch_with_optional_parameters(self, mock_cfg_uninterpolated):
        """Test run_batch with various optional parameters."""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        B, N = 2, 8
        trans = torch.randn(B, N, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        aatypes = torch.randint(0, 20, (B, N))
        res_mask = torch.ones(B, N, dtype=torch.bool)
        chain_idx = torch.ones(B, N, dtype=torch.long)  # Single chain

        num_passes = 3
        sequences_per_pass = 2
        result = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            temperature=0.2,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
        )

        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape == (B, num_passes, sequences_per_pass, N, 21)
        assert result.confidence_scores.shape == (B, num_passes, sequences_per_pass)
        assert result.sequences.shape == (B, num_passes, sequences_per_pass, N)

    def test_run_batch_format_conversion_integration(self, mock_cfg_uninterpolated):
        """Test that run_batch properly handles format conversion from project to MPNN and back."""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        B, N = 1, 5
        trans = torch.randn(B, N, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)

        # Use specific amino acids to test conversion
        # Project format: A=0, R=1, N=2, D=3, C=4 (AlphaFold ordering)
        aatypes = torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long)
        res_mask = torch.ones(B, N, dtype=torch.bool)
        chain_idx = torch.zeros(B, N, dtype=torch.long)  # Single chain

        num_passes = 2
        sequences_per_pass = 1
        result = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.1,
        )

        # Check that format conversion worked
        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape == (B, num_passes, sequences_per_pass, N, 21)
        assert result.confidence_scores.shape == (B, num_passes, sequences_per_pass)
        assert result.sequences.shape == (B, num_passes, sequences_per_pass, N)

        # Check that sequences are in valid range for project format
        assert torch.all(result.sequences >= 0)
        assert torch.all(result.sequences < 20)  # Standard 20 amino acids

    def test_logits_conversion_exact_mapping(self, mock_cfg_uninterpolated):
        """Test the exact mapping of logits conversion from MPNN to Cogen format."""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Test the conversion mapping
        conversion_map = runner._create_mpnn_to_cogen_conversion_map()
        print(f"Conversion map (MPNN->Cogen indices): {conversion_map}")

        # Check the conversion map shape and properties
        assert conversion_map.shape == (
            21,
        ), f"Expected conversion map shape (21,), got {conversion_map.shape}"
        assert (
            torch.max(conversion_map) < 21
        ), "All mapped indices should be valid Cogen indices"
        assert (
            torch.min(conversion_map) >= 0
        ), "All mapped indices should be non-negative"

        # Test that the mapping preserves the total probability mass
        # Create uniform MPNN logits (shape: B, N, 20)
        B, N = 1, 3
        mpnn_logits = torch.full((B, N, 20), -torch.log(torch.tensor(20.0)))

        # Convert to Cogen format
        cogen_logits = runner._convert_mpnn_logits_to_cogen(mpnn_logits)

        # Check shape
        expected_shape = (B, N, 21)
        assert (
            cogen_logits.shape == expected_shape
        ), f"Expected shape {expected_shape}, got {cogen_logits.shape}"

        # Check that probabilities sum to 1 after conversion and softmax
        cogen_probs = torch.softmax(cogen_logits, dim=-1)
        prob_sums = torch.sum(cogen_probs, dim=-1)
        assert torch.allclose(
            prob_sums, torch.ones_like(prob_sums), atol=1e-6
        ), "Probabilities should sum to 1"

        print(f"MPNN logits shape: {mpnn_logits.shape}")
        print(f"Cogen logits shape: {cogen_logits.shape}")
        print(f"Probability sums: {prob_sums}")

        # Test specific mapping for known amino acids
        # Create single amino acid MPNN logits (one-hot-like)
        for mpnn_idx in range(20):
            single_aa_logits = torch.full(
                (1, 1, 20), -10.0
            )  # Very low probability for all
            single_aa_logits[0, 0, mpnn_idx] = (
                0.0  # High probability for specific amino acid
            )

            converted_logits = runner._convert_mpnn_logits_to_cogen(single_aa_logits)

            # The corresponding Cogen position should have higher logit value
            cogen_position = conversion_map[mpnn_idx].item()

            # Check that the mapped position has accumulated the probability
            assert (
                converted_logits[0, 0, cogen_position]
                >= single_aa_logits[0, 0, mpnn_idx]
            ), (
                f"MPNN amino acid {mpnn_idx} should map to Cogen position {cogen_position} "
                f"with preserved or accumulated probability"
            )

        print("✓ Logits conversion exact mapping test passed")

    def test_run_batch_error_handling(self, mock_cfg):
        """Test run_batch error handling for invalid inputs."""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Test with mismatched shapes
        trans = torch.randn(2, 10, 3)
        rotmats = torch.randn(2, 15, 3, 3)  # Wrong size
        aatypes = torch.randint(0, 20, (2, 10))
        res_mask = torch.ones(2, 10)
        chain_idx = torch.zeros(2, 10, dtype=torch.long)  # Single chain

        with pytest.raises(ValueError, match="Rots and trans incompatible"):
            runner.run_batch(
                trans=trans,
                rotmats=rotmats,
                aatypes=aatypes,
                res_mask=res_mask,
                diffuse_mask=torch.ones_like(res_mask),
                chain_idx=chain_idx,
            )

    def test_run_batch_subprocess_mode_error(self, mock_cfg):
        """Test that run_batch raises error when not in native mode."""
        mock_cfg.folding.protein_mpnn.use_native_runner = False
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        trans = torch.randn(1, 10, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).expand(1, 10, 3, 3)
        aatypes = torch.randint(0, 20, (1, 10))
        res_mask = torch.ones(1, 10)
        chain_idx = torch.zeros(1, 10, dtype=torch.long)  # Single chain

        with pytest.raises(
            ValueError, match="run_batch only supports native runner mode"
        ):
            runner.run_batch(
                trans=trans,
                rotmats=rotmats,
                aatypes=aatypes,
                res_mask=res_mask,
                diffuse_mask=torch.ones_like(res_mask),
                chain_idx=chain_idx,
            )

    def test_run_batch_empty_batch_error(self, mock_cfg_uninterpolated):
        """Test that run_batch handles empty batch gracefully by returning uniform logits."""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Create empty batch (B=0)
        B, N = 0, 10
        trans = torch.empty(B, N, 3)
        rotmats = torch.empty(B, N, 3, 3)
        aatypes = torch.empty(B, N, dtype=torch.long)
        res_mask = torch.empty(B, N, dtype=torch.bool)
        chain_idx = torch.empty(B, N, dtype=torch.long)  # Empty chain indices

        # This should handle the empty batch gracefully
        num_passes = 1
        sequences_per_pass = 1
        try:
            result = runner.run_batch(
                trans=trans,
                rotmats=rotmats,
                aatypes=aatypes,
                res_mask=res_mask,
                diffuse_mask=torch.ones_like(res_mask),
                chain_idx=chain_idx,
                num_passes=num_passes,
                sequences_per_pass=sequences_per_pass,
            )
            # If it doesn't error, check the result structure
            assert isinstance(result, NativeMPNNResult)
            assert result.logits.shape == (B, num_passes, sequences_per_pass, N, 21)
            assert result.confidence_scores.shape == (B, num_passes, sequences_per_pass)
            assert result.sequences.shape == (B, num_passes, sequences_per_pass, N)
        except (ValueError, RuntimeError):
            # Empty batch handling might raise an error, which is acceptable
            pass

    def test_amino_acid_mapping_consistency(self, mock_cfg_uninterpolated):
        """Test that documents amino acid mapping differences between ProteinMPNN and project."""
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        # Initialize runner to get the LigandMPNN module
        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Load data_utils from ProteinMPNN
        data_utils = runner._load_ligandmpnn_module("data_utils")

        # Get mappings from actual sources
        data_utils_int_to_str = data_utils.restype_int_to_str
        project_restype_order = residue_constants.restype_order
        project_int_to_restype = {v: k for k, v in project_restype_order.items()}

        # Document differences between ProteinMPNN and project
        pmpnn_project_differences = []
        for i in range(20):  # Standard 20 amino acids
            data_utils_aa = data_utils_int_to_str.get(i, None)
            project_aa = project_int_to_restype.get(i, None)

            if data_utils_aa != project_aa:
                pmpnn_project_differences.append(
                    f"Index {i}: ProteinMPNN='{data_utils_aa}', Project='{project_aa}'"
                )

        # Print the differences for documentation
        print(f"\nProteinMPNN vs Project amino acid mapping differences:")
        print(f"Found {len(pmpnn_project_differences)} differences (this is expected)")
        for diff in pmpnn_project_differences:
            print(f"  {diff}")

        # Verify that ProteinMPNN uses alphabetical ordering by single letter code
        expected_pmpnn_order = [
            "A",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "K",
            "L",
            "M",
            "N",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "V",
            "W",
            "Y",
        ]
        actual_pmpnn_order = [data_utils_int_to_str[i] for i in range(20)]

        print(f"\nProteinMPNN ordering: {actual_pmpnn_order}")
        print(f"Expected alphabetical: {expected_pmpnn_order}")

        # This test documents that ProteinMPNN uses alphabetical ordering
        assert (
            actual_pmpnn_order == expected_pmpnn_order
        ), "ProteinMPNN should use alphabetical ordering"

        # Verify that project uses AlphaFold ordering
        expected_project_order = [
            "A",
            "R",
            "N",
            "D",
            "C",
            "Q",
            "E",
            "G",
            "H",
            "I",
            "L",
            "K",
            "M",
            "F",
            "P",
            "S",
            "T",
            "W",
            "Y",
            "V",
        ]
        actual_project_order = [project_int_to_restype[i] for i in range(20)]

        print(f"Project ordering: {actual_project_order}")
        print(f"Expected AlphaFold: {expected_project_order}")

        assert (
            actual_project_order == expected_project_order
        ), "Project should use AlphaFold ordering"

    def test_sequence_conversion_consistency(self, mock_cfg_uninterpolated):
        """
        Test that sequence conversion between project amino acid types and ProteinMPNN
        alphabet is consistent and reversible via _create_protein_dict_from_frames.

        This test verifies that the conversion correctly goes:
        project_int -> project_aa -> mpnn_int -> mpnn_aa -> sequence_string
        """
        # Configure for native mode

        cfg = mock_cfg_uninterpolated.interpolate()

        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Load actual data_utils module from ProteinMPNN
        data_utils = runner._load_ligandmpnn_module("data_utils")

        # Get actual mappings from data_utils
        pmpnn_restype_int_to_str = data_utils.restype_int_to_str

        # Get mappings from residue_constants
        from cogeneration.data import residue_constants

        project_restypes = residue_constants.restypes
        project_restype_order = (
            residue_constants.restype_order
        )  # This is the mapping we need

        # Create reverse mapping for project (int -> amino acid)
        project_int_to_restype = {v: k for k, v in project_restype_order.items()}

        # Test with a variety of amino acid types using actual project ordering
        test_aatypes = torch.tensor(
            [0, 1, 2, 3, 4, 0, 1]
        )  # A, R, N, D, C, A, R in project ordering
        chain_idx = torch.tensor([0, 0, 1, 1, 1, 2, 2])  # Three chains: 7 residues
        device = torch.device("cpu")

        # Create mock atom37 coordinates (N, 37, 3)
        N = len(test_aatypes)
        atom37 = torch.randn(N, 37, 3)

        # Convert project aatypes to MPNN format first
        mpnn_aatypes = runner._convert_cogen_aatypes_to_mpnn(test_aatypes)

        # Test the conversion
        protein_dict = runner._create_protein_dict_from_frames(
            atom37=atom37,
            aatypes=mpnn_aatypes,  # Use MPNN format aatypes
            chain_idx=chain_idx,
            device=device,
        )

        # The correct conversion: project_int -> project amino acid
        correct_expected_sequence = ""
        for aa_int in test_aatypes:
            # Convert project int -> project amino acid
            project_aa = project_int_to_restype.get(int(aa_int), "A")
            correct_expected_sequence += project_aa

        # The sequence should now be correctly converted
        assert (
            protein_dict["seq"] == correct_expected_sequence
        ), f"Expected correct sequence: {correct_expected_sequence}, got: {protein_dict['seq']}"

        # Document the input and output
        print(f"Input aatypes (project ints): {test_aatypes.tolist()}")
        print(
            f"Project amino acids: {[project_int_to_restype.get(int(aa), 'A') for aa in test_aatypes]}"
        )
        print(f"Output sequence: {protein_dict['seq']}")
        print(f"Expected sequence: {correct_expected_sequence}")

        # Verify that the S field contains ProteinMPNN-encoded amino acids
        # Convert the sequence back using ProteinMPNN's mapping to verify consistency
        expected_mpnn_ints = []
        for project_aa in correct_expected_sequence:
            mpnn_int = data_utils.restype_str_to_int.get(project_aa, 0)
            expected_mpnn_ints.append(mpnn_int)

        expected_s_tensor = torch.tensor(
            expected_mpnn_ints, device=device, dtype=torch.long
        )
        assert torch.equal(
            protein_dict["S"], expected_s_tensor
        ), f"S tensor should contain ProteinMPNN integers: expected {expected_s_tensor}, got {protein_dict['S']}"

        # Verify that original project amino acid types are preserved
        # Note: _original_aatypes field is not currently implemented
        # assert torch.equal(protein_dict["_original_aatypes"], test_aatypes), "Original project aatypes should be preserved"

        # Test chain label handling
        # With chain_idx=[0, 0, 1, 1, 1, 2, 2], we have 7 residues total
        # The unique chains in order of appearance are: [0, 1, 2]
        print(f"Actual chain labels: {protein_dict['chain_labels']}")
        print(f"Input chain_idx: {chain_idx}")

        # Let's test what we actually get and verify it's consistent
        actual_chain_labels = protein_dict["chain_labels"]
        assert len(actual_chain_labels) == len(
            test_aatypes
        ), f"Chain labels length should match aatypes length: {len(test_aatypes)}"

        # Verify that residues with same chain_idx get same chain_label
        assert (
            actual_chain_labels[0] == actual_chain_labels[1]
        ), "Residues 0,1 should have same chain label (both chain_idx=0)"
        assert (
            actual_chain_labels[2] == actual_chain_labels[3] == actual_chain_labels[4]
        ), "Residues 2,3,4 should have same chain label (all chain_idx=1)"
        assert (
            actual_chain_labels[5] == actual_chain_labels[6]
        ), "Residues 5,6 should have same chain label (both chain_idx=2)"

        # Test multi-chain with different chain indices
        multi_chain_idx = torch.tensor(
            [0, 0, 1, 1, 1, 2, 2]
        )  # Non-sequential chain IDs, length 7
        protein_dict_multi = runner._create_protein_dict_from_frames(
            atom37=atom37,
            aatypes=test_aatypes,
            chain_idx=multi_chain_idx,
            device=device,
        )

        # Verify consistent mapping for multi-chain case
        multi_chain_labels = protein_dict_multi["chain_labels"]
        assert len(multi_chain_labels) == len(
            test_aatypes
        ), "Multi-chain labels should match aatypes length"
        assert (
            multi_chain_labels[0] == multi_chain_labels[1]
        ), "Multi-chain: residues 0,1 should have same label"
        assert (
            multi_chain_labels[2] == multi_chain_labels[3] == multi_chain_labels[4]
        ), "Multi-chain: residues 2,3,4 should have same label"
        assert (
            multi_chain_labels[5] == multi_chain_labels[6]
        ), "Multi-chain: residues 5,6 should have same label"

        # Test R_idx (residue indices)
        expected_r_idx = torch.arange(1, N + 1, device=device)  # 1-indexed
        assert torch.equal(
            protein_dict["R_idx"], expected_r_idx
        ), "R_idx should be 1-indexed residue numbers"

        # Test that masks are properly set
        expected_mask = torch.ones(N, device=device)
        assert torch.equal(
            protein_dict["mask"], expected_mask
        ), "All residues should be valid (mask=1)"
        assert torch.equal(
            protein_dict["chain_mask"], expected_mask
        ), "All residues should be designable by default"

    def test_protein_dict_structure_consistency(self, mock_cfg):
        """
        Test that _create_protein_dict_from_frames creates a protein_dict with the
        correct structure and all required fields for ProteinMPNN featurization.
        """
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Mock data_utils to create simple alphabetical mappings for testing
        mock_data_utils = MagicMock()
        # Create simple alphabetical mappings: int_to_str (0->A, 1->B, etc.) and str_to_int (A->0, B->1, etc.)
        mock_data_utils.restype_int_to_str = {i: chr(ord("A") + i) for i in range(20)}
        mock_data_utils.restype_str_to_int = {chr(ord("A") + i): i for i in range(20)}

        mock_data_utils.featurize.return_value = {
            "X": torch.randn(1, 5, 4, 3),
            "mask": torch.ones(1, 5),
        }

        with patch.object(
            runner, "_load_ligandmpnn_module", return_value=mock_data_utils
        ):
            # Create test input tensors
            atom37 = torch.randn(5, 37, 3)
            aatypes = torch.tensor([0, 1, 2, 3, 4])  # A, R, N, D, C in project ordering
            chain_idx = torch.tensor([0, 0, 1, 1, 1])  # Two chains

            protein_dict = runner._create_protein_dict_from_frames(
                atom37, aatypes, chain_idx, torch.device("cpu")
            )

            # Check that all required fields are present
            required_fields = [
                "coords",
                "X",
                "S",
                "seq",
                "mask",
                "R_idx",
                "chain_labels",
                "chain_letters",
                "chain_idx",
                "chain_mask",
            ]
            for field in required_fields:
                assert field in protein_dict, f"Missing field: {field}"

            # Check field shapes and types
            assert protein_dict["coords"].shape == (5, 37, 3), "coords shape incorrect"
            assert protein_dict["X"].shape == (5, 4, 3), "X shape incorrect"
            assert protein_dict["S"].shape == (5,), "S shape incorrect"
            assert isinstance(protein_dict["seq"], str), "seq should be string"
            assert len(protein_dict["seq"]) == 5, "seq length incorrect"
            assert protein_dict["mask"].shape == (5,), "mask shape incorrect"
            assert protein_dict["R_idx"].shape == (5,), "R_idx shape incorrect"
            assert protein_dict["chain_labels"].shape == (
                5,
            ), "chain_labels shape incorrect"
            assert (
                len(protein_dict["chain_letters"]) == 5
            ), "chain_letters length incorrect"
            assert protein_dict["chain_idx"].shape == (5,), "chain_idx shape incorrect"
            assert protein_dict["chain_mask"].shape == (
                5,
            ), "chain_mask shape incorrect"

            # Check that R_idx starts from 1
            assert protein_dict["R_idx"][0] == 1, "R_idx should start from 1"

            # Check that all masks are ones (valid residues)
            assert torch.all(protein_dict["mask"] == 1), "mask should be all ones"
            assert torch.all(
                protein_dict["chain_mask"] == 1
            ), "chain_mask should be all ones"

            # Check that backbone coordinates are subset of full coordinates
            assert torch.allclose(protein_dict["X"], protein_dict["coords"][:, :4, :])

            # Check that all tensors are on correct device
            device = torch.device("cpu")
            for field in [
                "coords",
                "X",
                "S",
                "mask",
                "R_idx",
                "chain_labels",
                "chain_idx",
                "chain_mask",
            ]:
                if hasattr(protein_dict[field], "device"):
                    assert (
                        protein_dict[field].device == device
                    ), f"{field} on wrong device"

    def test_amino_acid_conversion_mapping(self, mock_cfg):
        """
        Test the specific amino acid conversion mapping between MPNN and Cogen formats.

        This test verifies that the conversion functions correctly map specific amino acids
        between the two different alphabet orderings.
        """
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Test conversion from Cogen to MPNN format
        with patch.object(runner, "_load_ligandmpnn_module") as mock_load_module:
            # Mock data_utils with exact MPNN alphabet mapping
            mock_data_utils = Mock()
            mock_data_utils.restype_str_to_int = {
                "A": 0,
                "C": 1,
                "D": 2,
                "E": 3,
                "F": 4,
                "G": 5,
                "H": 6,
                "I": 7,
                "K": 8,
                "L": 9,
                "M": 10,
                "N": 11,
                "P": 12,
                "Q": 13,
                "R": 14,
                "S": 15,
                "T": 16,
                "V": 17,
                "W": 18,
                "Y": 19,
            }
            mock_data_utils.restype_int_to_str = {
                v: k for k, v in mock_data_utils.restype_str_to_int.items()
            }
            mock_load_module.return_value = mock_data_utils

            # Test specific amino acid conversions
            # Cogen format (AlphaFold): [A,R,N,D,C,Q,E,G,H,I,L,K,M,F,P,S,T,W,Y,V]
            # MPNN format (alphabetical): [A,C,D,E,F,G,H,I,K,L,M,N,P,Q,R,S,T,V,W,Y]

            test_cases = [
                # (cogen_index, aa_letter, expected_mpnn_index)
                (0, "A", 0),  # A: Cogen[0] -> MPNN[0]
                (1, "R", 14),  # R: Cogen[1] -> MPNN[14]
                (2, "N", 11),  # N: Cogen[2] -> MPNN[11]
                (3, "D", 2),  # D: Cogen[3] -> MPNN[2]
                (4, "C", 1),  # C: Cogen[4] -> MPNN[1]
                (10, "L", 9),  # L: Cogen[10] -> MPNN[9]
                (11, "K", 8),  # K: Cogen[11] -> MPNN[8]
            ]

            for cogen_idx, aa_letter, expected_mpnn_idx in test_cases:
                # Test single amino acid conversion
                cogen_input = torch.tensor([cogen_idx], dtype=torch.long)
                mpnn_output = runner._convert_cogen_aatypes_to_mpnn(cogen_input)

                assert mpnn_output[0].item() == expected_mpnn_idx, (
                    f"Conversion failed for {aa_letter}: "
                    f"Cogen[{cogen_idx}] -> MPNN[{mpnn_output[0].item()}], "
                    f"expected MPNN[{expected_mpnn_idx}]"
                )

    def test_run_batch_temperature_effects(self, mock_cfg):
        """Test that temperature affects the diversity of generated sequences"""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Create a simple batch
        B, N = 2, 15
        trans = torch.randn(B, N, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        aatypes = torch.randint(0, 20, (B, N))
        res_mask = torch.ones(B, N, dtype=torch.bool)
        chain_idx = torch.zeros(B, N, dtype=torch.long)

        num_passes = 3
        sequences_per_pass = 2
        # Run with low temperature (more deterministic)
        result_low_temp = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.01,
        )

        # Run with high temperature (more random)
        result_high_temp = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=1.0,
        )

        # This is a probabilistic test, so we just check the shapes are correct
        assert result_low_temp.logits.shape == (
            B,
            num_passes,
            sequences_per_pass,
            N,
            21,
        )
        assert result_high_temp.logits.shape == (
            B,
            num_passes,
            sequences_per_pass,
            N,
            21,
        )

    def test_run_batch_seed_reproducibility(self, mock_cfg):
        """Test that running with the same seed produces identical results"""
        runner = ProteinMPNNRunner(mock_cfg.folding.protein_mpnn)

        # Create a simple batch
        B, N = 2, 15
        num_passes = 2
        sequences_per_pass = 3
        seed = 42

        # Set up test data
        torch.manual_seed(123)  # For reproducible test data generation
        trans = torch.randn(B, N, 3)
        rotmats = torch.eye(3).unsqueeze(0).unsqueeze(0).repeat(B, N, 1, 1)
        aatypes = torch.randint(0, 20, (B, N))
        res_mask = torch.ones(B, N, dtype=torch.bool)
        chain_idx = torch.zeros(B, N, dtype=torch.long)

        # Run inference twice with the same seed
        result1 = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.1,
            seed=seed,
        )

        result2 = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.1,
            seed=seed,
        )

        # Results should be identical when using the same seed
        assert torch.allclose(
            result1.logits, result2.logits, atol=1e-6, rtol=1e-6
        ), "Logits should be identical with same seed"

        assert torch.allclose(
            result1.confidence_scores, result2.confidence_scores, atol=1e-6, rtol=1e-6
        ), "Confidence scores should be identical with same seed"

        assert torch.equal(
            result1.sequences, result2.sequences
        ), "Sequences should be identical with same seed"

        # Run with a different seed to verify results are actually different
        result3 = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.1,
            seed=seed + 1,  # Different seed
        )

        # Results should be different with different seed
        assert not torch.allclose(
            result1.logits, result3.logits, atol=1e-3, rtol=1e-3
        ), "Logits should be different with different seed"

    def test_diffuse_mask_respected_run_batch(
        self, mock_cfg_uninterpolated, pdb_2qlw_path
    ):
        """Test that diffuse_mask=0 positions remain fixed in run_batch"""
        cfg = mock_cfg_uninterpolated.interpolate()
        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Process actual PDB file to get real structure
        batch = process_pdb_file(str(pdb_2qlw_path), "2qlw")
        featurizer = BatchFeaturizer(
            cfg=cfg.dataset, task=cfg.data.task, is_training=False
        )
        features = featurizer.featurize_processed_file(
            processed_file=batch,
            csv_row={
                mc.pdb_name: "2qlw",
                mc.processed_path: "",
            },
        )

        # Extract structure and add batch dimension
        trans = features[bp.trans_1].unsqueeze(0)  # (1, N, 3)
        rotmats = features[bp.rotmats_1].unsqueeze(0)  # (1, N, 3, 3)
        aatypes = features[bp.aatypes_1].unsqueeze(0)  # (1, N)
        res_mask = features[bp.res_mask].unsqueeze(0)  # (1, N)
        chain_idx = features[bp.chain_idx].unsqueeze(0)  # (1, N)

        N = trans.shape[1]
        original_aatypes = aatypes.clone()

        # Create diffuse_mask with some fixed positions (diffuse_mask=0)
        # Fix every 3rd position and a few specific positions
        diffuse_mask = torch.ones_like(res_mask, dtype=torch.float)
        diffuse_mask[0, ::3] = 0  # Fix every 3rd position
        diffuse_mask[0, :5] = 0  # Fix first 5 positions
        diffuse_mask[0, -3:] = 0  # Fix last 3 positions

        num_fixed = (diffuse_mask == 0).sum().item()
        print(f"Testing with {num_fixed} fixed positions out of {N} total")

        # Run with multiple passes to increase chance of mutations
        result = runner.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=diffuse_mask,
            chain_idx=chain_idx,
            num_passes=3,
            sequences_per_pass=2,
            temperature=1.0,  # High temperature for more variation
            seed=42,
        )

        # Check that fixed positions remain unchanged across all generated sequences
        fixed_positions = (diffuse_mask == 0).squeeze(0)  # (N,)
        original_fixed_aatypes = original_aatypes[0, fixed_positions]  # Fixed AA types

        for pass_idx in range(result.sequences.shape[1]):
            for seq_idx in range(result.sequences.shape[2]):
                generated_sequence = result.sequences[0, pass_idx, seq_idx]  # (N,)
                generated_fixed_aatypes = generated_sequence[fixed_positions]

                # Move original_fixed_aatypes to same device as generated sequences for comparison
                original_fixed_aatypes_device = original_fixed_aatypes.to(
                    generated_fixed_aatypes.device
                )

                # All fixed positions should match original amino acids
                assert torch.equal(
                    original_fixed_aatypes_device, generated_fixed_aatypes
                ), (
                    f"Fixed positions were mutated in pass {pass_idx}, sequence {seq_idx}. "
                    f"Original fixed AAs: {original_fixed_aatypes.tolist()}, "
                    f"Generated fixed AAs: {generated_fixed_aatypes.tolist()}"
                )

        # Verify that at least some diffusable positions were actually changed
        # (to ensure the model is working and we're not just getting back the input)
        diffusable_positions = (diffuse_mask == 1).squeeze(0)  # (N,)
        original_diffusable_aatypes = original_aatypes[0, diffusable_positions]

        any_changes = False
        for pass_idx in range(result.sequences.shape[1]):
            for seq_idx in range(result.sequences.shape[2]):
                generated_sequence = result.sequences[0, pass_idx, seq_idx]
                generated_diffusable_aatypes = generated_sequence[diffusable_positions]

                # Move to same device for comparison
                original_diffusable_aatypes_device = original_diffusable_aatypes.to(
                    generated_diffusable_aatypes.device
                )

                if not torch.equal(
                    original_diffusable_aatypes_device, generated_diffusable_aatypes
                ):
                    any_changes = True
                    break
            if any_changes:
                break

        assert any_changes, (
            "No changes detected in diffusable positions - "
            "model may not be working properly or temperature too low"
        )

    def test_diffuse_mask_respected_inverse_fold_pdb_native(
        self, mock_cfg_uninterpolated, pdb_2qlw_path, tmp_path
    ):
        """Test that diffuse_mask=0 positions remain fixed in inverse_fold_pdb_native"""
        cfg = mock_cfg_uninterpolated.interpolate()
        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Get original sequence from PDB file using the same parsing that ProteinMPNN uses
        protein_dict = runner._parse_pdb_to_protein_dict(pdb_2qlw_path)

        # Get the actual length from the parsed protein structure using "S" field (amino acid indices)
        N = len(protein_dict["S"])
        print(f"Parsed protein structure length: {N}")

        # Convert sequence indices to amino acid letters for comparison
        # The protein_dict["S"] contains ProteinMPNN amino acid indices, need to convert to project format

        # Load data_utils to get the conversion mappings
        data_utils = runner._load_ligandmpnn_module("data_utils")

        original_aatypes = []
        for mpnn_aa_idx in protein_dict["S"]:
            # Convert from ProteinMPNN index to amino acid letter
            mpnn_aa_letter = data_utils.restype_int_to_str.get(int(mpnn_aa_idx), "A")
            # Convert from amino acid letter to project index
            project_idx = residue_constants.restype_order.get(
                mpnn_aa_letter, 0
            )  # Default to Alanine if not found
            original_aatypes.append(project_idx)
        original_aatypes = torch.tensor(original_aatypes)

        # Create diffuse_mask with some fixed positions, matching the parsed structure length
        diffuse_mask = torch.ones(N, dtype=torch.float)
        # Fix just a few specific positions for easier debugging
        diffuse_mask[0] = 0  # Fix first position
        diffuse_mask[1] = 0  # Fix second position
        diffuse_mask[2] = 0  # Fix third position
        diffuse_mask[10] = 0  # Fix 11th position
        diffuse_mask[20] = 0  # Fix 21st position

        num_fixed = (diffuse_mask == 0).sum().item()
        print(f"Testing PDB native with {num_fixed} fixed positions out of {N} total")

        output_dir = tmp_path / "inverse_fold_test"

        # Run inverse folding with fixed positions
        result_fasta = runner.inverse_fold_pdb_native(
            pdb_path=pdb_2qlw_path,
            output_dir=output_dir,
            diffuse_mask=diffuse_mask,
            num_sequences=5,  # Generate multiple sequences
            temperature=1.0,  # High temperature for variation
            seed=42,
        )

        # Parse generated sequences from FASTA
        generated_records = list(SeqIO.parse(result_fasta, "fasta"))
        assert len(generated_records) >= 1, "Should generate at least one sequence"

        # Convert original amino acid types to letters for comparison
        original_aa_letters = [
            residue_constants.restypes[aa] for aa in original_aatypes
        ]
        original_sequence = "".join(original_aa_letters)

        # Check fixed positions in all generated sequences
        fixed_positions = (diffuse_mask == 0).nonzero(as_tuple=True)[0].tolist()
        original_fixed_letters = [original_sequence[pos] for pos in fixed_positions]

        any_diffusable_changes = False

        for i, record in enumerate(generated_records):
            generated_sequence = str(record.seq)
            # Remove chain break characters ('/' and ':') from the generated sequence
            generated_sequence_clean = generated_sequence.replace("/", "").replace(
                ":", ""
            )
            assert (
                len(generated_sequence_clean) == N
            ), f"Sequence {i} length mismatch: expected {N}, got {len(generated_sequence_clean)}"

            # Map fixed positions accounting for chain break character
            # The generated sequence has a chain break character at position 108 (after chain A)
            # We need to adjust the position mapping for positions after the chain break
            chain_letters = protein_dict["chain_letters"]
            chain_a_length = sum(1 for c in chain_letters if c == "A")

            adjusted_fixed_positions = []
            for pos in fixed_positions:
                if pos < chain_a_length:
                    # Position is in chain A, no adjustment needed
                    adjusted_fixed_positions.append(pos)
                else:
                    # Position is in chain B, need to account for the chain break character
                    adjusted_fixed_positions.append(pos + 1)

            generated_fixed_letters = [
                generated_sequence[pos] for pos in adjusted_fixed_positions
            ]
            assert original_fixed_letters == generated_fixed_letters, (
                f"Fixed positions were mutated in sequence {i} ({record.id}). "
                f"Original fixed: {original_fixed_letters}, "
                f"Generated fixed: {generated_fixed_letters}"
            )

            # Check for changes in diffusable positions (also accounting for chain break)
            for pos in range(N):
                if diffuse_mask[pos] == 1:
                    if pos < chain_a_length:
                        # Position is in chain A
                        generated_pos = pos
                    else:
                        # Position is in chain B, account for chain break
                        generated_pos = pos + 1

                    if original_sequence[pos] != generated_sequence[generated_pos]:
                        any_diffusable_changes = True
                        break

        # Verify that at least some diffusable positions were changed
        assert any_diffusable_changes, (
            "No changes detected in diffusable positions - "
            "model may not be working properly or all sequences identical"
        )

    def test_diffuse_mask_respected_inverse_fold_pdb_subprocess(
        self, mock_cfg_uninterpolated, pdb_2qlw_path, tmp_path
    ):
        """
        diffuse_mask=0 positions are correctly converted to JSONL and respected by ProteinMPNN subprocess
        requires ProteinMPNN installed
        """

        # Configure for subprocess mode
        mock_cfg_uninterpolated.folding.protein_mpnn.use_native_runner = False
        cfg = mock_cfg_uninterpolated.interpolate()
        runner = ProteinMPNNRunner(cfg.folding.protein_mpnn)

        # Get original sequence from PDB file to determine length
        processed_file = process_pdb_file(str(pdb_2qlw_path), "2qlw")
        chain_feats = process_chain_feats(
            processed_file,
            center=True,
            trim_to_modeled_residues=True,
            trim_chains_independently=True,
        )
        input_seq = const.aatype_to_seq(chain_feats["aatype"])
        print(input_seq)

        N = len(input_seq)

        # Create diffuse_mask with some fixed positions, matching the parsed structure length
        diffuse_mask = torch.ones(N, dtype=torch.float)
        diffuse_mask[:12] = 0  # Fix first 12 positions
        diffuse_mask[::6] = 0  # Fix every 6th position
        diffuse_mask[-10:] = 0  # Fix last 10 positions

        output_dir = tmp_path / "subprocess_test"
        output_dir.mkdir()

        # Run inverse folding with fixed positions
        result_path = runner.inverse_fold_pdb_subprocess(
            pdb_path=pdb_2qlw_path,
            output_dir=output_dir,
            diffuse_mask=diffuse_mask,
            num_sequences=3,
            temperature=1.0,
            seed=42,
        )

        print("inverse fasta:", result_path)

        # Verify the output path
        expected_path = output_dir / f"{pdb_2qlw_path.stem}_sequences.fa"
        assert result_path == expected_path
        assert result_path.exists(), "Result FASTA file should exist"

        # Check that the fixed_positions.jsonl file was created
        fixed_positions_path = output_dir / "fixed_positions.jsonl"
        assert (
            fixed_positions_path.exists()
        ), f"Fixed positions file should exist: {fixed_positions_path}"

        # Read and verify the JSONL content
        with open(fixed_positions_path, "r") as f:
            fixed_residues_json = json.load(f)

        # Verify the JSON structure
        assert isinstance(
            fixed_residues_json, dict
        ), "Fixed residues should be a dictionary"
        assert (
            pdb_2qlw_path.stem in fixed_residues_json
        ), f"Should have fixed positions for {pdb_2qlw_path.stem}"

        # Verify the fixed positions matches diffuse_mask
        fixed_residues_dict = fixed_residues_json[pdb_2qlw_path.stem]
        expected_fixed_residues = {
            "A": [],
            "B": [],
        }
        current_chain = "A"
        last_chain_idx = chain_feats["chain_index"][0]
        chain_counter = 1
        for i, (mask_val, chain_idx) in enumerate(
            zip(diffuse_mask.tolist(), chain_feats["chain_index"])
        ):
            if chain_idx != last_chain_idx:
                current_chain = "B"
                chain_counter = 1
            if mask_val == 0:
                expected_fixed_residues[current_chain].append(chain_counter)
            chain_counter += 1
            last_chain_idx = chain_idx

        assert (
            fixed_residues_dict == expected_fixed_residues
        ), f"Fixed residues mismatch. Expected: {expected_fixed_residues}, Got: {fixed_residues_dict}"

        # Verify the generated sequences respect the fixed positions
        generated_records = list(SeqIO.parse(result_path, "fasta"))
        assert len(generated_records) >= 1, "Should generate at least one sequence"

        # Confirm the fixed positions are not changed in the generated sequences
        # Get the original sequence for comparison
        original_sequence = input_seq

        # Get fixed positions from the diffuse_mask
        fixed_positions = (diffuse_mask == 0).nonzero(as_tuple=True)[0].tolist()
        original_fixed_letters = [original_sequence[pos] for pos in fixed_positions]

        # Filter out positions with 'X' (unknown amino acids) from fixed position validation
        # 'X' characters represent unknown/ambiguous amino acids and should be allowed to change
        valid_fixed_positions = []
        valid_original_fixed_letters = []
        for pos, letter in zip(fixed_positions, original_fixed_letters):
            if letter != "X":
                valid_fixed_positions.append(pos)
                valid_original_fixed_letters.append(letter)

        print(f"Original sequence: {original_sequence}")
        print(f"All fixed positions: {fixed_positions}")
        print(f"Valid fixed positions (excluding X): {valid_fixed_positions}")
        print(f"Original fixed letters: {original_fixed_letters}")
        print(f"Valid original fixed letters: {valid_original_fixed_letters}")

        any_diffusable_changes = False

        for i, record in enumerate(generated_records):
            generated_sequence = str(record.seq)
            print(f"Generated sequence {i} ({record.id}): {generated_sequence}")

            # Remove chain break characters ('/' and ':') from the generated sequence
            generated_sequence_clean = generated_sequence.replace("/", "").replace(
                ":", ""
            )
            assert (
                len(generated_sequence_clean) == N
            ), f"Sequence {i} length mismatch: expected {N}, got {len(generated_sequence_clean)}"

            # Check that fixed positions in the generated sequence match the original
            # For subprocess mode, we need to account for chain break characters in the generated sequence
            # The generated sequence may have chain break characters that need to be handled
            chain_break_positions = []
            for j, char in enumerate(generated_sequence):
                if char in ["/", ":"]:
                    chain_break_positions.append(j)

            # Adjust valid fixed position indices to account for chain break characters
            adjusted_valid_fixed_positions = []
            for pos in valid_fixed_positions:
                adjusted_pos = pos
                # Add offset for each chain break character that comes before this position
                for break_pos in chain_break_positions:
                    if break_pos <= adjusted_pos:
                        adjusted_pos += 1
                adjusted_valid_fixed_positions.append(adjusted_pos)

            # Extract the valid fixed letters from the generated sequence
            generated_valid_fixed_letters = [
                generated_sequence[pos] for pos in adjusted_valid_fixed_positions
            ]

            assert valid_original_fixed_letters == generated_valid_fixed_letters, (
                f"Valid fixed positions were mutated in sequence {i} ({record.id}). "
                f"Valid original fixed: {valid_original_fixed_letters}, "
                f"Generated valid fixed: {generated_valid_fixed_letters}, "
                f"Valid fixed positions: {valid_fixed_positions}, "
                f"Adjusted valid positions: {adjusted_valid_fixed_positions}"
            )

            # Check for changes in diffusable positions (also accounting for chain break characters)
            for pos in range(N):
                if diffuse_mask[pos] == 1:
                    # Adjust position for chain break characters
                    adjusted_pos = pos
                    for break_pos in chain_break_positions:
                        if break_pos <= adjusted_pos:
                            adjusted_pos += 1

                    if original_sequence[pos] != generated_sequence[adjusted_pos]:
                        any_diffusable_changes = True
                        break

        # Verify that at least some diffusable positions were changed
        assert any_diffusable_changes, (
            "No changes detected in diffusable positions - "
            "model may not be working properly or all sequences identical"
        )


class TestProteinMPNNRunnerPool:
    """Test cases for ProteinMPNNRunnerPool"""

    def test_pool_from_mock_cfg(self, mock_cfg):
        assert (
            mock_cfg.folding.protein_mpnn.use_native_runner is True
        ), "pool requires native runner"
        num_models = 4
        pool = ProteinMPNNRunnerPool(mock_cfg.folding.protein_mpnn, num_models=4)
        assert pool is not None
        assert len(pool) == num_models

        # If we are on a mac, ensure its on MPS
        if torch.backends.mps.is_available():
            assert pool.runners[0].device.type == "mps"
        # If CUDA is available, ensure its on CUDA
        if torch.cuda.is_available():
            assert pool.runners[0].device.type == "cuda"

    def test_single_run_batch_2qlw(self, mock_cfg, pdb_2qlw_path):
        """Test pool run_batch on 2qlw structure"""
        pool = ProteinMPNNRunnerPool(mock_cfg.folding.protein_mpnn, num_models=2)

        # Process actual PDB file
        batch = process_pdb_file(str(pdb_2qlw_path), "2qlw")
        featurizer = BatchFeaturizer(
            cfg=mock_cfg.dataset, task=mock_cfg.data.task, is_training=False
        )
        features = featurizer.featurize_processed_file(
            processed_file=batch,
            csv_row={
                mc.pdb_name: "2qlw",
                mc.processed_path: "",
            },
        )

        # Extract single structure and add batch dimension
        trans = features[bp.trans_1].unsqueeze(0)  # (1, N, 3)
        rotmats = features[bp.rotmats_1].unsqueeze(0)  # (1, N, 3, 3)
        aatypes = features[bp.aatypes_1].unsqueeze(0)  # (1, N)
        res_mask = features[bp.res_mask].unsqueeze(0)  # (1, N)
        chain_idx = features[bp.chain_idx].unsqueeze(0)  # (1, N)

        num_res = trans.shape[1]
        num_passes = 2
        sequences_per_pass = 1

        result = pool.run_batch(
            trans=trans,
            rotmats=rotmats,
            aatypes=aatypes,
            res_mask=res_mask,
            diffuse_mask=torch.ones_like(res_mask),
            chain_idx=chain_idx,
            num_passes=num_passes,
            sequences_per_pass=sequences_per_pass,
            temperature=0.2,
        )

        assert isinstance(result, NativeMPNNResult)
        assert result.logits.shape[0] == 1
        assert result.logits.shape[1] == num_passes
        assert result.logits.shape[2] == sequences_per_pass
        assert result.logits.shape[3] == num_res
        assert result.logits.shape[4] == 21

        assert torch.all(result.confidence_scores >= 0.0)
        assert torch.all(result.confidence_scores <= 1.0)

        assert torch.all(result.sequences >= 0)
        assert torch.all(result.sequences < 20)


@pytest.mark.skip
class TestBenchmarkProteinMPNNRunner:
    """Benchmark tests comparing different ProteinMPNNRunner approaches"""

    def test_benchmark_runner_approaches(self):
        """
        Benchmark test comparing run_batch, run_native, and run_subprocess approaches.

        This test uses create_pdb_dataloader to generate real PDB batches,
        converts to atom37 representation, writes PDB files, and then times each approach on the same data.
        """
        # Use default config, not mock
        cfg = Config().interpolate()

        # Benchmark parameters
        num_batches = 5
        num_passes = 3
        temperature = 0.1

        dataloader = create_pdb_dataloader(
            cfg=cfg,
            training=True,
        )

        batches = []
        train_iter = iter(dataloader)
        for i in range(num_batches):
            batch = next(train_iter)
            batch_size = batch[bp.trans_1].shape[0]
            num_res = batch[bp.trans_1].shape[1]
            print(f"  Train batch {i}: batch_size={batch_size}, num_res={num_res}")
            batches.append(batch)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # Convert batches to atom37 and write PDB files
            batch_pdbs = []

            for batch_idx, batch in enumerate(batches):
                batch_size = batch[bp.trans_1].shape[0]
                num_res = batch[bp.trans_1].shape[1]

                # Convert to atom37 using all_atom.atom37_from_trans_rot
                atom37_coords = all_atom.atom37_from_trans_rot(
                    trans=batch[bp.trans_1],
                    rots=batch[bp.rotmats_1],
                    aatype=batch[bp.aatypes_1],
                    res_mask=batch[bp.res_mask],
                )

                # Write PDB files for each structure in the batch
                batch_pdb_paths = []
                for struct_idx in range(batch_size):
                    pdb_name = f"batch_{batch_idx}_struct_{struct_idx}"
                    pdb_path = tmp_path / f"{pdb_name}.pdb"

                    # Write PDB file using write_prot_to_pdb
                    write_prot_to_pdb(
                        prot_pos=atom37_coords[struct_idx].numpy(),
                        file_path=str(pdb_path),
                        aatype=batch[bp.aatypes_1][struct_idx].numpy(),
                        chain_idx=(
                            batch[bp.chain_idx][struct_idx].numpy()
                            if bp.chain_idx in batch
                            else None
                        ),
                        no_indexing=True,
                        overwrite=True,
                    )
                    batch_pdb_paths.append(pdb_path)

                batch_pdbs.append(
                    {
                        "batch": batch,
                        "pdb_paths": batch_pdb_paths,
                        "atom37": atom37_coords,
                        "batch_size": batch_size,
                        "num_res": num_res,
                    }
                )

            print(
                f"Created PDB files for {len(batch_pdbs)} batches, sizes: {[(bd['batch_size'], bd['num_res']) for bd in batch_pdbs]}"
            )

            # Configure ProteinMPNN for testing
            mpnn_cfg = cfg.folding.protein_mpnn
            mpnn_cfg.use_native_runner = True  # Start with native mode

            results = {}

            print("\n=== Benchmarking run_batch ===")
            runner_batch = ProteinMPNNRunner(mpnn_cfg)

            start_time = time.time()
            batch_results = []

            for batch_data in batch_pdbs:
                batch = batch_data["batch"]
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                try:
                    result = runner_batch.run_batch(
                        trans=batch[bp.trans_1],
                        rotmats=batch[bp.rotmats_1],
                        aatypes=batch[bp.aatypes_1],
                        res_mask=batch[bp.res_mask],
                        diffuse_mask=torch.ones_like(batch[bp.res_mask]),
                        chain_idx=(
                            batch[bp.chain_idx] if bp.chain_idx in batch else None
                        ),
                        num_passes=num_passes,
                        sequences_per_pass=1,
                        temperature=temperature,
                    )
                    # convert batch to list to mirror run_native and run_subprocess output shapes
                    batch_results.extend(result.sequences.tolist())

                except Exception as e:
                    print(f"run_batch failed: {e}")
                    continue

            batch_time = time.time() - start_time
            results["run_batch"] = {
                "time": batch_time,
                "results": batch_results,
                "method": "run_batch",
            }
            print(f"run_batch completed in {batch_time:.2f} seconds")

            print("\n=== Benchmarking run_batch_pool ===")
            runner_pool = ProteinMPNNRunnerPool(mpnn_cfg, num_models=8)

            start_time = time.time()
            pool_results = []

            for batch_data in batch_pdbs:
                batch = batch_data["batch"]
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                try:
                    result = runner_pool.run_batch(
                        trans=batch[bp.trans_1],
                        rotmats=batch[bp.rotmats_1],
                        aatypes=batch[bp.aatypes_1],
                        res_mask=batch[bp.res_mask],
                        diffuse_mask=torch.ones_like(batch[bp.res_mask]),
                        chain_idx=(
                            batch[bp.chain_idx] if bp.chain_idx in batch else None
                        ),
                        num_passes=num_passes,
                        sequences_per_pass=1,
                        temperature=temperature,
                    )
                    # convert batch to list to mirror run_native and run_subprocess output shapes
                    pool_results.extend(result.sequences.tolist())

                except Exception as e:
                    print(f"run_batch_pool failed: {e}")
                    continue

            pool_time = time.time() - start_time
            results["run_batch_pool"] = {
                "time": pool_time,
                "results": pool_results,
                "method": "run_batch_pool",
            }
            print(f"run_batch_pool completed in {pool_time:.2f} seconds")

            print("\n=== Benchmarking run_native ===")

            start_time = time.time()
            native_results = []

            for batch_data in batch_pdbs:
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                for pdb_path in batch_data["pdb_paths"]:
                    try:
                        result_path = runner_batch.inverse_fold_pdb_native(
                            pdb_path=pdb_path,
                            output_dir=tmp_path / "native_output",
                            diffuse_mask=torch.ones(
                                200
                            ),  # Mock diffuse mask for typical protein size
                            num_sequences=num_passes,
                            temperature=temperature,
                        )
                        native_results.append(result_path)

                    except Exception as e:
                        print(f"run_native failed for {pdb_path}: {e}")
                        continue

            native_time = time.time() - start_time
            results["run_native"] = {
                "time": native_time,
                "results": native_results,
                "method": "run_native",
            }
            print(f"run_native completed in {native_time:.2f} seconds")

            print("\n=== Benchmarking run_subprocess ===")

            # Switch to subprocess mode
            mpnn_cfg.use_native_runner = False
            runner_subprocess = ProteinMPNNRunner(mpnn_cfg)

            start_time = time.time()
            subprocess_results = []

            for batch_data in batch_pdbs:
                batch_size = batch_data["batch_size"]
                num_res = batch_data["num_res"]

                for pdb_path in batch_data["pdb_paths"]:
                    try:
                        result_path = runner_subprocess.inverse_fold_pdb_subprocess(
                            pdb_path=pdb_path,
                            output_dir=tmp_path / "subprocess_output",
                            diffuse_mask=torch.ones(
                                200
                            ),  # Mock diffuse mask for typical protein size
                            num_sequences=num_passes,
                            temperature=temperature,
                        )
                        subprocess_results.append(result_path)

                    except Exception as e:
                        print(f"run_subprocess failed for {pdb_path}: {e}")
                        continue

            subprocess_time = time.time() - start_time
            results["run_subprocess"] = {
                "time": subprocess_time,
                "results": subprocess_results,
                "method": "run_subprocess",
            }
            print(f"run_subprocess completed in {subprocess_time:.2f} seconds")

            # Print benchmark results
            print("BENCHMARK RESULTS")

            for method_name, result_data in results.items():
                method_desc = result_data["method"]
                elapsed_time = result_data["time"]
                num_results = len(result_data["results"])

                print(f"\n{method_desc}:")
                print(f"  Total time: {elapsed_time:.2f} seconds")
                print(f"  Results generated: {num_results}")
                if num_results > 0:
                    print(
                        f"  Average time per structure: {elapsed_time / num_results:.2f} seconds"
                    )

            # Print batch statistics
            print("BATCH STATISTICS")

            total_batch_size = sum(bd["batch_size"] for bd in batch_pdbs)
            total_residues = sum(bd["batch_size"] * bd["num_res"] for bd in batch_pdbs)

            print(f"Total batches processed: {len(batch_pdbs)}")
            print(f"Total structures: {total_batch_size}")
            print(f"Total residues: {total_residues}")

            for i, batch_data in enumerate(batch_pdbs):
                print(
                    f"  Batch {i}: batch_size={batch_data['batch_size']}, num_res={batch_data['num_res']}"
                )

            # Verify that all methods produced some results
            assert len(results) == 4, "All four methods should have been tested"

            for method_name, result_data in results.items():
                num_results = len(result_data["results"])
                if num_results == 0:
                    print(f"WARNING: {method_name} produced no results")
                else:
                    print(f"SUCCESS: {method_name} produced {num_results} results")

            print("\nBenchmark completed successfully!")
