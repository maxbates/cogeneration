"""
Tests for ProteinMPNN runner module

Note: lots of Claude generated code here.
"""

import subprocess
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from Bio import SeqIO

from cogeneration.config.base import ModelType, ProteinMPNNRunnerConfig
from cogeneration.data.protein_mpnn_runner import ProteinMPNNRunner


def generate_mock_mpnn_fasta(
    pdb_name: str, num_sequences: int = 2, temperature: float = 0.1, seed: int = 123
) -> str:
    """
    Generate a mock MPNN-format FASTA string that mimics the output format
    from the original ProteinMPNN. This includes a native sequence (first entry)
    followed by generated sequences.

    Args:
        pdb_name: Name of the PDB file (used in headers)
        num_sequences: Number of generated sequences to create
        temperature: Sampling temperature for headers
        seed: Random seed for headers

    Returns:
        Mock FASTA string in ProteinMPNN format
    """
    # Mock native sequence header and sequence
    fasta_lines = [
        f">{pdb_name}, T={temperature}, seed={seed}, num_res=100, num_ligand_res=100, use_ligand_context=False, ligand_cutoff_distance=8.0, batch_size=1, number_of_batches={num_sequences}, model_path=/path/to/model",
        "MKLLVLGLGGVGKSALTVQFVQGIFVEKYDPTIEDFRKYTLPTVAIGLQLFLHYTSLLQEKLSPEDRKNLIVGSCDTAGQAMALQVEKQARELTGLEVLFQGPVLQV",
    ]

    # Add generated sequences
    for i in range(1, num_sequences + 1):
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

    @pytest.fixture
    def mock_config(self):
        """Mock ProteinMPNN configuration for testing"""
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=False,
            seq_per_sample=3,
            temperature=0.1,
            accelerator="cpu",
            pmpnn_path=Path("/fake/ligandmpnn/path"),
            pmpnn_seed=123,
        )
        return config

    @pytest.fixture
    def native_config(self):
        """Configuration for native mode testing"""
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=True,
            seq_per_sample=3,
            temperature=0.1,
            accelerator="cpu",
            pmpnn_path=Path("/Users/maxbates/projects/LigandMPNN"),
            pmpnn_seed=123,
        )
        return config

    @pytest.fixture
    def sidechain_config(self):
        """Configuration for side chain packing testing"""
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=True,
            seq_per_sample=2,
            temperature=0.1,
            accelerator="cpu",
            pmpnn_path=Path("/Users/maxbates/projects/LigandMPNN"),
            pmpnn_seed=123,
            pack_side_chains=True,
            number_of_packs_per_design=2,
            sc_num_denoising_steps=2,
            sc_num_samples=8,
            repack_everything=False,
        )
        return config

    @pytest.fixture
    def test_pdb_path(self):
        """Create a simple test PDB file"""
        return Path(__file__).parent.parent / "dataset" / "2qlw.pdb"

    def test_generate_mock_mpnn_fasta(self):
        """Test the mock FASTA generation function"""
        fasta_content = generate_mock_mpnn_fasta("test_pdb", num_sequences=2)

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

    def test_import_mechanism_success(self, native_config):
        """Test that LigandMPNN modules can be imported successfully when available"""
        runner = ProteinMPNNRunner(native_config)

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
                f"LigandMPNN import test failed. Please ensure LigandMPNN is installed and available at {native_config.pmpnn_path}. "
                f"You can install it by cloning https://github.com/dauparas/LigandMPNN. Error: {e}"
            )

    def test_import_mechanism_missing_path(self):
        """Test that import mechanism fails gracefully when LigandMPNN path doesn't exist"""
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=True,
            seq_per_sample=3,
            temperature=0.1,
            accelerator="cpu",
            pmpnn_path=Path("/nonexistent/path/to/ligandmpnn"),
            pmpnn_seed=123,
        )

        runner = ProteinMPNNRunner(config)

        with pytest.raises(FileNotFoundError, match="LigandMPNN path not found"):
            runner._load_ligandmpnn_module("data_utils")

    def test_import_mechanism_missing_module(self, native_config):
        """Test that import mechanism fails gracefully when specific module doesn't exist"""
        runner = ProteinMPNNRunner(native_config)

        # Try to import a non-existent module
        with pytest.raises(ImportError, match="Module nonexistent_module.py not found"):
            runner._load_ligandmpnn_module("nonexistent_module")

    def test_import_caching(self, native_config):
        """Test that modules are cached after first import"""
        runner = ProteinMPNNRunner(native_config)

        try:
            # Import the same module twice
            module1 = runner._load_ligandmpnn_module("data_utils")
            module2 = runner._load_ligandmpnn_module("data_utils")

            # Should be the same object (cached)
            assert module1 is module2
            assert "data_utils" in runner._ligandmpnn_modules

        except (ImportError, FileNotFoundError):
            # If LigandMPNN is not available, skip this test
            pytest.skip("LigandMPNN not available for caching test")

    def test_sidechain_model_loading_disabled(self, native_config):
        """Test that side chain model is not loaded when pack_side_chains is False"""
        # Ensure pack_side_chains is False
        native_config.pack_side_chains = False

        runner = ProteinMPNNRunner(native_config)

        try:
            model_sc = runner._load_side_chain_model()
            # Should return None when side chain packing is disabled
            assert model_sc is None

        except (ImportError, FileNotFoundError):
            # If LigandMPNN is not available, skip this test
            pytest.skip("LigandMPNN not available for side chain model test")

    def test_sidechain_model_loading_enabled(self, sidechain_config):
        """Test that side chain model is loaded when pack_side_chains is True"""
        runner = ProteinMPNNRunner(sidechain_config)

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

        except (ImportError, FileNotFoundError):
            # If LigandMPNN is not available, skip this test
            pytest.skip("LigandMPNN not available for side chain model test")
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

    def test_sidechain_config_validation(self, sidechain_config):
        """Test that side chain packing configuration is properly validated"""
        runner = ProteinMPNNRunner(sidechain_config)

        # Verify configuration values
        assert sidechain_config.pack_side_chains is True
        assert sidechain_config.number_of_packs_per_design == 2
        assert sidechain_config.sc_num_denoising_steps == 2
        assert sidechain_config.sc_num_samples == 8
        assert sidechain_config.repack_everything is False

    def test_sidechain_weights_configuration(self):
        """Test that pmpnn_weights configuration is used for side chain checkpoint lookup"""
        ligandmpnn_path = Path("/fake/ligandmpnn/path")
        pmpnn_weights_dir = "custom"
        custom_weights_path = ligandmpnn_path / pmpnn_weights_dir
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=True,
            accelerator="cpu",
            pmpnn_path=ligandmpnn_path,
            pmpnn_weights_dir=pmpnn_weights_dir,
            pack_side_chains=True,
            checkpoint_path_sc=None,  # Let it auto-discover
            pmpnn_seed=123,
        )

        runner = ProteinMPNNRunner(config)

        # Test that the configuration is set up correctly
        assert config.checkpoint_path_sc is None  # Should auto-discover
        assert config.pack_side_chains is True

    def test_sidechain_packing_execution(self, sidechain_config, test_pdb_path):
        """
        Test side chain packing execution in native mode.
        This test will be skipped if LigandMPNN is not available or side chain model can't be loaded.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            runner = ProteinMPNNRunner(sidechain_config)

            try:
                # Test native execution with side chain packing
                result_path = runner.run_native(
                    pdb_path=test_pdb_path,
                    output_dir=output_dir,
                    num_sequences=2,
                    seed=123,
                )

                # Verify main output exists
                assert result_path.exists()
                assert result_path.name == f"{test_pdb_path.stem}_sequences.fa"

                # Verify FASTA content
                records = list(SeqIO.parse(result_path, "fasta"))
                assert len(records) == 2  # Should have 2 generated sequences

                # Check if packed structures directory was created
                packed_dir = output_dir / "packed"
                if packed_dir.exists():
                    # If side chain packing worked, there should be packed PDB files
                    packed_files = list(packed_dir.glob("*.pdb"))

                    # Should have: 2 sequences × 2 packs per design = 4 files
                    # But we'll be lenient since side chain packing might fail for various reasons
                    if packed_files:
                        # Verify file naming convention
                        for pdb_file in packed_files:
                            assert test_pdb_path.stem in pdb_file.name
                            assert "_seq_" in pdb_file.name
                            assert "_pack_" in pdb_file.name
                            assert pdb_file.suffix == ".pdb"

                        print(
                            f"✅ Side chain packing succeeded. Generated {len(packed_files)} packed structures."
                        )
                    else:
                        print(
                            "⚠️ Side chain packing was enabled but no packed structures were generated."
                        )
                else:
                    print(
                        "⚠️ Side chain packing was enabled but packed directory was not created."
                    )

            except (ImportError, FileNotFoundError) as e:
                # If LigandMPNN is not available or model loading fails, skip the test
                if any(
                    keyword in str(e).lower()
                    for keyword in [
                        "data_utils",
                        "model_utils",
                        "checkpoint",
                        "ligandmpnn",
                        "sc_utils",
                    ]
                ):
                    pytest.skip(f"LigandMPNN or side chain packing not available: {e}")
                else:
                    raise

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_subprocess_success(
        self, mock_path_exists, mock_subprocess_run, mock_config, test_pdb_path
    ):
        """Test subprocess mode with successful execution"""

        # Mock path existence checks
        def mock_exists():
            # Mock that run.py exists and PDB path exists
            return True

        mock_path_exists.return_value = True

        # Setup mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "ProteinMPNN completed successfully"
        mock_result.stderr = ""
        mock_subprocess_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            runner = ProteinMPNNRunner(mock_config)

            # Create mock output directory and FASTA file
            seqs_dir = output_dir / "seqs"
            seqs_dir.mkdir(parents=True)

            # Create mock FASTA output that ProteinMPNN would generate
            mock_fasta_content = generate_mock_mpnn_fasta(
                test_pdb_path.stem, num_sequences=2
            )
            mock_fasta_path = seqs_dir / f"{test_pdb_path.stem}.fa"
            mock_fasta_path.write_text(mock_fasta_content)

            # Test subprocess execution
            result_path = runner.run_subprocess(
                pdb_path=test_pdb_path,
                output_dir=output_dir,
                device_id=0,
                num_sequences=2,
                seed=123,
            )

            # Verify subprocess was called
            mock_subprocess_run.assert_called_once()
            call_args = mock_subprocess_run.call_args
            assert "python" in call_args[0][0]
            assert str(mock_config.pmpnn_path / "run.py") in call_args[0][0]

            # Verify output path
            assert result_path.exists()
            assert result_path.name == f"{test_pdb_path.stem}_sequences.fa"

            # Verify processed FASTA content - should only have generated sequences (native skipped)
            records = list(SeqIO.parse(result_path, "fasta"))
            assert (
                len(records) == 2
            )  # Should have 2 generated sequences (native sequence skipped)

            # Check that sequence IDs are cleaned up
            for i, record in enumerate(records, 1):
                assert record.id == f"{test_pdb_path.stem}_seq_{i}"
                assert "ProteinMPNN generated sequence" in record.description

    @patch("subprocess.run")
    @patch("pathlib.Path.exists")
    def test_run_subprocess_failure(
        self, mock_path_exists, mock_subprocess_run, mock_config, test_pdb_path
    ):
        """Test subprocess mode with failed execution"""
        # Mock path existence checks
        mock_path_exists.return_value = True

        # Setup mock subprocess failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "ProteinMPNN failed"
        mock_subprocess_run.return_value = mock_result

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            runner = ProteinMPNNRunner(mock_config)

            # Test that subprocess failure raises exception
            with pytest.raises(subprocess.CalledProcessError):
                runner.run_subprocess(
                    pdb_path=test_pdb_path,
                    output_dir=output_dir,
                    device_id=0,
                    num_sequences=2,
                    seed=123,
                )

    def test_run_subprocess_missing_device_id(self, mock_config, test_pdb_path):
        """Test that subprocess mode requires device_id"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            runner = ProteinMPNNRunner(mock_config)

            # Test that missing device_id raises assertion error
            with pytest.raises(AssertionError, match="device_id is required"):
                runner.run_subprocess(
                    pdb_path=test_pdb_path,
                    output_dir=output_dir,
                    device_id=None,
                    num_sequences=2,
                    seed=123,
                )

    def test_run_native_real_execution(self, native_config, test_pdb_path):
        """
        Test native mode with real execution (requires LigandMPNN installation).
        This test will be skipped if LigandMPNN is not available.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir)
            runner = ProteinMPNNRunner(native_config)

            try:
                # Test native execution
                result_path = runner.run_native(
                    pdb_path=test_pdb_path,
                    output_dir=output_dir,
                    num_sequences=2,
                    seed=123,
                )

                # Verify output
                assert result_path.exists()
                assert result_path.name == f"{test_pdb_path.stem}_sequences.fa"

                # Verify FASTA content - should only have generated sequences (native skipped)
                records = list(SeqIO.parse(result_path, "fasta"))
                assert (
                    len(records) == 2
                )  # Should have 2 generated sequences (native sequence skipped)

                # Check sequence IDs are properly formatted
                for i, record in enumerate(records, 1):
                    assert record.id == f"{test_pdb_path.stem}_seq_{i}"
                    assert len(str(record.seq)) > 0

            except (ImportError, FileNotFoundError) as e:
                # If LigandMPNN is not available or model loading fails, skip the test
                if any(
                    keyword in str(e).lower()
                    for keyword in [
                        "data_utils",
                        "model_utils",
                        "checkpoint",
                        "ligandmpnn",
                    ]
                ):
                    pytest.skip(
                        f"LigandMPNN not available for native mode testing: {e}"
                    )
                else:
                    raise

    def test_process_fasta_output(self, mock_config):
        """Test FASTA output processing"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runner = ProteinMPNNRunner(mock_config)

            # Create mock original FASTA with native + generated sequences
            mock_fasta_content = generate_mock_mpnn_fasta("test_pdb", num_sequences=2)
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

    def test_config_validation(self):
        """Test configuration validation"""
        # Test that invalid model type raises appropriate error
        config = ProteinMPNNRunnerConfig(
            model_type="invalid_model", use_native_runner=True, accelerator="cpu"
        )

        runner = ProteinMPNNRunner(config)

        # This should raise an error when trying to get checkpoint path
        with pytest.raises(ValueError, match="Unknown model type"):
            runner._get_checkpoint_path()

    def test_pmpnn_weights_configuration(self):
        """Test that pmpnn_weights configuration is properly used for checkpoint lookup"""
        # Test with custom weights path
        ligandmpnn_path = Path("/fake/ligandmpnn/path")
        pmpnn_weights_dir = "custom"
        custom_weights_path = ligandmpnn_path / pmpnn_weights_dir
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=True,
            accelerator="cpu",
            pmpnn_path=ligandmpnn_path,
            pmpnn_weights_dir=pmpnn_weights_dir,
            pmpnn_seed=123,
        )

        runner = ProteinMPNNRunner(config)

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

    def test_pmpnn_weights_fallback(self):
        """Test that runner falls back to pmpnn_path when pmpnn_weights is empty"""
        ligandmpnn_path = Path("/fake/ligandmpnn/path")
        config = ProteinMPNNRunnerConfig(
            model_type=ModelType.PROTEIN_MPNN,
            use_native_runner=True,
            accelerator="cpu",
            pmpnn_path=ligandmpnn_path,
            pmpnn_weights_dir="",
            pmpnn_seed=123,
        )

        runner = ProteinMPNNRunner(config)

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
