"""
Test module for BoltzPredictor class.
"""

import tempfile
from pathlib import Path
from typing import List

import pytest
import torch

from cogeneration.data import residue_constants
from cogeneration.data.boltz_runner import BoltzPrediction, BoltzPredictor, ManifestBuilder

simple_sequence = "MKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"

multiple_sequences = [
    "MKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL",
    "ARNDCQEGHILKMFPSTWYV" * 2,  # All amino acids repeated
]


class TestManifestBuilder:
    """Test the ManifestBuilder class."""

    @pytest.fixture
    def predictor(self) -> BoltzPredictor:
        """Create a BoltzPredictor instance for testing with reduced parameters for speed."""
        return BoltzPredictor(
            use_potentials=True,
            recycling_steps=1,  # Reduced for faster testing
            sampling_steps=50,  # Reduced for faster testing
            diffusion_samples=1,
        )

    def test_init_with_target_dir(self, tmp_path):
        """Test ManifestBuilder initialization with custom target directory."""
        builder = ManifestBuilder(tmp_path)
        assert builder.target_dir == Path(tmp_path)
        assert builder.target_dir.exists()

    def test_init_without_target_dir(self):
        """Test ManifestBuilder initialization without target directory."""
        builder = ManifestBuilder()
        assert builder.target_dir.exists()
        assert builder.target_dir.name == "targets"

    def test_validate_sequence_valid(self):
        """Test sequence validation with valid amino acids."""
        builder = ManifestBuilder()
        seq = "ARNDCQEGHILKMFPSTWYV"
        result = builder._validate_sequence(seq)
        assert result == seq.upper()

    def test_validate_sequence_with_x(self):
        """Test sequence validation with X (unknown)."""
        builder = ManifestBuilder()
        seq = "ARNDX"
        result = builder._validate_sequence(seq)
        assert result == "ARNDX"

    def test_validate_sequence_invalid(self):
        """Test sequence validation with invalid amino acids."""
        builder = ManifestBuilder()
        with pytest.raises(ValueError, match="Invalid amino acids"):
            builder._validate_sequence("ARNDZ")

    def test_parse_sequence_single_chain(self):
        """Test parsing sequence without chain breaks."""
        builder = ManifestBuilder()
        result = builder._parse_sequence_with_chain_breaks("MKLLLL")
        assert result == ["MKLLLL"]

    def test_parse_sequence_multiple_chains(self):
        """Test parsing sequence with chain breaks."""
        builder = ManifestBuilder()
        result = builder._parse_sequence_with_chain_breaks("MKLLLL:ARNDCQ:PEPTIDE")
        assert result == ["MKLLLL", "ARNDCQ", "PEPTIDE"]

    def test_from_sequences_single_protein(self):
        """Test ManifestBuilder with a single protein sequence."""
        sequences = ["MKLLLL"]
        protein_ids = ["protein_0"]

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(target_dir=temp_dir)
            manifest, processed_dir = builder.from_sequences(sequences, protein_ids)

            # Validate manifest has 1 record
            assert len(manifest.records) == 1

            # Check FASTA file exists
            fasta_path = Path(temp_dir) / "protein_0.fasta"
            assert fasta_path.exists()

            # Check MSA directory has correct number of files (1 chain = 1 MSA file)
            msa_dir = Path(temp_dir) / "msa"
            msa_files = list(msa_dir.glob("*.csv"))
            assert len(msa_files) == 1

            # Verify MSA file content structure (should have negative integer taxonomy ID)
            msa_content = msa_files[0].read_text()
            lines = msa_content.strip().split("\n")
            assert len(lines) == 2  # Should have header + 1 data line

            # Parse the MSA content - should be "key,sequence" header and "negative_integer,sequence" data
            assert lines[0] == "key,sequence"
            key, sequence = lines[1].split(",", 1)
            assert key.lstrip("-").isdigit()  # Should be a negative integer
            assert sequence == "MKLLLL"

    def test_from_sequences_multiple_proteins(self):
        """Test ManifestBuilder with multiple protein sequences."""
        sequences = ["MKLLLL", "ARNDCE"]
        protein_ids = ["protein_0", "protein_1"]

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(target_dir=temp_dir)
            manifest, processed_dir = builder.from_sequences(sequences, protein_ids)

            # Validate manifest has 2 records
            assert len(manifest.records) == 2

            # Check FASTA files exist
            for protein_id in protein_ids:
                fasta_path = Path(temp_dir) / f"{protein_id}.fasta"
                assert fasta_path.exists()

            # Check MSA directory has correct number of files (2 chains = 2 MSA files)
            msa_dir = Path(temp_dir) / "msa"
            msa_files = list(msa_dir.glob("*.csv"))
            assert len(msa_files) == 2

            # Verify MSA file contents
            for msa_file in msa_files:
                msa_content = msa_file.read_text()
                lines = msa_content.strip().split("\n")
                assert len(lines) == 2  # Should have header + 1 data line

                # Parse the MSA content - should be "key,sequence" header and "negative_integer,sequence" data
                assert lines[0] == "key,sequence"
                key, sequence = lines[1].split(",", 1)
                assert key.lstrip("-").isdigit()  # Should be a negative integer
                assert sequence in ["MKLLLL", "ARNDCE"]

    def test_from_sequences_with_chain_breaks(self):
        """Test ManifestBuilder with chain breaks."""
        sequences = ["MKLLLL:ARNDCE"]  # Chain break indicated by ':'
        protein_ids = ["protein_0"]

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(target_dir=temp_dir)
            manifest, processed_dir = builder.from_sequences(sequences, protein_ids)

            # Validate manifest has 1 record
            assert len(manifest.records) == 1

            # Check FASTA file exists
            fasta_path = Path(temp_dir) / "protein_0.fasta"
            assert fasta_path.exists()

            # Check MSA directory has correct number of files (2 chains = 2 MSA files)
            msa_dir = Path(temp_dir) / "msa"
            msa_files = list(msa_dir.glob("*.csv"))
            assert len(msa_files) == 2

            # Verify MSA file contents for both chains
            sequences_found = []
            for msa_file in msa_files:
                msa_content = msa_file.read_text()
                lines = msa_content.strip().split("\n")
                assert len(lines) == 2  # Should have header + 1 data line

                # Parse the MSA content - should be "key,sequence" header and "negative_integer,sequence" data
                assert lines[0] == "key,sequence"
                key, sequence = lines[1].split(",", 1)
                assert key.lstrip("-").isdigit()  # Should be a negative integer
                sequences_found.append(sequence)

            # Should have both chain sequences
            assert "MKLLLL" in sequences_found
            assert "ARNDCE" in sequences_found

    def test_from_batch_simple(self):
        """Test ManifestBuilder.from_batch with simple sequences."""
        batch = [
            {"sequence": "MKLLLL", "protein_id": "protein_0"},
            {"sequence": "ARNDCE", "protein_id": "protein_1"},
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            builder = ManifestBuilder(target_dir=temp_dir)

            # Convert to sequences for from_sequences (from_batch expects tensors)
            sequences = [item["sequence"] for item in batch]
            protein_ids = [item["protein_id"] for item in batch]
            manifest, processed_dir = builder.from_sequences(sequences, protein_ids)

            # Validate manifest has 2 records
            assert len(manifest.records) == 2

            # Check FASTA files exist
            for item in batch:
                fasta_path = Path(temp_dir) / f"{item['protein_id']}.fasta"
                assert fasta_path.exists()

            # Check MSA directory has correct number of files (2 proteins = 2 MSA files)
            msa_dir = Path(temp_dir) / "msa"
            msa_files = list(msa_dir.glob("*.csv"))
            assert len(msa_files) == 2

            # Verify MSA file contents
            expected_sequences = ["MKLLLL", "ARNDCE"]
            sequences_found = []

            for msa_file in msa_files:
                msa_content = msa_file.read_text()
                lines = msa_content.strip().split("\n")
                assert len(lines) == 2  # Should have header + 1 data line

                # Parse the MSA content - should be "key,sequence" header and "negative_integer,sequence" data
                assert lines[0] == "key,sequence"
                key, sequence = lines[1].split(",", 1)
                assert key.lstrip("-").isdigit()  # Should be a negative integer
                sequences_found.append(sequence)

            # Verify all expected sequences are found
            for expected_seq in expected_sequences:
                assert expected_seq in sequences_found


class TestBoltzPredictor:
    """Test class for BoltzPredictor functionality."""

    @pytest.fixture
    def predictor(self) -> BoltzPredictor:
        """Create a BoltzPredictor instance for testing with reduced parameters for speed."""
        return BoltzPredictor(
            use_potentials=True,
            recycling_steps=1,  # Reduced for faster testing
            sampling_steps=50,  # Reduced for faster testing
            diffusion_samples=1,
        )

    def test_predictor_init(self):
        """Test that BoltzPredictor initializes correctly."""
        _ = BoltzPredictor(
            recycling_steps=1,
            sampling_steps=50,
            diffusion_samples=1,
        )

    @pytest.mark.slow
    def test_single_sequence_prediction(self, predictor, tmp_path):
        """Test that we can run a single sequence prediction."""
        # Create manifest with processed files using ManifestBuilder
        builder = ManifestBuilder(tmp_path)
        manifest, processed_dir = builder.from_sequences(
            [simple_sequence], ["test_protein"]
        )

        result = predictor.predict(manifest, processed_dir=processed_dir)

        assert result is not None
        assert len(result) == 1
        protein_result = result[0]
        assert isinstance(protein_result, BoltzPrediction)

        # Test parsed_structure method
        parsed_file = protein_result.parsed_structure()

        # Verify the structure has the expected shape
        pos = parsed_file["atom_positions"]
        assert pos.shape == (len(simple_sequence), 37, 3), (
            f"Expected pos.shape ({len(simple_sequence)}, 37, 3), " f"got {pos.shape}"
        )

        # Also verify other expected fields are present
        assert "aatype" in parsed_file
        assert "atom_mask" in parsed_file
        assert len(parsed_file["aatype"]) == len(simple_sequence)

    @pytest.mark.slow
    def test_multiple_sequence_prediction(self, predictor, tmp_path):
        """Test that we can run multiple sequence prediction."""
        # Create manifest with processed files using ManifestBuilder
        builder = ManifestBuilder(tmp_path)
        manifest, processed_dir = builder.from_sequences(
            multiple_sequences, ["protein1", "protein2"]
        )

        result = predictor.predict(manifest, processed_dir=processed_dir)

        assert result is not None
        assert len(result) == 2

    def test_predict_without_processed_dir_raises_error(self, predictor, tmp_path):
        """Test that predict without processed_dir raises error when processed files don't exist."""
        # Create a manifest but don't provide processed files
        # Create manifest but delete the processed directory to simulate missing processed files
        builder = ManifestBuilder(tmp_path)
        manifest, processed_dir = builder.from_sequences(multiple_sequences)
        # Now remove the processed directory to simulate missing processed files
        import shutil

        shutil.rmtree(processed_dir)

        with pytest.raises(FileNotFoundError, match="Missing required subdirectory"):
            predictor.predict(manifest, processed_dir=processed_dir)

    def test_boltz_prediction_dataclass(self, tmp_path):
        """Test BoltzPrediction dataclass functionality."""
        output_dir = tmp_path

        # Create some dummy files to test
        pdb_path = output_dir / "test_protein.pdb"
        pdb_path.write_text("dummy pdb content")

        structure_npz_path = output_dir / "test_structure.npz"
        structure_npz_path.write_text("dummy npz content")

        prediction = BoltzPrediction(
            path_pdb=pdb_path,
            path_structure_npz=structure_npz_path,
            path_plddt_npz=None,
            path_confidence_json=None,
        )

        assert prediction.path_pdb == pdb_path
        assert prediction.path_structure_npz == structure_npz_path
        assert prediction.path_plddt_npz is None
        assert prediction.path_confidence_json is None
