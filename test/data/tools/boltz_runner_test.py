"""
Test module for BoltzRunner class.
"""

import shutil
from pathlib import Path
from typing import List

import numpy as np
import pytest
import torch

from cogeneration.config.base import BoltzConfig
from cogeneration.data import residue_constants
from cogeneration.data.const import CHAIN_BREAK_STR, aatype_to_seq
from cogeneration.data.tools.boltz_runner import (
    BoltzManifestBuilder,
    BoltzPrediction,
    BoltzRunner,
)
from cogeneration.type.dataset import DatasetProteinColumn
from cogeneration.type.metrics import MetricName

simple_sequence = "MKYTGLP"

multiple_sequences = [
    "ARNDCQEGHI",
    "LKMFPSTWYV",
]


class TestManifestBuilder:
    """Test the ManifestBuilder class."""

    def test_init_with_outputs_dir(self, tmp_path, mock_cfg):
        """Test ManifestBuilder initialization with custom outputs directory."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        assert builder.paths.outputs_dir == tmp_path
        assert builder.paths.processed_dir.exists()

    def test_init_with_outputs_dir_creates_structure(self, tmp_path, mock_cfg):
        """Test ManifestBuilder initialization creates proper directory structure."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        assert builder.paths.processed_dir.exists()
        assert builder.paths.records_dir.exists()
        assert builder.paths.structures_dir.exists()

    def test_validate_sequence_valid(self, tmp_path, mock_cfg):
        """Test sequence validation with valid amino acids."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        seq = "ARNDCQEGHILKMFPSTWYV"
        result = builder._validate_sequence(seq)
        assert result == seq.upper()

    def test_validate_sequence_with_x(self, tmp_path, mock_cfg):
        """Test sequence validation with X (unknown)."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        seq = "ARNDX"
        result = builder._validate_sequence(seq)
        assert result == "ARNDX"

    def test_validate_sequence_invalid(self, tmp_path, mock_cfg):
        """Test sequence validation with invalid amino acids."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        with pytest.raises(ValueError, match="Invalid amino acids"):
            builder._validate_sequence("ARNDZ")

    def test_parse_sequence_single_chain(self, tmp_path, mock_cfg):
        """Test parsing sequence without chain breaks."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        result = builder._parse_sequence_with_chain_breaks("MKLLLL")
        assert result == ["MKLLLL"]

    def test_parse_sequence_multiple_chains(self, tmp_path, mock_cfg):
        """Test parsing sequence with chain breaks."""
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        result = builder._parse_sequence_with_chain_breaks("MKLLLL:ARNDCQ:PEPTIDE")
        assert result == ["MKLLLL", "ARNDCQ", "PEPTIDE"]

    def test_from_sequences_single_protein(self, tmp_path, mock_cfg):
        """Test ManifestBuilder with a single protein sequence."""
        sequences = ["MKLLLL"]
        protein_ids = ["protein_0"]

        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        manifest = builder.from_sequences(sequences, protein_ids)

        # Validate manifest has 1 record
        assert len(manifest.records) == 1

        # Check FASTA file exists
        fasta_path = builder.paths.outputs_dir / "protein_0.fasta"
        assert fasta_path.exists()

        # Check MSA directory has correct number of files (1 chain = 1 MSA file)
        msa_dir = builder.paths.msa_dir
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

    def test_from_sequences_multiple_proteins(self, tmp_path, mock_cfg):
        """Test ManifestBuilder with multiple protein sequences."""
        sequences = ["MKLLLL", "ARNDCE"]
        protein_ids = ["protein_0", "protein_1"]

        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        manifest = builder.from_sequences(sequences, protein_ids)

        # Validate manifest has 2 records
        assert len(manifest.records) == 2

        # Check FASTA files exist
        for protein_id in protein_ids:
            fasta_path = builder.paths.outputs_dir / f"{protein_id}.fasta"
            assert fasta_path.exists()

        # Check MSA directory has correct number of files (2 chains = 2 MSA files)
        msa_dir = builder.paths.msa_dir
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

    def test_from_sequences_with_chain_breaks(self, tmp_path, mock_cfg):
        """Test ManifestBuilder with chain breaks."""
        sequences = ["MKLLLL:ARNDCE"]  # Chain break indicated by ':'
        protein_ids = ["protein_0"]

        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        manifest = builder.from_sequences(sequences, protein_ids)

        # Validate manifest has 1 record
        assert len(manifest.records) == 1

        # Check FASTA file exists
        fasta_path = builder.paths.outputs_dir / "protein_0.fasta"
        assert fasta_path.exists()

        # Check MSA directory has correct number of files (2 chains = 2 MSA files)
        msa_dir = builder.paths.msa_dir
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

    def test_from_batch_simple(self, tmp_path, mock_cfg):
        """Test ManifestBuilder.from_batch with simple sequences."""
        batch = [
            {"sequence": "MKLLLL", "protein_id": "protein_0"},
            {"sequence": "ARNDCE", "protein_id": "protein_1"},
        ]

        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )

        # Convert to sequences for from_sequences (from_batch expects tensors)
        sequences = [item["sequence"] for item in batch]
        protein_ids = [item["protein_id"] for item in batch]
        manifest = builder.from_sequences(sequences, protein_ids)

        # Validate manifest has 2 records
        assert len(manifest.records) == 2

        # Check FASTA files exist
        for item in batch:
            fasta_path = builder.paths.outputs_dir / f"{item['protein_id']}.fasta"
            assert fasta_path.exists()

        # Check MSA directory has correct number of files (2 proteins = 2 MSA files)
        msa_dir = builder.paths.msa_dir
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


class TestBoltzRunner:
    """Test class for BoltzRunner functionality."""

    def test_runner_init(self, mock_cfg):
        """Test that BoltzRunner initializes correctly."""
        _ = BoltzRunner(cfg=mock_cfg.folding.boltz)

    @pytest.mark.slow
    def test_single_sequence_prediction(self, mock_cfg):
        """Test that we can run a single sequence prediction."""
        protein_id = "test_protein"

        predictor = BoltzRunner(cfg=mock_cfg.folding.boltz)
        result = predictor.fold_sequences([simple_sequence], [protein_id])

        assert result is not None
        assert len(result) == 1
        protein_result = result[0]
        assert isinstance(protein_result, BoltzPrediction)

        # confirm namespaced by protein_id
        assert protein_result.protein_id == protein_id
        assert protein_result.predictions_dir.name == protein_id

        # Confirm expected outputs exist
        assert protein_result.path_pdb.exists()
        assert protein_result.path_confidence_json.exists()
        assert protein_result.path_plddt_npz.exists()
        assert (
            protein_result.path_structure_npz is None
            or protein_result.path_structure_npz.exists()
        )

        folding_dir = protein_result.predictions_dir.parent.parent
        print(folding_dir)

        # sanity check
        assert (
            folding_dir / "predictions" / protein_id
        ) == protein_result.predictions_dir

        # Ensure other prediction files have been cleaned up
        assert len(list(protein_result.predictions_dir.glob("**/*"))) <= 4

        # processed and MSA directories should exist and contain only directories (no files)
        processed_dir = folding_dir / "processed"
        msa_dir = folding_dir / "msa"
        assert processed_dir.exists()
        assert msa_dir.exists()
        processed_files = [p for p in processed_dir.glob("**/*") if p.is_file()]
        msa_files = [p for p in msa_dir.glob("**/*") if p.is_file()]
        assert len(processed_files) == 0
        assert len(msa_files) == 0

        # Test parsed_structure method
        parsed_file = protein_result.parsed_structure()

        # verify structure shape
        pos = parsed_file[DatasetProteinColumn.atom_positions]
        assert pos.shape == (len(simple_sequence), 37, 3), (
            f"Expected pos.shape ({len(simple_sequence)}, 37, 3), " f"got {pos.shape}"
        )

        # verify sequence
        assert len(parsed_file[DatasetProteinColumn.aatype]) == len(simple_sequence)
        assert (
            aatype_to_seq(parsed_file[DatasetProteinColumn.aatype]) == simple_sequence
        )

    @pytest.mark.slow
    def test_multiple_sequence_prediction(self, mock_cfg):
        """Test that we can run multiple sequence prediction."""
        predictor = BoltzRunner(cfg=mock_cfg.folding.boltz)
        result = predictor.fold_sequences(multiple_sequences, ["protein1", "protein2"])

        assert result is not None
        assert len(result) == 2

        # check the sequences
        for i, protein_result in enumerate(result):
            assert protein_result.get_sequence() == multiple_sequences[i]

    @pytest.mark.slow
    def test_multiplicity_multiple_sequence_prediction(self, mock_cfg):
        """Test batched inference with multiple sequences."""
        mock_cfg.folding.boltz.diffusion_samples = 2  # multiplicity
        predictor = BoltzRunner(cfg=mock_cfg.folding.boltz)
        result = predictor.fold_sequences(multiple_sequences, ["protein1", "protein2"])

        assert result is not None
        assert len(result) == 2

        # check the sequences
        for i, protein_result in enumerate(result):
            assert protein_result.get_sequence() == multiple_sequences[i]

    @pytest.mark.slow
    def test_multimer_prediction(self, mock_cfg):
        """Test that we can run a multimer prediction."""
        protein_id = "test_multimer"
        multimer_sequence = "MKALGL" + CHAIN_BREAK_STR + "ATYPRKDA"

        predictor = BoltzRunner(cfg=mock_cfg.folding.boltz)
        result = predictor.fold_sequences([multimer_sequence], [protein_id])

        protein_result = result[0]
        assert isinstance(protein_result, BoltzPrediction)

        parsed_file = protein_result.parsed_structure()

        # Check that we have exactly 2 unique chain indices for the 2-chain multimer
        unique_chains = np.unique(parsed_file[DatasetProteinColumn.chain_index])
        assert (
            len(unique_chains) == 2
        ), f"Expected 2 chains, got {len(unique_chains)}: {unique_chains.tolist()}"

        parsed_seq = aatype_to_seq(parsed_file[DatasetProteinColumn.aatype])
        assert parsed_seq == multimer_sequence.replace(CHAIN_BREAK_STR, "")

    def test_predict_handles_missing_processed_files(self, mock_cfg, tmp_path):
        """Test that predict raises error when processed files don't exist in the BoltzPaths structure."""
        # Create manifest but delete the processed directory to simulate missing processed files
        builder = BoltzManifestBuilder(
            outputs_dir=tmp_path, cache_dir=mock_cfg.folding.boltz.cache_dir
        )
        manifest = builder.from_sequences(multiple_sequences, ["protein1", "protein2"])

        # Now remove the processed directory to simulate missing processed files
        shutil.rmtree(builder.paths.processed_dir)

        # Update the mock config to use the same tmp_path as outputs_path
        mock_cfg.folding.boltz.outputs_path = tmp_path
        predictor = BoltzRunner(cfg=mock_cfg.folding.boltz)
        with pytest.raises(FileNotFoundError, match="Missing record files"):
            predictor.predict(manifest, paths=builder.paths)

    @pytest.mark.slow
    def test_fold_fasta_subprocess(self, mock_cfg, tmp_path):
        """Test that fold_fasta_subprocess correctly handles multiple sequences including chain breaks."""
        # Create a temporary FASTA file with multiple sequences, including one with chain break
        fasta_path = tmp_path / "test_input.fasta"

        single_chain_seq = "MKLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL"
        multimer_seq = f"MKALGLTSRQWERTASDFGHIKLACVWNM{CHAIN_BREAK_STR}ATYPRKDAWERTYVIAPLKIHGFDSACVWNM"

        fasta_content = (
            ">single_chain\n"
            f"{single_chain_seq}\n"
            ">multimer_protein\n"
            f"{multimer_seq}\n"
        )
        fasta_path.write_text(fasta_content)

        # Create output directory
        output_dir = tmp_path / "subprocess_output"

        # Run fold_fasta_subprocess
        predictor = BoltzRunner(cfg=mock_cfg.folding.boltz)

        result_df = predictor.fold_fasta_subprocess(
            fasta_path=fasta_path, output_dir=output_dir
        )

        print(output_dir)
        print(result_df)

        # Validate the output DataFrame
        assert result_df is not None
        assert len(result_df) == 2

        # Check expected columns exist
        expected_columns = [
            MetricName.header,
            MetricName.sequence,
            MetricName.folded_pdb_path,
            MetricName.plddt_mean,
        ]
        for col in expected_columns:
            assert col in result_df.columns

        # Check the row data
        row = result_df.iloc[0]
        assert (
            row[MetricName.header] == "single_chain"
        )  # CLI uses filename as protein ID
        assert row[MetricName.sequence] == single_chain_seq
        assert row[MetricName.folded_pdb_path] is not None
        assert Path(row[MetricName.folded_pdb_path]).exists()
        assert row[MetricName.plddt_mean] is not None
        assert isinstance(row[MetricName.plddt_mean], (int, float))

        # Check the multimer row
        row = result_df.iloc[1]
        assert row[MetricName.header] == "multimer_protein"
        assert row[MetricName.sequence] == multimer_seq
        assert row[MetricName.folded_pdb_path] is not None
        assert Path(row[MetricName.folded_pdb_path]).exists()

        # Check 2 fasta files were created
        fasta_dir = output_dir / "fasta"
        fasta_files = list(fasta_dir.glob("*.fasta"))
        assert (
            len(fasta_files) == 2
        ), f"Expected 2 FASTA files, found {len(fasta_files)}"

        # Check that 3 MSA files were created (one single, 2 for multimer)
        msa_dir = output_dir / "msa"
        msa_files = list(msa_dir.glob("*.csv"))
        assert len(msa_files) == 3, f"Expected 2 MSA files, found {len(msa_files)}"


class TestBoltzPrediction:
    def test_boltz_prediction_dataclass(self, tmp_path):
        """Test BoltzPrediction dataclass functionality."""
        protein_id = "test_protein"
        output_dir = tmp_path / protein_id

        output_dir.mkdir(parents=True, exist_ok=True)

        # Create some dummy files to test
        pdb_path = output_dir / f"{protein_id}.pdb"
        pdb_path.write_text("dummy pdb content")

        structure_npz_path = output_dir / f"{protein_id}_structure.npz"
        structure_npz_path.write_text("dummy npz content")

        plddt_npz_path = output_dir / f"{protein_id}_plddt.npz"
        plddt_npz_path.write_text("dummy npz content")

        confidence_json_path = output_dir / f"{protein_id}_confidence.json"
        confidence_json_path.write_text("dummy json content")

        prediction = BoltzPrediction(
            protein_id=protein_id,
            predictions_dir=output_dir,
            path_pdb=pdb_path,
            path_structure_npz=structure_npz_path,
            path_plddt_npz=plddt_npz_path,
            path_confidence_json=confidence_json_path,
        )

        assert prediction.path_pdb == pdb_path
        assert prediction.path_structure_npz == structure_npz_path
        assert prediction.path_plddt_npz == plddt_npz_path
        assert prediction.path_confidence_json == confidence_json_path
