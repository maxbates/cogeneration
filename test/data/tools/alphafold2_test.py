"""
Test module for AlphaFold2Tool class.
"""

from pathlib import Path

import pytest

from cogeneration.data.const import CHAIN_BREAK_STR
from cogeneration.data.tools.alphafold2 import AlphaFold2Tool
from cogeneration.type.metrics import MetricName


class TestAlphaFold2Tool:
    """Test the AlphaFold2Tool class."""

    @pytest.mark.slow
    def test_fold_fasta(self, mock_cfg, tmp_path):
        """Test that fold_fasta correctly handles multiple sequences including chain breaks."""
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
        output_dir = tmp_path / "alphafold2_output"

        # Run fold_fasta
        tool = AlphaFold2Tool(cfg=mock_cfg.folding.alphafold)
        tool.set_device_id(0)

        result_df = tool.fold_fasta(fasta_path=fasta_path, output_dir=output_dir)

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

        # Check the single chain row data
        single_chain_row = result_df[
            result_df[MetricName.header] == "single_chain"
        ].iloc[0]
        assert single_chain_row[MetricName.header] == "single_chain"
        assert single_chain_row[MetricName.sequence] == single_chain_seq
        assert single_chain_row[MetricName.folded_pdb_path] is not None
        assert Path(single_chain_row[MetricName.folded_pdb_path]).exists()
        assert single_chain_row[MetricName.plddt_mean] is not None
        assert isinstance(single_chain_row[MetricName.plddt_mean], (int, float))

        # Check the multimer row
        multimer_row = result_df[
            result_df[MetricName.header] == "multimer_protein"
        ].iloc[0]
        assert multimer_row[MetricName.header] == "multimer_protein"
        assert multimer_row[MetricName.sequence] == multimer_seq
        assert multimer_row[MetricName.folded_pdb_path] is not None
        assert Path(multimer_row[MetricName.folded_pdb_path]).exists()
        assert multimer_row[MetricName.plddt_mean] is not None
        assert isinstance(multimer_row[MetricName.plddt_mean], (int, float))
