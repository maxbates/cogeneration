from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd

from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.data.protein import write_prot_to_pdb
from cogeneration.type.metrics import MetricName
from cogeneration.type.task import InferenceTask


def test_assess_sample_fold_only_does_not_inverse_fold(mock_cfg, tmp_path, monkeypatch):
    validator = FoldingValidator(cfg=mock_cfg.folding, device="cpu")

    # Avoid depending on mdtraj / geometry details here; this test is about control flow.
    monkeypatch.setattr(
        "cogeneration.data.folding_validation.calc_mdtraj_metrics",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "cogeneration.data.folding_validation.calc_ca_ca_metrics",
        lambda *args, **kwargs: {},
    )

    sample_length = 8
    pred_atom37 = np.random.randn(sample_length, 37, 3).astype(np.float32)
    pred_aa = np.random.randint(0, 20, size=(sample_length,), dtype=np.int64)
    chain_idx = np.zeros((sample_length,), dtype=np.int64)
    res_idx = np.arange(sample_length, dtype=np.int64)

    pred_pdb_path = tmp_path / "pred.pdb"
    write_prot_to_pdb(
        prot_pos=pred_atom37,
        file_path=str(pred_pdb_path),
        aatype=pred_aa,
        chain_idx=chain_idx,
        res_idx=res_idx,
        no_indexing=True,
        overwrite=True,
    )

    sample_dir = tmp_path / "sample"

    def mock_fold_fasta(*args, **kwargs) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            [
                {
                    MetricName.header: "sample",
                    MetricName.sequence: "A" * sample_length,
                    MetricName.folded_pdb_path: str(pred_pdb_path),
                    MetricName.plddt_mean: 50.0,
                }
            ]
        )

    def mock_assess_folded_structures(
        *,
        sample_pdb_path,
        pdb_name,
        folded_df,
        true_bb_positions,
        motif_mask,
        task,
    ) -> pd.DataFrame:
        df = folded_df.copy()
        df[MetricName.sample_pdb_path] = str(sample_pdb_path)
        df[MetricName.bb_rmsd_folded] = 0.0
        df[MetricName.is_designable] = True
        return df

    with patch.object(
        FoldingValidator, "inverse_fold_pdb", side_effect=AssertionError
    ) as _mock_inverse_fold, patch.object(
        FoldingValidator, "fold_fasta", side_effect=mock_fold_fasta
    ) as mock_fold, patch.object(
        FoldingValidator,
        "assess_folded_structures",
        side_effect=mock_assess_folded_structures,
    ):
        top_sample, saved = validator.assess_sample(
            task=InferenceTask.unconditional,
            sample_name="sample",
            sample_dir=str(sample_dir),
            pred_pdb_path=str(pred_pdb_path),
            pred_bb_positions=pred_atom37,
            pred_aa=pred_aa,
            sample_aa_traj=np.expand_dims(pred_aa, axis=0),
            diffuse_mask=np.ones_like(pred_aa, dtype=np.int8),
            motif_mask=None,
            chain_idx=chain_idx,
            res_idx=res_idx,
            true_bb_positions=None,
            true_aa=None,
            inverse_fold=False,
            also_fold_pmpnn_seq=False,
        )

    assert mock_fold.call_count == 1
    assert saved.inverse_folded_fasta is None
    assert saved.designability_df is None
    assert MetricName.bb_rmsd_folded in top_sample
