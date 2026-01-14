import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from cogeneration.config.base import DatasetConfig, DatasetFilterConfig
from cogeneration.data.const import CHAIN_TO_INT
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.featurizer import BatchFeaturizer
from cogeneration.type.batch import BatchFeatures
from cogeneration.type.batch import BatchProp as bp
from cogeneration.type.dataset import DatasetColumn, DatasetCSVRow
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.dataset import MetadataDataFrame, ProcessedFile
from cogeneration.type.task import DataTask

# Spans are predefined for expedience (anarci hard to install on remote).
# To regenerate spans with ANARCI (IMGT), e.g.:
# anarci -i vhh.fasta -s imgt --csv | awk -F',' '$3 ~ /CDR/ {print}'
VHH_CDR_SPANS_BY_PDB: Dict[str, Dict[str, Tuple[Tuple[int, int], ...]]] = {
    "5VLV": {"A": ((27, 38), (56, 65), (105, 117))},
    "5U64": {"B": ((27, 38), (56, 65), (105, 117))},
    "5DXW": {"A": ((27, 38), (56, 65), (105, 117))},
    "5VNV": {"A": ((27, 38), (56, 65), (105, 117))},
    "9MDZ": {"E": ((27, 38), (56, 65), (105, 117))},
    # complexes
    "1ZVY": {"A": ((27, 38), (56, 65), (105, 117))},
    "4KRL": {"B": ((27, 38), (56, 65), (105, 117))},
    "5M2J": {"D": ((27, 38), (56, 65), (105, 117))},
    "5O2U": {
        "B": ((27, 38), (56, 65), (105, 117)),
        "D": ((27, 38), (56, 65), (105, 117)),
    },
    "8PII": {"A": ((27, 38), (56, 65), (105, 117))},
}


@dataclass
class VHHMotifMasker:
    _log: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    def compute_motif_mask(
        self,
        aatypes: torch.Tensor,  # (N,)
        chain_idx: torch.Tensor,  # (N,)
        res_mask: torch.Tensor,  # (N,)
        pdb_name: str,
    ) -> torch.Tensor:
        motif_mask = res_mask.clone().int()
        pdb_key = pdb_name.strip().upper()
        chain_spans = VHH_CDR_SPANS_BY_PDB.get(pdb_key)
        if not chain_spans:
            raise ValueError(f"{pdb_key} VHH CDR spans not configured")

        cdr_log_entries: List[str] = []
        total_len = int(res_mask.sum().item())

        for chain_id, spans in chain_spans.items():
            chain_idx_val = CHAIN_TO_INT.get(chain_id)
            if chain_idx_val is None:
                raise ValueError(
                    f"{pdb_key} VHH CDR spans invalid chain ID: {chain_id}"
                )

            chain_mask = (chain_idx == chain_idx_val) & (res_mask == 1)
            if not chain_mask.any():
                raise ValueError(f"{pdb_key} VHH chain {chain_id} not found in sample")

            positions = torch.nonzero(chain_mask, as_tuple=False).flatten()
            chain_len = int(positions.shape[0])
            applied_spans: List[Tuple[int, int]] = []

            for start, end in spans:
                start_idx = max(1, int(start))
                end_idx = min(int(end), chain_len)
                if start_idx > end_idx:
                    continue
                applied_spans.append((start_idx, end_idx))
                motif_mask[positions[start_idx - 1 : end_idx]] = 0

            if applied_spans:
                span_str = ",".join(
                    [f"{s}-{e}" if s != e else f"{s}" for s, e in applied_spans]
                )
                cdr_log_entries.append(
                    f"chain={chain_id} cdrs={span_str} len={chain_len}"
                )
            else:
                raise ValueError(
                    f"{pdb_key} VHH CDR spans empty for chain {chain_id} (len={chain_len})"
                )

        if cdr_log_entries:
            self._log.info(
                f"{pdb_key} VHH CDR spans total_len={total_len}; {' | '.join(cdr_log_entries)}"
            )

        return motif_mask


@dataclass
class VHHBatchFeaturizer(BatchFeaturizer):
    masker: VHHMotifMasker = field(default_factory=VHHMotifMasker)

    def get_motif_mask(self, feats: BatchFeatures) -> torch.Tensor:
        if self.task != DataTask.inpainting:
            raise ValueError("VHHDataset only supports inpainting tasks.")
        return self.masker.compute_motif_mask(
            aatypes=feats[bp.aatypes_1],
            chain_idx=feats[bp.chain_idx],
            res_mask=feats[bp.res_mask],
            pdb_name=feats[bp.pdb_name],
        )


class VHHDataset(BaseDataset):
    def __init__(
        self,
        cfg: DatasetConfig,
        task: DataTask,
        pdb_ids: Optional[Sequence[str]] = None,
    ):
        if task != DataTask.inpainting:
            raise ValueError("VHHDataset only supports inpainting tasks.")

        # Patch config
        cfg = cfg.clone()
        # Make config lenient, so all PDBs are present
        cfg.filter = DatasetFilterConfig.lenient()
        # some VHH structures are more recent, let them pass through
        cfg.test_date_cutoff = "2030-01-01"

        super().__init__(
            cfg=cfg,
            task=task,
            eval=False,  # avoid length subset filtering
            use_test=False,  # PDBs not in test set
        )
        self.featurizer = VHHBatchFeaturizer(cfg=cfg, task=task, eval=self.is_eval)

        self.pdb_ids = self.normalize_pdb_ids(
            pdb_ids if pdb_ids is not None else self.default_pdb_ids()
        )
        self.csv = self.filter_metadata_pdb_ids(
            self.csv, self.pdb_ids, logger=self._log
        )
        self.csv = self.csv.reset_index(drop=True)
        self.csv[DatasetColumn.index] = list(range(len(self.csv)))

        self._log.info(
            f"🐪 Using VHH Dataset, with {len(self.csv)} of {len(self.pdb_ids)} PDBs in dataset"
        )

    @staticmethod
    def default_pdb_ids() -> List[str]:
        return sorted(VHH_CDR_SPANS_BY_PDB.keys())

    @staticmethod
    def normalize_pdb_ids(pdb_ids: Sequence[str]) -> List[str]:
        return [pdb_id.strip().upper() for pdb_id in pdb_ids]

    @staticmethod
    def filter_metadata_pdb_ids(
        metadata: MetadataDataFrame,
        pdb_ids: Sequence[str],
        logger: Optional[logging.Logger] = None,
    ) -> MetadataDataFrame:
        if mc.pdb_name not in metadata.columns:
            raise ValueError("Metadata missing pdb_name column for VHH filtering.")

        normalized = {pdb_id.strip().upper() for pdb_id in pdb_ids}
        pdb_series = metadata[mc.pdb_name].astype(str).str.upper()
        filtered = metadata[pdb_series.isin(normalized)]

        if logger is not None:
            found = set(filtered[mc.pdb_name].astype(str).str.upper())
            missing = sorted(normalized - found)
            if missing:
                logger.warning(f"VHH dataset missing PDB IDs: {missing}")

        return filtered

    def featurize_processed_file(
        self, processed_file: ProcessedFile, csv_row: DatasetCSVRow
    ) -> BatchFeatures:
        return self.featurizer.featurize_processed_file(
            processed_file=processed_file,
            csv_row=csv_row,
        )
