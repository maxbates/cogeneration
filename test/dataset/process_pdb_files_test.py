import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cogeneration.config.base import DatasetConfig, DatasetFilterConfig
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.process_pdb import read_processed_file
from cogeneration.scripts.process_pdb_files import process_pdb_with_metadata
from cogeneration.type.dataset import DatasetColumns as dc
from cogeneration.type.dataset import DatasetProteinColumns
from cogeneration.type.dataset import DatasetProteinColumns as dpc
from cogeneration.type.task import DataTaskEnum

# https://www2.rcsb.org/structure/2QLW
example_pdb_path = Path(__file__).parent / "2qlw.pdb"


class TestProcessPDBFiles:
    def test_process_file(self, tmp_path):
        metadata, _ = process_pdb_with_metadata(
            pdb_file_path=str(example_pdb_path.absolute()),
            write_dir=str(tmp_path),
        )

        # check metadata
        assert metadata is not None
        assert metadata[dc.seq_len] == 454
        assert metadata[dc.modeled_seq_len] == 335
        assert metadata[dc.moduled_num_res] == 200
        assert metadata[dc.num_chains] == 2
        assert metadata[dc.oligomeric_detail] == "dimeric"
        assert metadata[dc.quaternary_category] == "homomer"
        assert metadata[dc.helix_percent] > 0.1

        # check written file
        written_file = metadata[dc.processed_path]
        with open(written_file, "rb") as f:
            pkl = pickle.load(f)

        # modeled sequence includes non-AA residues (i.e. 20 = unknown)
        assert len(pkl[dpc.aatype]) == metadata[dc.seq_len]
        # check number actual residues in seq
        assert np.sum(pkl[dpc.aatype] != 20) == metadata[dc.moduled_num_res]
        # modeled positions only for valid residues
        assert len(pkl[dpc.modeled_idx]) == metadata[dc.moduled_num_res]

        # check all expected keys are present
        expected_keys = [key for key in DatasetProteinColumns]
        observed_keys = list(pkl.keys())
        extra_keys = set(observed_keys) - set(expected_keys)
        missing_keys = set(expected_keys) - set(observed_keys)
        assert (
            len(extra_keys) == 0 and len(missing_keys) == 0
        ), f"Extra keys: {extra_keys}, Missing keys: {missing_keys}"

    def test_parsing_processed_file(self, tmp_path):
        metadata, _ = process_pdb_with_metadata(
            pdb_file_path=str(example_pdb_path.absolute()),
            write_dir=str(tmp_path),
        )
        # Test can load and process
        _ = read_processed_file(processed_file_path=metadata[dc.processed_path])

    def test_dataset_using_processed_file(self, tmp_path):
        metadata, _ = process_pdb_with_metadata(
            pdb_file_path=str(example_pdb_path.absolute()),
            write_dir=str(tmp_path),
        )

        # write a dummy dataset csv
        csv_path = tmp_path / "pdb_metadata.csv"
        df = pd.DataFrame([metadata])
        df.to_csv(csv_path, index=False)

        # Use BaseDataset because does not require clusters
        dataset = BaseDataset(
            dataset_cfg=DatasetConfig(
                seed=0,
                processed_data_path=str(tmp_path),
                csv_path=csv_path,
                cluster_path=None,
                use_synthetic=False,
                use_redesigned=False,
                filter=DatasetFilterConfig(
                    # allow oligomers for our example PDB
                    num_chains=[1, 2],
                    oligomeric=["monomeric", "dimeric"],
                ),
            ),
            is_training=True,
            task=DataTaskEnum.hallucination,
        )

        assert len(dataset) > 0

        # just test it works
        processed_file = read_processed_file(
            processed_file_path=metadata[dc.processed_path]
        )
        _ = dataset.process_processed_file(
            processed_file=processed_file,
            csv_row=metadata,
        )
