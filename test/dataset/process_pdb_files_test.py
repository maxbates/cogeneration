import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from cogeneration.config.base import DatasetConfig, DatasetFilterConfig, DataTaskEnum
from cogeneration.data.enum import DatasetColumns as dc
from cogeneration.data.enum import DatasetProteinColumns as dpc
from cogeneration.dataset.data_utils import parse_chain_feats
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.process_pdb_files import process_file

# https://www2.rcsb.org/structure/2QLW
example_pdb_path = Path(__file__).parent / "2qlw.pdb"


class TestProcessPDBFiles:
    def test_process_file(self, tmp_path):
        metadata = process_file(
            file_path=str(example_pdb_path.absolute()),
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

    def test_parsing_processed_file(self, tmp_path):
        metadata = process_file(
            file_path=str(example_pdb_path.absolute()),
            write_dir=str(tmp_path),
        )
        with open(metadata[dc.processed_path], "rb") as f:
            pkl = pickle.load(f)

        # Test can load and process
        parse_chain_feats(pkl)

    def test_dataset_using_processed_file(self, tmp_path):
        metadata = process_file(
            file_path=str(example_pdb_path.absolute()),
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

        # just test it works
        _ = dataset.process_processed_path(
            processed_file_path=metadata[dc.processed_path],
        )
