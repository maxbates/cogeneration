from pathlib import Path

import numpy as np
import pandas as pd

from cogeneration.config.base import DatasetConfig, DatasetFilterConfig
from cogeneration.data.io import read_pkl
from cogeneration.data.residue_constants import unk_restype_index
from cogeneration.dataset.datasets import BaseDataset
from cogeneration.dataset.process_pdb import (
    read_processed_file,
    trim_chain_feats_to_modeled_residues,
)
from cogeneration.scripts.process_pdb_files import process_pdb_with_metadata
from cogeneration.type.dataset import DatasetProteinColumn
from cogeneration.type.dataset import DatasetProteinColumn as dpc
from cogeneration.type.dataset import MetadataColumn as mc
from cogeneration.type.task import DataTask

# https://www2.rcsb.org/structure/2QLW
# This protein has a fair amount of weird stuff going on to make it a good test case:
# - it is a dimer
# - has small molcules after chains that should be trimmed
# - has small molecules between chains (MG and FMT)
# - has non-residue atom types (MG atoms)
# - has some unknown residues (like seleniummethionine)
# - has preceding sequence without atoms
# - glycine starts at position -1 in each chain
example_pdb_path = Path(__file__).parent / "2qlw.pdb"


def shallow_copy_dict(d):
    return {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in d.items()}


def unk_run(aatype) -> int:
    """max count UNK residues in a row"""
    unk_mask = np.array(aatype == unk_restype_index).astype(int)
    edges = np.flatnonzero(np.diff(np.concatenate(([0], unk_mask, [0]))))
    max_run = (edges[1::2] - edges[::2]).max(initial=0)
    return max_run


class TestProcessPDBFiles:
    def test_process_file(self, tmp_path):
        metadata, processed_pdb = process_pdb_with_metadata(
            pdb_file_path=str(example_pdb_path.absolute()),
            write_dir=str(tmp_path),
        )
        assert metadata is not None

        # check written file exists
        pkl = read_pkl(metadata[mc.processed_path])

        # DEBUG convert to PDB to inspect
        # pdb_path = tmp_path / "2qlw_rewrite.pdb"
        # protein = create_full_prot(
        #     atom37=pkl[dpc.atom_positions],
        #     atom37_mask=pkl[dpc.atom_mask],
        #     aatype=pkl[dpc.aatype],
        #     b_factors=pkl[dpc.b_factors],
        # )
        # with open(pdb_path, "w") as f:
        #     f.write(to_pdb(protein))
        # print(pdb_path)

        # check metadata
        assert metadata[mc.num_chains] == 2
        # 2x 100 residues with known aa types
        assert metadata[mc.moduled_num_res] == 200
        # 2x 227 molecules/residues
        assert metadata[mc.seq_len] == 454
        # 100 modeled + intervening molecules + 100 modeled
        assert metadata[mc.modeled_seq_len] == 335
        assert metadata[mc.oligomeric_detail] == "dimeric"
        assert metadata[mc.quaternary_category] == "homomer"
        assert metadata[mc.helix_percent] > 0.1

        # modeled sequence includes non-AA residues (i.e. 20 = unknown)
        assert len(pkl[dpc.aatype]) == metadata[mc.seq_len]
        # check number actual residues in seq
        assert np.sum(pkl[dpc.aatype] != 20) == metadata[mc.moduled_num_res]
        # modeled positions only for valid residues
        assert len(pkl[dpc.modeled_idx]) == metadata[mc.moduled_num_res]

        # check multimer interactions
        assert metadata[mc.num_backbone_res_interacting] == 22
        assert metadata[mc.num_chains_clashing] == 0

        # check non-res interactions
        assert metadata[mc.num_metal_interactions] == 2  # mg ions

        # check all expected keys are present
        expected_keys = [key for key in DatasetProteinColumn]
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
        _ = read_processed_file(processed_file_path=metadata[mc.processed_path])

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
            task=DataTask.hallucination,
        )

        assert len(dataset) > 0

        # just test it works
        processed_file = read_processed_file(
            processed_file_path=metadata[mc.processed_path]
        )
        _ = dataset.process_processed_file(
            processed_file=processed_file,
            csv_row=metadata,
        )

    def test_trim_chain_feats_to_modeled_residues(self, tmp_path):
        _, processed_pdb = process_pdb_with_metadata(
            pdb_file_path=str(example_pdb_path.absolute()),
            write_dir=str(tmp_path),
        )

        # track positions
        processed_pdb["FLAG"] = np.arange(processed_pdb[dpc.aatype].shape[0])

        # check the modeled_idx is present
        assert dpc.modeled_idx in processed_pdb, "`modeled_idx` key should be present"
        modeled_idx = processed_pdb[dpc.modeled_idx].copy()
        modeled_mask = np.zeros(processed_pdb[dpc.aatype].shape, dtype=bool)
        modeled_mask[modeled_idx] = True

        assert (
            unk_run(processed_pdb[dpc.aatype]) > 20
        ), f"expected long runs of UNK (between chains)"

        trimmed = trim_chain_feats_to_modeled_residues(
            chain_feats=shallow_copy_dict(processed_pdb),
            trim_chains_independently=True,
        )

        assert dpc.modeled_idx not in trimmed, "`modeled_idx` key should be deleted"
        assert len(trimmed[dpc.chain_index]) > len(
            modeled_idx
        ), "should be longer than modeled_idx because UNK tokens"

        # all modeled_idx positions are present in trimmed
        assert np.all(np.isin(modeled_idx, trimmed["FLAG"]))

        # first / last residue in each chain not UNK
        for cid in np.unique(trimmed[dpc.chain_index]):
            pos = np.flatnonzero(trimmed[dpc.chain_index] == cid)
            first_res, last_res = (
                trimmed[dpc.aatype][pos[0]],
                trimmed[dpc.aatype][pos[-1]],
            )
            assert first_res != unk_restype_index and last_res != unk_restype_index

        # trims atom_positions sanely
        assert trimmed[dpc.atom_positions].shape == (
            len(trimmed[dpc.aatype]),
            37,
            3,
        ), "atom_positions should be same length as aatype"

        # no long UNK stretches
        assert (
            unk_run(trimmed[dpc.aatype]) <= 5
        ), f"no long runs of UNK (between chains)"

        # per chain trimming yields shorter feats
        trimmed_not_independently = trim_chain_feats_to_modeled_residues(
            chain_feats=shallow_copy_dict(processed_pdb),
            trim_chains_independently=False,
        )
        assert len(trimmed_not_independently[dpc.aatype]) > len(
            trimmed[dpc.aatype]
        ), "should be longer than independently trimmed"
        # should retain long UNK stretches
        assert (
            unk_run(trimmed_not_independently[dpc.aatype]) > 20
        ), f"expected long runs of UNK (between chains)"
