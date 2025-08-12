#! /usr/bin/env python3

"""
CLI program to generate redesigned sequences using ProteinMPNN for structures in a dataset.

By default, uses dataset specified in `config.dataset.datasets`, and applies filters specified by config.
Optionally can pass a metadata CSV directly.
Must contain path to raw pdb file.

This is a slow process: ProteinMPNN is relatively quick but Boltz/AlphaFold is slow.
Outputs are streamed into the output CSVs and directory.

Example usage:
    python redesign_structures.py 

Specify a metadata CSV instead of using dataset
    python redesign_structures.py redesign.metadata_csv=$HOME/pdb/rcsb/processed/metadata.csv

Use AlphaFold2 only for monomers:
    python redesign_structures.py folding.folding_model=alphafold2 'dataset.filter.num_chains=[1]'

Specify dataset and output:
    python redesign_structures.py dataset.datasets=[dataset_spec] redesign.output_dir=/path/to/redesigns

Use pre-defined candidates to bypass inverse folding:
    python redesign_structures.py redesign.best_redesigns_csv=/path/to/best_redesigns.csv redesign.all_csv=/path/to/redesigns_prespecified.csv
"""

import hydra
from omegaconf import OmegaConf

from cogeneration.config.base import Config
from cogeneration.data.folding_validation import FoldingValidator
from cogeneration.dataset.redesign import SequenceRedesigner


@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(cfg: Config) -> None:
    # Convert to object and interpolate
    config = cfg if isinstance(cfg, Config) else OmegaConf.to_object(cfg)
    config = config.interpolate()

    validator = FoldingValidator(cfg=config.folding)

    redesigner = SequenceRedesigner(
        cfg=config.redesign, validator=validator, dataset_cfg=config.dataset
    )

    redesigner.run()


if __name__ == "__main__":
    main()
