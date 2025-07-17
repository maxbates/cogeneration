# Cogeneration

Cogeneration is a protein generative model that simultaneously generates protein sequences and structures.

Proteins are represented as frames, each with a translation and rotation and torsions, and a sequence of amino acids.
It is based on MultiFlow, which applies flow matching across several domains:
Translations are interpolated in Euclidean space, rotations are interpolated in SO(3), and the sequence with discrete flow matching.

This project introduces several extensions over MultiFlow:
- **Inpainting (conditional generation)** given partial sequences / structures
  - MultiFlow only supports per-domain conditioning via seperate t (i.e. folding and inverse folding)
- **Multimer** support, enabling binder design, and specifying RFDiffusion style **hot spot** residues
- **Stochastic paths**, for the structure and sequence, enabling e.g. conformation sampling and sequence redesign
- **Feynman-Kac steering** for sequential monte-carlo sampling guided by potentials, defined only at inference time
- **existing protein language models (e.g. ESM)** to get frozen embeddings, enriching the node and edge representations
  - particularly for sequence-conditioned tasks like inpainting
- **Improved sequence prediction** and inverse folding, using a deeper sequence prediction network
- **Torsion angle prediction** for more accurate side chain placement
- **B-factor and pLDDT prediction**, improving model understanding of flexible regions, and embedding structure experimental method
- additional losses (e.g. for atomic interactions, clashes)
- Support for **LigandMPNN (in memory) for inverse folding** during validation / redesigning sequences
- Support for **Boltz-2 (in memory) for structure prediction** or AlphaFold2 duiring validation / redesigning sequences
- **Complete PDB processing data pipeline** to generate or augment training data, with several fields added to metadata
  - Scripts to download and process **AlphaFold Database** (by default, only part of it)
  - Track information about chains, multimer interactions, presence of non-residues, etc.
- Adds a trunk with **choice of attention mechanisms, e.g. IPA, Pairformer** (triangle attention)
- Enables **recyling** through the trunk + IPA
- **CUDA optimizations**, e.g. Flash Attention, Flash IPA, cuEquivariant triangle attention
- many improvements to code base: typing, enums, documentation, tests, etc.
- Many of these new features and modules are **optional - easily reverse compatible with MultiFlow**, and public Multiflow weights

## Installation, Training, and Sampling

See directions in [installation.md](docs/installation.md) for installation, training, and sampling.

## Project Structure

```
cogeneration/ - main directory containing all source code
├── config/
│   ├── base.py - project configuration. Hydra dataclasses + enums. Base class `Config`, many subclasses
│   ├── curriculum.py - `Curriculum` class for serial training on different configurations
│   └── dict_utils.py - dictionary utilities
├── data/
│   ├── tools/
│   │   ├── abc.py - base classes `FoldingTool` and `InverseFoldingTool`
│   │   ├── alphafold2.py - `AlphaFold2Tool` wrapper for ColabFold. subprocess only
│   │   ├── boltz_runner.py - `BoltzRunner` wrapper for Boltz2. subprocess / native.
│   │   └── protein_mpnn_runner.py - `ProteinMPNNRunner` for ProteinMPNN and LigandMPNN. subprocess / native.
│   ├── all_atom.py - (~Openfold) frames/rigids to atom14 and atom37 representations
│   ├── const.py - (~Openfold) sequence, structure and amino acid constants
│   ├── data_transforms.py - (~Openfold)data pipeline for rigid features and angles
│   ├── folding_validation.py - `FoldingValidator` for sample assessment and metrics, runs folding / inverse folding.
│   ├── interpolant.py - Big file. `Interpolant` class with `corrupt_batch()` and `sample()` methods. core training and sampling. 
│   ├── io.py - save/load utilities for pkl and json
│   ├── metrics.py - sample metrics computation
│   ├── noise_mask.py - masking and noise generation utilities
│   ├── potentials.py - Feynman-Kac steering. `Potential`, `FKSTeeringCalculator`, `FKSteeringResampler` classes
│   ├── protein.py - (~Openfold) `Protein` class for PDB processing into `Chain` objects
│   ├── residue_constants.py - (~Openfold)atom types, residue constants, bond lengths, representations and masks
│   ├── rigid.py - (~Openfold) rigid utilities for centering and alignment
│   ├── rigid_utils.py - (~Openfold) `Rotation` and `Rigid` classes, quaternion utilities
│   ├── so3_utils.py - (~Openfold) SO(3) sampling and interpolation
│   ├── superimposition.py - (~Openfold) structure superimposition and tm_score computation
│   ├── tensor_utils.py - (~Openfold) tensor manipulation utilities
│   ├── trajectory.py - `SamplingStep` and `SamplingTrajectory` classes for model predictions
│   └── trajectory_save.py - trajectory plotting and `save_trajectory()` function
├── dataset/
│   ├── scripts/ - data downloading and processing scripts
│   │   ├── cluster_pdbs.py - Generate `foldseek` clusters for downloaded PDB data
│   │   ├── download_alphafold.py - AlphaFold PDB database download (SwissProt proteins)
│   │   ├── download_pdb.py - PDB database download
│   │   ├── process_pdb_files.py - PDB to Metadata CSV and pkl ProcessedFiles
│   │   ├── redesign_structures.py - ProteinMPNN structure redesign
│   │   └── update_dataset_metadata.py - metadata CSV updates (~deprecated)
│   ├── datasets.py - `BaseDataset` and `PdbDataset` classes. `BatchFeaturizer` for `BatchFeatures`
│   ├── filterer.py - `DatasetFilterer` for dataset filtering using metadata
│   ├── interaction.py - `NonResidueInteractions` and `MultimerInteraction` for computing interactions + clashes
│   ├── mmcif_parsing.py - mmCIF file parsing utilities
│   ├── motif_factory.py - `MotifFactory` for motif and scaffold generation (inpainting)
│   ├── process_pdb.py - PDB parsing to `ProcessedFile` and `MetadataCSVRow`
│   ├── protein_downloader.py - dataloader with DDP and `LengthBatcher`
│   └── test_utils.py - test utilities for mock features and datasets
├── datasets/ - training and test data directory
│   └── install.sh - MultiFlow dataset download script
├── models/
│   ├── attention/
│   │   ├── attention_pair_bias.py - (~Boltz) `AttentionPairBias` module 
│   │   ├── attention_trunk.py - `AttentionTrunk` switch module
│   │   ├── double_attention_pair.py - hacky cheaper triangle attention alternative
│   │   ├── dropout.py - (~Boltz) dropout module
│   │   ├── ipa_attention.py - `AttentionIPATrunk` module
│   │   ├── ipa_flash.py - FlashIPA wrapper
│   │   ├── ipa_pytorch.py - (~Openfold) IPA submodules
│   │   ├── pairformer.py - (~Boltz) `Pairformer` block and `NoSeq` variant
│   │   ├── transition.py - MLP transition module
│   │   ├── triangular_attention.py - (~Boltz) triangle attention module
│   │   └── triangular_mult.py - (~Boltz) triangle multiplication module
│   ├── aa_pred.py - simple MLP sequence prediction network
│   ├── bfactors.py - B-factor prediction module
│   ├── confidence.py - pLDDT prediction module
│   ├── edge_feature_net.py - edge feature embedding network
│   ├── embed.py - position, time, distrogram embeddings
│   ├── esm_combiner.py - ESM embedding combination module
│   ├── esm_frozen.py - frozen ESM model for embeddings using FAPLM
│   ├── loss_calculator.py - `BatchLossCalculator` for losses and metrics
│   ├── model.py - complete PyTorch model
│   ├── module.py - Lightning module for training/validation/prediction
│   └── node_feature_net.py - initial node feature representation network
├── scripts/
│   ├── predict.py - inference script using `EvalRunner`
│   ├── train.py - training script using `Experiment`
│   ├── utils.py - GPU utilities and timing
│   └── utils_ddp.py - distributed data parallel helpers, Mac support
└── type/ - enums, type aliases, structs
    ├── batch.py - batch property enums: `BatchProp`, `NoisyBatchProp`, `PredBatchProp`
    ├── dataset.py - dataset enums: `MetadataColumn`, `RedesignColumn`, `DatasetProteinColumn`
    ├── embed.py - `PositionalEmbeddingMethod` enum
    ├── metrics.py - `MetricName` and `OutputFileName` enums
    ├── str_enum.py - base `StrEnum` class
    ├── structure.py - `StructureExperimentalMethod` enum
    └── task.py - `DataTask` and `InferenceTask` enums

doc/ - documentation and feature specs

test/ - unit tests (run with `pytest`)
```

### Ignored Directories

There are several directories that should be ignored:
```
.cache
ckpt
/inference_outputs
/lightning_logs
/multiflow_weights
/venv
/wandb
```

## Project Conventions

### Code Style

- Code is formatted using `black` and `isort`
- Use `dataclasses` for structs and for classes. Prefer classes to several global functions.
- Prefer single line comments. Short comments can proceed code on the same line followed by 2 spaces. 
- Use multiline comments to explain complex logic. Avoid formatting comments with `#` blocks.
- Use type annotations. Comment shape annotations for tensors.
- Use kwargs generally, especially if there is more than one argument or the function is imported.

### Tests

- Tests are in `/test`
- Test by running `pytest`
- Test directory structure should ~ match the `/cogeneration` directory structure, with `_test.py` suffix.

## License

This project is licensed under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0). This means you are free to share and adapt the material for non-commercial purposes, but you must give appropriate attribution and cannot use it for commercial purposes.

See the [LICENSE.md](LICENSE.md) file for full details.

## Attribution

This project is based on MultiFlow:

```
@article{campbell2024generative,
  title={Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design},
  author={Campbell, Andrew and Yim, Jason and Barzilay, Regina and Rainforth, Tom and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2402.04997},
  year={2024}
}
```

This repo uses Openfold, but copies its source in many places.
This is primarily because OpenFold requires building kernels on install, which requires an Nvidia GPU. 
The MSA transformer etc. are not necessary for the model in this repo.
This should simplify install, testing, etc.
```
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

And benefits from the following works, among many others:

FoldFlow-2: https://github.com/DreamFold/FoldFlow
FrameFlow: https://github.com/microsoft/protein-frame-flow
Boltz: https://github.com/jwohlwend/boltz
ESM: https://github.com/facebookresearch/esm
RFDiffusion: https://github.com/RosettaCommons/RFdiffusion
ProteinMPNN: https://github.com/dauparas/ProteinMPNN
AlphaFold2: https://github.com/google-deepmind/alphafold