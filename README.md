# Cogeneration

Cogeneration is a protein generative model that simultaneously generates protein sequences and structures.

Proteins are represented as frames, each with a translation and rotation and torsions, and a sequence of amino acids.
It is based on MultiFlow, which applies flow matching across several domains:
Translations are interpolated in Euclidean space, rotations are interpolated in SO(3), and the sequence with discrete flow matching.

This project introduces several extensions over MultiFlow:
- **Inpainting (conditional generation)** given partial sequences / structures
  - MultiFlow only supports per-domain conditioning via seperate t (i.e. folding and inverse folding)
- **Multimer** support, enabling binder design
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
  - Track information about chains, multimer interactions, presence of non-residues, etc.
- Adds a trunk with **choice of attention mechanisms, e.g. IPA, Pairformer** (triangle attention)
- Enables **recyling** through the trunk + IPA
- **CUDA optimizations**, e.g. Flash Attention, Flash IPA, cuEquivariant triangle attention
- many improvements to code base: typing, enums, documentation, tests, etc.
- Many of these new features and modules are **optional - easily reverse compatible with MultiFlow**, and public Multiflow weights

## Installation, Training, and Sampling

See directions in [installation.md](docs/installation.md) for installation, training, and sampling.

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

## Project Structure

`/cogeneration` - the main directory for the project, containing all source code.

`/cogeneration/config/`
`/cogeneration/config/base.py` - contains the all configuration for the project, specified using Hydra dataclasses and enums. The base class is `Config` and there are many subclasses.
`/cogeneration/config/curriculum.py` - defines a Curriculum class to serially train the model on different configurations.
`/cogeneration/config/dict_utils.py` - helpers for working with dicts

`/cogeneration/data`
`/cogeneration/data/tools`
`/cogeneration/data/tools/abc.py` - Base classes for folding and inverse folding tools, `FoldingTool` and `InverseFoldingTool`
`/cogeneration/data/tools/alphafold2.py` - `AlphaFold2Tool` wrapper to run AlphaFold2 using ColabFold in subprocess
`/cogeneration/data/tools/boltz_runner.py` - `BoltzRunner` wrapper to run Boltz natively 
`/cogeneration/data/tools/protein_mpnn_runner.py` -  `ProteinMPNNRunner` to run ProteinMPNN and LigandMPNN, in subprocess or natively
`/cogeneration/data`
`/cogeneration/data/all_atom.py` - mostly from Openfold. Frames / rigids -> atom14 and atom37 representations.
`/cogeneration/data/const.py` - mostly from Openfold. sequence and structure and aatypes constants.
`/cogeneration/data/data_transforms.py` - mostly from Openfold. Openfold data pipeline for getting rigid features, angles, etc.
`/cogeneration/data/folding_validation.py` - `FoldingValidator` class to assess samples, inverse fold with ProteinMPNN, fold with ColabFold (AlphaFold2), and compute metrics
`/cogeneration/data/interpolant.py` - Large file, work horse for training and sampling. `Interpolant` class for corrupting batches during training (`Interpolant.corrupt_batch()`) and sampling from the model (`Interpolant.sample()`). Many utilities for sampling from the source distributions, and Euler-Maruyama integration across each domain.
`/cogeneration/data/io.py` - utilities for saving/loading pkl, json.
`/cogeneration/data/metrics.py` - helpers for computing metrics of samples
`/cogeneration/data/noise_mask.py` - Utilities for masking and generating noise for each domain
`/cogeneration/data/potentials.py` - Feynman-Kac Steering. Several `Potential` instances. `FKSTeeringCalculator` stateless class for computing potentials. `FKSteeringResambler` stateful class for initializing particles and resampling during inference.
`/cogeneration/data/protein.py` - mostly from Openfold. `Protein` class for processing PDBs into a `Protein` `Chain`. 
`/cogeneration/data/residue_constants.py` - mostly from Openfold. Atom types, residue constants, bond lengths, atom14 and atom37 representations and masks, etc.
`/cogeneration/data/rigid.py` - mostly from Openfold. utilities for interacting with rigids, like centering and aligning.
`/cogeneration/data/rigid_utils.py` - largely from Openfold. `Rotation` and `Rigid` classes, utilities for working with quaternions.
`/cogeneration/data/so3_utils.py` - largely from Openfold. SO(3) sampling and interpolation utilities.
`/cogeneration/data/superimposition.py` - mostly from Openfold. Superimposition and tm_score to compare protein structures.
`/cogeneration/data/tensor_utils.py` - mostly from Openfold. utilities for working with tensors.
`/cogeneration/data/trajectory.py` - `SamplingStep` and `SamplingTrajectory` classes for capturing model predictions and protein intermediate states
`/cogeneration/data/trajectory_save.py` - plotting utilities for sampled trajectories, `save_trajectory()` for writing relevant files.

`/cogeneration/dataset`
`/cogeneration/dataset/scripts` - scripts for downloading and processing data, mostly PDBs.
`/cogeneration/dataset/scripts/download_pdb.py` - Download PDB database
`/cogeneration/dataset/scripts/process_pdb_files.py` - process a set of PDB files into a metadata CSV and pre-processed pkls for each PDB file
`/cogeneration/dataset/scripts/redesign_structures.py` - use ProteinMPNN to redesign structures in the dataset
`/cogeneration/dataset/scripts/update_dataset_metadata.py` - update an existing metadata CSV with new metadata
`/cogeneration/dataset`
`/cogeneration/dataset/datasets.py` - `BaseDataset` and child class `PdbDataset` for loading Metadata CSV, redesigned structures, synthetic structures, and creating dataset. `ProcessedFiles` are loaded on the fly in `__get_item__`. `BatchFeaturizer` for generating `BatchFeatures` from a `ProcessedFile`.
`/cogeneration/dataset/filterer.py` - `DatasetFilterer` class for filtering Metadata
`/cogeneration/dataset/interaction.py` - `NonResidueInteractions` and `MultimerInteraction` classes for computing atom / backbone interactions and clashes.
`/cogeneration/dataset/mmcif_parsing.py` - utils for parsing mmCIF files
`/cogeneration/dataset/motif_factory.py` - `MotifFactory` class for generating motifs and scaffolds, several strategies for picking motif regions.
`/cogeneration/dataset/process_pdb.py` - parsing PDBs into `ProcessedFile` and `MetadataCSVRow` instances, loading a `ProcessedFile` from file.
`/cogeneration/dataset/protein_downloader.py` - Dataloader using DDP and `LengthBatcher`
`/cogeneration/dataset/test_utils.py` - Utilities for tests to construct mock features, datasets, dataloaders

`/cogeneration/datasets` - Directory containing training and test data
`/cogeneration/datasets/install.sh` - Script to download MultiFlow datasets

`/cogeneration/models`
`/cogeneration/models/attention`
`/cogeneration/models/attention/attention_pair_bias.py` - Adapted from Boltz, `AttentionPairBias` module
`/cogeneration/models/attention/attention_trunk.py` - `AttentionTrunk` switch module, to create attention trunks
`/cogeneration/models/attention/double_attention_pair.py` - Double Attention pair, hacky triangle attention alternative
`/cogeneration/models/attention/dropout.py` - Adapted from Boltz, Dropout module
`/cogeneration/models/attention/ipa_attention.py` - AttentionIPATrunk module
`/cogeneration/models/attention/ipa_flash.py` - FlashIPA wrapper 
`/cogeneration/models/attention/ipa_pytorch.py` - Mostly from Openfold, IPA submodules
`/cogeneration/models/attention/pairformer.py` - Adapted from Boltz, Pairformer block / module, and `NoSeq` variant.
`/cogeneration/models/attention/transition.py` - MLP transition module
`/cogeneration/models/attention/trianglar_attention.py`  - Adapted from Boltz, triangle attention 
`/cogeneration/models/attention/trianglar_mult.py` - Adapted from Boltz, triangle attention 
`/cogeneration/models`
`/cogeneration/models/aa_pred.py` - Simple Sequence prediction network using linear layer / MLP
`/cogeneration/models/bfactors.py` - Module for predicting B-factors
`/cogeneration/models/confidence.py` - Module for predicting pLDDT (potentially PAE, PTM in the future?)
`/cogeneration/models/edge_feature_net.py` - Simple network for embedding edge features / pair representations. Embed edges using distrogram, plus self-conditioned dist, chain, masks etc.
`/cogeneration/models/embed.py` - Embedding utilites for positions, time, distrogram
`/cogeneration/models/esm_combiner.py` - Module which combines initial node and edge embeddings with ESM single and pair embeddings
`/cogeneration/models/esm_frozen.py` - Frozen ESM model for ESM single and pair embeddings
`/cogeneration/models/loss_calculator.py` - class `BatchLossCalculator` which computes losses and serializes them into `TrainingLosses` and `AuxiliaryMetrics`.
`/cogeneration/models/model.py` - complete Pytorch model
`/cogeneration/models/module.py` - Lightning Module which defines losses, training + validation + prediction steps
`/cogeneration/models/node_feature_net.py` - Simple network for initial representation of structure, sequence, masks, positional embeddings, time embeddings.

`/cogeneration/scripts`
`/cogeneration/scripts/predict.py` - script for inference / sampling, as an `EvalRunner`
`/cogeneration/scripts/train.py` - script for training the model, as an `Experiment`
`/cogeneration/scripts/utils.py` - GPU utilities, timing, etc.
`/cogeneration/scripts/utils_ddp.py` - Helpers for setting up distributed data parallel, Mac-friendly. 

`/cogeneration/type` - enums, type aliases, structs
`/cogeneration/type/batch.py` - defines enums `BatchProp` for properties from metadata / input structure and sequence, `NoisyBatchProp` for corrupted batch, `PredBatchProp` for model outputs, and util `empty_feats` for generating empty batch
`/cogeneration/type/dataset.py` - defines enums `MetadataColumn` for columns in Metadata CSV, `RedesignColumn` for redesign metadata, `DatasetProteinColumn` for fields of a `Protein` or `ProcessedFile`, `DatasetTransformColumn` for fields generated by OpenFold transforms
`/cogeneration/type/embed.py` - embedding types `PositionalEmbeddingMethod`
`/cogeneration/type/metrics.py` - `MetricName` enum defines sampling metrics calculated, `OutputFileName` defines files written by sampling + metrics
`/cogeneration/type/str_enum.py` - base class `StrEnum` for enums
`/cogeneration/type/structure.py` - `StructureExperimentalMethod` enum and parsing PDB for experimental method
`/cogeneration/type/task.py` - `DataTask` (training task) and  `InferenceTask` (sampling task) enums

`/doc` contains documentation, including some feature specs.

`/test` contains unit tests for the project. All tests are run using `pytest`. The structure roughly matches `/cogeneration` and test files are suffixed with `_test.py`.

### Ignored Directories

There are several directories that should be ignored:
`.cache`
`ckpt`
`/inference_outputs`
`/lightning_logs`
`/multiflow_weights`
`/venv`
`/wandb`

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