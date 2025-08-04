# Datasets refactor

Our data needs are diverging from MultiFlow's.

Multiflow data is broken across:
- main dataset
- redesigned sequences
- synthetic structures (AFDB)
- clusters mapping
- test dataset ids
- (for inference) length sampling dataset

## Major changes we have / want

willing to drop compatibility with MultiFlow data

- [x] drop PdbDataset class, just use BaseDataset
  - [x] drop DatasetTypeEnum
- [x] easy to specify multiple dataset files
  - e.g. PDB, AFDB, Dayhoff, and parse similarly
- [x] manage train test splits ourselves, drop test id spec
- [x] synthetic structures just treated as another dataset
- [x] clarify training / eval modes, use `eval` flag for dataset

- [ ] improve cfg for inference using PDB data for inpainting
  - [x] clearer separation of training and inference configurations
  - [x] unify inference cfg across length sampling or PDB data
  - [x] helper to get PDB dataset for inference using inference.samples cfg
    - Determine if we just want PDBs, or want length-sampling version of PDB for inpainting
  - [x] use helper EvalDatasetConstructor instead of length sampling + eval BaseDataset throughout
  - [x] update cfg for inpainting
  - [ ] ? separate subset for unconditional and conditional config (after use helper)
  - [ ] ? refactor DatasetConfig validation params, should not be required for defining eval dataset
    - [ ] ? esp handling for multimers

- [ ] reprocess PDB

- [ ] clearer dataset splits across a single dataset
  - !! Requires reprocessing PDB and tracking date 
  - [ ] using date to split
  - [ ] deprecate test dataset
  - avoid use of file to specify ids

- [ ] add Dayhoff dataset, esp if have redesigned -> backbone mapping
  - says they should but doesnt appear to be available in hugging face

## Redesigns

Original multiflow only cared about backbone, and so could substitute the redesigned sequence and consider the structure unchanged. 
It also deterministically always took the redesign, so that caching was consistent (thus, it was sort of like a new dataset).
However, we also predict torsions etc., so we rely on a new structure for each redesign.
Therefore, it's easiest to treat redesigns as a separate dataset.
The redesign script should just generate new metadata file into a new dataset, but also track the source structure and sequence.

- [ ] refactor redesign generation
  - [ ] refactor scripts to generate redesigns
  - [ ] keep all redesigns under some threshold, and track RMSD -- dont only keep best
  - [ ] track source of generation + tools used + time, combine with multiflow data
  
- [ ] Script to convert multiflow redesigns... may need to predict structures from redesigns?
  - Already have the needles in the haystack, so probably worth it?

- [ ] Generate own redesign dataset, e.g. multimers

Old plan

- XXXX support multiple sequences per structure, picking one at __get_item__
  - e.g. either native sequence or a redesigned sequence
  - e.g. pick from an MSA 
  - the Dayhoff dataset defines multiple sequences per structure
  - [ ] determine how to handle caching - seq changes atoms present
  - [ ] redesign may be sequence and structure, since if sequence changes, and want to predict torsions etc. need new structure
  - [x] redesigns are optional.
  - [x] refactor dataset filter for redesigns, move to dataset config
  - [ ] cfg option to disable using redesigns
  - [ ] BaseDataset tracks all redesigns in 1 -> N mapping, only drop bad ones
    - [ ] BaseDataset.load_dataset returns metadata + redesigns as separate object
  - [ ] don't drop if low RMSD - just use original -- treat as filter
    - [ ] need to separately apply filter to redesigns, since won't be in CSV
  - [ ] support sampling from native and all redesigns at __get_item__
    - [ ] featurize_processed_file can take sequence to override default / cached value
    - [ ] cfg to specify proportion of native vs redesign sampling