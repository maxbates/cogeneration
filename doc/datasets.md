# Datasets refactor

Our data needs are diverging from MultiFlow's.

Multiflow data is broken across:
- main dataset
- redesigned sequences
- synthetic structures (AFDB)
- clusters mapping
- test dataset ids
- (for inference) length sampling dataset

Major changes we have / want:
- [x] willing to drop compatibility with MultiFlow data
- [x] drop PdbDataset class, just use BaseDataset
  - [x] drop DatasetTypeEnum
- [x] easy to specify multiple dataset files
  - e.g. PDB, AFDB, Dayhoff, and parse similarly
- [x] manage train test splits ourselves, drop test id spec
- [x] synthetic structures just treated as another dataset
- [x] clarify training / eval modes, use `eval` flag for dataset
- [ ] inference using PDB data for inpainting
  - [ ] unify inference cfg across length sampling or PDB data
  - [ ] helper to get PDB dataset for inference using inference.samples cfg
  - [x] clearer separation of training and inference configurations
- [ ] support multiple sequences per structure, picking one at __get_item__
  - e.g. either native sequence or a redesigned sequence
  - e.g. pick from an MSA 
  - the Dayhoff dataset defines multiple sequences per structure
  - [ ] track all redesigns in 1 -> N mapping, only drop bad ones
  - [ ] don't drop if low RMSD - just use original  
  - [x] redesigns are optional. 
  - [ ] cfg option to disable using redesigns
- [ ] clearer dataset splits across a single dataset
  - deprecate test dataset
  - e.g. using date to split, or a list of clusters
  - avoid use of file to specify ids
- [ ] add Dayhoff dataset, esp if have redesigned -> backbone mapping
  - says they should but doesnt appear to be available?
