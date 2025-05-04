# Multimers

We want to be able to generate multimers. 
Generating binders is basically inpainting, with multiple chains.

Requires considering more residues, and therefore more memory. 
However, probably want to continue training from a checkpoint trained on generating/inpainting monomers.

## Current State + Questions

There are many aspects of the model and framework that should already support multiple chains.
For example, We already pass around `chain_idx` alongside `res_idx`.
However, There are some assumptions we make that we are working with monomers.
Most of these assumptions are in the `data` and `dataset` side of things.

Items in the `dataset` are currently single chains. 
When we load a PDB, we currently take out chain `"A"` in `parse_pdb_feats()`
We pass around `chain_feats` but need to move to `all_chain_feats` which is a dictionary of chains.
Or, `ProcessedFile` needs to represent the chains mashed together appropriately.

Conversely, `process_pdb_files()` already merges multiple chains. 
We need to inspect the multimer files available and see how they were encoded (presumably by `process_pdb_files()`).
We need to see what the indexing looks like in those multi-chain files.
We need to merge this behavior with `parse_pdb_feats()`.

Currently, we center the monomer to maintain translation invariance.
We instead need to center the multimer i.e. center of mass of all chains. 
There is an option to do this in `parse_pdb_feats()` for the dataset.
During sampling, Interpolant's current centering procedure using COM should be fine.

We currently start with centered guassian noise.
(?) We could instead use the "harmonic prior" from EigenFold / ESMFlow.

The model and module should mostly "just work."
We probably need to update how we embed things. Changing the `idx` may be sufficient.

Validation etc. assumes a single chain, and will be a reasonable lift to update where appropriate.

## TODO

- Theory
    - read RFDiffusion2
    - read AlphaFold-Multimer chain-break handling and relative positional embeddings
        - What is a `Protein`, what is a `Complex`, what should we use
    - read ESM-Flow / EigenFold "harmonic prior"

- Data
    - Parsing
        - [x] refactor `parse_chain_feats` to take `Protein` and handle multiple chains
        - [x] update `parse_pdb_feats()`
            - [x] remove default `chain="A"` argument, require passing it
        - [x] update `protein.py`, need multi-chain version of `process_chain() -> Protein`
        - [x] center all chains, not single chain
            - more important happens in dataset 
        - [x] `cfg.dataset.chain_gap`: integer offset between chains
            - We already support a space between chains when randomize chains
        - [x] inspect public multiflow multimer files, see how they are written, how non-residues handled
        - Per chain trimming
            - [x] Enable per-chain trimming in `process_pdb`, e.g. to clear out solvent atoms between chains
            - [x] test
            - [x] create enum for trimming methods
            - [x] add column to metadata enum
            - [x] reprocess PDB metadata (and other datasets) to add to CSV
                - [x] write a little script to add it
            - [x] update metadata creation to include new column - should generate both columns
            - [x] LengthBatcher uses correct `modeled_seq_len` column depending on method
            - [x] update eval dataset to use correct column
            - [x] check other references to `modeled_seq_len`
            - [ ] update `cluster` file to support other trimming strategy?
                - not sure what this cluster file is doing. maybe based on 2d structure?
                - should generate in data pipeline
    - Dataset
        - Filtering
            - [ ] convert `num_chains` filter to `dataset.filter.min_chains` / `dataset.filter.max_chains`
        - Dynamic chain selection
        - (?) Differentiate oligomers vs heteromers; target sampling weights per class per epoch
    
- Interpolant
    - Noise
        - [x] Introduce harmonic prior for translations
            - [x] option for single gaussian, or harmonic prior
            - [x] `cfg.interpolant.trans.noise_type`: “gaussian” vs “harmonic” (once implement alternative like harmonic prior)
       - [x] update interpolant if stochastic for sampling and training to use correct prior
       - [x] confirm stochastic paths still reasonable
    - Batch OT
        - [x] Do we need to make updates to OT implementation to support multiple chains
            - Currently, cost depends on positions, everything is a monomer
            - If use harmonic prior, `chain_idx` will vary between samples, not really meaningful to swap within batch
               - However, we are just aligning residue positions 1-to-1, chain_idx not explicitly important 
            - Maybe a multimer will just fall through with different point clouds esp if harmonic
            - But may want to be explicit, and only center + align 
        - [x] enable center + align without batch OT
    - Sample
        - [x] Require passing in `chain_idx` and `res_idx` so don't default to `torch.ones()`

- Model
    - [x] update positional embeddings
         - [ ] note that each chain is indexed from 1. Update positional embedding strategy accordingly?
         - [ ] consider clipping positional embeddings so across-chain out of range (like AF2 multimer) 
    - NodeFeatureNet
         - [x] require `embed_chain=True` in `cfg.node_features` to inject per-residue sinusoidal chain embeddings via `chain_idx`
         - [x] ensure `chain_idx` is integer (cast to `long`) before calling `get_index_embedding`
         - (?) optionally replace or augment sinusoidal chain encoding with a learned `nn.Embedding(num_chains, dim)` for small chain counts
    - EdgeFeatureNet
         - [x] enable `embed_chain=True` in `cfg.edge_features` to add the same-chain binary feature
    - Embedders
         - existing `get_index_embedding` (sinusoidal) covers both `res_idx` and `chain_idx`; no core change required
         - (?) add a learned chain embedding helper
    - SequenceIPANet
         - (?) Pass `chain_idx` to `sequence_ipa_net`, or positional embeddings better somehow?
            - Do already provide option to use `init_node_embed`...
    - [x] Update `ipa_pytorch.py` to support multimers if necessary
    
- Training
    -  ~~Update losses~~ losses fine, handle hotspots separately
        - [x] no neighbor loss across chains
        - [x] C-alpha distance loss accounts for chains
        - (?) consider explicit cross-chain distances?
            - not sure what beyond hot spot and interaction distance is necessary...
            - Want to support training on non-interacting pairs too - see how RosettaFold did this in the paper where they predict binding

- Inpainting
    - [x] Update motif selections to account for multiple chains
         - e.g. break motif on chain breaks for window-based methods
    - [x] Update motif segmenting to account for multiple chains
    - [x] Add new motif selection strategies
         - [x] masking binding interfaces of one chain, preserve rest
         - [x] masking binding interfaces of both chains, preserve rest
         - [x] mask entire interacting chain
    - [x] support chain breaks
        - Update MotifFactory
            - [x] new Segment subclass ChainBreak
            - [x] ensure chain breaks respected in motif generation
            - [x] pass `chain_idx` to `segments_from_diffuse_mask`, add ChainBreak segments
        - [x] update chain shuffling + residue indexing to respect breaks
            - [x] should manage chain_idx and res_idx while building up segments
                - chain break size should use config `chain_gap_dist`
                - do need to reset chain_idx to be 0 indexed
                - break up `reset_res_idx` and `reset_chain_idx` into separate functions
            - ~~remove chain shuffling and randomization - just use for unconditional~~
        - [x] allow specifying chain break in contigmap
            - [x] use RFDiffusion style `/0` break
        - [x] update tests
        - [x] drop `chain_gap_dist`... not needed
        

- Sampling
    - [x] new eval dataset for multimers? or update PDB dataset with new motif strategies
    - [x] enable multimers in length sampling dataset for inference
        - [x] include in sampling cfg

- Folding Validation
    - New Metrics
        - [ ] Binding interface metrics
            - [ ] binding precision / recall
            - requires refactor to pass through the full original structure, or similar
            - `true_bb_positions` e.g. for inpainting (most relevant task) is limited to the motifs
        - (?) Split some metrics per chain, e.g. RMSD, LDDT

- IO
    - [x] update PDB/JSON exporters for multiple chains

- Visualization
    - [x] Ensure visualization works with multiple chains

- Tests
    - [x] multi-chain `parse_pdb_feats()`
    - [x] `process_pdb_file()` -> `read_processed_file()`
    - [x] Parse `all_chain_feats` -> batch in BaseDataset
    - [x] fixture for dummy/real multimer batch 
    - [x] dataset loader multimers
    - [ ] `model.forward()` multimer
    - [ ] `module.training_step()` multimer
    - [ ] `module.validation_step()` multimer
    - [ ] `module.predict_step()` multimer
  
- Misc
    - [x] Remove default values for `chain_idx` and `res_idx`, require passing them.
    - [ ] address `TODO(multimer)`
  
## Future Work

- New Multimer dataset + metadata
    - [ ] Check how many multimers are included in MultiFlow data dump    
    - [ ] consider processing PDB multimers manually?
        - [ ] better small molecule handling, especially if in binding interface
    - [ ] Acquire more binders for training
    - [ ] ? expose per-chain metadata (chain IDs, lengths) when generating metadata file
        - however wont be available in public multiflow metadata file, use for new dataset?
    - [ ] Ability to filter to chains of particular lengths,
        - **to support length batching, need to know chain lengths ahead of time**
        - may make sense to filter and data augment in processing pipeline - i.e. each combination can be its own item
            - avoid things with too much symmetry, too many chains
        - e.g. drop a little peptide if 2 primary interacting chains
        - or only keep 2 random interacting pairs
    - (?) ability to filter chains with internal gaps
    - [ ] Review RFDiffusion training curriculum
    - [ ] filter when small molecule is interacting or between chains
        - How do we account for this, if the small molecules have been removed in the processed file?
        - see e.g. https://www.rcsb.org/3d-view/6DY1
            - it's just called `dimeric` in `pdb_metadata.csv`, ignores TON / MYR fatty acid chains completely
        - May need to process multimers again ourselves...
        - Can look for molecules with interactions.
            - Check for any presence of interacting molecules (ignore water etc.)  
            - Check for overlapping ranges with protein interactions.
    - [ ] generate cluster as part of pipeline

- contigmap parsing
    - [ ] better differentiate `Segments` from feats vs `Segments` from contigmap
    - Segments from feats and diffuse mask have start / end in feats, i.e. flattened chains
    - Segments from contigmap specify `chain` or `chain_id` and start/end are chain specific
    - Need to be able to convert contigmap segments -> feats segments

- Support "hotspots"
    - [ ] RFDiffusion style specification of interacting residues
        - pass as additional channel
    - [ ] Method to determine hotspots from a complete structure
        - pick a few residues, match something like RFDiffusion
    - [ ] update losses to ensure important binding residues are interacting across chains
        - optimize for picking a few, non-exhaustive, i.e. loss should focus on precision, recall within a region rather than all residues
    
- Consider residue-specific time schedules
    - e.g. for inpainting a binder, freeze the target chain at t=1
    - requires "fixed" version of inpainting rather than "guided" version