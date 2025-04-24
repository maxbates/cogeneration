# Multimers

We want to be able to generate multimers. 
Generating binders is basically inpainting, with multiple chains.

Requires considering more residues, and therefore more memory. 
However, probably want to continue training from a checkpoint trained on generating/inpainting monomers.

## Current State + Questions

We already pass around `chain_idx` alongside `res_idx`.
There are many aspects of the model and framework that should already support multiple chains.
However, there are several assumptions we make that we are working with monomers.

Items in the dataset are currently single chains. 
When we load a PDB, we currently take out chain `"A"` in `parse_pdb_feats()`
We pass around `chain_feats` but need to move to `all_chain_feats` which is a dictionary of chains.

Currently, we center the monomer to maintain translation invariance.
We instead need to center the multimer i.e. center of mass of all chains. 
There is an option to do this in `parse_pdb_feats()`

We currently start with centered guassian noise.
(?) We could instead use the "harmonic prior" from EigenFold / ESMFlow.

Validation etc. assumes a single chain, and will be a reasonable lift to update where appropriate.

## TODO

- Theory
    - read RFDiffusion2
    - read AlphaFold-Multimer chain-break handling and relative positional embeddings
    - read ESM-Flow / EigenFold "harmonic prior"      

- Config
    - [ ] `chain_gap`: integer offset between chains
    - [ ] `noise.type`: “gaussian” vs “harmonic”
    - [ ] `dataset.filter.min_chains` / `dataset.filter.max_chains`
    - [ ] sampling weights for homo vs hetero
  
- Data
    - Parsing
        - [ ] update `parse_pdb_feats()`
            - [ ] remove default `chain="A"` argument, require passing it
        - [ ] update indexing, choose some strategy for chain gaps
             - [ ] include a gap (like ~200 AlphaFold Multimer) between chains
        - [ ] center all chains, not single chain
        - [ ] Parse `all_chain_feats` -> batch
    - Dataset
        - [ ] Establish options for multiple chains in dataset filter (`min_chains`, `max_chains`)
        - [ ] Differentiate oligomers vs heteromers; sampling weights
        - [ ] support dynamic padding/cropping for variable total residues
        - [ ] expose per-chain metadata (chain IDs, lengths)   
    
- Interpolant
    - Noise
        - [ ] Introduce harmonic prior for translations
            - [ ] option for single gaussian, or harmonic prior
    - Batch OT
        - [ ] Do we need to make updates to OT implementation to support multiple chains
        - Currently, cost depends on positions, everything is a monomer
        - If use harmonic prior, `chain_idx` will vary between samples, not really meaningful to swap within batch
    - Sample
        - Require passing in `chain_idx` and `res_idx` so don't default to `torch.ones()`

- Model
    - [ ] update positional embeddings
         - [ ] consider clipping positional embeddings so across-chain out of range (like AF2 multimer) 
    - NodeFeatureNet
         - [ ] require `embed_chain=True` in `cfg.node_features` to inject per-residue sinusoidal chain embeddings via `chain_idx`
         - [ ] ensure `chain_idx` is integer (cast to `long`) before calling `get_index_embedding`
         - [ ] optionally replace or augment sinusoidal chain encoding with a learned `nn.Embedding(num_chains, dim)` for small chain counts
    - EdgeFeatureNet
         - [ ] enable `embed_chain=True` in `cfg.edge_features` to add the same-chain binary feature
    - Embedders
         - existing `get_index_embedding` (sinusoidal) covers both `res_idx` and `chain_idx`; no core change required
         - [ ] (?) add a learned chain embedding helper
    - SequenceIPANet
         - [ ] Pass `chain_idx` to `sequence_ipa_net`, or positional embeddings better somehow?
    - [ ] Update `ipa_pytorch.py` to support multimers if necessary
    
- Training
    - [ ] Update losses
          - [ ] no neighbor loss across chains
          - [ ] contact-presrvation loss
              - e.g. for residues more than N residues apart, is contact preserved
          - [ ] consider explicit cross-chain distances? 
          - Want to support training on non-interacting pairs too - see how RosettaFold did this in the paper where they predict binding
    - [ ] Enable larger residue window, e.g. 256 -> 384 for multimers 

- Sampling
    - 

- Inpainting
    - [ ] Update motif selections to account for multiple chains
         - e.g. break motif on chain breaks for window-based methods
    - [ ] Update motif segmenting to account for multiple chains
    - [ ] Add new motif selection strategies
         - [ ] masking binding interfaces of one chain, preserve rest
         - [ ] masking binding interfaces of both chains, preserve rest
         - [ ] mask entire interacting chain

- Folding Validation
    - New Metrics
        - [ ] Split some metrics per chain, e.g. RMSD, LDDT
        - [ ] Binding interface metrics
            - [ ] binding precision / recall

- IO
    - [ ] update PDB/JSON exporters for multiple chains

- Visualization
    - [ ] Ensure visualization works with multiple chains

- Tests
    - [ ] multi-chain `parse_pdb_feats()`
    - [ ] fixture for dummy/real multimer batch 
    - [ ] dataset loader multimers
    - [ ] `model.forward()` multimer
    - [ ] `module.training_step()` multimer
    - [ ] `module.validation_step()` multimer
    - [ ] `module.predict_step()` multimer
  
- Misc
    - [ ] Remove default values for `chain_idx` and `res_idx`, require passing them.
    - [ ] address `TODO(multimer)`
  
## Future Work

- Data
    - [ ] Check how many multimers are included in MultiFlow data dump
    - [ ] Acquire more binders for training
    - [ ] Review RFDiffusion training curriculum

- Support "hotspots"
    - RFDiffusion style specification of interacting residues
    
- Consider residue-specific time schedules
    - e.g. for inpainting a binder, freeze the target chain at t=1
    - requires "fixed" version of inpainting rather than "guided" version