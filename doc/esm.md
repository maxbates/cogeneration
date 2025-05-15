## ESM embeddings

Particularly if only sequence is fixed, and structure is interpolated, should get embedding given sequence + structure.  Currently, its primarily structure.

Because some sequence is always defined (the motifs), and the intermediate sequence is discrete, can use ESM to get an embedding

In theory, should improve our ability to unmask sequences... And we should get better single and pair representations by re-using pretrained model.

Can get both a single and pair representation, merging with representations from  `node_feature_net` and `edge_feature_net`
This is basically what FoldFlow2 does.

FoldFlow2 uses IPA (bb_encoder and bb_decoder) and ESM FoldingBlocks (trunk) rather than just IPA.
The folding blocks are invariant, enrich the representations. They do not update the backbone. 
But ultimately, it performs a backbone update (and predicts torsion angles).
The ultimate updates use IPA and SE(3) equivariant.

Can probably take ESM module from FoldFlow2

### Questions

- [ ] Confirm handles multiple chains

- Inpainting
    - should ESM representation only take motifs (sequence fixed, structure chains) or update as `aatypes` change?
        - probably update...

### TODO

- [x] Add ESM module
    - [x] ESM should be frozen
    - [x] Convert AF2 style representation to ESM and back
       - alphabet
       - padding / EOS
       - structure representation?
       - [x] support multiple chains

- [ ] Cfg option to use ESM
    - [ ] Specify ESM model size, adjusts hyperparameters
- Should only work with tasks where sequence is provided, e.g. inpainting and forward_folding
    - [ ] validate approach with forward_folding
    - [ ] compare IPA to folding blocks approach

- [ ] Merge `node_feature_net` and `edge_feature_net` with ESM embeddings
    - [ ] be sure time + positional information is preserved
    - [ ] By default, don't `embed_aatype` in `node_network` if using ESM?
        - [ ] how embed `aatype_sc` ?
        - [ ] important to embed `cat_t`?
    
- [ ] Pass these merged representations to sequence prediction net

- [x] dummy model for testing in registry
    - [ ] enable in test suite setup

- [ ] Update checkpointing
    - include in checkpoint?
    - load ESM params directly into network?


