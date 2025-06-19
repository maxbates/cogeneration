# Attention Speed-ups

Want to introduce flash attention and kernels to speed up various attention mechanisms in the project.

Should enable faster training, faster inference, lower memory footprint.

## Current State

`IPA` from Openfold makes up the bulk of the trunk. 
IPA is fairly slow, so we only have a few blocks in the trunk, which reduces richness of representation.
It also limits size of the model due to memory constaints.
However, `IPA` is SE(3) equivariant, which is important for performing backbone updates.

In `FrozenESM` we use a PLM to enrich the node and edge representations.
This model is much larger than the trainable parameters for flow matching model, so it's performance is important.

In `ESMCombiner` we have some hacky `DoubleAttentionPairBlock` blocks that are ~ a cheaper alternative to triangle attention. 
We should allow substituting these with actual triangle attention / pairformer.

In `SequenceIPANet` we use `IPA` blocks before predicting logits. 
Because this module does not modify the backbone, it does not need to be equivariant (invariant ok).
We should move this module to use a faster attention mechanism, e.g. triangle attention.

## Constraints

- maintain backwards compatibility with Multiflow
- able to run modules on a Mac with CPU and/or MPS (fallbacks ok)

## Alternatives 

### ESM LM 

- FAPLM project support using flash attention (or fallback to pytorch SDPA) for ESM.

Flash attention etc. do not output attention weights. 
So e.g. need option to only enrich node representation if use flash attention in ESM

### Invariant Point Attention

- Flash attention for IPA
- cuEquivariance fused IPA kernels (a bit faster, H100+ only)

### non-IPA Attention

- Pairformer from AF/ Boltz using Triangle attention, esp with `trifast` or NVidia `cuequivariant` package.

IPA is required to do the backbone updates. 
However, representation enrichment can be done with invariant blocks.
IPA-alternatives can be used in `SequenceIPANet`, `ESMCombiner`, and preceding `IPA` blocks.

Need to determine a reasonable ratio of pairformer:IPA blocks, or a minimum number of IPA blocks. 

## Implementation

- Pre-work
  - [x] update to python 3.2

- Add New Modules
  - [ ] Adopt FAPLM package to replace fair-esm
    - see branch `faplm`
  - [x] Adapt triangle attention from Boltz2 to project
    - has some dependencies to work through
  - [x] Adapt `Pairformer` from Boltz2 to project
    - has some dependencies to work through
  - ? FoldingBlocks from ESM3
    - is there any advantage to them over optimized blocks?
  
- Migrate new modules
  - [x] variable names consistent
  - [x] each major module has its own config

- [x] switch Attention block switch to create desired attention block
  - [x] Base class / switch class with single interface
  - [x] implement `enabled` prop
  - [x] be consistent which trunks have layer norm. 
    - Maybe we just have a trunk factory that creates N blocks, given a factory function to make a block


- Integrate new trunk switch module
  - [x] after ESMCombiner, before IPA trunk
    - [x] drop attention trunk from ESMCombiner
  - [x] before SequenceIPANet
    - [x]  drop this module + config etc., use existing aa pred net MLP

- Config
  - [x] easy way to pass config, update with number of layers, etc.
    - may need to define default shared configs, and override?
    - Don't want to re-define each one where may be used
  - [x] ensure generic way to specify node_dim and edge_dim
  - [x] let's define a single `model.attention` with subconfig for each attention type
  - [x] then define 2 trunks, pre IPA and pre AA pred, each which takes a type + num layers
  - [x] better group attention mechanism configs
  - [x] `ESMCombiner` pair representation is optional, to enable flash attention
  - [x] Enum to define block type for rep enrichment, shared default cfg for each 
    - `IPA`, `Pairformer`, `DoublePairAttention`
  - [x] update `tiny` hyperparam config where appropriate
  
- Speed up IPA
  - [x] FlashIPA import 
  - [x] wrapper to patch IPA?

- Tests
  - [x] test model pass for each model configuration 

- Clean up Modules
  - [x] Break up `models/` directory
  
- Support backlog
  - [ ] pull `num_layers` out of config, per trunk, into an (override?) argument
      - avoid patching config per trunk
      - [ ] consider add more attn parameters to model hyper param config, e.g. num heads
  - [x] expose `training` / use pytorch prop for dropout mask, or different values for training / inference
  - [ ] cuEquivariance env setup helper, e.g. set `CUQUI_ENABLE_BF16` etc.
  
- [ ] shared config `kernel` instead of only `local` option
    - Or, detect what is available and use helper throughout -- probably want explicit option
  - [ ] consistent `use_kernels` vs `kernel` config

- [ ] `setup.py` optional dependencies in `[cuda]` extras
  - [ ] improve `torch`
  - [ ] `cuequivariance` install
  - [ ] install flash-ipa
  - [ ] `pip install flash-attn --no-build-isolation`

- [ ] address TODO(attn)

- [x] update README code layout