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

- New Modules
  - [ ] Adopt FAPLM package to replace fair-esm
    - see branch `faplm`
  - [ ] Adapt triangle attention from Boltz2 to project
    - has some dependencies to work through
  - [ ] Adapt `Pairformer` from Boltz2 to project
    - has some dependencies to work through
  - ? FoldingBlocks from ESM3
    - is there any advantage to them over optimized blocks?

- Speed up Modules
  - [ ] FlashIPA / cuEquivariance wrapper
  - [ ] env setup helper, e.g. set `CUQUI_ENABLE_BF16` etc.

- Support backlog
  - expose `is_training` for dropout mask, or different values for training / inference

- Config
  - [ ] shared config `kernel` instead of only `local` option
    - Or, detect what is available and use helper throughout -- probably want explicit option
  - [x] `ESMCombiner` pair representation is optional, to enable flash attention
  - [ ] Enum to define block type for rep enrichment, shared default cfg for each 
    - `IPA`, `Pairformer`, `DoublePairAttention`
  

- Tests
  - [ ] test model pass for each model configuration 

- [ ] `setup.py` optional dependencies in `[cuda]` extras
  - [ ] improve `torch` and `cuequivariance` install