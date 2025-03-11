doc to plan for scaffolding feature

Review FrameFlow (second version), which enables scaffolding within a similar codebase (just for protein scaffolding)
https://github.com/microsoft/protein-frame-flow

## TODOs

- Theory
    - consider how scaffolding will work with logits, in both scaffolding regimes.
- Codebase
    - [ ] new training task `scaffolding`
    - [ ] inference task `scaffolding`
    - [ ] scaffolding type: `motif amortization` (fixed / conditional) vs `motif guidance` (move over time)
    - [ ] determine how we switch between unconditional generation and scaffolding
        - e.g. support switching behavior if `diffuse_mask` is defined to support unconditional generation, with some fraction of batches each
- Datasets
    - [ ] factory to generate motif masks
    - [ ] new eval dataset
        - Length sampler not sufficient. Take PDB structures, mask portions of them. Independent remaining length.
- Interpolant
    - [ ] update `corrupt_batch()` 
    - [ ] update sample() 
        - [ ] partial `diffuse_mask`. require trans_1 and rotmats_1 defined.
        - [ ] update centering procedure, if needed
        - [ ] update self-conditioning to use `diffuse_mask`
    - [ ] enable `twisting` for motif guidance sampling like FrameFlow
        - see https://github.com/microsoft/protein-frame-flow/blob/main/data/interpolant.py#L327
- Module
    - [ ] `validation_step()` shouldn't only default to unconditional generation 
- Training
    - [ ] log scaffolding metrics like scaffolding percent, motif size, `diffuse_mask.mean()`
- Sampling
    - [ ] new eval dataset (see above)
    - [ ] modify sampling procedure for non-diffused residues
- Metrics
    - [ ] update assess_sample() 
    - [ ] New sample metrics
    - [ ] New summary metrics
- Benchmarks
    - [ ] Take benchmark dataset + dataloader from FrameFlow
        - https://github.com/microsoft/protein-frame-flow/blob/main/experiments/utils.py#L45
- Helpers
    - [ ] make inference easier, e.g. to sample around some motif taken from a protein
- Misc
    - [ ] update relevant switch statements
    - [ ] update all `(diffuse_mask == 1.0).all()` references
    - [ ] update `twisting` reference for multiflow config check