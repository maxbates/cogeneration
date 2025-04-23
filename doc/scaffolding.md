doc to plan for scaffolding feature

Review FrameFlow (second version), which enables scaffolding within a similar codebase (just for protein scaffolding)
https://github.com/microsoft/protein-frame-flow

## Initial Version

- Pre-work
    - [x] see frameflow 2 https://github.com/microsoft/protein-frame-flow/commit/20293ab04994a5bfe97f2d8cf8603a0b63af0609
- Theory
    - [x] consider how scaffolding will work with logits, in both scaffolding regimes.
- Codebase
    - [x] new training task `scaffolding`
    - [x] inference task `scaffolding`
    - [x] scaffolding type: `motif amortization` (fixed / conditional)
    - [x] ensure `forward_folding` and `inverse_folding` defined correctly, cannot just index on rotmats_1/trans_1 being defined any more.
        - move to switch statements
    - [x] determine how we switch between unconditional generation and scaffolding
        - e.g. support switching behavior if `diffuse_mask` is defined to support unconditional generation, with some fraction of batches each
    - [x] cfg option `inpainting_percent`, defaults to unconditional otherwise
- Training Dataset
    - [x] factory to generate motif masks given a batch
        - [x] initially, just simple motifs
        - [x] motif factory config: window size, number motifs, motif size
    - [x] training dataset, or modify interpolant to generate motifs, which defines `diffuse_mask`
- Interpolant
    - [x] update `corrupt_batch()` 
        - [x] pass `diffuse_mask` to `corrupt_trans` and `corrupt_rotmats` and `corrupt_aatype`
            - see FrameFlow2 https://github.com/microsoft/protein-frame-flow/commit/f50d8dbbdae827be291e9f73d732b61b195f8816#diff-fa71335c2193d39db0b87deb7b1e8ba0943d147e774dc46af54a34c1208cab6f
    - [x] need to center motif.
        - See if in frameflow, the motif is centered, and the noised residues are also centered    
    - [x] update OT 
        - [x] OT needs to respect `diffuse_mask`    
        - [x] need to fix centering, currently part of OT. need to center everything, not just in diffuse_mask
    - [x] update `sample()` 
        - [x] partial `diffuse_mask`. require trans_1 and rotmats_1 defined.
        - [x] update centering procedure, if needed
            - needs to match what happens in training
        - [x] update self-conditioning to use `diffuse_mask`
- Training
    - [x] log scaffolding metrics like scaffolding percent, motif size, `diffuse_mask.mean()`
- Eval Dataset
    - [x] new eval dataset
        - [x] Modify scaffold lengths of masked PDB structures. Remove t=1 values from scaffold.
        - see `proteina` MotifFactory https://github.com/NVIDIA-Digital-Bio/proteina/blob/main/proteinfoundation/nn/motif_factory.py#L450
- Sampling
    - [x] modify sampling procedure for non-diffused residues i.e. `diffuse_mask`
    - [x] update predict.py with scaffolding dataset 
    - [x] determine if should fix motif at final time step, depending on metrics calculated
        - e.g. we might want to calculate RMSD of motifs, look for breaks, etc.
- Module
    - [x] inpainting `validation_step()` shouldn't only default to unconditional generation
    - [x] need to align with eval dataset? or just generate motifs from PDB validation dataset?
- Metrics
    - [x] support metrics for inpainting
    - [x] take scaffolding metrics from FrameFlow
        - There arent really any present
- Folding Validation
    - [x] update `assess_sample` 
        - Inpainting specific metrics. account for `diffuse_mask` appropriately.
        - [x] sequence recovery of motifs
        - [x] RMSD of motifs
- Visualizing
    - [x] update `save_trajectory()` for coloring logits - right now they are all green
    - [x] support b-factors by `diffuse_mask`

- Centering - move to interpolating rather than fixed
    - Notes
        - Fundamental tension if fixed: motifs and scaffolds will have a different center of mass.
            - We can be invariant to the motifs, keeping them fixed, and learn the drift of the scaffolds => conditional invariance
            - We can be invariant to everything, but the motifs will be "pushed" off-center and introduce bias and blurs conditioning
            - **If we interpolate scaffold over time, centering is much easier! Just center everything.**
        - Centering in FrameFlow
            - t=1 (and t=0) motifs are centered by the dataset
            - amortization (fixed motifs) there is no additional centering
                - want condition to be fixed... may lead to off-center structure... I guess that's ok?
                - would it make more sense to center at each intermediate time points
            - guidance (interpolation) uses twisting to align diffused residues
                - makes sense, "looks like" unconditional generation as residues move over time with diffuse residues.
        - Centering in FoldFlow2
            - FoldFlow2 samples in reverse, i.e. model predicts vectorfield and data sample is corrupted to yield trajectory
            - `center` in `reverse()` at each time step
            - in a presentation, they say they iterated on centering procedure, but didnt describe it...
            - Unclear if actually includes scaffolding in codebase...
            - Looks like it: 
                - takes `trans_t` and `v_t` and `flow_mask`
                - generates a `perturb`, adds noise, masks it 
                - applies perturb to `trans_t`
                - centers `trans_t` using COM of only residues in `flow_mask`
                - updates `trans_t_1` at `flow_mask` positions
                - ... which seems like is fine if `flow_mask` is everything 
                  but will recenter scaffolds — and not move motifs — and merge them
            - in their images, the motifs seem to interpolate over time...
    - [x] Training
        - [x] Interpolate motifs rather than just masking
        - [x] Interpolate rotations as well 
        - [x] add noise to motif positions (i.e. all positions)
        - [x] Center everything after adding noise 
        - Still predict changes for every residue
        - [x] For amino acids, keep them fixed
            - In future, later can start as masks and mess with logits. 
        - [x] Fix loss calculation
        - [x] Remove motif centering in dataset
    - [x] Sampling
        - [x] Center after adding noise
        - [x] Interpolate motifs over time, rather than just masking
        - [x] add noise to all residues
        - [x] Final time step, fix motifs
        - [x] For amino acids, keep them fixed

- Folding Validation
    - [x] confirm existing metrics make sense for inpainting
        - [x] masks used appropriately
        - [x] true sequence and bb positions used sanely
    - [x] Update expectations that only doing unconditional `validation_step()`
        - [x] pass true_bb_positions and true_aa when appropriate in `validation_step()`
        - [x] update `is_codesign` check in `assess_sample()`
    - [x] Update metrics to support when inpainting actually run like unconditional
        - Or, just override the task to unconditional if full diffuse_mask

- Misc / Backlog
    - [x] update relevant task switch statements
    - [x] ensure `diffuse_mask` is obeyed throughout
    - [x] update all `(diffuse_mask == 1.0).all()` references
    - [x] address `TODO(inpainting)`
    - [x] confirm masks used appropriately for metrics calculating: `res_mask`, `diffuse_mask`, `plddt_mask`

- Tests
    - [x] ScaffoldingDataset - diffuse_mask, motif sizes
    - [x] interpolant - corrupt_batch() 
    - [x] interpolant - sample()
    - [x] inference for inpainting works 
    - [-] evalrunner test for inpainting -- unnecessary
    - [x] metrics for inpainting

## Future Work

- [ ] Do `forward_folding` (full sequence conditioning) some percentage of the time

- Motif Selection
    - [x] set up to allow other methods. abstract into class.
    - [x] allow trimming low pLDDT ends before selection
    - [x] interacting residues or potential active sites
    - [x] Allow selecting a residue and then removing / keeping residues within some distance
    - [ ] cross-chain interactions
        - can we augment dataset looking for large monomers with interacting motifs?
    - [ ] based on secondary structure? e.g. enrich for beta sheets

- Amino acid corruption
    - [ ] Currently, sequence is fixed, i.e. whole process is motif-sequence-conditioned
    - [ ] If e.g. we worked in logits, it would be possible to interpolate in a clear way
    - [ ] **May make sense to introduce another task to differentiate sequence-conditioned interpolation vs scaffolding given fixed motifs + sequence**

- Training
    - [ ] define a training curriculum
        - train on small monomers + scaffolding, then larger + complexes, preferring infilling interacting domains.
    
- Inference
    - [ ] make inference easier, e.g. to sample around some motif taken from a protein
    - [x] support parsing RFDiffusion style contigmaps. See FrameFlow.
    - [ ] script to take contigmap and sample
    
- Metrics
    - [ ] secondary structure of scaffolds, clashes in scaffolds
    - [ ] novelty - foldseek against PDB (cluster designs first)

- Time embeddings
    - [ ] set time to ~t=1 (e.g. t = 1-1e-5) in embedding sequence or structure  ?

- Benchmarks
    - [ ] Take benchmark dataset + dataloader from FrameFlow
        - https://github.com/microsoft/protein-frame-flow/blob/main/experiments/utils.py#L45
    - [ ] look at MotifBench
    
- Scaffolding modeling Features
    - [ ] Option to only fix structure, not sequence
        - [ ] Add metrics for sequence recovery
    - [ ] alternative scaffolding method: `motif guidance` (move over time)
        - [ ] update `rotmats1` and `trans1` to support interpolation for motifc guidance, not just fixed
    - [ ] enable `twisting` for motif guidance sampling like FrameFlow
        - see https://github.com/microsoft/protein-frame-flow/blob/main/data/interpolant.py#L327
        - [ ] update `twisting` reference for multiflow config check. This is currently how we differentiate it.