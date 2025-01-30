This project is heavily based on `multiflow`, which introduced sequence-structure co-design with discrete flow matching:

- https://arxiv.org/pdf/2402.04997
- https://github.com/jasonkyuyim/multiflow

`FrameFlow` introduced SE(3) flow matching for protein structures:

- https://arxiv.org/abs/2310.05297
- https://github.com/microsoft/protein-frame-flow
- 
and an updated version of `FrameFlow` demonstrated scaffolding (inpainting):

- https://arxiv.org/abs/2401.04082
- https://github.com/microsoft/protein-frame-flow

`Discrete Flow Matching` from Meta extends the approach in `MultiFlow` with example results for larger models and comparing to LLMs:

- https://arxiv.org/abs/2407.15595
- https://github.com/gle-bellier/discrete-fm (unofficial)

The idea to use `ESM` as the sequence embedder comes from `FoldFlow-2`:

- https://arxiv.org/abs/2405.20313
- https://github.com/DreamFold/FoldFlow

`ESM` is now a company, originally from Meta's FAIR lab:

- https://github.com/facebookresearch/esm

`Stochastic Flow Matching` is discussed in many places, such as the `Stochastic Interpolants` paper:

- https://arxiv.org/abs/2303.08797

and applied to SE(3) for protein structures in a `FoldFlow` paper:

- https://arxiv.org/abs/2310.02391