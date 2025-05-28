# Cogeneration

Cogeneration is a protein generative model that simultaneously generates protein sequences and structures.

This project is based on MultiFlow, but introduces several extensions over the original:
- support for multimers
- support for inpainting (conditional generation)
- stochastic paths, for the structure and sequence
- model architecture changes:
    - support for existing protein language models (e.g. ESM) to get frozen embeddings
    - options to use a deeper sequence prediction network
    - side chain prediction for more accurate side chain placement
    - additional losses (e.g. for atomic interactions, clashes)
- data pipeline to generate or augment training data
- many improvements to code base: typing, enums, documentation, tests, etc.

### Attribution

This project is based on MultiFlow:

```
@article{campbell2024generative,
  title={Generative Flows on Discrete State-Spaces: Enabling Multimodal Flows with Applications to Protein Co-Design},
  author={Campbell, Andrew and Yim, Jason and Barzilay, Regina and Rainforth, Tom and Jaakkola, Tommi},
  journal={arXiv preprint arXiv:2402.04997},
  year={2024}
}
```