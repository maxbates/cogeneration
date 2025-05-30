# Cogeneration

Cogeneration is a protein generative model that simultaneously generates protein sequences and structures.

This project is based on MultiFlow, but introduces several extensions over the original:
- support for multimers
- inpainting (conditional generation) given partial structures
    - MultiFlow supports per-domain conditioning via seperate t (i.e. folding and inverse folding)
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

And uses code from OpenFold:
```
@article {Ahdritz2022.11.20.517210,
	author = {Ahdritz, Gustaf and Bouatta, Nazim and Floristean, Christina and Kadyan, Sachin and Xia, Qinghui and Gerecke, William and O{\textquoteright}Donnell, Timothy J and Berenberg, Daniel and Fisk, Ian and Zanichelli, Niccolò and Zhang, Bo and Nowaczynski, Arkadiusz and Wang, Bei and Stepniewska-Dziubinska, Marta M and Zhang, Shang and Ojewole, Adegoke and Guney, Murat Efe and Biderman, Stella and Watkins, Andrew M and Ra, Stephen and Lorenzo, Pablo Ribalta and Nivon, Lucas and Weitzner, Brian and Ban, Yih-En Andrew and Sorger, Peter K and Mostaque, Emad and Zhang, Zhao and Bonneau, Richard and AlQuraishi, Mohammed},
	title = {{O}pen{F}old: {R}etraining {A}lpha{F}old2 yields new insights into its learning mechanisms and capacity for generalization},
	elocation-id = {2022.11.20.517210},
	year = {2022},
	doi = {10.1101/2022.11.20.517210},
	publisher = {Cold Spring Harbor Laboratory},
	URL = {https://www.biorxiv.org/content/10.1101/2022.11.20.517210},
	eprint = {https://www.biorxiv.org/content/early/2022/11/22/2022.11.20.517210.full.pdf},
	journal = {bioRxiv}
}
```

And benefits from the following works:

FoldFlow-2: https://github.com/DreamFold/FoldFlow
FrameFlow: https://github.com/microsoft/protein-frame-flow
ESM: https://github.com/facebookresearch/esm
RFDiffusion: https://github.com/RosettaCommons/RFdiffusion
Boltz: https://github.com/jwohlwend/boltz
ProteinMPNN: https://github.com/dauparas/ProteinMPNN
AlphaFold2: https://github.com/google-deepmind/alphafold