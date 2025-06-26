The current standard way to assess the generated structures from protein generative models is to check the *designability*, i.e. to inverse fold the structure with a tool like *LigandMPNN* and re-fold it with a tool like *Boltz* or *AlphaFold*. 

Unfortunately, the two programs primarily used in this project - *LigandMPNN* and *Boltz2* - are meant to be run from the command line, not to hold the model in memory and run inference on demand.

This directory contains wrappers around 3rd party software to enable running inference more quickly, against models kept in memory. 

The code in this directory is messy because most of it is AI-generated.

`LigandMPNN` in particular is a mess, because it's `main` function is a monolith that had to be largely recreated to run inference on the model.

`Boltz` is less bad, but its `predict` function is still a long string of logic, which hides away many decisions (like where all the input and processed files go) that had to be re-created in the wrapper.