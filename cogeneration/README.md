# Notes

## OpenFold

This repo uses Openfold, but copies its source in many places. 

This is primarily because OpenFold requires building kernels on install, which requires an Nvidia GPU. 
The MSA transformer etc. are not necessary for the model in this repo. 

This should simplify install, testing, etc.