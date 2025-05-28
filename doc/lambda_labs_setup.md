## Notes for getting set up on Lambda Labs machine:

Sync local to remote - set up ssh.

e.g. using PyCharm - set up Pycharm remote deployment.
Ensure all source files are copied to remote.

```
cd cogeneration
```

install datasets
```
bash ./cogeneration/datasets/install.sh
```

install pytorch etc. (required special install)
```
pip install torch==2.4.1 torchaudio==2.4.1 torchvision
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
```

install other dependencies
```
pip install -r requirements.txt
```

install cogeneration package as editable
```
pip install -e .
```

setup wandb
```
wandb login
```

for animations, install ffmpeg
```
sudo apt-get install ffmpeg
```

TODO install ProteinMPNN and colabfold (localcolabfold?)

### Data Pipeline

If you want to download PDB and process (or reprocess) it to support new metadata or structure processing etc.

Both are long running processes (~1 hr each) but support resuming. 

```
python cogeneration/dataset/scripts/download_pdb.py --pdb_dir pdbs

python cogeneration/dataset/scripts/process_pdb_files.py --pdb_dir pdbs --output_dir processed_pdbs
```

#### Inverse Fold to Redesign Structures

You can also generate redesigned sequences using ProteinMPNN. 

This takes a very long time. You should merge with the ones provided by Multiflow. See notes in script.

```
python cogeneration/dataset/scripts/redesign_structures.py --pdb_dir processed_pdbs --output_dir redesigned_pdbs
```

### Training

Simple training with logging
```
python cogeneration/scripts/train.py 2>&1 | tee train.log
```

view results at [wandb.ai](https://wandb.ai/)

#### Curricula

You can also specify a `Curriculum` to train a series of checkpoints with different configurations.

Each `Curriculum` takes its own `Config`, which is specified in python.

#### Copy desired results

note the timestamp (`shared.now`, e.g. "20250130_204335") and W&B run id (e.g. `maxbates-org/cogeneration/6jo9519l`) 

copy log to local:
```
scp ubuntu@lambda_labs_tester:/home/ubuntu/cogeneration/train.log train.log
```

copy checkpoints all to local:
```
scp -r ubuntu@lambda_labs_tester:/home/ubuntu/cogeneration/ckpt ckpt
```

copy a specific checkpoint:

```
CHECKPOINT="hallucination_pdb_20250130_204335/20250130_204335"
mkdir -p "ckpt/cogeneration/${CHECKPOINT}"
scp -r ubuntu@lambda_labs_tester:"/home/ubuntu/cogeneration/ckpt/cogeneration/${CHECKPOINT}/" "ckpt/cogeneration/${CHECKPOINT}/"
```

### Sampling

To sample from a trained model, you can use the `predict.py` script.

All configuration, including checkpoints, number of samples, lengths etc. are specified in the `Config`

```
python cogeneration/scripts/predict.py --output_dir samples
```

### Troubleshooting

If you still have an issue at run-time with torch scatter, install and build without caching:

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html --no-cache-dir
```