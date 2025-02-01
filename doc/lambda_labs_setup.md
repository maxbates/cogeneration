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

### Training

train with logging
```
python cogeneration/scripts/train.py 2>&1 | tee train.log
```

view results at [wandb.ai](https://wandb.ai/)

#### Copy desired results

note the timestamp (`metadata.now`, e.g. "20250130_204335") and W&B run id (e.g. `maxbates-org/cogeneration/6jo9519l`) 

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

### Troubleshooting

If you still have an issue at run-time with torch scatter, install and build without caching:

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html --no-cache-dir
```