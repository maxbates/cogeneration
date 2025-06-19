Notes for getting set up on Lambda Labs machine:

## Sync local to remote

set up ssh

**Option 1: Using rsync (recommended):**
```bash
# Sync from local to remote (run from local machine)
rsync -avz --filter=':- .gitignore' --exclude='.git' ./ username@remote_host:~/cogeneration/
```

**Option 2: Using PyCharm:**
Set up PyCharm remote deployment to ensure all source files are copied to remote.

```
cd cogeneration
```

## Install package + dependencies

Install package

**Option 1: CUDA installation (recommended for training on GPU machines):**
```
pip install -e .[cuda]
```

**Option 2: CPU-only installation:**
```
pip install -e .
```

**Option 3: Development installation with CUDA:**
```
pip install -e .[cuda,dev]
```

**Fallback**: If you encounter issues with the CUDA installation, you can install manually:
```
# Install PyTorch with CUDA first
pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 torchaudio==2.4.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install torch-scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html

# Install flash-attention (may require --no-build-isolation)
pip install flash-attn --no-build-isolation

# Install remaining dependencies
pip install -e .
```

## Install data, tools

install datasets
```
bash ./cogeneration/datasets/install.sh
```

for animations, install ffmpeg
```
sudo apt-get install ffmpeg
```

Install ProteinMPNN

```bash
# Clone ProteinMPNN to ~/tools directory
mkdir -p ~/tools
cd ~/tools
git clone https://github.com/dauparas/ProteinMPNN.git

# Install dependencies in current environment
pip install prody pyparsing==3.1.1

# Add to PATH
echo 'export PATH="$HOME/tools/ProteinMPNN:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

Install Colabfold using localcolabfold

```bash
# Install to ~/tools/localcolabfold
cd ~/tools
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
chmod +x install_colabbatch_linux.sh
bash install_colabbatch_linux.sh

# Add to PATH
echo 'export PATH="$HOME/tools/localcolabfold/colabfold-conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Training

setup wandb
```
wandb login
```

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

## Sampling

To sample from a trained model, you can use the `predict.py` script.

All configuration, including checkpoints, number of samples, lengths etc. are specified in the `Config`

```
python cogeneration/scripts/predict.py --output_dir samples
```


## Data Pipeline

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

## Troubleshooting

If you still have an issue at run-time with torch scatter, install and build without caching:

```
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html --no-cache-dir
```
