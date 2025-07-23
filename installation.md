Notes for getting set up on local or remote machine:

See also `install.sh`, which automates most steps in this file (assumes repo has been downloaded/copied).

## (If Remote) sync local to remote

set up ssh

**Option 1: Using rsync (recommended):**

Run from local machine:

```bash
# Sync from local `./` (cogeneration directory) to remote ~/cogeneration
rsync -avz --filter=':- .gitignore' --exclude='.git' --exclude '*.tar' ./ username@remote_host:~/cogeneration/
```

**Option 2: Using PyCharm:**
Set up PyCharm remote deployment to ensure all source files are copied to remote.

## (If required) Set up venv / conda on remote machine

```
# download mamba / conda
wget -O Mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/25.3.0-3/Miniforge3-25.3.0-3-Linux-x86_64.sh
bash Mambaforge.sh -b -p $HOME/mambaforge
source ~/mambaforge/bin/activate

# create env and activate
mamba create -n cogeneration python=3.12 pip -c conda-forge
mamba activate cogeneration
```

## Install package

```
cd cogeneration
```

**Option 1: CUDA 12.8 installation (for training on GPU machines):**
```
pip install -e .[cu128]
```

**Option 2: CPU-only installation:**
```
pip install -e .[cpu]
```

**Option 3: Development installation with CUDA:**
```
pip install -e .[cu128,dev]
```

**Debugging CUDA Setup**

If you encounter issues with the CUDA installation, you may need to update `setup.py` to match your machine. There are configuration variables at the top for CUDA version that must match your machine, and may limit / impact the appropriate versions for other torch packages. 

Otherwise, you can install manually
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

## Install data, 3rd party tools

### Install ProteinMPNN / LigandMPNN

```bash
# Clone ligandMPNN (for local) ProteinMPNN (for subprocess) to ~/projects directory as siblings of cogeneration
mkdir -p ~/projects
cd ~/projects
git clone https://github.com/dauparas/ProteinMPNN.git
git clone https://github.com/dauparas/LigandMPNN.git

# Download ligandMPNN weights to run natively (much faster)
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt -O "~/projects/LigandMPNN/proteinmpnn_v_48_020.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt -O "~/projects/LigandMPNN/ligandmpnn_v_32_010_25.pt"
wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt -O "~/projects/LigandMPNN/ligandmpnn_sc_v_32_002_16.pt"

# Install dependencies in current environment
pip install prody pyparsing==3.1.1

# Add to PATH
echo 'export PATH="$HOME/projects/ProteinMPNN:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### Install Colabfold using localcolabfold

```bash
# Install to ~/projects/localcolabfold
cd ~/projects
wget https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
chmod +x install_colabbatch_linux.sh
bash install_colabbatch_linux.sh

# Add to PATH
echo 'export PATH="$HOME/projects/localcolabfold/colabfold-conda/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### for animations, install ffmpeg

Animations are enabled by default when trajectories are saved.

```
sudo apt-get install ffmpeg
```

### Install datasets

Datasets are required for training and evaluation, or provide a base set of structures for inpainting.

```
bash ./cogeneration/datasets/install_multiflow_datasets.sh
```

### To run Public Multiflow, download weights

Using the `public_multiflow` config, the model will match the public MultiFlow model, and its weights can be used for inference / fine-tuning.

```
# Get multiflow weights
wget https://zenodo.org/records/10714631/files/weights.tar.gz

# Uncompress weights
# creates a directory `weights` containing `config.yaml` and `last.ckpt`
tar -xzf weights.tar.gz

# move weights to `multiflow_weights` at project root (may need to adjust rel path here)
mv weights multiflow_weights

# Clean up weight tarball
rm weights.tar.gz 
```

## Training

setup wandb
```
wandb login
```

Load CUDA environment variables
```
set -a  # automatically export all variables
source "$(dirname "$0")/.env.cuda"
set +a
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

These long running processes (>= 1 hr each) but some (PDB, processing) support resuming. 

Using the default paths, which install into `~/pdb`:

```
python cogeneration/dataset/scripts/download_pdb.py
python cogeneration/dataset/scripts/process_pdb_files.py 

python cogeneration/dataset/scripts/download_alphafold.py
python cogeneration/dataset/scripts/process_pdb_files.py --alphafold

python cogeneration/dataset/scripts/cluster_pdbs.py
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

If encounter SSL issues downloading weights, try:
```
python -m pip install --upgrade certifi
bash /Applications/Python\\ 3.12/Install\\ Certificates.command
```