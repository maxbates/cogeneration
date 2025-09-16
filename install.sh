#!/usr/bin/env bash
set -euo pipefail

# install.sh - convenience script to set up Cogeneration with all optional
# dependencies and tools for training. Intended for running on a fresh
# remote instance.

# Determine repository root (directory containing this script)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_PARENT="$(dirname "$REPO_DIR")"
cd "$REPO_DIR"

if [[ ! -d "$REPO_DIR/cogeneration" ]]; then
    echo "Expected cogeneration directory at $REPO_DIR/cogeneration" >&2
    exit 1
fi

usage() {
    echo "Usage: $0 [--cpu] [--dev]"
    echo "  --cpu  Install CPU-only dependencies"
    echo "  --dev  Include development dependencies"
}

WITH_CUDA=1
EXTRA=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --cpu)
      WITH_CUDA=0
      shift
      ;;
    --dev)
      EXTRA=",dev"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 1
      ;;
  esac
done

# 0. Pre-install some dependencies for simplicity
# Install torch
if [[ "$WITH_CUDA" -eq 1 ]]; then
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
else
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# 1. Install Python package
if [[ "$WITH_CUDA" -eq 1 ]]; then
    # include torch-scatter wheel, so don't build
    pip install -e .[cu128${EXTRA}] -f https://data.pyg.org/whl/torch-2.7.0+cu128.html
else
    # include torch-scatter wheel, so don't build
    pip install -e .[cpu${EXTRA}] -f https://data.pyg.org/whl/torch-2.7.0+cpu.html
fi

# 2. Install ProteinMPNN / LigandMPNN
cd $REPO_PARENT
if [[ ! -d ProteinMPNN ]]; then
    git clone https://github.com/dauparas/ProteinMPNN.git
fi
if [[ ! -d LigandMPNN ]]; then
    git clone https://github.com/dauparas/LigandMPNN.git
fi

# Download LigandMPNN weights if they are missing
if [[ ! -f LigandMPNN/proteinmpnn_v_48_020.pt ]]; then
    wget -q https://files.ipd.uw.edu/pub/ligandmpnn/proteinmpnn_v_48_020.pt -O LigandMPNN/proteinmpnn_v_48_020.pt
fi
if [[ ! -f LigandMPNN/ligandmpnn_v_32_010_25.pt ]]; then
    wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_v_32_010_25.pt -O LigandMPNN/ligandmpnn_v_32_010_25.pt
fi
if [[ ! -f LigandMPNN/ligandmpnn_sc_v_32_002_16.pt ]]; then
    wget -q https://files.ipd.uw.edu/pub/ligandmpnn/ligandmpnn_sc_v_32_002_16.pt -O LigandMPNN/ligandmpnn_sc_v_32_002_16.pt
fi

# Add ProteinMPNN to PATH for convenience
if ! grep -q "ProteinMPNN" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/projects/ProteinMPNN:$PATH"' >> ~/.bashrc
fi

# 3. Install ColabFold using localcolabfold
cd ~/projects
if [[ ! -d localcolabfold ]]; then
    wget -q https://raw.githubusercontent.com/YoshitakaMo/localcolabfold/main/install_colabbatch_linux.sh
    chmod +x install_colabbatch_linux.sh
    bash install_colabbatch_linux.sh
fi
if ! grep -q "localcolabfold" ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/projects/localcolabfold/colabfold-conda/bin:$PATH"' >> ~/.bashrc
fi

# 4. Install ffmpeg for trajectory animations
if ! command -v ffmpeg >/dev/null 2>&1; then
    sudo apt-get update
    sudo apt-get install -y ffmpeg
fi

# 5. Install datasets
cd "$REPO_DIR"
bash cogeneration/datasets/install_multiflow_datasets.sh

# 6. Download public MultiFlow weights
if [[ ! -d multiflow_weights ]]; then
    wget https://zenodo.org/records/10714631/files/weights.tar.gz
    tar -xzf weights.tar.gz
    mv weights multiflow_weights
    rm weights.tar.gz
fi

# 7. Setup environment
# can safely set MPS env vars regardless of environment
set -a && source $REPO_DIR/cogeneration/.env.mps && set +a
if [[ "$WITH_CUDA" -eq 1 ]]; then
    set -a && source $REPO_DIR/cogeneration/.env.cuda && set +a
fi

# 8. Dummy boltz run, to trigger downloading of weights + mols
echo "Running Boltz to trigger weights + mols download"
echo ">dummy|protein|empty\nACGTDK\n" > boltz_dummy.fasta
boltz predict boltz_dummy.fasta
rm boltz_dummy.fasta
rm -r boltz_results_boltz_dummy

echo "\nSetup complete. You should run 'wandb login' before training."
