# Usage: export $(cat .env.mps)

# Environment variables for PyTorch Mac / MPS development
# These must be set BEFORE importing PyTorch

# Enable fallback to CPU for unsupported MPS operations
PYTORCH_ENABLE_MPS_FALLBACK=1

# Optimize MPS memory management
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# General PyTorch optimizations
OMP_NUM_THREADS=4
TOKENIZERS_PARALLELISM=false

# Optional: Enable MPS profiling (uncomment if needed)
# PYTORCH_MPS_PROFILING=1