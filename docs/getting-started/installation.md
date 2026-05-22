# Installation

This guide covers all installation methods for GSGE (Group-SELFIES Graph Embeddings).

## System Requirements

- **Python**: 3.10, 3.11, or 3.12 (tested primarily on 3.12)
- **Operating System**: Linux, macOS, or Windows with WSL
- **GPU** (optional): CUDA-compatible GPU recommended for GAE training (CPU works but is slower)

## Quick Install (Users)

Install GSGE in your existing Python environment:

```bash
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
pip install .
```

This installs all required dependencies, including the required fork of `group-selfies` from GitHub.

!!! note "PyTorch and GPU support"
    `pip install .` installs PyTorch from PyPI, which is **CPU-only by default**. If you need GPU acceleration, install PyTorch with CUDA **before** installing GSGE. See [PyTorch & CUDA](#pytorch-cuda) below.

For optional features:

```bash
pip install ".[viz]"         # Visualization (plotly, umap-learn, etc.)
pip install ".[notebooks]"   # Jupyter notebook support
pip install ".[viz,notebooks]"  # Both
```

### Verify Installation

```bash
# Test imports
python -c "import GSGE; from GSGE import GS_Vocab, GSGE_Corpus; print('Installation successful!')"

# Test CLI
GSGE_CLI run_test --help
```

## Developer Install

For contributors who need a complete development environment with testing, docs, and visualization tools:

```bash
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
bash install.sh
```

This script will:

1. Create a `gsge-dev` conda environment (from `environment.yml`)
2. Install all required and optional dependencies
3. Install `group-selfies` from GitHub
4. Install GSGE in **editable mode** (`pip install -e ".[all]"`)

After installation, activate the environment:

```bash
conda activate gsge-dev
```

### Manual Developer Setup

If you prefer manual control:

```bash
conda env create -f environment.yml
conda activate gsge-dev
pip install -e ".[all]"  # Installs dev, docs, viz, and notebooks extras
```

## PyTorch & CUDA

By default, `pip install` fetches PyTorch from PyPI which provides CPU-only builds. To use GPU acceleration, install PyTorch with CUDA **before** installing GSGE:

```bash
# For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch --index-url https://download.pytorch.org/whl/cu124

# For CUDA 12.8
pip install torch --index-url https://download.pytorch.org/whl/cu128

# For CUDA 13.0
pip install torch --index-url https://download.pytorch.org/whl/cu130
```

Then install GSGE (it will use the already-installed PyTorch):

```bash
pip install .
```

For the latest install commands for your specific platform and CUDA version, see the [PyTorch Get Started page](https://pytorch.org/get-started/locally/).

### Verify GPU Support

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Troubleshooting

### NumPy Version Issues

GSGE requires `numpy<=2.3.0` for RDKit compatibility. If you encounter version conflicts:

```bash
pip install "numpy<=2.3.0" --force-reinstall
```

### group-selfies Installation Issues

If the group-selfies installation fails during `pip install .`:

1. Ensure you have git installed
2. Check your internet connection
3. Install it manually first:

```bash
git clone https://github.com/JasperDurinck/group-selfies
cd group-selfies
pip install .
cd ..
```

### CUDA Not Detected

If `torch.cuda.is_available()` returns `False`:

1. Verify you have CUDA toolkit installed: `nvidia-smi`
2. Check you installed the correct PyTorch CUDA build (see [PyTorch & CUDA](#pytorch-cuda))
3. Ensure your CUDA version matches the PyTorch build

### Import Errors

```bash
# Verify GSGE is installed
pip show GSGE

# Reinstall in editable mode
pip install -e .
```

## Testing Your Installation

```bash
# Run all tests
GSGE_CLI run_test

# Run specific test
GSGE_CLI run_test --file test_make_cg.py

# Using pytest directly
pytest tests/
```

## Upgrading

```bash
cd /path/to/GSGE
git pull origin main
pip install -e . --upgrade
```

## Uninstallation

```bash
pip uninstall GSGE

# If using conda environment
conda env remove -n gsge-dev
```

## Next Steps

After successful installation:

- Follow the [Quick Start Guide](quickstart.md) to create your first vocabulary
- Explore [User Guide](../user-guide/index.md) for detailed tutorials
- Check out [example notebooks](../tutorials/index.md) for real-world use cases
