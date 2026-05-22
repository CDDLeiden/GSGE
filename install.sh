#!/bin/bash
# Installation script for GSGE package

set -e  # Exit on error

echo "================================================"
echo "GSGE Installation Script"
echo "================================================"

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create conda environment
echo ""
echo "Step 1: Creating conda environment 'gsge-dev'..."
conda env create -f environment.yml -y 2>/dev/null || {
    echo "Environment already exists, updating..."
    conda env update -f environment.yml -y
}

# Activate environment
echo ""
echo "Step 2: Activating environment..."
eval "$(conda shell.bash hook)"
conda activate gsge-dev

# PyTorch installation
echo ""
echo "Step 3: Installing PyTorch..."
echo ""
echo "Do you need GPU (CUDA) support?"
echo "  1) CPU only (default)"
echo "  2) CUDA 11.8"
echo "  3) CUDA 12.1"
echo "  4) CUDA 12.4"
echo "  5) CUDA 12.8"
echo "  6) CUDA 13.0"
echo ""
read -r -p "Select option [1-6] (default: 1): " cuda_choice

case "${cuda_choice:-1}" in
    1)
        echo "Installing PyTorch (CPU)..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
    2)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch --index-url https://download.pytorch.org/whl/cu121
        ;;
    4)
        echo "Installing PyTorch with CUDA 12.4..."
        pip install torch --index-url https://download.pytorch.org/whl/cu124
        ;;
    5)
        echo "Installing PyTorch with CUDA 12.8..."
        pip install torch --index-url https://download.pytorch.org/whl/cu128
        ;;
    6)
        echo "Installing PyTorch with CUDA 13.0..."
        pip install torch --index-url https://download.pytorch.org/whl/cu130
        ;;
    *)
        echo "Invalid option. Installing PyTorch (CPU)..."
        pip install torch --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# Install GSGE with all dependencies from pyproject.toml
echo ""
echo "Step 4: Installing GSGE with all dependencies..."
echo "This will install runtime, dev, docs, viz, and notebooks extras."
cd "$SCRIPT_DIR"
pip install -e ".[all]"

# Verify installation
echo ""
echo "Step 5: Verifying installation..."
python -c "import GSGE; print(f'GSGE imported successfully: {GSGE.__file__}')"

if command -v GSGE_CLI &> /dev/null; then
    echo "CLI command available: GSGE_CLI"
else
    echo "Note: GSGE_CLI command will be available after shell restart"
fi

echo ""
echo "================================================"
echo "Installation completed successfully!"
echo "================================================"
echo ""
echo "To use GSGE, activate the environment with:"
echo "  conda activate gsge-dev"
echo ""
