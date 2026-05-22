# Contributing to GSGE

Thank you for your interest in contributing to GSGE (Group-SELFIES Graph Embeddings)! This document provides guidelines and instructions for contributors.

## Table of Contents

- [Development Setup](#development-setup)
- [Code Contributions](#code-contributions)
- [Testing](#testing)
- [Code Style](#code-style)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Conda (Miniconda or Anaconda)
- Git

### Quick Setup (Recommended)

The fastest way to set up a development environment is using the automated installation script:

```bash
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
bash install.sh
```

This script will:
1. Create a dedicated conda environment named `gsge-dev`
2. Install all required dependencies (including optional ones)
3. Install the `group-selfies` fork from GitHub
4. Install GSGE in **editable mode** (changes to code are immediately reflected)

After installation, activate the environment:
```bash
conda activate gsge-dev
```

### Manual Setup

If you prefer to set up manually or need more control:

#### 1. Clone the Repository

```bash
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
```

#### 2. Create Conda Environment

```bash
conda env create -f environment.yml
conda activate gsge-dev
```

#### 3. Install Dependencies

```bash
# Main dependencies
pip install tqdm rdkit "numpy>=1.26.4,<2.0" Pillow torch torch_geometric joblib selfies pyarrow pandas scikit-learn scipy

# Optional dependencies for development
pip install plotly seaborn matplotlib ipykernel nbformat ipywidgets
```

#### 4. Install group-selfies Fork

```bash
cd /tmp
git clone https://github.com/JasperDurinck/group-selfies
cd group-selfies
pip install .
cd -
```

#### 5. Install GSGE in Editable Mode

```bash
pip install -e ".[dev,viz,notebooks]"
```

This installs GSGE with all optional dependencies, allowing you to modify the code and see changes immediately.

### Windows Setup

On Windows, use the manual setup process:

```cmd
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
conda env create -f environment.yml
conda activate gsge-dev
pip install -e ".[dev,viz,notebooks]"
```

Note: The `install.sh` script requires Git Bash or WSL. If you have WSL installed, you can use the quick setup:

```bash
# In WSL
bash install.sh
```

### Verify Your Setup

```bash
# Test imports
python -c "import GSGE; from GSGE import GS_Vocab, GSGE_Corpus; print('Setup successful!')"

# Run tests
GSGE_CLI run_test
```

## Code Contributions

### Architecture Overview

GSGE follows a **Facade pattern** with modular class decomposition. 
Before contributing, please review [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture documentation.

Key modules:
- `gsge.py` - Main facade class with specialized managers
- `vocab.py` - Vocabulary and corpus management
- `tokenizer.py` - Tokenization logic
- `embedding.py` - Embedding management
- `graphs/` - Graph autoencoder and compound graph implementations

### Coding Guidelines

1. **Follow the existing patterns**: GSGE delegates functionality to specialized manager classes rather than adding everything to the main class.

2. **Use type hints**: New code should include type hints for function parameters and return values.

3. **Add docstrings**: All public functions and classes should have clear docstrings.

4. **Maintain compatibility**: The package requires `numpy<2.0` for RDKit compatibility. Ensure your changes don't break this.

5. **Parallel processing**: When adding batch operations, follow the existing parallel processing patterns (see `_parallel_tokenize_df` in `core_gsge.py`).

## Testing

### Running Tests

Run all tests:
```bash
GSGE_CLI run_test
```

Run a specific test file:
```bash
GSGE_CLI run_test --file test_make_cg.py
```

### Available Tests

- `test_lookup_table_emb_layer.py` - Embedding layer tests
- `test_make_gsge_corpus.py` - Corpus creation tests
- `test_make_cg.py` - Compound graph tests
- `test_make_gsge_vocab.py` - Vocabulary building tests
- `test_gsge_tokenization.py` - Tokenization tests

### Adding New Tests

When adding new features:
1. Create a new test file in `tests/` following the naming convention `test_*.py`
2. Include any required test data as `.pkl` files in `tests/`
3. Update the test file list in the CLI if necessary

## Code Style

### Python Style

- Follow PEP 8 guidelines
- Use meaningful variable names
- Keep functions focused and concise
- Prefer composition over inheritance

### Import Organization

```python
# Standard library
import os
import sys

# Third-party packages
import numpy as np
import torch
from rdkit import Chem

# Local imports
from .vocab import GS_Vocab
from .core_gsge import CoreGSGE
```

### Docstring Format

```python
def my_function(param1: str, param2: int) -> bool:
    """
    Brief description of what the function does.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Description of return value

    Example:
        >>> my_function("test", 42)
        True
    """
    pass
```

## Pull Request Process

1. **Fork the repository** and create a new branch from `dev`:
   ```bash
   git checkout -b feature/your-feature-name dev
   ```

2. **Make your changes**:
   - Write clear, concise commit messages
   - Keep commits focused and atomic
   - Add tests for new functionality

3. **Test your changes**:
   ```bash
   GSGE_CLI run_test
   ```

4. **Update documentation**:
   - Update README.md if you've changed installation or usage
   - Update CLAUDE.md if you've changed architecture
   - Add docstrings to new functions/classes

5. **Submit a pull request**:
   - Target the `dev` branch (not `main`)
   - Provide a clear description of your changes
   - Reference any related issues
   - Wait for review and address feedback

### Commit Message Format

```
type: Brief description (max 72 chars)

More detailed explanation if necessary. Explain what and why,
not how (the code shows how).

Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

## Reporting Issues

### Bug Reports

When reporting bugs, please include:

1. **Environment details**:
   - Python version
   - GSGE version
   - Operating system
   - Relevant package versions (`numpy`, `rdkit`, `torch`, etc.)

2. **Steps to reproduce**:
   - Minimal code example
   - Input data (if applicable)
   - Expected vs. actual behavior

3. **Error messages**:
   - Full traceback
   - Any warning messages

### Feature Requests

For feature requests:
1. Describe the use case
2. Explain why existing functionality doesn't suffice
3. Suggest a possible implementation (if you have one)

### Questions

For questions about usage:
- Check [ARCHITECTURE.md](ARCHITECTURE.md) for architecture details
- Review the [installation guide](docs/getting-started/installation.md) for installation issues
- Check existing issues on GitHub
- Look at `use_examples/` for usage examples

If your question isn't answered, open a GitHub issue with the "question" label.

## Development Workflow

### Branching Strategy

- `main` - Stable releases only
- `dev` - Active development (target for PRs)
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates

### Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update CHANGELOG (if applicable)
3. Merge `dev` into `main`
4. Tag the release
5. Push to PyPI (when ready)

## Getting Help

- **Issues**: https://github.com/CDDLeiden/GSGE/issues
- **Email maintainers**:
  - Bola Khalil: b.a.a.khalil@lacdr.leidenuniv.nl
  - Jasper Durinck: jasper.j.durinck@gmail.com

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to advance molecular representation learning!

## License

By contributing to GSGE, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to GSGE! 🚀
