# Contributing to GSGE

Thank you for your interest in contributing to GSGE! This guide will help you get started with development.

## Quick Start for Contributors

```bash
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
bash install.sh  # Sets up gsge-dev conda environment
conda activate gsge-dev
```

## Development Setup

### Prerequisites

- Python 3.10, 3.11, or 3.12
- Conda (Miniconda or Anaconda)
- Git

### Automated Setup (Recommended)

The `install.sh` script creates a complete development environment:

```bash
bash install.sh
```

This will:

1. Create `gsge-dev` conda environment
2. Install all dependencies (including dev tools)
3. Install group-selfies fork from GitHub
4. Install GSGE in editable mode

### Manual Setup

For more control over the installation:

#### 1. Clone Repository

```bash
git clone https://github.com/CDDLeiden/GSGE
cd GSGE
```

#### 2. Create Environment

```bash
conda env create -f environment.yml
conda activate gsge-dev
```

#### 3. Install Dependencies

```bash
# Core dependencies
pip install tqdm rdkit "numpy<=2.3.0" Pillow torch torch_geometric \
    joblib selfies pyarrow pandas scikit-learn scipy

# Development dependencies
pip install pytest pytest-cov

# Optional visualization
pip install plotly seaborn matplotlib

# Optional notebook support
pip install ipykernel nbformat ipywidgets
```

#### 4. Install group-selfies Fork

!!! warning "Required Fork"
    GSGE requires a forked version of group-selfies with critical fixes.

```bash
git clone https://github.com/JasperDurinck/group-selfies
cd group-selfies
pip install .
cd ..
```

#### 5. Install GSGE in Editable Mode

```bash
pip install -e ".[dev,viz,notebooks]"
```

### Verify Setup

```bash
# Test imports
python -c "import GSGE; print('✓ Setup successful!')"

# Run test suite
pytest tests/

# Or use CLI
GSGE_CLI run_test
```

## Project Architecture

GSGE follows a **Facade pattern** with modular class decomposition. Key architectural principles:

- **Main facade**: `GSGE` class delegates to specialized managers
- **Manager classes**: Handle specific functionality (vocabulary, embeddings, GAE training, etc.)
- **Static utilities**: `CoreGSGE` provides stateless helper functions
- **Parallel processing**: Batch operations use multiprocessing

### Module Structure

```
GSGE/
├── gsge.py                 # Main facade + managers
├── core_gsge.py            # Static utilities
├── vocab.py                # GS_Vocab, GSGE_Corpus
├── tokenizer.py            # GSGE_tokenizer
├── embedding.py            # EmbeddingManager
├── clustering.py           # GSGE_clustering
├── fragment_descriptors.py # Descriptor calculation
├── plots.py                # Visualization
├── graphs/
│   ├── fragment_graph/     # GAE components
│   └── compound_graph/     # Compound graph class
└── tests/                  # Test suite
```

## Coding Guidelines

### General Principles

1. **Follow existing patterns**: Delegate to manager classes, don't bloat the main facade
2. **Use type hints**: All public APIs should have type annotations
3. **Add docstrings**: Google-style docstrings for all public functions/classes
4. **Maintain compatibility**: NumPy <= 2.3.0
5. **Parallel processing**: Follow existing patterns for batch operations

### Code Style

#### PEP 8 Compliance

- 4 spaces for indentation
- Max line length: 88 characters (Black formatter)
- Meaningful variable names
- Clear, concise functions

#### Import Organization

```python
# Standard library
import os
from pathlib import Path
from typing import List, Optional

# Third-party
import numpy as np
import torch
from rdkit import Chem

# Local
from GSGE.vocab import GS_Vocab
from GSGE.core_gsge import CoreGSGE
```

#### Docstring Format

Use Google-style docstrings:

```python
def build_vocab(
    self,
    m_set: List[str],
    convert: bool = True,
    target: int = 100
) -> None:
    """
    Build molecular fragment vocabulary from SMILES list.

    Args:
        m_set: List of SMILES strings or RDKit Mol objects.
        convert: If True, input is SMILES; if False, input is Mol objects.
        target: Target number of fragments to extract.

    Raises:
        ValueError: If m_set is empty or contains invalid molecules.

    Example:
        >>> vocab = GS_Vocab()
        >>> vocab.build_vocab(['CCO', 'c1ccccc1'], convert=True, target=50)
        >>> print(vocab.num_fragments)
        50
    """
```

## Testing

### Running Tests

```bash
# All tests
pytest

# Specific test file
pytest tests/test_vocab.py

# With coverage
pytest --cov=GSGE --cov-report=html

# Using CLI
GSGE_CLI run_test
GSGE_CLI run_test --file test_make_cg.py
```

### Test Organization

Tests are organized by component:

- `tests/test_vocab.py` - Vocabulary and corpus tests
- `tests/test_tokenizer_comprehensive.py` - Tokenization tests
- `tests/test_gae.py` - Graph Autoencoder tests
- `tests/test_embedding.py` - Embedding generation tests
- `tests/test_integration.py` - End-to-end workflow tests
- `tests/test_edge_cases.py` - Error handling and edge cases

### Writing New Tests

When adding functionality:

1. **Create corresponding tests** in `tests/`
2. **Use pytest conventions**: Classes named `Test*`, methods named `test_*`
3. **Use fixtures**: Leverage `conftest.py` fixtures for common setups
4. **Mark slow tests**: Use `@pytest.mark.slow` for tests >1 second
5. **Test edge cases**: Include error handling and boundary conditions

Example test structure:

```python
import pytest
from GSGE import GS_Vocab

class TestVocabularyBuilding:
    """Tests for vocabulary building functionality."""

    def test_build_from_smiles(self, simple_smiles_list):
        """Test building vocabulary from SMILES list."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=50
        )
        assert vocab.num_fragments > 0
        assert vocab.num_fragments <= 50

    @pytest.mark.slow
    def test_large_vocabulary(self, sample_smiles_1000):
        """Test building large vocabulary."""
        vocab = GS_Vocab()
        vocab.build_vocab(m_set=sample_smiles_1000, convert=True, target=500)
        assert vocab.num_fragments > 0
```

### Test Fixtures

Reusable fixtures are defined in `tests/conftest.py`:

- `gsge_with_descriptors` - Pre-loaded GSGE with descriptors (session scope)
- `simple_smiles_list` - List of simple molecules (function scope)
- `minimal_vocab` - Small vocabulary for fast tests
- `mock_encoder_cpu` - Lightweight encoder for testing

## Pull Request Process

### 1. Create a Branch

From `dev` branch:

```bash
git checkout dev
git pull origin dev
git checkout -b feature/your-feature-name
```

Branch naming conventions:

- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation updates
- `test/*` - Test improvements

### 2. Make Changes

- Write clear, atomic commits
- Add tests for new functionality
- Update docstrings
- Follow code style guidelines

### 3. Test Thoroughly

```bash
# Run all tests
pytest

# Check code coverage
pytest --cov=GSGE --cov-report=term-missing

# Verify no import errors
python -c "from GSGE import *"
```

### 4. Update Documentation

- Add docstrings to new functions/classes
- Update user guides if adding features
- Update API reference if needed

### 5. Submit PR

- Target the `dev` branch
- Provide clear description
- Reference related issues
- Await review and address feedback

### Commit Message Format

```
type: Brief description (max 72 characters)

More detailed explanation if necessary. Explain what and why,
not how (the code shows how).

Fixes #123
```

**Types**:

- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding/updating tests
- `chore`: Maintenance tasks

## Reporting Issues

### Bug Reports

Include:

1. **Environment**:
   - Python version
   - GSGE version
   - OS
   - Package versions (`numpy`, `rdkit`, `torch`)

2. **Reproduction**:
   - Minimal code example
   - Input data
   - Expected vs. actual behavior

3. **Error messages**:
   - Full traceback
   - Warning messages

### Feature Requests

Describe:

- Use case and motivation
- Why existing functionality is insufficient
- Suggested implementation (optional)

### Questions

Before asking:

- Review [Installation Guide](../getting-started/installation.md)
- Search existing issues
- Look at `use_examples/` notebooks

## Development Workflow

### Branching Strategy

- `main` - Stable releases only
- `dev` - Active development (target for PRs)
- `feature/*` - New features
- `fix/*` - Bug fixes
- `docs/*` - Documentation

### Release Process

(For maintainers)

1. Update version in `pyproject.toml`
2. Update `CHANGES.md`
3. Merge `dev` → `main`
4. Tag release: `git tag v1.0.0`
5. Push to PyPI

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/CDDLeiden/GSGE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CDDLeiden/GSGE/discussions)
- **Email**:
  - Bola Khalil: b.a.a.khalil@lacdr.leidenuniv.nl
  - Jasper Durinck: jasper.j.durinck@gmail.com

## Code of Conduct

Be respectful and constructive in all interactions. We're building tools to advance molecular representation learning together!

## License

By contributing to GSGE, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to GSGE!** 🚀
