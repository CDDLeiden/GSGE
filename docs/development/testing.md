# Testing Guide

Comprehensive guide to testing GSGE.

## Running Tests

### Quick Test

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=GSGE --cov-report=html

# Using CLI
GSGE_CLI run_test
```

### Test Selection

```bash
# Specific test file
pytest tests/test_vocab.py

# Specific test class
pytest tests/test_vocab.py::TestGSVocabBuilding

# Specific test method
pytest tests/test_vocab.py::TestGSVocabBuilding::test_build_from_smiles

# Using patterns
pytest -k "vocab"  # All tests matching "vocab"
```

### Test Markers

```bash
# Skip slow tests (fast unit tests only)
pytest -m "not slow"

# Only integration tests
pytest -m "integration"

# Only GAE tests
pytest -m "gae"

# Combine markers
pytest -m "slow and integration"
```

## Test Organization

### Test Structure

```
tests/
├── conftest.py                        # Shared fixtures
├── test_vocab.py                      # Vocabulary tests
├── test_tokenizer_comprehensive.py    # Tokenization tests
├── test_gae.py                        # GAE tests
├── test_embedding.py                  # Embedding tests
├── test_clustering.py                 # Clustering tests
├── test_descriptors.py                # Descriptor tests
├── test_plots.py                      # Visualization tests
├── test_core_gsge.py                  # Core utilities tests
├── test_integration.py                # End-to-end tests
├── test_edge_cases.py                 # Error handling tests
├── test_compound_graph_comprehensive.py  # Compound graph tests
├── test_make_cg.py                    # Compound graph creation tests
├── test_gsge_tokenization.py          # Tokenization tests
├── test_make_gsge_corpus.py           # Corpus tests
├── test_make_gsge_vocab.py            # Vocab tests
├── test_lookup_table_emb_layer.py     # Embedding layer tests
├── test_backwards_compat.py           # Backward compatibility tests
├── test_save_load_verification.py     # Save/load verification tests
└── test_import.py                     # Import tests
```

### Test Categories

| Category | Purpose | Speed | Markers |
|----------|---------|-------|---------|
| **Unit** | Test individual functions | Fast (<0.1s) | None |
| **Integration** | Test complete workflows | Medium (0.1-2s) | `@pytest.mark.integration` |
| **Slow** | Heavy computation, I/O | Slow (>2s) | `@pytest.mark.slow` |
| **GAE** | Graph autoencoder specific | Varies | `@pytest.mark.gae` |

## Fixtures

Reusable test fixtures are defined in `conftest.py`.

### Session-Scoped Fixtures

Loaded once per test session (expensive operations):

```python
@pytest.fixture(scope="session")
def gsge_with_descriptors():
    """Pre-loaded GSGE with descriptors from saved state."""
    pkl_path = pkg_resources.files(tests).joinpath('test_gsge_save_with_descriptors.pkl')
    return GSGE(GSGE_load_path=pkl_path)

@pytest.fixture(scope="session")
def sample_smiles_1000():
    """1000 SMILES for large-scale tests."""
    # Load from pickled dataset
    return smiles_list
```

### Function-Scoped Fixtures

Created for each test (fast, isolated):

```python
@pytest.fixture
def simple_smiles_list():
    """Small list of simple molecules."""
    return ['CCO', 'c1ccccc1', 'CC(C)O', 'CC(=O)O']

@pytest.fixture
def minimal_vocab(simple_smiles_list):
    """Minimal vocabulary for fast tests."""
    vocab = GS_Vocab()
    vocab.build_vocab(m_set=simple_smiles_list, convert=True, target=30)
    return vocab
```

### Mock Objects

Lightweight mocks for testing without expensive operations:

```python
@pytest.fixture
def mock_encoder_cpu():
    """Lightweight AttentiveFP encoder for CPU testing."""
    return AttentiveFP(
        in_channels=9,
        hidden_channels=32,  # Reduced for speed
        out_channels=32,
        edge_dim=3,
        num_layers=1,
        num_timesteps=1
    )
```

## Writing Tests

### Test Structure

```python
import pytest
from GSGE import GS_Vocab

class TestVocabularyBuilding:
    """Tests for vocabulary building functionality."""

    def test_build_from_smiles(self, simple_smiles_list):
        """Test building vocabulary from SMILES list."""
        # Arrange
        vocab = GS_Vocab()

        # Act
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=50
        )

        # Assert
        assert vocab.num_fragments > 0
        assert vocab.num_fragments <= 50

    @pytest.mark.slow
    def test_large_vocabulary(self, sample_smiles_1000):
        """Test building large vocabulary (marked slow)."""
        vocab = GS_Vocab()
        vocab.build_vocab(m_set=sample_smiles_1000, convert=True, target=500)
        assert vocab.num_fragments > 0
```

### Test Naming

- Test files: `test_*.py`
- Test classes: `Test*`
- Test methods: `test_*`
- Descriptive names: `test_build_vocab_from_smiles_list`

### Assertions

```python
# Exact equality
assert vocab.num_fragments == 50

# Comparison
assert vocab.num_fragments > 0
assert vocab.num_fragments <= target

# Type checking
assert isinstance(tokens, list)

# Membership
assert 'GS_frag_0' in token_dict

# Boolean
assert vocab is not None

# Exceptions
with pytest.raises(ValueError):
    vocab.build_vocab(m_set=[], convert=True)

# Exception messages
with pytest.raises(ValueError, match="empty"):
    vocab.build_vocab(m_set=[], convert=True)
```

### Parametrized Tests

Test multiple inputs efficiently:

```python
@pytest.mark.parametrize("smiles,expected_atoms", [
    ('C', 1),
    ('CCO', 3),
    ('c1ccccc1', 6),
])
def test_atom_count(smiles, expected_atoms):
    """Test atom counting for various molecules."""
    mol = Chem.MolFromSmiles(smiles)
    assert mol.GetNumAtoms() == expected_atoms
```

## Coverage

### Generating Coverage Reports

```bash
# Terminal output
pytest --cov=GSGE --cov-report=term-missing

# HTML report
pytest --cov=GSGE --cov-report=html
# Opens htmlcov/index.html

# XML report (for CI/CD)
pytest --cov=GSGE --cov-report=xml
```

### Coverage Configuration

Configured in `.coveragerc`:

```ini
[run]
source = GSGE
omit =
    */tests/*
    */test_*.py
    */__init__.py
    */scripts/CLI.py

[report]
precision = 2
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

### Coverage Goals

| Module | Current | Target |
|--------|---------|--------|
| vocab.py | 90% | 95% |
| tokenizer.py | 85% | 90% |
| embedding.py | 85% | 90% |
| GAE.py | 75% | 85% |
| clustering.py | 80% | 85% |
| Overall | ~17% | 85% |

## Continuous Integration

### GitHub Actions

Tests run automatically on:

- Push to `main` or `dev`
- Pull requests
- Manual workflow dispatch

Configuration: `.github/workflows/tests.yml`

```yaml
name: Tests

on:
  push:
    branches: [main, dev]
  pull_request:
    branches: [main, dev]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: ${{ matrix.python-version }}
      - name: Install dependencies
        run: |
          pip install -e ".[dev]"
      - name: Run tests
        run: |
          pytest --cov=GSGE --cov-report=xml
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Performance Testing

### Profiling Tests

```bash
# Profile test execution
pytest --profile

# Sort by cumulative time
pytest --profile-svg
```

### Benchmarking

Create benchmark tests for critical paths:

```python
def test_vocabulary_building_performance(benchmark, large_smiles_dataset):
    """Benchmark vocabulary building speed."""
    vocab = GS_Vocab()

    result = benchmark(
        vocab.build_vocab,
        m_set=large_smiles_dataset,
        convert=True,
        target=500
    )

    assert vocab.num_fragments > 0
```

## Debugging Tests

### Running with Verbose Output

```bash
# Verbose output
pytest -v

# Show print statements
pytest -s

# Show local variables on failure
pytest -l

# Enter debugger on failure
pytest --pdb
```

### Debugging Individual Tests

```python
def test_my_feature():
    """Test with debugging."""
    import pdb; pdb.set_trace()  # Breakpoint

    vocab = GS_Vocab()
    vocab.build_vocab(...)
```

## Best Practices

### 1. Test Isolation

Each test should be independent:

```python
# ✓ Good - uses fixture
def test_vocab_building(minimal_vocab):
    # Test uses fresh fixture
    assert minimal_vocab.num_fragments > 0

# ✗ Bad - shared state
vocab = GS_Vocab()  # Module level

def test_vocab_building():
    # Tests share state
    vocab.build_vocab(...)
```

### 2. Clear Test Names

```python
# ✓ Good
def test_build_vocab_with_empty_smiles_list_raises_error():
    pass

# ✗ Bad
def test_1():
    pass
```

### 3. Arrange-Act-Assert

```python
def test_vocabulary_building():
    # Arrange
    vocab = GS_Vocab()
    smiles = ['CCO', 'c1ccccc1']

    # Act
    vocab.build_vocab(m_set=smiles, convert=True, target=50)

    # Assert
    assert vocab.num_fragments > 0
```

### 4. Test Edge Cases

```python
def test_empty_input():
    """Test with empty input."""
    with pytest.raises(ValueError):
        vocab.build_vocab(m_set=[], convert=True)

def test_single_molecule():
    """Test with single molecule."""
    vocab.build_vocab(m_set=['C'], convert=True, target=10)
    assert vocab.num_fragments >= 1
```

### 5. Use Markers

```python
@pytest.mark.slow
def test_expensive_operation():
    """Long-running test."""
    pass

@pytest.mark.integration
def test_full_workflow():
    """End-to-end test."""
    pass
```

## Troubleshooting

### Common Issues

**Tests fail with import errors:**

```bash
# Reinstall in editable mode
pip install -e .
```

**Fixture not found:**

```bash
# Check conftest.py is present
ls tests/conftest.py
```

**Tests hang:**

```bash
# Check for infinite loops or deadlocks
# Use timeout
pytest --timeout=60
```

**Coverage not showing:**

```bash
# Ensure .coveragerc is present
# Run with explicit config
pytest --cov=GSGE --cov-config=.coveragerc
```

## Additional Resources

- [pytest documentation](https://docs.pytest.org/)
- [pytest-cov documentation](https://pytest-cov.readthedocs.io/)
- [Testing best practices](https://docs.python-guide.org/writing/tests/)
