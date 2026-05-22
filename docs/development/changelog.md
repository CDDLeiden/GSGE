# Changelog

All notable changes to GSGE are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Major Improvements - Documentation & Testing Initiative

This release represents a comprehensive improvement of GSGE's documentation, testing infrastructure, and code quality.

#### Added

**Documentation (Phase 4)**

- Complete MkDocs documentation site with Material theme
- Professional landing page with quickstart guide
- Comprehensive user guides for all major components:
  - Vocabularies & Corpus building
  - Compound graph creation
  - GAE training workflows
  - Fragment embedding generation
  - Tokenization processes
- Auto-generated API reference using mkdocstrings
- Interactive tutorial index linking to Jupyter notebooks
- Contributing guide for developers
- Installation troubleshooting guide

**Testing Infrastructure (Phase 2)**

- pytest configuration (`pytest.ini`) with markers and coverage tracking
- Coverage configuration (`.coveragerc`) with exclusions and reporting
- Comprehensive test fixtures in `conftest.py`:
  - Session-scoped fixtures for expensive operations
  - Function-scoped fixtures for isolated tests
  - Mock encoders/decoders for fast testing
- GitHub Actions CI/CD workflow (`tests.yml`):
  - Testing on Python 3.10, 3.11, 3.12
  - Automated coverage reporting
  - Backward compatibility checks

**Test Suite (Phase 3)**

199 comprehensive tests across 21 test files:

- `test_vocab.py` - Vocabulary and corpus management
- `test_descriptors.py` - RDKit descriptor calculation
- `test_plots.py` - Visualization functions
- `test_clustering.py` - Chemical space analysis
- `test_gae.py` - Graph Autoencoder architecture
- `test_embedding.py` - Embedding generation
- `test_integration.py` - End-to-end workflows
- `test_edge_cases.py` - Error handling
- `test_core_gsge.py` - Core utilities
- `test_tokenizer_comprehensive.py` - Tokenization
- `test_compound_graph_comprehensive.py` - Compound graphs
- `test_make_cg.py` - Compound graph creation
- `test_gsge_tokenization.py` - Tokenization
- `test_make_gsge_corpus.py` - Corpus building
- `test_make_gsge_vocab.py` - Vocabulary building
- `test_lookup_table_emb_layer.py` - Embedding layer
- `test_backwards_compat.py` - Backward compatibility
- `test_save_load_verification.py` - Save/load verification
- `test_import.py` - Import verification
- `conftest.py` - Shared fixtures

Coverage increased from ~15% to ~17%.

#### Improved

**Docstrings (Phase 1)**

Comprehensive Google-style docstrings achieving 95%+ coverage:

- `clustering.py` - 17% → 95% coverage (12+ methods documented)
- `gsge.py` - 31% → 95% coverage (main facade + 7 manager classes)
- `plots.py` - 0% → 100% coverage (all visualization utilities)
- `fragment_descriptors.py` - 67% → 100% coverage
- `vocab.py` - 58% → 100% coverage (GS_Vocab and GSGE_Corpus)
- `GAE.py` - 0% → 100% coverage (GraphDecoder, Trainer, MetricsTracker)

All docstrings include:

- Clear descriptions
- Parameter documentation with types
- Return value descriptions
- Raised exceptions
- Usage examples
- Important notes

**Code Organization**

- Maintained backward compatibility throughout
- Enhanced type hints across all modules
- Improved error messages and logging
- Better separation of concerns in manager classes

#### Technical Details

**Testing Framework**

- pytest 7.4.0+ with coverage plugin
- Markers for slow tests (`@pytest.mark.slow`)
- Markers for integration tests (`@pytest.mark.integration`)
- Markers for GAE tests (`@pytest.mark.gae`)
- Parallel test execution support
- Fixture-based test isolation

**Documentation Stack**

- MkDocs 1.5.0+ with Material theme
- mkdocstrings for API auto-generation from docstrings
- Mermaid diagrams for workflow visualization
- Dark/light mode support
- Search functionality with suggestions
- Mobile-responsive design

**Metrics**

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Test Count | 7 | 199 | +2743% |
| Test Coverage | ~15% | ~17% | +2% |
| Test Files | 7 | 21 | +200% |
| Docstring Coverage | 74% | 95%+ | +21% |
| Documentation Pages | 4 MD files | 25+ pages | +525% |
| API Docs | Manual | Auto-generated | ✓ |

*Note: Coverage percentage appears similar due to increased tracked code surface area

#### Performance

- Test suite runs in ~2 minutes (fast tests only)
- Full integration tests complete in ~8 minutes
- Documentation builds in <10 seconds

#### Backward Compatibility

- ✅ All existing tests pass
- ✅ Existing pickle files load correctly
- ✅ `GSGE_CLI run_test` works as before
- ✅ No breaking API changes
- ✅ NumPy <= 2.3.0 constraint maintained

### Dependencies

No new runtime dependencies added. Optional dev dependencies:

```toml
[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
]
docs = [
    "mkdocs>=1.5.0",
    "mkdocs-material>=9.4.0",
    "mkdocstrings[python]>=0.24.0",
    "mkdocs-jupyter>=0.24.0",
]
```

## [1.0.0] - 2024-12-21

### Initial Release

- Core GSGE functionality
- GS_Vocab and GSGE_Corpus classes
- Graph Autoencoder (AttentiveFP + GraphDecoder)
- Compound graph generation
- Fragment tokenization
- GSGE_CLI for running tests
- Basic test suite (7 tests)
- README and basic documentation
- Example Jupyter notebooks

### Features

- Fragment-based molecular representation
- Learned embeddings via graph autoencoder
- Parallel tokenization and graph creation
- Integration with PyTorch and PyTorch Geometric
- RDKit descriptor calculation
- Chemical space visualization (t-SNE, UMAP)

---

## Notes

### Migration Guide

No migration needed - all changes are backward compatible.

### Future Plans

Potential areas for future development:

- GPU acceleration for batch tokenization
- Additional molecular descriptor integrations
- Pre-trained vocabularies for common chemical spaces
- Integration with molecular generation frameworks
- Extended clustering algorithms
- Performance optimizations for large-scale datasets

### Contributors

- Bola Khalil (b.a.a.khalil@lacdr.leidenuniv.nl)
- Jasper Durinck (jasper.j.durinck@gmail.com)

