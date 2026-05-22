## [1.0.0] - 2026-02-20

### Added
- `ARCHITECTURE.md`: Comprehensive architecture documentation with data flow pipeline, module hierarchy, and typical usage workflow
- `GSGE/__init__.py`: Added `__version__` variable for runtime version access

### Changed
- Standardized versioning to semantic versioning (1.0.0) across `pyproject.toml` and `GSGE/__init__.py`
- Removed unused `version.json` package-data reference from `pyproject.toml`
- `GSGE/visualization.py`: Added `seed` parameter for reproducible plots and cleaned up imports
- `GSGE/graphs/compound_graph/data.py`: Improved variable naming for clarity in graph processing
- `GSGE/tokenizer.py`: Updated type hints and import paths for better clarity
- `GSGE/utils_chem.py`: Moved from `GSGE/graphs/fragment_graph/` to `GSGE/` root directory for better module organization

### Fixed
- `GSGE/gsge.py`: Made node features 2D with dtype long for PyTorch Geometric compatibility
- `GSGE/graphs/compound_graph/data.py`: Handle bondless compound graphs (e.g., very small compounds) in compound_graph reordering
- `tests/test_backwards_compat.py`: Fixed SMILES notation mistake that caused test failure

### Documentation
- Added detailed `ARCHITECTURE.md` with architecture overview, data flow pipeline, and usage workflow
- Updated `CONTRIBUTING.md` with architecture references and new test details
- Updated `docs/development/changelog.md`, `docs/development/testing.md` with improved documentation
- Updated `docs/getting-started/installation.md` with clarified dependency constraints
- Updated references in `use_examples/README.md` and `use_examples/00_making_vocabs/README.md`

## [0.1.0.dev2] - 2026-01-23

### Fixed
- `GSGE/gsge.py`: Fixed `set_decoder()` method which was incorrectly calling `set_encoder()` instead
- `GSGE/gsge.py`: Auto-generate `GSGE_vocab` token-to-ID dictionary and `OHE_tokens_mask` in `make_GS_fragment_embedding_dict()` when they are None
- `GSGE/core_gsge.py`: Fixed optimizer parameter handling to accept both optimizer class (e.g., `torch.optim.Adam`) and instantiated optimizer objects
- `GSGE/vocab.py`: Added validation warnings for `n_limit` parameter to help users avoid empty vocabularies when dataset size is too small
- `GSGE/fragment_functions.py`: Added `Chem.SanitizeMol()` calls after fragmentation steps to prevent valence errors on complex molecules
- `use_examples/06_end_to_end/property_prediction_tutorial.ipynb`: Fixed tutorial to work with current API after manager pattern refactoring:
  - Updated to use `vocab_manager.GS_vocab` and `vocab_manager.GSGE_corpus` attribute access
  - Fixed `n_limit=80` to `n_limit=1` for compatibility with small datasets
  - Added encoder/decoder setup before GAE training
  - Added defensive `get_molecular_features()` with proper PyTorch tensor to numpy conversion
  - Added dynamic cross-validation fold calculation for small datasets

### Changed
- `use_examples/06_end_to_end/property_prediction_tutorial.ipynb`: Expanded sample dataset from 5 to 10 molecules for better training

## [0.1.0.dev1] - 2026-01-21

### Added
- `GSGE/visualization.py` module with standalone visualization utilities:
  - `plot_cluster_grid()` - Grid visualization of example molecules per cluster
  - `plot_descriptor_distribution()` - Descriptor distribution histograms
- End-to-end property prediction tutorial (`use_examples/06_end_to_end/`)
  - Complete ML pipeline from vocabulary building to model deployment
  - Property prediction tutorial notebook with 16 detailed steps
- CLI Usage section in README.md
- Windows Setup section in CONTRIBUTING.md
- System requirements section in INSTALLATION.md
- Optional dependencies documentation in INSTALLATION.md
- Expected outputs table in use_examples/README.md
- CPU/GPU time estimates in tutorial overview table
- Common Pitfalls section with 6 common issues and solutions
- Under-Documented Features section in CLAUDE.md exposing:
  - GSGE_Embedding Layer usage
  - Large-Scale Tokenization
  - MLM Masking for self-supervised learning
  - Descriptor + Embedding combination
  - Clustering with pre-computed embeddings
  - Graph visualization
  - Custom fragmentation functions

### Changed
- Updated use_examples/README.md tutorial table with CPU/GPU time estimates
- Enhanced INSTALLATION.md numpy version documentation (changed from `<=2.3.0` to `>=1.26.4`)
- Reorganized use_examples/README.md with structured Expected Results section

### Fixed
- README.md line 93: Missing comma between `GSGE` and `CUSTOM_fragment_mol` in import statement
- README.md lines 135-136: Broken import syntax (`from core_gsje` → `from GSGE.core_gsge import add_all_single_elements`)
- CONTRIBUTING.md line 149: Wrong test path (`GSGE/tests/` → `tests/`)

### Documentation
- Comprehensive documentation improvements across all key files
- Better visibility for advanced features through new CLAUDE.md section
- Improved troubleshooting guidance with common pitfalls

## [1.0.0] - 2026-01-12

### Added
- Comprehensive GSGE documentation suite including installation guides, API reference, tutorials, and usage examples
- Molecular fragment tokenization tutorial with detailed explanations and examples
- Complete test suite with unit tests, integration tests, and edge case coverage for all GSGE components
- GitHub Actions CI/CD workflow for testing across multiple Python versions with artifact uploading
- Test coverage configuration with pytest-xdist and pytest-timeout for improved performance
- Custom documentation site configuration with mkdocs, CSS themes, and MathJax support
- Shared pytest fixtures for consistent testing across the GSGE test suite

### Changed
- Updated package configuration with modern pyproject.toml setup and proper dependency management
- Enhanced all core GSGE components with comprehensive docstrings following the project style guide
- Improved Graph Autoencoder components with enhanced functionality and documentation
- Updated citation information and author order for proper attribution
- Renamed readme.md to README.md for consistency across platforms
- Added badges to README for test status and PyPI version tracking

### Fixed
- Corrected test logic for API mismatches and marked RDKit-crashing tests as slow tests
- Fixed segfault issues in CI/CD pipeline and reduced timeouts for clustering tests
- Added missing matplotlib and networkx dependencies to pyproject.toml
- Resolved CI/CD workflow issues by adding tkinter and group-selfies fork dependencies
- Added coverage configuration and proper test discovery setup
- Updated GitHub Actions upload-artifact action to version 4 for compatibility

### Documentation
- Added comprehensive work summary detailing GSGE improvement phases and testing infrastructure
- Created user guide and vocabulary management documentation
- Added tutorials for GSGE usage with Jupyter notebooks
- Implemented proper docstring style guide for consistent documentation practices
- Added license and citation documentation for proper attribution
- Enhanced README with tags, tutorials, and learning resources

### Testing
- Added unit tests for GS_Vocab and GSGE_Corpus classes
- Implemented tests for GSGE tokenization and preprocessing functionality
- Added tests for molecular visualization and fragment highlighting
- Created edge case and error handling tests for the GSGE framework
- Added integration tests for GSGE workflows and pipelines
- Implemented tests for Graph Autoencoder training and inference
- Added tests for GSGE embedding layer and embedding manager
- Created tests for molecular fragment descriptor calculations
- Added tests for compound graph generation and manipulation
- Implemented tests for GSGE clustering functionality

### Infrastructure
- Updated .gitignore with additional patterns for build artifacts, virtual environments, and IDE files
- Configured pytest for test discovery and coverage reporting
- Added environment configuration for GSGE development
- Updated workflows for better performance and reliability
- Configured proper artifact uploading and version management