from pathlib import Path
import importlib.resources

__version__ = "1.0.0"

from .fragment_tools import FragmentTools, GS_FragmentTools
from .gsge import GSGE
from .vocab import GS_Vocab, GSGE_Corpus, CUSTOM_fragment_mol
from .plots import highlight_fragments
from .tokenizer import GSGE_tokenizer
from .chem import (
    _GRAMMAR_TOKENS,
    _ELEMENT_TOKENS,
    _REDIRECT_TOKENS,
    _ELEMENTS_BOND_COUNTS,
)
from .core_gsge import CoreGSGE
from .embedding import GSGE_Embedding

# =============================================================================
# Path Resolution Utilities
# =============================================================================
#
# GSGE provides canonical path constants and utilities for consistent path
# handling across all installation modes (editable, site-packages, source).
#
# Usage patterns:
#   1. For package resources (data files shipped with GSGE):
#      Use get_package_resource() or GSGE_PACKAGE_DIR directly
#
#   2. For test fixtures (when running from source checkout):
#      Use get_project_root() to find the project root, then access tests/
#
#   3. For user-provided paths:
#      Use pathlib.Path directly, no GSGE utilities needed
#
# These constants are primarily for internal GSGE use and CLI tools.
# =============================================================================

# Path to the installed GSGE package directory
# Works in both editable installs and site-packages
GSGE_PACKAGE_DIR: Path = Path(__file__).parent.resolve()


def get_package_resource(resource_name: str) -> Path:
    """
    Get the path to a resource file within the GSGE package.

    This function provides a consistent way to access package resources
    (data files, configs, etc.) that are shipped with GSGE. It works
    correctly in all installation modes.

    Args:
        resource_name: Relative path to the resource within the GSGE package.
            Example: 'version.json' or 'data/default_vocab.pkl'

    Returns:
        Path object pointing to the resource.

    Example:
        >>> from GSGE import get_package_resource
        >>> version_path = get_package_resource('version.json')
        >>> print(version_path.exists())
        True
    """
    return GSGE_PACKAGE_DIR / resource_name


def get_project_root() -> Path | None:
    """
    Attempt to find the project root directory (contains pyproject.toml).

    This function walks up from GSGE_PACKAGE_DIR looking for pyproject.toml.
    Useful for development scenarios where you need access to the full
    repository structure (tests/, use_examples/, docs/, etc.).

    Returns:
        Path to project root if found (editable install or source checkout),
        None if running from a standard pip install (no source available).

    Example:
        >>> from GSGE import get_project_root
        >>> root = get_project_root()
        >>> if root:
        ...     tests_dir = root / 'tests'
        ...     examples_dir = root / 'use_examples'

    Warning:
        This function returns None for standard pip installs. Always check
        the return value before using. For package data, use get_package_resource()
        instead.
    """
    current = GSGE_PACKAGE_DIR
    # Walk up at most 5 levels to find project root
    for _ in range(5):
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:  # Reached filesystem root
            break
        current = parent
    return None


def get_tests_dir() -> Path | None:
    """
    Get the path to the tests directory (at project root).

    Convenience function for accessing test fixtures during development.
    Returns None if running from a standard pip install.

    Returns:
        Path to tests/ directory if found, None otherwise.

    Example:
        >>> from GSGE import get_tests_dir
        >>> tests_dir = get_tests_dir()
        >>> if tests_dir:
        ...     fixture_path = tests_dir / 'test_gsge_save_with_descriptors.pkl'
    """
    root = get_project_root()
    if root is not None:
        tests_dir = root / "tests"
        if tests_dir.exists():
            return tests_dir
    return None


def get_use_examples_dir() -> Path | None:
    """
    Get the path to the use_examples directory (at project root).

    Convenience function for accessing examples and tutorials during development.
    Returns None if running from a standard pip install.

    Returns:
        Path to use_examples/ directory if found, None otherwise.

    Example:
        >>> from GSGE import get_use_examples_dir
        >>> examples_dir = get_use_examples_dir()
        >>> if examples_dir:
        ...     gae_dir = examples_dir / '03_GAE' / 'v2'
        ...     checkpoint_dir = gae_dir / 'model_checkpoints'
    """
    root = get_project_root()
    if root is not None:
        examples_dir = root / "use_examples"
        if examples_dir.exists():
            return examples_dir
    return None
