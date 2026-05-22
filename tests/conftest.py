"""
Shared pytest fixtures for GSGE test suite.

Provides reusable fixtures for testing vocabularies, embeddings, models,
and sample data across all test modules.
"""

import pytest
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from typing import List
import os

from GSGE import GSGE, GS_Vocab, GSGE_Corpus
from GSGE.graphs.fragment_graph.GAE import AttentiveFP, GraphDecoder

# Get the tests directory path (now at project root, not in GSGE package)
_tests_dir = Path(__file__).parent

# Disable tqdm monitor thread to prevent RDKit segfaults
os.environ['TQDM_DISABLE'] = '0'  # Keep tqdm enabled
try:
    from tqdm import tqdm
    # Disable the monitor thread that causes RDKit segfaults
    tqdm.monitor_interval = 0
except ImportError:
    pass


# ============================================================================
# Session-scoped fixtures (expensive, loaded once)
# ============================================================================

@pytest.fixture(scope="session")
def gsge_with_descriptors():
    """
    Load pre-saved GSGE instance with descriptors (session scope).

    Expensive to load, reused across all tests in session.
    Contains complete GSGE with vocabulary, embeddings, and descriptors.
    """
    pkl_path = _tests_dir / 'test_gsge_save_with_descriptors.pkl'
    gsge = GSGE(GSGE_load_path=pkl_path)
    return gsge


@pytest.fixture(scope="session")
def gsge_v5a2():
    """
    Load v5a2 GSGE instance (session scope).

    Alternative GSGE configuration for compatibility testing.
    """
    pkl_path = _tests_dir / 'gsge_save_v5a2.pkl'
    gsge = GSGE(GSGE_load_path=pkl_path)
    return gsge


@pytest.fixture(scope="session")
def sample_smiles_1000():
    """
    Load 1000 sample SMILES strings from pickle (session scope).

    Provides diverse molecular structures for testing tokenization,
    graph generation, and other batch operations.
    """
    pkl_path = _tests_dir / 'subset_smiles_1000.pkl'
    with open(pkl_path, 'rb') as f:
        smiles_list = pickle.load(f)
    return smiles_list


@pytest.fixture(scope="session")
def minimal_vocab():
    """
    Create minimal vocabulary with common fragments for fast unit tests.

    Contains only ~30 frequently occurring fragments to enable fast
    testing without full vocabulary overhead.
    """
    common_frags = [
        'C', 'O', 'N', 'S', 'Cl', 'Br',  # Elements
        'CC', 'CO', 'CN', 'c1ccccc1',   # Simple fragments
        'C(C)(C)C', 'c1ccc(*)cc1',       # Branched/aromatic
    ]
    vocab = GS_Vocab()
    for frag in common_frags:
        vocab.add_GS_fragment(frag)
    return vocab


# ============================================================================
# Function-scoped fixtures (created fresh for each test)
# ============================================================================

@pytest.fixture
def peptide_smiles():
    """
    Small set of cyclic peptide SMILES for testing (function scope).

    Returns list of 10 structurally diverse cyclic peptides for
    testing fragmentation, vocabulary building, and GAE training.
    """
    return [
        'C[C@H](NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CCCN1C(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](CO)NC(=O)[C@@H]1CSSC[C@@H]2NC(=O)[C@H](Cc3ccccc3)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](N)Cc3ccccc3)CSSC[C@@H](NC2=O)C(=O)N1)C(=O)O',
        'CC[C@H](C)[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CO)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]1CSSC[C@H](NC(=O)[C@H](Cc2ccccc2)NC(=O)[C@H](Cc2c[nH]c3c2cccc3)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(N)=O)NC(=O)CNC1=O)C(=O)N[C@@H](Cc1c[nH]c2c1cccc2)C(=O)O)C(=O)O',
        'C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H]1CCCN1C(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H]1CSSC[C@@H]2NC(=O)[C@H](Cc3ccccc3)NC(=O)[C@H](CC(C)C)NC(=O)[C@@H](NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@@H](N)Cc3ccccc3)CSSC[C@@H](NC2=O)C(=O)N1)C(=O)O',
    ]


@pytest.fixture
def mock_encoder_cpu():
    """
    Lightweight AttentiveFP encoder for CPU testing (function scope).

    Smaller architecture for fast unit tests without GPU requirement.
    """
    return AttentiveFP(
        in_channels=9,
        hidden_channels=64,
        out_channels=32,
        edge_dim=3,
        num_layers=2,
        num_timesteps=1
    )


@pytest.fixture
def mock_decoder_cpu():
    """
    Lightweight GraphDecoder for CPU testing (function scope).

    Matches mock_encoder_cpu dimensions for testing.
    """
    return GraphDecoder(latent_dim=32, hidden_dim=64)


@pytest.fixture
def simple_smiles_list():
    """
    Very simple SMILES for basic unit tests (function scope).

    Returns list of 5 simple molecules for quick testing.
    """
    return ['CCO', 'CC(C)O', 'c1ccccc1', 'CC(=O)O', 'NCCO']


# ============================================================================
# Temporary directory fixtures
# ============================================================================

@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """
    Temporary directory for saving model checkpoints (function scope).

    Automatically cleaned up after test completion.
    """
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    return checkpoint_dir


@pytest.fixture
def temp_vocab_dir(tmp_path):
    """
    Temporary directory for saving vocabularies (function scope).

    Automatically cleaned up after test completion.
    """
    vocab_dir = tmp_path / "vocabs"
    vocab_dir.mkdir()
    return vocab_dir


# ============================================================================
# Parametrize helpers
# ============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: integration tests requiring full setup"
    )
    config.addinivalue_line(
        "markers", "unit: fast unit tests"
    )
    config.addinivalue_line(
        "markers", "gae: GAE-related tests"
    )
    config.addinivalue_line(
        "markers", "vocab: vocabulary tests"
    )
    config.addinivalue_line(
        "markers", "tokenization: tokenization tests"
    )
    config.addinivalue_line(
        "markers", "compound_graph: compound graph tests"
    )
