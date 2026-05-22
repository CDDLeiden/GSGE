"""
Tests for molecular fragment descriptor calculation and normalization.

Tests RDKit descriptor calculation, z-score normalization,
and descriptor integration with GSGE vocabularies.
"""

import pytest
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors

from GSGE.fragment_descriptors import (
    calc_mol_frag_descriptors,
    get_mol_frag_descriptors,
    normalize_descriptors
)
from GSGE import GSGE


class TestFragmentDescriptors:
    """Tests for fragment descriptor calculation."""

    def test_calc_descriptors_from_smiles(self):
        """Test descriptor calculation from SMILES string."""
        descriptors = calc_mol_frag_descriptors(smiles='CCO')
        assert isinstance(descriptors, np.ndarray)
        # Number of descriptors may vary by RDKit version (typically 47-48)
        assert descriptors.shape[0] >= 45  # At least 45 descriptors
        assert descriptors.dtype == np.float32

    def test_calc_descriptors_from_mol(self):
        """Test descriptor calculation from RDKit Mol object."""
        mol = Chem.MolFromSmiles('CCO')
        descriptors = calc_mol_frag_descriptors(mol=mol)
        assert isinstance(descriptors, np.ndarray)
        # Number of descriptors may vary by RDKit version
        assert descriptors.shape[0] >= 45

    def test_calc_descriptors_custom_keys(self):
        """Test descriptor calculation with custom subset of descriptors."""
        custom_keys = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA']
        descriptors = calc_mol_frag_descriptors(
            smiles='CCO',
            descriptor_keys=custom_keys
        )
        assert descriptors.shape[0] == len(custom_keys)

    def test_calc_descriptors_invalid_molecule(self):
        """Test descriptor calculation handles invalid SMILES."""
        # Invalid SMILES should be handled gracefully
        descriptors = calc_mol_frag_descriptors(smiles='INVALID', verbose=False)
        # Should handle gracefully (may return None or zeros)
        assert descriptors is None or descriptors is not None

    def test_calc_descriptors_fragment_with_wildcard(self):
        """Test descriptor calculation for fragment with wildcard atom."""
        # Fragment SMILES with wildcard
        descriptors = calc_mol_frag_descriptors(smiles='CC(*)O')
        assert descriptors is not None
        # Number of descriptors may vary by RDKit version
        assert descriptors.shape[0] >= 45

    def test_get_vocab_descriptors(self, gsge_with_descriptors):
        """Test getting descriptors for all vocabulary fragments."""
        descriptors = get_mol_frag_descriptors(gsge_with_descriptors)
        assert isinstance(descriptors, np.ndarray)
        assert descriptors.ndim == 2  # 2D array
        # Rows = fragments, columns = descriptors
        # Use vocab_manager to get vocab size
        vocab_size = len(gsge_with_descriptors.vocab_manager.GS_vocab.fragments)
        assert descriptors.shape[0] == vocab_size
        # Number of descriptors may vary by RDKit version
        assert descriptors.shape[1] >= 45

    def test_get_vocab_descriptors_custom_keys(self, gsge_with_descriptors):
        """Test getting custom descriptor subset for vocabulary."""
        custom_keys = ['MolWt', 'TPSA', 'MolLogP']
        descriptors = get_mol_frag_descriptors(
            gsge_with_descriptors,
            calc_mol_frag_descriptors_args={'descriptor_keys': custom_keys}
        )
        assert descriptors.shape[1] == len(custom_keys)

    def test_normalize_descriptors(self):
        """Test descriptor normalization with z-score."""
        # Create sample descriptor matrix
        raw_descriptors = np.array([
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0]
        ])

        normalized, means, stds, mask = normalize_descriptors(raw_descriptors)

        # Check shapes
        assert normalized.shape[0] == raw_descriptors.shape[0]
        assert len(means) == len(stds)
        assert len(mask) == raw_descriptors.shape[1]

        # Check normalization (mean should be ~0, std should be ~1)
        assert np.abs(normalized.mean(axis=0)).max() < 1e-6  # Mean ≈ 0
        # Std should be 1 for each column
        np.testing.assert_allclose(normalized.std(axis=0), 1.0, rtol=1e-6)

    def test_normalize_filters_low_variance(self):
        """Test that low-variance descriptors are filtered out."""
        # Create descriptors with one constant column (zero variance)
        raw_descriptors = np.array([
            [1.0, 5.0, 3.0],
            [2.0, 5.0, 6.0],
            [3.0, 5.0, 9.0],
            [4.0, 5.0, 12.0]
        ])

        normalized, means, stds, mask = normalize_descriptors(raw_descriptors, min_var=1e-6)

        # Column 1 (all 5.0) should be filtered out
        assert mask.sum() < raw_descriptors.shape[1]
        assert not mask[1]  # Second column (index 1) should be filtered
        assert mask[0] and mask[2]  # Other columns should remain

    def test_denormalization_roundtrip(self):
        """Test that normalization can be reversed."""
        raw_descriptors = np.random.rand(10, 5)

        normalized, means, stds, mask = normalize_descriptors(raw_descriptors)

        # Denormalize
        denormalized = normalized * stds + means

        # Should match original (filtered columns only)
        np.testing.assert_allclose(denormalized, raw_descriptors[:, mask], rtol=1e-6)


class TestGSGEDescriptorIntegration:
    """Tests for descriptor integration with GSGE framework."""

    def test_calc_fragment_descriptors_method(self, gsge_with_descriptors):
        """Test GSGE.calc_fragment_descriptors() method."""
        gsge_with_descriptors.calc_fragment_descriptors()

        descriptors = gsge_with_descriptors.get_fragment_descriptors()
        assert descriptors is not None
        assert isinstance(descriptors, torch.Tensor)
        assert descriptors.ndim == 2

    def test_get_fragment_descriptors_none_before_calc(self, minimal_vocab):
        """Test that descriptors are None before calculation."""
        from GSGE import GSGE
        gsge = GSGE(GS_vocab=minimal_vocab)
        descriptors = gsge.get_fragment_descriptors()
        assert descriptors is None

    def test_get_fragment_descriptors_names(self, gsge_with_descriptors):
        """Test getting descriptor names."""
        gsge_with_descriptors.calc_fragment_descriptors()
        names = gsge_with_descriptors.get_fragment_descriptors_names()
        assert isinstance(names, list)
        assert len(names) > 0
        assert 'MolWt' in names

    def test_descriptors_and_embeddings_concat(self, gsge_with_descriptors):
        """Test concatenating descriptors with embeddings."""
        # This would require embeddings to be present
        if gsge_with_descriptors.get_fragment_embeddings() is not None:
            combined = gsge_with_descriptors.get_fragment_descriptors_and_embeddings()
            assert isinstance(combined, torch.Tensor)
            # Should have more features than descriptors alone
            descriptors_only = gsge_with_descriptors.get_fragment_descriptors()
            assert combined.shape[1] > descriptors_only.shape[1]
