"""
Tests for molecular visualization and fragment highlighting functions.

Tests color generation, fragment cleaning, molecule sorting,
and fragment highlighting visualization.
"""

import pytest
from rdkit import Chem
from PIL import Image

from GSGE.plots import (
    clean_fragment,
    generate_colors,
    sort_mols_atom_num,
    highlight_fragments
)


class TestPlotUtilities:
    """Tests for plotting utility functions."""

    def test_clean_fragment_removes_wildcards(self):
        """Test that clean_fragment removes dummy atoms ([*])."""
        frag_smiles = 'CC(*)O'
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        cleaned = clean_fragment(frag_mol)

        # Should remove wildcard atom
        assert cleaned.GetNumAtoms() == frag_mol.GetNumAtoms() - 1
        # Resulting molecule should be valid
        assert Chem.MolToSmiles(cleaned) == 'CCO'

    def test_clean_fragment_with_multiple_wildcards(self):
        """Test cleaning fragment with multiple wildcard atoms."""
        frag_smiles = 'C(*)C(*)*'
        frag_mol = Chem.MolFromSmiles(frag_smiles)
        cleaned = clean_fragment(frag_mol)

        # Should remove all 3 wildcards
        original_atoms = frag_mol.GetNumAtoms()
        cleaned_atoms = cleaned.GetNumAtoms()
        assert cleaned_atoms < original_atoms

    def test_clean_fragment_no_wildcards(self):
        """Test cleaning fragment without wildcards (no change expected)."""
        mol_smiles = 'CCO'
        mol = Chem.MolFromSmiles(mol_smiles)
        cleaned = clean_fragment(mol)

        # Should be unchanged
        assert cleaned.GetNumAtoms() == mol.GetNumAtoms()

    def test_generate_colors_count(self):
        """Test that generate_colors returns correct number of colors."""
        num_colors = 10
        colors = generate_colors(num_colors, seed=42)

        assert len(colors) == num_colors
        # Each color should be RGBA tuple
        for color in colors:
            assert len(color) == 4  # R, G, B, A

    def test_generate_colors_rgba_range(self):
        """Test that generated colors have values in valid range [0, 1]."""
        colors = generate_colors(num_colors=5, seed=123)

        for r, g, b, a in colors:
            assert 0 <= r <= 1
            assert 0 <= g <= 1
            assert 0 <= b <= 1
            assert 0 <= a <= 1

    def test_generate_colors_reproducible(self):
        """Test that same seed produces same colors."""
        colors1 = generate_colors(num_colors=5, seed=42)
        colors2 = generate_colors(num_colors=5, seed=42)

        assert colors1 == colors2

    def test_generate_colors_different_seeds(self):
        """Test that different seeds produce different colors."""
        colors1 = generate_colors(num_colors=5, seed=42)
        colors2 = generate_colors(num_colors=5, seed=123)

        assert colors1 != colors2

    def test_sort_mols_descending(self, simple_smiles_list):
        """Test sorting molecules by atom count (largest first)."""
        mols = [Chem.MolFromSmiles(s) for s in simple_smiles_list]
        sorted_mols = sort_mols_atom_num(mols, reverse=True)

        # Check descending order
        atom_counts = [mol.GetNumAtoms() for mol in sorted_mols]
        assert atom_counts == sorted(atom_counts, reverse=True)

    def test_sort_mols_ascending(self, simple_smiles_list):
        """Test sorting molecules by atom count (smallest first)."""
        mols = [Chem.MolFromSmiles(s) for s in simple_smiles_list]
        sorted_mols = sort_mols_atom_num(mols, reverse=False)

        # Check ascending order
        atom_counts = [mol.GetNumAtoms() for mol in sorted_mols]
        assert atom_counts == sorted(atom_counts)


class TestFragmentHighlighting:
    """Tests for fragment highlighting visualization."""

    def test_highlight_fragments_basic(self, gsge_with_descriptors):
        """Test basic fragment highlighting for simple molecule."""
        mol = Chem.MolFromSmiles('CCO')
        img = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            verbose=False
        )

        # Should return PIL Image
        assert isinstance(img, Image.Image)
        # Image should have reasonable dimensions
        assert img.size[0] > 0
        assert img.size[1] > 0

    def test_highlight_fragments_custom_size(self, gsge_with_descriptors):
        """Test fragment highlighting with custom image size."""
        mol = Chem.MolFromSmiles('CCO')
        custom_size = (1200, 900)

        img = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            img_size=custom_size,
            verbose=False
        )

        # Image size should match custom size
        assert img.size == custom_size

    def test_highlight_fragments_with_annotations(self, gsge_with_descriptors):
        """Test fragment highlighting with atom index annotations."""
        mol = Chem.MolFromSmiles('CCO')

        img_with_idx = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            annotate_atoms=True,
            annotate_with_index=False,
            verbose=False
        )

        assert isinstance(img_with_idx, Image.Image)

    def test_highlight_fragments_same_color_mode(self, gsge_with_descriptors):
        """Test fragment highlighting with same color for identical fragments."""
        # Molecule with repeated fragments
        mol = Chem.MolFromSmiles('CC(C)C')  # Has multiple C fragments

        img = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            same_color_for_same_fragment=True,
            verbose=False
        )

        assert isinstance(img, Image.Image)

    def test_highlight_fragments_different_color_mode(self, gsge_with_descriptors):
        """Test fragment highlighting with different colors for each occurrence."""
        mol = Chem.MolFromSmiles('CC(C)C')

        img = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            same_color_for_same_fragment=False,
            verbose=False
        )

        assert isinstance(img, Image.Image)

    def test_highlight_fragments_reproducible_colors(self, gsge_with_descriptors):
        """Test that same color seed produces reproducible highlighting."""
        mol = Chem.MolFromSmiles('CCO')

        img1 = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            color_seed=42,
            verbose=False
        )

        img2 = highlight_fragments(
            mol,
            vocab=gsge_with_descriptors.vocab_manager.GS_vocab,
            color_seed=42,
            verbose=False
        )

        # Images should be identical (same seed)
        assert list(img1.getdata()) == list(img2.getdata())

    def test_highlight_fragments_invalid_mol_raises(self, gsge_with_descriptors):
        """Test that invalid molecule raises appropriate error."""
        with pytest.raises(ValueError, match="must be an RDKit molecule"):
            highlight_fragments(
                "not_a_mol",  # String instead of Mol object
                vocab=gsge_with_descriptors.vocab_manager.GS_vocab
            )
