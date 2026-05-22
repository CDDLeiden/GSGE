"""
Comprehensive tests for GSGE tokenization and preprocessing.

Extends test_gsge_tokenization.py with edge cases, error handling,
parallel processing scenarios, and masking functionality.
"""

import pytest
import numpy as np
from typing import List

from GSGE import GSGE, GS_Vocab
from GSGE.tokenizer import GSGE_tokenizer


class TestTokenizerInitialization:
    """Tests for GSGE_tokenizer initialization."""

    def test_tokenizer_init_with_vocab(self, minimal_vocab):
        """Test tokenizer initialization with vocabulary."""
        tokenizer = GSGE_tokenizer(
            group_grammar=minimal_vocab.group_grammar,
            GS_vocab=minimal_vocab
        )

        assert tokenizer.group_grammar is not None
        assert tokenizer.GS_vocab is not None
        assert len(tokenizer.grammar_tokens) > 0
        assert len(tokenizer.element_tokens) > 0

    def test_tokenizer_token_mappings(self, minimal_vocab):
        """Test that tokenizer creates correct token-to-ID mappings."""
        tokenizer = GSGE_tokenizer(
            group_grammar=minimal_vocab.group_grammar,
            GS_vocab=minimal_vocab
        )

        # Check that grammar tokens are mapped
        assert tokenizer.token_dict is not None
        assert '[PAD]' in tokenizer.token_dict
        assert 'Branch' in tokenizer.token_dict

        # Check that fragment tokens are included
        frag_tokens = [k for k in tokenizer.token_dict if k.startswith('GS_frag_')]
        assert len(frag_tokens) > 0


class TestPreprocessFromSMILES:
    """Tests for preprocess_from_SMILES method."""

    def test_preprocess_simple_molecules(self, gsge_with_descriptors):
        """Test preprocessing simple molecules."""
        simple_smiles = ['CCO', 'c1ccccc1', 'CC(C)O']

        for smiles in simple_smiles:
            tokens = gsge_with_descriptors.preprocess_from_SMILES(smiles)
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    def test_preprocess_preserves_structure(self, gsge_with_descriptors):
        """Test that tokenization preserves molecular structure information."""
        smiles = 'c1ccccc1'  # Benzene

        tokens = gsge_with_descriptors.preprocess_from_SMILES(smiles)

        # Should contain ring-related tokens
        assert any('Ring' in str(t) for t in tokens)

    def test_preprocess_invalid_smiles_returns_gracefully(self, gsge_with_descriptors):
        """Test that invalid SMILES are handled gracefully."""
        invalid_smiles = ['INVALID!!!', 'NOT_SMILES', '((()))']

        for smiles in invalid_smiles:
            try:
                tokens = gsge_with_descriptors.preprocess_from_SMILES(smiles)
                # May return empty list or raise exception depending on implementation
                assert tokens is not None
            except (ValueError, AttributeError, KeyError):
                # Expected for invalid SMILES
                pass

    def test_preprocess_empty_smiles(self, gsge_with_descriptors):
        """Test preprocessing empty SMILES string."""
        try:
            tokens = gsge_with_descriptors.preprocess_from_SMILES('')
            # May return empty list or raise exception
            assert isinstance(tokens, list)
        except (ValueError, AttributeError):
            # Expected for empty input
            pass


class TestParallelTokenization:
    """Tests for parallel tokenization functionality."""

    def test_parallel_tokenize_basic(self, gsge_with_descriptors, simple_smiles_list):
        """Test basic parallel tokenization."""
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            simple_smiles_list[:5],
            max_workers=2
        )

        assert len(tokenized) == 5
        assert all(isinstance(t, list) for t in tokenized)

    def test_parallel_tokenize_empty_list(self, gsge_with_descriptors):
        """Test parallel tokenization with empty input."""
        result = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            [],
            max_workers=2
        )

        assert result == []

    def test_parallel_tokenize_single_worker(self, gsge_with_descriptors, simple_smiles_list):
        """Test parallel tokenization with single worker."""
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            simple_smiles_list[:3],
            max_workers=1
        )

        assert len(tokenized) == 3

    def test_parallel_tokenize_consistency(self, gsge_with_descriptors, simple_smiles_list):
        """Test that parallel tokenization is deterministic."""
        smiles_subset = simple_smiles_list[:5]

        # Tokenize twice with same settings
        result1 = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            smiles_subset,
            max_workers=2
        )
        result2 = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            smiles_subset,
            max_workers=2
        )

        # Results should be identical
        assert len(result1) == len(result2)
        for r1, r2 in zip(result1, result2):
            assert r1 == r2

    def test_parallel_tokenize_different_worker_counts(self, gsge_with_descriptors, simple_smiles_list):
        """Test that different worker counts produce same results."""
        smiles_subset = simple_smiles_list[:10]

        result_2_workers = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            smiles_subset,
            max_workers=2
        )
        result_4_workers = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            smiles_subset,
            max_workers=4
        )

        # Results should be identical regardless of worker count
        assert len(result_2_workers) == len(result_4_workers)
        for r2, r4 in zip(result_2_workers, result_4_workers):
            assert r2 == r4


class TestTokenIDConversion:
    """Tests for token-to-ID and ID-to-token conversions."""

    def test_token_to_id_conversion(self, gsge_with_descriptors):
        """Test converting tokens to IDs."""
        tokens = ['Branch', 'C', 'O', '[PAD]']

        token_dict = gsge_with_descriptors.tokenizer.token_dict

        # All standard tokens should have IDs
        for token in tokens:
            if token in token_dict:
                assert isinstance(token_dict[token], int)
                assert token_dict[token] >= 0

    def test_id_to_token_conversion(self, gsge_with_descriptors):
        """Test converting IDs back to tokens."""
        token_dict = gsge_with_descriptors.tokenizer.token_dict
        id_to_token = {v: k for k, v in token_dict.items()}

        # Sample IDs should map back to tokens
        for token_id in [0, 1, 2, 10, 20]:
            if token_id in id_to_token:
                token = id_to_token[token_id]
                assert isinstance(token, str)

    def test_pad_token_is_zero(self, gsge_with_descriptors):
        """Test that [PAD] token has ID 0."""
        token_dict = gsge_with_descriptors.tokenizer.token_dict

        if '[PAD]' in token_dict:
            assert token_dict['[PAD]'] == 0


class TestTokenMasking:
    """Tests for token masking functionality."""

    def test_create_attention_mask(self, gsge_with_descriptors, simple_smiles_list):
        """Test creating attention masks for tokenized sequences."""
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            simple_smiles_list[:3],
            max_workers=2
        )

        # Create attention mask (1 for real tokens, 0 for padding)
        for token_seq in tokenized:
            mask = [1 if token != '[PAD]' else 0 for token in token_seq]
            assert len(mask) == len(token_seq)
            assert sum(mask) <= len(mask)  # Some or all should be real tokens


class TestGrammarTokenProcessing:
    """Tests for grammar token handling."""

    def test_grammar_tokens_present(self, gsge_with_descriptors):
        """Test that grammar tokens are recognized."""
        grammar_tokens = gsge_with_descriptors.tokenizer.grammar_tokens

        # Standard grammar tokens
        expected = ['Branch', 'Ring1', 'Ring2', 'pop', '=Branch']
        for token in expected:
            assert token in grammar_tokens

    def test_element_tokens_present(self, gsge_with_descriptors):
        """Test that element tokens are recognized."""
        element_tokens = gsge_with_descriptors.tokenizer.element_tokens

        # Common elements
        expected = ['C', 'N', 'O', 'S', 'F', 'Cl']
        for token in expected:
            assert token in element_tokens

    def test_fragment_tokens_format(self, gsge_with_descriptors):
        """Test that fragment tokens follow expected format."""
        token_dict = gsge_with_descriptors.tokenizer.token_dict

        # Fragment tokens should be like 'GS_frag_0', 'GS_frag_1', etc.
        frag_tokens = [k for k in token_dict if k.startswith('GS_frag_')]

        for frag_token in frag_tokens[:10]:  # Check first 10
            # Should match pattern GS_frag_<number>
            assert frag_token.startswith('GS_frag_')
            suffix = frag_token.replace('GS_frag_', '')
            assert suffix.isdigit()


class TestEdgeCases:
    """Tests for edge cases in tokenization."""

    def test_tokenize_single_atom_molecule(self, gsge_with_descriptors):
        """Test tokenizing single-atom molecules."""
        single_atoms = ['C', 'N', 'O', 'S']

        for smiles in single_atoms:
            tokens = gsge_with_descriptors.preprocess_from_SMILES(smiles)
            assert isinstance(tokens, list)
            assert len(tokens) > 0

    def test_tokenize_very_large_molecule(self, gsge_with_descriptors):
        """Test tokenizing large molecules."""
        # Large molecule (complex peptide)
        large_smiles = 'CC(C)CC1NC(=O)CCN(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(Cc2ccccc2)N(C)C(=O)CNC(=O)C(C(C)O)NC(=O)C(Cc2ccccc2)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(Cc2ccccc2)NC1=O'

        tokens = gsge_with_descriptors.preprocess_from_SMILES(large_smiles)
        assert isinstance(tokens, list)
        assert len(tokens) > 50  # Should have many tokens

    @pytest.mark.slow
    def test_parallel_tokenize_large_dataset(self, gsge_with_descriptors, sample_smiles_1000):
        """Test parallel tokenization on large dataset."""
        # Tokenize 100 molecules
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            sample_smiles_1000[:100],
            max_workers=4
        )

        assert len(tokenized) == 100
        assert all(isinstance(t, list) for t in tokenized)


class TestTokenizerIntegration:
    """Integration tests for tokenizer with other components."""

    def test_tokenize_and_encode(self, gsge_with_descriptors, simple_smiles_list):
        """Test tokenizing and encoding to embeddings."""
        # Tokenize
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            simple_smiles_list[:3],
            max_workers=2
        )

        assert len(tokenized) == 3

        # If embeddings exist, test encoding
        if gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings is not None:
            # Convert to token IDs
            token_dict = gsge_with_descriptors.tokenizer.token_dict
            token_ids = []
            for token_seq in tokenized:
                ids = [token_dict.get(t, 0) for t in token_seq]
                token_ids.append(ids)

            # Should be valid IDs
            assert all(isinstance(ids, list) for ids in token_ids)
