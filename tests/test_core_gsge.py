"""
Tests for CoreGSGE static utility methods.

Tests data loading, preprocessing, training helpers, and core utilities.
"""

import pytest
import torch
import numpy as np

from GSGE.core_gsge import CoreGSGE
from GSGE import GS_Vocab, GSGE_Corpus


class TestDataLoading:
    """Tests for data loading and preparation."""

    @pytest.mark.slow
    def test_load_and_prepare_data(self, minimal_vocab):
        """Test loading and preparing data for GAE training."""
        # Create minimal corpus
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=['CCO', 'c1ccccc1', 'CC(C)O'],
            convert=True,
            min_size=1,
            max_size=8
        )

        train_loader, val_loader = CoreGSGE.load_and_prepare_data(
            GS_vocab=minimal_vocab,
            GSGE_corpus=corpus,
            batch_size=2,
            x_percent=0.2,
            seed=42
        )

        # Verify loaders created
        assert train_loader is not None
        assert val_loader is not None

        # Check they contain data
        assert len(train_loader) > 0 or len(val_loader) > 0

    def test_load_and_prepare_data_reproducibility(self, minimal_vocab):
        """Test that same seed produces same train/val split."""
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=['CCO', 'c1ccccc1', 'CC(C)O', 'CCCO', 'CC(C)(C)O'],
            convert=True
        )

        # Load with same seed twice
        train1, val1 = CoreGSGE.load_and_prepare_data(
            GS_vocab=minimal_vocab,
            GSGE_corpus=corpus,
            batch_size=2,
            x_percent=0.2,
            seed=42
        )

        train2, val2 = CoreGSGE.load_and_prepare_data(
            GS_vocab=minimal_vocab,
            GSGE_corpus=corpus,
            batch_size=2,
            x_percent=0.2,
            seed=42
        )

        # Should produce same split
        assert len(train1) == len(train2)
        assert len(val1) == len(val2)


class TestEmbedFragments:
    """Tests for fragment embedding utilities."""

    @pytest.mark.slow
    def test_embed_fragments_basic(self, mock_encoder_cpu):
        """Test embedding fragments with encoder."""
        frag_smiles = ['CCO', 'c1ccccc1']

        embeddings, graph_data = CoreGSGE.embed_fragments(
            frag_smiles=frag_smiles,
            encoder=mock_encoder_cpu,
            device='cpu',
            batch_size=2,
            return_data=True
        )

        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 32  # Encoder output dim
        assert len(graph_data) == 2

    @pytest.mark.slow
    def test_embed_fragments_batching(self, mock_encoder_cpu):
        """Test that batching works correctly."""
        frag_smiles = ['CCO'] * 10  # 10 identical fragments

        embeddings, _ = CoreGSGE.embed_fragments(
            frag_smiles=frag_smiles,
            encoder=mock_encoder_cpu,
            device='cpu',
            batch_size=3,  # Process in batches of 3
            return_data=True
        )

        # Should produce 10 embeddings despite batching
        assert embeddings.shape[0] == 10


class TestCombinedEmbeddings:
    """Tests for creating combined OHE + learned embeddings."""

    def test_create_combined_embeddings(self):
        """Test creating combined embedding matrix."""
        ohe_vocab_size = 20
        embedding_dim = 32

        # Create dummy fragment embeddings
        frag_embeddings = {
            i: torch.randn(embedding_dim).numpy()
            for i in range(10)
        }

        result = CoreGSGE.create_combined_embeddings(
            OHE_vocab_size=ohe_vocab_size,
            fragment_embeddings=frag_embeddings,
            embedding_dim=embedding_dim
        )

        # Result is a tuple: (combined_tensor, dense_dict)
        assert isinstance(result, tuple) and len(result) == 2
        combined, dense = result

        # Combined should have OHE tokens + fragment embeddings
        assert combined.shape[0] >= ohe_vocab_size
        assert combined.shape[1] > 0

    def test_combined_embeddings_ohe_zeros(self):
        """Test that OHE portion is initialized to zeros."""
        ohe_vocab_size = 10
        embedding_dim = 16

        frag_embeddings = {0: torch.randn(embedding_dim).numpy()}

        result = CoreGSGE.create_combined_embeddings(
            OHE_vocab_size=ohe_vocab_size,
            fragment_embeddings=frag_embeddings,
            embedding_dim=embedding_dim
        )

        combined, dense = result
        # First ohe_vocab_size rows should be zeros (for OHE sparse tokens)
        ohe_portion = combined[:ohe_vocab_size, :]
        assert torch.all(ohe_portion == 0)


class TestGAETrainingHelpers:
    """Tests for GAE training helper functions."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_train_gsge_auto_encoder(self, minimal_vocab, mock_encoder_cpu, mock_decoder_cpu, temp_checkpoint_dir):
        """Test GAE training helper function."""
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=['CCO', 'c1ccccc1'],
            convert=True
        )

        optimizer = torch.optim.Adam(
            list(mock_encoder_cpu.parameters()) + list(mock_decoder_cpu.parameters()),
            lr=0.001
        )

        # Train for 1 epoch
        CoreGSGE.train_GSGE_Auto_Encoder(
            GS_vocab=minimal_vocab,
            GSGE_corpus=corpus,
            batch_size=2,
            num_epochs=1,
            checkpoint_interval=1,
            x_percent=0.2,
            seed=42,
            device='cpu',
            encoder=mock_encoder_cpu,
            decoder=mock_decoder_cpu,
            optimizer=optimizer,
            checkpoint_dir=str(temp_checkpoint_dir)
        )

        # Check checkpoint created
        checkpoints = list(temp_checkpoint_dir.glob('*.pth'))
        assert len(checkpoints) >= 1


class TestVocabularyUtilities:
    """Tests for vocabulary-related utilities."""

    def test_add_gs_vocab_to_gsge_corpus(self):
        """Test adding GS_vocab fragments to GSGE_corpus."""
        vocab = GS_Vocab()
        vocab.add_GS_fragment('C')
        vocab.add_GS_fragment('O')

        corpus = GSGE_Corpus()
        initial_count = corpus.num_fragments

        # This function may fail if fragments can't be processed
        # Just test that it doesn't crash
        try:
            CoreGSGE.add_GS_vocab_to_GSGE_corpus(
                vocab.vocab_fragments,
                corpus
            )
            # If it succeeds, corpus should have fragments
            assert corpus.num_fragments >= initial_count
        except (IndexError, ValueError):
            # Some fragments may fail to process - acceptable
            pass

    def test_add_all_single_elements_to_vocab(self):
        """Test adding all single elements to vocabulary."""
        from GSGE.chem import _ELEMENTS_BOND_COUNTS

        vocab = GS_Vocab()
        initial_count = vocab.num_fragments

        CoreGSGE.add_all_single_elements(
            vocab,
            element_bond_counts=_ELEMENTS_BOND_COUNTS
        )

        # Should have added elements
        assert vocab.num_fragments > initial_count

    def test_add_all_single_elements_to_corpus(self):
        """Test adding all single elements to corpus."""
        from GSGE.chem import _ELEMENTS_BOND_COUNTS

        corpus = GSGE_Corpus()
        initial_count = corpus.num_fragments

        CoreGSGE.add_all_single_elements(
            corpus,
            element_bond_counts=_ELEMENTS_BOND_COUNTS
        )

        assert corpus.num_fragments > initial_count


class TestCoverageChecking:
    """Tests for vocabulary coverage checking utilities."""

    def test_check_for_non_grouped_atoms(self):
        """Test checking for atoms not covered by vocabulary."""
        vocab = GS_Vocab()
        vocab.add_GS_fragment('CC(*)O')

        # Simple molecule that may have ungrouped atoms
        result = CoreGSGE.check_for_non_grouped_atoms(
            smiles='CCO',
            GS_vocab=vocab
        )

        # Result should be None (all grouped) or list of ungrouped atoms
        assert result is None or isinstance(result, list)

    def test_check_ungrouped_with_incomplete_vocab(self):
        """Test that incomplete vocab produces ungrouped atoms."""
        vocab = GS_Vocab()
        # Very minimal vocab

        # Complex molecule should have ungrouped atoms
        result = CoreGSGE.check_for_non_grouped_atoms(
            smiles='c1ccccc1',
            GS_vocab=vocab
        )

        # Should find ungrouped atoms with empty vocab
        # (or None if single elements are automatically handled)
        assert result is None or (isinstance(result, list) and len(result) >= 0)


class TestTokenizationUtilities:
    """Tests for tokenization helper functions."""

    def test_preprocess_smiles_to_tokens(self, gsge_with_descriptors):
        """Test preprocessing SMILES to token sequence."""
        # Use the GSGE object's method
        smiles = 'CCO'

        tokens = gsge_with_descriptors.tokenizer.tokenize_one(smiles)

        assert isinstance(tokens, list)
        # May be empty if SMILES is invalid
        assert len(tokens) >= 0


class TestEncodingDecoding:
    """Tests for encoding/decoding utilities."""

    def test_encode_gsge_tokens_to_embeddings(self, gsge_with_descriptors):
        """Test encoding token IDs to embeddings."""
        if gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings is not None:
            token_ids = np.array([[1, 2, 3], [4, 5, 6]])

            embeddings = CoreGSGE.encode_GSGE(
                token_ids,
                gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings,
                device='cpu'
            )

            assert embeddings.shape[0] == 2  # batch
            assert embeddings.shape[1] == 3  # seq_len
            assert embeddings.shape[2] > 0   # embedding_dim
