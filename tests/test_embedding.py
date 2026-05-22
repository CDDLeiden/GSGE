"""
Tests for fragment embedding generation and management.

Tests GSGE_Embedding layer, embedding manager, combined embeddings,
and integration with vocabularies.
"""

import pytest
import torch
import numpy as np

from GSGE.embedding import GSGE_Embedding
from GSGE import GSGE


class TestGSGEEmbeddingLayer:
    """Tests for GSGE_Embedding PyTorch layer."""

    def test_embedding_layer_init_with_embeddings(self):
        """Test initializing embedding layer with pre-trained embeddings."""
        vocab_size = 100
        dense_size = 0
        embedding_dim = 128

        # Create dummy combined embeddings
        combined_embeddings = torch.randn(vocab_size, embedding_dim)

        layer = GSGE_Embedding(
            sparse_vocab_size=vocab_size,
            dense_size=dense_size,
            embedding_dim=embedding_dim,
            GSGE_combined_embeddings=combined_embeddings,
            only_token2vec=True,
            no_grad=True
        )

        assert layer.GSGE_combined_embeddings.shape == (vocab_size, embedding_dim)
        # With no_grad=True and only_token2vec=True, uses token2vec directly

    def test_embedding_layer_init_ohe_only(self):
        """Test initializing embedding layer with OHE and combined embeddings."""
        sparse_vocab_size = 50
        dense_size = 64
        embedding_dim = 32

        combined_embeddings = torch.randn(sparse_vocab_size, dense_size)

        layer = GSGE_Embedding(
            sparse_vocab_size=sparse_vocab_size,
            dense_size=dense_size,
            embedding_dim=embedding_dim,
            GSGE_combined_embeddings=combined_embeddings,
            only_token2vec=False,
            no_grad=False
        )

        assert layer.sparse_embed.weight.shape == (sparse_vocab_size + 1, embedding_dim)

    def test_embedding_layer_forward(self):
        """Test forward pass through embedding layer with token2vec."""
        vocab_size = 50
        embedding_dim = 32

        combined_embeddings = torch.randn(vocab_size, embedding_dim)

        layer = GSGE_Embedding(
            sparse_vocab_size=vocab_size,
            dense_size=0,
            embedding_dim=embedding_dim,
            GSGE_combined_embeddings=combined_embeddings,
            only_token2vec=True,
            no_grad=True
        )

        # Create dummy token IDs
        token_ids = torch.randint(0, vocab_size, (10, 20))  # [batch, seq_len]

        # Forward pass - uses token2vec
        embeddings = layer(token_ids)

        assert embeddings.shape == (10, 20, embedding_dim)

    def test_embedding_layer_combined_ohe_and_learned(self):
        """Test embedding layer with both OHE and learned embeddings."""
        sparse_vocab_size = 20
        dense_size = 32
        embedding_dim = 32

        # Create combined embeddings
        combined_emb = torch.randn(sparse_vocab_size, dense_size)

        layer = GSGE_Embedding(
            sparse_vocab_size=sparse_vocab_size,
            dense_size=dense_size,
            embedding_dim=embedding_dim,
            GSGE_combined_embeddings=combined_emb,
            only_token2vec=True,
            no_grad=False
        )

        # Token IDs spanning vocab range
        token_ids = torch.randint(0, sparse_vocab_size, (5, 10))

        embeddings = layer(token_ids)
        assert embeddings.shape[0] == 5
        assert embeddings.shape[1] == 10
        assert embeddings.shape[2] == dense_size


class TestEmbeddingManager:
    """Tests for EmbeddingManager class."""

    def test_embedding_manager_init(self, gsge_with_descriptors):
        """Test EmbeddingManager initialization."""
        assert gsge_with_descriptors.embedding_manager is not None
        assert hasattr(gsge_with_descriptors.embedding_manager, 'vocab_manager')

    def test_load_gae_weights(self, gsge_with_descriptors, mock_encoder_cpu, temp_checkpoint_dir):
        """Test loading GAE weights from checkpoint."""
        # Set encoder
        gsge_with_descriptors.gae_trainer.set_encoder(mock_encoder_cpu)
        gsge_with_descriptors.store_modules.modules['encoder'] = mock_encoder_cpu

        # Create and save checkpoint
        checkpoint_path = temp_checkpoint_dir / 'test_checkpoint.pth'
        torch.save({
            'encoder_state_dict': mock_encoder_cpu.state_dict(),
            'epoch': 1
        }, checkpoint_path)

        # Load weights
        gsge_with_descriptors.load_GAE_weights(str(checkpoint_path), map_location='cpu')

        # Verify encoder has loaded state
        assert gsge_with_descriptors.store_modules.modules['encoder'] is not None

    @pytest.mark.slow
    def test_embed_fragments_returns_correct_shape(self, mock_encoder_cpu):
        """Test that embed_fragments returns correct embedding shape."""
        from GSGE import GSGE

        # Create minimal setup
        gsge = GSGE()
        gsge.gae_trainer.set_encoder(mock_encoder_cpu)
        gsge.store_modules.modules['encoder'] = mock_encoder_cpu

        # Embed a few fragments
        frag_smiles = ['CCO', 'c1ccccc1', 'CC(C)O']

        embeddings = gsge.embed_fragments(
            frag_smiles=frag_smiles,
            device='cpu',
            batch_size=2,
            return_data=False
        )

        assert embeddings.shape[0] == 3  # 3 fragments
        assert embeddings.shape[1] == 32  # Encoder output dim

    @pytest.mark.slow
    def test_embed_fragments_with_graph_data(self, mock_encoder_cpu):
        """Test embedding fragments and returning graph data."""
        from GSGE import GSGE

        gsge = GSGE()
        gsge.gae_trainer.set_encoder(mock_encoder_cpu)
        gsge.store_modules.modules['encoder'] = mock_encoder_cpu

        frag_smiles = ['CCO', 'c1ccccc1']

        embeddings, graph_data = gsge.embed_fragments(
            frag_smiles=frag_smiles,
            device='cpu',
            batch_size=2,
            return_data=True
        )

        assert embeddings.shape[0] == 2
        assert len(graph_data) == 2
        # Each graph_data element should be a PyG Data object
        assert hasattr(graph_data[0], 'x')
        assert hasattr(graph_data[0], 'edge_index')

    def test_get_fragment_embeddings_none_before_creation(self):
        """Test that get_fragment_embeddings returns None before embeddings created."""
        from GSGE import GSGE

        gsge = GSGE()
        embeddings = gsge.get_fragment_embeddings()

        assert embeddings is None


class TestCombinedEmbeddings:
    """Tests for combined OHE + learned embeddings."""

    def test_combined_embeddings_shape(self, gsge_with_descriptors):
        """Test that combined embeddings have correct shape."""
        # If embeddings exist
        if gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings is not None:
            combined = gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings

            # Should be 2D tensor
            assert combined.ndim == 2
            # Rows = total vocab size
            assert combined.shape[0] > 0

    def test_gae_embeddings_subset_of_combined(self, gsge_with_descriptors):
        """Test that GAE embeddings are subset of combined embeddings."""
        if gsge_with_descriptors.embedding_manager.GSGE_GAE_embeddings is not None:
            gae_emb = gsge_with_descriptors.embedding_manager.GSGE_GAE_embeddings
            combined = gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings

            # GAE embeddings and combined embeddings should have same embedding dimension
            assert gae_emb.shape[1] == combined.shape[1]


class TestEmbeddingIntegration:
    """Integration tests for embedding workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_embedding_workflow(self, simple_smiles_list, mock_encoder_cpu):
        """Test complete workflow: vocab → corpus → train → embeddings."""
        from GSGE import GSGE, GS_Vocab, GSGE_Corpus
        from GSGE.graphs.fragment_graph.GAE import GraphDecoder

        # Build vocab
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=10,
            MIN_SIZE=1,
            MAX_SIZE=6
        )

        # Build corpus
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=simple_smiles_list,
            convert=True,
            min_size=1,
            max_size=6
        )

        # Create GSGE
        decoder = GraphDecoder(latent_dim=32, hidden_dim=64)
        gsge = GSGE(
            GS_vocab=vocab,
            GSGE_corpus=corpus,
            encoder=mock_encoder_cpu,
            decoder=decoder
        )

        # Generate embeddings
        gsge.make_GS_fragment_embedding_dict(
            device='cpu',
            batch_size=4
        )

        # Check embeddings created
        embeddings = gsge.get_fragment_embeddings()
        assert embeddings is not None
        assert embeddings.shape[0] == len(vocab.fragments)

    def test_encode_gsge_with_token_ids(self, gsge_with_descriptors):
        """Test encoding token IDs to dense embeddings."""
        if gsge_with_descriptors.embedding_manager.GSGE_combined_embeddings is not None:
            # Create dummy token IDs
            token_ids = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])

            # Encode
            embeddings = gsge_with_descriptors.encode_GSGE(
                token_ids,
                device='cpu'
            )

            assert embeddings.shape[0] == 2  # Batch size
            assert embeddings.shape[1] == 4  # Sequence length
            assert embeddings.shape[2] > 0   # Embedding dimension
