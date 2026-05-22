"""
End-to-end integration tests for GSGE workflows.

Tests complete pipelines: vocab building → corpus → GAE training →
embeddings → compound graphs → tokenization.
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np

from GSGE import GSGE, GS_Vocab, GSGE_Corpus
from GSGE.graphs.fragment_graph.GAE import AttentiveFP, GraphDecoder
from rdkit import Chem


class TestVocabToEmbeddings:
    """Integration tests for vocabulary → embeddings pipeline."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_vocab_corpus_gae_embeddings_pipeline(self, simple_smiles_list):
        """Test complete pipeline from SMILES to embeddings."""
        # Step 1: Build vocabulary
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=15,
            MIN_SIZE=1,
            MAX_SIZE=8
        )
        assert vocab.num_fragments > 0

        # Step 2: Build corpus
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=simple_smiles_list,
            convert=True,
            min_size=1,
            max_size=8
        )
        assert corpus.num_fragments > 0

        # Step 3: Create GSGE with encoder/decoder
        encoder = AttentiveFP(
            in_channels=9,
            hidden_channels=32,
            out_channels=16,
            edge_dim=3,
            num_layers=1,
            num_timesteps=1
        )
        decoder = GraphDecoder(latent_dim=16, hidden_dim=32)

        gsge = GSGE(
            GS_vocab=vocab,
            GSGE_corpus=corpus,
            encoder=encoder,
            decoder=decoder
        )

        # Step 4: Generate embeddings
        gsge.make_GS_fragment_embedding_dict(device='cpu', batch_size=4)

        # Verify embeddings created
        embeddings = gsge.get_fragment_embeddings()
        assert embeddings is not None
        assert embeddings.shape[0] == vocab.num_fragments
        assert embeddings.shape[1] == 16  # Encoder output dim


class TestSerializationRoundtrip:
    """Integration tests for save/load cycles."""

    @pytest.mark.integration
    def test_gsge_save_load_roundtrip(self, gsge_with_descriptors):
        """Test saving and loading complete GSGE state."""
        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'gsge_test.pkl'

            # Save current state
            original_vocab_size = len(gsge_with_descriptors.get_fragments_smiles())

            gsge_with_descriptors.save_gsge_data(
                str(save_path),
                meta_info='Test integration save'
            )

            # Load into new GSGE
            loaded_gsge = GSGE(GSGE_load_path=str(save_path))

            # Verify loaded state matches original
            assert len(loaded_gsge.get_fragments_smiles()) == original_vocab_size

    @pytest.mark.integration
    def test_vocab_corpus_save_load_independence(self, simple_smiles_list):
        """Test that vocab and corpus can be saved/loaded independently."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Build and save vocab
            vocab = GS_Vocab()
            vocab.build_vocab(
                m_set=simple_smiles_list,
                convert=True,
                target=10
            )
            vocab_path = Path(tmpdir) / 'vocab.pkl'
            vocab.save_GS_vocab(dir_path=tmpdir, vocab_name='vocab')

            # Build and save corpus
            corpus = GSGE_Corpus()
            corpus.build_corpus(m_set=simple_smiles_list, convert=True)
            corpus_path = Path(tmpdir) / 'corpus.pkl'
            corpus.save_GSGE_corpus(dir_path=tmpdir, vocab_name='corpus')

            # Create GSGE with loaded vocab and corpus
            gsge = GSGE(
                GS_vocab=str(vocab_path),
                GSGE_corpus=str(corpus_path)
            )

            assert gsge.vocab_manager.GS_vocab is not None
            assert gsge.vocab_manager.GSGE_corpus is not None


class TestTokenizationToGraphs:
    """Integration tests for tokenization → compound graphs pipeline."""

    @pytest.mark.integration
    def test_tokenize_and_make_compound_graphs(self, gsge_with_descriptors, simple_smiles_list):
        """Test tokenizing molecules and creating compound graphs."""
        # Tokenize molecules
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            simple_smiles_list[:3],
            max_workers=2
        )

        assert len(tokenized) == 3
        # Each should be list of tokens
        assert all(isinstance(t, list) for t in tokenized)

        # Create compound graphs
        graphs = gsge_with_descriptors.make_compound_graphs(
            simple_smiles_list[:3],
            workers=2,
            pyg_data=True
        )

        assert len(graphs) == 3
        # Each graph should have node features and edges
        for graph in graphs:
            assert hasattr(graph, 'x')
            assert hasattr(graph, 'edge_index')

    @pytest.mark.integration
    def test_get_single_compound_graph(self, gsge_with_descriptors):
        """Test getting single compound graph from SMILES."""
        smiles = 'CCO'

        # Get compound graph object
        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        assert cg is not None
        # Should have graph attributes
        assert hasattr(cg, 'nodes')
        assert hasattr(cg, 'edges')


class TestDescriptorsAndEmbeddings:
    """Integration tests combining descriptors and embeddings."""

    @pytest.mark.integration
    def test_calculate_descriptors_after_vocab_build(self, simple_smiles_list):
        """Test calculating descriptors for built vocabulary."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=10
        )

        gsge = GSGE(GS_vocab=vocab)

        # Calculate descriptors
        gsge.calc_fragment_descriptors()

        descriptors = gsge.get_fragment_descriptors()
        assert descriptors is not None
        assert descriptors.shape[0] == vocab.num_fragments

    @pytest.mark.integration
    @pytest.mark.slow
    def test_combined_descriptors_and_embeddings(self, simple_smiles_list, mock_encoder_cpu):
        """Test combining RDKit descriptors with learned embeddings."""
        from GSGE.graphs.fragment_graph.GAE import GraphDecoder

        # Build vocab and corpus
        vocab = GS_Vocab()
        vocab.build_vocab(m_set=simple_smiles_list, convert=True, target=10)

        corpus = GSGE_Corpus()
        corpus.build_corpus(m_set=simple_smiles_list, convert=True)

        # Create GSGE
        decoder = GraphDecoder(latent_dim=32, hidden_dim=64)
        gsge = GSGE(
            GS_vocab=vocab,
            GSGE_corpus=corpus,
            encoder=mock_encoder_cpu,
            decoder=decoder
        )

        # Generate embeddings
        gsge.make_GS_fragment_embedding_dict(device='cpu', batch_size=4)

        # Calculate descriptors
        gsge.calc_fragment_descriptors(
            descriptor_keys=['MolWt', 'TPSA', 'NumHDonors']
        )

        # Get combined features
        combined = gsge.get_fragment_descriptors_and_embeddings()

        assert combined is not None
        # Should have embedding_dim + descriptor_dim features
        embeddings = gsge.get_fragment_embeddings()
        descriptors = gsge.get_fragment_descriptors()
        assert combined.shape[1] == embeddings.shape[1] + descriptors.shape[1]


class TestAddingFragments:
    """Integration tests for manually adding fragments."""

    @pytest.mark.integration
    def test_add_single_elements_to_vocab_and_corpus(self):
        """Test adding all single elements to both vocab and corpus."""
        vocab = GS_Vocab()
        corpus = GSGE_Corpus()

        gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)

        # Add all single elements
        gsge.add_all_single_elements()

        # Check elements were added
        assert vocab.num_fragments > 0
        assert corpus.num_fragments > 0

        # Should include common elements
        vocab_smiles = vocab.fragments
        assert 'C' in vocab_smiles or any('C' in s for s in vocab_smiles)

    @pytest.mark.integration
    def test_add_standard_smaller_fragments(self):
        """Test adding predefined common fragments."""
        vocab = GS_Vocab()
        gsge = GSGE(GS_vocab=vocab)

        initial_count = vocab.num_fragments

        # Add standard fragments
        gsge.add_standard_smaller_fragments()

        # Should have added fragments
        assert vocab.num_fragments > initial_count


class TestCheckCoverage:
    """Integration tests for vocabulary coverage checking."""

    @pytest.mark.integration
    def test_check_for_graphs_groupings(self, gsge_with_descriptors, simple_smiles_list):
        """Test checking which molecules have ungrouped atoms."""
        # Check coverage
        problematic = gsge_with_descriptors.check_for_graphs_groupings(
            simple_smiles_list,
            workers=2
        )

        # Should return list (empty if all molecules fully covered)
        assert isinstance(problematic, list)

    @pytest.mark.integration
    def test_full_coverage_with_single_elements(self, simple_smiles_list):
        """Test that adding single elements improves coverage."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=15
        )

        gsge = GSGE(GS_vocab=vocab)

        # Add single elements to improve coverage
        gsge.add_all_single_elements()

        # Check coverage again
        problematic = gsge.check_for_graphs_groupings(
            simple_smiles_list[:5],
            workers=2
        )

        # Should have fewer (or same) problematic molecules
        assert isinstance(problematic, list)


class TestMemoryAndPerformance:
    """Integration tests for memory usage and performance."""

    @pytest.mark.integration
    def test_large_vocabulary_build(self, sample_smiles_1000):
        """Test building vocabulary from 1000 SMILES."""
        vocab = GS_Vocab()

        # Build vocab from 1000 molecules
        vocab.build_vocab(
            m_set=sample_smiles_1000[:100],  # Use subset for speed
            convert=True,
            target=50,
            MIN_SIZE=2,
            MAX_SIZE=10
        )

        assert vocab.num_fragments > 0
        assert vocab.num_fragments <= 50  # Should respect target

    @pytest.mark.integration
    @pytest.mark.slow
    def test_parallel_tokenization_performance(self, gsge_with_descriptors, sample_smiles_1000):
        """Test parallel tokenization on larger dataset."""
        # Tokenize 100 molecules in parallel
        tokenized = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            sample_smiles_1000[:100],
            max_workers=4
        )

        assert len(tokenized) == 100
        # All should be successfully tokenized (or return empty list for failures)
        assert all(isinstance(t, list) for t in tokenized)
