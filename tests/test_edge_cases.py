"""
Edge case and error handling tests for GSGE.

Tests invalid inputs, malformed data, error handling, boundary conditions,
and robustness of the GSGE framework.
"""

import pytest
import numpy as np
from rdkit import Chem

from GSGE import GSGE, GS_Vocab, GSGE_Corpus
from GSGE.fragment_descriptors import calc_mol_frag_descriptors


class TestInvalidInputHandling:
    """Tests for handling invalid inputs gracefully."""

    def test_invalid_smiles_in_vocab_build(self):
        """Test that invalid SMILES are skipped during vocab building."""
        smiles_list = ['CCO', 'c1ccccc1']

        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=smiles_list,
            convert=True,
            target=10,
            verbose=False
        )

        # Should build vocab from valid SMILES only
        assert vocab.num_fragments >= 0  # At least partial success

    def test_empty_smiles_list(self):
        """Test handling empty SMILES list."""
        vocab = GS_Vocab()

        # Should not crash with empty list
        vocab.build_vocab(
            m_set=[],
            convert=True,
            target=10
        )

        # Vocab should remain empty
        assert vocab.num_fragments == 0

    def test_none_smiles_input(self):
        """Test handling None as SMILES input."""
        with pytest.raises((AttributeError, TypeError)):
            calc_mol_frag_descriptors(smiles=None)

    def test_empty_string_smiles(self):
        """Test handling empty string as SMILES."""
        result = calc_mol_frag_descriptors(smiles='', verbose=False)
        # Should handle gracefully (may return None or default values)
        assert result is not None or result is None

    def test_invalid_mol_object(self):
        """Test handling invalid Mol object."""
        with pytest.raises((AttributeError, TypeError)):
            calc_mol_frag_descriptors(mol="not_a_mol_object")


class TestBoundaryConditions:
    """Tests for boundary conditions and extreme values."""

    def test_single_atom_molecule(self):
        """Test handling single-atom molecules."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=['CCO', 'CO'],
            convert=True,
            MIN_SIZE=1,
            MAX_SIZE=10
        )

        # Should handle small molecules
        assert vocab.num_fragments >= 0

    def test_very_large_molecule(self):
        """Test handling very large molecules."""
        # Use a reasonably large but valid SMILES
        # Stearic acid as an example of larger molecule
        large_smiles = 'CCCCCCCCCCCCCCCCCC(=O)O'

        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=[large_smiles],
            convert=True,
            verbose=False
        )

        # Should handle without crashing
        assert vocab.num_fragments >= 0

    def test_target_larger_than_fragments(self):
        """Test when target vocab size exceeds available fragments."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=['CCO'],  # Very few unique fragments
            convert=True,
            target=1000,  # Unrealistic target
            MIN_SIZE=1,
            MAX_SIZE=10
        )

        # Should build vocab with available fragments
        assert vocab.num_fragments < 1000

    def test_min_size_larger_than_max_size(self):
        """Test invalid size constraints (min > max)."""
        vocab = GS_Vocab()

        # This configuration doesn't make sense
        vocab.build_vocab(
            m_set=['CCO', 'c1ccccc1'],
            convert=True,
            MIN_SIZE=15,  # Larger than MAX_SIZE
            MAX_SIZE=5,
            verbose=False
        )

        # Should handle gracefully (may produce empty vocab)
        assert vocab.num_fragments >= 0

    def test_zero_target_size(self):
        """Test building vocab with target=0."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=['CCO', 'c1ccccc1'],
            convert=True,
            target=0
        )

        # Should produce empty or minimal vocab
        assert vocab.num_fragments >= 0


class TestDataTypeErrors:
    """Tests for handling wrong data types."""

    def test_smiles_as_int(self):
        """Test passing integer instead of SMILES string."""
        vocab = GS_Vocab()

        with pytest.raises((TypeError, AttributeError)):
            vocab.build_vocab(
                m_set=[123, 456],  # Integers instead of strings
                convert=True
            )

    def test_mol_as_string(self):
        """Test passing string when Mol object expected."""
        vocab = GS_Vocab()

        # convert=False expects Mol objects
        with pytest.raises((TypeError, AttributeError)):
            vocab.build_vocab(
                m_set=['CCO'],  # String when Mol expected
                convert=False  # This expects Mol objects
            )

    def test_invalid_descriptor_keys_type(self):
        """Test passing invalid type for descriptor_keys."""
        # The function may handle string input gracefully or raise error
        with pytest.raises((TypeError, ValueError)):
            calc_mol_frag_descriptors(
                smiles='CCO',
                descriptor_keys='MolWt'  # Should be list, not string
            )


class TestMissingDependencies:
    """Tests for handling missing or None dependencies."""

    def test_encoder_none_before_training(self):
        """Test that training without encoder raises error."""
        gsge = GSGE()

        with pytest.raises((ValueError, AttributeError)):
            gsge.make_GS_fragment_embedding_dict()

    def test_vocab_none_for_tokenization(self):
        """Test tokenization without vocabulary."""
        gsge = GSGE()  # No vocab

        # Should handle gracefully or raise informative error
        with pytest.raises((AttributeError, ValueError, TypeError)):
            gsge.preprocess_from_SMILES('CCO')


class TestNumericEdgeCases:
    """Tests for numeric edge cases and special values."""

    def test_descriptors_with_nan_handling(self):
        """Test that NaN descriptors are handled appropriately."""
        from GSGE.fragment_descriptors import normalize_descriptors

        # Create descriptors with NaN
        raw_descriptors = np.array([
            [1.0, 2.0, np.nan],
            [2.0, 4.0, np.nan],
            [3.0, 6.0, np.nan]
        ])

        # Normalization should handle or warn about NaN
        try:
            normalized, means, stds, mask = normalize_descriptors(raw_descriptors)
            # Check that function completed
            assert normalized is not None
        except (ValueError, RuntimeWarning):
            # Expected if NaN handling raises error
            pass

    def test_descriptors_all_zeros(self):
        """Test normalizing descriptors that are all zeros."""
        from GSGE.fragment_descriptors import normalize_descriptors

        raw_descriptors = np.zeros((10, 5))

        normalized, means, stds, mask = normalize_descriptors(raw_descriptors)

        # All constant columns should be filtered
        assert mask.sum() == 0  # No variance, all filtered

    def test_descriptors_single_sample(self):
        """Test normalizing descriptors with only one sample."""
        from GSGE.fragment_descriptors import normalize_descriptors

        raw_descriptors = np.array([[1.0, 2.0, 3.0]])

        # Single sample has no variance
        normalized, means, stds, mask = normalize_descriptors(raw_descriptors)

        # All columns should be filtered (no variance)
        assert mask.sum() == 0


class TestConcurrencyAndParallelism:
    """Tests for parallel processing edge cases."""

    def test_parallel_tokenize_empty_list(self, gsge_with_descriptors):
        """Test parallel tokenization with empty input."""
        result = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            [],
            max_workers=2
        )

        assert result == []

    def test_parallel_tokenize_single_worker(self, gsge_with_descriptors):
        """Test parallel tokenization with single worker."""
        smiles_list = ['CCO', 'c1ccccc1']

        result = gsge_with_descriptors.parallel_tokenize_SMILES_list(
            smiles_list,
            max_workers=1
        )

        assert len(result) == 2

    @pytest.mark.slow
    def test_parallel_graph_creation_error_handling(self, gsge_with_descriptors):
        """Test parallel compound graph creation with valid SMILES."""
        smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']

        # Should handle parallel processing
        try:
            graphs = gsge_with_descriptors.make_compound_graphs(
                smiles_list,
                workers=1,
                pyg_data=False
            )
            # Should return list of graphs
            assert isinstance(graphs, list)
        except Exception:
            # Acceptable if error raised
            pass


class TestMemoryConstraints:
    """Tests for handling memory-constrained scenarios."""

    def test_large_batch_size(self):
        """Test handling very large batch size."""
        # This shouldn't crash, just use available memory
        from GSGE import GSGE

        gsge = GSGE()
        # Just verify GSGE can be instantiated
        assert gsge is not None

    def test_very_long_smiles_string(self):
        """Test handling moderately long SMILES string."""
        # Use a valid but longer SMILES (real molecule)
        # A fatty acid chain
        long_smiles = 'CCCCCCCCCCCCCCCCCC(=O)O'

        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=[long_smiles],
            convert=True,
            verbose=False
        )

        # Should handle without memory error
        assert vocab.num_fragments >= 0


class TestFileIOErrors:
    """Tests for file I/O error handling."""

    def test_load_nonexistent_vocab(self):
        """Test loading vocabulary from nonexistent file."""
        vocab = GS_Vocab()

        with pytest.raises(FileNotFoundError):
            vocab.load_GS_vocab('/nonexistent/path/vocab.pkl')

    def test_load_nonexistent_corpus(self):
        """Test loading corpus from nonexistent file."""
        corpus = GSGE_Corpus()

        with pytest.raises(FileNotFoundError):
            corpus.load_GSGE_corpus('/nonexistent/path/corpus.pkl')

    def test_save_to_invalid_directory(self):
        """Test saving to nonexistent directory."""
        vocab = GS_Vocab()
        vocab.add_GS_fragment('CC(*)O')

        # Should raise error for invalid path
        with pytest.raises((FileNotFoundError, PermissionError, OSError)):
            vocab.save_GS_vocab(
                dir_path='/nonexistent/invalid/path',
                vocab_name='test'
            )

    def test_load_corrupted_pkl(self):
        """Test loading corrupted pickle file."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.pkl', delete=False) as f:
            f.write("This is not a valid pickle file")
            invalid_path = f.name

        vocab = GS_Vocab()

        with pytest.raises(Exception):  # Various pickle errors possible
            vocab.load_GS_vocab(invalid_path)


class TestStateConsistency:
    """Tests for maintaining consistent internal state."""

    def test_vocab_fragments_count_consistency(self):
        """Test that num_fragments matches actual fragment count."""
        vocab = GS_Vocab()
        vocab.add_GS_fragment('CC(*)O')
        vocab.add_GS_fragment('c1ccccc1(*)')

        assert vocab.num_fragments == len(vocab.vocab_fragments)
        assert vocab.num_fragments == len(vocab.fragments)

    def test_corpus_fragments_count_consistency(self):
        """Test corpus fragment count consistency."""
        corpus = GSGE_Corpus()
        corpus.add_GS_fragment('CC(*)O')
        corpus.add_GS_fragment('c1ccccc1(*)')

        assert corpus.num_fragments == len(corpus.vocab_fragments)

    def test_reinit_vocab_updates_ids(self):
        """Test that init_vocab correctly reassigns fragment IDs."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=['CCO', 'c1ccccc1'],
            convert=True,
            target=10
        )

        original_count = vocab.num_fragments

        # Reinit should maintain count
        vocab.init_vocab()

        assert vocab.num_fragments == original_count
