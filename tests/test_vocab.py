"""
Comprehensive tests for GS_Vocab and GSGE_Corpus vocabulary classes.

Tests vocabulary building, fragment addition, serialization,
and corpus management for GAE training.
"""

import pytest
import tempfile
from pathlib import Path
from collections import defaultdict

from GSGE import GS_Vocab, GSGE_Corpus
from GSGE.vocab import BaseGSVocab
from GSGE.fragment_functions import CUSTOM_fragment_mol
from rdkit import Chem


class TestGSVocab:
    """Tests for GS_Vocab class."""

    def test_init_empty_vocab(self):
        """Test creating empty vocabulary."""
        vocab = GS_Vocab()
        assert vocab.num_fragments == 0
        assert len(vocab.vocab_fragments) == 0
        assert len(vocab.fragments) == 0

    def test_add_single_fragment(self):
        """Test adding single fragment to vocabulary."""
        vocab = GS_Vocab()
        vocab.add_GS_fragment('CC(*)O')
        assert vocab.num_fragments == 1
        assert 'GS_frag_0' in vocab.vocab_fragments
        assert 'CC(*)O' in vocab.fragments

    def test_add_duplicate_fragment_canonical(self):
        """Test that duplicate fragments (same canonical form) aren't added twice."""
        vocab = GS_Vocab()
        vocab.add_GS_fragment('CC(*)O')
        vocab.add_GS_fragment('OC(*)C')  # Same as CC(*)O canonically
        # Should only have 1 fragment (canonical duplicates filtered)
        assert vocab.num_fragments == 1

    def test_build_vocab_from_smiles(self, simple_smiles_list):
        """Test building vocabulary from SMILES list."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=50,
            MIN_SIZE=1,
            MAX_SIZE=10,
            fragment_mol_fn=CUSTOM_fragment_mol
        )
        assert vocab.num_fragments > 0
        assert len(vocab.fragments) == vocab.num_fragments

    def test_build_vocab_with_mol_objects(self, simple_smiles_list):
        """Test building vocabulary from RDKit Mol objects."""
        mols = [Chem.MolFromSmiles(s) for s in simple_smiles_list]
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=mols,
            convert=False,
            target=50,
            MIN_SIZE=1,
            MAX_SIZE=10
        )
        assert vocab.num_fragments > 0

    def test_save_and_load_vocab(self, simple_smiles_list):
        """Test vocabulary serialization round-trip."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=30,
            MIN_SIZE=1,
            MAX_SIZE=8
        )
        original_count = vocab.num_fragments

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_vocab.pkl'
            vocab.save_GS_vocab(dir_path=tmpdir, vocab_name='test_vocab')

            # Load into new vocab
            loaded_vocab = GS_Vocab()
            loaded_vocab.load_GS_vocab(str(save_path))

            assert loaded_vocab.num_fragments == original_count
            assert len(loaded_vocab.fragments) == len(vocab.fragments)
            assert list(loaded_vocab.vocab_fragments.keys()) == list(vocab.vocab_fragments.keys())

    def test_get_hashes(self, simple_smiles_list):
        """Test retrieving canonical hashes."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=20
        )
        hashes = vocab.get_hashes()
        assert isinstance(hashes, list)
        assert len(hashes) == vocab.num_fragments

    def test_merge_into_core(self):
        """Test fragment merging into generalized cores."""
        vocab = GS_Vocab()
        # Add similar fragments that should merge
        vocab.add_GS_fragment('CC(*)O')
        vocab.add_GS_fragment('CCC(*)O')

        # Build with merging enabled
        vocab.build_vocab(
            m_set=['CCO', 'CCCO', 'CCCCO'],
            convert=True,
            target=50,
            MIN_SIZE=2,
            MAX_SIZE=10,
            n_limit=2  # Require at least 2 occurrences
        )
        # Verify merging occurred (exact count depends on fragmentation)
        assert vocab.num_fragments > 0

    def test_export_fragments_to_csv(self, simple_smiles_list):
        """Test exporting fragments to CSV."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=20
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / 'fragments.csv'
            vocab.export_fragments_to_csv(str(csv_path))

            assert csv_path.exists()
            # Verify CSV has content
            with open(csv_path, 'r') as f:
                lines = f.readlines()
                assert len(lines) > 1  # Header + at least one fragment


class TestGSGECorpus:
    """Tests for GSGE_Corpus class."""

    def test_init_empty_corpus(self):
        """Test creating empty corpus."""
        corpus = GSGE_Corpus()
        assert corpus.num_fragments == 0
        assert len(corpus.vocab_fragments) == 0

    def test_add_fragment_to_corpus(self):
        """Test adding fragment to corpus."""
        corpus = GSGE_Corpus()
        corpus.add_GS_fragment('CC(*)O')
        assert corpus.num_fragments == 1
        assert 'GSGE_frag_0' in corpus.vocab_fragments  # Uses GSGE_frag prefix

    def test_build_corpus_from_smiles(self, simple_smiles_list):
        """Test building corpus from SMILES (non-merged fragments)."""
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=simple_smiles_list,
            min_size=1,
            max_size=10,
            convert=True,
            fragmented=False
        )
        assert corpus.num_fragments > 0

    def test_corpus_vs_vocab_fragment_counts(self, simple_smiles_list):
        """Test that corpus has more fragments than vocab (non-merged vs merged)."""
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=30,
            MIN_SIZE=1,
            MAX_SIZE=10
        )

        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=simple_smiles_list,
            min_size=1,
            max_size=10,
            convert=True
        )

        # Corpus should have equal or more fragments (non-merged)
        assert corpus.num_fragments >= vocab.num_fragments

    def test_save_and_load_corpus(self, simple_smiles_list):
        """Test corpus serialization round-trip."""
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=simple_smiles_list,
            min_size=1,
            max_size=8,
            convert=True
        )
        original_count = corpus.num_fragments

        with tempfile.TemporaryDirectory() as tmpdir:
            save_path = Path(tmpdir) / 'test_corpus.pkl'
            corpus.save_GSGE_corpus(dir_path=tmpdir, vocab_name='test_corpus')

            # Load into new corpus
            loaded_corpus = GSGE_Corpus()
            loaded_corpus.load_GSGE_corpus(str(save_path))

            assert loaded_corpus.num_fragments == original_count

    def test_get_hashes_corpus(self, simple_smiles_list):
        """Test retrieving corpus hashes."""
        corpus = GSGE_Corpus()
        corpus.build_corpus(m_set=simple_smiles_list, convert=True)
        hashes = corpus.get_hashes()
        assert isinstance(hashes, list)
        assert len(hashes) > 0

    def test_corpus_with_meta_info(self):
        """Test saving corpus with metadata."""
        corpus = GSGE_Corpus()
        corpus.add_GS_fragment('CC(*)O')

        with tempfile.TemporaryDirectory() as tmpdir:
            corpus.save_GSGE_corpus(
                dir_path=tmpdir,
                vocab_name='test_corpus',
                meta_info='Test metadata string'
            )
            save_path = Path(tmpdir) / 'test_corpus'

            loaded = GSGE_Corpus()
            loaded.load_GSGE_corpus(str(save_path))
            # Meta info should be loaded
            assert hasattr(loaded, 'meta_info')


class TestCustomBaseGSVocab:
    """Tests for custom classes inheriting from BaseGSVocab."""

    def test_custom_class_inheritance(self):
        """Test that a custom class can inherit from BaseGSVocab and work correctly."""
        from collections import defaultdict

        # Define a custom vocabulary class
        class CustomTestVocab(BaseGSVocab):
            """Custom vocabulary for testing BaseGSVocab extensibility."""

            def __init__(self, load_path=None):
                super().__init__(load_path)
                # Initialize core_dict (required but not in BaseGSVocab.__init__)
                self.core_dict = defaultdict(list)

            def get_frag_id_prefix(self):
                """Return custom fragment ID prefix."""
                return 'CUSTOM_frag_'

        # Create instance and verify inheritance
        custom_vocab = CustomTestVocab()
        assert isinstance(custom_vocab, BaseGSVocab)
        assert custom_vocab.num_fragments == 0
        assert len(custom_vocab.vocab_fragments) == 0
        assert custom_vocab.get_frag_id_prefix() == 'CUSTOM_frag_'

    def test_custom_class_add_fragments(self):
        """Test adding fragments to custom class."""
        from collections import defaultdict

        class CustomTestVocab(BaseGSVocab):
            """Custom vocabulary for testing."""

            def __init__(self, load_path=None):
                super().__init__(load_path)
                self.core_dict = defaultdict(list)

            def get_frag_id_prefix(self):
                return 'TEST_frag_'

        custom_vocab = CustomTestVocab()

        # Add fragments and verify custom prefix is used
        custom_vocab.add_GS_fragment('CC(*)O')
        assert custom_vocab.num_fragments == 1
        assert 'TEST_frag_0' in custom_vocab.vocab_fragments
        assert 'CC(*)O' in custom_vocab.fragments

        custom_vocab.add_GS_fragment('c1ccccc1')
        assert custom_vocab.num_fragments == 2
        assert 'TEST_frag_1' in custom_vocab.vocab_fragments

    def test_custom_class_get_hashes(self):
        """Test get_hashes method works on custom class."""
        from collections import defaultdict

        class CustomTestVocab(BaseGSVocab):
            """Custom vocabulary for testing."""

            def __init__(self, load_path=None):
                super().__init__(load_path)
                self.core_dict = defaultdict(list)

            def get_frag_id_prefix(self):
                return 'HASH_frag_'

        custom_vocab = CustomTestVocab()
        custom_vocab.add_GS_fragment('CC(*)O')
        custom_vocab.add_GS_fragment('c1ccccc1')

        hashes = custom_vocab.get_hashes()
        assert isinstance(hashes, list)
        assert len(hashes) == 2

    def test_custom_class_init_vocab(self):
        """Test init_vocab regenerates fragment IDs correctly."""
        from collections import defaultdict

        class CustomTestVocab(BaseGSVocab):
            """Custom vocabulary for testing."""

            def __init__(self, load_path=None):
                super().__init__(load_path)
                self.core_dict = defaultdict(list)

            def get_frag_id_prefix(self):
                return 'REINIT_frag_'

        custom_vocab = CustomTestVocab()
        custom_vocab.add_GS_fragment('CC(*)O')
        custom_vocab.add_GS_fragment('c1ccccc1')

        # Store original IDs
        original_ids = list(custom_vocab.vocab_fragments.keys())
        assert original_ids == ['REINIT_frag_0', 'REINIT_frag_1']

        # Reinit vocabulary
        custom_vocab.init_vocab()

        # Verify fragments are regenerated with correct prefix
        assert custom_vocab.num_fragments == 2
        new_ids = list(custom_vocab.vocab_fragments.keys())
        assert new_ids == ['REINIT_frag_0', 'REINIT_frag_1']
        assert 'CC(*)O' in custom_vocab.fragments
        assert 'c1ccccc1' in custom_vocab.fragments

    def test_custom_class_duplicate_handling(self):
        """Test that duplicate fragments are handled correctly in custom class."""
        from collections import defaultdict

        class CustomTestVocab(BaseGSVocab):
            """Custom vocabulary for testing."""

            def __init__(self, load_path=None):
                super().__init__(load_path)
                self.core_dict = defaultdict(list)

            def get_frag_id_prefix(self):
                return 'DUP_frag_'

        custom_vocab = CustomTestVocab()

        # Add same fragment twice (different representations)
        custom_vocab.add_GS_fragment('CC(*)O')
        custom_vocab.add_GS_fragment('OC(*)C')  # Same canonical form

        # Should only have 1 fragment (duplicates filtered)
        assert custom_vocab.num_fragments == 1
        assert 'DUP_frag_0' in custom_vocab.vocab_fragments
