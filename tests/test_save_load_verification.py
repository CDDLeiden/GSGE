"""
Test script to verify save/load methods work correctly for both GS_Vocab and GSGE_Corpus.

This script tests:
1. Save and load round-trip for GS_Vocab
2. Save and load round-trip for GSGE_Corpus
3. Data integrity after save/load
4. Fragment ID prefix preservation (GS_frag_ vs GSGE_frag_)
5. Settings preservation
6. Meta info preservation
"""

import os
import sys
import tempfile
import shutil
from collections import defaultdict

# Mock the missing dependencies for testing save/load functionality
class MockGroup:
    """Mock Group object for testing."""
    def __init__(self, smiles):
        self.smiles = smiles

class MockMol:
    """Mock molecule for testing."""
    def __init__(self, smiles):
        self.smiles = smiles

def test_save_load_functionality():
    """
    Test save and load functionality for both GS_Vocab and GSGE_Corpus.

    This test verifies that:
    1. All data is correctly saved to disk
    2. All data is correctly loaded back
    3. Fragment IDs are regenerated correctly with proper prefixes
    4. Settings and meta_info are preserved
    """

    print("=" * 80)
    print("Testing Save/Load Functionality for GS_Vocab and GSGE_Corpus")
    print("=" * 80)

    # Create temporary directory for test files
    temp_dir = tempfile.mkdtemp(prefix='vocab_save_load_test_')
    print(f"\nCreated temporary directory: {temp_dir}")

    try:
        # Import after creating temp dir to ensure imports work
        # We'll import the actual classes to test
        from GSGE.vocab import GS_Vocab, GSGE_Corpus, BaseGSVocab

        # Test 1: GS_Vocab save/load
        print("\n" + "-" * 80)
        print("TEST 1: GS_Vocab Save/Load")
        print("-" * 80)

        # Create a minimal GS_Vocab instance for testing
        # We'll manually set the attributes instead of using build_vocab
        vocab = GS_Vocab()
        vocab.fragments = ['C', 'CC', 'CCC', 'c1ccccc1']
        vocab.settings = {
            'convert': True,
            'n_limit': 1000,
            'target': 'vocab',
            'fragmented': False,
            'method': 'default',
            'MIN_SIZE': 1,
            'MAX_SIZE': 10,
            'fragment_mol_fn': 'test_fragment_fn'
        }
        vocab.core_dict = defaultdict(list)
        vocab.core_dict['[C]'] = ['C', 'CC']
        vocab.core_dict['[CC]'] = ['CCC']
        vocab.core_counter = defaultdict(int)
        vocab.core_counter['[C]'] = 2
        vocab.core_counter['[CC]'] = 1

        # Initialize vocab to generate fragment IDs
        vocab.init_vocab()

        print(f"Before save:")
        print(f"  - num_fragments: {vocab.num_fragments}")
        print(f"  - Fragment IDs: {list(vocab.vocab_fragments.keys())[:5]}...")

        # Verify fragment ID prefix
        first_frag_id = list(vocab.vocab_fragments.keys())[0]
        assert first_frag_id.startswith('GS_frag_'), f"Expected GS_frag_ prefix, got {first_frag_id}"
        print(f"  - Fragment ID prefix: ✓ 'GS_frag_'")

        # Save the vocabulary
        meta_info = "Test GS_Vocab for save/load verification"
        vocab_path = os.path.join(temp_dir, 'test_vocab')
        vocab.save_GS_vocab(dir_path=temp_dir, vocab_name='test_vocab', meta_info=meta_info)
        print(f"\n✓ Saved vocabulary to {vocab_path}")

        # Create a new instance and load
        vocab2 = GS_Vocab()
        vocab2.load_GS_vocab(vocab_path)
        print(f"✓ Loaded vocabulary from {vocab_path}")

        # Verify data integrity
        print(f"\nAfter load:")
        print(f"  - num_fragments: {vocab2.num_fragments}")
        print(f"  - Fragment IDs: {list(vocab2.vocab_fragments.keys())[:5]}...")

        # Check fragments are preserved
        assert vocab2.num_fragments == vocab.num_fragments, \
            f"Fragment count mismatch: {vocab2.num_fragments} != {vocab.num_fragments}"
        print(f"  ✓ Fragment count preserved: {vocab2.num_fragments}")

        # Check fragment ID prefix is preserved
        first_frag_id_2 = list(vocab2.vocab_fragments.keys())[0]
        assert first_frag_id_2.startswith('GS_frag_'), \
            f"Expected GS_frag_ prefix after load, got {first_frag_id_2}"
        print(f"  ✓ Fragment ID prefix preserved: 'GS_frag_'")

        # Check settings are preserved
        for key in vocab.settings:
            assert vocab2.settings.get(key) == vocab.settings[key], \
                f"Setting {key} not preserved: {vocab2.settings.get(key)} != {vocab.settings[key]}"
        print(f"  ✓ Settings preserved")

        # Check meta_info is preserved
        assert hasattr(vocab2, 'meta_info'), "meta_info attribute not loaded"
        assert vocab2.meta_info == meta_info, \
            f"meta_info not preserved: {vocab2.meta_info} != {meta_info}"
        print(f"  ✓ Meta info preserved: '{meta_info}'")

        # Check core_dict is preserved
        assert len(vocab2.core_dict) == len(vocab.core_dict), \
            f"core_dict size mismatch: {len(vocab2.core_dict)} != {len(vocab.core_dict)}"
        print(f"  ✓ core_dict preserved ({len(vocab2.core_dict)} cores)")

        # Check core_counter is preserved
        assert len(vocab2.core_counter) == len(vocab.core_counter), \
            f"core_counter size mismatch: {len(vocab2.core_counter)} != {len(vocab.core_counter)}"
        print(f"  ✓ core_counter preserved ({len(vocab2.core_counter)} entries)")

        print("\n✅ GS_Vocab save/load test PASSED")

        # Test 2: GSGE_Corpus save/load
        print("\n" + "-" * 80)
        print("TEST 2: GSGE_Corpus Save/Load")
        print("-" * 80)

        # Create a minimal GSGE_Corpus instance
        corpus = GSGE_Corpus()
        corpus.fragments = ['N', 'CN', 'CCN', 'c1ncccc1']
        corpus.settings = {
            'convert': False,
            'fragmented': False,
            'method': 'default',
            'MIN_SIZE': 1,
            'MAX_SIZE': 8,
            'fragment_mol_fn': 'test_fragment_fn_module'
        }
        corpus.core_dict = {}
        corpus.core_dict['[N]'] = ['N', 'CN']
        corpus.core_dict['[CN]'] = ['CCN']
        corpus.core_counter = defaultdict(int)
        corpus.core_counter['[N]'] = 2
        corpus.core_counter['[CN]'] = 1

        # Initialize corpus to generate fragment IDs
        corpus.init_vocab()

        print(f"Before save:")
        print(f"  - num_fragments: {corpus.num_fragments}")
        print(f"  - Fragment IDs: {list(corpus.vocab_fragments.keys())[:5]}...")

        # Verify fragment ID prefix
        first_frag_id = list(corpus.vocab_fragments.keys())[0]
        assert first_frag_id.startswith('GSGE_frag_'), \
            f"Expected GSGE_frag_ prefix, got {first_frag_id}"
        print(f"  - Fragment ID prefix: ✓ 'GSGE_frag_'")

        # Save the corpus
        meta_info = "Test GSGE_Corpus for save/load verification"
        corpus_path = os.path.join(temp_dir, 'test_corpus')
        corpus.save_GSGE_corpus(dir_path=temp_dir, vocab_name='test_corpus', meta_info=meta_info)
        print(f"\n✓ Saved corpus to {corpus_path}")

        # Create a new instance and load
        corpus2 = GSGE_Corpus()
        corpus2.load_GSGE_corpus(corpus_path)
        print(f"✓ Loaded corpus from {corpus_path}")

        # Verify data integrity
        print(f"\nAfter load:")
        print(f"  - num_fragments: {corpus2.num_fragments}")
        print(f"  - Fragment IDs: {list(corpus2.vocab_fragments.keys())[:5]}...")

        # Check fragments are preserved
        assert corpus2.num_fragments == corpus.num_fragments, \
            f"Fragment count mismatch: {corpus2.num_fragments} != {corpus.num_fragments}"
        print(f"  ✓ Fragment count preserved: {corpus2.num_fragments}")

        # Check fragment ID prefix is preserved
        first_frag_id_2 = list(corpus2.vocab_fragments.keys())[0]
        assert first_frag_id_2.startswith('GSGE_frag_'), \
            f"Expected GSGE_frag_ prefix after load, got {first_frag_id_2}"
        print(f"  ✓ Fragment ID prefix preserved: 'GSGE_frag_'")

        # Check settings are preserved
        for key in corpus.settings:
            assert corpus2.settings.get(key) == corpus.settings[key], \
                f"Setting {key} not preserved: {corpus2.settings.get(key)} != {corpus.settings[key]}"
        print(f"  ✓ Settings preserved")

        # Check meta_info is preserved
        assert hasattr(corpus2, 'meta_info'), "meta_info attribute not loaded"
        assert corpus2.meta_info == meta_info, \
            f"meta_info not preserved: {corpus2.meta_info} != {meta_info}"
        print(f"  ✓ Meta info preserved: '{meta_info}'")

        # Check core_dict is preserved
        assert len(corpus2.core_dict) == len(corpus.core_dict), \
            f"core_dict size mismatch: {len(corpus2.core_dict)} != {len(corpus.core_dict)}"
        print(f"  ✓ core_dict preserved ({len(corpus2.core_dict)} cores)")

        # Check core_counter is preserved
        assert len(corpus2.core_counter) == len(corpus.core_counter), \
            f"core_counter size mismatch: {len(corpus2.core_counter)} != {len(corpus.core_counter)}"
        print(f"  ✓ core_counter preserved ({len(corpus2.core_counter)} entries)")

        print("\n✅ GSGE_Corpus save/load test PASSED")

        # Test 3: Verify fragment ID mappings are correctly regenerated
        print("\n" + "-" * 80)
        print("TEST 3: Fragment ID Mapping Regeneration")
        print("-" * 80)

        # After loading, init_vocab() should have regenerated all mappings
        # Check that hash_to_frag_info, frag_to_canonical, and frag_id_to_noncanonical are populated
        assert len(vocab2.hash_to_frag_info) > 0, "hash_to_frag_info is empty after load"
        print(f"  ✓ hash_to_frag_info populated: {len(vocab2.hash_to_frag_info)} entries")

        assert len(vocab2.frag_to_canonical) > 0, "frag_to_canonical is empty after load"
        print(f"  ✓ frag_to_canonical populated: {len(vocab2.frag_to_canonical)} entries")

        assert len(vocab2.frag_id_to_noncanonical) > 0, "frag_id_to_noncanonical is empty after load"
        print(f"  ✓ frag_id_to_noncanonical populated: {len(vocab2.frag_id_to_noncanonical)} entries")

        assert len(corpus2.hash_to_frag_info) > 0, "hash_to_frag_info is empty after load"
        print(f"  ✓ Corpus hash_to_frag_info populated: {len(corpus2.hash_to_frag_info)} entries")

        assert len(corpus2.frag_to_canonical) > 0, "frag_to_canonical is empty after load"
        print(f"  ✓ Corpus frag_to_canonical populated: {len(corpus2.frag_to_canonical)} entries")

        assert len(corpus2.frag_id_to_noncanonical) > 0, "frag_id_to_noncanonical is empty after load"
        print(f"  ✓ Corpus frag_id_to_noncanonical populated: {len(corpus2.frag_id_to_noncanonical)} entries")

        print("\n✅ Fragment ID mapping regeneration test PASSED")

        # Test 4: Verify inheritance structure
        print("\n" + "-" * 80)
        print("TEST 4: Inheritance Structure")
        print("-" * 80)

        assert isinstance(vocab2, BaseGSVocab), "GS_Vocab is not instance of BaseGSVocab"
        print(f"  ✓ GS_Vocab inherits from BaseGSVocab")

        assert isinstance(corpus2, BaseGSVocab), "GSGE_Corpus is not instance of BaseGSVocab"
        print(f"  ✓ GSGE_Corpus inherits from BaseGSVocab")

        assert hasattr(vocab2, 'get_frag_id_prefix'), "get_frag_id_prefix method not found"
        assert vocab2.get_frag_id_prefix() == 'GS_frag_', \
            f"get_frag_id_prefix returned wrong value: {vocab2.get_frag_id_prefix()}"
        print(f"  ✓ GS_Vocab.get_frag_id_prefix() returns 'GS_frag_'")

        assert hasattr(corpus2, 'get_frag_id_prefix'), "get_frag_id_prefix method not found"
        assert corpus2.get_frag_id_prefix() == 'GSGE_frag_', \
            f"get_frag_id_prefix returned wrong value: {corpus2.get_frag_id_prefix()}"
        print(f"  ✓ GSGE_Corpus.get_frag_id_prefix() returns 'GSGE_frag_'")

        print("\n✅ Inheritance structure test PASSED")

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✅")
        print("=" * 80)
        print("\nSave/load functionality verified for both GS_Vocab and GSGE_Corpus:")
        print("  ✓ Data integrity maintained")
        print("  ✓ Fragment ID prefixes preserved (GS_frag_ vs GSGE_frag_)")
        print("  ✓ Settings and meta_info preserved")
        print("  ✓ Fragment mappings correctly regenerated")
        print("  ✓ Inheritance structure correct")
        print("=" * 80)

        return True

    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("Note: This test requires GSGE module to be importable.")
        print("If rdkit or group-selfies are missing, the test cannot run.")
        return False

    except AssertionError as e:
        print(f"\n❌ Assertion Failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            print(f"\nCleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"\nWarning: Could not clean up temporary directory: {e}")


if __name__ == '__main__':
    success = test_save_load_functionality()
    sys.exit(0 if success else 1)
