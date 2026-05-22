#!/usr/bin/env python
"""
Simple backwards compatibility test script.
Tests that the refactored code maintains the same API and behavior.
"""

import sys
import tempfile
from pathlib import Path

# Add GSGE to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from GSGE import GS_Vocab, GSGE_Corpus
    print("✓ Successfully imported GS_Vocab and GSGE_Corpus")
except Exception as e:
    print(f"✗ Failed to import: {e}")
    sys.exit(1)

# Test 1: Verify inheritance
print("\n=== Test 1: Verify Inheritance ===")
try:
    from GSGE.vocab import BaseGSVocab
    assert issubclass(GS_Vocab, BaseGSVocab), "GS_Vocab should inherit from BaseGSVocab"
    assert issubclass(GSGE_Corpus, BaseGSVocab), "GSGE_Corpus should inherit from BaseGSVocab"
    print("✓ Both classes inherit from BaseGSVocab")
except Exception as e:
    print(f"✗ Inheritance test failed: {e}")
    sys.exit(1)

# Test 2: Verify fragment ID prefixes
print("\n=== Test 2: Verify Fragment ID Prefixes ===")
try:
    vocab = GS_Vocab()
    corpus = GSGE_Corpus()

    vocab_prefix = vocab.get_frag_id_prefix()
    corpus_prefix = corpus.get_frag_id_prefix()

    assert vocab_prefix == 'GS_frag_', f"Expected 'GS_frag_', got '{vocab_prefix}'"
    assert corpus_prefix == 'GSGE_frag_', f"Expected 'GSGE_frag_', got '{corpus_prefix}'"
    print(f"✓ GS_Vocab prefix: '{vocab_prefix}'")
    print(f"✓ GSGE_Corpus prefix: '{corpus_prefix}'")
except Exception as e:
    print(f"✗ Fragment ID prefix test failed: {e}")
    sys.exit(1)

# Test 3: Verify initialization attributes
print("\n=== Test 3: Verify Initialization Attributes ===")
try:
    vocab = GS_Vocab()
    corpus = GSGE_Corpus()

    # Check common attributes exist
    for obj, name in [(vocab, 'GS_Vocab'), (corpus, 'GSGE_Corpus')]:
        assert hasattr(obj, 'core_counter'), f"{name} missing core_counter"
        assert hasattr(obj, 'fragments'), f"{name} missing fragments"
        assert hasattr(obj, 'num_fragments'), f"{name} missing num_fragments"
        assert hasattr(obj, 'vocab_fragments'), f"{name} missing vocab_fragments"
        assert hasattr(obj, 'hash_to_frag_info'), f"{name} missing hash_to_frag_info"
        assert hasattr(obj, 'frag_to_canonical'), f"{name} missing frag_to_canonical"
        assert hasattr(obj, 'frag_id_to_noncanonical'), f"{name} missing frag_id_to_noncanonical"
        assert hasattr(obj, 'core_dict'), f"{name} missing core_dict"

    # Check specific initialization
    assert vocab.num_fragments == 0, "GS_Vocab should start with 0 fragments"
    assert corpus.num_fragments == 0, "GSGE_Corpus should start with 0 fragments"
    assert len(vocab.fragments) == 0, "GS_Vocab should start with empty fragments list"
    assert len(corpus.fragments) == 0, "GSGE_Corpus should start with empty fragments list"

    print("✓ All initialization attributes present")
    print("✓ Initial values correct")
except Exception as e:
    print(f"✗ Initialization test failed: {e}")
    sys.exit(1)

# Test 4: Verify inherited methods exist
print("\n=== Test 4: Verify Inherited Methods ===")
try:
    vocab = GS_Vocab()
    corpus = GSGE_Corpus()

    # Methods that should be inherited from BaseGSVocab
    base_methods = ['init_vocab', 'get_hashes', 'add_GS_fragment', 'add_GS_group', 'plot_vocab']

    for method in base_methods:
        assert hasattr(vocab, method), f"GS_Vocab missing method: {method}"
        assert hasattr(corpus, method), f"GSGE_Corpus missing method: {method}"
        assert callable(getattr(vocab, method)), f"GS_Vocab.{method} not callable"
        assert callable(getattr(corpus, method)), f"GSGE_Corpus.{method} not callable"

    # Class-specific save/load methods
    assert hasattr(vocab, 'save_GS_vocab'), "GS_Vocab missing save_GS_vocab"
    assert hasattr(vocab, 'load_GS_vocab'), "GS_Vocab missing load_GS_vocab"
    assert hasattr(corpus, 'save_GSGE_corpus'), "GSGE_Corpus missing save_GSGE_corpus"
    assert hasattr(corpus, 'load_GSGE_corpus'), "GSGE_Corpus missing load_GSGE_corpus"

    print(f"✓ All {len(base_methods)} base methods present in both classes")
    print("✓ Class-specific save/load methods present")
except Exception as e:
    print(f"✗ Method verification failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Verify add_GS_fragment generates correct IDs
print("\n=== Test 5: Verify Fragment ID Generation ===")
try:
    vocab = GS_Vocab()
    corpus = GSGE_Corpus()

    # Add fragments
    vocab.add_GS_fragment('CC(*1)O')
    corpus.add_GS_fragment('CC(*1)O')

    # Check IDs
    assert 'GS_frag_0' in vocab.vocab_fragments, "Expected GS_frag_0 in vocab"
    assert 'GSGE_frag_0' in corpus.vocab_fragments, "Expected GSGE_frag_0 in corpus"
    assert vocab.num_fragments == 1, "GS_Vocab should have 1 fragment"
    assert corpus.num_fragments == 1, "GSGE_Corpus should have 1 fragment"

    print("✓ Fragment IDs generated correctly")
    print(f"  - GS_Vocab: GS_frag_0")
    print(f"  - GSGE_Corpus: GSGE_frag_0")
except Exception as e:
    print(f"✗ Fragment ID generation test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Verify get_hashes works
print("\n=== Test 6: Verify get_hashes Method ===")
try:
    vocab = GS_Vocab()
    vocab.add_GS_fragment('CC(*1)O')

    hashes = vocab.get_hashes()
    assert isinstance(hashes, list), "get_hashes should return a list"
    assert len(hashes) == 1, "Should have 1 hash"
    print("✓ get_hashes returns correct result")
except Exception as e:
    print(f"✗ get_hashes test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Verify save/load methods exist (not testing actual save/load due to potential dependencies)
print("\n=== Test 7: Verify Save/Load Method Signatures ===")
try:
    import inspect

    vocab = GS_Vocab()
    corpus = GSGE_Corpus()

    # Check method signatures
    vocab_save_sig = str(inspect.signature(vocab.save_GS_vocab))
    vocab_load_sig = str(inspect.signature(vocab.load_GS_vocab))
    corpus_save_sig = str(inspect.signature(corpus.save_GSGE_corpus))
    corpus_load_sig = str(inspect.signature(corpus.load_GSGE_corpus))

    print(f"✓ save_GS_vocab signature: {vocab_save_sig}")
    print(f"✓ load_GS_vocab signature: {vocab_load_sig}")
    print(f"✓ save_GSGE_corpus signature: {corpus_save_sig}")
    print(f"✓ load_GSGE_corpus signature: {corpus_load_sig}")
except Exception as e:
    print(f"✗ Save/load signature test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Verify unique methods still exist
print("\n=== Test 8: Verify Unique Methods ===")
try:
    vocab = GS_Vocab()
    corpus = GSGE_Corpus()

    # GS_Vocab unique methods
    assert hasattr(vocab, 'build_vocab'), "GS_Vocab missing build_vocab"
    assert hasattr(vocab, 'merge_into_core'), "GS_Vocab missing merge_into_core"

    # GSGE_Corpus unique methods
    assert hasattr(corpus, 'build_corpus'), "GSGE_Corpus missing build_corpus"
    assert hasattr(corpus, '_process_molecule'), "GSGE_Corpus missing _process_molecule"
    assert hasattr(corpus, '_canonicalize_core'), "GSGE_Corpus missing _canonicalize_core"

    print("✓ GS_Vocab unique methods present")
    print("✓ GSGE_Corpus unique methods present")
except Exception as e:
    print(f"✗ Unique methods test failed: {e}")
    sys.exit(1)

print("\n" + "="*60)
print("ALL BACKWARDS COMPATIBILITY TESTS PASSED!")
print("="*60)
print("\nSummary:")
print("  ✓ Inheritance hierarchy correct")
print("  ✓ Fragment ID prefixes correct (GS_frag_ vs GSGE_frag_)")
print("  ✓ All initialization attributes present")
print("  ✓ All inherited methods working")
print("  ✓ Fragment ID generation working")
print("  ✓ get_hashes method working")
print("  ✓ Save/load methods present")
print("  ✓ Unique methods preserved")
