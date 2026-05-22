import sys
sys.path.insert(0, '.')

try:
    from GSGE.vocab import GS_Vocab, GSGE_Corpus, BaseGSVocab
    print("✓ Import successful")
    print(f"  GS_Vocab: {GS_Vocab}")
    print(f"  GSGE_Corpus: {GSGE_Corpus}")
    print(f"  BaseGSVocab: {BaseGSVocab}")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
