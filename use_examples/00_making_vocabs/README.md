# Building Vocabularies and Corpora

This tutorial demonstrates how to build custom molecular fragment vocabularies and corpora using GSGE.

## Tutorials in This Module

| Tutorial | Time Category | Time Estimate | Difficulty | Description |
|----------|---------------|---------------|------------|-------------|
| vocabulary_and_corpus_tutorial.ipynb | Medium | 30 min (CPU/GPU) | Beginner | Build vocabularies and corpora |

## Prerequisites

- [x] GSGE installed (see [Installation Guide](../../docs/getting-started/installation.md))
- [ ] Basic understanding of SMILES notation
- [ ] Completed: None (beginner-friendly, independent module)

## Learning Objectives

After completing this module, you will be able to:
- Extract molecular fragments from datasets
- Create custom vocabularies for your chemical space
- Build corpora for GAE training
- Save and load vocabularies

## Overview

Vocabularies and corpora are the foundation of GSGE:

- **GS_Vocab (Vocabulary)**: Merged, generalized fragments used as nodes in compound graphs
- **GSGE_Corpus**: Non-merged fragments for training the Graph Autoencoder

## Tutorial Notebook

**`vocabulary_and_corpus_tutorial.ipynb`**

This comprehensive tutorial covers:

1. **Dataset Preparation**
   - Loading molecular datasets (small molecules, linear peptides, cyclic peptides)
   - SMILES validation and preprocessing

2. **Vocabulary Building**
   - Extracting fragments with custom bond-cutting rules
   - Setting fragment size constraints (MIN_SIZE, MAX_SIZE)
   - Controlling vocabulary size (target parameter)
   - Adding single elements for complete coverage

3. **Corpus Creation**
   - Building non-merged fragment corpus for GAE training
   - Preserving R-groups without generalization
   - Adding vocabulary fragments to corpus

4. **Visualization**
   - Inspecting extracted fragments
   - Analyzing fragment distributions
   - Checking coverage

5. **Saving/Loading**
   - Serializing vocabularies and corpora
   - Adding metadata
   - Loading for later use

## Quick Start

```python
from GSGE import GS_Vocab, GSGE_Corpus, CUSTOM_fragment_mol

# Build vocabulary
vocab = GS_Vocab()
vocab.build_vocab(
    m_set=smiles_list,
    convert=True,
    target=200,
    MIN_SIZE=1,
    MAX_SIZE=15,
    fragment_mol_fn=CUSTOM_fragment_mol
)

# Build corpus
corpus = GSGE_Corpus()
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=15,
    fragment_mol_fn=CUSTOM_fragment_mol,
    fragmented=False  # Non-merged for GAE training
)

# Save
vocab.save_GS_vocab(dir_path='./vocabs', vocab_name='my_vocab')
corpus.save_GSGE_corpus(dir_path='./vocabs', vocab_name='my_corpus')
```

## Pre-built Vocabularies

The `vocabs/` directory contains pre-built vocabularies and corpora:

| File | Description | Use Case |
|------|-------------|----------|
| `GS_vocab_v5` | General vocabulary | Small molecules |
| `GS_vocab_v5_a` | Extended vocabulary | Linear peptides |
| `GS_vocab_v5_a2` | Comprehensive vocabulary | Cyclic peptides |
| `GSGE_corpus_v5` | Training corpus | GAE training |
| `gsge_save_v5a2.pkl` | Complete GSGE object | Ready-to-use instance |

### Loading Pre-built Vocabularies

```python
from GSGE import GS_Vocab, GSGE

# Load vocabulary only
vocab = GS_Vocab()
vocab.load_GS_vocab('vocabs/GS_vocab_v5_a2')

# Or load complete GSGE object
gsge = GSGE(GSGE_load_path='vocabs/gsge_save_v5a2.pkl')
```

## Key Parameters

### Vocabulary Building

- **target**: Target number of fragments to extract (e.g., 200)
- **MIN_SIZE**: Minimum fragment size in atoms (typically 1)
- **MAX_SIZE**: Maximum fragment size in atoms (typically 10-15)
- **n_limit**: Maximum wildcard connections per fragment (typically 80)
- **fragment_mol_fn**: Bond-cutting function (use `CUSTOM_fragment_mol` for peptides)

### Corpus Building

- **min_size**: Minimum fragment size (typically 1)
- **max_size**: Maximum fragment size (typically 10-15)
- **fragmented**: False for GAE training (keeps non-merged fragments)
- **method**: 'default' or custom fragmentation method

## Custom Bond-Cutting

The `CUSTOM_fragment_mol` function is designed for cyclic peptides:

- Removes outgoing ring bonds
- Cuts amide bonds: `[C,c:1](=O)[N:2][C,c:3]`
- Cuts disulfide bonds: `[C][S,v2]~[S,v2][C]`
- Handles ring systems: `[Rx2;D3,D4]`

For different chemistries, you can create custom fragmenters:

```python
from rdkit import Chem

def my_fragmenter(mol):
    """Custom bond-cutting logic."""
    # Your implementation
    return fragments

vocab.build_vocab(..., fragment_mol_fn=my_fragmenter)
```

## Best Practices

1. **Start Small**: Test with 100-500 molecules before scaling up
2. **Check Coverage**: Use `check_for_graphs_groupings()` to verify full coverage
3. **Add Single Elements**: Always run `add_all_single_elements()` for complete coverage
4. **Save Metadata**: Include meta_info when saving vocabularies
5. **Version Control**: Use clear naming (e.g., vocab_v1, vocab_v2) for reproducibility

## Troubleshooting

### Vocabulary Too Small

**Problem**: Extracted fewer fragments than expected

**Solutions**:
- Increase `MAX_SIZE` parameter
- Increase `target` parameter
- Check that SMILES are valid
- Use default fragmenter instead of custom

### Coverage Issues

**Problem**: Some molecules cannot be fully represented

**Solutions**:
- Run `gsge.add_all_single_elements()`
- Manually add missing fragments with `vocab.add_GS_fragment()`
- Increase vocabulary size

### Memory Errors

**Problem**: Out of memory during vocabulary building

**Solutions**:
- Process dataset in batches
- Reduce `target` parameter
- Use smaller `MAX_SIZE`

## Next Steps

After building vocabularies:

1. **Create Compound Graphs**: See [`make_compound_graphs/`](../01_make_compound_graphs/README.md)
2. **Train GAE**: See [`GAE/`](../03_GAE/README.md)
3. **Tokenize Molecules**: See [`tokenization_example/`](../02_tokenization_example/README.md)

## Additional Resources

- [User Guide: Vocabularies](../../docs/user-guide/vocabularies.md)
- [API Reference: vocab.py](../../docs/api-reference/vocab.md)


---

[Back to Tutorials Overview](../README.md)
- [ARCHITECTURE.md](../../ARCHITECTURE.md) - Architecture details
