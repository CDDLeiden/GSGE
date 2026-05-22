# Vocabularies & Corpus

Vocabularies and corpora are the foundation of GSGE. This guide explains how to build, customize, and manage molecular fragment collections.

## Overview

GSGE uses two types of fragment collections:

- **GS_Vocab (Vocabulary)**: Merged, generalized fragments used as nodes in compound graphs
- **GSGE_Corpus**: Non-merged fragments for training the Graph Autoencoder

## GS_Vocab (Vocabulary)

The vocabulary contains molecular fragments that serve as the "alphabet" for representing molecules as graphs.

### Building a Vocabulary

```python
from GSGE import GS_Vocab, CUSTOM_fragment_mol

# Create vocabulary object
vocab = GS_Vocab()

# Build from SMILES list
smiles_list = ['CCO', 'c1ccccc1', 'CC(C)O', '...']

vocab.build_vocab(
    m_set=smiles_list,
    convert=True,  # Input is SMILES strings
    target=200,  # Target vocabulary size
    MIN_SIZE=1,  # Minimum fragment size
    MAX_SIZE=15,  # Maximum fragment size
    n_limit=80,  # Max wildcard connections
    fragment_mol_fn=CUSTOM_fragment_mol  # Bond-cutting function
)

print(f"Built vocabulary with {vocab.num_fragments} fragments")
```

### Key Parameters

#### Fragment Size

- **MIN_SIZE**: Minimum number of atoms in a fragment (default: 1)
- **MAX_SIZE**: Maximum number of atoms in a fragment (default: 10)

```python
# Small fragments (amino acids, functional groups)
vocab.build_vocab(..., MIN_SIZE=1, MAX_SIZE=10)

# Larger fragments (peptide sequences)
vocab.build_vocab(..., MIN_SIZE=5, MAX_SIZE=20)
```

#### Vocabulary Size

- **target**: Target number of unique fragments to extract

```python
# Small vocabulary for simple molecules
vocab.build_vocab(..., target=100)

# Large vocabulary for diverse chemical space
vocab.build_vocab(..., target=500)
```

#### Bond-Cutting Function

The `fragment_mol_fn` parameter controls how molecules are cut into fragments.

**CUSTOM_fragment_mol** (recommended for peptides):

- Removes outgoing ring bonds
- Cuts amide bonds: `[C,c:1](=O)[N:2][C,c:3]`
- Cuts disulfide bonds: `[C][S,v2]~[S,v2][C]`
- Handles ring systems: `[Rx2;D3,D4]`

```python
from GSGE import CUSTOM_fragment_mol

vocab.build_vocab(
    m_set=smiles_list,
    fragment_mol_fn=CUSTOM_fragment_mol
)
```

### Manually Adding Fragments

Add specific fragments to ensure coverage:

```python
# Add individual fragments
vocab.add_GS_fragment('O=C(*1)(*1)')  # Carbonyl
vocab.add_GS_fragment('N=C(*1)(*1)')  # Imine

# Add all single elements (recommended)
from GSGE.core_gsge import CoreGSGE
CoreGSGE.add_all_single_elements(
    vocab,
    element_bond_counts=_ELEMENTS_BOND_COUNTS
)
```

### Inspecting Vocabulary

```python
# Number of fragments
print(f"Total fragments: {vocab.num_fragments}")

# List all fragments
for i, frag_smiles in enumerate(vocab.fragments[:10]):
    print(f"{i}: {frag_smiles}")

# Get fragment hashes (for deduplication)
hashes = vocab.get_hashes()

# Visualize vocabulary fragments
vocab.plot_vocab(
    n_frags=20,  # Number of fragments to plot
    save_path='vocab_fragments.png'
)
```

### Saving and Loading

```python
# Save vocabulary
vocab.save_GS_vocab(
    dir_path='./my_vocabularies',
    vocab_name='peptide_vocab',
    meta_info='Cyclic peptide vocabulary, MIN_SIZE=1, MAX_SIZE=15'
)

# Load vocabulary
loaded_vocab = GS_Vocab()
loaded_vocab.load_GS_vocab('./my_vocabularies/peptide_vocab.pkl')

print(f"Loaded {loaded_vocab.num_fragments} fragments")
```

## GSGE_Corpus

The corpus contains non-merged, non-generalized fragments for training the Graph Autoencoder.

### Building a Corpus

```python
from GSGE import GSGE_Corpus, CUSTOM_fragment_mol

# Create corpus object
corpus = GSGE_Corpus()

# Build from SMILES
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=15,
    fragment_mol_fn=CUSTOM_fragment_mol,
    fragmented=False  # Keep non-merged fragments
)

print(f"Built corpus with {corpus.num_fragments} fragments")
```

### Vocabulary vs. Corpus

| Aspect | GS_Vocab | GSGE_Corpus |
|--------|----------|-------------|
| **Purpose** | Compound graph nodes | GAE training data |
| **Fragments** | Merged, generalized | Non-merged, specific |
| **R-groups** | Collapsed to wildcards | Preserved |
| **Size** | Smaller (100-500) | Larger (1000-10000) |
| **Usage** | Tokenization, graphs | Training only |

### Corpus Parameters

- **fragmented**: Whether to merge fragments (default: False for GAE training)
- **method**: Fragmentation method ('default' or custom)

```python
# Non-merged for GAE training (recommended)
corpus.build_corpus(..., fragmented=False)

# Merged (same as vocabulary)
corpus.build_corpus(..., fragmented=True)
```

### Adding Vocabulary to Corpus

For GAE training, include vocabulary fragments in corpus:

```python
from GSGE.core_gsge import CoreGSGE

CoreGSGE.add_GS_vocab_to_GSGE_corpus(
    vocab.vocab_fragments,
    corpus
)

print(f"Corpus now has {corpus.num_fragments} fragments")
```

### Saving and Loading Corpus

```python
# Save
corpus.save_GSGE_corpus(
    dir_path='./my_corpora',
    vocab_name='peptide_corpus',
    meta_info='Non-merged fragments for GAE training'
)

# Load
loaded_corpus = GSGE_Corpus()
loaded_corpus.load_GSGE_corpus('./my_corpora/peptide_corpus.pkl')
```

## Combining Vocabulary and Corpus

Use GSGE object to manage both:

```python
from GSGE import GSGE

# Create GSGE with both
gsge = GSGE(
    GS_vocab=vocab,
    GSGE_corpus=corpus
)

# Or load from paths
gsge = GSGE(
    GS_vocab='./my_vocabularies/peptide_vocab.pkl',
    GSGE_corpus='./my_corpora/peptide_corpus.pkl'
)

# Add single elements (required for compound graphs)
gsge.add_all_single_elements()
```

## Custom Bond-Cutting

Create your own fragment extraction logic:

```python
from rdkit import Chem

def my_fragmenter(mol):
    """Custom fragmenter for specific chemistry."""
    # Your custom bond-cutting logic
    fragments = []

    # Example: Cut only at specific bond types
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.BondType.SINGLE:
            # Fragment at this bond
            pass

    return fragments

# Use custom fragmenter
vocab.build_vocab(
    m_set=smiles_list,
    fragment_mol_fn=my_fragmenter
)
```

## Vocabulary Coverage

### Checking Coverage

Verify that your vocabulary can fully represent molecules:

```python
from GSGE import GSGE

gsge = GSGE(GS_vocab=vocab)
gsge.add_all_single_elements()

# Check ungrouped atoms
problematic = gsge.check_for_graphs_groupings(
    smiles_list,
    workers=4
)

if problematic:
    print(f"Warning: {len(problematic)} molecules have ungrouped atoms")
    for smiles, ungrouped_atoms in problematic:
        print(f"  {smiles}: {ungrouped_atoms} ungrouped")
else:
    print("✓ All molecules fully covered")
```

### Improving Coverage

If molecules aren't fully covered:

1. **Add single elements** (most common solution):
   ```python
   gsge.add_all_single_elements()
   ```

2. **Add specific missing fragments**:
   ```python
   vocab.add_GS_fragment('your_fragment(*)')
   ```

3. **Increase vocabulary size**:
   ```python
   vocab.build_vocab(..., target=500)  # Larger target
   ```

4. **Adjust fragment sizes**:
   ```python
   vocab.build_vocab(..., MIN_SIZE=1, MAX_SIZE=20)
   ```

## Best Practices

### Vocabulary Building

1. **Start with representative dataset**: 100-1000 molecules covering your chemical space
2. **Set reasonable sizes**: MIN_SIZE=1, MAX_SIZE=10-15 for most use cases
3. **Use CUSTOM_fragment_mol**: Especially for peptides and complex molecules
4. **Always add single elements**: Ensures complete coverage
5. **Validate coverage**: Check that all molecules can be fully represented

### Corpus Creation

1. **Keep size manageable**: 1000-10000 unique fragments
2. **Use fragmented=False**: For GAE training
3. **Include vocabulary**: Add vocab fragments to corpus for better training
4. **Match sizes**: Use same MIN/MAX_SIZE as vocabulary

### Performance Tips

1. **Batch processing**: Build vocabularies in batches for very large datasets
2. **Parallel workers**: Use multiprocessing for fragment extraction
3. **Cache results**: Save and reuse vocabularies across experiments

## Example Workflows

### Small Molecules

```python
# Small molecule vocabulary
vocab = GS_Vocab()
vocab.build_vocab(
    m_set=drug_like_smiles,
    convert=True,
    target=150,
    MIN_SIZE=1,
    MAX_SIZE=10
)
vocab.save_GS_vocab(dir_path='.', vocab_name='druglike_vocab')
```

### Cyclic Peptides

```python
# Cyclic peptide vocabulary
vocab = GS_Vocab()
vocab.build_vocab(
    m_set=cyclic_peptide_smiles,
    convert=True,
    target=300,
    MIN_SIZE=1,
    MAX_SIZE=15,
    fragment_mol_fn=CUSTOM_fragment_mol
)

# Add single elements
from GSGE.core_gsge import CoreGSGE
CoreGSGE.add_all_single_elements(vocab)

vocab.save_GS_vocab(dir_path='.', vocab_name='cycpep_vocab')
```

### Corpus for GAE

```python
# Build corpus from same dataset
corpus = GSGE_Corpus()
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=15,
    fragment_mol_fn=CUSTOM_fragment_mol,
    fragmented=False
)

# Add vocabulary fragments
CoreGSGE.add_GS_vocab_to_GSGE_corpus(vocab.vocab_fragments, corpus)

corpus.save_GSGE_corpus(dir_path='.', vocab_name='training_corpus')
```

## Troubleshooting

### Issue: Vocabulary too small

**Solution**: Increase `target` parameter or reduce `MIN_SIZE`

```python
vocab.build_vocab(..., target=300, MIN_SIZE=1)
```

### Issue: Fragments not covering molecules

**Solution**: Add single elements

```python
from GSGE import GSGE
gsge = GSGE(GS_vocab=vocab)
gsge.add_all_single_elements()
```

### Issue: Memory errors during building

**Solution**: Process in batches

```python
# Build vocabulary in batches
batch_size = 100
for i in range(0, len(smiles_list), batch_size):
    batch = smiles_list[i:i+batch_size]
    vocab.build_vocab(m_set=batch, ...)
```

## Next Steps

- Explore the [API Reference](../api-reference/index.md) for detailed documentation
- See [Quick Start](../getting-started/quickstart.md) for examples
