# API Reference

Complete API documentation for GSGE modules and classes.

## Core Modules

### [GSGE Class](gsge.md)
Main facade class that provides the primary interface to GSGE functionality.

- VocabularyManager - Manages vocabularies and corpora
- EmbeddingManager - Handles fragment embeddings
- GAETrainer - Graph Autoencoder training
- DescriptorCalculator - RDKit descriptor computation
- GraphProcessor - Compound graph operations
- ClusteringAnalyzer - Chemical space visualization

### [Vocabularies](vocab.md)
Fragment vocabulary and corpus management.

- `GS_Vocab` - Vocabulary of merged, generalized fragments
- `GSGE_Corpus` - Corpus of non-merged fragments

### [Tokenizer](tokenizer.md)
Tokenization and preprocessing utilities.

- `GSGE_tokenizer` - Fragment-based tokenization
- Token-to-ID mappings
- Parallel tokenization

### [Embeddings](embedding.md)
Fragment embedding generation and management.

- `GSGE_Embedding` - PyTorch embedding layer
- `EmbeddingManager` - Embedding lookup tables
- Combined OHE + learned embeddings

### [Graph Autoencoder](gae.md)
Graph neural network for learning fragment embeddings.

- `AttentiveFP` - Graph attention encoder
- `GraphDecoder` - Graph reconstruction decoder
- `GraphAutoencoderTrainer` - Training loop
- `MetricsTracker` - Performance metrics

### [Clustering](clustering.md)
Chemical space visualization and analysis.

- `GSGE_clustering` - Fragment clustering utilities
- t-SNE and UMAP visualization
- MCS-based clustering

### [Descriptors](descriptors.md)
RDKit molecular descriptor calculation.

- `calc_mol_frag_descriptors` - Compute descriptors
- `normalize_descriptors` - Z-score normalization
- Descriptor lookup tables

### [Plots](plots.md)
Visualization utilities for molecules and fragments.

- `highlight_fragments` - Highlight fragments in molecules
- `generate_colors` - Color generation with seeds
- `draw_mol_with_atom_index` - Annotated molecule drawing

## Quick Navigation

### By Task

**Building Vocabularies**
→ [vocab.GS_Vocab](vocab.md#GS_Vocab)

**Training GAE**
→ [gae.GraphAutoencoderTrainer](gae.md#GraphAutoencoderTrainer)

**Creating Compound Graphs**
→ [gsge.GSGE.make_compound_graphs](gsge.md#make_compound_graphs)

**Tokenization**
→ [tokenizer.GSGE_tokenizer](tokenizer.md#GSGE_tokenizer)

**Using Embeddings**
→ [embedding.GSGE_Embedding](embedding.md#GSGE_Embedding)

**Visualization**
→ [clustering.GSGE_clustering](clustering.md#GSGE_clustering)

### By Data Type

**Working with SMILES**
- `GS_Vocab.build_vocab(m_set, convert=True, ...)`
- `GSGE.preprocess_from_SMILES(smiles)`
- `GSGE.get_CG_from_smiles(smiles)`

**Working with RDKit Mols**
- `calc_mol_frag_descriptors(mol=mol_object)`
- `GS_Vocab.build_vocab(m_set, convert=False, ...)`

**Working with Embeddings**
- `GSGE.make_GS_fragment_embedding_dict()`
- `GSGE.get_fragment_embeddings()`
- `GSGE_Embedding(frag_emb=embeddings)`

**Working with Graphs**
- `GSGE.make_compound_graphs(smiles_list, pyg_data=True)`
- `compound_graph.plot_graph_rd_c_style()`

## Module Structure

```
GSGE/
├── gsge.py                 # Main GSGE class
├── vocab.py                # GS_Vocab, GSGE_Corpus
├── tokenizer.py            # GSGE_tokenizer
├── embedding.py            # GSGE_Embedding, EmbeddingManager
├── clustering.py           # GSGE_clustering
├── fragment_descriptors.py # Descriptor calculation
├── plots.py                # Visualization utilities
├── core_gsge.py            # Core static utilities
├── graphs/
│   ├── fragment_graph/
│   │   └── GAE.py          # Graph Autoencoder
│   └── compound_graph/
│       └── data.py         # compound_graph class
└── chem.py                 # Chemical constants
```

## Common Workflows

### Vocabulary → Compound Graphs

```python
from GSGE import GSGE, GS_Vocab

# Build vocabulary
vocab = GS_Vocab()
vocab.build_vocab(m_set=smiles, convert=True, target=200)

# Create GSGE and compound graphs
gsge = GSGE(GS_vocab=vocab)
gsge.add_all_single_elements()
cgs = gsge.make_compound_graphs(smiles, pyg_data=True)
```

### Corpus → GAE → Embeddings

```python
from GSGE import GSGE, GSGE_Corpus
from GSGE.graphs.fragment_graph.GAE import GraphAutoencoderTrainer

# Build corpus
corpus = GSGE_Corpus()
corpus.build_corpus(m_set=smiles, convert=True)

# Train GAE
gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)
gsge.train_GSGE_Auto_Encoder(num_epochs=100, batch_size=64)

# Generate embeddings
gsge.make_GS_fragment_embedding_dict()
embeddings = gsge.get_fragment_embeddings()
```

### Tokenization → Embeddings

```python
# Tokenize molecules
tokens = gsge.parallel_tokenize_SMILES_list(smiles, max_workers=4)

# Convert to embeddings
from GSGE.embedding import GSGE_Embedding
emb_layer = GSGE_Embedding(
    0, None, 0,
    gsge.get_fragment_embeddings(),
    only_token2vec=True
)
# Use in PyTorch: emb_layer(token_ids) → embeddings
```

## Conventions

### Function Signatures

All API documentation follows these conventions:

- **Args**: Parameters with types and descriptions
- **Returns**: Return values with types
- **Raises**: Exceptions that may be raised
- **Example**: Usage examples
- **Note**: Important implementation details

### Type Hints

Type hints are provided for all public APIs:

```python
def build_vocab(
    self,
    m_set: List[Union[str, Chem.Mol]],
    convert: bool = True,
    target: int = 100
) -> None:
```

### Default Values

Default values are specified in signatures and documented in parameter descriptions.

## Version Information

This documentation corresponds to GSGE version 1.0.0.

For changes between versions, see the [Changelog](../development/changelog.md).
