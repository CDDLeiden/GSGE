# GSGE Class

The main facade class providing the primary interface to GSGE functionality.

::: GSGE.gsge.GSGE
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Manager Classes

### VocabularyManager

::: GSGE.gsge.VocabularyManager
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### EmbeddingManager

::: GSGE.gsge.EmbeddingManager
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### GAETrainer

::: GSGE.gsge.GAETrainer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### DescriptorCalculator

::: GSGE.gsge.DescriptorCalculator
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### GraphProcessor

::: GSGE.gsge.GraphProcessor
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### ClusteringAnalyzer

::: GSGE.gsge.ClusteringAnalyzer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Compound Graphs

### make_compound_graphs

::: GSGE.gsge.GSGE.make_compound_graphs
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### get_CG_from_SMILES

::: GSGE.gsge.GSGE.get_CG_from_SMILES
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### CompoundGraph

::: GSGE.graphs.compound_graph.CompoundGraph
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Usage Examples

### Basic Initialization

```python
from GSGE import GSGE, GS_Vocab, GSGE_Corpus

# Initialize with vocabulary and corpus
gsge = GSGE(
    GS_vocab=vocab,
    GSGE_corpus=corpus
)

# Or load from saved state
gsge = GSGE(GSGE_load_path='gsge_save.pkl')
```

### Complete Workflow

```python
# Build vocabulary and corpus
vocab = GS_Vocab()
vocab.build_vocab(m_set=smiles, convert=True, target=200)

corpus = GSGE_Corpus()
corpus.build_corpus(m_set=smiles, convert=True)

# Create GSGE
gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)
gsge.add_all_single_elements()

# Train GAE
gsge.train_GSGE_Auto_Encoder(
    batch_size=64,
    num_epochs=100,
    device='cuda'
)

# Generate embeddings
gsge.make_GS_fragment_embedding_dict(device='cuda')

# Create compound graphs
cgs = gsge.make_compound_graphs(smiles, workers=4, pyg_data=True)

# Save
gsge.save_gsge_data('my_gsge.pkl')
```
