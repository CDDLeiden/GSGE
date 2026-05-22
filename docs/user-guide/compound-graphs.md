# Compound Graphs

Compound graphs are GSGE's core representation for molecules. Instead of representing molecules at the atomic level, compound graphs use molecular fragments as nodes, enabling more chemically meaningful representations for machine learning tasks.

## Overview

In traditional molecular graphs, each node represents an atom. In GSGE compound graphs:

- **Nodes** represent molecular fragments (functional groups, ring systems, etc.)
- **Edges** represent connections between fragments
- **Fragment types** come from your GSGE vocabulary

This approach captures higher-level chemical semantics that atomic-level graphs miss.

## Creating Compound Graphs

### From SMILES

```python
from GSGE import GSGE

# Load or create GSGE instance
gsge = GSGE(GSGE_load_path='my_gsge.pkl')

# Create compound graphs from SMILES
smiles_list = ['CCO', 'c1ccccc1', 'CC(=O)O']
cgs = gsge.make_compound_graphs(smiles_list)

# Returns list of compound graph objects
```

### As PyTorch Geometric Data Objects

For training graph neural networks:

```python
# Create PyG Data objects
cgs = gsge.make_compound_graphs(smiles_list, pyg_data=True)

# Each CG is a torch_geometric.data.Data object with:
# - x: node features (fragment embeddings)
# - edge_index: graph connectivity
# - edge_attr: edge features
# - num_fragments: number of fragments
```

## Compound Graph Structure

Each compound graph contains:

| Attribute | Type | Description |
|-----------|------|-------------|
| `fragments` | list[str] | Fragment SMILES in the molecule |
| `fragment_ids` | list[int] | Token IDs for each fragment |
| `adjacency` | list[tuple] | (source, target) connections between fragments |
| `num_fragments` | int | Total number of fragments |

### Example: Ethanol (CCO)

```python
from GSGE import GSGE

gsge = GSGE(GSGE_load_path='my_gsge.pkl')
gsge.add_all_single_elements()

cg = gsge.get_CG_from_smiles('CCO', return_CG_object=True)

print(f"Fragments: {cg.fragments}")
# ['[C]', '[C]', '[O]']  # Methyl, Methylene, Hydroxyl

print(f"Fragment IDs: {cg.fragment_ids}")
# [23, 23, 47]  # Token IDs from vocabulary

print(f"Connections: {cg.adjacency}")
# [(0, 1), (1, 2)]  # C-C, C-O bonds
```

## Graph Features

### Node Features (Fragment Embeddings)

```python
# Get embeddings for compound graph
gsge.set_encoder()
gsge.load_GAE_weights('checkpoint.pth')

cg = gsge.get_CG_from_smiles('CCO', return_CG_object=True)
cg.get_node_features(gsge)

# Node features shape: (num_fragments, embedding_dim)
print(cg.x.shape)  # (3, 128) for ethanol with 128-dim embeddings
```

### Edge Features

Edges encode bond information between fragments:

- **Bond type**: single, double, triple, aromatic
- **Connection atoms**: which atoms in each fragment are bonded
- **Distance**: topological distance between fragments

## Working with Compound Graphs

### Visualizing Compound Graphs

```python
# Visualize with RDKit-style layout
cg = gsge.get_CG_from_smiles('c1ccccc1', return_CG_object=True)
cg.plot_graph_rd_c_style()
```

### Batch Processing

```python
from tqdm import tqdm

# Process large datasets
smiles_list = [...]  # Your dataset

cgs = []
for smiles in tqdm(smiles_list):
    try:
        cg = gsge.get_CG_from_smiles(smiles, return_CG_object=True)
        cgs.append(cg)
    except Exception as e:
        print(f"Failed for {smiles}: {e}")
        # Handle invalid molecules
```

### Parallel Processing

```python
# Use multiprocessing for large datasets
from joblib import Parallel, delayed

def process_smiles(s):
    return gsge.get_CG_from_smiles(s, return_CG_object=True)

cgs = Parallel(n_jobs=4)(
    delayed(process_smiles)(s)
    for s in smiles_list
)
```

## Use Cases

### Property Prediction

Compound graphs are ideal for graph neural network-based property prediction:

```python
import torch
from torch_geometric.loader import DataLoader

# Create PyG Data objects
cgs = gsge.make_compound_graphs(smiles_list, pyg_data=True)

# Add labels
for cg, label in zip(cgs, labels):
    cg.y = torch.tensor([label])

# Create DataLoader
loader = DataLoader(cgs, batch_size=32, shuffle=True)

# Train your GNN
for batch in loader:
    pred = model(batch.x, batch.edge_index, batch.batch)
    loss = criterion(pred, batch.y)
    # ...
```

### Molecular Similarity

Compare molecules based on fragment composition:

```python
cg1 = gsge.get_CG_from_smiles('CCO', return_CG_object=True)
cg2 = gsge.get_CG_from_smiles('CC(C)O', return_CG_object=True)

# Jaccard similarity of fragment sets
set1 = set(cg1.fragments)
set2 = set(cg2.fragments)
similarity = len(set1 & set2) / len(set1 | set2)
```

### Fragment Analysis

Analyze which fragments contribute to properties:

```python
cg = gsge.get_CG_from_smiles('CC(=O)Oc1ccccc1', return_CG_object=True)

# Get fragment embeddings
embeddings = gsge.get_fragment_embeddings()
cg.get_node_features(gsge)

# Identify most important fragments
for frag, emb in zip(cg.fragments, cg.x):
    print(f"{frag}: {emb.norm().item():.3f}")
```

## Common Issues

### Issue: "Fragment not in vocabulary"

**Cause**: SMILES contains a fragment not in your vocabulary.

**Solution**:
```python
# Always add single elements after building vocabulary
gsge.add_all_single_elements()

# Or rebuild vocabulary with more diverse fragments
vocab = GS_Vocab()
vocab.build_vocab(smiles_list, convert=True, target=500)  # Larger target
```

### Issue: Graph has no edges

**Cause**: Molecule is a single fragment, or fragments aren't connected.

**Solution**: Check your fragmentation function and vocabulary coverage.

```python
cg = gsge.get_CG_from_smiles('CCO', return_CG_object=True)
print(f"Edges: {len(cg.edge_index[0])}")  # Should be > 0
```

### Issue: Different graphs for same SMILES

**Cause**: Fragmentation is deterministic, but vocabulary differences cause different tokenizations.

**Solution**: Always use the same GSGE instance with the same vocabulary.

## See Also

- [Vocabularies](vocabularies.md) - Building custom fragment vocabularies
- [Graph Autoencoder](../api-reference/gae.md) - Training fragment embeddings
- [API Reference: GraphProcessor](../api-reference/gsge.md#graphprocessor) - Low-level graph operations
