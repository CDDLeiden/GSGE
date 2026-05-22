# Creating Compound Graphs

This tutorial demonstrates how to create fragment-based compound graph representations of molecules using GSGE.

## Tutorials in This Module

| Tutorial | Time Category | Time Estimate | Difficulty | Description |
|----------|---------------|---------------|------------|-------------|
| compound_graphs_tutorial.ipynb | Medium | 20 min (CPU/GPU) | Beginner | Create fragment-based graphs |

## Prerequisites

- [x] GSGE installed (see [Installation Guide](../../docs/getting-started/installation.md))
- [ ] Basic understanding of graph data structures
- [ ] Completed: [00_making_vocabs](../00_making_vocabs/README.md) recommended (or load pre-built)

## Learning Objectives

After completing this module, you will be able to:
- Create compound graphs from SMILES
- Visualize graph structures
- Understand graph node and edge features
- Use graphs for GNN training

## Overview

Compound graphs represent molecules as graphs where:
- **Nodes**: Molecular fragments from your vocabulary
- **Edges**: Bonds connecting fragments in the original molecule

This representation enables fragment-level molecular modeling instead of atom-level approaches.

## Tutorial Notebook

**`compound_graphs_tutorial.ipynb`**

This tutorial covers:

1. **Setup and Data Loading**
   - Loading GSGE with pre-built vocabulary
   - Preparing molecular datasets

2. **Single Compound Graph Creation**
   - Converting SMILES to compound graph
   - Understanding graph structure (nodes, edges, features)
   - Extracting adjacency matrices

3. **Batch Processing**
   - Creating graphs for multiple molecules in parallel
   - PyTorch Geometric Data format
   - Performance optimization

4. **Visualization**
   - RDKit-style graph layouts
   - Fragment highlighting in molecules
   - Node and edge annotations

5. **Graph Analysis**
   - Inspecting node features
   - Analyzing graph topology
   - Validating coverage

## Quick Start

```python
from GSGE import GSGE

# Load GSGE with vocabulary
gsge = GSGE(GS_vocab='path/to/vocab.pkl')
gsge.add_all_single_elements()  # Required for full coverage

# Create single compound graph
smiles = 'c1ccc(cc1)O'  # Phenol
cg = gsge.get_CG_from_smiles(smiles, return_CG_object=True)

# Inspect graph
print(f"Nodes: {len(cg.nodes)}")
print(f"Edges: {len(cg.edges)}")

# Visualize
cg.plot_graph_rd_c_style()
```

## Creating Multiple Graphs

### Sequential Processing

```python
# Create graphs one by one
cg_list = []
for smiles in smiles_list:
    cg = gsge.get_CG_from_smiles(smiles, return_CG_object=True)
    cg_list.append(cg)
```

### Parallel Processing (Recommended)

```python
# Create graphs in parallel
cgs_data = gsge.make_compound_graphs(
    smiles_list,
    workers=4,  # Number of parallel workers
    pyg_data=True  # Return PyTorch Geometric format
)

print(f"Created {len(cgs_data)} compound graphs")
```

## Graph Formats

### Compound Graph Object

```python
cg = gsge.get_CG_from_smiles(smiles, return_CG_object=True)

# Access properties
cg.nodes          # List of fragment IDs
cg.edges          # List of edges
cg.graph          # NetworkX graph
cg.smiles         # Original SMILES
cg.x              # Node features

# Visualization
cg.plot_graph_rd_c_style(show=True)
```

### Raw Format (Adjacency Matrix + Features)

```python
adj_matrix, features = gsge.get_CG_from_smiles(
    smiles,
    return_CG_object=False
)

# adj_matrix: [2, num_edges] edge index
# features: list of fragment IDs
```

### PyTorch Geometric Format

```python
cgs_data = gsge.make_compound_graphs(
    smiles_list,
    pyg_data=True
)

# Each data object has:
data = cgs_data[0]
data.x            # Node features [num_nodes, feature_dim]
data.edge_index   # Edge connectivity [2, num_edges]
data.num_nodes    # Number of nodes
data.num_edges    # Number of edges
```

## Visualization Options

### RDKit-Style Layout

```python
cg.plot_graph_rd_c_style(
    show=True,           # Display plot
    save_path='graph.png',  # Save to file
    figsize=(12, 8)      # Figure size
)
```

### Fragment Highlighting

```python
# Highlight fragments in original molecule
gsge.plot_GS_fragments_in_mol(
    smiles,
    args={
        'color_method': 'method1',
        'color_seed': 42,
        'annotate_with_index': True,
        'annotate_atoms': True
    }
)
```

## Use Cases

### Graph Neural Networks

```python
import torch
from torch_geometric.nn import GCNConv

# Use compound graphs for GNN training
cgs_data = gsge.make_compound_graphs(smiles_list, pyg_data=True)

# Define GNN model
class FragmentGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x

# Train on compound graphs
model = FragmentGNN(128, 64, 1)
```

### Molecular Comparison

```python
# Compare molecules at fragment level
cg1 = gsge.get_CG_from_smiles(smiles1, return_CG_object=True)
cg2 = gsge.get_CG_from_smiles(smiles2, return_CG_object=True)

# Check for common fragments
common_fragments = set(cg1.nodes).intersection(set(cg2.nodes))
print(f"Shared fragments: {len(common_fragments)}")
```

### Virtual Screening

```python
# Screen library based on fragment composition
library_graphs = gsge.make_compound_graphs(library_smiles, pyg_data=True)

# Filter by fragment content
target_fragments = {10, 25, 42}  # Fragment IDs of interest
hits = []
for i, data in enumerate(library_graphs):
    fragments = set(data.x.squeeze().tolist())
    if target_fragments.issubset(fragments):
        hits.append(library_smiles[i])
```

## Performance Tips

1. **Use Parallel Processing**: Set `workers=4` or higher for large datasets
2. **Check Coverage First**: Ensure all molecules are fully covered by vocabulary
3. **Cache Graphs**: Save graphs to avoid recomputation
4. **PyG Format**: Use `pyg_data=True` for GNN applications

## Troubleshooting

### Graph Creation Fails

**Problem**: Cannot create compound graph for some molecules

**Solutions**:
- Run `gsge.add_all_single_elements()` before graph creation
- Check vocabulary coverage with `check_for_graphs_groupings()`
- Add missing fragments to vocabulary

### Memory Issues

**Problem**: Out of memory when creating many graphs

**Solutions**:
- Process in batches
- Reduce number of workers
- Use raw format instead of PyG format initially

### Visualization Issues

**Problem**: Graph visualization fails or looks incorrect

**Solutions**:
- Ensure matplotlib and rdkit are installed
- Try different layout algorithms
- Check that compound graph has valid structure

## Expected Output

### Typical Compound Graph Statistics

| Molecule Type | Nodes | Edges | Example |
|---------------|-------|-------|---------|
| Small molecule | 2-5 | 1-4 | Ethanol: 2 nodes, 1 edge |
| Drug-like | 5-15 | 4-14 | Ibuprofen: ~8 nodes |
| Peptide | 10-30 | 9-29 | Tripeptide: ~12 nodes |
| Cyclic peptide | 15-40 | 15-41 | Cyclic decapeptide: ~25 nodes |

### Adjacency Matrix Format

```python
adj_matrix = array([
    [0, 2, 3, ...],  # Source nodes
    [1, 1, 2, ...]   # Target nodes
])
# Each column represents an edge
```

## Next Steps

After creating compound graphs:

1. **Train GNN**: Use graphs for property prediction
2. **Generate Embeddings**: See [`GAE/`](../03_GAE/README.md)
3. **Analyze Chemical Space**: See [`mol_frag_features/`](../05_mol_frag_features/README.md)

## Additional Resources

- [User Guide: Compound Graphs](../../docs/user-guide/compound-graphs.md)
- [API Reference: compound_graph](../../docs/api-reference/gsge.md#compound-graphs)


---

[Back to Tutorials Overview](../README.md)
- [PyTorch Geometric Documentation](https://pytorch-geometric.readthedocs.io/)
