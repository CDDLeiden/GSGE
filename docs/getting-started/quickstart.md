# Quick Start

This guide will walk you through the basics of using GSGE to create vocabularies, generate embeddings, and create compound graphs.

## Prerequisites

Ensure you have [installed GSGE](installation.md) successfully.

## Your First GSGE Workflow

We'll go through a complete workflow: building a vocabulary → creating a corpus → making compound graphs.

### 1. Import GSGE

```python
from GSGE import GSGE, GS_Vocab, GSGE_Corpus, CUSTOM_fragment_mol
```

### 2. Prepare Your Molecules

Start with a list of SMILES strings representing your molecules:

```python
smiles_list = [
    'CCO',  # Ethanol
    'c1ccccc1',  # Benzene
    'CC(C)O',  # Isopropanol
    'CC(=O)O',  # Acetic acid
    'c1ccc(cc1)O',  # Phenol
    'CC(C)(C)O',  # Tert-butanol
    'CCN',  # Ethylamine
    'c1ccncc1',  # Pyridine
]
```

### 3. Build a Vocabulary

The vocabulary contains merged, generalized fragments used as nodes in compound graphs:

```python
# Create vocabulary object
vocab = GS_Vocab()

# Build vocabulary from molecules
vocab.build_vocab(
    m_set=smiles_list,
    convert=True,  # Input is SMILES (not RDKit Mol objects)
    target=50,  # Target vocabulary size
    MIN_SIZE=1,  # Minimum fragment size
    MAX_SIZE=10,  # Maximum fragment size
    fragment_mol_fn=CUSTOM_fragment_mol  # Custom bond-cutting function
)

print(f"Vocabulary contains {vocab.num_fragments} fragments")

# View some fragments
print("Example fragments:")
for i, frag in enumerate(vocab.fragments[:5]):
    print(f"  {i}: {frag}")
```

### 4. Build a Corpus (Optional)

The corpus contains non-merged fragments for training the Graph Autoencoder:

```python
# Create corpus object
corpus = GSGE_Corpus()

# Build corpus from molecules
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=10,
    fragment_mol_fn=CUSTOM_fragment_mol,
    fragmented=False  # Keep non-merged fragments
)

print(f"Corpus contains {corpus.num_fragments} fragments")
```

### 5. Create GSGE Instance

Combine vocabulary and corpus into a GSGE instance:

```python
# Create GSGE object
gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)

# Add single elements for complete coverage
gsge.add_all_single_elements()

print(f"GSGE initialized with {len(gsge.get_fragments_smiles())} total fragments")
```

### 6. Create Compound Graphs

Represent molecules as graphs of fragments:

```python
# Create a single compound graph
smiles = 'c1ccc(cc1)O'  # Phenol
cg = gsge.get_CG_from_smiles(smiles, return_CG_object=True)

print(f"Compound graph for {smiles}:")
print(f"  Nodes (fragments): {len(cg.nodes)}")
print(f"  Edges: {len(cg.edges)}")

# Visualize the graph
cg.plot_graph_rd_c_style()
```

### 7. Batch Create Compound Graphs

Process multiple molecules in parallel:

```python
# Create graphs for all molecules
cgs_data = gsge.make_compound_graphs(
    smiles_list,
    workers=4,  # Use 4 parallel workers
    pyg_data=True  # Return PyTorch Geometric Data format
)

print(f"Created {len(cgs_data)} compound graphs")

# Inspect first graph
data = cgs_data[0]
print(f"First graph: {data.num_nodes} nodes, {data.num_edges} edges")
```

### 8. Tokenize Molecules

Convert molecules to sequences of fragment tokens:

```python
# Tokenize a single molecule
smiles = 'CCO'
tokens = gsge.preprocess_from_SMILES(smiles)

print(f"Tokens for {smiles}:")
print(tokens[:10])  # First 10 tokens

# Tokenize multiple molecules in parallel
tokenized = gsge.parallel_tokenize_SMILES_list(
    smiles_list,
    max_workers=4
)

print(f"Tokenized {len(tokenized)} molecules")
print(f"First tokenized sequence length: {len(tokenized[0])}")
```

### 9. Save and Load

Save your GSGE instance for later use:

```python
# Save
gsge.save_gsge_data(
    'my_gsge.pkl',
    meta_info='Quick start example vocabulary'
)

# Load
gsge_loaded = GSGE(GSGE_load_path='my_gsge.pkl')
print(f"Loaded GSGE with {len(gsge_loaded.get_fragments_smiles())} fragments")
```

## Using Pre-trained Vocabulary

GSGE includes a pre-trained vocabulary for small molecules and peptides:

```python
from GSGE import GSGE, get_tests_dir

# Load pre-trained vocabulary
tests_dir = get_tests_dir()
pkl_path = tests_dir / 'test_gsge_save_with_descriptors.pkl'
gsge = GSGE(GSGE_load_path=pkl_path)

print(f"Pre-trained vocabulary loaded: {len(gsge.get_fragments_smiles())} fragments")

# Use it immediately
cg = gsge.get_CG_from_smiles('CC(C)CC1NC(=O)C(Cc2ccccc2)NC1=O', return_CG_object=True)
cg.plot_graph_rd_c_style()
```

## Complete Example Script

Here's a complete working example:

```python
from GSGE import GSGE, GS_Vocab, GSGE_Corpus, CUSTOM_fragment_mol

# Prepare data
smiles_list = ['CCO', 'c1ccccc1', 'CC(C)O', 'CC(=O)O']

# Build vocabulary
vocab = GS_Vocab()
vocab.build_vocab(
    m_set=smiles_list,
    convert=True,
    target=30,
    MIN_SIZE=1,
    MAX_SIZE=8,
    fragment_mol_fn=CUSTOM_fragment_mol
)

# Build corpus
corpus = GSGE_Corpus()
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=8,
    fragment_mol_fn=CUSTOM_fragment_mol
)

# Create GSGE
gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)
gsge.add_all_single_elements()

# Create compound graphs
cgs = gsge.make_compound_graphs(smiles_list, workers=2, pyg_data=False)

# Tokenize molecules
tokens = gsge.parallel_tokenize_SMILES_list(smiles_list, max_workers=2)

# Save
gsge.save_gsge_data('quickstart_gsge.pkl')

print(f"✓ Created vocabulary with {vocab.num_fragments} fragments")
print(f"✓ Created corpus with {corpus.num_fragments} fragments")
print(f"✓ Generated {len(cgs)} compound graphs")
print(f"✓ Tokenized {len(tokens)} molecules")
```

## Common Parameters

### Vocabulary Building

- `target`: Target number of fragments in vocabulary (default: 100)
- `MIN_SIZE`: Minimum fragment size in atoms (default: 1)
- `MAX_SIZE`: Maximum fragment size in atoms (default: 10)
- `n_limit`: Maximum number of wildcard connections per fragment (default: 6)
- `fragment_mol_fn`: Custom bond-cutting function (default: `CUSTOM_fragment_mol`)

### Corpus Building

- `min_size`: Minimum fragment size (default: 2)
- `max_size`: Maximum fragment size (default: 12)
- `fragmented`: Whether to keep fragments as-is (False) or merge them (True)

### Compound Graph Creation

- `workers`: Number of parallel workers (default: 1)
- `pyg_data`: Return PyTorch Geometric Data format (True) or raw format (False)

### Tokenization

- `max_workers`: Number of parallel workers for tokenization

## Next Steps

Now that you've completed the quick start:

- **Learn about vocabularies**: See [Vocabularies & Corpus Guide](../user-guide/vocabularies.md)
- **Explore the API Reference**: See [detailed API documentation](../api-reference/index.md)

## Troubleshooting

### "No fragments generated"

If vocabulary building produces no fragments:

- Increase `MAX_SIZE` parameter
- Check that your SMILES are valid
- Try using default bond-cutting instead of `CUSTOM_fragment_mol`

### "Cannot create compound graph"

If compound graph creation fails:

- Run `gsge.add_all_single_elements()` first
- Check coverage: `gsge.check_for_graphs_groupings([smiles])`
- Add missing fragments manually

### Memory issues with large datasets

For large molecule sets:

- Process in batches
- Reduce `target` vocabulary size
- Use more workers for parallel processing

## Getting Help

- **API Reference**: See [detailed API documentation](../api-reference/index.md)
- **Examples**: Browse [tutorials](../tutorials/index.md) for more examples
- **Issues**: Report bugs on [GitHub](https://github.com/CDDLeiden/GSGE/issues)
