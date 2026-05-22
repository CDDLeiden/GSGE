# Group-SELFIES Graph Embeddings (GSGE)

**GSGE** (Group-SELFIES Graph Embeddings) extends molecular fragment tokenization and node information using learned molecular fragment graph embeddings. It is **functional group aware** while preserving **learned fragment molecular structural information** via graph-based autoencoding.

## Key Features

- **Compact molecular representations** using molecular fragment nodes instead of atoms
- **Learned fragment embeddings** in continuous latent space via graph autoencoder
- **Functional group awareness** preserving chemical knowledge
- **Designed and tested** on complex molecular structures, particularly cyclic peptides

![GSGE Compound Graph](images/compound_graph.png)
*Figure: GSGE compound graph representation of a cyclic peptide*

## What is GSGE?

GSGE provides a framework for representing molecules as graphs of molecular fragments rather than individual atoms. Each fragment is embedded in a learned continuous latent space that captures its chemical and structural properties.

### Core Components

1. **GS_Vocab (Vocabulary)**: Merged, generalized molecular fragments used as nodes in compound graphs
2. **GSGE_Corpus**: Non-merged fragments for training the Graph Autoencoder
3. **Graph Autoencoder (GAE)**: Neural network that learns continuous embeddings for fragments
4. **Compound Graphs**: Molecule representations where nodes are fragments and edges connect bonded fragments
5. **Tokenization**: Convert molecules to sequences of fragment tokens for downstream tasks

### Workflow Overview

![GSGE Workflow](images/workflow_overview.png)
*Figure: Complete GSGE workflow from molecules to embeddings*

The typical GSGE workflow involves:

1. **Build vocabulary** from your molecular dataset
2. **Create corpus** of fragments for autoencoder training
3. **Train Graph Autoencoder** to learn fragment embeddings
4. **Generate embeddings** for all vocabulary fragments
5. **Use embeddings** for tokenization, compound graphs, or downstream tasks

## Quick Example

```python
from GSGE import GSGE, GS_Vocab, GSGE_Corpus

# Build vocabulary from SMILES
vocab = GS_Vocab()
vocab.build_vocab(
    m_set=smiles_list,
    convert=True,
    target=200,
    MIN_SIZE=1,
    MAX_SIZE=15
)

# Build corpus for GAE training
corpus = GSGE_Corpus()
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=15
)

# Create GSGE instance
gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)
gsge.add_all_single_elements()

# Create compound graphs
cg = gsge.get_CG_from_smiles('CCO', return_CG_object=True)
cg.plot_graph_rd_c_style()
```

## Visualization Example

Below is an example of learned fragment embeddings visualized using t-SNE. The clustering demonstrates that chemically similar fragments are embedded near each other in latent space.

![Fragment Embeddings](images/GSGE_img1.png)
*Figure: t-SNE visualization of 9607 unique molecular fragment embeddings*

## Getting Started

- [Installation Guide](getting-started/installation.md) - Set up GSGE in your environment
- [Quick Start Tutorial](getting-started/quickstart.md) - Build your first vocabulary and compound graphs
- [User Guide](user-guide/index.md) - Comprehensive guides for all GSGE features
- [API Reference](api-reference/index.md) - Detailed API documentation

## Use Cases

GSGE is particularly well-suited for:

- **Cyclic peptides** and complex macrocycles
- **Molecular design** with fragment-based approaches
- **Chemical space visualization** and analysis
- **Molecule generation** using learned fragment representations
- **Property prediction** leveraging structural embeddings

## Requirements

- Python 3.10, 3.11, or 3.12
- PyTorch and PyTorch Geometric
- RDKit for molecular operations
- NumPy <= 2.3.0
- group-selfies (forked version with critical fixes)

See [Installation](getting-started/installation.md) for complete dependency details.

## Citation

If you use GSGE in your research, please cite:

```bibtex
@software{gsge_package2026,
  title = {GSGE: Group-SELFIES Graph Embeddings},
  author = {Durinck, Jasper and Khalil, Bola},
  year = {2026},
  url = {https://github.com/CDDLeiden/GSGE}
}
```
and
```bibtex
@article{gsge_paper2026,
  title = {GSGE: A Group-SELFIES-Based Molecular Graph Representation Evaluated for Estimation of Cyclic Peptide Membrane Permeability},
  authors = {Khalil, Bola and Durinck, Jasper J. and Combs, Steven and Wasserman, Anne Mai and Dyubankova, Natalia and van Vlijmen, Herman and van Westen, Gerard J.P.},
  year = {2026},
  publisher = {},
}
```

## License

GSGE is released under the [MIT License](about/license.md).

## Contributing

We welcome contributions! Please see our [Contributing Guide](development/contributing.md) for details on:

- Setting up development environment
- Running tests
- Submitting pull requests
- Code style guidelines

## Support

- **Issues**: [GitHub Issues](https://github.com/CDDLeiden/GSGE/issues)
- **Discussions**: [GitHub Discussions](https://github.com/CDDLeiden/GSGE/discussions)
- **Documentation**: This site provides comprehensive guides and API reference

## Next Steps

Ready to get started? Follow our [Installation Guide](getting-started/installation.md) to set up GSGE in your environment.
