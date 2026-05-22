# Graph Autoencoder (GAE) Training

This tutorial demonstrates how to train a Graph Autoencoder to learn continuous embeddings for molecular fragments.

> **GPU Recommendation**
> GAE training is significantly faster with GPU:
> - **CPU:** 4-6 hours for 300 epochs
> - **GPU:** 1-2 hours for 300 epochs
>
> If you don't have GPU access, consider using the pre-trained model in `GSGE/tests/`.

## Tutorials in This Module

| Tutorial | Time Category | Time Estimate | Difficulty | Description |
|----------|---------------|---------------|------------|-------------|
| train_GAE.py | Long | 4-6h CPU / 1-2h GPU | Advanced | Train graph autoencoder |
| gae_training_monitor.ipynb | Short | 10 min | Intermediate | Monitor training |
| embedding_visualization.ipynb | Medium | 30 min | Intermediate | Visualize and analyze embeddings |

## Prerequisites

- [x] GSGE installed (see [Installation Guide](../../docs/getting-started/installation.md))
- [ ] Vocabulary and corpus built ([00_making_vocabs](../00_making_vocabs/README.md))
- [ ] GPU recommended (but CPU works)

## Learning Objectives

After completing this module, you will be able to:
- Configure GAE architecture
- Train encoder-decoder on fragment graphs
- Monitor training metrics
- Analyze learned embeddings
- Save and load trained checkpoints

## Overview

The Graph Autoencoder (GAE) learns to:
1. **Encode** fragment molecular graphs into continuous latent vectors
2. **Decode** latent vectors back to reconstruct fragment graphs
3. **Embed** chemical knowledge into the learned representations

This enables using learned embeddings as informative node features instead of sparse one-hot encodings.

## Tutorial Files

| File | Description | Usage |
|------|-------------|-------|
| **`train_GAE.py`** | Training script | Run GAE training |
| **`gae_training_monitor.ipynb`** | Log analysis notebook | Monitor training progress |
| **`embedding_visualization.ipynb`** | Embedding visualization | Visualize and analyze learned embeddings |
| **`model_checkpoints/`** | Saved models | Checkpoint directory |
| **`train_GAE_v5_vocab.log`** | Example log | Training metrics |

## Quick Start

### 1. Prepare Data

First, build vocabulary and corpus (see [`making_vocabs/`](../00_making_vocabs/README.md)).:

```python
from GSGE import GS_Vocab, GSGE_Corpus, CUSTOM_fragment_mol

# Build vocabulary
vocab = GS_Vocab()
vocab.build_vocab(m_set=smiles, convert=True, target=200)
vocab.save_GS_vocab(dir_path='.', vocab_name='my_vocab')

# Build corpus (non-merged fragments)
corpus = GSGE_Corpus()
corpus.build_corpus(
    m_set=smiles,
    convert=True,
    fragmented=False  # Keep non-merged
)
corpus.save_GSGE_corpus(dir_path='.', vocab_name='my_corpus')
```

### 2. Train GAE

**Using Python Script** (Recommended for long training):

```bash
# Edit train_GAE.py to set paths and hyperparameters
python train_GAE.py
```

**Using GSGE API**:

```python
from GSGE import GSGE

gsge = GSGE(
    GS_vocab='my_vocab.pkl',
    GSGE_corpus='my_corpus.pkl'
)

# Train autoencoder
gsge.train_GSGE_Auto_Encoder(
    batch_size=64,
    num_epochs=300,
    checkpoint_interval=5,
    val_percentage=0.2,
    device='cuda',
    checkpoint_dir='./checkpoints'
)
```

### 3. Generate Embeddings

```python
# Load trained model
gsge.set_encoder()  # Uses AttentiveFP
gsge.load_GAE_weights('checkpoints/checkpoint_epoch_100.pth')

# Generate embeddings for all vocabulary fragments
gsge.make_GS_fragment_embedding_dict(device='cuda')

# Get embeddings
embeddings = gsge.get_fragment_embeddings()
print(embeddings.shape)  # [num_fragments, embedding_dim]

# Save for later use
gsge.save_gsge_data('gsge_with_embeddings.pkl')
```

## Training Configuration

### Model Architecture

**Encoder (AttentiveFP)**:
```python
from torch_geometric.nn.models.attentive_fp import AttentiveFP

encoder = AttentiveFP(
    in_channels=9,        # Node features (atom type, etc.)
    hidden_channels=256,  # Hidden layer size
    out_channels=128,     # Embedding dimension
    edge_dim=3,           # Edge features (bond type)
    num_layers=3,         # Number of graph convolution layers
    num_timesteps=2       # Attention timesteps
)
```

**Decoder (Graph Reconstruction)**:
```python
from GSGE.graphs.fragment_graph.GAE import GraphDecoder

decoder = GraphDecoder(
    latent_dim=128,       # Must match encoder out_channels
    hidden_dim=256        # Hidden layer size
)
```

### Hyperparameters

| Parameter | Typical Value | Description |
|-----------|---------------|-------------|
| `batch_size` | 64 | Fragments per batch |
| `num_epochs` | 100-300 | Training epochs |
| `learning_rate` | 0.001 | Adam optimizer LR |
| `val_percentage` | 0.2 | Validation split |
| `checkpoint_interval` | 5 | Save every N epochs |
| `embedding_dim` | 64-256 | Latent space dimension |
| `hidden_dim` | 128-512 | Hidden layer size |

### Recommended Settings

**Small Vocabulary** (<500 fragments):
- Embedding dim: 64
- Hidden dim: 128
- Epochs: 100
- Batch size: 32

**Medium Vocabulary** (500-2000 fragments):
- Embedding dim: 128
- Hidden dim: 256
- Epochs: 200
- Batch size: 64

**Large Vocabulary** (>2000 fragments):
- Embedding dim: 256
- Hidden dim: 512
- Epochs: 300
- Batch size: 128

## Monitoring Training

### Using Log Files

```python
# Training prints metrics to console and log file
# Example output:
Epoch 1/300, Train Loss: 2.45, Val Loss: 2.38
Epoch 2/300, Train Loss: 2.12, Val Loss: 2.08
...
```

### Using gae_training_monitor.ipynb

```python
import pandas as pd
import matplotlib.pyplot as plt

# Parse log file
df = pd.read_csv('train_GAE_v5_vocab.log', ...)

# Plot training curves
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Validation')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
```

## Analyzing Learned Embeddings

### Embedding Visualization

**embedding_visualization.ipynb** demonstrates:

1. **Generating embeddings** from trained GAE models
2. **Dimensionality reduction** (t-SNE and UMAP, 2D and 3D)
3. **Cluster identification** using hierarchical clustering
4. **Interactive visualization** with Plotly
5. **Chemical interpretation** of clusters
6. **Exporting plots** for publications

```python
from GSGE import GSGE

# Load GSGE with embeddings
gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')

# Create clustering object
gsge_clustering = gsge.get_GSGE_clustering(
    embeddings=gsge.get_fragment_embeddings()
)

# 2D UMAP visualization
fig_umap = gsge_clustering.plot_2D_UMAP(random_state=42)
fig_umap.show()

# 3D t-SNE visualization
fig_tsne = gsge_clustering.plot_3D_TSNE()
fig_tsne.show()
```

### What the Visualization Shows

- **t-SNE plots**: 2D and 3D projections for exploratory analysis
- **UMAP plots**: More stable, publication-quality visualizations
- **Cluster grids**: Example molecules from each cluster
- **Interactive plots**: Hover to see SMILES, zoom/pan to explore
- **Export options**: Save as HTML (interactive) or PNG (static)

## Evaluation Metrics

### Reconstruction Quality

The GAE is evaluated on its ability to reconstruct fragment graphs:

**Atom Reconstruction**:
- Predicts atom features (type, hybridization, etc.)
- Measured by cross-entropy loss

**Edge Reconstruction**:
- Predicts bond existence and types
- Measured by binary cross-entropy + type loss

**Graph Size**:
- Predicts number of atoms
- Measured by MSE

### Total Loss

```
Total Loss = α × Atom Loss + β × Edge Loss + γ × Size Loss
```

Typical weights: α=1.0, β=1.0, γ=0.1

### Validation Metrics

Monitor validation loss to detect overfitting:
- **Good**: Val loss decreases steadily
- **Overfitting**: Train loss ↓, val loss ↑
- **Underfitting**: Both losses high and plateau

## Checkpointing

Checkpoints are saved automatically during training:

```
checkpoints/
├── checkpoint_epoch_5.pth
├── checkpoint_epoch_10.pth
├── checkpoint_epoch_15.pth
...
```

Each checkpoint contains:
- Encoder state dict
- Decoder state dict
- Optimizer state dict
- Epoch number
- Train/val losses

### Resuming Training

```python
trainer = GraphAutoencoderTrainer(
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_dir='./checkpoints',
    device='cuda',
    load_checkpoint_path='checkpoints/checkpoint_epoch_50.pth'  # Resume
)

trainer.train(num_epochs=100)  # Continue for 100 more epochs
```

## GPU Training

### Requirements

- CUDA-compatible GPU (recommended: 8GB+ VRAM)
- PyTorch with CUDA support
- PyTorch Geometric

### Setup

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install PyTorch with CUDA (if needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Memory Management

For large vocabularies (>5000 fragments):
- Reduce batch size
- Use gradient accumulation
- Use mixed precision training
- Monitor GPU memory with `nvidia-smi`

## Troubleshooting

### Training Issues

**Loss not decreasing:**
- Check data quality (valid fragments)
- Reduce learning rate
- Increase model capacity
- Verify data preprocessing

**Out of memory:**
- Reduce batch size
- Reduce model hidden dimensions
- Use CPU for small vocabularies
- Enable gradient checkpointing

**NaN loss:**
- Reduce learning rate
- Check for invalid data
- Add gradient clipping
- Normalize inputs

### Checkpoint Issues

**Checkpoint not loading:**
- Verify model architecture matches
- Check PyTorch version compatibility
- Ensure checkpoint path is correct

### Visualization Issues

**Clusters not meaningful:**
- Train for more epochs
- Increase embedding dimension
- Use more training data
- Try different perplexity values for t-SNE

## Best Practices

1. **Data Quality**: Ensure fragments are valid SMILES
2. **Early Stopping**: Monitor validation loss, stop if overfitting
3. **Save Often**: Use checkpoint_interval=5 or less
4. **Validate Embeddings**: Check that similar fragments cluster together
5. **Multiple Runs**: Try different random seeds for robustness

## Next Steps

After training GAE:

1. **Use Embeddings**: See [`use_embeddings/`](../04_use_embeddings/README.md)
2. **Tokenize with Embeddings**: See [`tokenization_example/`](../02_tokenization_example/README.md)
3. **Property Prediction**: Train GNN on compound graphs with fragment embeddings

## Additional Resources

- [API Reference: GAE](../../docs/api-reference/gae.md)
- [AttentiveFP Paper](https://pubs.acs.org/doi/10.1021/acs.jmedchem.9b00959)


---

[Back to Tutorials Overview](../README.md)
- [PyTorch Geometric Docs](https://pytorch-geometric.readthedocs.io/)
