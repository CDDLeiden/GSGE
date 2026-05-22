# Graph Autoencoder

Graph neural network components for learning fragment embeddings.

## AttentiveFP

Graph attention encoder from PyTorch Geometric.

```python
from torch_geometric.nn.models.attentive_fp import AttentiveFP

encoder = AttentiveFP(
    in_channels=9,
    hidden_channels=256,
    out_channels=128,
    edge_dim=3,
    num_layers=3,
    num_timesteps=2
)
```

## GraphDecoder

Decoder for reconstructing molecular graphs from latent embeddings.

::: GSGE.graphs.fragment_graph.GAE.GraphDecoder
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## GraphAutoencoderTrainer

Training loop for the Graph Autoencoder.

::: GSGE.graphs.fragment_graph.GAE.GraphAutoencoderTrainer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## MetricsTracker

Tracks and computes epoch-level training metrics.

::: GSGE.graphs.fragment_graph.GAE.MetricsTracker
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Loss Functions

### compute_atom_loss

::: GSGE.graphs.fragment_graph.GAE.compute_atom_loss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

### compute_edge_loss

::: GSGE.graphs.fragment_graph.GAE.compute_edge_loss
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Usage Examples

### Training Setup

```python
from GSGE import GSGE, GS_Vocab, GSGE_Corpus
from GSGE.graphs.fragment_graph.GAE import AttentiveFP, GraphDecoder, GraphAutoencoderTrainer
from GSGE.core_gsge import CoreGSGE
import torch

# Build vocabulary and corpus
vocab = GS_Vocab()
vocab.build_vocab(m_set=smiles, convert=True, target=200)

corpus = GSGE_Corpus()
corpus.build_corpus(m_set=smiles, convert=True)

# Load data
train_loader, val_loader = CoreGSGE.load_and_prepare_data(
    GS_vocab=vocab,
    GSGE_corpus=corpus,
    batch_size=64,
    val_percentage=0.2,
    split_seed=42
)

# Create models
encoder = AttentiveFP(
    in_channels=9,
    hidden_channels=256,
    out_channels=128,
    edge_dim=3,
    num_layers=3,
    num_timesteps=2
)

decoder = GraphDecoder(
    latent_dim=128,
    hidden_dim=256
)

# Optimizer
optimizer = torch.optim.Adam(
    list(encoder.parameters()) + list(decoder.parameters()),
    lr=0.001
)

# Create trainer
trainer = GraphAutoencoderTrainer(
    encoder=encoder,
    decoder=decoder,
    optimizer=optimizer,
    train_loader=train_loader,
    val_loader=val_loader,
    checkpoint_dir='./checkpoints',
    device='cuda',
    batch_size=64
)

# Train
trainer.train(
    num_epochs=300,
    checkpoint_interval=5
)
```

### Using with GSGE

```python
from GSGE import GSGE

gsge = GSGE(GS_vocab=vocab, GSGE_corpus=corpus)

# Train via GSGE interface
gsge.train_GSGE_Auto_Encoder(
    batch_size=64,
    num_epochs=300,
    checkpoint_interval=5,
    val_percentage=0.2,
    split_seed=42,
    device='cuda',
    checkpoint_dir='./checkpoints'
)
```

### Loading Checkpoints

```python
# Load checkpoint
checkpoint = torch.load('checkpoints/checkpoint_epoch_100.pth')

# Restore model states
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

# Or use GSGE method
gsge.load_GAE_weights('checkpoints/checkpoint_epoch_100.pth')
```
