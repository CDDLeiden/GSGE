# Embeddings

PyTorch embedding layer for efficient fragment embedding lookup.

## GSGE_Embedding

::: GSGE.embedding.GSGE_Embedding
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Usage Examples

### Creating Embedding Layer

```python
from GSGE import GSGE
from GSGE.embedding import GSGE_Embedding

# Get embeddings from GSGE
gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')
embeddings = gsge.get_fragment_embeddings()

# Create PyTorch embedding layer
emb_layer = GSGE_Embedding(
    num_OHE_tokens=0,
    OHE_emb_dim=None,
    num_frag_tokens=0,
    frag_emb=embeddings,
    only_token2vec=True,
    no_grad=True  # Freeze embeddings
)

# Use in model
import torch
token_ids = torch.tensor([[1, 2, 3, 4]])
embedded = emb_layer(token_ids)  # [batch, seq_len, emb_dim]
```

### Generating Embeddings

```python
from GSGE import GSGE

gsge = GSGE(GS_vocab=vocab)

# Set encoder
gsge.set_encoder()  # Uses AttentiveFP by default

# Load trained weights
gsge.load_GAE_weights('checkpoint_epoch_100.pth')

# Generate embeddings for all fragments
gsge.make_GS_fragment_embedding_dict(
    device='cuda',
    batch_size=64
)

# Get embeddings
embeddings = gsge.get_fragment_embeddings()
print(embeddings.shape)  # [num_fragments, embedding_dim]
```

### Combined Embeddings and Descriptors

```python
# Calculate descriptors
gsge.calc_fragment_descriptors(
    descriptor_keys=['MolWt', 'TPSA', 'NumHDonors']
)

# Get combined features
combined = gsge.get_fragment_descriptors_and_embeddings()
print(combined.shape)  # [num_fragments, emb_dim + desc_dim]
```
