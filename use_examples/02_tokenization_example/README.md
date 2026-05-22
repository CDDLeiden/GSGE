# Tokenization Examples

This tutorial demonstrates how to tokenize molecules into sequences of fragment tokens using GSGE.

## Tutorials in This Module

| Tutorial | Time Category | Time Estimate | Difficulty | Description |
|----------|---------------|---------------|------------|-------------|
| tokenization_tutorial.ipynb | Short | 15 min (CPU/GPU) | Beginner | Tokenize molecules into sequences |

## Prerequisites

- [x] GSGE installed (see [Installation Guide](../../docs/getting-started/installation.md))
- [ ] Basic Python knowledge
- [ ] Completed: None (independent module)

## Learning Objectives

After completing this module, you will be able to:
- Tokenize molecules into fragment sequences
- Understand token types (grammar, elements, fragments)
- Create token vocabularies
- Handle batch processing and padding

## Overview

Tokenization converts molecules into sequences of tokens representing:
- **Grammar tokens**: SELFIES syntax (Branch, Ring1, pop, etc.)
- **Element tokens**: Single atoms (C, N, O, S, etc.)
- **Fragment tokens**: Molecular fragments from vocabulary (GS_frag_0, GS_frag_1, etc.)

This enables using GSGE with sequence models like RNNs, Transformers, and language models.

## Tutorial Notebook

**`tokenization_tutorial.ipynb`**

This tutorial covers:

1. **Single Molecule Tokenization**
   - Converting SMILES to token sequences
   - Understanding token types
   - Token-to-ID conversion

2. **Batch Tokenization**
   - Parallel processing of multiple molecules
   - Padding sequences to equal length
   - Creating attention masks

3. **Token Vocabularies**
   - Token-to-ID mappings
   - Vocabulary size and composition
   - Special tokens ([PAD], [MASK], etc.)

4. **Integration with Models**
   - Using tokens with PyTorch
   - Embedding lookup
   - Sequence modeling workflows

## Quick Start

### Single Molecule

```python
from GSGE import GSGE

# Load GSGE with vocabulary
gsge = GSGE(GS_vocab='path/to/vocab.pkl')

# Tokenize molecule
smiles = 'CCO'  # Ethanol
tokens = gsge.preprocess_from_SMILES(smiles)

print(f"Tokens: {tokens}")
# Output: ['GS_frag_5', 'GS_frag_12', 'O', ...]
```

### Batch Processing

```python
smiles_list = ['CCO', 'c1ccccc1', 'CC(C)O', 'CC(=O)O']

# Parallel tokenization
tokenized = gsge.parallel_tokenize_SMILES_list(
    smiles_list,
    max_workers=4
)

print(f"Tokenized {len(tokenized)} molecules")
for tokens in tokenized:
    print(f"Length: {len(tokens)}, Tokens: {tokens[:5]}...")
```

### Token-to-ID Conversion

```python
# Get token dictionary
token_dict = gsge.tokenizer.token_dict

# Convert tokens to IDs
token_ids = []
for tokens in tokenized:
    ids = [token_dict.get(token, 0) for token in tokens]  # 0 = [PAD]
    token_ids.append(ids)

# Convert to PyTorch tensor
import torch
token_tensor = torch.tensor(token_ids)
print(token_tensor.shape)  # [batch_size, seq_len]
```

## Token Types

### Grammar Tokens

SELFIES syntax elements:

| Token | Description | Example Use |
|-------|-------------|-------------|
| `Branch` | Start branch | Branching points |
| `Ring1`, `Ring2` | Ring closures | Cyclic structures |
| `pop` | End branch | Return from branch |
| `=Branch` | Double bond branch | Unsaturated branches |
| `#Branch` | Triple bond branch | Alkynes |

### Element Tokens

Single atom representations:

| Token | Description |
|-------|-------------|
| `C` | Carbon |
| `N` | Nitrogen |
| `O` | Oxygen |
| `S` | Sulfur |
| `F`, `Cl`, `Br`, `I` | Halogens |
| `:0`, `:1`, etc. | Bond descriptors |

### Fragment Tokens

Vocabulary fragments:

```
GS_frag_0, GS_frag_1, GS_frag_2, ...
```

Each represents a molecular fragment from your vocabulary.

## Padding and Masking

### Padding Sequences

```python
from torch.nn.utils.rnn import pad_sequence

# Convert to tensors
token_tensors = [torch.tensor(ids) for ids in token_ids]

# Pad to equal length
padded = pad_sequence(token_tensors, batch_first=True, padding_value=0)
print(padded.shape)  # [batch_size, max_seq_len]
```

### Creating Attention Masks

```python
# Create mask (1 for real tokens, 0 for padding)
attention_mask = (padded != 0).long()

# Use in Transformer
outputs = transformer_model(
    input_ids=padded,
    attention_mask=attention_mask
)
```

## Integration with Sequence Models

### RNN Example

```python
import torch.nn as nn

class MoleculeRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)  # Property prediction

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)  # [batch, seq, emb]
        _, (hidden, _) = self.lstm(embedded)  # [1, batch, hidden]
        output = self.fc(hidden.squeeze(0))   # [batch, 1]
        return output

# Create model
vocab_size = len(gsge.tokenizer.token_dict)
model = MoleculeRNN(vocab_size, embedding_dim=128, hidden_dim=256)

# Train on tokenized molecules
predictions = model(padded)
```

### Transformer Example

```python
from transformers import BertConfig, BertForSequenceClassification

# Configure BERT for molecules
config = BertConfig(
    vocab_size=len(gsge.tokenizer.token_dict),
    hidden_size=256,
    num_hidden_layers=6,
    num_attention_heads=8,
    max_position_embeddings=512
)

model = BertForSequenceClassification(config, num_labels=1)

# Train on tokenized molecules
outputs = model(
    input_ids=padded,
    attention_mask=attention_mask,
    labels=property_labels
)
```

### Using Learned Fragment Embeddings

```python
from GSGE.embedding import GSGE_Embedding

# Load GSGE with trained embeddings
gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')

# Get pre-trained fragment embeddings
fragment_embeddings = gsge.get_fragment_embeddings()
token_vocab = gsge.get_GSGE_vocab()

# Create embedding layer with learned embeddings
# When only_token2vec=True, sparse_vocab_size is the vocabulary size
# and GSGE_combined_embeddings provides the pre-trained fragment embeddings
embedding_layer = GSGE_Embedding(
    sparse_vocab_size=len(token_vocab),  # Vocabulary size
    dense_size=fragment_embeddings.shape[1],  # Embedding dimension
    embedding_dim=128,  # For sparse tokens (not used when only_token2vec=True)
    GSGE_combined_embeddings=fragment_embeddings,  # Pre-trained embeddings
    only_token2vec=True,  # Only use pre-trained embeddings (no learned sparse embeddings)
    no_grad=True  # Freeze embeddings
)

# Use in model
embedded = embedding_layer(padded)  # [batch, seq, emb_dim]
```

## Use Cases

### Molecular Generation

```python
# Train generative model on tokenized molecules
# Generate new molecules by sampling token sequences

# Example: Train GPT-style model
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config(vocab_size=len(token_dict), n_positions=512)
model = GPT2LMHeadModel(config)

# Train on tokenized molecules
# Generate by sampling from model
```

### Property Prediction

```python
# Use tokens as input to prediction model
# Example: Predict logP, solubility, bioactivity

# Tokenize molecules
tokens = gsge.parallel_tokenize_SMILES_list(smiles_list)
token_ids = convert_to_ids(tokens)

# Predict properties
predictions = model(token_ids)
```

### Molecular Similarity

```python
# Compare molecules via token overlap

def token_similarity(tokens1, tokens2):
    set1 = set(tokens1)
    set2 = set(tokens2)
    jaccard = len(set1 & set2) / len(set1 | set2)
    return jaccard

# Compare two molecules
smiles1 = 'CCO'
smiles2 = 'CCCO'
tokens1 = gsge.preprocess_from_SMILES(smiles1)
tokens2 = gsge.preprocess_from_SMILES(smiles2)
similarity = token_similarity(tokens1, tokens2)
print(f"Similarity: {similarity:.2f}")
```

## Vocabulary Statistics

### Checking Vocabulary Composition

```python
token_dict = gsge.tokenizer.token_dict

# Count token types
grammar_tokens = [t for t in token_dict if t in ['Branch', 'Ring1', 'pop', ...]]
element_tokens = [t for t in token_dict if t in ['C', 'N', 'O', ...]]
fragment_tokens = [t for t in token_dict if t.startswith('GS_frag_')]

print(f"Grammar tokens: {len(grammar_tokens)}")
print(f"Element tokens: {len(element_tokens)}")
print(f"Fragment tokens: {len(fragment_tokens)}")
print(f"Total vocabulary size: {len(token_dict)}")
```

### Token Frequency Analysis

```python
from collections import Counter

# Tokenize dataset
all_tokens = []
for smiles in smiles_list:
    tokens = gsge.preprocess_from_SMILES(smiles)
    all_tokens.extend(tokens)

# Count frequencies
token_freq = Counter(all_tokens)

# Most common tokens
print("Top 10 tokens:")
for token, count in token_freq.most_common(10):
    print(f"{token}: {count}")
```

## Performance Tips

1. **Parallel Processing**: Use `parallel_tokenize_SMILES_list` for >100 molecules
2. **Caching**: Cache tokenized sequences to avoid reprocessing
3. **Batch Size**: Use appropriate batch sizes for your GPU memory
4. **Padding**: Minimize padding by sorting by length before batching

## Troubleshooting

### Tokenization Fails

**Problem**: Cannot tokenize some SMILES

**Solutions**:
- Check SMILES validity with RDKit
- Ensure vocabulary covers all fragments
- Add single elements to vocabulary
- Check for special characters

### Memory Issues

**Problem**: Out of memory during batch tokenization

**Solutions**:
- Reduce `max_workers` parameter
- Process in smaller batches
- Use sequential tokenization for large molecules

### Token ID Mismatch

**Problem**: Token not found in dictionary

**Solutions**:
- Ensure same vocabulary used for tokenization and model
- Check for case sensitivity
- Verify vocabulary completeness

## Expected Output

### Typical Token Sequence Lengths

| Molecule Type | Token Length | Example |
|---------------|--------------|---------|
| Small molecule | 5-15 | Ethanol: ~8 tokens |
| Drug-like | 15-50 | Ibuprofen: ~30 tokens |
| Peptide | 30-100 | Tripeptide: ~50 tokens |
| Cyclic peptide | 50-200 | Cyclic decapeptide: ~120 tokens |

### Token Distribution

Typical vocabulary breakdown:
- Grammar tokens: ~20
- Element tokens: ~30
- Fragment tokens: 100-500 (depends on vocabulary)
- Total: 150-550 tokens

## Next Steps

After tokenizing molecules:

1. **Train Sequence Model**: RNN, LSTM, Transformer
2. **Molecular Generation**: Generate new molecules
3. **Property Prediction**: Predict molecular properties
4. **Use with Embeddings**: See [`use_embeddings/`](../04_use_embeddings/README.md)

## Additional Resources

- [User Guide: Tokenization](../../docs/user-guide/tokenization.md)
- [API Reference: Tokenizer](../../docs/api-reference/tokenizer.md)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)


---

[Back to Tutorials Overview](../README.md)
- [PyTorch RNN Tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
