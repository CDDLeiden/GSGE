# Tokenizer

Tokenization and preprocessing utilities for converting molecules to fragment sequences.

## GSGE_tokenizer

::: GSGE.tokenizer.GSGE_tokenizer
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Usage Examples

### Single Molecule Tokenization

```python
from GSGE import GSGE

gsge = GSGE(GS_vocab=vocab)

# Tokenize single molecule
tokens = gsge.preprocess_from_SMILES('CCO')
print(tokens)  # ['GS_frag_5', 'GS_frag_12', 'O', ...]
```

### Parallel Tokenization

```python
# Tokenize list of molecules
smiles_list = ['CCO', 'c1ccccc1', 'CC(C)O']

tokenized = gsge.parallel_tokenize_SMILES_list(
    smiles_list,
    max_workers=4
)

# Returns list of token sequences
print(f"Tokenized {len(tokenized)} molecules")
```

### Token-to-ID Conversion

```python
# Get token dictionary
token_dict = gsge.tokenizer.token_dict

# Convert tokens to IDs
token_ids = [token_dict.get(token, 0) for token in tokens]
```
