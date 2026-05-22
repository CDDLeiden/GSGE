# Vocabularies

Fragment vocabulary and corpus management classes.

## GS_Vocab

Vocabulary of merged, generalized molecular fragments used as nodes in compound graphs.

::: GSGE.vocab.GS_Vocab
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## GSGE_Corpus

Corpus of non-merged fragments for Graph Autoencoder training.

::: GSGE.vocab.GSGE_Corpus
    options:
      show_root_heading: true
      show_source: false
      heading_level: 3

## Usage Examples

### Building a Vocabulary

```python
from GSGE import GS_Vocab
from GSGE.fragment_functions import CUSTOM_fragment_mol

vocab = GS_Vocab()
vocab.build_vocab(
    m_set=smiles_list,
    convert=True,
    target=200,
    MIN_SIZE=1,
    MAX_SIZE=15,
    fragment_mol_fn=CUSTOM_fragment_mol
)

# Add fragments manually
vocab.add_GS_fragment('O=C(*1)(*1)')

# Save
vocab.save_GS_vocab(
    dir_path='./vocabs',
    vocab_name='my_vocab',
    meta_info='Custom vocabulary'
)
```

### Building a Corpus

```python
from GSGE import GSGE_Corpus

corpus = GSGE_Corpus()
corpus.build_corpus(
    m_set=smiles_list,
    convert=True,
    min_size=1,
    max_size=15,
    fragmented=False  # Non-merged for GAE
)

# Save
corpus.save_GSGE_corpus(
    dir_path='./corpora',
    vocab_name='my_corpus'
)
```
