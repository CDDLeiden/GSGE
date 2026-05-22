# Fragment Descriptors

RDKit molecular descriptor calculation and normalization.

## calc_mol_frag_descriptors

Calculate RDKit descriptors for a molecular fragment.

::: GSGE.fragment_descriptors.calc_mol_frag_descriptors
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## get_mol_frag_descriptors

Get descriptor lookup table for multiple fragments.

::: GSGE.fragment_descriptors.get_mol_frag_descriptors
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## normalize_descriptors

Z-score normalization with variance filtering.

::: GSGE.fragment_descriptors.normalize_descriptors
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Usage Examples

### Calculate Descriptors for Single Fragment

```python
from GSGE.fragment_descriptors import calc_mol_frag_descriptors

# From SMILES
descriptors = calc_mol_frag_descriptors(
    smiles='CCO',
    descriptor_keys=['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors']
)

print(descriptors)  # {'MolWt': 46.07, 'TPSA': 20.23, ...}
```

### Calculate Descriptors for Multiple Fragments

```python
from GSGE import GSGE

gsge = GSGE(GS_vocab=vocab)

# Calculate for all fragments in vocabulary
gsge.calc_fragment_descriptors(
    descriptor_keys=['MolWt', 'TPSA', 'NumHDonors', 'NumHAcceptors', 'NumRotatableBonds']
)

# Get descriptor matrix
descriptors = gsge.get_fragment_descriptors()
print(descriptors.shape)  # [num_fragments, num_descriptors]
```

### Normalize Descriptors

```python
from GSGE.fragment_descriptors import normalize_descriptors
import numpy as np

# Raw descriptors
raw_descriptors = np.array([
    [46.07, 20.23, 1, 1],
    [78.11, 40.46, 2, 2],
    [60.10, 37.30, 2, 2]
])

# Normalize (z-score)
normalized, means, stds, mask = normalize_descriptors(raw_descriptors)

print(f"Normalized shape: {normalized.shape}")
print(f"Kept {mask.sum()} / {len(mask)} features (removed zero-variance)")
```

### Combine with Embeddings

```python
from GSGE import GSGE

gsge = GSGE(GSGE_load_path='gsge_with_embeddings.pkl')

# Calculate descriptors
gsge.calc_fragment_descriptors(
    descriptor_keys=['MolWt', 'TPSA', 'NumHDonors']
)

# Get combined features
combined = gsge.get_fragment_descriptors_and_embeddings()

embeddings = gsge.get_fragment_embeddings()
descriptors = gsge.get_fragment_descriptors()

print(f"Embeddings: {embeddings.shape}")
print(f"Descriptors: {descriptors.shape}")
print(f"Combined: {combined.shape}")  # [num_fragments, emb_dim + desc_dim]
```

### Available RDKit Descriptors

Common descriptor keys include:

- **Molecular properties**: `MolWt`, `ExactMolWt`, `HeavyAtomMolWt`
- **Topological**: `NumHDonors`, `NumHAcceptors`, `NumRotatableBonds`
- **Surface area**: `TPSA`, `LabuteASA`
- **Counts**: `NumHeteroatoms`, `NumAromaticRings`, `NumSaturatedRings`
- **Electronic**: `Chi0`, `Chi1`, `BalabanJ`

See [RDKit documentation](https://www.rdkit.org/docs/GettingStartedInPython.html#list-of-available-descriptors) for complete list.
