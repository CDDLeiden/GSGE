# Visualization Utilities

Molecular and fragment visualization functions.

## highlight_fragments

Highlight molecular fragments in molecule visualization.

::: GSGE.plots.highlight_fragments
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## generate_colors

Generate reproducible colors with seed.

::: GSGE.plots.generate_colors
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## sort_mols_atom_num

Sort molecules by number of atoms.

::: GSGE.plots.sort_mols_atom_num
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## clean_fragment

Clean and standardize fragment SMILES.

::: GSGE.plots.clean_fragment
    options:
      show_root_heading: true
      show_source: false
      heading_level: 2

## Usage Examples

### Highlight Fragments in Molecule

```python
from GSGE import GSGE
from GSGE.plots import highlight_fragments

gsge = GSGE(GS_vocab=vocab)
smiles = 'c1ccc(cc1)O'  # Phenol

# Highlight fragments
img = gsge.plot_GS_fragments_in_mol(
    smiles,
    args={
        'color_method': 'method1',
        'color_seed': 42,
        'annotate_with_index': True,
        'annotate_atoms': True
    }
)
```

### Generate Consistent Colors

```python
from GSGE.plots import generate_colors

# Generate colors with seed for reproducibility
colors = generate_colors(
    n=10,
    seed=42,
    method='method1'
)

# Each color is RGBA tuple
print(colors[0])  # (0.8, 0.2, 0.3, 0.6)
```

### Visualize Vocabulary Fragments

```python
from GSGE import GS_Vocab

vocab = GS_Vocab()
vocab.build_vocab(m_set=smiles, convert=True, target=100)

# Plot first 20 fragments
vocab.plot_vocab(
    n_frags=20,
    save_path='vocabulary_fragments.png'
)
```

### Sort Molecules by Size

```python
from GSGE.plots import sort_mols_atom_num
from rdkit import Chem

mols = [Chem.MolFromSmiles(s) for s in smiles_list]
sorted_mols = sort_mols_atom_num(mols)

# Smallest to largest
for mol in sorted_mols:
    print(f"{Chem.MolToSmiles(mol)}: {mol.GetNumAtoms()} atoms")
```

### Clean Fragment SMILES

```python
from GSGE.plots import clean_fragment

# Clean up fragment SMILES
raw_fragment = 'C(*)C(*)'
cleaned = clean_fragment(raw_fragment)

print(f"Raw: {raw_fragment}")
print(f"Cleaned: {cleaned}")
```

## Color Methods

### method1 (Default)

Generates vibrant colors with good contrast:

```python
colors = generate_colors(n=5, method='method1', seed=42)
```

### Custom Color Schemes

You can create custom color generation logic:

```python
import numpy as np

def custom_colors(n, seed=None):
    if seed:
        np.random.seed(seed)
    return [(np.random.rand(), np.random.rand(), np.random.rand(), 0.6)
            for _ in range(n)]
```

## Visualization Best Practices

### High-Quality Molecule Images

```python
from rdkit import Chem
from rdkit.Chem import Draw

mol = Chem.MolFromSmiles(smiles)

# High resolution
img = Draw.MolToImage(
    mol,
    size=(800, 800),
    kekulize=True,
    wedgeBonds=True
)
```

### Fragment Highlighting

For clear fragment visualization:

1. Use consistent color seeds across visualizations
2. Annotate atoms for complex molecules
3. Use high DPI for publication-quality images
4. Consider dark/light backgrounds for different contexts

### Batch Visualization

For visualizing many molecules:

```python
from rdkit.Chem import Draw

# Grid of molecules
mols = [Chem.MolFromSmiles(s) for s in smiles_list[:20]]
img = Draw.MolsToGridImage(
    mols,
    molsPerRow=5,
    subImgSize=(200, 200),
    legends=[f"Mol {i}" for i in range(len(mols))]
)
```
