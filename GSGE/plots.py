
from rdkit import Chem
import random
import itertools
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io
import group_selfies


def clean_fragment(fragment: str):
    """
    Remove dummy atoms ([*]) from a molecular fragment.

    Dummy atoms (represented as [*] in SMILES) are placeholder atoms used to
    indicate connection points in molecular fragments. This function removes
    all dummy atoms to obtain the clean fragment structure.

    Args:
        fragment: SMILES string or RDKit Mol object representing a molecular
            fragment, potentially containing dummy atoms ([*]).

    Returns:
        RDKit Mol object with all dummy atoms removed.

    Example:
        >>> from rdkit import Chem
        >>> frag_smiles = 'CC(*)O'  # Fragment with dummy atom
        >>> clean_mol = clean_fragment(frag_smiles)
        >>> Chem.MolToSmiles(clean_mol)
        'CCO'  # Dummy atom removed

    Note:
        Atoms are removed in reverse order by index to avoid index shifting
        issues during removal.
    """
    fragment = Chem.RWMol(fragment)
    dummy_atoms = [atom.GetIdx() for atom in fragment.GetAtoms() if atom.GetAtomicNum() == 0]
    for idx in sorted(dummy_atoms, reverse=True):
        fragment.RemoveAtom(idx)
    return Chem.Mol(fragment)

def generate_colors(num_colors: int, num_shades: int = 5, seed: int = 42):
    """
    Generate visually distinct RGBA colors for fragment highlighting.

    Creates a set of perceptually distinct colors by maximizing color space distance.
    Avoids dark colors (uses range 0.3-0.9) and adds transparency (alpha 0.5-0.8)
    for better visualization when highlighting molecular fragments.

    Args:
        num_colors: Number of distinct colors to generate.
        num_shades: Number of shades per RGB channel to sample. Higher values
            provide more color options but increase computation. Default is 5.
        seed: Random seed for reproducible color generation. Default is 42.

    Returns:
        List of RGBA color tuples where each tuple is (R, G, B, A) with
        values in range [0.0, 1.0]. Length equals num_colors.

    Example:
        Generate colors for 10 fragments:

        >>> colors = generate_colors(num_colors=10, seed=123)
        >>> len(colors)
        10
        >>> colors[0]
        (0.7, 0.5, 0.3, 0.65)  # Example RGBA values

        Use with RDKit highlighting:

        >>> colors = generate_colors(num_colors=5)
        >>> # Use in highlight_fragments() or RDKit drawing

    Note:
        Colors are selected to maximize perceptual distance using Euclidean
        distance in RGB space. If more colors are requested than can be
        generated from the color pool, random bright colors are added.
        Alpha transparency ranges from 0.5 to 0.8 for visual clarity.
    """
    random.seed(seed)

    # Generate a pool of colors using itertools.product with a brighter range (0.3 to 0.9)
    color_pool = list(itertools.product(
        np.linspace(0.3, 0.9, num_shades),  # R
        np.linspace(0.3, 0.9, num_shades),  # G
        np.linspace(0.3, 0.9, num_shades)   # B
    ))
    random.shuffle(color_pool)

    # Define color distance function (for RGB only, ignoring alpha for now)
    def color_distance(c1, c2):
        return sum((a - b) ** 2 for a, b in zip(c1[:3], c2[:3])) ** 0.5

    # Select distinct colors
    if not color_pool:
        return []

    sorted_colors = [color_pool.pop(0)]  # Start with first color
    while color_pool and len(sorted_colors) < num_colors:
        # Pick the color furthest from all already selected colors
        next_color = max(color_pool, key=lambda c: min(color_distance(c, x) for x in sorted_colors))
        sorted_colors.append(next_color)
        color_pool.remove(next_color)

    # If we need more colors than generated, cycle through with some variation
    while len(sorted_colors) < num_colors:
        sorted_colors.append((random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)))

    # Add transparency (alpha channel) to all colors, range 0.5 to 0.8
    sorted_colors_with_alpha = [(r, g, b, random.uniform(0.5, 0.8)) for r, g, b in sorted_colors]

    return sorted_colors_with_alpha

def sort_mols_atom_num(molecule_list: list, reverse: bool = True):
    """
    Sort molecules by number of atoms.

    Args:
        molecule_list: List of RDKit Mol objects to sort.
        reverse: If True, sort from largest to smallest (descending).
            If False, sort from smallest to largest (ascending). Default is True.

    Returns:
        Sorted list of RDKit Mol objects ordered by atom count.

    Example:
        >>> from rdkit import Chem
        >>> mols = [Chem.MolFromSmiles(s) for s in ['C', 'CCO', 'CC']]
        >>> sorted_mols = sort_mols_atom_num(mols, reverse=True)
        >>> [mol.GetNumAtoms() for mol in sorted_mols]
        [3, 2, 1]  # CCO, CC, C (largest to smallest)
    """
    # Sorting the list by the number of atoms (large to small by default)
    return sorted(molecule_list, key=lambda mol: mol.GetNumAtoms(), reverse=reverse)

def highlight_fragments(
    full_mol,
    vocab: dict,  # GS_Vocab or GSGE_corpus
    img_size: tuple = (1800, 1350),
    annotate_atoms: bool = True,
    annotate_with_index: bool = False,
    same_color_for_same_fragment: bool = True,
    verbose: bool = False,
    color_seed: int = 42,
    color_method: str = 'standard',  # method1
    return_color_info: bool = False,
    fragment_colors: dict = None,
    transparent_background: bool = False
):
    """
    Visualize molecular fragments with color-coded highlighting in molecule.

    Extracts and highlights molecular fragments from a molecule using the provided
    vocabulary. Each fragment is colored distinctly, with bonds highlighted to show
    fragment boundaries. Supports custom annotations and color schemes.

    Args:
        full_mol: RDKit Mol object of the complete molecule to visualize.
        vocab: GS_Vocab or GSGE_Corpus object containing fragment vocabulary.
        img_size: Tuple of (width, height) in pixels for output image.
            Default is (1800, 1350) for high-quality visualization.
        annotate_atoms: If True, show atom indices on the molecule.
            Default is True.
        annotate_with_index: If True, annotate atoms with fragment indices
            in format "atom_idx(frag_idx|frag_id)". Overrides annotate_atoms.
            Default is False.
        same_color_for_same_fragment: If True, identical fragments across the
            molecule use the same color. If False, each occurrence gets a
            different color. Default is True.
        verbose: If True, print debugging information including Group-SELFIES
            encoding and fragment details. Default is False.
        color_seed: Random seed for reproducible color generation. Default is 42.
        color_method: Color generation method. Options:
            - 'standard': Uses generate_colors() for distinct colors
            - Other: Uses predefined color palette
            Default is 'standard'.

    Returns:
        PIL Image object showing the molecule with highlighted fragments.

    Raises:
        ValueError: If full_mol is not a valid RDKit Mol object.

    Example:
        Basic fragment highlighting:

        >>> from rdkit import Chem
        >>> from GSGE import GSGE
        >>> gsge = GSGE(GS_vocab='vocab.pkl')
        >>> mol = Chem.MolFromSmiles('CCO')
        >>> img = highlight_fragments(mol, gsge.vocab_manager.GS_vocab)
        >>> img.show()  # Display highlighted molecule

        Custom annotations and colors:

        >>> img = highlight_fragments(
        ...     mol,
        ...     vocab=gsge.vocab_manager.GS_vocab,
        ...     img_size=(2400, 1800),
        ...     annotate_with_index=True,
        ...     same_color_for_same_fragment=False,
        ...     color_seed=123
        ... )

    Note:
        - Atoms not included in any fragment are reported as warnings
        - Fragment colors are chosen to maximize visual distinction
        - Bonds are only highlighted if both atoms belong to the same fragment
        - Alpha transparency (0.5-0.8) makes overlapping highlights visible
        - The function modifies atom properties temporarily but clears them
          before returning
        - When return_color_info=True, returns (img, color_map) where color_map
          maps fragment vocab ID (int) to RGBA color tuple
        - When fragment_colors is provided, it should map fragment vocab ID (int)
          to RGBA color tuple, overriding internal color generation
    """

    grammar_fragment = group_selfies.GroupGrammar(vocab=vocab.vocab_fragments)
    extracted_groups=grammar_fragment.extract_groups(full_mol)

    if verbose: 
        print(grammar_fragment.encoder(full_mol, extracted_groups))

    # Set random seed for reproducible colors
    random.seed(color_seed)

    # Validate input
    if not isinstance(full_mol, Chem.rdchem.Mol):
        raise ValueError("full_mol must be an RDKit molecule object")
    
    num_atoms = full_mol.GetNumAtoms()
    if verbose:
        print(f"Molecule has {num_atoms} atoms")
    
    if color_method == 'method1':
        num_fragments = len(extracted_groups)
        colors = generate_colors(num_fragments, num_shades=5, seed=color_seed)
    else:
        # Generate distinct colors
        predefined_colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (0.5, 0.0, 0.5),  # Purple
            (1.0, 0.5, 0.0),  # Orange
        ]

        num_fragments = len(extracted_groups)
        colors = predefined_colors[:min(num_fragments, len(predefined_colors))] + \
                [(random.random(), random.random(), random.random()) 
                for _ in range(max(0, num_fragments - len(predefined_colors)))]


    # Map fragments to colors
    fragment_to_color = {}
    color_index = 0
    for fragment in extracted_groups:
        group_name = fragment[0]
        frag_id = int(group_name.name.split('_')[-1])
        if same_color_for_same_fragment and group_name.canonsmiles in fragment_to_color:
            continue
        if fragment_colors is not None and frag_id in fragment_colors:
            fragment_to_color[fragment[0].canonsmiles] = fragment_colors[frag_id]
        else:
            fragment_to_color[fragment[0].canonsmiles] = colors[color_index % len(colors)]
        color_index += 1
        if verbose:
            print(f"Group {group_name}: frag: {fragment[1]}") # Color {fragment_to_color[fragment[0].canonsmiles]},
    
    # Collect highlighting information
    highlight_atoms = set()
    highlight_bonds = set()
    atom_colors = {}
    bond_colors = {}
    atom_labels = {}  # For custom fragment index labels

    # Track encountered atom indices
    encountered_atoms = set()
    
    # Process each fragment independently
    for frag_idx, fragment in enumerate(extracted_groups):
        group_name, atom_indices, bond_info = fragment
        color = fragment_to_color[fragment[0].canonsmiles]
        
        # Get all atoms in this fragment
        fragment_atoms = set(atom_idx for atom_idx, _ in atom_indices if atom_idx < num_atoms)
        
        # Add these atoms to the encountered set
        encountered_atoms.update(fragment_atoms)
        
        # Add atom indices for this fragment
        for atom_idx in fragment_atoms:
            highlight_atoms.add(atom_idx)
            if atom_idx not in atom_colors:  # Only set color if not already set
                atom_colors[atom_idx] = color
            if annotate_with_index:
                #atom_labels[atom_idx] = f"{frag_idx}({atom_idx}|{list(fragment_to_color.keys()).index(fragment[0].canonsmiles)})" #fragment index, fragment id (same across same fragments), atom_num
                atom_labels[atom_idx] = f"{atom_idx}({frag_idx}|{fragment[0].name.split('_')[-1]})" #atom_index, fragment index, fragment GS_vocab id (same across same fragments)
        
        # Find and highlight all bonds within this fragment
        for atom1 in fragment_atoms:
            atom_obj = full_mol.GetAtomWithIdx(atom1)
            for bond in atom_obj.GetBonds():
                atom2 = bond.GetOtherAtomIdx(atom1)
                if atom2 in fragment_atoms:  # Only highlight if both atoms are in this fragment
                    bond_idx = bond.GetIdx()
                    highlight_bonds.add(bond_idx)
                    if bond_idx not in bond_colors:
                        bond_colors[bond_idx] = color
                    elif verbose and bond_colors[bond_idx] != color:
                        print(f"Note: Bond {bond_idx} between atoms {atom1}-{atom2} already colored differently")
    
    # Check if any atoms were not included in fragments
    all_atoms_in_mol = set(range(num_atoms))
    missing_atoms = all_atoms_in_mol - encountered_atoms

    if missing_atoms:
        missing_info = [
            (atom_idx, full_mol.GetAtomWithIdx(atom_idx).GetSymbol()) for atom_idx in sorted(missing_atoms)
        ]
        print("Warning: The following atoms were not included in any fragment (single element are still present in the group-selfies):")
        for atom_idx, atom_symbol in missing_info:
            print(f" - Atom {atom_idx} ({atom_symbol})")
    
    # Convert sets to lists
    highlight_atoms = list(highlight_atoms)
    highlight_bonds = list(highlight_bonds)
    
    # Prepare drawing options
    drawer = rdMolDraw2D.MolDrawOptions()
    drawer.useBWAtomPalette()  # Black and white base for contrast
    drawer.highlightBondWidth = 2.0  # Thicker highlights for better visibility
    if annotate_atoms and not annotate_with_index:
        drawer.addAtomIndices = True  # Show default atom indices
    elif annotate_with_index:
        for atom_idx, label in atom_labels.items():
            full_mol.GetAtomWithIdx(atom_idx).SetProp("atomNote", label)
    
    # Generate image using MolDraw2DCairo
    if transparent_background:
        drawer.setBackgroundColour((1, 1, 1, 0))
    d2d = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
    d2d.SetDrawOptions(drawer)
    d2d.DrawMolecule(
        full_mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors=atom_colors,
        highlightBondColors=bond_colors
    )
    d2d.FinishDrawing()
    
    # Get the PNG data and convert to PIL Image
    png_data = d2d.GetDrawingText()
    img = Image.open(io.BytesIO(png_data))
    
    # Clear atom notes to avoid persisting changes
    if annotate_with_index:
        for atom_idx in atom_labels:
            full_mol.GetAtomWithIdx(atom_idx).ClearProp("atomNote")
    
    if return_color_info:
        # Build mapping from fragment vocab ID to color
        frag_id_to_color = {}
        for fragment in extracted_groups:
            frag_id = int(fragment[0].name.split('_')[-1])
            frag_id_to_color[frag_id] = fragment_to_color[fragment[0].canonsmiles]
        return img, frag_id_to_color
    return img
