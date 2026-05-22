from rdkit import Chem
from rdkit.Chem import Descriptors
import numpy as np


def calc_mol_frag_descriptors(
    mol=None,
    smiles=None,
    verbose=True,
    dtype=np.float32,
    descriptor_calc_fns_list=Descriptors._descList,
    descriptor_keys=[
        'MaxEStateIndex', 'MinEStateIndex', 'MolWt',
        'NumValenceElectrons', 'FpDensityMorgan1',
        'AvgIpc', 'BalabanJ',
        'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
        'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
        'Kappa1', 'Kappa2', 'Kappa3', 'TPSA', 'EState_VSA7', 'EState_VSA8',
        'VSA_EState4', 'VSA_EState7', 'VSA_EState8', 'FractionCSP3', 'HeavyAtomCount',
        'NHOHCount', 'NOCount', 'NumAmideBonds', 'NumAtomStereoCenters', 'NumHAcceptors', 'NumHDonors',
        'NumHeteroatoms', 'NumHeterocycles', 'NumRotatableBonds', 'NumUnspecifiedAtomStereoCenters',
        'Phi', 'RingCount', 'MolLogP', 'MolMR'
    ],
    smiles_input=False
):
    """
    Calculate RDKit molecular descriptors for a fragment.

    Computes a selected subset of RDKit molecular descriptors for a molecular
    fragment. Default selection includes 48 descriptors covering molecular weight,
    topological indices, electronic properties, and structural features optimized
    for fragment characterization.

    Args:
        mol: RDKit Mol object to calculate descriptors for. If None, smiles
            parameter is used to create Mol object.
        smiles: SMILES string representing the molecule/fragment. Only used if
            mol is None.
        verbose: If True, print error messages when descriptor calculation fails.
            Default is True.
        dtype: NumPy data type for output array. Default is np.float32 for
            memory efficiency.
        descriptor_calc_fns_list: List of (name, function) tuples for descriptor
            calculation. Default is RDKit's full descriptor list (Descriptors._descList).
        descriptor_keys: List of descriptor names to calculate. Only descriptors
            in this list are computed. Default is a curated set of 48 descriptors
            relevant for fragment characterization.
        smiles_input: If True, treat smiles parameter as direct mol input.
            Internal parameter for optimization. Default is False.

    Returns:
        NumPy array of descriptor values with shape (n_descriptors,) and specified
        dtype. Descriptors that fail calculation are set to 0.

    Example:
        Calculate descriptors from SMILES:

        >>> descriptors = calc_mol_frag_descriptors(smiles='CCO')
        >>> descriptors.shape
        (48,)  # 48 descriptor values
        >>> descriptors.dtype
        dtype('float32')

        Calculate from RDKit Mol:

        >>> from rdkit import Chem
        >>> mol = Chem.MolFromSmiles('CC(*)O')
        >>> descriptors = calc_mol_frag_descriptors(mol=mol)

        Custom descriptor subset:

        >>> custom_keys = ['MolWt', 'NumHDonors', 'NumHAcceptors']
        >>> descriptors = calc_mol_frag_descriptors(
        ...     smiles='CCO',
        ...     descriptor_keys=custom_keys
        ... )
        >>> descriptors.shape
        (3,)

    Note:
        - Failed descriptor calculations are replaced with 0 to maintain array shape
        - Descriptor order matches descriptor_keys order
        - Default descriptor set optimized for fragment representations in GSGE
        - For full list of available descriptors, see RDKit.Chem.Descriptors._descList
    """

    if mol is None and smiles_input==False:
        mol = Chem.MolFromSmiles(smiles)  
    else:
        mol = smiles

    # Calculate descriptors
    results = {}
    x=[]
    for name, func in descriptor_calc_fns_list:

        if name in descriptor_keys:
            try:
                results[name] = func(mol)
            except Exception as e:
                if verbose: print('error for ', name, func)
                results[name] = 0  

    # Print top few
    for k in list(results.keys()):
        #if verbose: print(f"{k}: {results[k]}")
        x.append(results[k])
    return np.array(x, dtype=dtype)

def get_mol_frag_descriptors(gsge, calc_mol_frag_descriptors_args={}, smiles_input=False):
    """
    Calculate descriptors for all fragments in GSGE vocabulary.

    Computes molecular descriptors for every fragment in the GS_vocab and
    returns them as a 2D array suitable for machine learning models.

    Args:
        gsge: GSGE instance or object with GS_vocab.vocab_fragments attribute.
        calc_mol_frag_descriptors_args: Dictionary of arguments passed to
            calc_mol_frag_descriptors() for each fragment. Can include
            'descriptor_keys', 'verbose', 'dtype', etc.
        smiles_input: If True, use SMILES strings instead of Mol objects for
            descriptor calculation. Default is False (use Mol objects).

    Returns:
        NumPy array of shape (n_fragments, n_descriptors) where each row
        contains descriptor values for one fragment.

    Example:
        Calculate descriptors for all vocabulary fragments:

        >>> from GSGE import GSGE
        >>> gsge = GSGE(GS_vocab='vocab.pkl')
        >>> descriptors = get_mol_frag_descriptors(gsge)
        >>> descriptors.shape
        (200, 48)  # 200 fragments, 48 descriptors each

        Custom descriptor subset:

        >>> custom_args = {
        ...     'descriptor_keys': ['MolWt', 'NumHDonors', 'TPSA'],
        ...     'verbose': False
        ... }
        >>> descriptors = get_mol_frag_descriptors(gsge, custom_args)
        >>> descriptors.shape
        (200, 3)

    Note:
        - Returns stacked array with one row per fragment
        - Fragment order matches GS_vocab.vocab_fragments.keys() order
        - Descriptors are not normalized (use normalize_descriptors() if needed)
    """
    return np.stack([calc_mol_frag_descriptors(
        mol=gsge.GS_vocab.vocab_fragments[frag_key].mol if smiles_input is False else None,
        smiles=gsge.GS_vocab.vocab_fragments[frag_key].canonsmiles if smiles_input is True else None,
        **calc_mol_frag_descriptors_args, smiles_input=smiles_input) for frag_key in gsge.GS_vocab.vocab_fragments.keys()])


# === Normalization and Filtering ===

def normalize_descriptors(descriptors, min_var=1e-6):
    """
    Normalize molecular descriptors using z-score and filter low-variance features.

    Applies z-score normalization (mean=0, std=1) to descriptors and removes
    features with variance below threshold to eliminate uninformative descriptors.

    Args:
        descriptors: NumPy array of shape (n_samples, n_descriptors) containing
            raw descriptor values.
        min_var: Minimum variance threshold for keeping descriptors. Descriptors
            with variance below this value are filtered out. Default is 1e-6.

    Returns:
        Tuple of (normalized_descriptors, means, stds, useful_mask) where:
            - normalized_descriptors (np.ndarray): Z-score normalized array of
              shape (n_samples, n_useful_descriptors).
            - means (np.ndarray): Mean values of useful descriptors before
              normalization. Shape: (n_useful_descriptors,).
            - stds (np.ndarray): Standard deviations of useful descriptors before
              normalization. Shape: (n_useful_descriptors,).
            - useful_mask (np.ndarray): Boolean mask indicating which original
              descriptors were kept. Shape: (n_descriptors,).

    Example:
        Normalize fragment descriptors:

        >>> raw_descriptors = get_mol_frag_descriptors(gsge)
        >>> normalized, means, stds, mask = normalize_descriptors(raw_descriptors)
        >>> normalized.shape
        (200, 45)  # 3 low-variance descriptors filtered out
        >>> mask.sum()
        45  # 45 descriptors kept

        Denormalize for interpretation:

        >>> denormalized = normalized * stds + means
        >>> np.allclose(denormalized, raw_descriptors[:, mask])
        True

    Note:
        - Z-score formula: (x - mean) / std
        - std calculation includes +1e-8 offset to prevent division by zero
        - Low-variance descriptors are often constants or near-constants that
          don't contribute to model discrimination
        - Returns both statistics (means, stds) and mask for potential denormalization
    """
    means = descriptors.mean(axis=0)
    stds = descriptors.std(axis=0) + 1e-8  # prevent divide by zero

    variances = descriptors.var(axis=0)
    useful_mask = variances > min_var

    descriptors_filtered = descriptors[:, useful_mask]
    means_filtered = means[useful_mask]
    stds_filtered = stds[useful_mask]

    normalized = (descriptors_filtered - means_filtered) / stds_filtered

    return normalized, means_filtered, stds_filtered, useful_mask

# === Run Everything ===


# if __name__  == '__main__':
#     from GSGE import GSGE, get_use_examples_dir
#     examples_dir = get_use_examples_dir()
#     if examples_dir is None:
#         raise RuntimeError("Cannot find use_examples directory.")
#     GSGE_load_path = examples_dir / 'GAE' / 'test_gsge_save.pkl'
#     gsge = GSGE(GSGE_load_path=str(GSGE_load_path)) 

#     raw_descriptors = get_mol_frag_descriptors(gsge)
#     normalized_descriptors, means, stds, useful_mask = normalize_descriptors(raw_descriptors)