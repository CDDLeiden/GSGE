
"Custom version of torch geometric"

from typing import Any, Dict, List
import torch
import torch_geometric

x_map: Dict[str, List[Any]] = {
    'atomic_num':
    list(range(0, 119)),
    'chirality': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER',
        'CHI_TETRAHEDRAL',
        'CHI_ALLENE',
        'CHI_SQUAREPLANAR',
        'CHI_TRIGONALBIPYRAMIDAL',
        'CHI_OCTAHEDRAL',
    ],
    'degree':
    list(range(0, 11)),
    'formal_charge':
    list(range(-5, 7)),
    'num_hs':
    list(range(0, 9)),
    'num_radical_electrons':
    list(range(0, 5)),
    'hybridization': [
        'UNSPECIFIED',
        'S',
        'SP',
        'SP2',
        'SP3',
        'SP3D',
        'SP3D2',
        'OTHER',
    ],
    'is_aromatic': [False, True],
    'is_in_ring': [False, True],
}

e_map: Dict[str, List[Any]] = {
    'bond_type': [
        'UNSPECIFIED',
        'SINGLE',
        'DOUBLE',
        'TRIPLE',
        # 'QUADRUPLE',
        # 'QUINTUPLE',
        # 'HEXTUPLE',
        # 'ONEANDAHALF',
        # 'TWOANDAHALF',
        # 'THREEANDAHALF',
        # 'FOURANDAHALF',
        # 'FIVEANDAHALF',
        'AROMATIC',
        # 'IONIC',
        # 'HYDROGEN',
        # 'THREECENTER',
        # 'DATIVEONE',
        # 'DATIVE',
        # 'DATIVEL',
        # 'DATIVER',
        'OTHER',
        # 'ZERO',
    ],
    'stereo': [
        'STEREONONE',
        'STEREOANY',
        'STEREOZ',
        'STEREOE',
        'STEREOCIS',
        'STEREOTRANS',
    ],
    'is_conjugated': [False, True],
}


def from_smiles(smiles: str, with_hydrogen: bool = False,
                kekulize: bool = False, atom_to_token_id = None, allow_unk=False) -> 'torch_geometric.data.Data':
    r"""Converts a SMILES string to a :class:`torch_geometric.data.Data`
    instance.

    Args:
        smiles (str): The SMILES string.
        with_hydrogen (bool, optional): If set to :obj:`True`, will store
            hydrogens in the molecule graph. (default: :obj:`False`)
        kekulize (bool, optional): If set to :obj:`True`, converts aromatic
            bonds to single/double bonds. (default: :obj:`False`)
    """
    from rdkit import Chem, RDLogger

    from torch_geometric.data import Data

    RDLogger.DisableLog('rdApp.*')

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        mol = Chem.MolFromSmiles('')
    if with_hydrogen:
        mol = Chem.AddHs(mol)
    if kekulize:
        Chem.Kekulize(mol)

    xs: List[List[int]] = []
    for atom in mol.GetAtoms():
        row: List[int] = []

        if atom_to_token_id is not None: #atom num is linked to token id
            ATI = atom_to_token_id.get(x_map['atomic_num'].index(atom.GetAtomicNum()), atom_to_token_id[-1])
            if ATI == atom_to_token_id[-1]:
                print(atom.GetAtomicNum(), 'this atom/node type is not in the vocab, set as UNK instead:', atom_to_token_id[-1], 'for:', smiles, )
                if allow_unk is False:
                    raise ValueError('UNK not allowed, set allow_unk to True for allowing it.')
            row.append(atom_to_token_id.get(x_map['atomic_num'].index(atom.GetAtomicNum()), -1))
        else:
            row.append(x_map['atomic_num'].index(atom.GetAtomicNum()))
        
        row.append(x_map['chirality'].index(str(atom.GetChiralTag())))
        row.append(x_map['degree'].index(atom.GetTotalDegree()))
        row.append(x_map['formal_charge'].index(atom.GetFormalCharge()))
        row.append(x_map['num_hs'].index(atom.GetTotalNumHs()))
        row.append(x_map['num_radical_electrons'].index(
            atom.GetNumRadicalElectrons()))
        row.append(x_map['hybridization'].index(str(atom.GetHybridization())))
        row.append(x_map['is_aromatic'].index(atom.GetIsAromatic()))
        row.append(x_map['is_in_ring'].index(atom.IsInRing()))
        xs.append(row)

    x = torch.tensor(xs, dtype=torch.long).view(-1, 9)

    edge_indices, edge_attrs = [], []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()

        e = []

        bond_type_str = str(bond.GetBondType())
        try:
            bond_type_idx = e_map['bond_type'].index(bond_type_str)
        except ValueError:
            print('bond_type', bond_type_str, 'not included in bond_types, set as class value', len(e_map['bond_type']), 'for UNK in stead for', smiles)
            bond_type_idx = len(e_map['bond_type'])  # Default to 5 if not found

        e.append(bond_type_idx)
        e.append(e_map['stereo'].index(str(bond.GetStereo())))
        e.append(e_map['is_conjugated'].index(bond.GetIsConjugated()))

        edge_indices += [[i, j], [j, i]]
        edge_attrs += [e, e]

    edge_index = torch.tensor(edge_indices)
    edge_index = edge_index.t().to(torch.long).view(2, -1)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.long).view(-1, 3)

    if edge_index.numel() > 0:  # Sort indices.
        perm = (edge_index[0] * x.size(0) + edge_index[1]).argsort()
        edge_index, edge_attr = edge_index[:, perm], edge_attr[perm]

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)


# Mapping of atomic numbers to token IDs with chemical symbols for clarity
atom_to_token_id: Dict[int, int] = {
    0:  0,  # *   (Wildcard/Placeholder)
    1:  1,  # H   (Hydrogen)
    5:  2,  # B   (Boron)
    6:  3,  # C   (Carbon)
    7:  4,  # N   (Nitrogen)
    8:  5,  # O   (Oxygen)
    9:  6,  # F   (Fluorine)
    14: 7,  # Si  (Silicon)
    15: 8,  # P   (Phosphorus)
    16: 9,  # S   (Sulfur)
    17: 10, # Cl  (Chlorine)
    32: 11, # Ge  (Germanium)
    33: 12, # As  (Arsenic)
    34: 13, # Se  (Selenium)
    35: 14, # Br  (Bromine)
    40: 15, # Zr  (Zirconium)
    50: 16, # Sn  (Tin)
    52: 17, # Te  (Tellurium)
    53: 18, # I   (Iodine)
    80: 19, # Hg  (Mercury)
    82: 20, # Pb  (Lead)
    -1: 21  # UNK (UNKNOWN)
}

if __name__ == '__main__':

    import joblib
    #from torch_geometric.utils import from_smiles, to_smiles

    from GSGE import GSGE, get_tests_dir

    tests_dir = get_tests_dir()
    if tests_dir is None:
        raise RuntimeError("Cannot find tests directory. Run from source checkout.")
    pkl_path = tests_dir / 'gsge_save_v5a2.pkl'
    gsge = GSGE(GSGE_load_path=pkl_path)

    #Get data
    fragments = gsge.vocab_manager.GS_vocab.vocab_fragments

    list_smiles_frags = []
    for i, g in enumerate(fragments.values()):
        list_smiles_frags.append(g.canonsmiles.replace('*1', '*'))

    smiles_list = list_smiles_frags

    # Convert SMILES to Data objects and ensure correct dtypes
    data_list = [from_smiles(smiles, atom_to_token_id=atom_to_token_id) for smiles in smiles_list]

    print(data_list[0])

    print(data_list[0]['x'][:,0])
    print(data_list[0]['smiles'])

