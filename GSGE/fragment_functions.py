
""" adaptations from official group_selfies fragment_functions"""

from group_selfies.utils.group_utils import hash_molecule, bond_to_order, get_core
from rdkit import Chem


aliases = {'default': ['default']}
alias_translation = dict([(alias, k) for k, v in aliases.items() for alias in v])

from rdkit import Chem

def CUSTOM_fragment_mol(m, MIN_SIZE:int=4, MAX_SIZE:int=15, method:str='default'):
    """" Used for making GSGE_Corpus or GS_Vocab, and does the inital bond breaking for making molecular fragments"""

    # Fragment a molecule based on various strategies
    if alias_translation[method.lower()] == 'default':
        m_clone = Chem.Mol(m)
        Chem.Kekulize(m_clone, clearAromaticFlags=True)

        # **Step 1: Cut outgoing ring bonds**
        ring_bond_smarts = Chem.MolFromSmarts('[Rx2;D3,D4][$([Rx2;D3,D4]),$([!R])]')
        ring_bonds = [m.GetBondBetweenAtoms(a1, a2).GetIdx() for a1, a2 in m_clone.GetSubstructMatches(ring_bond_smarts)]

        if ring_bonds:
            m_clone = Chem.FragmentOnBonds(m_clone, ring_bonds, addDummies=True)

        # **Step 2: Cut peptide bonds (amide bonds)**
        # Sanitize after ring fragmentation to avoid valence errors
        if ring_bonds:
            Chem.SanitizeMol(m_clone)

        peptide_bond_smarts = Chem.MolFromSmarts("[C,c:1](=O)[N:2][C,c:3]")
        peptide_bonds = [m_clone.GetBondBetweenAtoms(match[0], match[2]).GetIdx() for match in m_clone.GetSubstructMatches(peptide_bond_smarts)]

        if peptide_bonds:
            m_clone = Chem.FragmentOnBonds(m_clone, peptide_bonds, addDummies=True)

        # **Step 3: Cut disulfide bonds**
        # Sanitize after peptide fragmentation to avoid valence errors
        if peptide_bonds:
            Chem.SanitizeMol(m_clone)

        disulfide_bond_smarts = Chem.MolFromSmarts("[C][S,v2]~[S,v2][C]")#"[SX2]-[SX2]")
        disulfide_bonds = [m_clone.GetBondBetweenAtoms(match[0], match[1]).GetIdx() for match in m_clone.GetSubstructMatches(disulfide_bond_smarts)]

        if disulfide_bonds:
            m_clone = Chem.FragmentOnBonds(m_clone, disulfide_bonds, addDummies=True)

        # **Get the final fragments**
        fragments = Chem.GetMolFrags(m_clone, asMols=True, sanitizeFrags=True)
    else:
        raise ValueError('Not a valid method:', method)

    new_fragments = []
    unique_frag_hashes = set()
    for fragment in fragments:
        if fragment is None:
            continue
        atom_count = sum(1 for atom in fragment.GetAtoms() if atom.GetAtomicNum())
        
        if atom_count < MIN_SIZE or atom_count > MAX_SIZE:
            continue
        
        new_f = fragment
        for idx, atom in enumerate(new_f.GetAtoms()):
            if not atom.GetAtomicNum():
                parent = atom.GetNeighbors()[0].GetIdx()
                valence = bond_to_order[fragment.GetBondBetweenAtoms(parent, idx).GetBondType()]
                atom.SetIntProp('valAvailable', valence)
                atom.SetIsotope(0)

        hm = hash_molecule(new_f)  # Hash fragment to avoid repeats
        if hm not in unique_frag_hashes:
            try:
                Chem.Kekulize(new_f)
                new_f_core = get_core(new_f)  # Extract core from fragment
                new_fragments.append((new_f, new_f_core))
                unique_frag_hashes.add(hm)
            except Exception as e:
                print('ISSUE', e)
                pass

    return new_fragments



