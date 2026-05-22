
from rdkit import Chem
from group_selfies.utils.group_utils import HashableMolecule
from group_selfies import Group
import re

class FragmentTools:

    @staticmethod  
    def get_hash_by_smiles(smiles:str):
        """Retrieve the hash corresponding to a given SMILES."""
        return HashableMolecule(FragmentTools.get_canonical_mol_from_smiles(smiles))
    
    @staticmethod    
    def clean_wildcards(smiles:str):
        return re.sub(r'\[\*:?\d*\]', '*', smiles).replace('*1', '*')

    @staticmethod
    def get_canonical_smiles_from_mol(mol):
        """Makes sure atom order in graph is canonical"""
        smiles = FragmentTools.clean_wildcards(Chem.MolToSmiles(mol, canonical=True))
        return Chem.MolToSmiles(Chem.MolFromSmiles(smiles), canonical=True)

    @staticmethod
    def canonicalize_smiles(smiles:str):
        mol = FragmentTools.get_canonical_mol_from_smiles(smiles)
        return Chem.MolToSmiles(mol, canonical=True)
    
    @staticmethod
    def canonicalize_mol(mol):
        """Makes sure atom order in graph is canonical"""
        return Chem.MolFromSmiles(FragmentTools.get_canonical_smiles_from_mol(mol))
    
    @staticmethod
    def get_canonical_mol_from_smiles(smiles:str):
        """Makes sure atom order in graph is canonical"""
        return Chem.MolFromSmiles(FragmentTools.clean_wildcards(smiles))

class GS_FragmentTools(FragmentTools):

    @staticmethod
    def make_element_GS(element:str, num_bonds:int=1, fragment_name:str=None) -> Group : 
        """ 
        Create a molecular fragment representation for an element with a specified number of bonds.

        Args:
            element (str): The atomic symbol (e.g., 'C', 'Cl', 'N').
            num_bonds (int): The number of bonds the element should have (default is 1).
            fragment_name (str, optional): Custom name for the fragment.

        Returns:
            tuple: (<Group object>, hash, canonical SMILES)

        # Example usage:
        make_element_GS('C', 1)
        # Output: (<Group frag_C_*1 C(*1)>, -2475059342658138438, '*C')

        make_element_GS('C', 4) 
        # Output: (<Group frag_C_*4 C(*1)(*1)(*1)(*1)>, 7242397962174154240, '*C(*)(*)*')

        make_element_GS('Cl', 1) 
        # Output: (<Group frag_Cl_*1 Cl(*1)>, some_hash_value, '*Cl')
        """

        # Elements that require brackets in SMILES
        bracketed_elements = ['Si', 'Ge', 'As', 'Se', 'Zr', 'Sn', 'Te', 'Hg', 'Pb']

        if num_bonds == 1:

            if element in bracketed_elements:
                smi_gs_format = f"[{element}]*1" # Ensures correct parsing for Si
            else:
                smi_gs_format = f"{element}*1"

        else:

            if element in bracketed_elements:
                smi_gs_format = f"[{element}]" + "(*1)" * num_bonds  # Ensures correct parsing for Si
            else:
                smi_gs_format = f"{element}" + "(*1)" * num_bonds


        frag_id = fragment_name if fragment_name is not None else f'GS_frag_{element}_*{num_bonds}'
        
        mol = FragmentTools.get_canonical_mol_from_smiles(smi_gs_format)
        cn_smi = FragmentTools.get_canonical_smiles_from_mol(mol)
        hash = FragmentTools.get_hash_by_smiles(cn_smi).hash
        group = Group(frag_id, smi_gs_format)

        return group, hash, cn_smi
