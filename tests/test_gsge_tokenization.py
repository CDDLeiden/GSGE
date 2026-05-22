

#standardized peptides
peptide_smiles = ['CC(C)CC1NC(=O)CCN(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(Cc2ccccc2)N(C)C(=O)CNC(=O)C(C(C)O)NC(=O)C(Cc2ccccc2)NC(=O)C(CC(C)C)N(C)C(=O)C(CC(C)C)NC(=O)C(Cc2ccccc2)NC1=O',
 'CCC(C)C1NC(=O)C(CC(C)C)N(C)C(=O)CCN(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)NC(=O)CN(C)C(=O)C(C(C)O)NC(=O)C(C)N(C)C(=O)C(C(C)C)N(C)C1=O',
 'CCC(C)C1NC(=O)C(Cc2ccccc2)NC(=O)C(C)N(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(CC(C)C)NC(=O)C(Cc2ccccc2)N(C)C(=O)C(Cc2ccccc2)NC(=O)C(CC(C)C)NC(=O)C(C(C)O)NC(=O)C(CC(C)C)NC1=O',
 'CC(C)CC1NC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)CNC(=O)C(C(C)O)NC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)NC(=O)C(C)N(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(COC(C)(C)C)NC1=O',
 'CCC(C)C1NC(=O)C(CC(C)C)N(C)C(=O)C(C)NC(=O)CCCN(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(C(C)O)NC(=O)C(Cc2ccccc2)NC(=O)C(C(C)C)N(C)C(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C1=O',
 'CC(C)CC1C(=O)N(C)C(Cc2ccccc2)C(=O)NC(C(C)O)C(=O)N(C)C(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(CC(C)C)C(=O)NC(C(=O)N2CCCCC2)CC(=O)N(C)CCCC(=O)NC(COC(C)(C)C)C(=O)N(C)CC(=O)N1C',
 'CC(C)CC1C(=O)N(C)C(CC(C)C)C(=O)NC(C(=O)N(C)C(Cc2ccccc2)C(=O)NC(C)C(=O)N2CCCCC2)CC(=O)NC(C(C)C)C(=O)N(C)C(C)C(=O)N(C)C(Cc2ccccc2)C(=O)N(C)C(CC(C)C)C(=O)NC(C(C)O)C(=O)N(C)CC(=O)NC(C(C)C)C(=O)N1C',
 'CCC(C)C1NC(=O)C(C)N(C)C(=O)CC(C(=O)N(C)C(Cc2ccccc2)C(=O)NC(C)C(=O)N2CCCCC2)NC(=O)C(Cc2ccccc2)N(C)C(=O)C(C)NC(=O)C(CC(C)C)N(C)C(=O)C(C)N(C)C(=O)C(C(C)O)NC(=O)C(C(C)C)N(C)C(=O)C(C)N(C)C1=O',
 'CC(C)CC1NC(=O)C(Cc2ccccc2)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)NC(=O)C(C)N(C)C(=O)C(C(C)O)NC(=O)C(C)N(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(CC(C)C)N(C)C1=O',
 'CC(C)CC1NC(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)N(C)C(=O)CCN(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)C(C(C)O)NC(=O)C(Cc2ccccc2)N(C)C1=O',
 'CC(C)CC1NC(=O)C(C)N(C)C(=O)CC(C(=O)N2CCCCC2)NC(=O)C(CC(C)C)N(C)C(=O)C(Cc2ccccc2)N(C)C(=O)C(CC(C)C)NC(=O)C(Cc2ccccc2)N(C)C(=O)C(C)N(C)C(=O)C(C(C)O)NC1=O',
 'CC(C)CC1C(=O)NC(C(C)O)C(=O)N(C)C(C)C(=O)N(C)C(CC(C)C)C(=O)NC(C(=O)N2CCCCC2)CC(=O)N(C)CCCC(=O)NC(C)C(=O)N(C)C(Cc2ccccc2)C(=O)NC(Cc2ccccc2)C(=O)N1C',
 'CC(C)CC1C(=O)N(C)C(Cc2ccccc2)C(=O)NCC(=O)NC(C(=O)N2CCCCC2)CC(=O)N(C)CCCC(=O)NC(C(C)O)C(=O)N(C)C(C)C(=O)N(C)C(Cc2ccccc2)C(=O)NC(COC(C)(C)C)C(=O)N1C',
 'CC(C)CC1C(=O)NC(C(C)O)C(=O)N(C)C(C)C(=O)N(C)C(CC(C)C)C(=O)N(C)C(Cc2ccccc2)C(=O)NC(COC(C)(C)C)C(=O)NC(C(=O)N2CCCCC2)CC(=O)N(C)CCCC(=O)N(C)C(C)C(=O)N1C']


import unittest
from GSGE import GSGE
from pathlib import Path

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

class TestGSGEpreprocess_from_SMILES(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the standard gsge vocab/data once for all tests
        pkl_path = Path(__file__).parent / 'gsge_save_v5a2.pkl'
        cls.gsge = GSGE(GSGE_load_path=pkl_path)

        cls.expected = [':0','GS_frag_38','C',':0','GS_frag_76','Ring1',':3','GS_frag_107','O',':3',
                      'GS_frag_22','Ring1',':0','GS_frag_69','pop','Ring1',':0','GS_frag_76','Ring1',
                      ':6','GS_frag_107','Branch',':4','GS_frag_108','Ring1',':4','GS_frag_77','Ring1',
                      ':4','GS_frag_108','Branch',':0','GS_frag_47','Ring1',':0','GS_frag_68','pop','pop',
                      'Branch',':0','GS_frag_76','Ring1',':9','GS_frag_93','Ring1',':0','GS_frag_47','pop',
                      'Ring1',':1','GS_frag_38','Ring1',':4','GS_frag_108','Branch',':0','GS_frag_47','Ring1',
                      ':0','GS_frag_68','pop','pop','Branch','Ring2','#Branch','C','pop','pop','pop','=Branch',
                      ':0','GS_frag_47','pop','pop','pop','pop','pop','pop','Ring1',':0','GS_frag_68','pop','pop',
                      'pop','pop','pop','pop']

        cls.expected_tokens = [ 
            13,  85,  29,  13, 123,   7,  16, 154,  31,  16,  69,   7,  13,
            116,   6,   7,  13, 123,   7,  19, 154,  10,  17, 155,   7,  17,
            124,   7,  17, 155,  10,  13,  94,   7,  13, 115,   6,   6,  10,
            13, 123,   7,  22, 140,   7,  13,  94,   6,   7,  14,  85,   7,
            17, 155,  10,  13,  94,   7,  13, 115,   6,   6,  10,   8,  12,
            29,   6,   6,   6,  11,  13,  94,   6,   6,   6,   6,   6,   6,
            7,  13, 115,   6,   6,   6,   6,   6,   6,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0]

        global peptide_smiles

    def test_preprocess_from_SMILES_output(self):
        print('GSGE string test: preprocess_from_SMILES')
        data_ = [self.gsge.preprocess_from_SMILES(peptide_smi)
                 for peptide_smi in peptide_smiles]
        
        assert data_[0] == self.expected, 'list of string tokens do not match'

    def test_parallel_tokenize_SMILES_list_output(self):
        print('test: parallel_tokenize_SMILES_list')
        padded_results, smiles_list = self.gsge.parallel_tokenize_SMILES_list(peptide_smiles)
        
        assert padded_results[0].tolist() == self.expected_tokens, 'list of string tokens do not match'
    
if __name__ == '__main__':
    unittest.main()