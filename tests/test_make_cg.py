

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
import numpy as np
from GSGE import GSGE
from pathlib import Path

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*fork.*")

class TestGSGEGetCG(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the standard gsge vocab/data once for all tests
        pkl_path = Path(__file__).parent / 'gsge_save_v5a2.pkl'
        cls.gsge = GSGE(GSGE_load_path=pkl_path)

        cls.expected_array = np.array([
            [ 0,  2,  3,  3,  3,  6,  7,  6,  9,  9, 10, 12, 10, 14, 14, 14,
             17, 17, 18, 20,  0,  1,  1,  2,  4,  5,  5,  6,  8,  8, 10, 11,
             11, 13, 13, 15, 16, 14, 18, 19, 19, 18],
            [ 1,  1,  2,  4,  5,  5,  6,  8,  8, 10, 11, 11, 13, 13, 15, 16,
             14, 18, 19, 19, 18,  0,  2,  3,  3,  3,  6,  7,  6,  9,  9, 10,
             12, 10, 14, 14, 14, 17, 17, 18, 20,  0]
        ])
        cls.expected_features = [38, 76, 107, 22, 69, 76, 107, 68, 108, 77, 108, 47,
                             68, 76, 93, 47, 47, 38, 108, 47, 68]

        global peptide_smiles

    def test_get_CG_from_smiles_output(self):
        print('making compound graph test: get_CG_from_smiles')
        data_ = [self.gsge.get_CG_from_smiles(peptide_smi, return_CG_object=False)
                 for peptide_smi in peptide_smiles]
        
        adj_matrix, features = data_[0]

        self.assertTrue(np.array_equal(adj_matrix, self.expected_array), "Adjacency matrix mismatch")
        self.assertEqual(features, self.expected_features, "Feature list mismatch")
    
    def test_make_compound_graphs_output(self):
        print('making compound graph test: make_compound_graphs, (note: using mp.set_start_method("spawn", force=True) to handle unittest, will behave slower)')
        data_ = self.gsge.make_compound_graphs(peptide_smiles, workers=2, pyg_data=False)
        
        adj_matrix, features = data_[0]

        self.assertTrue(np.array_equal(adj_matrix, self.expected_array), "Adjacency matrix mismatch")
        self.assertEqual(features, self.expected_features, "Feature list mismatch")

if __name__ == '__main__':
    unittest.main()