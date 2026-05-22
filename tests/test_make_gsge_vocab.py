from GSGE import GS_Vocab, CUSTOM_fragment_mol
import unittest
import random
import pytest
from pathlib import Path

random.seed(42)

class TestGSGECorpus(unittest.TestCase):

    def setUp(self):
        self.expected_num_fragments = 25
        self.expected_fragment_smiles = [
            'O(*1)*1', 'C1=C2C(=C(*1)C(*1)=C1*1)N(*1)C(*1)=C2*1', 'N1C2=C(*1)C(*1)=C(*1)C(*1)=C2C(*1)=C1*1', 'N1=C(*1)N(*1)C(*1)=C1*1', 'N1=C(*1)NC(*1)=C1*1',
            'C(C(CC(*1)(*1)*1)*1)N(*1)*1', 'CC(C*1)C(N*1)C(=O)*1', 'C(=C(C(=C(*1)*1)*1)*1)(C(=C(*1)*1)*1)*1', 'O=C(O)CCC(N*1)C(=O)*1', 'NCCCCC(N*1)C(=O)*1',
            'NC(=O)CC(N*1)C(=O)*1', 'N=C(N)NCCCC(N*1)C(=O)*1', 'CSCCC(N*1)C(=O)*1', 'C(*1)(*1)(*1)*1', 'O=C(C(CS*1)N*1)*1', 'NC(=O)CCC(N*1)C(=O)*1', 'O=C(*1)*1',
            'N(*1)(*1)*1', 'O=C(C(N*1)*1)*1', 'CC(C)CC(N*1)C(=O)*1', 'O=C(C(N*1)C(O)*1)*1', 'C1=C(*1)C=C(*1)C(*1)=C1*1', 'Cl*1', 'O=C(O)CC(N*1)C(=O)*1',
            'S=C(*1)(*1)']

    @pytest.mark.slow
    def test_fragmentation_and_vocab(self):
        pkl_path = Path(__file__).parent / 'subset_smiles_1000.pkl'
        import pickle

        with open(pkl_path, "rb") as f:
            subset = pickle.load(f)

        GS_vocab = GS_Vocab()
        GS_vocab.build_vocab(
            m_set=subset, 
            convert=True, 
            n_limit=80, 
            target=200, 
            MIN_SIZE=1, 
            MAX_SIZE=15, 
            method='default', 
            fragment_mol_fn=CUSTOM_fragment_mol
            )
        
        
        #test ading fragments manually
        GS_vocab.add_GS_fragment('S=C(*1)(*1)') 

        self.assertEqual(GS_vocab.num_fragments, self.expected_num_fragments)

        actual_fragment_smiles = [v.canonsmiles for k, v in GS_vocab.vocab_fragments.items()]
        self.assertCountEqual(actual_fragment_smiles, self.expected_fragment_smiles)

        from GSGE import GSGE
        gsge = GSGE(GS_vocab=GS_vocab)
        gsge.add_all_single_elements()

        self.assertEqual(gsge.get_GS_vocab().fragments.__len__(), 80)
        self.assertEqual(gsge.get_fragments_smiles().__len__(), 80)

 


if __name__ == '__main__':
    unittest.main()
    
    # Example usage

    # import importlib.resources as pkg_resources

    # pkl_path = pkg_resources.files(tests).joinpath('subset_smiles_1000.pkl')
    # import pickle

    # with open(pkl_path, "rb") as f:
    #     subset = pickle.load(f)
    

    # GS_vocab = GS_Vocab()
    # GS_vocab.build_vocab(
    #     m_set=subset, 
    #     convert=True, 
    #     n_limit=80, 
    #     target=200, 
    #     MIN_SIZE=1, 
    #     MAX_SIZE=15, 
    #     method='default', 
    #     fragment_mol_fn=CUSTOM_fragment_mol
    #     )

    # print(GS_vocab.num_fragments)
    # print([v.canonsmiles for k,v in GS_vocab.vocab_fragments.items()])

