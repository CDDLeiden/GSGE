import unittest
import torch
import random
import numpy as np
import os
from pathlib import Path

from GSGE import GSGE_Embedding
from GSGE import GSGE

def set_deterministic(seed=42):
    """Seed everything for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

class TestGSGEEmbedding(unittest.TestCase):

    def setUp(self):
        set_deterministic(seed=1234)  # You can change this seed if needed

        # Load GSGE with saved fragment data
        pkl_path = Path(__file__).parent / 'test_gsge_save_with_descriptors.pkl'
        self.gsge = GSGE(GSGE_load_path=pkl_path)

        # Select look-up table type and get embedding layer
        self.look_up_table = self.gsge.get_fragment_descriptors()
        emb_size = self.look_up_table.shape[1]
        self.embedding_layer = GSGE_Embedding(
            0, emb_size, 0, self.look_up_table,
            only_token2vec=True, no_grad=True
        )

        self.expected_sum = -137.03770446777344 

    def test_embedding_determinism(self):
        # Fixed input for determinism test
        input_tensor = torch.randint(low=0, high=185, size=(4, 10), dtype=torch.long)

        # Apply embedding
        embedded_output = self.embedding_layer(input_tensor)

        # Reduce to a single value for easy comparison
        result_sum = embedded_output.sum().item()


        self.assertAlmostEqual(result_sum, self.expected_sum, places=5, msg="Embedding output is not deterministic.")

if __name__ == '__main__':
    unittest.main()
