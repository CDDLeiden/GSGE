from .gsge import GSGE
from .core_gsge import CoreGSGE
import torch
import torch.nn as nn
import numpy as np

class GSGE_Embedding(nn.Module):
    def __init__(
        self, 
        sparse_vocab_size: int, 
        dense_size: int, 
        embedding_dim: int, 
        GSGE_combined_embeddings: str|torch.Tensor, 
        padding_idx: int = None,
        fine_tune_embeddings: bool = False,  # allows the vectors to be set a trainable parameters
        input_token_ids: bool=True,
        only_token2vec: bool=False,
        no_grad=False
    ):
        """
        Combines sparse one-hot encodings and dense embeddings into a hybrid representation.
        
        Args:
            sparse_vocab_size (int): Number of sparse tokens (vocabulary size for one-hot encodings).
            dense_size (int): Dimensionality of the pre-trained dense embeddings.
            embedding_dim (int): Dimensionality of the learned embeddings for sparse tokens.
            GSGE_combined_embeddings (torch.Tensor): Embedding vectors in order of token id ints. (can also provide path to load it from GSGE instance)
            padding_idx (int, optional): Index to use for padding (dense positions). Defaults to sparse_vocab_size.
            fine_tune_embeddings (bool, optional): If True, make GSGE_combined_embeddings trainable. Defaults to False.
            input_token_ids (bool): 
        """
        super().__init__()
        self.sparse_vocab_size = sparse_vocab_size
        self.dense_size = dense_size
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else sparse_vocab_size

        if isinstance(GSGE_combined_embeddings, str):
            gsge = GSGE(GSGE_load_path=GSGE_combined_embeddings)
            GSGE_combined_embeddings = gsge.embedding_manager.GSGE_combined_embeddings

        
        # TODO: Consider making GSGE_combined_embeddings fine-tunable
        if fine_tune_embeddings:
            self.GSGE_combined_embeddings = nn.Parameter(GSGE_combined_embeddings)
        else:
            self.register_buffer('GSGE_combined_embeddings', GSGE_combined_embeddings)
        
        # Embedding layer for sparse grammar tokens
        self.sparse_embed = nn.Embedding(sparse_vocab_size + 1, embedding_dim, padding_idx=self.padding_idx)

        self.input_token_ids = input_token_ids
        self.only_token2vec = only_token2vec
        self.no_grad = no_grad

    def token2vec(self, token_ids: torch.Tensor | np.ndarray):
        return CoreGSGE.encode_GSGE(token_ids, self.GSGE_combined_embeddings, device=self.GSGE_combined_embeddings.device)

    def forward(self, input_tensor, verbose=False):
        """
        Processes a tensor combining sparse one-hot encodings and dense embeddings.
        
        Args:
            input_tensor (torch.Tensor): Shape [batch_size, seq_len, sparse_vocab_size + dense_size].
                                         First sparse_vocab_size dims are one-hot, rest are dense.
            verbose (bool): If True, print debugging info.
        
        Returns:
            torch.Tensor: Shape [batch_size, seq_len, embedding_dim + dense_size].
                          Hybrid representation with embedded sparse and original dense parts.
        """
        if self.input_token_ids:
            if self.no_grad:
                with torch.no_grad():
                    input_tensor = self.token2vec(input_tensor)
            else:
                input_tensor = self.token2vec(input_tensor)
            
            if self.only_token2vec:
                return input_tensor

        # #batch_size, seq_len, total_size = input_tensor.shape
        # assert total_size == self.sparse_vocab_size + self.dense_size, \
        #     f"Expected total size {self.sparse_vocab_size + self.dense_size}, got {total_size}"
        
        if verbose:
            print('______input______')
            print('sparse_vocab_size:', self.sparse_vocab_size)
            print('embedding_dim of sparse embeddings:', self.embedding_dim)
            print('input_tensor', input_tensor.shape)
            print('expected output for hybrid:', input_tensor.shape[-1] - self.sparse_vocab_size + self.embedding_dim)

        # Split into sparse and dense parts
        sparse_part = input_tensor[:, :, :self.sparse_vocab_size]  # [batch_size, seq_len, sparse_vocab_size]
        dense_part = input_tensor[:, :, self.sparse_vocab_size:]   # [batch_size, seq_len, dense_size]

        # Convert sparse one-hot to indices
        sparse_indices = torch.argmax(sparse_part, dim=-1)  # [batch_size, seq_len]
        sparse_sum = sparse_part.sum(dim=-1)  # [batch_size, seq_len]
        sparse_indices = torch.where(sparse_sum > 0, sparse_indices, torch.full_like(sparse_indices, self.padding_idx))

        # Embed sparse indices
        sparse_embedded = self.sparse_embed(sparse_indices)  # [batch_size, seq_len, embedding_dim]

        # Masks: 1 where sparse, 0 where dense; 0 where sparse, 1 where dense
        mask_sparse = (sparse_sum > 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        mask_dense = (sparse_sum == 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]

        # Concatenate embedded sparse and original dense parts with masking
        hybrid_output = torch.cat([
            sparse_embedded * mask_sparse,  # Zero out sparse where dense
            dense_part * mask_dense         # Zero out dense where sparse
        ], dim=-1)  # [batch_size, seq_len, embedding_dim + dense_size]

        if verbose:
            print('______hybrid_output______')
            print('hybrid_output:', hybrid_output.shape)
            print('_________________________')
            assert hybrid_output.shape[-1] == input_tensor.shape[-1] - self.sparse_vocab_size + self.embedding_dim

        return hybrid_output