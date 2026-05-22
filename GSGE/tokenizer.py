from __future__ import annotations

from group_selfies import GroupGrammar
from numpy import dtype, ndarray

from .chem import (_GRAMMAR_TOKENS, _ELEMENT_TOKENS, _REDIRECT_TOKENS)
from .core_gsge import CoreGSGE
from functools import partial
import torch
import pandas as pd
import numpy as np
from rdkit import Chem
from selfies import decoder as sf_decoder
import os
from typing import Callable, Any
from pathlib import Path
import logging
import pyarrow as pa
import pyarrow.parquet as pq

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gsge import VocabularyManager


class GSGE_tokenizer:

    """
    Tokenizer for fragment-based molecular representation using Grammar-SMILES-Group-Element (GSGE) scheme.

    This class handles preprocessing and tokenization of molecules into fragment tokens based on
    grammar tokens, element tokens, redirect tokens, and special tokens. It integrates closely with a
    VocabularyManager instance and supports parallel tokenization for large datasets.

    Attributes:
        vocab_manager (VocabularyManager | None): Manages vocabulary and grammar information.
        grammar_tokens (list): List of grammar tokens used in tokenization.
        element_tokens (list): List of element tokens used in tokenization.
        redirect_tokens (list): List of redirect tokens for special handling.
        special_tokens (list): List of special tokens like [PAD], [UNK], etc.
        frag_regex_pattern (str): Regex pattern used to extract fragments from strings.

    Main Methods:
        frag_regex(frag_id): Extracts fragment info based on regex pattern.
        preprocess(GS_string, gsge_vocab=None): Preprocesses a string into tokens.
        preprocess_from_mol(mol, gsge_vocab=None): Preprocesses an RDKit molecule.
        preprocess_from_SMILES(smiles, gsge_vocab=None): Preprocesses from a SMILES string.
        preprocess_from_SELFIES(selfies, gsge_vocab=None): Preprocesses from a SELFIES string.
        get_fragments_in_GS(GS_str_tokens): Retrieves fragments from tokenized GS string.
        plot_GS_fragments_in_mol(smiles, args): Visualizes fragments in a molecule.
        parallel_tokenize_SMILES_df(df, ...): Parallel tokenization for a DataFrame of SMILES.
        parallel_tokenize_SMILES_list(smiles_list, ...): Parallel tokenization for a list of SMILES.
        large_scale_tokenizer(...): Batch tokenization for very large datasets with checkpointing.
        MLM_masking(id_tokens, ...): Applies masked language modeling token corruption for training.

    Usage example:
        tokenizer = GSGE_tokenizer(vocab_manager=my_vocab_manager)
        tokens = tokenizer.preprocess_from_SMILES("CCO")
        embeddings = tokenizer.MLM_masking(tokens)
    """

    def __init__(self, 
            vocab_manager:VocabularyManager|None, 
            grammer_tokens:list=_GRAMMAR_TOKENS, 
            element_tokens:list=_ELEMENT_TOKENS,
            redirect_tokens:list=_REDIRECT_TOKENS, 
            special_tokens:list=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]'], 
            frag_regex_pattern:str=r"([^A-Za-z]*)(GS.*)",
            ) :

        self.grammar_tokens = grammer_tokens
        self.element_tokens = element_tokens
        self.redirect_tokens = redirect_tokens
        self.special_tokens = special_tokens
        self.frag_regex_pattern = frag_regex_pattern
        
        self.vocab_manager = vocab_manager

    def frag_regex(self, frag_id:str):
        return CoreGSGE._frag_regex(frag_id, self.frag_regex_pattern)
    
    def preprocess(self, GS_string:str, gsge_vocab: None | dict = None):
        return CoreGSGE._preprocess(
            GS_string, 
            self.grammar_tokens, 
            self.element_tokens, 
            self.redirect_tokens, 
            self.frag_regex_pattern,
            gsge_vocab
            )
    
    def preprocess_from_mol(self, mol, gsge_vocab: None | dict=None):
        extracted = self.vocab_manager.group_grammar.extract_groups(mol)
        return self.preprocess(self.vocab_manager.group_grammar.encoder(mol, extracted), gsge_vocab)

    def preprocess_from_SMILES(self, smiles:str, gsge_vocab: None | dict=None):
        return self.preprocess_from_mol(Chem.MolFromSmiles(smiles), gsge_vocab)
    
    def preprocess_from_SELFIES(self, smiles:str, gsge_vocab: None | dict=None):
        return self.preprocess_from_SMILES(sf_decoder(smiles), gsge_vocab)

    def get_fragments_in_GS(self, GS_str_tokens:str):
        """Note that GS_str_tokens is our post GS generation processing"""
        return CoreGSGE.get_fragments_in_GS(GS_str_tokens, self.vocab_manager.GS_vocab)

    def plot_GS_fragments_in_mol(
            self, 
            smiles:str, 
            args:dict={
                'img_size':(1600, 1200), 
                'verbose':False,
                'color_method':'standard',
                'color_seed':42,
                'annotate_with_index':True,
                'annotate_atoms':True,
                }
            ):
        
        """Note that GS_tokenized is our post GS generation processing"""

        return CoreGSGE.plot_GS_fragments_in_mol(
            smiles=smiles, 
            GS_vocab=self.vocab_manager.GS_vocab,
            args=args
            )

    def parallel_tokenize_SMILES_df(
            self, 
            df:pd.DataFrame, 
            standardize_fn: None | Callable = None, 
            standardize_args:dict={},
            max_workers: int | None = None,
            smiles_column:str='SMILES'
            )-> tuple[Any, Any]:
        
        # Define the specific processing function
        process_func = partial(CoreGSGE._preprocess_from_smiles,
                            group_grammar=self.vocab_manager.group_grammar,
                            grammar_tokens=self.grammar_tokens,
                            element_tokens=self.element_tokens,
                            redirect_tokens=self.redirect_tokens,
                            frag_regex_pattern=self.frag_regex_pattern,
                            vocab=self.vocab_manager.GSGE_vocab)
        
        # Use the general parallel processing function
        padded_results, smiles_list = CoreGSGE._parallel_tokenize_df(
            batch_df=df,
            process_func=process_func,
            standardize_fn=standardize_fn,
            standardize_args=standardize_args,
            max_workers=max_workers,
            smiles_column=smiles_column
        )

        return padded_results, smiles_list
    
    def parallel_tokenize_SMILES_list(
        self,
        smiles_list: list[str],
        standardize_fn: None | Callable = None,
        standardize_args: dict = {},
        max_workers: int | None = None
        ) -> tuple[Any, Any]:
        """
        Wrapper to tokenize a list of SMILES strings by converting it to a DataFrame first.
        """

        df = pd.DataFrame(smiles_list, columns=['SMILES'])

        return self.parallel_tokenize_SMILES_df(
            df=df,
            standardize_fn=standardize_fn,
            standardize_args=standardize_args,
            max_workers=max_workers,
            smiles_column='SMILES'
        )

    def large_scale_tokenizer(
        self,
        input_file:str,
        output_dir_name:str='GSGE_tokens_batches',
        output_file_name:str='GSGE_tokens.csv',
        batch_size:int=1000000, 
        checkpoint_file:str='checkpoint.txt',
        standardize_fn: None | Callable = None,
        standardize_args:dict={'suppress_exception':True},
        max_workers: None | int = None,
        smiles_column:str='SMILES'
        ):
        """
        Description: Use for pre-tokenizing large pre-training dataset to parquet format
        Example use:
        if __name__ == '__main__':
            import logging
            from GSGE import GSGE_tokenizer, GS_Vocab, get_use_examples_dir
            from GSGE.utils_chem import parallel_standardize

            examples_dir = get_use_examples_dir()
            if examples_dir is None:
                raise RuntimeError("Cannot find use_examples directory.")
            GS_vocab_path = str(examples_dir / '00_making_vocabs' / 'vocabs' / 'GS_vocab_v1')
            input_file = '25k_test_smiles_df.csv'
            output_dir_name = 'GSGE_tokens_batches'
            checkpoint_file = 'checkpoint.txt'
            logging_file = 'GSGE_tokens_processing.log'
            output_file_name = 'GSGE_tokens.csv'
            batch_size = 10**3

            GS_vocab = GS_Vocab()
            GS_vocab.load_GS_vocab(file_path=GS_vocab_path)
            GSGEtokenizer = GSGE_tokenizer(GS_vocab=GS_vocab)

            #can also use and call gsge.large_scale_tokenize: gsge = GSGE(GS_vocab=GS_vocab_path)

            standardize_fn=parallel_standardize
            standardize_args= {'suppress_exception':True}
            
            # Set up logging
            logging.basicConfig(filename=logging_file, level=logging.INFO,
                            format='%(asctime)s - %(message)s')
            
            GSGEtokenizer.large_scale_tokenizer(
                input_file=input_file, 
                output_dir_name=output_dir_name, 
                output_file_name=output_file_name,
                batch_size=batch_size, 
                checkpoint_file=checkpoint_file,
                standardize_fn=standardize_fn,
                standardize_args=standardize_args,
                )
        
        """

        _large_scale_tokenizer(
            self,
            input_file=input_file, 
            output_dir_name=output_dir_name,
            output_file_name=output_file_name,
            batch_size=batch_size, 
            checkpoint_file=checkpoint_file,
            standardize_fn=standardize_fn,
            standardize_args=standardize_args, 
            max_workers=max_workers,
            smiles_column=smiles_column
            )

    def MLM_masking(self, id_tokens:np.ndarray,  mask_probability=0.15, masking_fraction=0.8, corruption_fraction=0.1):
        gsge_mlm_masking = GSGE_MLM_masking(token_to_id=self.vocab_manager.GSGE_vocab)
        return gsge_mlm_masking.masking(torch.tensor(id_tokens), mask_probability, masking_fraction, corruption_fraction)

#MLM tools
class GSGE_MLM_masking:
    def __init__(self, token_to_id):
        self.token_to_id = token_to_id
        self.id_to_token = {id:token for token, id in self.token_to_id.items()}

    def create_attention_mask(self,sequences, special_tokens):
         return create_attention_mask(sequences, special_tokens)
    
    def masking(self, sequences, mask_probability=0.15, masking_fraction=0.8, corruption_fraction=0.1):
        attention_masks = create_attention_mask(sequences, {int(self.token_to_id['[PAD]']), int(self.token_to_id['[CLS]'])})  # assuming 0 is a special token like padding
        return masking(sequences, attention_masks, int(self.token_to_id['[MASK]']), list(self.id_to_token.keys()), mask_probability, masking_fraction, corruption_fraction)

def masking(sequences, attention_masks, mask_token_id, vocab, mask_probability=0.15, making_fraction=0.8, corruption_fraction=0.1):
    """
    Fully vectorized MLM masking implementation for higher performance.

    Returns:
    - masked_sequences (torch.Tensor): Sequences after applying the masking procedure.
    - decision_mask (torch.Tensor): Mask indicating all selected tokens for MLM (1 for masked, 0 for unmasked).
    - mask2 (torch.Tensor): Mask tracking which tokens were replaced by [MASK] in decision_mask.
    """
    vocab = torch.tensor(vocab, dtype=torch.int64, device=sequences.device)
    batch_size, seq_len = sequences.shape

    decision_mask = torch.zeros_like(sequences, dtype=torch.int64)  # Full 15% selection
    mask2 = torch.zeros_like(sequences, dtype=torch.int64)  # Only `[MASK]` tokens
    masked_sequences = sequences.clone()

    rand_vals = torch.rand(batch_size, seq_len, device=sequences.device)
    valid_positions = attention_masks == 1

    # Step 1: Generate decision_mask (15% of valid tokens)
    masking_decision = (rand_vals < mask_probability) & valid_positions  # Full 15%
    decision_mask[masking_decision] = 1

    # Step 2: Split the 15% selected tokens into `[MASK]`, corruption, and unchanged
    selected_indices = masking_decision.nonzero(as_tuple=True)  # Get indices of selected tokens
    num_selected = selected_indices[0].shape[0]  # Total count of selected tokens

    num_mask = int(num_selected * making_fraction)  # 80% `[MASK]`
    num_corrupt = int(num_selected * corruption_fraction)  # 10% corrupted

    # Shuffle selected indices so we can assign them randomly
    shuffled_indices = torch.randperm(num_selected, device=sequences.device)

    mask_indices = tuple(idx[shuffled_indices[:num_mask]] for idx in selected_indices)
    corrupt_indices = tuple(idx[shuffled_indices[num_mask:num_mask + num_corrupt]] for idx in selected_indices)

    # Apply `[MASK]` token
    masked_sequences[mask_indices] = mask_token_id
    mask2[mask_indices] = 1  # Track `[MASK]` replacements

    # Apply corruption (random tokens)
    random_tokens = vocab[torch.randint(0, len(vocab), (num_corrupt,), device=sequences.device)]
    masked_sequences[corrupt_indices] = random_tokens.to(masked_sequences.dtype)

    # Unchanged tokens stay the same (no need to modify)

    return masked_sequences, decision_mask, mask2

def create_attention_mask(sequences, special_tokens):
    """
    Create an attention mask where 1 represents a valid token (non-special) and 0 represents any special token (like padding or mask).
    
    Args:
    - sequences (torch.Tensor): A tensor of shape (batch_size, seq_len) representing the tokenized sequences.
    - special_tokens (set or torch.Tensor): A set or tensor of special token IDs that should be considered non-valid (e.g., PAD, MASK, etc.).
    
    Returns:
    - torch.Tensor: A tensor of attention masks of shape (batch_size, seq_len) with 1s for valid tokens and 0s for special tokens.
    """
    # Convert special_tokens to a tensor for fast comparison
    special_tokens = torch.tensor(list(special_tokens), dtype=torch.int64, device=sequences.device)

    # Step 1: Create an attention mask where 1 means valid token and 0 means special token
    # Check if each token in sequences is in special_tokens (using broadcasting)
    attention_mask = ~torch.isin(sequences, special_tokens)  # Invert, so valid tokens become True (1)

    # Step 2: Convert boolean mask to integer (1 for True, 0 for False)
    attention_mask = attention_mask.to(torch.int64)

    return attention_mask

###LARGE SCALE TOKENIZATION

def _process_GS_tokenization_batch(
    batch_df: pd.DataFrame,
    batch_num: int,
    output_dir: str,
    group_grammar: GroupGrammar,
    GSGE_vocab: dict,
    grammar_tokens: list = _GRAMMAR_TOKENS,
    element_tokens: list = _ELEMENT_TOKENS,
    redirect_tokens: list = _REDIRECT_TOKENS,
    frag_regex_pattern: str = r"([^A-Za-z]*)(GS.*)",
    standardize_fn: None | Callable = None,
    standardize_args: dict = {'suppress_exception': True},
    max_workers: None | int = None,
    smiles_column='SMILES'
) -> int:
    """
    Process a batch of SMILES, tokenize them, and save results to files.
    
    Args:
        batch_df: Input DataFrame with SMILES column
        batch_num: Batch number for file naming
        output_dir: Directory to save output files
        ... (other GS-specific parameters)
    
    Returns:
        batch_num: The processed batch number
    """
    # Define the specific processing function
    process_func = partial(CoreGSGE._preprocess_from_smiles,
                          group_grammar=group_grammar,
                          grammar_tokens=grammar_tokens,
                          element_tokens=element_tokens,
                          redirect_tokens=redirect_tokens,
                          frag_regex_pattern=frag_regex_pattern,
                          vocab=GSGE_vocab)

    # Use the general parallel processing function
    padded_results, smiles_list = CoreGSGE._parallel_tokenize_df(
        batch_df=batch_df,
        process_func=process_func,
        standardize_fn=standardize_fn,
        standardize_args=standardize_args,
        max_workers=max_workers,
        smiles_column=smiles_column
    )

    # Create SMILES DataFrame
    smiles_df = pd.DataFrame({smiles_column: smiles_list})
    
    # Define and create output paths
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    smiles_file = output_dir / f'batch_{batch_num}_smiles.csv'
    arrays_file = output_dir / f'batch_{batch_num}_arrays.parquet'
    
    # Save results
    smiles_df.to_csv(smiles_file, index=False)
    
    table = pa.Table.from_pydict({
        'GSGE_tokens': padded_results.tolist()
    })
    pq.write_table(table, arrays_file, compression='snappy')
    
    logging.info(f'Completed batch {batch_num}')
    return batch_num

def _combine_results(output_dir: str, total_batches: int, output_file_name: str):
    """Combine all batch files into final result files:
    - SMILES in a single CSV
    - Token arrays in a single Parquet
    Stored in a 'combined' subdirectory"""
    
    output_dir = Path(output_dir)
    combined_dir = output_dir / 'combined'
    combined_dir.mkdir(exist_ok=True)  # Create 'combined' dir if it doesn't exist
    
    all_smiles_dfs = []
    all_arrays = []

    # Process each batch
    for batch_num in range(total_batches):
        # SMILES file
        smiles_file = output_dir / f'batch_{batch_num}_smiles.csv'
        if smiles_file.exists():
            all_smiles_dfs.append(pd.read_csv(smiles_file))
        
        # Arrays file
        arrays_file = output_dir / f'batch_{batch_num}_arrays.parquet'
        if arrays_file.exists():
            table = pq.read_table(arrays_file)
            arrays = table['GSGE_tokens'].to_pylist()
            all_arrays.extend(arrays)

    # Combine SMILES into final CSV
    if all_smiles_dfs:
        final_smiles_df = pd.concat(all_smiles_dfs, ignore_index=True)
        smiles_output_file = combined_dir / f"{output_file_name}_smiles.csv"
        final_smiles_df.to_csv(smiles_output_file, index=False)
        logging.info(f'Combined {len(all_smiles_dfs)} SMILES batches into {smiles_output_file}')
    else:
        logging.warning('No SMILES batches found to combine')

    # Combine arrays into final Parquet
    if all_arrays:
        # Find the maximum length across ALL arrays
        max_len = max(len(arr) for arr in all_arrays)
        
        # Pad all arrays to the global maximum length
        padded_arrays = [
            np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=0)
            for arr in all_arrays
        ]
        
        # Convert to 2D numpy array
        final_arrays = np.array(padded_arrays)
        
        arrays_output_file = combined_dir / f"{output_file_name}_tokens.parquet"
        
        # Create Arrow Table and write to Parquet
        table = pa.Table.from_pydict({
            'GSGE_tokens': final_arrays.tolist()
        })
        pq.write_table(table, arrays_output_file, compression='snappy')
        logging.info(f'Combined {len(all_arrays)} token arrays into {arrays_output_file} with shape {final_arrays.shape}')
    else:
        logging.warning('No array batches found to combine')

    print('Done')

def _large_scale_tokenizer(
    GSGEtokenizer: GSGE_tokenizer, 
    input_file:str, 
    output_dir_name:str='GSGE_tokens_batches',
    output_file_name:str='GSGE_tokens.csv',
    batch_size:int=1000000, 
    checkpoint_file:str='checkpoint.txt',
    standardize_fn: str | None =None,
    standardize_args:dict={'suppress_exception':True},
    max_workers: None | int =None,
    smiles_column='SMILES'
    ):

    """
    if __name__ == '__main__':
        from GSGE import get_use_examples_dir

        examples_dir = get_use_examples_dir()
        if examples_dir is None:
            raise RuntimeError("Cannot find use_examples directory.")
        GS_vocab_path = str(examples_dir / '00_making_vocabs' / 'vocabs' / 'GS_vocab_v1')
        input_file = '25k_test_smiles_df.csv'
        output_dir_name = 'GSGE_tokens_batches'
        checkpoint_file = 'checkpoint.txt'
        logging_file = 'GSGE_tokens_processing.log'
        output_file_name = 'GSGE_tokens.csv'
        batch_size = 10**3

        GS_vocab = GS_Vocab()
        GS_vocab.load_GS_vocab(file_path=GS_vocab_path)
        GSGEtokenizer = GSGE_tokenizer(GS_vocab=GS_vocab)

        standardize_fn=parallel_standardize
        standardize_args= {'suppress_exception':True}
        
        # Set up logging
        logging.basicConfig(filename=logging_file, level=logging.INFO,
                        format='%(asctime)s - %(message)s')
        
        large_scale_tokenizer(
            GSGEtokenizer=GSGEtokenizer, 
            input_file=input_file, 
            output_dir_name=output_dir_name, 
            output_file_name=output_file_name,
            batch_size=batch_size, 
            checkpoint_file=checkpoint_file,
            standardize_fn=standardize_fn,
            standardize_args=standardize_args,
            )
            
    """
    
    # Create output directory
    output_dir = Path(output_dir_name)
    output_dir.mkdir(exist_ok=True)
    
    # Load full dataset
    logging.info('Loading dataset...')
    df = pd.read_csv(input_file)

    # Calculate total batches
    total_rows = len(df)
    total_batches = (total_rows + batch_size - 1) // batch_size
    
    # Check for existing checkpoint
    start_batch = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            start_batch = int(f.read().strip())
        logging.info(f'Resuming from batch {start_batch}')
        print(f'Resuming from batch {start_batch}')
    
    # Process in batches
    for batch_num in range(start_batch, total_batches):
        print('Batch:',batch_num+1, 'Out of:', total_batches)
        start_idx = batch_num * batch_size
        end_idx = min((batch_num + 1) * batch_size, total_rows)
        
        batch_df = df.iloc[start_idx:end_idx]
        
        try:
            _process_GS_tokenization_batch(
                batch_df, 
                batch_num, 
                output_dir, 
                GSGEtokenizer.vocab_manager.group_grammar,
                GSGEtokenizer.vocab_manager.GSGE_vocab, 
                GSGEtokenizer.grammar_tokens, 
                GSGEtokenizer.element_tokens, 
                GSGEtokenizer.redirect_tokens, 
                GSGEtokenizer.frag_regex_pattern,
                standardize_fn=standardize_fn,
                standardize_args=standardize_args,
                max_workers=max_workers,
                smiles_column=smiles_column
                )
            
            # Update checkpoint
            with open(checkpoint_file, 'w') as f:
                f.write(str(batch_num + 1))
                
        except Exception as e:
            logging.error(f'Error processing batch {batch_num}: {str(e)}')
            raise
    
    _combine_results(output_dir, total_batches, output_file_name)