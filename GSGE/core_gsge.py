
from group_selfies import GroupGrammar
import torch_geometric.data, torch_geometric.loader, torch_geometric
from .vocab import GS_Vocab, GSGE_Corpus
from .chem import _ELEMENT_TOKENS, _GRAMMAR_TOKENS, _REDIRECT_TOKENS, _ELEMENTS_BOND_COUNTS
from rdkit import Chem
from tqdm import tqdm
from typing import Callable
import numpy as np
import pandas as pd
from .plots import highlight_fragments
from .graphs.fragment_graph.GAE import GraphDecoder, AttentiveFP, GraphAutoencoderTrainer, ATOM_MAX_NUM
from torch_geometric.loader import DataLoader
from .graphs.fragment_graph.from_smiles_to_graph import from_smiles, atom_to_token_id
from .fragment_tools import FragmentTools, GS_FragmentTools
import copy, re, torch, concurrent

class CoreGSGE:

    """ 
    Core functions
    
    """   

    @staticmethod
    def _frag_regex(frag_id:str, frag_regex_pattern:str):
        match = re.match(frag_regex_pattern, frag_id)
        return match.groups() if match else ('[UNK]',)
    
    @staticmethod
    def _preprocess(
        GS_string:str, 
        grammar_tokens:list[str]=_GRAMMAR_TOKENS, 
        element_tokens:list[str]=_ELEMENT_TOKENS,
        redirect_tokens:dict[str, str]=_REDIRECT_TOKENS,
        frag_regex_pattern:str=r"([^A-Za-z]*)(GS.*)",
        vocab: None | dict = None 
        ): 

        GS_strings = GS_string.replace('[', '').split(']')
        tokens = []

        for GS_str in GS_strings:
            if 'frag' in GS_str and 'GS' in GS_str:
                str_tokens = CoreGSGE._frag_regex(GS_str, frag_regex_pattern)
            elif GS_str in grammar_tokens:
                str_tokens = [GS_str]
            elif GS_str in element_tokens:
                str_tokens = [GS_str]
            elif GS_str in redirect_tokens:
                str_tokens = redirect_tokens[GS_str]
            elif GS_str != '':
                str_tokens = ['[UNK]']
            else:
                if GS_str == '':
                    continue
                else:
                    raise ValueError('Unexpected behavior during tokenization')
            
            tokens.extend(str_tokens)

        # Remove empty strings
        tokens_ = [token for token in tokens if token and token != '']
        
        if vocab is not None: #token to id
            tokens_ = np.array([vocab[token] for token in tokens_ if token])

        return tokens_
    
    @staticmethod
    def _preprocess_from_mol(mol, group_grammar:GroupGrammar, grammar_tokens:list, element_tokens:list, redirect_tokens:list, frag_regex_pattern:str, vocab=None):
        extracted = group_grammar.extract_groups(mol)
        return CoreGSGE._preprocess(group_grammar.encoder(mol, extracted), grammar_tokens, element_tokens, redirect_tokens, frag_regex_pattern, vocab=vocab)
    
    @staticmethod
    def _preprocess_from_smiles(smiles:str, group_grammar:GroupGrammar, grammar_tokens:list, element_tokens:list, redirect_tokens:list, frag_regex_pattern:str, vocab=None):
        mol = Chem.MolFromSmiles(smiles)
        return CoreGSGE._preprocess_from_mol(mol, group_grammar, grammar_tokens, element_tokens, redirect_tokens, frag_regex_pattern, vocab)
    
    @staticmethod
    def get_fragments_in_GS(GS_str_tokens:list, GS_vocab:GS_Vocab):
        """Note that GS_str_tokens is our post GS generation processing"""
        smi_frags_in_GS = []
        for GS_str_token in GS_str_tokens:
            frag_ = GS_vocab.vocab_fragments.get(GS_str_token, False)
            if frag_ is not False:
                smi_frags_in_GS.append(frag_.canonsmiles)
        return smi_frags_in_GS
    
    @staticmethod
    def _parallel_tokenize_df(
        batch_df: pd.DataFrame,
        process_func: Callable,
        standardize_fn: None | Callable = None,
        standardize_args: dict = {'suppress_exception': True},
        max_workers: None | int = None,
        smiles_column='SMILES'
    ) -> np.ndarray | list[str]:
        """
        General function to process a DataFrame in parallel and return tokenized results.
        
        Args:
            batch_df: Input DataFrame with SMILES column
            process_func: Function to apply to each SMILES string
            standardize_fn: Optional standardization function
            standardize_args: Arguments for standardization function
            max_workers: Number of parallel workers
        
        Returns:
            numpy array of padded tokenized results
        """
        # Handle standardization
        if callable(standardize_fn):
            stnd_batch_df = standardize_fn(batch_df.copy(), **standardize_args)
            stnd_batch_df = stnd_batch_df.dropna().reset_index(drop=True).copy()
        elif standardize_fn is None:
            stnd_batch_df = batch_df
        else:
            raise ValueError('standardize_fn must be callable or None')

        smiles_list = stnd_batch_df[smiles_column].values

        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_func, smiles_list), 
                            total=len(smiles_list), 
                            desc="parallel_tokenize_batch"))

        # Pad results to uniform length
        max_len = max(len(arr) for arr in results)
        padded_results = np.array([
            np.pad(arr, (0, max_len - len(arr)), mode='constant', constant_values=0)
            for arr in results
        ])

        return padded_results, smiles_list

    @staticmethod
    def plot_GS_fragments_in_mol(
        smiles:str,
        GS_vocab:GS_Vocab,
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

        full_mol = Chem.MolFromSmiles(smiles)
        highlighted_image = highlight_fragments(full_mol, GS_vocab, **args)
        return highlighted_image
    
    @staticmethod
    def check_for_non_grouped_atoms(
        smiles:str,
        GS_vocab:GS_Vocab,
        verbose=True
        ):
        
        """Note that GS_tokenized is our post GS generation processing"""

        full_mol = Chem.MolFromSmiles(smiles)
        highlighted_image = CoreGSGE.find_missing_atoms_in_fragments_(full_mol, GS_vocab, verbose=verbose)
        return highlighted_image

    @staticmethod
    def find_missing_atoms_in_fragments_(full_mol, vocab, verbose=True):
        """
        Extracts fragments using GroupGrammar and identifies atoms 
        not included in any fragment.

        Parameters:
        - full_mol: RDKit molecule (RDKit Mol)
        - vocab: a GS_Vocab or compatible object with .vocab_fragments
        - verbose: whether to print missing atom info

        Returns:
        - List of (atom_idx, atom_symbol) for atoms not included in any fragment
        """
        if not isinstance(full_mol, Chem.rdchem.Mol):
            raise ValueError("full_mol must be an RDKit molecule object")
        
        grammar_fragment = GroupGrammar(vocab=vocab.vocab_fragments)
        extracted_groups = grammar_fragment.extract_groups(full_mol)

        num_atoms = full_mol.GetNumAtoms()
        encountered_atoms = set()

        for fragment in extracted_groups:
            atom_indices = fragment[1]
            for atom_idx, _ in atom_indices:
                if atom_idx < num_atoms:
                    encountered_atoms.add(atom_idx)

        all_atoms = set(range(num_atoms))
        missing_atoms = sorted(all_atoms - encountered_atoms)

        missing_info = [
            (atom_idx, full_mol.GetAtomWithIdx(atom_idx).GetSymbol())
            for atom_idx in missing_atoms
        ]

        if missing_info and verbose:
            print("Warning: The following atoms were not included in any fragment (single elements are still present in the group-selfies):")
            for atom_idx, atom_symbol in missing_info:
                print(f" - Atom {atom_idx} ({atom_symbol})")

        return missing_info

    @staticmethod
    def train_GSGE_Auto_Encoder(
        GS_vocab:GS_Vocab|str, 
        GSGE_corpus:GSGE_Corpus|str, 
        batch_size:int=64,
        learning_rate=0.001,
        num_epochs:int=300,
        checkpoint_interval:int=5,
        val_percentage:float=0.2,
        split_seed:int=42,
        device= 'cuda' if torch.cuda.is_available() else 'cpu',
        encoder=AttentiveFP(in_channels=9, hidden_channels=256, out_channels=128, edge_dim=3, num_layers=3, num_timesteps=2),
        decoder:GraphDecoder=GraphDecoder(latent_dim=128, hidden_dim=256),
        optimizer= torch.optim.Adam,
        checkpoint_dir:str = "./model_checkpoints",
        load_checkpoint_path: None | str = None
        ):
        
        # #load in data
        train_loader, val_loader = CoreGSGE.load_and_prepare_data(
            GS_vocab, GSGE_corpus, x_percent=val_percentage, seed=split_seed, batch_size=batch_size)

        # Handle optimizer: if already instantiated, use it; otherwise create from class
        if hasattr(optimizer, 'param_groups') or hasattr(optimizer, 'step'):
            # optimizer is already an instantiated optimizer object
            pass  # use as-is
        else:
            # optimizer is a class (like torch.optim.Adam), instantiate it
            optimizer = optimizer(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)
        
        # Initialize trainer
        trainer = GraphAutoencoderTrainer(
            encoder=encoder,
            decoder=decoder,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=checkpoint_dir,
            device=device,
            batch_size=batch_size,
            load_checkpoint_path=load_checkpoint_path
        )

        # Run training
        trainer.train(num_epochs=num_epochs, checkpoint_interval=checkpoint_interval)
    
    @staticmethod
    def embed_fragments(
        frag_smiles:list, 
        encoder, 
        max_atom_size:int=20, 
        process_smiles:Callable=from_smiles, 
        atom_to_token_id:dict[int,int]=atom_to_token_id,
        return_data:bool = False,
        device='cuda',
        batch_size=64
        )-> np.ndarray | torch_geometric.data.Data:

        import torch

        graph_data = []
        for j, smiles in enumerate(frag_smiles):
            try:
                data_ = process_smiles(smiles.replace('*1', '*'), atom_to_token_id=atom_to_token_id)
                if data_.x.shape[0] > max_atom_size:
                    print('skipping', j, 'larger them max size') 
                    continue
                graph_data.append(data_)
            except:
                continue

        encoder.eval()

        data_loader = DataLoader(graph_data, batch_size=batch_size, shuffle=False)
        embeddings = []
        for batch in tqdm(data_loader): 
            batch = batch.to(device)
            with torch.no_grad():
                z = encoder(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                embeddings.append(np.array(z.detach().cpu()))

        embeddings = np.vstack(embeddings)  # Shape: (num_samples, embedding_dim)

        if return_data is True:
            return embeddings, graph_data
        else:
            return embeddings

    @staticmethod
    def add_GS_vocab_to_GSGE_corpus(GS_vocab:dict, GSGE_corpus:GSGE_Corpus):
        for GS in GS_vocab.values():
            hash = FragmentTools.get_hash_by_smiles(GS.canonsmiles).hash
            in_vocab_flag = GSGE_corpus.hash_to_frag_info.get(hash, False)
            if in_vocab_flag is False:
                GSGE_corpus.add_GS_group(copy.deepcopy(GS))
    
    @staticmethod
    def add_single_elements(element_args, vocab: GSGE_Corpus | GS_Vocab):
        for element_arg in element_args: 
            GS = GS_FragmentTools.make_element_GS(element_arg)
            vocab.add_GS_group(GS)
    
    @staticmethod
    def add_all_single_elements(vocab: GSGE_Corpus | GS_Vocab, element_bond_counts: None | dict[str,list[int]]= _ELEMENTS_BOND_COUNTS):

        for element, bonds in element_bond_counts.items():
            for num_bonds in bonds:
                try:
                    group, hash, cn_smi = GS_FragmentTools.make_element_GS(element, num_bonds)
                    vocab.add_GS_group(group)
                except Exception as e:
                    print('Warning:', e)
                    continue

    @staticmethod
    def random_split_train_test(
        GSGE_corpus:GSGE_Corpus, 
        GS_vocab:GS_Vocab, 
        x_percent:float=0.2, 
        seed:int=42, 
        GS_vocab_oversample:int=10
        ):

        """
        Oversamples GS_vocab.fragments by a factor of 10, merges with GSGE_corpus.fragments,
        then samples x% into test_data, ensuring test_data does not contain GS_vocab.fragments.

        Parameters:
        - GSGE_corpus: Object with a `.fragments` list (augmented data)
        - GS_vocab: Object with a `.fragments` list (core used data)
        - x_percent: Float (0-1), percentage of total data to sample into test_data
        - seed: Optional random seed for reproducibility

        Returns:
        - train_data: List of strings for training
        - test_data: List of strings for testing
        """
        import random

        if seed is not None:
            random.seed(seed)  # Set seed for reproducibility

        # Oversample GS_vocab.fragments by a factor of 10
        train_data = GSGE_corpus.fragments + GS_vocab.fragments * GS_vocab_oversample

        # Create a test pool excluding GS_vocab.fragments
        test_pool = [frag for frag in train_data if frag not in GS_vocab.fragments]

        # Determine the number of samples for test_data
        num_test_samples = int(len(train_data) * x_percent)
        num_test_samples = min(num_test_samples, len(test_pool))  # Avoid oversampling

        # Randomly sample test_data without replacement
        test_data = random.sample(test_pool, num_test_samples)

        # Remove test_data from train_data
        train_data = [frag for frag in train_data if frag not in test_data]

        return train_data, test_data

    @staticmethod
    def load_and_prepare_data(
        GS_vocab:GS_Vocab|str, 
        GSGE_corpus:GSGE_Corpus|str, 
        x_percent:float=0.2, 
        seed:int=42, 
        max_atom_size:int=ATOM_MAX_NUM, 
        GS_vocab_oversample:int=10, 
        batch_size:int=64,
        process_smiles=from_smiles,
        ):
        
        """
        Load GSGE vocabulary and corpus, split into train/test, and convert SMILES to graph data.
        
        Args:
            vocab_path (str): Path to the GS_Vocab file.
            corpus_path (str): Path to the GSGE_Corpus file.
            x_percent (float): Fraction of data to use for test split (default: 0.2).
            train_limit (int): Maximum number of training samples to use (default: 2000).
            seed (int): Random seed for reproducibility (default: 42).
        
        Returns:
            tuple: (train_data, val_data) as lists of PyTorch Geometric Data objects.
        """
        
        if isinstance(GS_vocab, str):
            # Load vocabulary
            print("Loading vocabulary...")
            GS_vocab = GS_Vocab(load_path=GS_vocab)

        if isinstance(GSGE_corpus, str):
            # Load corpus
            print("Loading corpus...")
            GSGE_corpus = GSGE_Corpus(load_path=GSGE_corpus)

        # Split data
        print("Splitting data...")
        train_data_list, test_data_list = CoreGSGE.random_split_train_test(
            GSGE_corpus, GS_vocab, x_percent=x_percent, seed=seed, GS_vocab_oversample=GS_vocab_oversample)
        
        print(f"Train size: {len(train_data_list)}, Test size: {len(test_data_list)}")

        # Convert SMILES to graphs
        print("Converting SMILES to graphs for training data...")
        train_data = []
        for j, smiles in enumerate(train_data_list):
            try:
                data_ = process_smiles(smiles.replace('*1', '*'), atom_to_token_id=atom_to_token_id)
                if data_.x.shape[0] > max_atom_size:
                    print('skipping', j, 'larger them max size') 
                    continue
                train_data.append(data_)
            except:
                continue
                
        print("Converting SMILES to graphs for validation data...")
        val_data = [process_smiles(smiles.replace('*1', '*'), atom_to_token_id=atom_to_token_id) 
                    for smiles in test_data_list]
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader
  
    @staticmethod
    def create_combined_embeddings(OHE_vocab_size: int, fragment_embeddings: dict, embedding_dim: int, device='cpu') -> torch.Tensor | dict:
        """
        Create a combined embedding matrix for sparse (OHE) and dense (fragment embeddings).
        
        Args:
            OHE_vocab_size (int): The size of the OHE vocabulary (grammar tokens).
            fragment_embeddings (dict): A dictionary mapping fragment IDs to dense embeddings (e.g., word2vec-like).
            embedding_dim (int): The dimensionality of the dense fragment embeddings.
            device (torch.device): The device (CPU or CUDA) to store the embeddings.
        
        Returns:
            torch.Tensor: A combined embedding matrix for both sparse and dense tokens.
        """
        # The final combined embedding size will be OHE_vocab_size + embedding_dim
        final_embedding_size = OHE_vocab_size + embedding_dim
        
        # Initialize a list to hold all combined embeddings
        combined_embeddings = []
        dense_embeddings = []
        
        # Add one-hot encoded sparse embeddings for grammar tokens (OHE tokens)
        for i in range(OHE_vocab_size):
            one_hot = torch.zeros(final_embedding_size, dtype=torch.float32, device=device)
            one_hot[i] = 1  # One-hot encoding for each grammar token
            combined_embeddings.append(one_hot)
        
        # Add dense fragment embeddings
        for token_id, embedding in fragment_embeddings.items():
            # Create an embedding that has OHE_vocab_size zeros in the front, followed by the dense embedding
            dense_embedding_with_ohe = torch.cat([torch.zeros(OHE_vocab_size, device=device), torch.tensor(embedding, dtype=torch.float32, device=device)])
            combined_embeddings.append(dense_embedding_with_ohe)
            dense_embeddings.append(torch.tensor(embedding, dtype=torch.float32, device=device))
        
        # Stack the combined embeddings into a single tensor
        combined_embeddings = torch.stack(combined_embeddings)  # Shape: [total_vocab_size, OHE_vocab_size + embedding_dim]
        dense_embeddings = torch.stack(dense_embeddings)  # Shape: [total_vocab_size, OHE_vocab_size + embedding_dim]

        GSGE_combined_vocab = {i:emb for i, emb in enumerate(np.array(combined_embeddings))}
        
        return combined_embeddings, GSGE_combined_vocab, dense_embeddings
    
    @staticmethod
    def encode_GSGE(token_ids: np.ndarray | torch.Tensor , combined_embeddings: torch.Tensor, device='cpu') -> torch.Tensor:
        """
        Creates the hybrid sparse-dense embeddings for a batch of tokenized sequences using the combined embeddings.
        
        Args:
            token_ids: Shape [batch_size, seq_len], token IDs from GSGE_vocab. Can be either np.ndarray or torch.Tensor.
            combined_embeddings (torch.Tensor): Shape [total_vocab_size, OHE_vocab_size + embedding_dim], combined sparse and dense embeddings.
            device (torch.device): The device (CPU or CUDA) to store the embeddings.
        
        Returns:
            torch.Tensor: Shape [batch_size, seq_len, OHE_vocab_size + embedding_dim].
        """
        # Check if token_ids is a NumPy array or PyTorch tensor and convert accordingly
        if isinstance(token_ids, np.ndarray):
            token_ids_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        elif isinstance(token_ids, torch.Tensor):
            token_ids_tensor = token_ids.to(dtype=torch.long, device=device)
        else:
            raise TypeError("token_ids must be either a NumPy array or PyTorch tensor")
                
        # Efficient lookup using combined embeddings
        embeddings = combined_embeddings[token_ids_tensor]  # [batch_size, seq_len, OHE_vocab_size + embedding_dim]
        
        return embeddings