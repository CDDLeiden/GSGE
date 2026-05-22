from .vocab import GS_Vocab, GSGE_Corpus
from .fragment_tools import GS_FragmentTools
from group_selfies import GroupGrammar
from .chem import (_GRAMMAR_TOKENS, _ELEMENT_TOKENS, _REDIRECT_TOKENS, 
    _ELEMENTS_BOND_COUNTS, SPECIAL_TOKENS, COMMON_SMALLER_FRAGMENTS)
from .graphs.fragment_graph.GAE import AttentiveFP, GraphDecoder, ATOM_MAX_NUM
from .core_gsge import CoreGSGE
from  .graphs.fragment_graph.from_smiles_to_graph import from_smiles, atom_to_token_id
from .fragment_descriptors import normalize_descriptors, get_mol_frag_descriptors, Descriptors
from typing import List, Dict, Optional, Tuple, Union
from torch_geometric.data import Data
from .graphs.compound_graph.data import parallel_, preprocess_graph, smiles_to_group_graph
from functools import partial
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle

from .clustering import GSGE_clustering
from .tokenizer import GSGE_tokenizer

class Store_Data:

    """
    Handles saving and loading of GSGE-related data to/from disk using pickle.

    Attributes:
        gsge (GSGE): An instance of the GSGE class containing tokenizer, vocab, embeddings, and descriptors.
        GSGE_load_path (str, optional): Path to load previously saved GSGE data.

    Methods:
        save_gsge_data(filepath: str, meta_info: str = ''):
            Saves relevant GSGE components and metadata to the specified filepath.

        load_gsge_data(filepath: str):
            Loads GSGE components from a file and updates the provided gsge instance accordingly.
    """

    def __init__(self, gsge: 'GSGE', GSGE_load_path: Optional[str] = None):
        """
        Initialize Store_Data handler for saving/loading GSGE state.

        Args:
            gsge: GSGE instance to manage persistence for.
            GSGE_load_path: Path to previously saved GSGE data file (.pkl).
                If provided, automatically loads data during initialization.

        Example:
            Create handler and load existing data:

            >>> gsge = GSGE()
            >>> store = Store_Data(gsge, GSGE_load_path='gsge_save.pkl')

            Create handler for later saving:

            >>> store = Store_Data(gsge)
            >>> store.save_gsge_data('new_save.pkl')
        """
        self.gsge = gsge
        self.GSGE_load_path = GSGE_load_path

        if GSGE_load_path is not None:
            self.load_gsge_data(GSGE_load_path) 

    def save_gsge_data(self, filepath: str, meta_info: str = '') -> None:
        """
        Save complete GSGE state to pickle file for later reuse.

        Serializes all vocabularies, tokenizer configuration, embeddings, and
        descriptors to a single pickle file. This allows complete reconstruction
        of GSGE instance without retraining or recomputing.

        Args:
            filepath: Output path for pickle file. Common convention is .pkl extension.
            meta_info: Optional metadata string to store with saved data (e.g.,
                training info, version, dataset used).

        Example:
            Save GSGE with metadata:

            >>> gsge.save_gsge_data(
            ...     'gsge_cyclic_peptides_v2.pkl',
            ...     meta_info='Trained on 5000 cyclic peptides, 300 epochs'
            ... )
            Data saved to gsge_cyclic_peptides_v2.pkl

        Note:
            Saved data includes:
            - GS_vocab and GSGE_corpus
            - Tokenizer configuration
            - Fragment embeddings (if generated)
            - Descriptor calculations (if computed)
            - OHE tokens mask
            File sizes typically range from 1-50 MB depending on vocabulary size.
        """
        GSGE_tokenizer_args = {
            'grammar_tokens': self.gsge.tokenizer.grammar_tokens,
            'element_tokens': self.gsge.tokenizer.element_tokens,
            'redirect_tokens': self.gsge.tokenizer.redirect_tokens,
            'special_tokens': self.gsge.tokenizer.special_tokens,
            'frag_regex_pattern': self.gsge.tokenizer.frag_regex_pattern
        }

        data = {
            'GSGE_corpus': self.gsge.vocab_manager.GSGE_corpus,
            'GS_vocab': self.gsge.vocab_manager.GS_vocab,
            'GSGE_vocab': self.gsge.vocab_manager.GSGE_vocab,
            'GSGE_tokenizer_args': GSGE_tokenizer_args,
            'GS_frag_id_to_embedding': self.gsge.embedding_manager.GS_frag_id_to_embedding,
            'GSGE_id_to_embedding': self.gsge.embedding_manager.GSGE_id_to_embedding,
            'GSGE_combined_embeddings': self.gsge.embedding_manager.GSGE_combined_embeddings,
            'GSGE_GAE_embeddings': self.gsge.embedding_manager.GSGE_GAE_embeddings,
            'GSGE_fragment_descriptors': self.gsge.descriptor_calculator.GSGE_fragment_descriptors,
            'fragment_descriptors_keys': self.gsge.descriptor_calculator.fragment_descriptors_keys,
            'OHE_tokens_mask': self.gsge.vocab_manager.OHE_tokens_mask,
            'meta_info': meta_info
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {filepath}")

    def load_gsge_data(self, filepath: str) -> None:
        """
        Load complete GSGE state from pickle file and restore all components.

        Deserializes previously saved GSGE data and updates all manager components
        (vocabulary, tokenizer, embeddings, descriptors) in the current instance.
        Allows resuming work without rebuilding vocabularies or retraining.

        Args:
            filepath: Path to saved GSGE pickle file (created by save_gsge_data).

        Raises:
            FileNotFoundError: If filepath does not exist.
            pickle.UnpicklingError: If file is corrupted or incompatible.

        Example:
            Load pre-trained GSGE:

            >>> gsge = GSGE()
            >>> gsge.load_gsge_data('gsge_save.pkl')
            Data loaded and restored into gsge from gsge_save.pkl

            >>> print(len(gsge.get_fragments_smiles()))
            200  # Vocabulary loaded successfully

        Note:
            Automatically updates:
            - vocab_manager (GS_vocab, GSGE_corpus, GSGE_vocab)
            - tokenizer (grammar, element, special tokens)
            - embedding_manager (fragment embeddings if available)
            - descriptor_calculator (descriptors if computed)
            Regenerates token-to-ID mappings after loading.
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)

        # Restore vocab_manager-related data
        self.gsge.vocab_manager.GSGE_corpus = data.get('GSGE_corpus')
        self.gsge.vocab_manager.GS_vocab = data.get('GS_vocab')
        self.gsge.vocab_manager.GSGE_vocab = data.get('GSGE_vocab')
        self.gsge.vocab_manager.OHE_tokens_mask = data.get('OHE_tokens_mask')

        # Restore tokenizer-related data
        tokenizer_args = data.get('GSGE_tokenizer_args', {})
        self.gsge.tokenizer.grammar_tokens = tokenizer_args.get('grammar_tokens')
        self.gsge.tokenizer.element_tokens = tokenizer_args.get('element_tokens')
        self.gsge.tokenizer.redirect_tokens = tokenizer_args.get('redirect_tokens')
        self.gsge.tokenizer.special_tokens = tokenizer_args.get('special_tokens')
        self.gsge.tokenizer.frag_regex_pattern = tokenizer_args.get('frag_regex_pattern')

        # Restore embedding_manager-related data
        self.gsge.embedding_manager.GS_frag_id_to_embedding = data.get('GS_frag_id_to_embedding')
        self.gsge.embedding_manager.GSGE_id_to_embedding = data.get('GSGE_id_to_embedding')
        self.gsge.embedding_manager.GSGE_combined_embeddings = data.get('GSGE_combined_embeddings')
        self.gsge.embedding_manager.GSGE_GAE_embeddings = data.get('GSGE_GAE_embeddings')

        # Restore descriptor_calculator-related data
        self.gsge.descriptor_calculator.GSGE_fragment_descriptors = data.get('GSGE_fragment_descriptors')
        self.gsge.descriptor_calculator.fragment_descriptors_keys = data.get('fragment_descriptors_keys')

        # Optional meta_info
        self.gsge.vocab_manager.meta_info = data.get('meta_info', '')

        self.gsge.vocab_manager.set_GS_vocab(self.gsge.vocab_manager.GS_vocab)
        self.gsge.vocab_manager.set_GSGE_corpus(self.gsge.vocab_manager.GSGE_corpus)
        self.gsge.vocab_manager.get_GSGE_vocab_token_to_id_dict(
            special_tokens=self.gsge.tokenizer.special_tokens,
            grammar_tokens=self.gsge.tokenizer.grammar_tokens,
            element_tokens=self.gsge.tokenizer.element_tokens,
        )

        print(f"Data loaded and restored into gsge from {filepath}")

class VocabularyManager:

    """
    Manages the vocabularies and corpus used by the GSGE model, including loading,
    processing, and generating token-to-ID mappings.

    Attributes:
        GSGE_vocab (dict): Token-to-ID mapping for the full GSGE vocabulary.
        GS_vocab (GS_Vocab): Grammar Structure vocabulary instance.
        GSGE_corpus (GSGE_Corpus): Corpus object containing training/usage data.
        OHE_tokens_mask (np.ndarray): Binary mask for one-hot-encoded special/grammar/element tokens.
        meta_info (str): Optional metadata information.
        group_grammar (GroupGrammar): Group-Selfies grammar object or extracting groups etc.

    Methods:
        set_GS_vocab(GS_vocab):
            Loads or assigns the GS vocabulary and initializes group grammar.

        set_GSGE_corpus(GSGE_corpus):
            Loads or assigns the GSGE corpus.

        get_GSGE_vocab_token_to_id_dict(special_tokens, grammar_tokens, element_tokens, return_OHE_tokens_mask):
            Generates a token-to-ID dictionary and optional OHE mask.

        get_GSGE_vocab(special_tokens, grammar_tokens, element_tokens):
            Returns the full ordered vocabulary list for GSGE.
    """

    def __init__(
        self,
        GSGE_vocab: Optional[Dict[str, int]] = None,
        GS_vocab: Optional[Union["GS_Vocab", str]] = None,
        GSGE_corpus: Optional[Union["GSGE_Corpus", str]] = None
    ):
        """
        Initialize VocabularyManager with vocabularies and corpus.

        Args:
            GSGE_vocab: Token-to-ID mapping dictionary. If None, will be generated
                via get_GSGE_vocab_token_to_id_dict().
            GS_vocab: GS_Vocab instance or path to saved vocab file (.pkl).
            GSGE_corpus: GSGE_Corpus instance or path to saved corpus file (.pkl).
        """
        self.GSGE_corpus = GSGE_corpus
        self.GS_vocab = GS_vocab
        self.GSGE_vocab = GSGE_vocab
        self.OHE_tokens_mask = None
        self.meta_info = None

        self.set_GS_vocab(GS_vocab)
        self.set_GSGE_corpus(GSGE_corpus)


    def set_GS_vocab(self, GS_vocab: Optional[GS_Vocab | str]) -> None:
        """
        Set or load GS vocabulary and initialize group grammar.

        Args:
            GS_vocab: Either GS_Vocab instance, path to .pkl file, or None.

        Raises:
            ValueError: If GS_vocab is not a valid type.
        """
        if isinstance(GS_vocab, GS_Vocab):
            self.GS_vocab = GS_vocab
        elif isinstance(GS_vocab, str):
            self.GS_vocab = GS_Vocab()
            self.GS_vocab.load_GS_vocab(GS_vocab)
        elif GS_vocab is None:
            self.GS_vocab = None
        else:
            raise ValueError(f'Invalid GS_vocab type: {type(GS_vocab)}')
        self.set_group_grammar()

    def set_group_grammar(self):
        """Initialize GroupGrammar with vocabulary fragments if available."""
        if self.GS_vocab is not None:  # if loaded in after initialization of the GSGE_tokenizer
            self.group_grammar = GroupGrammar(vocab=self.GS_vocab.vocab_fragments)
        else:
            self.group_grammar = GroupGrammar(vocab=None)

    def set_GSGE_corpus(self, GSGE_corpus: Optional[GSGE_Corpus | str]) -> None:
        """
        Set or load GSGE corpus for training data.

        Args:
            GSGE_corpus: Either GSGE_Corpus instance, path to .pkl file, or None.

        Raises:
            ValueError: If GSGE_corpus is not a valid type.
        """
        if isinstance(GSGE_corpus, GSGE_Corpus):
            self.GSGE_corpus = GSGE_corpus
        elif isinstance(GSGE_corpus, str):
            self.GSGE_corpus = GSGE_Corpus()
            self.GSGE_corpus.load_GSGE_corpus(GSGE_corpus)
        elif GSGE_corpus is None:
            self.GSGE_corpus = None
        else:
            raise ValueError(f'Invalid GSGE_corpus type: {type(GSGE_corpus)}')

    def get_GSGE_vocab_token_to_id_dict(
        self,
        special_tokens: List[str] = SPECIAL_TOKENS,
        grammar_tokens: List[str] = _GRAMMAR_TOKENS,
        element_tokens: List[str] = _ELEMENT_TOKENS,
        return_OHE_tokens_mask: bool = False,
    ) -> Dict[str, int]:
        """
        Generate token-to-ID mapping dictionary for complete GSGE vocabulary.

        Creates bidirectional mapping between tokens and IDs, and generates
        binary mask indicating which tokens should use one-hot encoding
        (special, grammar, and element tokens) vs learned embeddings (fragments).

        Args:
            special_tokens: Special tokens ([PAD], [UNK], etc.). Default is SPECIAL_TOKENS.
            grammar_tokens: Group-SELFIES grammar tokens. Default is _GRAMMAR_TOKENS.
            element_tokens: Elemental tokens (C, O, N, etc.). Default is _ELEMENT_TOKENS.
            return_OHE_tokens_mask: If True, return both vocab dict and OHE mask.
                Default is False.

        Returns:
            GSGE_vocab dictionary mapping tokens to integer IDs if return_OHE_tokens_mask=False,
            otherwise tuple of (GSGE_vocab, OHE_tokens_mask).

        Note:
            Token ID assignment order:
            1. Special tokens (e.g., [PAD], [UNK])
            2. Grammar tokens (e.g., [Branch1], [Ring1])
            3. Element tokens (e.g., C, O, N)
            4. Fragment tokens (e.g., GS_frag_0, GS_frag_1, ...)

            Sets three attributes:
            - self.GSGE_vocab: token -> ID mapping
            - self.GSGE_vocab_reverse: ID -> token mapping
            - self.OHE_tokens_mask: binary mask (1 for OHE tokens, 0 for fragment tokens)
        """
        vocab_list = self.get_GSGE_vocab(special_tokens, grammar_tokens, element_tokens)
        self.GSGE_vocab = {token: i for i, token in enumerate(vocab_list)}
        self.GSGE_vocab_reverse = {i: token for i, token in enumerate(vocab_list)}
        self.OHE_tokens_mask = np.array([
            1 if token in special_tokens or token in grammar_tokens or token in element_tokens else 0
            for token in vocab_list
        ])

        if return_OHE_tokens_mask is True:
                return self.GSGE_vocab, self.OHE_tokens_mask

        return self.GSGE_vocab

    def get_GSGE_vocab(
        self,
        special_tokens: List[str] = SPECIAL_TOKENS,
        grammar_tokens: List[str] = _GRAMMAR_TOKENS,
        element_tokens: List[str] = _ELEMENT_TOKENS
    ) -> List[str]:
        """
        Return complete ordered vocabulary list for GSGE.

        Args:
            special_tokens: Special tokens to include. Default is SPECIAL_TOKENS.
            grammar_tokens: Grammar tokens to include. Default is _GRAMMAR_TOKENS.
            element_tokens: Element tokens to include. Default is _ELEMENT_TOKENS.

        Returns:
            Ordered list of all vocabulary tokens: [special + grammar + element + fragments].
        """
        return special_tokens + grammar_tokens + element_tokens + list(self.GS_vocab.vocab_fragments.keys())

class EmbeddingManager:
    
    """
    Manages the creation, storage, and retrieval of fragment embeddings for the GSGE model.

    Attributes:
        vocab_manager (VocabularyManager): Provides vocab and fragment information.
        store_modules (Store_Modules): Expected to contatin a (encoder) neural encoder (e.g., GNN) used for generating fragment embeddings.
        GS_frag_id_to_embedding (dict): Maps GS fragment IDs to their embedding vectors.
        GSGE_id_to_embedding (dict): Maps GSGE token IDs to embedding vectors.
        GSGE_combined_embeddings (np.ndarray): Final embedding matrix including one-hot and learned embeddings.
        GSGE_GAE_embeddings (np.ndarray): Embeddings generated by the Graph AutoEncoder (GAE).

    Methods:
        make_GS_fragment_embedding_dict(...):
            Generates and stores embeddings for grammar structure fragments using the encoder.

        embed_fragments(...):
            Encodes a list of SMILES strings into fragment embeddings using the encoder.

        load_GAE_weights(load_checkpoint_path, map_location):
            Loads pretrained encoder weights from a saved checkpoint.

        get_fragment_embeddings():
            Returns the final GAE-based embeddings used in GSGE.
    """

    def __init__(self,
        vocab_manager: VocabularyManager,
        store_modules: 'Store_Modules' = None
        ):
        """
        Initialize EmbeddingManager.

        Args:
            vocab_manager: VocabularyManager instance providing vocabulary and fragments.
            store_modules: Store_Modules instance containing encoder/decoder models.
        """

        self.vocab_manager = vocab_manager
        self.store_modules = store_modules
        self.GS_frag_id_to_embedding = None
        self.GSGE_id_to_embedding = None
        self.GSGE_combined_embeddings = None
        self.GSGE_GAE_embeddings = None

    def make_GS_fragment_embedding_dict(
        self,
        load_checkpoint_path: Optional[str] = None,
        map_location: str = 'cuda',
        max_atom_size: int = ATOM_MAX_NUM,
        process_smiles: callable = from_smiles,
        atom_to_token_id: Dict = atom_to_token_id,
        device: str = 'cuda',
        batch_size: int = 64
    ) -> None:
        """
        Generate and store embeddings for all GS vocabulary fragments.

        Encodes all vocabulary fragments using trained GAE encoder and creates
        mapping dictionaries from fragment IDs to embeddings. Combines one-hot
        encodings for special/grammar/element tokens with learned embeddings.

        Args:
            load_checkpoint_path: Path to saved encoder checkpoint. If None, uses
                current encoder weights. Default is None.
            map_location: Device for loading checkpoint ('cuda' or 'cpu'). Default is 'cuda'.
            max_atom_size: Maximum number of atoms per fragment. Default is ATOM_MAX_NUM (20).
            process_smiles: Function to convert SMILES to PyG Data. Default is from_smiles.
            atom_to_token_id: Mapping from atom types to token IDs. Default is atom_to_token_id.
            device: Device for inference ('cuda' or 'cpu'). Default is 'cuda'.
            batch_size: Batch size for encoding fragments. Default is 64.

        Raises:
            ValueError: If encoder not set before calling this method.

        Note:
            Creates three embedding mappings:
            - GS_frag_id_to_embedding: Maps GS_frag_N IDs to embeddings
            - GSGE_id_to_embedding: Maps GSGE vocab token IDs to embeddings
            - GSGE_combined_embeddings: Full embedding matrix (OHE + learned)
            - GSGE_GAE_embeddings: Pure GAE embeddings for fragments only
        """
        if self.store_modules.modules['encoder'] is None:
            raise ValueError("Encoder must be set before creating embeddings")

        # Ensure GSGE_vocab (token-to-ID dictionary) is generated
        if self.vocab_manager.GSGE_vocab is None:
            self.vocab_manager.get_GSGE_vocab_token_to_id_dict()

        # Ensure OHE_tokens_mask is generated
        if self.vocab_manager.OHE_tokens_mask is None:
            self.vocab_manager.get_GSGE_vocab_token_to_id_dict(return_OHE_tokens_mask=True)

        embeddings, _ = self.embed_fragments(
            frag_smiles=self.vocab_manager.GS_vocab.fragments,
            load_checkpoint_path=load_checkpoint_path,
            map_location=map_location,
            max_atom_size=max_atom_size,
            process_smiles=process_smiles,
            atom_to_token_id=atom_to_token_id,
            device=device,
            batch_size=batch_size,
            return_data=True
        )
        self.GS_frag_id_to_embedding = {
            gs_frag: embeddings[i]
            for i, gs_frag in enumerate(self.vocab_manager.GS_vocab.frag_id_to_noncanonical.keys())
        }
        self.GSGE_id_to_embedding = {
            self.vocab_manager.GSGE_vocab[gsge_id]: emb
            for gsge_id, emb in self.GS_frag_id_to_embedding.items()
        }
        self.GSGE_combined_embeddings, self.GSGE_id_to_embedding, self.GSGE_GAE_embeddings = \
            CoreGSGE.create_combined_embeddings(
                sum(self.vocab_manager.OHE_tokens_mask),
                self.GSGE_id_to_embedding,
                embedding_dim=embeddings[0].__len__()
            )

    def embed_fragments(
        self,
        frag_smiles: List[str],
        load_checkpoint_path: Optional[str] = None,
        map_location: str = 'cuda',
        max_atom_size: int = ATOM_MAX_NUM,
        process_smiles: callable = from_smiles,
        atom_to_token_id: Dict = atom_to_token_id,
        device: str = 'cuda',
        batch_size: int = 64,
        return_data: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, List[Data]]:
        """
        Encode list of SMILES fragments into latent embeddings.

        Converts SMILES to graph representations and encodes them using
        trained GAE encoder to produce dense vector embeddings.

        Args:
            frag_smiles: List of fragment SMILES strings to encode.
            load_checkpoint_path: Path to encoder checkpoint. If provided, loads
                weights before encoding. Default is None.
            map_location: Device for loading checkpoint ('cuda' or 'cpu'). Default is 'cuda'.
            max_atom_size: Maximum atoms per fragment for padding. Default is ATOM_MAX_NUM (20).
            process_smiles: Function converting SMILES to PyG Data. Default is from_smiles.
            atom_to_token_id: Atom type to token ID mapping. Default is atom_to_token_id.
            device: Device for inference ('cuda' or 'cpu'). Default is 'cuda'.
            batch_size: Batch size for encoding. Default is 64.
            return_data: If True, return both embeddings and graph data.
                Default is False.

        Returns:
            If return_data=False: NumPy array of embeddings [n_fragments, embedding_dim].
            If return_data=True: Tuple of (embeddings, graph_data_list).

        Example:
            >>> frags = ['CC(*)O', 'c1ccccc1*', 'C(*)C(*)*']
            >>> embeddings = embedding_manager.embed_fragments(frags, device='cpu')
            >>> embeddings.shape
            (3, 128)  # 3 fragments, 128-dim embeddings
        """
        if load_checkpoint_path:
            self.load_GAE_weights(load_checkpoint_path, map_location)
        self.store_modules.modules['encoder'] = self.store_modules.modules['encoder'].to(device)
        embeddings, graph_data = CoreGSGE.embed_fragments(
            frag_smiles=frag_smiles,
            encoder=self.store_modules.modules['encoder'],
            max_atom_size=max_atom_size,
            process_smiles=process_smiles,
            atom_to_token_id=atom_to_token_id,
            return_data=True,
            device=device,
            batch_size=batch_size
        )
        return (embeddings, graph_data) if return_data else embeddings

    def load_GAE_weights(self, load_checkpoint_path: str, map_location: str = 'cuda') -> None:
        """
        Load pretrained encoder weights from checkpoint file.

        Args:
            load_checkpoint_path: Path to checkpoint file saved during training.
            map_location: Device to load weights onto ('cuda' or 'cpu'). Default is 'cuda'.

        Example:
            >>> embedding_manager.load_GAE_weights('checkpoints/checkpoint_epoch_300.pth')
            Encoder weights loaded
        """
        checkpoint = torch.load(load_checkpoint_path, map_location=map_location)
        if self.store_modules.modules['encoder']:
            self.store_modules.modules['encoder'].load_state_dict(checkpoint['encoder_state_dict'])
            print('Encoder weights loaded')

    def get_fragment_embeddings(self) -> Optional[np.ndarray]:
        """
        Return GAE-generated fragment embeddings.

        Returns:
            NumPy array of fragment embeddings [n_fragments, embedding_dim],
            or None if embeddings haven't been generated yet.
        """
        return self.GSGE_GAE_embeddings

    def export_embeddings_to_csv(self, filename: str = 'fragment_embeddings.csv') -> None:
        """
        Export fragment embeddings to CSV file.

        Writes fragment IDs and their corresponding embedding dimension values to CSV format.
        First column contains fragment IDs (GS_frag_N), subsequent columns contain
        embedding dimension values (dim_0, dim_1, ..., dim_N). Includes header row.

        Args:
            filename: Output CSV filename. Default is 'fragment_embeddings.csv'.

        Raises:
            ValueError: If embeddings haven't been generated yet.

        Example:
            >>> embedding_manager.make_GS_fragment_embedding_dict('checkpoint.pth')
            >>> embedding_manager.export_embeddings_to_csv('embeddings.csv')
            Embeddings exported to embeddings.csv
        """
        if self.GSGE_GAE_embeddings is None:
            raise ValueError("Embeddings must be generated before export. Call make_GS_fragment_embedding_dict() first.")

        import csv

        # Get fragment IDs in the same order as embeddings
        fragment_ids = list(self.vocab_manager.GS_vocab.frag_id_to_noncanonical.keys())

        # Get embedding dimension for header creation
        embedding_dim = self.GSGE_GAE_embeddings.shape[1]

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write header row
            header = ['fragment_id'] + [f'dim_{i}' for i in range(embedding_dim)]
            writer.writerow(header)

            # Write data rows
            for i, frag_id in enumerate(fragment_ids):
                row = [frag_id] + self.GSGE_GAE_embeddings[i].tolist()
                writer.writerow(row)

class GAETrainer:

    """
    Trains a Graph AutoEncoder (GAE) model for learning fragment embeddings from molecular graph data.

    Attributes:
        vocab_manager (VocabularyManager): Provides access to GS vocabulary and GSGE corpus.
        encoder (AttentiveFP): Graph neural network encoder for molecular fragments.
        decoder (GraphDecoder): Decoder network reconstructing molecular graphs from latent vectors.

    Methods:
        train_GSGE_Auto_Encoder(...):
            Trains the GAE model using the provided vocabulary and corpus, with support for checkpoints and validation.

        set_encoder(encoder):
            Manually sets a custom encoder instance.

        set_decoder(decoder):
            Manually sets a custom decoder instance.
    """

    def __init__(
        self,
        vocab_manager: VocabularyManager,
        store_modules: 'Store_Modules',
    ):
        """
        Initialize GAETrainer.

        Args:
            vocab_manager: VocabularyManager providing GS_vocab and GSGE_corpus.
            store_modules: Store_Modules instance containing encoder/decoder models.
        """
        self.vocab_manager = vocab_manager
        self.store_modules = store_modules

    def train_GSGE_Auto_Encoder(
        self,
        batch_size: int = 64,
        num_epochs: int = 300,
        checkpoint_interval: int = 5,
        val_percentage: float = 0.2,
        split_seed: int = 42,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = "./model_checkpoints",
        load_checkpoint_path: Optional[str] = None
    ) -> None:
        """
        Train Graph Autoencoder for learning fragment embeddings.

        Trains encoder-decoder model to learn latent representations of molecular
        fragments from GSGE_corpus. Includes automatic validation and checkpointing.

        Args:
            batch_size: Number of fragments per training batch. Default is 64.
            num_epochs: Total training epochs. Default is 300.
            checkpoint_interval: Save checkpoint every N epochs. Default is 5.
            val_percentage: Fraction of data for validation (0.0-1.0). Default is 0.2.
            split_seed: Random seed for train/val split. Default is 42.
            learning_rate: Adam optimizer learning rate. Default is 0.001.
            device: Training device ('cuda' or 'cpu'). Default is 'cuda' if available.
            checkpoint_dir: Directory for saving checkpoints. Default is './model_checkpoints'.
            load_checkpoint_path: Path to resume from checkpoint. If None, starts fresh.
                Default is None.

        Example:
            >>> gae_trainer.train_GSGE_Auto_Encoder(
            ...     num_epochs=300,
            ...     batch_size=64,
            ...     checkpoint_interval=10,
            ...     device='cuda'
            ... )
            Epoch [1/300] ------------------------------------- Train
              Train Atom Loss: 2.1234, ...
        """
        optimizer = torch.optim.Adam(
            list(self.store_modules.modules['encoder'].parameters()) +
            list(self.store_modules.modules['decoder'].parameters()),
            lr=learning_rate
        )
        CoreGSGE.train_GSGE_Auto_Encoder(
            GS_vocab=self.vocab_manager.GS_vocab,
            GSGE_corpus=self.vocab_manager.GSGE_corpus,
            batch_size=batch_size,
            num_epochs=num_epochs,
            checkpoint_interval=checkpoint_interval,
            val_percentage=val_percentage,
            split_seed=split_seed,
            device=device,
            encoder=self.store_modules.modules['encoder'],
            decoder=self.store_modules.modules['decoder'],
            optimizer=optimizer,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint_path=load_checkpoint_path
        )

    def set_encoder(self, encoder: nn.Module =
            AttentiveFP(in_channels=9, hidden_channels=256, out_channels=128, edge_dim=3, num_layers=3, num_timesteps=2)
        ) -> None:
        """
        Set custom encoder model for GAE.

        Args:
            encoder: PyTorch encoder module. Default is AttentiveFP with standard architecture.

        Example:
            >>> from torch_geometric.nn.models.attentive_fp import AttentiveFP
            >>> custom_encoder = AttentiveFP(in_channels=9, hidden_channels=512,
            ...                               out_channels=256, edge_dim=3)
            >>> gae_trainer.set_encoder(custom_encoder)
        """

        self.store_modules.modules['encoder'] = encoder

    def set_decoder(self, decoder: nn.Module =
            GraphDecoder(latent_dim=128, hidden_dim=256)
        ) -> None:
        """
        Set custom decoder model for GAE.

        Args:
            decoder: PyTorch decoder module. Default is GraphDecoder with standard architecture.

        Example:
            >>> custom_decoder = GraphDecoder(latent_dim=256, hidden_dim=512)
            >>> gae_trainer.set_decoder(custom_decoder)
        """

        self.store_modules.modules['decoder'] = decoder

class DescriptorCalculator:

    """
    Calculates and stores molecular fragment descriptors used in the GSGE framework.

    Attributes:
        vocab_manager (VocabularyManager): Provides access to molecular fragments for descriptor computation.
        GSGE_fragment_descriptors (torch.Tensor): Normalized descriptors for all fragments.
        fragment_descriptors_keys (List[str]): Names of computed descriptor types.

    Methods:
        calc_fragment_descriptors(...):
            Computes and normalizes molecular descriptors for fragments using specified RDKit descriptor functions.

        get_fragment_descriptors():
            Returns the tensor of normalized fragment descriptors.

        get_fragment_descriptors_names():
            Returns the list of descriptor names used during calculation.
    """

    def __init__(self, vocab_manager: VocabularyManager):
        """
        Initialize DescriptorCalculator.

        Args:
            vocab_manager: VocabularyManager providing fragment vocabulary for descriptor calculation.
        """
        self.vocab_manager = vocab_manager
        self.GSGE_fragment_descriptors = None
        self.fragment_descriptors_keys = None

    def calc_fragment_descriptors(
        self,
        descriptor_calc_fns_list: List[callable] = Descriptors._descList,
        descriptor_keys: Optional[List[str]] = None,
        smiles_input: bool = False
    ) -> None:
        """
        Compute and normalize RDKit molecular descriptors for all vocabulary fragments.

        Calculates specified subset of RDKit descriptors for each fragment,
        normalizes via z-score, and stores as PyTorch tensor for efficient use
        in downstream models.

        Args:
            descriptor_calc_fns_list: List of (name, function) tuples for descriptor
                calculation. Default is RDKit's full descriptor list (Descriptors._descList).
            descriptor_keys: List of descriptor names to calculate. If None, uses
                default set of 48 descriptors covering molecular weight, topological
                indices, electronic properties. Default is None.
            smiles_input: If True, use SMILES strings instead of Mol objects for
                calculation. Default is False.

        Note:
            Default descriptor set includes 48 descriptors:
            - Molecular weight and electronic properties
            - Topological indices (Chi, Kappa)
            - Surface area descriptors (TPSA, VSA)
            - Structural features (rings, stereocenters, rotatable bonds)

            Normalizes descriptors using z-score: (x - mean) / std
            Filters out low-variance descriptors (var < 1e-6)
        """
        self.fragment_descriptors_keys = descriptor_keys or [
            'MaxEStateIndex', 'MinEStateIndex', 'MolWt', 'NumValenceElectrons', 'FpDensityMorgan1',
            'AvgIpc', 'BalabanJ', 'BertzCT', 'Chi0', 'Chi0n', 'Chi0v', 'Chi1', 'Chi1n', 'Chi1v',
            'Chi2n', 'Chi2v', 'Chi3n', 'Chi3v', 'Chi4n', 'Chi4v', 'HallKierAlpha', 'Ipc',
            'Kappa1', 'Kappa2', 'Kappa3', 'TPSA', 'EState_VSA7', 'EState_VSA8', 'VSA_EState4',
            'VSA_EState7', 'VSA_EState8', 'FractionCSP3', 'HeavyAtomCount', 'NHOHCount', 'NOCount',
            'NumAmideBonds', 'NumAtomStereoCenters', 'NumHAcceptors', 'NumHDonors', 'NumHeteroatoms',
            'NumHeterocycles', 'NumRotatableBonds', 'NumUnspecifiedAtomStereoCenters', 'Phi',
            'RingCount', 'MolLogP', 'MolMR'
        ]
        raw_descriptors = get_mol_frag_descriptors(
            self.vocab_manager,
            calc_mol_frag_descriptors_args={
                'descriptor_keys': self.fragment_descriptors_keys,
                'descriptor_calc_fns_list': descriptor_calc_fns_list
            },
            smiles_input=smiles_input
        )
        normalized_descriptors, _, _, _ = normalize_descriptors(raw_descriptors)
        self.GSGE_fragment_descriptors = torch.tensor(normalized_descriptors)

    def get_fragment_descriptors(self) -> Optional[torch.Tensor]:
        """
        Return normalized fragment descriptors as PyTorch tensor.

        Returns:
            Tensor of shape [n_fragments, n_descriptors] with z-score normalized
            descriptors, or None if descriptors haven't been calculated yet.
        """
        return self.GSGE_fragment_descriptors

    def get_fragment_descriptors_names(self) -> Optional[List[str]]:
        """
        Return list of descriptor names used in calculation.

        Returns:
            List of descriptor name strings (e.g., ['MolWt', 'TPSA', ...]),
            or None if descriptors haven't been calculated yet.
        """
        return self.fragment_descriptors_keys

    def export_descriptors_to_csv(self, filename: str = 'fragment_descriptors.csv') -> None:
        """
        Export fragment descriptors to CSV file.

        Writes fragment IDs and their corresponding descriptor values to CSV format.
        First column contains fragment IDs (GS_frag_N), subsequent columns contain
        descriptor values. Includes header row with descriptor names.

        Args:
            filename: Output CSV filename. Default is 'fragment_descriptors.csv'.

        Raises:
            ValueError: If descriptors haven't been calculated yet.

        Example:
            >>> descriptor_calculator.calc_fragment_descriptors()
            >>> descriptor_calculator.export_descriptors_to_csv('descriptors.csv')
            Descriptors exported to descriptors.csv
        """
        if self.GSGE_fragment_descriptors is None:
            raise ValueError("Descriptors must be calculated before export. Call calc_fragment_descriptors() first.")

        import csv

        # Get fragment IDs in the same order as descriptors
        fragment_ids = list(self.vocab_manager.GS_vocab.frag_id_to_noncanonical.keys())

        # Convert tensor to numpy for easier handling
        descriptors_array = self.GSGE_fragment_descriptors.numpy()

        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)

            # Write header row
            header = ['fragment_id'] + self.fragment_descriptors_keys
            writer.writerow(header)

            # Write data rows
            for i, frag_id in enumerate(fragment_ids):
                row = [frag_id] + descriptors_array[i].tolist()
                writer.writerow(row)

class GraphProcessor:

    """
    Processes SMILES strings into graph-based representations used in GSGE.

    Attributes:
        vocab_manager (VocabularyManager): Provides vocabulary and grammar needed for graph construction.
        tokenizer (GSGE_tokenizer): Tokenizer for handling special and grammar tokens.

    Methods:
        01_make_compound_graphs(smiles_list, workers, pyg_data):
            Converts a list of SMILES strings into graph data, optionally in PyTorch Geometric format.

        check_for_graphs_groupings(smiles, workers):
            Checks if compounds in the input list can be grouped correctly based on GS vocabulary.

        get_CG_from_smiles(smiles, return_CG_object):
            Generates a compound graph (CG) object from a single SMILES string.
    """
    
    def __init__(self, vocab_manager: VocabularyManager, tokenizer:'GSGE_tokenizer'):
        """
        Initialize GraphProcessor.

        Args:
            vocab_manager: VocabularyManager providing vocabulary and group grammar.
            tokenizer: GSGE_tokenizer for handling special/grammar tokens.
        """
        self.vocab_manager = vocab_manager
        self.tokenizer = tokenizer

    def make_compound_graphs(
        self,
        smiles_list: List[str],
        workers: int = 8,
        pyg_data: bool = True
    ) -> List[Data] | List[tuple]:
        """
        Convert SMILES list to compound graph representations.

        Processes molecules in parallel, representing each as graph where nodes
        are molecular fragments (from vocabulary) and edges connect bonded fragments.

        Args:
            smiles_list: List of SMILES strings to convert.
            workers: Number of parallel workers for processing. Default is 8.
            pyg_data: If True, return PyTorch Geometric Data objects. If False,
                return raw (edge_index, node_features) tuples. Default is True.

        Returns:
            If pyg_data=True: List of PyG Data objects with x and edge_index attributes.
            If pyg_data=False: List of (edge_index, node_features) tuples.

        Example:
            >>> smiles_list = ['CCO', 'CC(C)O', 'c1ccccc1']
            >>> graphs = graph_processor.make_compound_graphs(smiles_list)
            >>> graphs[0].x.shape
            torch.Size([3, embedding_dim])  # 3 fragment nodes
        """
        process_func = partial(preprocess_graph, group_grammar=self.vocab_manager.group_grammar)
        data = parallel_(smiles_list, process_func=process_func, max_workers=workers)
        if pyg_data:
            return [
                Data(
                    x=torch.tensor(node_features, dtype=torch.long).unsqueeze(1),  # shape [num_nodes, 1]
                    edge_index=torch.tensor(edge_index, dtype=torch.long)          # shape [2, num_edges]
                )
                for edge_index, node_features in data
            ]
        return data

    def check_for_graphs_groupings(
            self,
            smiles: List[str],
            workers: int = 8
        ) -> List[tuple]:
        """
        Check which molecules have atoms not covered by vocabulary fragments.

        Identifies molecules that cannot be fully grouped into vocabulary fragments,
        useful for vocabulary coverage validation.

        Args:
            smiles: List of SMILES strings to check.
            workers: Number of parallel workers. Default is 8.

        Returns:
            List of (index, non_grouped_atoms, smiles) tuples for molecules with
            ungrouped atoms. Empty list if all molecules fully grouped.

        Example:
            >>> problematic = graph_processor.check_for_graphs_groupings(smiles_list)
            >>> if problematic:
            ...     print(f"Molecule {problematic[0][0]} has ungrouped atoms: {problematic[0][1]}")
        """
        process_func = partial(check_single_cg, vocab=self.vocab_manager.GS_vocab)
        results_raw = parallel_(smiles, process_func=process_func, max_workers=workers)
        return [(i, res, smiles[i]) for i, res in enumerate(results_raw) if res is not None]

    def get_CG_from_smiles(
        self,
        smiles: str,
        return_CG_object: bool = True
    ) -> object:
        """
        Generate compound graph from single SMILES string.

        Args:
            smiles: SMILES string representing molecule.
            return_CG_object: If True, return compound_graph object. If False,
                return raw graph data. Default is True.

        Returns:
            compound_graph object (if return_CG_object=True) or raw graph data.

        Example:
            >>> cg = graph_processor.get_CG_from_smiles('CCO', return_CG_object=True)
            >>> cg.plot_graph_rd_c_style()  # Visualize fragment graph
        """
        return smiles_to_group_graph(smiles, self.vocab_manager.group_grammar, return_CG_object=return_CG_object)

def check_single_cg(smi_, vocab):
    """
    Check if single SMILES has atoms not covered by vocabulary (for multiprocessing).

    Helper function for parallel checking of molecule coverage.

    Args:
        smi_: SMILES string to check.
        vocab: GS_Vocab instance with fragment vocabulary.

    Returns:
        List of non-grouped atom indices if any exist, None otherwise.
    """
    result_ = CoreGSGE.check_for_non_grouped_atoms(
        smiles=smi_, GS_vocab=vocab
    )
    return result_ if result_ else None

class ClusteringAnalyzer:

    """
    Performs clustering analysis on molecular fragments using embeddings or structure-based methods.

    Attributes:
        vocab_manager (VocabularyManager): Provides vocabulary and structural data required for clustering.

    Methods:
        get_GSGE_clustering(...):
            Computes or uses provided cluster labels to perform clustering analysis and visualization.
            Supports structure-based clustering (e.g., MCS) or embedding-based approaches.
    """

    def __init__(self, gsge:'GSGE', vocab_manager: VocabularyManager):
        """
        Initialize ClusteringAnalyzer.

        Args:
            gsge: GSGE instance for accessing embeddings and configurations.
            vocab_manager: VocabularyManager providing vocabulary and fragment data.
        """
        self.gsge = gsge
        self.vocab_manager = vocab_manager

    def get_GSGE_clustering(
        self,
        embeddings: Optional[np.ndarray] = None,
        graph_data: Optional[List[Data]] = None,
        smiles_df: Optional[pd.DataFrame] = None,
        cluster_labels: Optional[List[int]] = None,
        smiles_column: str = 'SMILES',
        cluster_args: dict = {},
        cluster:bool=True,
    ) -> object:
        """
        Create GSGE_clustering object for fragment clustering analysis.

        Performs clustering using Maximum Common Substructure (MCS) or uses
        provided cluster labels. Supports embedding-based visualization (TSNE, UMAP).

        Args:
            embeddings: Fragment embeddings array [n_fragments, embedding_dim].
                If None, uses embeddings from GSGE instance. Default is None.
            graph_data: List of PyG Data objects representing fragments.
                If provided, extracts SMILES from graph data. Default is None.
            smiles_df: DataFrame containing SMILES strings for fragments.
                Required if graph_data not provided. Default is None.
            cluster_labels: Pre-computed cluster labels for fragments.
                If None, computes using MCS clustering. Default is None.
            smiles_column: Column name in smiles_df containing SMILES strings.
                Default is 'SMILES'.
            cluster_args: Keyword arguments passed to MCS clustering algorithm.
                Default is {}.
            cluster: If True, perform clustering. If False, use provided labels.
                Default is True.

        Returns:
            GSGE_clustering object with clustering results and visualization methods.

        Example:
            >>> clustering = clustering_analyzer.get_GSGE_clustering(
            ...     embeddings=fragment_embeddings,
            ...     smiles_df=fragment_df
            ... )
            >>> fig = clustering.plot_2D_TSNE()
            >>> fig.show()
        """
        if graph_data is not None:
            if smiles_df is not None:
                print('Warning: Both smiles_df and graph_data provided, using graph_data')
            smiles_df = pd.DataFrame({'SMILES': [frag_.smiles for frag_ in graph_data]})
        if not cluster_labels:
            if smiles_df is None:
                raise ValueError('smiles_df or graph_data required for clustering')
            cluster_labels = GSGE_clustering._MCS_clustering(
                smiles_df, **cluster_args
            )
        return GSGE_clustering(
            gsge=self.gsge,
            embeddings=embeddings,
            smiles_df=smiles_df,
            cluster_labels=cluster_labels,
            smiles_column=smiles_column
        )

class Store_Modules:
    """
    Storage container for PyTorch encoder/decoder modules.

    Simple dictionary wrapper for managing neural network modules used in GAE training.

    Attributes:
        modules (dict): Dictionary storing PyTorch modules (e.g., {'encoder': ..., 'decoder': ...}).
    """
    def __init__(self, modules={}):
        """
        Initialize Store_Modules with module dictionary.

        Args:
            modules: Dictionary mapping module names to PyTorch modules.
                Default is empty dict.
        """
        self.modules = modules    

class GSGE(GS_FragmentTools):

    """
    Grammar-SMILES-Group-Element (GSGE) framework for fragment-based molecular modeling and representation.

    This (facade) class serves as the main interface to manage the full GSGE pipeline: tokenization, vocabulary management,
    fragment embedding, graph-based processing, descriptor calculation, and clustering.

    Attributes:
        vocab_manager (VocabularyManager): Manages vocabularies and corpus for fragments.
        tokenizer (GSGE_tokenizer): Tokenizes SMILES or molecules into fragment-based GSGE tokens.
        embedding_manager (EmbeddingManager): Handles GAE embedding generation and loading.
        gae_trainer (GAETrainer): Trains the Graph AutoEncoder for fragment embedding.
        descriptor_calculator (DescriptorCalculator): Computes and stores fragment-level molecular descriptors.
        graph_processor (GraphProcessor): Converts molecules to graph representations.
        clustering_analyzer (ClusteringAnalyzer): Clusters molecular fragmetns using embeddings and MCS.
        store_data (Store_Data): Saves and loads model state and metadata.

    Core Methods:
        save_gsge_data(filepath, meta_info): Save GSGE model state to file.
        load_gsge_data(filepath): Load GSGE model state from file.
        encode_GSGE(token_ids): Encode token IDs into dense embeddings using current GSGE vocabulary.
        train_GSGE_Auto_Encoder(...): Train the GAE model for fragment embedding.
        embed_fragments(...): Generate fragment embeddings using the trained encoder.
        calc_fragment_descriptors(...): Compute normalized descriptors for each fragment.
        make_compound_graphs(...): Convert SMILES to graph representations.
        get_GSGE_clustering(...): Perform clustering using MCS or embeddings.
        add_standard_smaller_fragments(fragments): Add predefined common fragments to GS_vocab.
        add_all_single_elements(...): Add single elements to both GS_vocab and GSGE_corpus.
        preprocess_from_SMILES(smiles): Tokenize a SMILES string to GSGE format.
        parallel_tokenize_SMILES_list(smiles_list): Batch tokenize a list of SMILES.
        get_fragment_embeddings(): Return GAE-generated fragment embeddings.
        get_fragment_descriptors(): Return RDKit-calculated descriptors for fragments.
        get_fragment_descriptors_and_embeddings(): Concatenate descriptors and embeddings.
        plot_GS_fragments_in_mol(smiles, args): Visualize tokenized fragments on molecule image.

    Accessors:
        get_GS_vocab(): Return GS_Vocab instance.
        get_GSGE_corpus(): Return GSGE_Corpus instance.
        get_GSGE_vocab(): Return token-to-ID dictionary of the GSGE vocabulary.
        get_group_grammar(): Return the GroupGrammar used for parsing fragments.
        get_smiles_from_GS_frag_id(frag_id): Get canonical SMILES from a GS fragment ID.
    """

    def __init__(
        self,
        GSGE_load_path: Optional[str] = None,
        GSGE_vocab: Optional[Dict[str, int]] = None,
        GS_vocab: Optional[GS_Vocab | str] = None,
        GSGE_corpus: Optional[GSGE_Corpus | str] = None,
        grammar_tokens: List[str] = _GRAMMAR_TOKENS,
        element_tokens: List[str] = _ELEMENT_TOKENS,
        redirect_tokens: List[str] = _REDIRECT_TOKENS,
        special_tokens: List[str] = SPECIAL_TOKENS,
        frag_regex_pattern: str = r"([^A-Za-z]*)(GS.*)",
        encoder: Optional[object] = None,
        decoder: Optional[object] = None,
        load_checkpoint_path: Optional[str] = None,
        make_dense_embeddings: bool = False
    ):
        """
        Initialize GSGE framework for fragment-based molecular modeling.

        Creates complete GSGE pipeline including vocabulary management, tokenization,
        embedding generation, graph processing, and clustering capabilities.

        Args:
            GSGE_load_path: Path to saved GSGE state (.pkl file). If provided, loads
                all components during initialization. Default is None.
            GSGE_vocab: Pre-built token-to-ID mapping dictionary. If None, will be
                generated from GS_vocab. Default is None.
            GS_vocab: GS_Vocab instance or path to vocab file. Merged vocabulary for
                molecule representation. Default is None.
            GSGE_corpus: GSGE_Corpus instance or path to corpus file. Non-merged
                fragments for GAE training. Default is None.
            grammar_tokens: Group-SELFIES grammar tokens. Default is _GRAMMAR_TOKENS.
            element_tokens: Element tokens (C, O, N, etc.). Default is _ELEMENT_TOKENS.
            redirect_tokens: Redirect tokens for SELFIES. Default is _REDIRECT_TOKENS.
            special_tokens: Special tokens ([PAD], [UNK], etc.). Default is SPECIAL_TOKENS.
            frag_regex_pattern: Regex pattern for fragment token detection.
                Default is r"([^A-Za-z]*)(GS.*)".
            encoder: PyTorch encoder model (e.g., AttentiveFP). If None, must be set
                before training/embedding. Default is None.
            decoder: PyTorch decoder model (e.g., GraphDecoder). If None, must be set
                before training. Default is None.
            load_checkpoint_path: Path to encoder checkpoint for loading pretrained
                weights. Default is None.
            make_dense_embeddings: If True, automatically generate embeddings for all
                vocabulary fragments after initialization. Requires encoder and GS_vocab.
                Default is False.

        Example:
            Load pre-trained GSGE:

            >>> gsge = GSGE(GSGE_load_path='gsge_save.pkl')
            >>> print(len(gsge.get_fragments_smiles()))
            200  # Vocabulary loaded

            Create new GSGE with custom vocabulary:

            >>> from torch_geometric.nn.models.attentive_fp import AttentiveFP
            >>> encoder = AttentiveFP(in_channels=9, hidden_channels=256,
            ...                       out_channels=128, edge_dim=3)
            >>> gsge = GSGE(
            ...     GS_vocab='vocab.pkl',
            ...     GSGE_corpus='corpus.pkl',
            ...     encoder=encoder,
            ...     make_dense_embeddings=True,
            ...     load_checkpoint_path='checkpoint_epoch_300.pth'
            ... )
        """
        super().__init__()
        self.vocab_manager = VocabularyManager(GSGE_vocab, GS_vocab, GSGE_corpus)
        self.tokenizer = GSGE_tokenizer(
            vocab_manager=self.vocab_manager,
            grammer_tokens=grammar_tokens,
            element_tokens=element_tokens,
            redirect_tokens=redirect_tokens,
            special_tokens=special_tokens,
            frag_regex_pattern=frag_regex_pattern,
        )
        self.store_modules = Store_Modules({'encoder':encoder, 'decoder':decoder})
        self.embedding_manager = EmbeddingManager(self.vocab_manager, self.store_modules)
        self.gae_trainer = GAETrainer(self.vocab_manager, self.store_modules)
        self.descriptor_calculator = DescriptorCalculator(self.vocab_manager)
        self.graph_processor = GraphProcessor(self.vocab_manager, self.tokenizer)
        self.clustering_analyzer = ClusteringAnalyzer(self, self.vocab_manager)
        self.store_data = Store_Data(self, GSGE_load_path)

        if load_checkpoint_path:
            self.embedding_manager.load_GAE_weights(load_checkpoint_path=load_checkpoint_path, map_location='cuda')
        if make_dense_embeddings and encoder and self.vocab_manager.GS_vocab:
            if not load_checkpoint_path:
                print('Warning: No load_checkpoint_path provided, using current encoder weights')
            self.embedding_manager.make_GS_fragment_embedding_dict(
                load_checkpoint_path=load_checkpoint_path,
                map_location='cuda',
                max_atom_size=ATOM_MAX_NUM,
                process_smiles=from_smiles,
                atom_to_token_id=atom_to_token_id,
                device='cuda',
                batch_size=64
            )

    def save_gsge_data(self, filepath: str, meta_info: str = '') -> None:
        self.store_data.save_gsge_data(filepath=filepath, meta_info=meta_info)
    
    def load_gsge_data(self, filepath: str) -> None:
        self.store_data.load_gsge_data(filepath)

    def encode_GSGE(
        self,
        token_ids: np.ndarray,
        dtype: type = np.float32,
        device: str = 'cpu'
    ) -> np.ndarray:
        return CoreGSGE.encode_GSGE(
            token_ids, self.embedding_manager.GSGE_combined_embeddings.to(device), device=device
        )

    def train_GSGE_Auto_Encoder(
        self,
        batch_size: int = 64,
        num_epochs: int = 300,
        checkpoint_interval: int = 5,
        val_percentage: float = 0.2,
        split_seed: int = 42,
        learning_rate: float = 0.001,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        checkpoint_dir: str = "./model_checkpoints",
        load_checkpoint_path: Optional[str] = None
    ) -> None:
        self.gae_trainer.train_GSGE_Auto_Encoder(
            batch_size=batch_size,
            num_epochs=num_epochs,
            checkpoint_interval=checkpoint_interval,
            val_percentage=val_percentage,
            split_seed=split_seed,
            learning_rate=learning_rate,
            device=device,
            checkpoint_dir=checkpoint_dir,
            load_checkpoint_path=load_checkpoint_path
        )

    def embed_fragments(
        self,
        frag_smiles: List[str],
        load_checkpoint_path: Optional[str] = None,
        map_location: str = 'cuda',
        max_atom_size: int = ATOM_MAX_NUM,
        process_smiles: callable = from_smiles,
        atom_to_token_id: Dict = atom_to_token_id,
        device: str = 'cuda',
        batch_size: int = 64,
        return_data: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, List[Data]]:
        return self.embedding_manager.embed_fragments(
            frag_smiles=frag_smiles,
            load_checkpoint_path=load_checkpoint_path,
            map_location=map_location,
            max_atom_size=max_atom_size,
            process_smiles=process_smiles,
            atom_to_token_id=atom_to_token_id,
            device=device,
            batch_size=batch_size,
            return_data=return_data
        )

    def calc_fragment_descriptors(
        self,
        descriptor_calc_fns_list: List[callable] = Descriptors._descList,
        descriptor_keys: Optional[List[str]] = None,
        smiles_input: bool = False
    ) -> None:
        self.descriptor_calculator.calc_fragment_descriptors(
            descriptor_calc_fns_list=descriptor_calc_fns_list,
            descriptor_keys=descriptor_keys,
            smiles_input=smiles_input
        )

    def make_compound_graphs(
        self,
        smiles_list: List[str],
        workers: int = 8,
        pyg_data: bool = True
    ) -> List[Data] | List[tuple]:
        return self.graph_processor.make_compound_graphs(
            smiles_list=smiles_list,
            workers=workers,
            pyg_data=pyg_data
        )

    def check_for_graphs_groupings(
        self,
        smiles: List[str],
        workers: int = 8
    ) -> List[tuple]:
        return self.graph_processor.check_for_graphs_groupings(
            smiles=smiles,
            workers=workers
        )

    def get_CG_from_smiles(
        self,
        smiles: str,
        return_CG_object: bool = True
    ) -> object:
        return self.graph_processor.get_CG_from_smiles(
            smiles=smiles,
            return_CG_object=return_CG_object
        )

    def get_GSGE_clustering(
        self,
        embeddings: Optional[np.ndarray] = None,
        graph_data: Optional[List[Data]] = None,
        smiles_df: Optional[pd.DataFrame] = None,
        cluster_labels: Optional[List[int]] = None,
        smiles_column: str = 'SMILES',
        cluster_args: dict = {},
        cluster: bool = True
    ) -> object:
        return self.clustering_analyzer.get_GSGE_clustering(
            embeddings=embeddings,
            graph_data=graph_data,
            smiles_df=smiles_df,
            cluster_labels=cluster_labels,
            smiles_column=smiles_column,
            cluster_args=cluster_args,
            cluster=cluster
        )
    
    def load_GAE_weights(self, load_checkpoint_path: str, map_location: str = 'cuda') -> None:
        self.embedding_manager.load_GAE_weights(load_checkpoint_path, map_location)

    def make_GS_fragment_embedding_dict(
            self,
            load_checkpoint_path: Optional[str] = None,
            map_location: str = 'cuda',
            max_atom_size: int = ATOM_MAX_NUM,
            process_smiles: callable = from_smiles,
            atom_to_token_id: Dict = atom_to_token_id,
            device: str = 'cuda',
            batch_size: int = 64
        ) -> None:
        self.embedding_manager.make_GS_fragment_embedding_dict(
            load_checkpoint_path=load_checkpoint_path,
            map_location=map_location,
            max_atom_size=max_atom_size,
            process_smiles=process_smiles,
            atom_to_token_id=atom_to_token_id,
            device=device,
            batch_size=batch_size
        )
    
    def set_encoder(self, encoder: nn.Module = 
            AttentiveFP(in_channels=9, hidden_channels=256, out_channels=128, edge_dim=3, num_layers=3, num_timesteps=2)
        ) -> None:
        self.gae_trainer.set_encoder(encoder)

    def set_decoder(self, decoder: nn.Module =
            GraphDecoder(latent_dim=128, hidden_dim=256)
        ) -> None:
        self.gae_trainer.set_decoder(decoder)
    
    def add_standard_smaller_fragments(self, fragments: List[str] = COMMON_SMALLER_FRAGMENTS) -> None:
        for frag in fragments:
            self.vocab_manager.GS_vocab.add_GS_fragment(frag)

    def add_GS_vocab_to_GSGE_corpus(self) -> None:
        CoreGSGE.add_GS_vocab_to_GSGE_corpus(
            self.vocab_manager.GS_vocab.vocab_fragments,
            self.vocab_manager.GSGE_corpus
        )

    def add_all_single_elements(self, element_bond_counts: Dict[str, List[int]] = _ELEMENTS_BOND_COUNTS) -> None:
        if self.vocab_manager.GS_vocab is not None:
            CoreGSGE.add_all_single_elements(self.vocab_manager.GS_vocab, element_bond_counts)
            print('Added single elements to GS_vocab')
        if self.vocab_manager.GSGE_corpus is not None:
            CoreGSGE.add_all_single_elements(self.vocab_manager.GSGE_corpus, element_bond_counts)
            print('Added single elements to GSGE_corpus')

    def get_fragments_smiles(self) -> List[str]:
        return self.vocab_manager.GS_vocab.fragments

    def get_fragments_mols(self) -> List[object]:
        return [self.vocab_manager.GS_vocab.vocab_fragments[key].mol
                for key in self.vocab_manager.GS_vocab.vocab_fragments]

    def get_smiles_from_GS_frag_id(self, frag_id: str) -> str:
        return self.vocab_manager.GS_vocab.frag_id_to_noncanonical[frag_id]

    def get_GS_frag_id_to_smiles_dict(self) -> Dict[str, str]:
        return self.vocab_manager.GS_vocab.frag_id_to_noncanonical

    def get_GS_from_GS_frag_id(self, frag_id: str) -> str:
        return self.vocab_manager.group_grammar[frag_id]

    def get_fragment_descriptors(self) -> Optional[torch.Tensor]:
        return self.descriptor_calculator.get_fragment_descriptors()

    def get_fragment_embeddings(self) -> Optional[np.ndarray]:
        return self.embedding_manager.get_fragment_embeddings()

    def export_fragment_embeddings_to_csv(self, filename: str = 'fragment_embeddings.csv') -> None:
        """
        Export fragment embeddings to CSV file.

        Facade method that delegates to embedding_manager.export_embeddings_to_csv().
        Provides a consistent API pattern matching other GSGE methods.

        Args:
            filename: Output CSV filename. Default is 'fragment_embeddings.csv'.

        Raises:
            ValueError: If embeddings haven't been generated yet.

        Example:
            >>> gsge.make_GS_fragment_embedding_dict()
            >>> gsge.export_fragment_embeddings_to_csv('embeddings.csv')
            Embeddings exported to embeddings.csv
        """
        return self.embedding_manager.export_embeddings_to_csv(filename)

    def get_fragment_descriptors_and_embeddings(self) -> torch.Tensor:
        return torch.cat([
            self.embedding_manager.get_fragment_embeddings(),
            self.descriptor_calculator.get_fragment_descriptors()
        ], dim=1)

    def get_fragment_descriptors_names(self) -> Optional[List[str]]:
        return self.descriptor_calculator.get_fragment_descriptors_names()

    def export_fragment_descriptors_to_csv(self, filename: str = 'fragment_descriptors.csv') -> None:
        """
        Export fragment descriptors to CSV file.

        Facade method that delegates to descriptor_calculator.export_descriptors_to_csv().
        Provides a consistent API pattern matching other GSGE methods.

        Args:
            filename: Output CSV filename. Default is 'fragment_descriptors.csv'.

        Raises:
            ValueError: If descriptors haven't been calculated yet.

        Example:
            >>> gsge.calc_fragment_descriptors()
            >>> gsge.export_fragment_descriptors_to_csv('descriptors.csv')
            Descriptors exported to descriptors.csv
        """
        return self.descriptor_calculator.export_descriptors_to_csv(filename)

    def preprocess_from_SMILES(self, smiles):
        return self.tokenizer.preprocess_from_SMILES(smiles=smiles)
    
    def parallel_tokenize_SMILES_list(self, smiles_list, max_workers=8):
        return self.tokenizer.parallel_tokenize_SMILES_list(smiles_list=smiles_list, max_workers=max_workers)
    
    def plot_GS_fragments_in_mol(self, smiles, args):
        return self.tokenizer.plot_GS_fragments_in_mol(smiles, args)
    
    def get_group_grammar(self):
        return self.vocab_manager.group_grammar
    
    def get_GS_vocab(self) -> GS_Vocab:
        return self.vocab_manager.GS_vocab

    def get_GSGE_corpus(self) -> GSGE_Corpus:
        return self.vocab_manager.GSGE_corpus
    
    def get_GSGE_vocab(self):
        return self.vocab_manager.GSGE_vocab
        
        