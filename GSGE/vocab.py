
import os
from collections import defaultdict
from tqdm import tqdm
import joblib
from abc import ABC, abstractmethod
from group_selfies import Group
from rdkit import Chem
from group_selfies.utils.fragment_utils import (
    merge_patterns, force_core, closely_contained, select_diverse_set, mol_to_group_s, get_core)
from group_selfies.utils.group_utils import HashableMolecule
from .fragment_tools import FragmentTools
from .fragment_functions import CUSTOM_fragment_mol


class BaseGSVocab(FragmentTools, ABC):
    """
    Base class for Group-SELFIES fragment vocabularies.

    This class provides common functionality for managing fragment vocabularies,
    including fragment addition, canonicalization, visualization, and persistence.
    It serves as an abstract base class that defines the interface and shared
    behavior for both GS_Vocab (vocabulary builder with diversity selection) and
    GSGE_Corpus (corpus builder allowing non-unique fragments).

    The class implements robust fragment canonicalization, ensuring that duplicate
    fragments (same molecular structure but different SMILES representations) are
    properly identified and tracked. All fragments are stored in both their original
    form and canonical form, enabling flexible querying and consistent representation.

    Core Functionality:
    - Fragment canonicalization and duplicate detection via hash-based lookup
    - Fragment ID generation with configurable prefixes (GS_frag_ or GSGE_frag_)
    - Bi-directional mapping between fragment IDs, SMILES, and canonical hashes
    - Interactive visualization in Jupyter notebooks
    - Save/load persistence for vocabulary sharing

    Design Pattern:
    This class uses the Template Method pattern - it defines the skeleton of
    vocabulary management while delegating specific behaviors to subclasses:
    - GS_Vocab: Enforces diversity selection and core merging
    - GSGE_Corpus: Allows non-unique fragments for training data

    Attributes:
        core_dict: Maps core patterns to fragment lists.
            Type differs by subclass:
            - GS_Vocab: defaultdict(list) with HashableMolecule keys
            - GSGE_Corpus: dict with canonical SMILES string keys
        core_counter: Tracks occurrence counts of core patterns (defaultdict).
        fragments: List of all fragment SMILES strings (may contain duplicates).
        num_fragments: Count of unique fragments in vocabulary (int).
        vocab_fragments: Maps fragment IDs to Group objects (dict).
        hash_to_frag_info: Maps canonical hashes to (frag_id, canonical_smiles) tuples.
        frag_to_canonical: Maps non-canonical fragments to their canonical hashes.
        frag_id_to_noncanonical: Maps fragment IDs to original fragment SMILES.

    Example:
        BaseGSVocab cannot be instantiated directly. Use subclasses:

        >>> vocab = GS_Vocab()  # For diverse, non-redundant vocabularies
        >>> corpus = GSGE_Corpus()  # For training data with all fragment variants

    Note:
        This class is abstract - use GS_Vocab or GSGE_Corpus for concrete implementations.
    """

    def __init__(self, load_path: None | str = None):
        """
        Initialize base vocabulary attributes and data structures.

        Sets up all common data structures needed for fragment tracking and
        canonicalization. The core_dict attribute is NOT initialized here
        as it has different types in subclasses (see Notes).

        Args:
            load_path: Optional path to load vocabulary from file.
                If provided, child classes will handle loading during initialization.
                Note: Actual loading logic is implemented by child class methods
                (load_GS_vocab for GS_Vocab, load_GSGE_corpus for GSGE_Corpus).
                Default is None (creates empty vocabulary).

        Example:
            >>> # Initialization is handled by subclasses
            >>> vocab = GS_Vocab()  # Empty vocabulary
            >>> loaded_vocab = GS_Vocab(load_path='my_vocab.pkl')

        Note:
            Attribute initialization details:
            - core_dict: NOT initialized here (differs by subclass)
            - core_counter: defaultdict(int) for tracking core frequencies
            - fragments: Empty list for storing all fragment SMILES
            - num_fragments: Counter starting at 0
            - vocab_fragments: Empty dict mapping IDs to Group objects
            - hash_to_frag_info: Empty dict for canonical hash lookups
            - frag_to_canonical: Empty dict for canonicalization mapping
            - frag_id_to_noncanonical: Empty dict for ID->SMILES mapping
            - settings: Empty dict for build parameters (overwritten by build_vocab/build_corpus)

            Subclass-specific core_dict types:
            - GS_Vocab: defaultdict(list) with HashableMolecule keys
            - GSGE_Corpus: Regular dict {} with canonical SMILES string keys
        """
        # Initialize common attributes
        self.core_counter = defaultdict(int)
        self.fragments = []
        self.num_fragments = 0
        self.vocab_fragments = {}
        self.hash_to_frag_info = {}
        self.frag_to_canonical = {}
        self.frag_id_to_noncanonical = {}
        self.settings = {}  # Initialize empty dict, populated by build_vocab/build_corpus

        # Note: load_path handling is deferred to child classes
        # which implement their own load methods

    @abstractmethod
    def get_frag_id_prefix(self) -> str:
        """
        Return the fragment ID prefix for this vocabulary type.

        This abstract method must be implemented by all subclasses to define
        their unique fragment ID prefix. The prefix is used to generate
        sequential fragment IDs (e.g., GS_frag_0, GS_frag_1, ...) and helps
        distinguish fragments from different vocabulary sources.

        Returns:
            Fragment ID prefix string. Concrete implementations return:
            - 'GS_frag_' for GS_Vocab (diverse vocabulary builder)
            - 'GSGE_frag_' for GSGE_Corpus (corpus for training)

        Example:
            >>> class MyVocab(BaseGSVocab):
            ...     def get_frag_id_prefix(self):
            ...         return 'MyVocab_frag_'
            >>> vocab = MyVocab()
            >>> vocab.get_frag_id_prefix()
            'MyVocab_frag_'

        Note:
            Fragment IDs are generated as f'{prefix}{num_fragments}' where
            num_fragments is the current count of unique fragments. This ensures
            unique, sequential IDs that can be easily parsed and sorted.
        """
        pass

    def init_vocab(self):
        """
        Reset vocabulary structures and re-add all fragments with fresh IDs.

        This method clears all fragment tracking dictionaries and re-processes
        all fragments from self.fragments, generating new sequential fragment IDs.
        It's useful for regenerating vocabulary after manual modifications to the
        fragment list, or for ensuring consistent ID numbering.

        The method preserves the self.fragments list but regenerates all derived
        data structures (vocab_fragments, hash_to_frag_info, frag_to_canonical,
        frag_id_to_noncanonical) and resets the fragment counter.

        Side Effects:
            - Resets num_fragments counter to 0
            - Clears vocab_fragments, hash_to_frag_info, frag_to_canonical, frag_id_to_noncanonical
            - Regenerates all mappings by re-adding each fragment from self.fragments
            - Displays progress bar during processing (via tqdm)

        Example:
            >>> vocab = GS_Vocab()
            >>> vocab.fragments.extend(['C(=O)O', 'C(=O)N'])  # Manual addition
            >>> vocab.init_vocab()  # Regenerate IDs and mappings
            >>> vocab.num_fragments
            2

        Note:
            Use cases for init_vocab:
            1. After manually modifying self.fragments list
            2. To reset fragment IDs to start from 0
            3. To regenerate canonicalization mappings
            4. To ensure data consistency after direct attribute manipulation

            Performance: O(n) where n is the number of fragments. For large
            vocabularies (>10,000 fragments), this may take several seconds.
        """
        self.num_fragments = 0
        self.vocab_fragments = {}  # Stores fragment ID -> Group object
        self.hash_to_frag_info = {}  # Maps canonical hash -> (frag_id, canonical smiles)
        self.frag_to_canonical = {}  # Maps non-canonical frag -> canonical hash
        self.frag_id_to_noncanonical = {}  # Maps frag_id -> original non-canonical frag
        frags = self.fragments.copy()
        for frag in tqdm(frags):
            self.add_GS_fragment(frag)

    def get_hashes(self):
        """
        Return list of canonical fragment hashes for all unique fragments.

        Retrieves the canonical hash keys from hash_to_frag_info, which represents
        the set of unique molecular structures in the vocabulary. Each hash corresponds
        to a distinct fragment regardless of SMILES representation variations.

        Returns:
            List of canonical hash strings for all unique fragments.
            The length equals num_fragments (count of unique fragments).
            Hashes are deterministic based on molecular structure and can be used
            for fragment comparison across different vocabularies.

        Example:
            >>> vocab = GS_Vocab()
            >>> vocab.add_GS_fragment('CCO')  # Ethanol
            >>> vocab.add_GS_fragment('C-C-O')  # Same fragment, different SMILES
            >>> vocab.num_fragments
            1
            >>> hashes = vocab.get_hashes()
            >>> len(hashes)
            1

        Note:
            Canonical hashes are computed using RDKit's canonicalization algorithm
            and provide a robust way to identify duplicate fragments regardless of
            their SMILES representation. Two fragments with the same hash are
            chemically identical.
        """
        return list(self.hash_to_frag_info.keys())

    def add_GS_fragment(self, frag):
        """
        Add molecular fragment to vocabulary with automatic canonicalization.

        This method adds a fragment to the vocabulary while handling canonicalization
        and duplicate detection. Fragments are stored in both their original form
        (non-canonical SMILES) and canonical form, enabling flexible querying.

        Canonicalization Process:
        1. Compute canonical hash from input SMILES (handles non-canonical input)
        2. If canonical hash is new (not seen before):
           - Generate new fragment ID (e.g., GS_frag_0)
           - Compute canonical SMILES representation
           - Create Group object with original SMILES
           - Store all mappings (ID->Group, hash->(ID, canonical), etc.)
           - Increment fragment counter
        3. Always map input SMILES to its canonical hash (enables lookup)

        Args:
            frag: Fragment SMILES string. Can be canonical or non-canonical.
                Must be a valid molecular fragment representation parseable by RDKit.
                Examples: 'CCO', 'C-C-O', 'C(=O)O', '[CH3][CH2][OH]'

        Side Effects:
            - Updates vocab_fragments if fragment is new
            - Updates hash_to_frag_info if fragment is new
            - Updates frag_to_canonical (always, for lookup)
            - Updates frag_id_to_noncanonical if fragment is new
            - Appends to self.fragments if fragment not already present
            - Increments num_fragments counter if fragment is new

        Example:
            >>> vocab = GS_Vocab()
            >>> vocab.add_GS_fragment('CCO')  # Add ethanol
            >>> vocab.num_fragments
            1
            >>> vocab.add_GS_fragment('C-C-O')  # Same molecule, different SMILES
            >>> vocab.num_fragments
            1  # Still 1 - duplicate detected

        Note:
            Error Handling:
            - Silently skips invalid SMILES with console message
            - Catches exceptions during canonicalization
            - Fragments must be parseable by RDKit's MolFromSmiles

            Duplicate Detection:
            - Uses molecular graph canonicalization (hash-based)
            - Different SMILES for same molecule detected as duplicate
            - Only unique molecular structures are counted in num_fragments
        """
        try:
            # Step 1: Get canonical hash (even if frag is non-canonical)
            hash_canonical_smi = FragmentTools.get_hash_by_smiles(frag).hash

            # Step 2: If hash is new, store it with a new ID
            if hash_canonical_smi not in self.hash_to_frag_info:
                frag_id = f'{self.get_frag_id_prefix()}{self.num_fragments}'
                canonical_smiles = FragmentTools.canonicalize_smiles(frag)  # Get canonical smiles
                group = Group(frag_id, frag)  # Store non-canonical fragment

        except Exception as e:
            print(f"Skipping fragment {frag} due to error: {e}")
            return

        if hash_canonical_smi not in self.hash_to_frag_info:

            self.vocab_fragments[frag_id] = group  # Store non-canonical fragment
            self.hash_to_frag_info[hash_canonical_smi] = (frag_id, canonical_smiles)
            self.frag_id_to_noncanonical[frag_id] = frag  # Save original non-canonical fragment
            self.num_fragments+=1
            if frag not in self.fragments: #prevents double adding whe  using self.inti_vocab
                self.fragments.append(frag)

        # Step 3: Always map the non-canonical fragment to its canonical hash
        self.frag_to_canonical[frag] = hash_canonical_smi

    def add_GS_group(self, group: Group):
        """
        Add Group-SELFIES Group object to vocabulary with canonicalization.

        This method is similar to add_GS_fragment but operates on Group objects
        from the group_selfies library. Groups contain pre-computed canonical SMILES
        and are used internally by Group-SELFIES for molecular representation.

        The method canonicalizes the group's canonical SMILES to ensure consistent
        fragment identification, and assigns a new fragment ID if the group represents
        a novel fragment. The Group object's name attribute is updated to reflect
        its new fragment ID.

        Args:
            group: Group-SELFIES Group object containing fragment information.
                Must have 'canonsmiles' attribute with canonical SMILES string.
                Expected format: Group-SELFIES style SMILES (e.g., 'C(1*)(1*)')
                where wildcard atoms are represented as (*) attachment points.

        Side Effects:
            - Updates group.name with assigned fragment ID (if fragment is new)
            - Updates vocab_fragments if fragment is new
            - Updates hash_to_frag_info if fragment is new
            - Updates frag_to_canonical (always, for lookup)
            - Updates frag_id_to_noncanonical if fragment is new
            - Appends to self.fragments if fragment not already present
            - Increments num_fragments counter if fragment is new

        Example:
            >>> from group_selfies import Group
            >>> vocab = GS_Vocab()
            >>> group = Group('temp_name', 'C(=O)O')
            >>> vocab.add_GS_group(group)
            >>> print(group.name)
            'GS_frag_0'

        Note:
            Group-SELFIES Format:
            - Groups use (*) notation for attachment points (wildcard atoms)
            - Example: 'C(1*)(1*)' not '*C*' (different from standard SMILES)
            - Groups must be created from valid molecular fragments

            Error Handling:
            - Silently skips Groups with invalid canonical SMILES
            - Prints console message for Groups that fail canonicalization
            - Groups must have valid 'canonsmiles' attribute
        """
        try:
            # Step 1: Get canonical hash (even if frag is non-canonical)
            hash_canonical_smi = FragmentTools.get_hash_by_smiles(group.canonsmiles).hash

            # Step 2: If hash is new, store it with a new ID
            if hash_canonical_smi not in self.hash_to_frag_info:
                frag_id = f'{self.get_frag_id_prefix()}{self.num_fragments}'
                canonical_smiles = FragmentTools.canonicalize_smiles(group.canonsmiles)  # Get canonical smiles

        except Exception as e:
            print(f"Skipping group {group} due to error: {e}")
            return

        if hash_canonical_smi not in self.hash_to_frag_info:
            group.name = frag_id
            self.vocab_fragments[frag_id] = group  # Store non-canonical fragment
            self.hash_to_frag_info[hash_canonical_smi] = (frag_id, canonical_smiles)
            self.frag_id_to_noncanonical[frag_id] = group.canonsmiles  # Save original non-canonical fragment
            self.num_fragments+=1
            if group.canonsmiles not in self.fragments:
                self.fragments.append(group.canonsmiles)

        # Step 3: Always map the non-canonical fragment to its canonical hash
        self.frag_to_canonical[group.canonsmiles] = hash_canonical_smi

    def plot_vocab(self, mols_per_page=10, mols_per_row=5, sub_img_size=(300, 300), show_slider=True):
        """
        Display vocabulary fragments interactively in Jupyter notebook with pagination.

        Creates an interactive grid visualization for browsing large fragment
        vocabularies/corpora. Displays molecular structures as 2D drawings
        organized in a grid layout, with optional slider-based navigation for
        paging through large fragment sets.

        The visualization is particularly useful for:
        - Quality checking fragment vocabularies after building
        - Identifying problematic fragments (invalid structures, unusual chem)
        - Exploring fragment diversity and chemical space coverage
        - Presenting vocabularies in publications or presentations

        Args:
            mols_per_page: Number of molecules to display per page. Default is 10.
                Higher values show more fragments but require more scrolling.
                Recommended: 10-50 for readability.
            mols_per_row: Number of molecules per row in the grid layout.
                Default is 5. Affects grid aspect ratio.
                Recommended: 3-6 depending on screen width.
            sub_img_size: Size of each molecule image as (width, height) tuple in pixels.
                Default is (300, 300). Larger values show more detail but consume more memory.
                Recommended: (200, 200) to (500, 500).
            show_slider: Whether to show interactive slider for page navigation.
                Default is True. If False, only displays first page.
                Set to False for static snapshots or non-interactive environments.

        Side Effects:
            - Creates and displays ipywidgets IntSlider (if show_slider=True)
            - Uses IPython.display for rendering molecules
            - Generates in-memory RDKit molecule drawings
            - Clears previous output in Jupyter cell on page updates

        Raises:
            ImportError: If ipywidgets or IPython.display are not available.
                Install with: pip install ipywidgets

        Example:
            Basic usage with default settings:
            >>> vocab = GS_Vocab(load_path='my_vocab.pkl')
            >>> vocab.plot_vocab()

            Custom layout for presentation:
            >>> vocab.plot_vocab(mols_per_page=20, mols_per_row=4, sub_img_size=(400, 400))

            Quick snapshot without slider:
            >>> vocab.plot_vocab(show_slider=False)

            Large vocabulary exploration:
            >>> corpus = GSGE_Corpus(load_path='large_corpus.pkl')
            >>> corpus.plot_vocab(mols_per_page=50, mols_per_row=5)

        Note:
            Requirements:
            - ipywidgets: For interactive slider widget
            - IPython.display: For Jupyter notebook rendering
            - RDKit.Chem.Draw: For molecular structure depiction

            Install dependencies:
            ```
            pip install ipywidgets
            jupyter nbextension enable --py widgetsnbextension
            ```

            Performance:
            - Rendering speed depends on sub_img_size and mols_per_page
            - Large images (>500px) or many fragments (>100 per page) may be slow
            - Total rendering time scales with vocabulary size

            Tips:
            - Use smaller sub_img_size (200-250) when browsing many fragments
            - Use larger sub_img_size (400-500) for detailed inspection or screenshots
            - Adjust mols_per_row to match your display width
            - Consider show_slider=False for generating static figures
        """
        import ipywidgets as widgets
        from IPython.display import display
        from rdkit.Chem import Draw

        # Convert vocab_fragment values to a list
        mols_list = [g.mol for g in self.vocab_fragments.values()]
        total_mols = len(mols_list)

        # Create an Output widget for the molecule images
        out = widgets.Output()

        # Function to update the displayed molecules
        def update_display(page):
            start = page * mols_per_page
            end = min(start + mols_per_page, total_mols)
            mols_to_draw = mols_list[start:end]

            # Clear only the output widget containing the images
            out.clear_output(wait=True)

            # Display the molecules in a grid within the output widget
            with out:
                img = Draw.MolsToGridImage(mols_to_draw, molsPerRow=mols_per_row, subImgSize=sub_img_size)
                display(img)

        # Create a slider to navigate through pages (if show_slider is True)
        if show_slider:
            page_slider = widgets.IntSlider(
                min=0, max=max(0, (total_mols // mols_per_page)), step=1, description="Page:"
            )

            # Observe changes to the slider
            page_slider.observe(lambda change: update_display(change['new']), names='value')

            # Show the slider and output widget
            display(page_slider)
            display(out)
            update_display(0)
        else:
            # No slider, just show the output widget with first page
            display(out)
            update_display(0)

    def _save_base(self, dir_path='.', vocab_name='vocab', meta_info=''):
        """
        Save vocabulary/corpus to pickle file (protected base implementation).

        This is the base implementation for saving vocabulary data structures to disk.
        Child classes (GS_Vocab and GSGE_Corpus) wrap this method with public
        save methods that provide appropriate default values for vocab_name.

        The method saves all essential vocabulary state including fragments, settings,
        core information, and optional metadata. This enables full vocabulary restoration
        when loading via _load_base.

        Saved Data Structure:
        {
            'fragments': List of fragment SMILES strings,
            'settings': Dict of build parameters (n_limit, target, MIN_SIZE, etc.),
            'meta_info': User-provided metadata string,
            'cores_dict': Core pattern mappings,
            'core_counter': Core occurrence counts
        }

        Args:
            dir_path: Directory path where vocabulary file will be saved.
                Directory will be created if it doesn't exist. Default is current directory.
            vocab_name: Base filename for saved vocabulary (without extension).
                File will be saved as '{dir_path}/{vocab_name}' using joblib.
                Note: No file extension is added automatically. Default is 'vocab'.
            meta_info: Optional metadata string to store with the vocabulary.
                Useful for recording provenance, dataset info, or notes.
                Examples: 'Built from ChEMBL v30, drug-like molecules only',
                          'Training set for GAE model, 2024-01-15'.

        Side Effects:
            - Creates/overwrites file at '{dir_path}/{vocab_name}'
            - Serializes vocabulary state using joblib (binary format)
            - Does NOT modify in-memory vocabulary state

        Example:
            This method is called internally by child classes:
            >>> vocab = GS_Vocab()
            >>> # ... build vocabulary ...
            >>> vocab._save_base(dir_path='./vocabs', vocab_name='my_vocab',
            ...                 meta_info='Built from 10k molecules')

        Note:
            Protected Method Design:
            - This method is protected (prefix: _) intended for internal use
            - Child classes provide public methods (save_GS_vocab, save_GSGE_corpus)
            - Users should call child class methods, not _save_base directly

            File Format:
            - Uses joblib for efficient Python object serialization
            - Binary format (not human-readable)
            - Compatible with joblib.load() for loading

            What's NOT Saved:
            - vocab_fragments (regenerated from fragments via init_vocab)
            - hash_to_frag_info (regenerated from fragments via init_vocab)
            - frag_to_canonical (regenerated from fragments via init_vocab)
            - frag_id_to_noncanonical (regenerated from fragments via init_vocab)

            These mappings are automatically regenerated on load, ensuring
            consistency without saving redundant data.
        """
        store_dict = {
            'fragments': self.fragments,
            'settings': self.settings,
            'meta_info': meta_info,
            'cores_dict': self.core_dict,
            'core_counter': self.core_counter
        }

        # Create directory if it doesn't exist (as documented)
        os.makedirs(dir_path, exist_ok=True)

        joblib.dump(store_dict, os.path.join(dir_path, vocab_name))

    def _load_base(self, file_path):
        """
        Load vocabulary/corpus from pickle file (protected base implementation).

        This is the base implementation for loading vocabulary data from disk.
        Child classes (GS_Vocab and GSGE_Corpus) wrap this method with public
        load methods. After loading the saved state, automatically calls
        init_vocab() to regenerate all fragment IDs and canonicalization mappings.

        The loading process restores vocabulary fragments, settings, core information,
        and metadata, then rebuilds the derived data structures (vocab_fragments,
        hash mappings, etc.) by re-adding all fragments.

        Args:
            file_path: Path to saved vocabulary/corpus file (created by _save_base).
                Must be a valid joblib-serialized pickle file.
                Can be relative or absolute path.
                Examples: 'vocab.pkl', './data/my_vocab', '/path/to/vocab'

        Side Effects:
            - Overwrites self.fragments with loaded data
            - Overwrites self.settings with loaded parameters
            - Overwrites self.meta_info with loaded metadata
            - Overwrites self.cores_dict with loaded core mappings
            - Overwrites self.core_counter with loaded counts
            - Calls init_vocab() which regenerates all fragment mappings
            - Resets num_fragments and rebuilds all derived structures

        Raises:
            FileNotFoundError: If file_path doesn't exist
            pickle.UnpicklingError: If file is corrupted or invalid format
            KeyError: If saved file is missing required keys
            Exception: For other loading errors (prints error message)

        Example:
            This method is called internally by child classes:
            >>> vocab = GS_Vocab()
            >>> vocab._load_base('path/to/vocab_file')
            Vocabulary loaded successfully!
            >>> # vocab.fragments, settings, and all mappings are now restored

        Note:
            Protected Method Design:
            - This method is protected (prefix: _) intended for internal use
            - Child classes provide public methods (load_GS_vocab, load_GSGE_corpus)
            - Users should call child class methods, not _load_base directly

            Automatic Regeneration:
            - Always calls init_vocab() after loading
            - Regenerates vocab_fragments (ID -> Group mappings)
            - Regenerates hash_to_frag_info (canonical hash lookups)
            - Regenerates frag_to_canonical (SMILES -> hash mappings)
            - Regenerates frag_id_to_noncanonical (ID -> SMILES mappings)
            - Resets num_fragments to actual count of unique fragments

            Error Handling:
            - Prints success message on successful load
            - Prints error message but doesn't raise exception on load failure
            - Continues with init_vocab() even if some keys are missing
            - Gracefully handles corrupted files

            Data Integrity:
            - Original fragment IDs may NOT be preserved after loading
            - init_vocab() regenerates IDs sequentially starting from 0
            - This ensures consistent ID numbering but changes from original
            - If you need stable IDs across sessions, consider saving/loading
              the entire object without rebuilding (advanced use case)
        """
        try:
            store_dict = joblib.load(file_path)
            self.fragments = store_dict.get('fragments')
            self.settings = store_dict.get('settings')
            self.meta_info = store_dict.get('meta_info')
            self.cores_dict = store_dict.get('cores_dict')
            self.core_counter = store_dict.get('core_counter')

            print("Vocabulary loaded successfully!")

        except Exception as e:
            print(f"Error loading vocabulary: {e}")

        self.init_vocab()


class GS_Vocab(BaseGSVocab):

    """
    Group-SELFIES Vocabulary Builder with Diversity Selection

    This class constructs and manages a diverse vocabulary of molecular fragments
    using Group-SELFIES representations. It inherits common fragment management
    functionality from BaseGSVocab (addition, canonicalization, visualization,
    persistence) and extends it with automated fragmentation, core merging,
    and diversity-based selection capabilities.

    The class specializes in creating compact, non-redundant fragment vocabularies
    by selecting diverse subsets based on frequency and structural similarity.
    This makes it ideal for building efficient molecular representations that
    capture chemical diversity while minimizing redundancy.

    Inherits from BaseGSVocab:
        Provides foundational fragment vocabulary management:
        - Fragment canonicalization and duplicate detection
        - Fragment ID generation with 'GS_frag_' prefix
        - Interactive visualization via plot_vocab()
        - Save/load functionality via save_GS_vocab()/load_GS_vocab()
        - Bi-directional fragment mapping (ID, SMILES, canonical hash)

    Key Features:
    - Fragment molecules using customizable fragmentation strategies
    - Merge and generalize fragments around common cores
    - Select diverse fragment subsets based on frequency and structure
    - Visualize, save, load, and export fragment vocabularies
    """

    def __init__(self, load_path: None | str = None):
        """
        Initialize GS_Vocab vocabulary builder.

        Args:
            load_path: Path to previously saved vocabulary (.pkl file). If provided,
                automatically loads vocabulary during initialization.

        Example:
            >>> vocab = GS_Vocab()  # Empty vocabulary
            >>> vocab = GS_Vocab(load_path='vocab.pkl')  # Load existing
        """
        # Initialize base class attributes (core_counter, fragments, num_fragments, etc.)
        super().__init__(load_path)
        # GS_Vocab-specific: core_dict uses defaultdict(list) with HashableMolecule keys
        self.core_dict = defaultdict(list)

        if load_path is not None:
            self.load_GS_vocab(load_path)

    def merge_into_core(self, new_h, new_f, new_f_core):
        """Merge new fragment into existing core pattern if compatible."""
        for idx, existing_pattern in enumerate(self.core_dict[new_h]):
            if (res := merge_patterns(existing_pattern, new_f, new_f_core)) is not None:
                if '*' in Chem.MolToSmiles(res):
                    self.core_dict[new_h][idx] = res
                    self.core_counter[new_h] += 1
                break
        else:
            if '*' in Chem.MolToSmiles(new_f):
                self.core_dict[new_h].append(new_f)
                self.core_counter[new_h] += 1

    def get_frag_id_prefix(self) -> str:
        """
        Return the fragment ID prefix for GS_Vocab.

        Returns:
            Fragment ID prefix string 'GS_frag_'

        Example:
            >>> vocab = GS_Vocab()
            >>> vocab.get_frag_id_prefix()
            'GS_frag_'
        """
        return 'GS_frag_'

    def build_vocab(
            self,
            m_set,
            convert=False,
            n_limit=1,
            target=100,
            fragmented=False,
            method='default',
            MIN_SIZE=4,
            MAX_SIZE=15,
            fragment_mol_fn=CUSTOM_fragment_mol,
            verbose=False
            ):
        """
        Build vocabulary of molecular fragments from molecule set.

        Fragments molecules, merges similar patterns, and selects diverse subset
        of fragments for vocabulary. Process includes fragmentation, core merging,
        filtering by frequency, and diversity-based selection.

        Args:
            m_set: List of molecules (RDKit Mol objects or SMILES strings).
            convert: If True, treat m_set as SMILES strings and convert to Mol objects.
                Default is False.
            n_limit: Minimum occurrence count for fragment cores. Cores appearing less
                than n_limit times are filtered out. Default is 1.
            target: Target vocabulary size after diversity selection. Default is 100.
            fragmented: If True, treat each molecule as single fragment (no fragmentation).
                Default is False.
            method: Fragmentation method passed to fragment_mol_fn. Default is 'default'.
            MIN_SIZE: Minimum fragment size in atoms. Default is 4.
            MAX_SIZE: Maximum fragment size in atoms. Default is 15.
            fragment_mol_fn: Function for fragmenting molecules. Default is CUSTOM_fragment_mol
                which removes ring bonds, amide bonds, and disulfide bonds.
            verbose: If True, print detailed progress and error messages. Default is False.

        Example:
            Build vocabulary from SMILES:

            >>> vocab = GS_Vocab()
            >>> smiles_list = ['CCO', 'CC(C)O', 'c1ccccc1']
            >>> vocab.build_vocab(
            ...     m_set=smiles_list,
            ...     convert=True,
            ...     n_limit=2,
            ...     target=50,
            ...     MIN_SIZE=1,
            ...     MAX_SIZE=10
            ... )

            Build from RDKit Mol objects:

            >>> mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            >>> vocab.build_vocab(m_set=mols, target=100)

        Note:
            Process steps:
            1. Fragment each molecule using fragment_mol_fn
            2. Merge similar fragments into generalized cores
            3. Filter cores by minimum occurrence (n_limit)
            4. Merge closely contained cores
            5. Select diverse subset up to target size
            6. Canonicalize and add to vocabulary
        """

        if verbose is False:
            #fragments can cause lot of error statemetns, prevents spam that is expected
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

        # Early validation: check if n_limit is reasonable for dataset size
        if n_limit > 1 and len(m_set) < n_limit * 2:
            print(f"Warning: n_limit={n_limit} may be too high for dataset size={len(m_set)}. "
                  f"Consider reducing n_limit to ~{max(1, len(m_set) // 10)} or using n_limit=1 for small datasets.")

        self.core_dict = defaultdict(list)
        self.core_counter = defaultdict(int)
        self.fragments = []
        self.num_fragments = 0
        self.vocab_fragments = {}  # Stores fragment ID -> Group object
        self.hash_to_frag_info = {}  # Maps canonical hash -> (frag_id, canonical smiles)
        self.frag_to_canonical = {}  # Maps non-canonical frag -> canonical hash
        self.frag_id_to_noncanonical = {}  # Maps frag_id -> original non-canonical frag
        
        self.settings = {
            'convert':convert, 
            'n_limit':n_limit, 
            'target':target, 
            'fragmented':fragmented, 
            'method':method, 
            'MIN_SIZE':MIN_SIZE, 
            'MAX_SIZE':MAX_SIZE,
            'fragment_mol_fn':CUSTOM_fragment_mol,  
            } 
    
        #Initial fragmentation
        for m in tqdm(m_set):
            try:
                if convert:
                    try:
                        m = Chem.MolFromSmiles(m)
                        if m is None:
                            if verbose:
                                raise ValueError(f"Invalid SMILES: {m}")
                    except Exception as e:
                        if verbose:
                            print(f"Skipping molecule {m} due to conversion error: {e}")
                        continue  # Skip this molecule and move to the next

                try:
                    fragments = fragment_mol_fn(m, method=method, MIN_SIZE=MIN_SIZE, MAX_SIZE=MAX_SIZE) if not fragmented else [(m, get_core(m))]
                except Exception as e:
                    if verbose:
                        print(f"Skipping molecule {m} due to fragmentation error: {e}")
                    continue  # Skip this molecule if fragmentation fails

                for new_f, new_f_core in fragments:
                    try:
                        new_h = HashableMolecule(new_f_core)
                        self.merge_into_core(new_h, new_f, new_f_core)
                    except Exception as e:
                        if verbose:
                            print(f"Skipping fragment due to merging error: {e}")
            except Exception as e:
                if verbose:
                    print(f"Skipping molecule {m} due to unexpected error: {e}")

        #filter based on counts
        self.filtered_cores = []
        for k in self.core_dict.keys():
            try:
                if self.core_counter[k] >= n_limit:
                    self.filtered_cores.append(k)
            except Exception as e:
                if verbose:
                    print(f"Skipping core {k} due to error: {e}")

        # Warn if n_limit filtered out all cores
        if len(self.filtered_cores) == 0 and len(self.core_dict) > 0:
            max_count = max(self.core_counter.values())
            print(f"Warning: n_limit={n_limit} filtered out all {len(self.core_dict)} cores. "
                  f"Maximum core occurrence was {max_count}. "
                  f"Try reducing n_limit to <= {max_count} or using a larger dataset.")

        #merging similar fragments into one general fragment 
        self.current_cores = []
        for core in tqdm(sorted(self.filtered_cores, key=lambda x: x.mol.GetNumAtoms())):
            try:
                for existing_core in self.current_cores:
                    try:
                        if closely_contained(core.mol, existing_core.mol):
                            for group in self.core_dict.get(core, []):  # Use `.get()` to avoid KeyError
                                try:
                                    self.merge_into_core(existing_core, force_core(group, existing_core.mol), existing_core.mol)
                                except Exception as e:
                                    if verbose:
                                        print(f"Skipping group {group} due to error: {e}")
                            break
                    except Exception as e:
                        if verbose:
                            print(f"Skipping existing_core {existing_core} due to error: {e}")
                else:
                    self.current_cores.append(core)
            except Exception as e:
                if verbose:
                    print(f"Skipping core {core} due to error: {e}")

        #obtaining cores (and weight to be used in selection)
        self.cores, self.weights = [], []
        for k in self.current_cores:
            try:
                if k not in self.core_dict:
                    if verbose:
                        raise KeyError(f"Key {k} not found in core_dict")

                self.cores.append(k)
                self.weights.append(len(self.core_dict[k]))
            except Exception as e:
                if verbose:
                    print(f"Skipping core {k} due to error: {e}")

        #select final fragments based on diversity
        # Handle edge case: RDKit's GetTanimotoDistMat requires at least 2 molecules
        if len(self.cores) < 2:
            # If too few cores for diversity selection, include all cores
            self.include_cores = self.cores
            if verbose:
                print(f"Warning: Only {len(self.cores)} core(s) available for diversity selection. "
                      f"Skipping diversity selection and using all cores. "
                      f"Consider using a larger dataset or adjusting MIN_SIZE/MAX_SIZE parameters.")
        else:
            # Cap target to number of available cores to avoid RDKit errors
            actual_target = min(target, len(self.cores))
            self.include_cores = select_diverse_set(self.cores, actual_target, weights=self.weights)
        for k in tqdm(self.include_cores):
            for x in self.core_dict[k]:
                try:
                    res = mol_to_group_s(x)
                    if '*' in res:
                        #self.fragments.append(res)
                        self.add_GS_fragment(res)
                except Exception as e:
                    pass

        self.settings['fragment_mol_fn'] = fragment_mol_fn.__module__

    def save_GS_vocab(self, dir_path='.', vocab_name='GS_Vocab', meta_info=''):
        """
        Save vocabulary to pickle file.

        Args:
            dir_path: Directory path for saving. Default is current directory.
            vocab_name: Base name for saved file. Default is 'GS_Vocab'.
            meta_info: Optional metadata string to store with vocabulary.

        Example:
            >>> vocab.save_GS_vocab(dir_path='./vocabs', vocab_name='peptides_v1')
        """
        self._save_base(dir_path, vocab_name, meta_info) 

    def load_GS_vocab(self, file_path):
        """
        Load GS vocabulary from pickle file.

        Args:
            file_path: Path to saved vocabulary file (created by save_GS_vocab).

        Example:
            >>> vocab = GS_Vocab()
            >>> vocab.load_GS_vocab('vocabs/peptides_v1')
            Vocabulary loaded successfully!
        """
        self._load_base(file_path)

    def export_fragments_to_csv(self, filename='GS_vocab_fragments.csv'):
        """
        Export vocabulary fragments to CSV file.

        Args:
            filename: Output CSV filename. Default is 'GS_vocab_fragments.csv'.
        """
        import csv
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["GS_frag"])
            for string in self.fragments:
                writer.writerow([string])

class GSGE_Corpus(BaseGSVocab):

    """
    Group-SELFIES Guided Enumeration Corpus Builder

    This class constructs comprehensive fragment corpora by systematically fragmenting
    molecules and organizing fragments around canonical core structures. It inherits
    common fragment management functionality from BaseGSVocab (addition, canonicalization,
    visualization, persistence) and specializes in allowing non-unique fragments for
    machine learning applications.

    Unlike GS_Vocab, which enforces diversity selection and creates compact vocabularies,
    GSGE_Corpus preserves all fragment variants, making it ideal for training Graph
    Autoencoders and other models that benefit from diverse training data with multiple
    fragment representations of similar molecular patterns.

    Inherits from BaseGSVocab:
        Provides foundational fragment vocabulary management:
        - Fragment canonicalization and duplicate detection
        - Fragment ID generation with 'GSGE_frag_' prefix
        - Interactive visualization via plot_vocab()
        - Save/load functionality via save_GSGE_corpus()/load_GSGE_corpus()
        - Bi-directional fragment mapping (ID, SMILES, canonical hash)

    Key Features:
    - Fragment molecules and associate fragments with canonicalized cores
    - Allow multiple fragment variations per core (non-unique cores)
    - Build, save, load, and visualize fragment corpora
    - Canonicalize and track fragment hashes for consistency
    """

    def __init__(self, load_path: None | str = None):
        """
        Initialize GSGE_Corpus builder.

        Args:
            load_path: Path to previously saved corpus (.pkl file). If provided,
                automatically loads corpus during initialization.

        Example:
            >>> corpus = GSGE_Corpus()  # Empty corpus
            >>> corpus = GSGE_Corpus(load_path='corpus.pkl')  # Load existing
        """
        # Initialize base class attributes (core_counter, fragments, num_fragments, etc.)
        super().__init__(load_path)
        # GSGE_Corpus-specific: core_dict uses regular dict {} with canonical SMILES string keys
        self.core_dict = {}

        if load_path is not None:
            self.load_GSGE_corpus(load_path)

    def get_frag_id_prefix(self) -> str:
        """
        Return the fragment ID prefix for GSGE_Corpus.

        Returns:
            Fragment ID prefix string 'GSGE_frag_'

        Example:
            >>> corpus = GSGE_Corpus()
            >>> corpus.get_frag_id_prefix()
            'GSGE_frag_'
        """
        return 'GSGE_frag_'

    @staticmethod
    def _canonicalize_core(mol):
        """
        Convert core molecule to canonical SMILES for uniqueness.

        Ensures consistent core representation by canonicalizing SMILES strings,
        enabling proper fragment grouping by core structure.

        Args:
            mol: RDKit Mol object representing a fragment core.

        Returns:
            Canonical SMILES string if successful, None if canonicalization fails.

        Note:
            Used internally by build_corpus to group fragments by canonical cores.
        """
        try:
            return Chem.MolToSmiles(mol, canonical=True)
        except Exception:
            return None

    def _process_molecule(self, m, convert=False, fragmented=False, min_size=1, max_size=15, fragment_mol_fn=CUSTOM_fragment_mol, method='default'):
        """
        Process single molecule and update fragment database.

        Fragments molecule, canonicalizes cores, merges patterns, and adds fragments
        to corpus. Called internally by build_corpus for each molecule.

        Args:
            m: RDKit Mol object or SMILES string (if convert=True).
            convert: If True, convert SMILES string to Mol object. Default is False.
            fragmented: If True, treat molecule as single fragment. Default is False.
            min_size: Minimum fragment size in atoms. Default is 1.
            max_size: Maximum fragment size in atoms. Default is 15.
            fragment_mol_fn: Function for fragmenting molecules. Default is CUSTOM_fragment_mol.
            method: Fragmentation method passed to fragment_mol_fn. Default is 'default'.

        Note:
            - Silently skips molecules that fail processing
            - Merges similar fragments around same canonical core
            - Only adds fragments containing wildcard atoms ([*])
            - Internal method called by build_corpus
        """
        try:
            if convert:
                m = Chem.MolFromSmiles(m) #Original mol
                if m is None:
                    return

            fragments = fragment_mol_fn(m, method=method, MIN_SIZE=min_size, MAX_SIZE=max_size) if not fragmented else [(m, get_core(m))] #fragments of original mol

            for frag, core in fragments:
                # Check fragment size before processing
                if frag.GetNumAtoms() > max_size:
                    continue

                core_smiles = GSGE_Corpus._canonicalize_core(core)
                smile_ = Chem.MolToSmiles(frag)
                if core_smiles is None or '*' not in smile_:
                    continue

                if core_smiles in self.core_dict:
                    for existing_frag in self.core_dict[core_smiles]:
                        res = merge_patterns(existing_frag, frag, core)
                        smi_ = Chem.MolToSmiles(res)
                        if res and '*' in smi_:
                            if res.GetNumAtoms() > max_size:
                                continue
                            self.core_dict[core_smiles].append(res)
                            self.core_counter[core_smiles] += 1
                            break
                else:
                    self.core_dict[core_smiles] = [frag]
                    self.core_counter[core_smiles] = 1

        except Exception:
            pass

    def build_corpus(
        self,
        m_set,
        min_size=4,
        max_size=15,
        method='default',
        fragment_mol_fn=CUSTOM_fragment_mol,
        convert=True,
        fragmented=False,
        verbose=False
        ):
        """
        Build fragment corpus from molecule set (non-unique fragments).

        Fragments molecules and organizes fragments by canonical core structures,
        allowing multiple fragment variations per core. Unlike GS_Vocab.build_vocab,
        this method does NOT enforce diversity selection or core merging, making it
        ideal for training Graph Autoencoders where fragment variety is beneficial.

        Args:
            m_set: List of molecules (RDKit Mol objects or SMILES strings).
            min_size: Minimum fragment size in atoms. Default is 4.
            max_size: Maximum fragment size in atoms. Default is 15.
            method: Fragmentation method passed to fragment_mol_fn. Default is 'default'.
            fragment_mol_fn: Function for fragmenting molecules. Default is CUSTOM_fragment_mol
                which removes ring bonds, amide bonds, and disulfide bonds.
            convert: If True, treat m_set as SMILES strings. Default is True.
            fragmented: If True, treat each molecule as single fragment (no fragmentation).
                Default is False.
            verbose: If True, print RDKit warnings and error messages. Default is False.

        Example:
            Build corpus from SMILES for GAE training:

            >>> corpus = GSGE_Corpus()
            >>> smiles_list = ['CCO', 'CC(C)O', 'c1ccccc1']
            >>> corpus.build_corpus(
            ...     m_set=smiles_list,
            ...     min_size=1,
            ...     max_size=15,
            ...     convert=True
            ... )

            Build from RDKit Mol objects:

            >>> mols = [Chem.MolFromSmiles(s) for s in smiles_list]
            >>> corpus.build_corpus(m_set=mols, convert=False)

        Note:
            Key differences from GS_Vocab.build_vocab:
            - Allows non-unique fragments (multiple variants per core)
            - No diversity selection or target size
            - No n_limit filtering by occurrence count
            - Ideal for Graph Autoencoder training data
        """

        if verbose is False:
            from rdkit import RDLogger
            RDLogger.DisableLog('rdApp.*')

        self.settings = {
            'convert':False, 
            'fragmented':False, 
            'method':method, 
            'MIN_SIZE':min_size, 
            'MAX_SIZE':max_size, 
            'fragment_mol_fn': fragment_mol_fn.__module__
            }
        
        self.core_dict = {}  # {canonical_core_smiles: list of fragments}
        self.core_counter = defaultdict(int)  # {canonical_core_smiles: count}

        for m in tqdm(m_set):
            self._process_molecule(m, 
                convert=convert, 
                fragmented=fragmented, 
                min_size=min_size, 
                max_size=max_size, 
                fragment_mol_fn=fragment_mol_fn, 
                method=method
                )
        
        self.include_cores = [] 
        for k in self.core_counter:
            self.include_cores.append(k)

        #self.fragments = []
        for k in tqdm(self.include_cores):
            for x in self.core_dict[k]:
                try:
                    res = mol_to_group_s(x)
                    if '*' in res:
                        #self.fragments.append(res)
                        self.add_GS_fragment(res)
                except Exception as e:
                    pass

    def save_GSGE_corpus(self, dir_path='.', vocab_name='GSGE_corpus', meta_info=''):
        """
        Save corpus to pickle file.

        Args:
            dir_path: Directory path for saving. Default is current directory.
            vocab_name: Base name for saved file. Default is 'GSGE_corpus'.
            meta_info: Optional metadata string to store with corpus.

        Example:
            >>> corpus.save_GSGE_corpus(dir_path='./corpora', vocab_name='peptides_corpus_v1')
        """
        self._save_base(dir_path, vocab_name, meta_info)
  
    def load_GSGE_corpus(self, file_path):
        """
        Load GSGE corpus from pickle file.

        Args:
            file_path: Path to saved corpus file (created by save_GSGE_corpus).

        Example:
            >>> corpus = GSGE_Corpus()
            >>> corpus.load_GSGE_corpus('corpora/peptides_corpus_v1')
            Vocabulary loaded successfully!

        Note:
            Automatically calls init_vocab() to regenerate fragment IDs and mappings.
        """
        self._load_base(file_path)
