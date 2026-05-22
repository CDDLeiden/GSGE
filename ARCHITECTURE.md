## Architecture

GSGE follows a **Facade pattern** with the main `GSGE` class (in `GSGE/gsge.py`) delegating functionality to specialized manager classes. The codebase is organized into functional modules:

### Architecture Overview

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "dagre"}} }%%
flowchart LR
    GSGE["GSGE<br/>(Facade)"]

    subgraph Layer2["Manager Classes"]
        VM["VocabularyManager"]
        EM["EmbeddingManager"]
        GT["GAETrainer"]
        DC["DescriptorCalculator"]
        GP["GraphProcessor"]
        CA["ClusteringAnalyzer"]
        SD["Store_Data"]
        SM["Store_Modules"]
    end

    subgraph Layer3["Core Modules"]
        TOK["GSGE_tokenizer"]
        VOC["GS_Vocab /<br/>GSGE_Corpus"]
        CORE["CoreGSGE<br/>(static utils)"]
    end

    subgraph RightBottom[" "]
        direction TB
        subgraph Layer5["Outputs"]
            EMB["GSGE_Embedding"]
            CLUST["GSGE_clustering"]
            CG["compound_graph"]
        end

        subgraph Layer4["Graph Neural Networks"]
            ENC["AttentiveFP<br/>(Encoder)"]
            DEC["GraphDecoder<br/>(Decoder)"]
        end
    end

    GSGE --> VM
    GSGE --> EM
    GSGE --> GT
    GSGE --> DC
    GSGE --> GP
    GSGE --> CA
    GSGE --> SD

    VM --> VOC
    VM --> TOK
    EM --> SM
    SM --> ENC
    GT --> ENC
    GT --> DEC
    GP --> CG
    CA --> CLUST
    EM --> EMB

    TOK -.-> CORE
    EM -.-> CORE
    GT -.-> CORE
    GP -.-> CORE
    EMB -.-> CORE

    classDef facadeStyle fill:#4A90D9,stroke:#2E5A8B,color:#fff
    classDef managerStyle fill:#7CB342,stroke:#558B2F,color:#fff
    classDef coreStyle fill:#FF9800,stroke:#E65100,color:#fff
    classDef neuralStyle fill:#9C27B0,stroke:#6A1B9A,color:#fff
    classDef outputStyle fill:#607D8B,stroke:#37474F,color:#fff

    class GSGE facadeStyle
    class VM,EM,GT,DC,GP,CA,SD,SM managerStyle
    class TOK,VOC,CORE coreStyle
    class ENC,DEC neuralStyle
    class EMB,CLUST,CG outputStyle
    style RightBottom fill:none,stroke:none
```

### Data Flow Pipeline

```mermaid
flowchart LR
    subgraph Input
        SMI["SMILES<br/>Molecules"]
    end
    
    subgraph "1. Vocabulary Building"
        FRAG["Fragmentation<br/>(CUSTOM_fragment_mol)"]
        VOC["GS_Vocab<br/>(diversity selection)"]
        CORP["GSGE_Corpus<br/>(training data)"]
    end
    
    subgraph "2. Tokenization"
        TOK["GSGE_tokenizer"]
        TKS["Token IDs"]
    end
    
    subgraph "3. Graph Construction"
        CG["Compound Graphs<br/>(smiles_to_group_graph)"]
        FG["Fragment Graphs<br/>(from_smiles)"]
    end
    
    subgraph "4. GAE Training"
        ENC["Encoder<br/>(AttentiveFP)"]
        DEC["Decoder<br/>(GraphDecoder)"]
        CKPT["Checkpoints"]
    end
    
    subgraph "5. Embedding Generation"
        EMB["Fragment<br/>Embeddings"]
        DESC["Molecular<br/>Descriptors"]
    end
    
    subgraph "6. Analysis"
        DOWN["Downstream<br/>Models"]
        CLUST["Clustering<br/>(MCS + t-SNE/UMAP)"]
    end
    
    SMI --> FRAG
    FRAG --> VOC
    FRAG --> CORP
    VOC --> TOK
    TOK --> TKS
    VOC --> FG
    CORP --> FG
    TKS --> DOWN
    CG --> DOWN
    FG --> ENC
    ENC --> DEC
    DEC --> CKPT
    CKPT --> EMB
    VOC --> DESC
    EMB --> CLUST
    EMB --> DOWN
    DESC --> DOWN
    
    classDef inputStyle fill:#4A90D9,stroke:#2E5A8B,color:#fff
    classDef vocabStyle fill:#66BB6A,stroke:#388E3C,color:#fff
    classDef tokStyle fill:#FFA726,stroke:#E65100,color:#fff
    classDef graphStyle fill:#AB47BC,stroke:#7B1FA2,color:#fff
    classDef trainStyle fill:#EF5350,stroke:#C62828,color:#fff
    classDef embStyle fill:#26C6DA,stroke:#00838F,color:#fff
    classDef analysisStyle fill:#8D6E63,stroke:#4E342E,color:#fff
    
    class SMI inputStyle
    class FRAG,VOC,CORP vocabStyle
    class TOK,TKS tokStyle
    class FG,CG graphStyle
    class ENC,DEC,CKPT trainStyle
    class EMB,DESC embStyle
    class CLUST,DOWN analysisStyle
```

### Module Hierarchy

```mermaid
classDiagram
    class GSGE {
        +VocabularyManager vocab_manager
        +GSGE_tokenizer tokenizer
        +EmbeddingManager embedding_manager
        +GAETrainer gae_trainer
        +DescriptorCalculator descriptor_calculator
        +GraphProcessor graph_processor
        +ClusteringAnalyzer clustering_analyzer
        +save_gsge_data()
        +load_gsge_data()
        +encode_GSGE()
    }
    
    class VocabularyManager {
        +GS_vocab GS_vocab
        +GSGE_corpus GSGE_corpus
        +dict GSGE_vocab
        +set_GS_vocab()
        +set_GSGE_corpus()
        +get_GSGE_vocab_token_to_id_dict()
    }
    
    class EmbeddingManager {
        +dict GS_frag_id_to_embedding
        +ndarray GSGE_combined_embeddings
        +make_GS_fragment_embedding_dict()
        +embed_fragments()
        +load_GAE_weights()
    }
    
    class GAETrainer {
        +train_GSGE_Auto_Encoder()
        +set_encoder()
        +set_decoder()
    }
    
    class DescriptorCalculator {
        +Tensor GSGE_fragment_descriptors
        +calc_fragment_descriptors()
        +get_fragment_descriptors()
    }
    
    class GraphProcessor {
        +make_compound_graphs()
        +check_for_graphs_groupings()
        +get_CG_from_smiles()
    }
    
    class ClusteringAnalyzer {
        +get_GSGE_clustering()
    }
    
    GSGE *-- VocabularyManager
    GSGE *-- EmbeddingManager
    GSGE *-- GAETrainer
    GSGE *-- DescriptorCalculator
    GSGE *-- GraphProcessor
    GSGE *-- ClusteringAnalyzer
    
    note for GSGE "Facade class providing\nunified interface to all\nGSGE functionality"
```

### Typical Usage Workflow

```mermaid
sequenceDiagram
    participant User
    participant GSGE
    participant VM as VocabularyManager
    participant GAE as GAETrainer
    participant EM as EmbeddingManager
    participant DC as DescriptorCalculator
    
    Note over User,DC: Phase 1: Setup & Vocabulary Building
    User->>GSGE: Initialize with SMILES dataset
    GSGE->>VM: Build GS_Vocab (diversity selection)
    GSGE->>VM: Build GSGE_Corpus (all fragments)
    VM-->>GSGE: Token-to-ID mappings ready
    
    Note over User,DC: Phase 2: GAE Training (Optional)
    User->>GSGE: set_encoder(), set_decoder()
    User->>GSGE: train_GSGE_Auto_Encoder()
    GSGE->>GAE: Train on fragment graphs
    GAE-->>GSGE: Checkpoints saved
    User->>GSGE: make_GS_fragment_embedding_dict()
    GSGE->>EM: Generate embeddings
    EM-->>GSGE: Embeddings ready
    
    Note over User,DC: Phase 3: Feature Computation
    User->>GSGE: calc_fragment_descriptors()
    GSGE->>DC: Compute RDKit descriptors
    DC-->>GSGE: Descriptors normalized
    
    Note over User,DC: Phase 4: Save & Reload
    User->>GSGE: save_gsge_data(filepath)
    GSGE-->>User: .pkl file saved
    User->>GSGE: GSGE(load_path=filepath)
    GSGE-->>User: State restored
```

### Core Modules

**`GSGE/gsge.py`** - Main facade class and managers
- `GSGE`: Primary interface for the entire framework
- `VocabularyManager`: Manages GS_Vocab and GSGE_Corpus loading and token-to-ID mappings
- `EmbeddingManager`: Handles GAE embedding generation and loading for fragments
- `GAETrainer`: Wraps GraphAutoencoderTrainer for training fragment autoencoders
- `DescriptorCalculator`: Computes RDKit molecular descriptors for fragments
- `GraphProcessor`: Converts SMILES to PyTorch Geometric graph representations
- `ClusteringAnalyzer`: Performs clustering using embeddings and Maximum Common Substructure (MCS)
- `Store_Data`: Handles pickle-based save/load of complete GSGE state
- `Store_Modules`: Storage container for PyTorch encoder/decoder modules

**`GSGE/vocab.py`** - Vocabulary and corpus management
- `BaseGSVocab`: Abstract base class with fragment canonicalization and persistence
- `GS_Vocab`: Vocabulary builder with diversity selection and core merging (for generalization)
- `GSGE_Corpus`: Corpus builder allowing non-unique fragments (for GAE training data)

**`GSGE/core_gsge.py`** - Core preprocessing functions
- Token-to-ID preprocessing functions
- Static methods for preparing GAE training data
- Parallel tokenization utilities

**`GSGE/tokenizer.py`** - Tokenization logic
- `GSGE_tokenizer`: Converts molecules/SMILES to GSGE token sequences

**`GSGE/chem.py`** - Chemical constants and utilities
- Grammar tokens, element tokens, special tokens
- Common smaller fragments (amide, peptide bond patterns)
- Element bond counts

**`GSGE/fragment_functions.py`** - Custom fragmentation
- `CUSTOM_fragment_mol`: Fragmentation function for cyclic peptides (removes ring bonds, amide bonds, disulfide bonds)

**`GSGE/fragment_tools.py`** - Fragment SMILES utilities
- `FragmentTools`: Base class for fragment canonicalization and SMILES handling
- `GS_FragmentTools`: Extended utilities including `make_element_GS` for creating element fragments

**`GSGE/fragment_descriptors.py`** - Molecular descriptor calculation
- `get_mol_frag_descriptors`: Computes RDKit descriptors for fragments
- `normalize_descriptors`: Normalizes descriptors using training data statistics

**`GSGE/clustering.py`** - Clustering analysis
- `GSGE_clustering`: Clusters fragments using UMAP/t-SNE visualization and hierarchical clustering with MCS

**`GSGE/embedding.py`** - Embedding layer for downstream models
- `GSGE_Embedding`: Combines sparse one-hot encodings (grammar/element tokens) with dense GAE embeddings (fragment tokens)

**`GSGE/plots.py`** - Fragment visualization
- `highlight_fragments`: Highlights molecular fragments in compound visualizations
- Color generation utilities for fragment highlighting

**`GSGE/visualization.py`** - Additional visualization utilities
- `plot_cluster_grid`: Grid visualization of molecules from each cluster
- Static matplotlib-based plots complementing interactive Plotly methods

**`GSGE/utils_chem.py`** - Chemical utilities
- Data validation and NaN checking functions
- Utility functions for chemical data processing

### Graph Modules

**`GSGE/graphs/fragment_graph/GAE.py`** - Graph Autoencoder
- `AttentiveFP`: Attentive Graph Neural Network encoder
- `GraphDecoder`: Decoder for reconstructing molecular graphs
- `GraphAutoencoderTrainer`: Training loop with checkpointing
- `ATOM_MAX_NUM`: Maximum atoms per fragment (default 20)

**`GSGE/graphs/fragment_graph/from_smiles_to_graph.py`**
- `from_smiles`: Converts fragment SMILES to PyTorch Geometric Data object
- `atom_to_token_id`: Mapping from atom types to token IDs

**`GSGE/graphs/fragment_graph/utils_chem.py`** - Chemical utilities for graphs
- Fragment-level chemical property computation

**`GSGE/graphs/fragment_graph/parsed_log_metric_plots.py`** - Training log utilities
- `parse_log_file`: Parses GAE training logs for metric extraction and plotting

**`GSGE/graphs/compound_graph/data.py`** - Compound graph creation
- `compound_graph`: Group-level molecular graph class extending MolecularGraph
- `smiles_to_group_graph`: Converts SMILES to compound graph (fragment nodes)
- `parallel_`: Parallel processing for batch compound graph generation
- `preprocess_graph`: Preprocessing for graph neural networks

### Scripts

**`GSGE/scripts/CLI.py`** - Command-line interface
- `GSGE_CLI run_test`: Run all or specific tests
- Tests are located in `tests/` directory at project root

### Package Exports

**`GSGE/__init__.py`** - Public API
- Main classes: `GSGE`, `GS_Vocab`, `GSGE_Corpus`, `GSGE_tokenizer`, `GSGE_Embedding`
- Utilities: `FragmentTools`, `GS_FragmentTools`, `CoreGSGE`, `highlight_fragments`
- Constants: `_GRAMMAR_TOKENS`, `_ELEMENT_TOKENS`, `_REDIRECT_TOKENS`, `_ELEMENTS_BOND_COUNTS`
- Path utilities: `get_package_resource()`, `get_project_root()`, `get_tests_dir()`, `get_use_examples_dir()`

