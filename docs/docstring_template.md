# GSGE Docstring Style Guide

This document defines the standard format for all docstrings in the GSGE codebase.

## Overview

GSGE uses **Google-style docstrings** with comprehensive documentation for all public APIs.

## Standard Template

### Complete Function/Method Docstring

```python
def function_name(param1: str, param2: int, param3: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Brief one-line description of what the function does.

    More detailed explanation spanning multiple lines if needed. Explain
    the function's purpose, behavior, and important implementation details.
    This section can include:
    - Algorithm descriptions
    - Implementation notes
    - Behavioral characteristics
    - Performance considerations

    Args:
        param1: Description of param1 with valid values and constraints.
            Use indentation for multi-line parameter descriptions.
        param2: Description of param2. Type hints in signature reduce
            verbosity in docstrings.
        param3: Description of optional param3. Always mention default
            behavior when None or default value is used.

    Returns:
        Description of return value. For complex types like Dict, describe
        the structure, keys, and value types. Example:
        Dict with keys:
            - 'status' (str): Success or error status
            - 'value' (int): The computed value
            - 'metadata' (dict): Additional information

    Raises:
        ValueError: When param1 is empty or contains invalid characters.
        TypeError: When param2 is not an integer.
        RuntimeError: When the operation fails due to external conditions.

    Example:
        Basic usage:

        >>> result = function_name("test", 42)
        >>> print(result)
        {'status': 'success', 'value': 42, 'metadata': {}}

        With optional parameter:

        >>> result = function_name("test", 42, ["option1", "option2"])
        >>> print(result['status'])
        'success'

    Note:
        Additional notes, warnings, edge cases, performance considerations,
        or important usage patterns. Use this section for:
        - Thread safety information
        - Side effects
        - Historical context
        - Related functions or methods
    """
    pass
```

### Class Docstring

```python
class ClassName:
    """
    Brief one-line description of the class purpose.

    More detailed explanation of the class, its role in the system,
    and typical usage patterns. Explain what the class represents
    and when to use it.

    Attributes:
        attribute1 (type): Description of attribute1.
        attribute2 (type): Description of attribute2.
        _private_attr (type): Private attributes should be documented
            if they're important for understanding class behavior.

    Example:
        >>> obj = ClassName(param1="value")
        >>> obj.method()
        'result'

    Note:
        Additional information about the class design, performance
        characteristics, or important usage considerations.
    """
```

### Property Docstring

```python
@property
def property_name(self) -> str:
    """
    Brief description of what the property represents.

    Returns:
        Description of the property value.
    """
    return self._property_name
```

### Simple Getter/Setter Docstring

```python
def get_value(self) -> int:
    """Return the current value."""
    return self._value

def set_value(self, value: int) -> None:
    """Set the value to the specified integer."""
    self._value = value
```

## Required Sections by Function Type

### User-Facing Functions (High Priority)

**Required sections:**
- Summary (one-line + detailed)
- Args (all parameters documented)
- Returns (comprehensive structure description)
- Example (at least one working example)
- Raises (if applicable)
- Note (if there are important usage details)

### Internal/Helper Functions

**Required sections:**
- Summary (one-line, brief detailed optional)
- Args (all parameters)
- Returns (brief description)

**Optional:**
- Example (for complex internal functions)
- Raises (for error handling)

### Simple Getters/Setters

**Required:**
- One-line summary only

```python
def get_vocab_size(self) -> int:
    """Return the size of the vocabulary."""
    return len(self.vocab)
```

## Examples by Module

### Clustering Module Example

```python
def plot_2D_TSNE(
    self,
    random_state: int = 42,
    scatter_2d_args: Optional[Dict[str, Any]] = None
) -> None:
    """
    Generate and display a 2D t-SNE visualization of fragment embeddings.

    Performs dimensionality reduction on fragment embeddings using t-SNE
    and creates an interactive Plotly scatter plot. Fragments are colored
    by their cluster assignments.

    Args:
        random_state: Random seed for t-SNE reproducibility. Default is 42.
        scatter_2d_args: Optional dictionary of arguments passed to
            plotly.express.scatter. Common options include 'height' and
            'width'. Defaults to {'height': 800, 'width': 1600}.

    Raises:
        ImportError: If plotly is not installed.
        ValueError: If embeddings have not been generated yet.

    Example:
        >>> clustering = GSGE_clustering(gsge_instance)
        >>> clustering.plot_2D_TSNE(random_state=123)
        # Opens interactive plot in browser

        Custom plot dimensions:

        >>> clustering.plot_2D_TSNE(
        ...     scatter_2d_args={'height': 600, 'width': 1200}
        ... )

    Note:
        Requires plotly to be installed. Install with: pip install plotly
        For large datasets (>10k fragments), consider using UMAP instead
        as it typically performs better.
    """
```

### GSGE Main Class Example

```python
def make_compound_graphs(
    self,
    smiles_list: List[str],
    pyg_data: bool = True,
    workers: int = 1
) -> Union[List[CompoundGraph], List[Data]]:
    """
    Create compound graph representations for a list of SMILES strings.

    Converts SMILES strings into compound graphs where nodes are molecular
    fragments (from the vocabulary) and edges connect bonded fragments.
    Supports both CompoundGraph objects and PyTorch Geometric Data format.

    Args:
        smiles_list: List of SMILES strings representing molecules.
        pyg_data: If True, return PyTorch Geometric Data objects suitable
            for GNN training. If False, return CompoundGraph objects with
            rich visualization methods. Default is True.
        workers: Number of parallel workers for processing. Use workers > 1
            for large datasets. Default is 1 (sequential processing).

    Returns:
        If pyg_data=True: List of torch_geometric.data.Data objects with:
            - x: Node features (fragment embeddings)
            - edge_index: Graph connectivity
            - edge_attr: Edge features
            - smiles: Original SMILES string

        If pyg_data=False: List of CompoundGraph objects with methods:
            - plot_graph_rd_c_style(): Visualize the compound graph
            - get_adjacency_matrix(): Get graph structure
            - to_pyg_data(): Convert to PyG format

    Raises:
        ValueError: If vocabulary is not initialized or empty.
        RuntimeError: If SMILES parsing fails for all molecules.

    Example:
        Create PyG Data objects for GNN training:

        >>> gsge = GSGE(GS_vocab='vocab.pkl')
        >>> smiles = ['CCO', 'CC(C)O', 'c1ccccc1']
        >>> graphs = gsge.01_make_compound_graphs(smiles, pyg_data=True)
        >>> print(type(graphs[0]))
        <class 'torch_geometric.data.Data'>

        Create CompoundGraph objects for visualization:

        >>> graphs = gsge.01_make_compound_graphs(smiles, pyg_data=False)
        >>> graphs[0].plot_graph_rd_c_style()
        # Displays molecular graph with RDKit-style layout

        Parallel processing for large datasets:

        >>> large_smiles_list = [...]  # 10000 SMILES
        >>> graphs = gsge.01_make_compound_graphs(
        ...     large_smiles_list,
        ...     pyg_data=True,
        ...     workers=4
        ... )

    Note:
        Requires vocabulary to be built first via build_vocab() or loaded
        from file. For best performance with workers > 1, use a batch size
        appropriate for your CPU cores. Multiprocessing uses 'spawn' method
        for cross-platform compatibility.
    """
```

### GAE Module Example

```python
def train(
    self,
    num_epochs: int,
    checkpoint_interval: int = 10,
    early_stopping_patience: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Train the Graph Autoencoder for the specified number of epochs.

    Executes the training loop with periodic checkpointing and optional
    early stopping. Tracks atom loss, edge loss, graph size loss, and
    total loss for both training and validation sets.

    Args:
        num_epochs: Total number of training epochs to run.
        checkpoint_interval: Save model checkpoint every N epochs.
            Default is 10. Checkpoints include encoder, decoder, optimizer
            state, and training metrics.
        early_stopping_patience: If specified, stop training if validation
            loss does not improve for N consecutive epochs. None disables
            early stopping. Default is None.

    Returns:
        Dictionary containing training history with keys:
            - 'train_loss': List of training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'train_atom_loss': Atom reconstruction losses
            - 'train_edge_loss': Edge reconstruction losses
            - 'val_atom_loss': Validation atom losses
            - 'val_edge_loss': Validation edge losses

    Raises:
        RuntimeError: If training data loaders are not initialized.
        ValueError: If num_epochs <= 0 or checkpoint_interval <= 0.

    Example:
        Basic training:

        >>> trainer = GraphAutoencoderTrainer(
        ...     encoder, decoder, optimizer,
        ...     train_loader, val_loader,
        ...     device='cuda'
        ... )
        >>> history = trainer.train(num_epochs=100)
        >>> print(f"Final loss: {history['val_loss'][-1]:.4f}")

        Training with checkpointing and early stopping:

        >>> history = trainer.train(
        ...     num_epochs=300,
        ...     checkpoint_interval=5,
        ...     early_stopping_patience=20
        ... )
        >>> # Automatically stops if no improvement for 20 epochs

    Note:
        Checkpoints are saved to self.checkpoint_dir and include:
        - Encoder and decoder model states
        - Optimizer state for resume training
        - Training metrics and epoch number
        - Timestamp of checkpoint

        Use load_checkpoint() to resume training from a saved state.
        For best results, use a learning rate scheduler (not included,
        should be added to optimizer before training).
    """
```

## Docstring Checklist

Before committing docstring changes, verify:

- [ ] **Summary**: One-line description present and clear
- [ ] **Detailed Description**: Multi-line explanation for complex functions
- [ ] **Args**: All parameters documented with types and descriptions
- [ ] **Returns**: Return value structure clearly described
- [ ] **Raises**: All exceptions documented (if applicable)
- [ ] **Example**: At least one working example for user-facing functions
- [ ] **Note**: Important usage details, warnings, or performance notes
- [ ] **Type Hints**: Function signature has proper type annotations
- [ ] **Grammar**: Proper grammar, spelling, and punctuation
- [ ] **Formatting**: Proper indentation and line breaks
- [ ] **Links**: References to related functions/classes (if applicable)

## Anti-Patterns to Avoid

### ❌ Too Vague

```python
def process_data(self, data):
    """Process the data."""
    pass
```

### ✅ Clear and Specific

```python
def process_data(self, data: List[str]) -> List[Dict[str, Any]]:
    """
    Convert SMILES strings to molecular fragment dictionaries.

    Args:
        data: List of SMILES strings to process.

    Returns:
        List of dictionaries, each containing fragment information.
    """
    pass
```

### ❌ Repeating Type Hints

```python
def calculate(self, x: int, y: int) -> int:
    """
    Calculate something.

    Args:
        x (int): First integer.
        y (int): Second integer.

    Returns:
        int: The result.
    """
    pass
```

### ✅ Focus on Meaning, Not Types

```python
def calculate(self, x: int, y: int) -> int:
    """
    Calculate the sum of two integers.

    Args:
        x: First addend.
        y: Second addend.

    Returns:
        The sum of x and y.
    """
    pass
```

### ❌ Missing Examples for Complex Functions

```python
def parallel_tokenize_SMILES_list(
    self, smiles_list, max_length=512, workers=4
):
    """Tokenize SMILES in parallel."""
    pass
```

### ✅ Include Working Examples

```python
def parallel_tokenize_SMILES_list(
    self,
    smiles_list: List[str],
    max_length: int = 512,
    workers: int = 4
) -> pd.DataFrame:
    """
    Tokenize a list of SMILES strings in parallel.

    Args:
        smiles_list: SMILES strings to tokenize.
        max_length: Maximum sequence length for padding.
        workers: Number of parallel workers.

    Returns:
        DataFrame with columns 'smiles', 'tokens', 'token_ids',
        'attention_mask'.

    Example:
        >>> tokenizer = GSGE_tokenizer(vocab_manager)
        >>> smiles = ['CCO', 'CC(C)O', 'c1ccccc1']
        >>> df = tokenizer.parallel_tokenize_SMILES_list(
        ...     smiles, max_length=128, workers=2
        ... )
        >>> print(df.columns)
        Index(['smiles', 'tokens', 'token_ids', 'attention_mask'])
    """
    pass
```

## Tools for Validation

### Manual Review

```bash
# Search for functions without docstrings
grep -E "^[[:space:]]*def " file.py | grep -v '"""'
```

### Coverage Measurement

```python
# Use interrogate for docstring coverage
pip install interrogate
interrogate -vv GSGE/
```

### Example Test

```python
# Test docstrings are valid
import doctest
doctest.testmod(module_name, verbose=True)
```

## References

- [Google Python Style Guide - Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- [PEP 257 - Docstring Conventions](https://peps.python.org/pep-0257/)
- [Napoleon - Google Style Documentation](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

---

*Last Updated: December 2025*
*GSGE Project Improvement - Phase 1*
