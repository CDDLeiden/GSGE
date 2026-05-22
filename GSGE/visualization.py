"""
Visualization utilities for GSGE fragment analysis.

This module provides standalone helper functions for visualizing molecular fragments,
embedding distributions, and clustering results. These complement the existing
Plotly-based interactive methods in GSGE_clustering by providing matplotlib-based
static visualizations and additional plot types.

Note: For interactive t-SNE/UMAP plots, use GSGE_clustering.plot_2D_TSNE(),
plot_2D_UMAP(), plot_3D_TSNE(), or plot_3D_UMAP() instead.
"""

from typing import List, Optional, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from rdkit import Chem
from rdkit.Chem import Draw
import random

def plot_cluster_grid(
    smiles_list: List[str],
    labels: np.ndarray,
    color_map: Dict[int, Tuple[float, float, float]],
    samples_per_cluster: int = 5,
    figsize_per_sample: int = 3,
    seed: Optional[int] = None
) -> plt.Figure:
    """
    Plot a grid showing example molecules from each cluster.

    Creates a grid visualization displaying random samples from each cluster,
    showing the actual chemical structures. This complements the embedding
    space plots by showing what fragments are in each cluster.

    Args:
        smiles_list: List of SMILES strings for all fragments
        labels: Cluster labels for each SMILES (same length as smiles_list)
        color_map: Mapping from cluster ID to RGB color tuple (0-1 range)
        samples_per_cluster: Number of examples to show per cluster
        figsize_per_sample: Size of each subplot in inches
        seed: Random seed for reproducibility. If None, visualization will vary between runs.

    Returns:
        matplotlib Figure object with cluster grid

    Example:
        >>> from GSGE.visualization import plot_cluster_grid
        >>> import seaborn as sns
        >>> import numpy as np
        >>>
        >>> # Create color map
        >>> unique_clusters = np.unique(labels)
        >>> palette = sns.color_palette("hsv", len(unique_clusters))
        >>> color_map = {label: palette[i] for i, label in enumerate(unique_clusters)}
        >>>
        >>> # Plot clusters with reproducible results
        >>> fig = plot_cluster_grid(smiles_list, labels, color_map, seed=42)
        >>> plt.show()

    Note:
        - Invalid SMILES are displayed as text placeholders
        - Clusters with label -1 (noise) are excluded from display
        - Samples are randomly selected from each cluster
        - Use the seed parameter for reproducible visualizations in documentation or debugging
        - This is a static matplotlib alternative to interactive Plotly visualizations
    """
    if seed is not None:
        random.seed(seed)
    unique_labels = sorted([l for l in np.unique(labels) if l != -1])

    fig, axes = plt.subplots(
        samples_per_cluster,
        len(unique_labels),
        figsize=(figsize_per_sample * len(unique_labels), figsize_per_sample * samples_per_cluster)
    )

    # Ensure axes is 2D
    if samples_per_cluster == 1:
        axes = axes[np.newaxis, :]
    if len(unique_labels) == 1:
        axes = axes[:, np.newaxis]

    for col, label in enumerate(unique_labels):
        # Get fragments in this cluster
        idxs = [i for i, lbl in enumerate(labels) if lbl == label]

        # Sample randomly
        chosen = random.sample(idxs, min(samples_per_cluster, len(idxs)))

        for row in range(samples_per_cluster):
            ax = axes[row, col]
            ax.axis("off")

            if row < len(chosen):
                smi = smiles_list[chosen[row]]
                mol = Chem.MolFromSmiles(smi.replace('*1', '*'))
                if mol is not None:
                    img = Draw.MolToImage(mol, size=(200, 200))
                    ax.imshow(img)
                else:
                    ax.text(0.5, 0.5, 'Invalid\nSMILES',
                           ha='center', va='center', fontsize=12)

        # Add cluster title
        rgb = color_map[label]
        hex_color = mcolors.to_hex(rgb)
        axes[0, col].set_title(
            f"Cluster {label}",
            fontsize=16,
            color=hex_color,
            pad=20,
            fontweight='bold'
        )

    plt.tight_layout()
    return fig


def plot_descriptor_distribution(
    descriptors: np.ndarray,
    descriptor_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (14, 10),
    max_descriptors: int = 48
) -> plt.Figure:
    """
    Plot distribution of molecular descriptors as a grid of histograms.

    Creates a grid visualization showing the distribution of each descriptor,
    useful for understanding descriptor ranges, identifying outliers, and
    checking for normalization needs.

    Args:
        descriptors: Descriptor matrix (n_samples, n_descriptors)
        descriptor_names: Optional names for each descriptor column
        figsize: Figure size as (width, height) in inches
        max_descriptors: Maximum number of descriptors to plot (for grid sizing)

    Returns:
        matplotlib Figure object with descriptor distribution grid

    Example:
        >>> from GSGE.visualization import plot_descriptor_distribution
        >>> from GSGE.fragment_descriptors import get_mol_frag_descriptors
        >>>
        >>> # Calculate descriptors
        >>> descriptors = get_mol_frag_descriptors(gsge)
        >>>
        >>> # Plot distributions
        >>> fig = plot_descriptor_distribution(
        ...     descriptors,
        ...     descriptor_names=['MolWt', 'TPSA', 'NumHDonors', ...]
        ... )
        >>> plt.show()

    Note:
        - Grid layout is automatically calculated based on number of descriptors
        - Uses efficient grid calculation (approximately square layout)
        - Useful for identifying descriptor normalization needs
    """
    n_descriptors = descriptors.shape[1]
    n_cols = int(np.ceil(np.sqrt(min(n_descriptors, max_descriptors))))
    n_rows = int(np.ceil(min(n_descriptors, max_descriptors) / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows > 1:
        axes_flat = axes.flat
    elif n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes

    for i in range(min(n_descriptors, max_descriptors)):
        ax = axes_flat[i]
        data = descriptors[:, i]

        ax.hist(data, bins=30, edgecolor='black', alpha=0.7)
        ax.set_title(descriptor_names[i] if descriptor_names else f'Descriptor {i}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')

    # Hide unused subplots
    for i in range(min(n_descriptors, max_descriptors), len(axes_flat)):
        axes_flat[i].axis('off')

    plt.tight_layout()
    return fig
