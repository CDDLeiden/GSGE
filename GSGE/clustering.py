from __future__ import annotations

from .graphs.fragment_graph.GAE import ATOM_MAX_NUM
from  .graphs.fragment_graph.from_smiles_to_graph import from_smiles, atom_to_token_id
import pandas as pd
import numpy as np

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .gsge import GSGE

class GSGE_clustering:
    """
    Clustering and visualization utilities for molecular fragment embeddings.

    Provides tools for embedding molecular fragments using trained Graph Autoencoder
    models, clustering fragments based on Maximum Common Substructure (MCS) similarity,
    and visualizing fragment chemical space using dimensionality reduction techniques
    (t-SNE and UMAP).

    Attributes:
        gsge (GSGE): GSGE instance containing vocabulary and encoder.
        embeddings (np.ndarray | None): Fragment embeddings from trained GAE model.
        smiles_df (pd.DataFrame | None): DataFrame containing fragment SMILES strings.
        smiles_column (str): Name of column containing SMILES in smiles_df.
        cluster_labels (np.ndarray | None): Cluster assignments for fragments.
        graph_data (List | None): Graph representations of fragments (set by _embed_fragments).
        plot_df (pd.DataFrame | None): DataFrame with dimensionality reduction results (set by plotting methods).
        fig (plotly.graph_objs.Figure | None): Last generated plot figure.

    Example:
        >>> from GSGE import GSGE
        >>> gsge = GSGE(GSGE_load_path='gsge_save.pkl')
        >>> clustering = GSGE_clustering(
        ...     gsge=gsge,
        ...     embeddings=None,
        ...     smiles_df=None,
        ...     cluster_labels=None
        ... )
        >>> clustering._embed_fragments(frag_smiles=None)
        >>> clustering._cluster()
        >>> fig = clustering.plot_2D_TSNE()
        >>> fig.show()

    Note:
        Requires trained GAE encoder to generate embeddings. Use
        gsge.set_encoder() and gsge.load_GAE_weights() before embedding.
        Visualization methods require plotly: pip install plotly
        UMAP visualization methods also require umap-learn: pip install umap-learn
    """

    def __init__(
        self,
        gsge: GSGE,
        embeddings: None | np.ndarray = None,
        smiles_df: None | pd.DataFrame = None,
        cluster_labels: None | list = None,
        smiles_column: str = 'SMILES'
    ):
        """
        Initialize GSGE_clustering with embeddings and clustering configuration.

        Args:
            gsge: GSGE instance with vocabulary and optionally loaded encoder.
            embeddings: Pre-computed fragment embeddings. If None, call
                _embed_fragments() to generate embeddings.
            smiles_df: DataFrame containing fragment SMILES. Created automatically
                by _embed_fragments() if None.
            cluster_labels: Pre-computed cluster assignments. If None, call
                _cluster() to generate clusters.
            smiles_column: Name of SMILES column in smiles_df. Default is 'SMILES'.

        Example:
            Create clustering instance and generate embeddings:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments(frag_smiles=None)

            Create clustering instance with pre-computed embeddings:

            >>> clustering = GSGE_clustering(
            ...     gsge=gsge_instance,
            ...     embeddings=embeddings_array,
            ...     smiles_df=smiles_dataframe,
            ...     cluster_labels=labels_array
            ... )
        """
        self.gsge = gsge
        self.embeddings = embeddings
        self.smiles_df = smiles_df
        self.smiles_column = smiles_column
        self.cluster_labels = cluster_labels
      
    def _embed_fragments(
        self,
        frag_smiles: list | None = None,
        encoder: None = None,
        load_checkpoint_path: None | str = None,
        map_location: str = 'cuda',
        max_atom_size: int = ATOM_MAX_NUM,
        process_smiles=from_smiles,
        atom_to_token_id: dict = atom_to_token_id,
        device: str = 'cuda',
        batch_size: int = 64,
    ):
        """
        Generate embeddings for molecular fragments using trained GAE encoder.

        Converts fragment SMILES strings to graph representations, processes them
        through the trained Graph Autoencoder encoder, and stores the resulting
        latent embeddings. Updates self.embeddings, self.graph_data, and self.smiles_df.

        Args:
            frag_smiles: List of fragment SMILES strings to embed. If None, uses
                all fragments from gsge.vocab_manager.GS_vocab.fragments.
            encoder: Pre-initialized encoder model. If None, uses self.gsge.encoder
                which should be loaded via gsge.set_encoder() and gsge.load_GAE_weights().
            load_checkpoint_path: Path to GAE checkpoint file to load encoder weights.
                If None and encoder is None, uses weights already loaded in gsge.encoder.
            map_location: Device to load checkpoint weights ('cuda' or 'cpu').
                Default is 'cuda'.
            max_atom_size: Maximum number of atoms in fragment graphs. Fragments
                exceeding this size will be truncated. Default is ATOM_MAX_NUM.
            process_smiles: Function to convert SMILES to PyG Data objects.
                Default is from_smiles from fragment_graph module.
            atom_to_token_id: Dictionary mapping atom symbols to token IDs for
                feature encoding. Default is atom_to_token_id from fragment_graph.
            device: Device for computation ('cuda' or 'cpu'). Default is 'cuda'.
            batch_size: Number of fragments to process per batch. Larger batches
                are faster but require more memory. Default is 64.

        Raises:
            RuntimeError: If encoder is not initialized and cannot be loaded.
            ValueError: If fragment SMILES cannot be parsed.

        Example:
            Embed vocabulary fragments with default settings:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments(frag_smiles=None)
            >>> print(clustering.embeddings.shape)
            (200, 128)  # 200 fragments, 128-dim embeddings

            Embed specific fragments on CPU:

            >>> custom_frags = ['CC(*)O', 'c1ccccc1(*)', 'N=C(*)N']
            >>> clustering._embed_fragments(
            ...     frag_smiles=custom_frags,
            ...     device='cpu',
            ...     batch_size=32
            ... )

        Note:
            Sets self.embeddings (np.ndarray), self.graph_data (list of Data objects),
            and self.smiles_df (DataFrame with 'SMILES' column). Requires encoder
            weights to be loaded before calling.
        """
        self.embeddings, self.graph_data = self.gsge.embed_fragments(
            frag_smiles=self.gsge.vocab_manager.GS_vocab.fragments if frag_smiles is None else frag_smiles,
            encoder=encoder,  # will load self.encoder in embed_fragments if encoder is None
            load_checkpoint_path=load_checkpoint_path,
            map_location=map_location,
            max_atom_size=max_atom_size,
            process_smiles=process_smiles,
            atom_to_token_id=atom_to_token_id,
            batch_size=batch_size,
            device=device,
            return_data=True,
        )

        self.smiles_df = pd.DataFrame({'SMILES': [frag_.smiles for frag_ in self.graph_data]})

    @staticmethod
    def _MCS_clustering(
        df: pd.DataFrame,
        fcluster_args: dict = {'t': 1.5, 'criterion': 'distance'},
        hierarchical_clustering_args: dict = {}
    ):
        """
        Cluster molecular fragments using Maximum Common Substructure (MCS) similarity.

        Performs hierarchical clustering based on Maximum Common Substructure similarity
        between molecular fragments. Fragments with higher structural similarity are
        grouped into the same cluster.

        Args:
            df: DataFrame containing fragment SMILES strings. Must have a 'SMILES'
                column for MCS comparison.
            fcluster_args: Arguments for scipy.cluster.hierarchy.fcluster() to form
                flat clusters from hierarchical linkage. Common arguments:
                    - 't': Threshold for cluster formation (default: 1.5)
                    - 'criterion': Clustering criterion ('distance', 'inconsistent',
                      'maxclust', etc.). Default is 'distance'.
            hierarchical_clustering_args: Arguments passed to hierarchical_clustering()
                function for computing MCS-based distance matrix and linkage.

        Returns:
            Cluster labels array where each element is the cluster ID for the
            corresponding fragment. Shape: (n_fragments,)

        Raises:
            ValueError: If required modules (scipy) are not installed.
            ImportError: If utils_chem module cannot be imported.

        Example:
            Basic clustering with default parameters:

            >>> smiles_df = pd.DataFrame({'SMILES': ['CC(*)O', 'CCC(*)O', 'c1ccccc1(*)']})
            >>> labels = GSGE_clustering._MCS_clustering(smiles_df)
            >>> print(labels)
            [1 1 2]  # First two fragments in same cluster (similar structure)

            Custom clustering threshold:

            >>> labels = GSGE_clustering._MCS_clustering(
            ...     smiles_df,
            ...     fcluster_args={'t': 2.0, 'criterion': 'distance'}
            ... )

        Note:
            MCS clustering can be computationally expensive for large numbers of fragments
            (>1000) due to pairwise comparison complexity. For large datasets, consider
            using embedding-based clustering (k-means on embeddings) instead.
        """
        try:
            from utils_chem import hierarchical_clustering, form_linkage
            from scipy.cluster.hierarchy import fcluster
            import pandas as pd
        except ModuleNotFoundError as e:
            raise ValueError(f"Module import error: {e}")
        except ImportError as e:
            raise ValueError(f"Import error: {e}")

        # Perform hierarchical clustering (linkage matrix)
        _, Z = form_linkage(hierarchical_clustering(df, **hierarchical_clustering_args))

        # Convert linkage matrix to cluster labels
        cluster_labels = fcluster(Z, **fcluster_args)
        return cluster_labels
    
    def _cluster(self, method=None, method_args: dict = {}):
        """
        Cluster fragments using specified method and store cluster labels.

        Applies clustering algorithm to fragment SMILES in self.smiles_df and stores
        the resulting cluster assignments in self.cluster_label. By default, uses
        Maximum Common Substructure (MCS) similarity clustering.

        Args:
            method: Custom clustering method or pre-computed cluster labels. If None,
                uses MCS clustering via _MCS_clustering(). If array-like, directly
                assigns as cluster labels. If callable, should accept smiles_df and
                return cluster labels.
            method_args: Arguments passed to _MCS_clustering() when method is None.
                See _MCS_clustering() docstring for available parameters.

        Example:
            Use default MCS clustering:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> print(clustering.cluster_label)
            [1 1 2 2 3 1 ...]

            Custom MCS clustering threshold:

            >>> clustering._cluster(method_args={
            ...     'fcluster_args': {'t': 2.0, 'criterion': 'distance'}
            ... })

            Use pre-computed labels:

            >>> custom_labels = np.array([1, 1, 2, 2, 3, 3])
            >>> clustering._cluster(method=custom_labels)

        Note:
            Sets self.cluster_label attribute. Requires self.smiles_df to be
            populated (call _embed_fragments() first).
        """
        self.cluster_label = GSGE_clustering._MCS_clustering(self.smiles_df, **method_args) if method is None else method
    
    def plot_2D_TSNE(self, random_state: int = 42, scatter_2d_args: dict = {'height': 800, 'width': 1600}):
        """
        Generate 2D t-SNE visualization of fragment embeddings colored by cluster.

        Performs t-Distributed Stochastic Neighbor Embedding (t-SNE) dimensionality
        reduction on fragment embeddings to 2D space and creates an interactive
        Plotly scatter plot. Fragments are colored and symbolized by their cluster
        assignments for easy visual interpretation of chemical space.

        Args:
            random_state: Random seed for t-SNE reproducibility. Same seed produces
                identical visualizations across runs. Default is 42.
            scatter_2d_args: Dictionary of keyword arguments passed to plotly.express.scatter().
                Common options include:
                    - 'height': Plot height in pixels (default: 800)
                    - 'width': Plot width in pixels (default: 1600)
                    - 'template': Plotly template ('plotly', 'plotly_white', 'ggplot2', etc.)

        Returns:
            Plotly Figure object containing the 2D scatter plot. Can be displayed
            with fig.show() or saved with fig.write_html().

        Raises:
            ImportError: If plotly is not installed.
            AttributeError: If self.embeddings or self.cluster_labels not set.

        Example:
            Basic 2D t-SNE visualization:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> fig = clustering.plot_2D_TSNE()
            >>> fig.show()  # Opens in browser

            Custom plot dimensions and template:

            >>> fig = clustering.plot_2D_TSNE(
            ...     random_state=123,
            ...     scatter_2d_args={'height': 600, 'width': 1200, 'template': 'plotly_white'}
            ... )
            >>> fig.write_html('tsne_plot.html')

        Note:
            - Sets self.plot_df (DataFrame with TSNE-1, TSNE-2, SMILES columns)
            - Sets self.fig (Plotly figure object)
            - Requires plotly: pip install plotly
            - For large datasets (>5000 fragments), t-SNE may be slow. Consider
              using UMAP instead (plot_2D_UMAP) for better performance.
            - Each cluster gets a unique color and symbol for easy identification.
        """
        from sklearn.manifold import TSNE
        try:
            import plotly.express as px
        except ImportError:
            print("The 'plotly' library is not installed. Please install it using 'pip install plotly'.")

        tsne = TSNE(n_components=2, random_state=random_state)
        tsne_results = tsne.fit_transform(self.embeddings)

        cluster_labels_frag_str = self.cluster_labels.astype(str)

        df = pd.DataFrame({
            'TSNE-1': tsne_results[:, 0],
            'TSNE-2': tsne_results[:, 1],
            'SMILES': self.smiles_df['SMILES'],
            'Cluster': ['Cluster ' + l for l in cluster_labels_frag_str],
        })

        fig = px.scatter(
            df, x="TSNE-1", y="TSNE-2",
            hover_name="SMILES",
            color="Cluster",
            symbol="Cluster",
            title="2D t-SNE of Molecular Embeddings",
            **scatter_2d_args
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
        )
        fig.update_xaxes(showline=True, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linecolor='black', mirror=True)

        self.plot_df = df
        self.fig = fig

        return fig

    def plot_3D_TSNE(self, random_state: int = 42, scatter_3d_args: dict = {'height': 800, 'width': 1600}):
        """
        Generate 3D t-SNE visualization of fragment embeddings colored by cluster.

        Performs t-Distributed Stochastic Neighbor Embedding (t-SNE) dimensionality
        reduction on fragment embeddings to 3D space and creates an interactive
        Plotly 3D scatter plot. The 3D visualization can reveal cluster structure
        not visible in 2D projections. Fragments are colored by cluster assignments.

        Args:
            random_state: Random seed for t-SNE reproducibility. Same seed produces
                identical visualizations across runs. Default is 42.
            scatter_3d_args: Dictionary of keyword arguments passed to plotly.express.scatter_3d().
                Common options include:
                    - 'height': Plot height in pixels (default: 800)
                    - 'width': Plot width in pixels (default: 1600)
                    - 'template': Plotly template ('plotly', 'plotly_white', etc.)

        Returns:
            Plotly Figure object containing the 3D scatter plot. Interactive plot
            allows rotation, zoom, and hover to explore chemical space. Can be
            displayed with fig.show() or saved with fig.write_html().

        Raises:
            ImportError: If plotly or sklearn is not installed.
            AttributeError: If self.embeddings or self.cluster_labels not set.

        Example:
            Basic 3D t-SNE visualization:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> fig = clustering.plot_3D_TSNE()
            >>> fig.show()  # Opens interactive 3D plot in browser

            Custom visualization with different random seed:

            >>> fig = clustering.plot_3D_TSNE(
            ...     random_state=456,
            ...     scatter_3d_args={'height': 900, 'width': 1400, 'template': 'plotly_dark'}
            ... )
            >>> fig.write_html('tsne_3d_plot.html')

            Explore clusters interactively:

            >>> fig = clustering.plot_3D_TSNE()
            >>> # Rotate plot to view from different angles
            >>> # Hover over points to see SMILES
            >>> fig.show()

        Note:
            - Sets self.plot_df (DataFrame with TSNE-1, TSNE-2, TSNE-3, SMILES columns)
            - Sets self.fig (Plotly 3D figure object)
            - Requires plotly: pip install plotly
            - 3D t-SNE provides additional dimensionality compared to 2D, which can
              better preserve local and global structure of high-dimensional data.
            - For large datasets (>5000 fragments), computation may take several
              minutes. Consider downsampling or using 2D t-SNE for exploration first.
            - Interactive 3D plots allow rotation and zoom for exploring cluster
              boundaries and overlap.
        """
        from sklearn.manifold import TSNE

        try:
            import plotly.express as px
        except ImportError:
            print("The 'plotly' library is not installed. Please install it using 'pip install plotly'.")

        tsne = TSNE(n_components=3, random_state=random_state)
        tsne_results = tsne.fit_transform(self.embeddings)

        cluster_labels_str = ['Cluster ' + s for s in self.cluster_labels.astype(str)]

        df = pd.DataFrame({
            'TSNE-1': tsne_results[:, 0],
            'TSNE-2': tsne_results[:, 1],
            'TSNE-3': tsne_results[:, 2],
            'SMILES': self.smiles_df['SMILES'],
            'Cluster': cluster_labels_str,
        })

        fig = px.scatter_3d(
            df,
            x="TSNE-1",
            y="TSNE-2",
            z="TSNE-3",
            hover_name="SMILES",
            color="Cluster",
            title="3D t-SNE of Molecular Embeddings",
            **scatter_3d_args
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
        )

        self.plot_df = df
        self.fig = fig

        return fig

    def plot_2D_UMAP(self, random_state: int = 42, scatter_2d_args: dict = {'height': 800, 'width': 1600}):
        """
        Generate 2D UMAP visualization of fragment embeddings colored by cluster.

        Performs Uniform Manifold Approximation and Projection (UMAP) dimensionality
        reduction on fragment embeddings to 2D space and creates an interactive
        Plotly scatter plot. Fragments are colored and symbolized by their cluster
        assignments for easy visual interpretation of chemical space. UMAP typically
        preserves more global structure than t-SNE and is faster for large datasets.

        Args:
            random_state: Random seed for UMAP reproducibility. Same seed produces
                identical visualizations across runs. Default is 42.
            scatter_2d_args: Dictionary of keyword arguments passed to plotly.express.scatter().
                Common options include:
                    - 'height': Plot height in pixels (default: 800)
                    - 'width': Plot width in pixels (default: 1600)
                    - 'template': Plotly template ('plotly', 'plotly_white', 'ggplot2', etc.)

        Returns:
            Plotly Figure object containing the 2D scatter plot. Can be displayed
            with fig.show() or saved with fig.write_html().

        Raises:
            ImportError: If plotly or umap-learn is not installed.
            AttributeError: If self.embeddings or self.cluster_labels not set.

        Example:
            Basic 2D UMAP visualization:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> fig = clustering.plot_2D_UMAP()
            >>> fig.show()  # Opens in browser

            Custom plot dimensions and template:

            >>> fig = clustering.plot_2D_UMAP(
            ...     random_state=123,
            ...     scatter_2d_args={'height': 600, 'width': 1200, 'template': 'plotly_white'}
            ... )
            >>> fig.write_html('umap_plot.html')

        Note:
            - Sets self.plot_df (DataFrame with UMAP-1, UMAP-2, SMILES columns)
            - Sets self.fig (Plotly figure object)
            - Requires plotly: pip install plotly
            - Requires umap-learn: pip install umap-learn
            - UMAP is generally faster than t-SNE and better preserves global structure.
            - Each cluster gets a unique color and symbol for easy identification.
            - For very large datasets (>10000 fragments), UMAP is recommended over t-SNE.
        """
        try:
            import umap
            import plotly.express as px
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else "required library"
            print(f"The '{missing_lib}' library is not installed. Please install it using 'pip install {missing_lib}'.")

        umap_model = umap.UMAP(n_components=2, random_state=random_state)
        umap_results = umap_model.fit_transform(self.embeddings)

        cluster_labels_frag_str = self.cluster_labels.astype(str)

        df = pd.DataFrame({
            'UMAP-1': umap_results[:, 0],
            'UMAP-2': umap_results[:, 1],
            'SMILES': self.smiles_df['SMILES'],
            'Cluster': ['Cluster ' + l for l in cluster_labels_frag_str],
        })

        fig = px.scatter(
            df, x="UMAP-1", y="UMAP-2",
            hover_name="SMILES",
            color="Cluster",
            symbol="Cluster",
            title="2D UMAP of Molecular Embeddings",
            **scatter_2d_args
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
        )
        fig.update_xaxes(showline=True, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linecolor='black', mirror=True)

        self.plot_df = df
        self.fig = fig

        return fig

    def plot_2D_PCA(self, scatter_2d_args: dict = {'height': 800, 'width': 1600}):
        """
        Generate 2D PCA visualization of fragment embeddings colored by cluster.

        Performs Principal Component Analysis (PCA) dimensionality reduction on
        fragment embeddings to 2D space and creates an interactive Plotly scatter
        plot. PCA is a linear method that preserves global variance structure and
        is deterministic (no random seed needed). Axis labels include the
        percentage of variance explained by each component.

        Args:
            scatter_2d_args: Dictionary of keyword arguments passed to plotly.express.scatter().
                Common options include:
                    - 'height': Plot height in pixels (default: 800)
                    - 'width': Plot width in pixels (default: 1600)

        Returns:
            Plotly Figure object containing the 2D scatter plot.

        Example:
            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> fig = clustering.plot_2D_PCA()
            >>> fig.show()
        """
        from sklearn.decomposition import PCA
        try:
            import plotly.express as px
        except ImportError:
            print("The 'plotly' library is not installed. Please install it using 'pip install plotly'.")

        pca = PCA(n_components=2)
        pca_results = pca.fit_transform(self.embeddings)
        var_explained = pca.explained_variance_ratio_ * 100

        cluster_labels_frag_str = self.cluster_labels.astype(str)

        df = pd.DataFrame({
            'PC-1': pca_results[:, 0],
            'PC-2': pca_results[:, 1],
            'SMILES': self.smiles_df['SMILES'],
            'Cluster': ['Cluster ' + l for l in cluster_labels_frag_str],
        })

        fig = px.scatter(
            df, x="PC-1", y="PC-2",
            hover_name="SMILES",
            color="Cluster",
            symbol="Cluster",
            title="2D PCA of Molecular Embeddings",
            labels={
                'PC-1': f'PC-1 ({var_explained[0]:.1f}%)',
                'PC-2': f'PC-2 ({var_explained[1]:.1f}%)',
            },
            **scatter_2d_args
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
        )
        fig.update_xaxes(showline=True, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linecolor='black', mirror=True)

        self.plot_df = df
        self.fig = fig

        return fig

    def _make_corpus_overlay_figure(
        self,
        vocab_2d: np.ndarray,
        corpus_2d: np.ndarray,
        axis_prefix: str,
        title: str,
        corpus_smiles: list[str] | None = None,
        scatter_2d_args: dict | None = None,
        corpus_marker_args: dict | None = None,
    ):
        """
        Build a layered plotly figure with corpus points behind vocab points.

        Args:
            vocab_2d: 2D coordinates for vocab fragments, shape (n_vocab, 2).
            corpus_2d: 2D coordinates for corpus fragments, shape (n_corpus, 2).
            axis_prefix: Label prefix for axes (e.g. 'TSNE' or 'UMAP').
            title: Figure title.
            corpus_smiles: SMILES strings for corpus hover text.
            scatter_2d_args: Layout kwargs (height, width, …).
            corpus_marker_args: Marker style overrides for corpus trace.

        Returns:
            plotly.graph_objs.Figure
        """
        import plotly.graph_objects as go

        if scatter_2d_args is None:
            scatter_2d_args = {'height': 800, 'width': 1600}
        if corpus_marker_args is None:
            corpus_marker_args = {}

        corpus_marker = dict(
            color='#808080',
            size=4,
            opacity=0.55,
        )
        corpus_marker.update(corpus_marker_args)

        fig = go.Figure()

        # --- corpus background layer ---
        corpus_hover = corpus_smiles if corpus_smiles is not None else None
        fig.add_trace(go.Scatter(
            x=corpus_2d[:, 0],
            y=corpus_2d[:, 1],
            mode='markers',
            marker=corpus_marker,
            text=corpus_hover,
            hoverinfo='text' if corpus_hover is not None else 'skip',
            name='Corpus fragments',
            showlegend=True,
        ))

        # --- vocab foreground layer (one trace per cluster) ---
        # Publication-quality palette: Okabe-Ito (colorblind-safe) extended with
        # additional distinguishable colors for larger cluster counts.
        _PUB_PALETTE = [
            '#0072B2',  # blue
            '#E69F00',  # orange
            '#009E73',  # bluish green
            '#D55E00',  # vermillion
            '#CC79A7',  # reddish purple
            '#56B4E9',  # sky blue
            '#F0E442',  # yellow
            '#882255',  # wine / dark magenta
            '#44AA99',  # teal
            '#332288',  # indigo
            '#117733',  # dark green
            '#AA4499',  # purple
        ]

        cluster_labels_str = self.cluster_labels.astype(str)
        unique_labels = sorted(set(cluster_labels_str), key=lambda x: int(x))
        color_map = {
            label: _PUB_PALETTE[i % len(_PUB_PALETTE)]
            for i, label in enumerate(unique_labels)
        }
        symbols = ['circle', 'square', 'diamond', 'cross', 'x',
                   'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star']

        for i, label in enumerate(unique_labels):
            mask = cluster_labels_str == label
            fig.add_trace(go.Scatter(
                x=vocab_2d[mask, 0],
                y=vocab_2d[mask, 1],
                mode='markers',
                marker=dict(
                    color=color_map[label],
                    size=8,
                    symbol=symbols[i % len(symbols)],
                    line=dict(width=0.5, color='DarkSlateGrey'),
                ),
                text=self.smiles_df['SMILES'][mask].values,
                hoverinfo='text',
                name=f'Cluster {label}',
                showlegend=True,
            ))

        fig.update_layout(
            title=title,
            xaxis_title=f'{axis_prefix}-1',
            yaxis_title=f'{axis_prefix}-2',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
            **scatter_2d_args,
        )
        fig.update_xaxes(showline=True, linecolor='black', mirror=True)
        fig.update_yaxes(showline=True, linecolor='black', mirror=True)

        return fig

    def plot_2D_TSNE_with_corpus(
        self,
        corpus_embeddings: np.ndarray,
        corpus_smiles: list[str] | None = None,
        random_state: int = 42,
        scatter_2d_args: dict | None = None,
        corpus_marker_args: dict | None = None,
    ):
        """
        2D t-SNE with corpus fragments as grey background and vocab coloured by cluster.

        Runs t-SNE on the **combined** vocab + corpus embeddings so both sets
        share the same coordinate space, then plots corpus as light-grey
        background markers with vocab fragments coloured by cluster on top.

        Args:
            corpus_embeddings: Embeddings for corpus fragments, shape (n_corpus, dim).
            corpus_smiles: SMILES for corpus hover text.
            random_state: Random seed for t-SNE.
            scatter_2d_args: Layout kwargs forwarded to plotly (height, width, …).
            corpus_marker_args: Marker style overrides for corpus points
                (defaults: color='lightgrey', size=4, opacity=0.4).

        Returns:
            plotly.graph_objs.Figure
        """
        from sklearn.manifold import TSNE

        combined = np.vstack([self.embeddings, corpus_embeddings])
        tsne = TSNE(n_components=2, random_state=random_state)
        coords = tsne.fit_transform(combined)

        n_vocab = len(self.embeddings)
        vocab_2d = coords[:n_vocab]
        corpus_2d = coords[n_vocab:]

        fig = self._make_corpus_overlay_figure(
            vocab_2d=vocab_2d,
            corpus_2d=corpus_2d,
            axis_prefix='TSNE',
            title='2D t-SNE — Vocab vs Corpus Fragments',
            corpus_smiles=corpus_smiles,
            scatter_2d_args=scatter_2d_args,
            corpus_marker_args=corpus_marker_args,
        )

        self.fig = fig
        return fig

    def plot_2D_UMAP_with_corpus(
        self,
        corpus_embeddings: np.ndarray,
        corpus_smiles: list[str] | None = None,
        random_state: int = 42,
        scatter_2d_args: dict | None = None,
        corpus_marker_args: dict | None = None,
    ):
        """
        2D UMAP with corpus fragments as grey background and vocab coloured by cluster.

        Runs UMAP on the **combined** vocab + corpus embeddings so both sets
        share the same coordinate space, then plots corpus as light-grey
        background markers with vocab fragments coloured by cluster on top.

        Args:
            corpus_embeddings: Embeddings for corpus fragments, shape (n_corpus, dim).
            corpus_smiles: SMILES for corpus hover text.
            random_state: Random seed for UMAP.
            scatter_2d_args: Layout kwargs forwarded to plotly (height, width, …).
            corpus_marker_args: Marker style overrides for corpus points
                (defaults: color='lightgrey', size=4, opacity=0.4).

        Returns:
            plotly.graph_objs.Figure
        """
        import umap as umap_lib

        combined = np.vstack([self.embeddings, corpus_embeddings])
        reducer = umap_lib.UMAP(n_components=2, random_state=random_state)
        coords = reducer.fit_transform(combined)

        n_vocab = len(self.embeddings)
        vocab_2d = coords[:n_vocab]
        corpus_2d = coords[n_vocab:]

        fig = self._make_corpus_overlay_figure(
            vocab_2d=vocab_2d,
            corpus_2d=corpus_2d,
            axis_prefix='UMAP',
            title='2D UMAP — Vocab vs Corpus Fragments',
            corpus_smiles=corpus_smiles,
            scatter_2d_args=scatter_2d_args,
            corpus_marker_args=corpus_marker_args,
        )

        self.fig = fig
        return fig

    def plot_2D_PCA_with_corpus(
        self,
        corpus_embeddings: np.ndarray,
        corpus_smiles: list[str] | None = None,
        scatter_2d_args: dict | None = None,
        corpus_marker_args: dict | None = None,
    ):
        """
        2D PCA with corpus fragments as grey background and vocab coloured by cluster.

        Runs PCA on the **combined** vocab + corpus embeddings so both sets
        share the same coordinate space, then plots corpus as light-grey
        background markers with vocab fragments coloured by cluster on top.
        Axis labels include the percentage of variance explained.

        Args:
            corpus_embeddings: Embeddings for corpus fragments, shape (n_corpus, dim).
            corpus_smiles: SMILES for corpus hover text.
            scatter_2d_args: Layout kwargs forwarded to plotly (height, width, …).
            corpus_marker_args: Marker style overrides for corpus points
                (defaults: color='lightgrey', size=4, opacity=0.4).

        Returns:
            plotly.graph_objs.Figure
        """
        from sklearn.decomposition import PCA

        combined = np.vstack([self.embeddings, corpus_embeddings])
        pca = PCA(n_components=2)
        coords = pca.fit_transform(combined)
        var_explained = pca.explained_variance_ratio_ * 100

        n_vocab = len(self.embeddings)
        vocab_2d = coords[:n_vocab]
        corpus_2d = coords[n_vocab:]

        fig = self._make_corpus_overlay_figure(
            vocab_2d=vocab_2d,
            corpus_2d=corpus_2d,
            axis_prefix=f'PC',
            title='2D PCA — Vocab vs Corpus Fragments',
            corpus_smiles=corpus_smiles,
            scatter_2d_args=scatter_2d_args,
            corpus_marker_args=corpus_marker_args,
        )

        fig.update_xaxes(title_text=f'PC-1 ({var_explained[0]:.1f}%)')
        fig.update_yaxes(title_text=f'PC-2 ({var_explained[1]:.1f}%)')

        self.fig = fig
        return fig

    def plot_3D_UMAP(self, random_state: int = 42, scatter_3d_args: dict = {'height': 800, 'width': 1600}):
        """
        Generate 3D UMAP visualization of fragment embeddings colored by cluster.

        Performs Uniform Manifold Approximation and Projection (UMAP) dimensionality
        reduction on fragment embeddings to 3D space and creates an interactive
        Plotly 3D scatter plot. The 3D visualization can reveal cluster structure
        not visible in 2D projections. UMAP typically preserves more global structure
        than t-SNE and is faster for large datasets. Fragments are colored by cluster
        assignments.

        Args:
            random_state: Random seed for UMAP reproducibility. Same seed produces
                identical visualizations across runs. Default is 42.
            scatter_3d_args: Dictionary of keyword arguments passed to plotly.express.scatter_3d().
                Common options include:
                    - 'height': Plot height in pixels (default: 800)
                    - 'width': Plot width in pixels (default: 1600)
                    - 'template': Plotly template ('plotly', 'plotly_white', etc.)

        Returns:
            Plotly Figure object containing the 3D scatter plot. Interactive plot
            allows rotation, zoom, and hover to explore chemical space. Can be
            displayed with fig.show() or saved with fig.write_html().

        Raises:
            ImportError: If plotly or umap-learn is not installed.
            AttributeError: If self.embeddings or self.cluster_labels not set.

        Example:
            Basic 3D UMAP visualization:

            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> fig = clustering.plot_3D_UMAP()
            >>> fig.show()  # Opens interactive 3D plot in browser

            Custom visualization with different random seed:

            >>> fig = clustering.plot_3D_UMAP(
            ...     random_state=456,
            ...     scatter_3d_args={'height': 900, 'width': 1400, 'template': 'plotly_dark'}
            ... )
            >>> fig.write_html('umap_3d_plot.html')

            Explore clusters interactively:

            >>> fig = clustering.plot_3D_UMAP()
            >>> # Rotate plot to view from different angles
            >>> # Hover over points to see SMILES
            >>> fig.show()

        Note:
            - Sets self.plot_df (DataFrame with UMAP-1, UMAP-2, UMAP-3, SMILES columns)
            - Sets self.fig (Plotly 3D figure object)
            - Requires plotly: pip install plotly
            - Requires umap-learn: pip install umap-learn
            - 3D UMAP provides additional dimensionality compared to 2D, which can
              better preserve local and global structure of high-dimensional data.
            - UMAP is generally faster than t-SNE and better preserves global structure.
            - For large datasets (>10000 fragments), UMAP is recommended over t-SNE.
            - Interactive 3D plots allow rotation and zoom for exploring cluster
              boundaries and overlap.
        """
        try:
            import umap
            import plotly.express as px
        except ImportError as e:
            missing_lib = str(e).split("'")[1] if "'" in str(e) else "required library"
            print(f"The '{missing_lib}' library is not installed. Please install it using 'pip install {missing_lib}'.")

        umap_model = umap.UMAP(n_components=3, random_state=random_state)
        umap_results = umap_model.fit_transform(self.embeddings)

        cluster_labels_str = ['Cluster ' + s for s in self.cluster_labels.astype(str)]

        df = pd.DataFrame({
            'UMAP-1': umap_results[:, 0],
            'UMAP-2': umap_results[:, 1],
            'UMAP-3': umap_results[:, 2],
            'SMILES': self.smiles_df['SMILES'],
            'Cluster': cluster_labels_str,
        })

        fig = px.scatter_3d(
            df,
            x="UMAP-1",
            y="UMAP-2",
            z="UMAP-3",
            hover_name="SMILES",
            color="Cluster",
            title="3D UMAP of Molecular Embeddings",
            **scatter_3d_args
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
        )

        self.plot_df = df
        self.fig = fig

        return fig

    def plot_3D_PCA(self, scatter_3d_args: dict = {'height': 800, 'width': 1600}):
        """
        Generate 3D PCA visualization of fragment embeddings colored by cluster.

        Performs Principal Component Analysis (PCA) dimensionality reduction on
        fragment embeddings to 3D space and creates an interactive Plotly 3D
        scatter plot. PCA is a linear method that preserves global variance
        structure and is deterministic. Axis labels include the percentage of
        variance explained by each component.

        Args:
            scatter_3d_args: Dictionary of keyword arguments passed to plotly.express.scatter_3d().
                Common options include:
                    - 'height': Plot height in pixels (default: 800)
                    - 'width': Plot width in pixels (default: 1600)

        Returns:
            Plotly Figure object containing the 3D scatter plot.

        Example:
            >>> clustering = GSGE_clustering(gsge=gsge_instance)
            >>> clustering._embed_fragments()
            >>> clustering._cluster()
            >>> fig = clustering.plot_3D_PCA()
            >>> fig.show()
        """
        from sklearn.decomposition import PCA
        try:
            import plotly.express as px
        except ImportError:
            print("The 'plotly' library is not installed. Please install it using 'pip install plotly'.")

        pca = PCA(n_components=3)
        pca_results = pca.fit_transform(self.embeddings)
        var_explained = pca.explained_variance_ratio_ * 100

        cluster_labels_str = ['Cluster ' + s for s in self.cluster_labels.astype(str)]

        df = pd.DataFrame({
            'PC-1': pca_results[:, 0],
            'PC-2': pca_results[:, 1],
            'PC-3': pca_results[:, 2],
            'SMILES': self.smiles_df['SMILES'],
            'Cluster': cluster_labels_str,
        })

        fig = px.scatter_3d(
            df,
            x="PC-1",
            y="PC-2",
            z="PC-3",
            hover_name="SMILES",
            color="Cluster",
            title="3D PCA of Molecular Embeddings",
            labels={
                'PC-1': f'PC-1 ({var_explained[0]:.1f}%)',
                'PC-2': f'PC-2 ({var_explained[1]:.1f}%)',
                'PC-3': f'PC-3 ({var_explained[2]:.1f}%)',
            },
            **scatter_3d_args
        )

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='black',
        )

        self.plot_df = df
        self.fig = fig

        return fig