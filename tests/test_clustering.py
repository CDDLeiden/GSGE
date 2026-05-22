"""
Tests for GSGE_clustering visualization and analysis.

Tests fragment embedding generation, TSNE/UMAP visualization,
MCS clustering, and clustering utilities.
"""

import pytest
import numpy as np
import pandas as pd

from GSGE.clustering import GSGE_clustering


class TestClusteringEmbedding:
    """Tests for fragment embedding in clustering context."""

    def test_embed_fragments_with_encoder(self, gsge_with_descriptors, mock_encoder_cpu):
        """Test embedding fragments using encoder."""
        # Set encoder
        gsge_with_descriptors.gae_trainer.set_encoder(mock_encoder_cpu)
        gsge_with_descriptors.store_modules.modules['encoder'] = mock_encoder_cpu

        # Get fragments
        frag_smiles = gsge_with_descriptors.get_fragments_smiles()[:10]  # First 10 for speed

        # Embed fragments
        embeddings, graph_data = gsge_with_descriptors.embed_fragments(
            frag_smiles=frag_smiles,
            device='cpu',
            batch_size=5,
            return_data=True
        )

        assert embeddings.shape[0] == 10  # 10 fragments
        assert embeddings.shape[1] == 32  # Encoder output dim
        assert len(graph_data) == 10

    @pytest.mark.slow
    def test_make_fragment_embedding_dict(self, gsge_with_descriptors, mock_encoder_cpu):
        """Test creating fragment embedding dictionary for vocabulary."""
        gsge_with_descriptors.gae_trainer.set_encoder(mock_encoder_cpu)
        gsge_with_descriptors.store_modules.modules['encoder'] = mock_encoder_cpu

        # Create embeddings (this is slow for full vocab)
        gsge_with_descriptors.make_GS_fragment_embedding_dict(
            device='cpu',
            batch_size=16
        )

        # Check that embeddings were created
        embeddings = gsge_with_descriptors.get_fragment_embeddings()
        assert embeddings is not None
        assert embeddings.shape[0] == len(gsge_with_descriptors.get_fragments_smiles())


class TestMCSClustering:
    """Tests for Maximum Common Substructure clustering."""

    @pytest.mark.slow
    def test_mcs_clustering_basic(self):
        """Test MCS clustering on simple molecules."""
        smiles_df = pd.DataFrame({
            'SMILES': ['CCO', 'CCCO', 'c1ccccc1', 'c1ccc(O)cc1']
        })

        # Run MCS clustering
        labels = GSGE_clustering._MCS_clustering(smiles_df)

        assert len(labels) == len(smiles_df)
        # Labels should be integers
        assert all(isinstance(l, (int, np.integer)) for l in labels)

    @pytest.mark.slow
    def test_mcs_clustering_with_custom_params(self):
        """Test MCS clustering with custom parameters."""
        smiles_df = pd.DataFrame({
            'SMILES': ['CCO', 'CC(C)O', 'CC(C)(C)O']
        })

        labels = GSGE_clustering._MCS_clustering(
            smiles_df,
            atomCompare_RingMatchesRingOnly=True
        )

        assert len(labels) == len(smiles_df)

    @pytest.mark.slow
    def test_mcs_clustering_different_column(self):
        """Test MCS clustering with non-default SMILES column."""
        smiles_df = pd.DataFrame({
            'molecule': ['CCO', 'CCCO', 'c1ccccc1']
        })

        labels = GSGE_clustering._MCS_clustering(
            smiles_df,
            smiles_column='molecule'
        )

        assert len(labels) == len(smiles_df)


class TestGSGEClusteringObject:
    """Tests for GSGE_clustering class instantiation and methods."""

    @pytest.mark.slow
    def test_clustering_init_basic(self, gsge_with_descriptors):
        """Test creating clustering object with minimal args."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:20]
        })

        # Get clustering with MCS
        clustering = gsge_with_descriptors.get_GSGE_clustering(
            smiles_df=smiles_df
        )

        assert clustering is not None
        assert hasattr(clustering, 'cluster_labels')
        assert hasattr(clustering, 'smiles_df')

    @pytest.mark.slow
    def test_clustering_with_embeddings(self, gsge_with_descriptors):
        """Test clustering with provided embeddings."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:10]
        })

        # Create dummy embeddings
        embeddings = np.random.rand(10, 128)

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            embeddings=embeddings,
            smiles_df=smiles_df
        )

        assert clustering.embeddings is not None
        assert clustering.embeddings.shape == (10, 128)

    def test_clustering_with_provided_labels(self, gsge_with_descriptors):
        """Test clustering with pre-computed labels."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:10]
        })

        # Provide manual cluster labels
        labels = [0, 0, 1, 1, 2, 2, 0, 1, 2, 0]

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            smiles_df=smiles_df,
            cluster_labels=labels
        )

        assert clustering.cluster_labels == labels


class TestClusteringVisualization:
    """Tests for clustering visualization methods."""

    @pytest.mark.slow
    def test_plot_2d_tsne(self, gsge_with_descriptors):
        """Test 2D t-SNE visualization."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:30]
        })

        # Create embeddings
        embeddings = np.random.rand(30, 128)

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            embeddings=embeddings,
            smiles_df=smiles_df
        )

        # Generate 2D t-SNE
        fig = clustering.plot_2D_TSNE(random_state=42)

        assert fig is not None
        # Check it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')

    @pytest.mark.slow
    def test_plot_2d_umap(self, gsge_with_descriptors):
        """Test 2D UMAP visualization."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:30]
        })

        # Create embeddings
        embeddings = np.random.rand(30, 128)

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            embeddings=embeddings,
            smiles_df=smiles_df
        )

        # Generate 2D UMAP
        fig = clustering.plot_2D_UMAP(random_state=42)

        assert fig is not None
        # Check it's a plotly figure
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')

    @pytest.mark.slow
    def test_plot_3d_tsne(self, gsge_with_descriptors):
        """Test 3D t-SNE visualization."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:30]
        })

        embeddings = np.random.rand(30, 128)

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            embeddings=embeddings,
            smiles_df=smiles_df
        )

        # Generate 3D t-SNE
        fig = clustering.plot_3D_TSNE(random_state=42)

        assert fig is not None
        assert hasattr(fig, 'data')

    @pytest.mark.slow
    def test_plot_3d_umap(self, gsge_with_descriptors):
        """Test 3D UMAP visualization."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:30]
        })

        embeddings = np.random.rand(30, 128)

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            embeddings=embeddings,
            smiles_df=smiles_df
        )

        # Generate 3D UMAP
        fig = clustering.plot_3D_UMAP(random_state=42)

        assert fig is not None
        assert hasattr(fig, 'data')

    @pytest.mark.slow
    def test_clustering_plot_df_created(self, gsge_with_descriptors):
        """Test that plot_df is created after visualization."""
        smiles_df = pd.DataFrame({
            'SMILES': gsge_with_descriptors.get_fragments_smiles()[:20]
        })

        embeddings = np.random.rand(20, 128)

        clustering = gsge_with_descriptors.get_GSGE_clustering(
            embeddings=embeddings,
            smiles_df=smiles_df
        )

        # Generate plot (creates plot_df internally)
        clustering.plot_2D_TSNE(random_state=42)

        # Check plot_df was created
        assert hasattr(clustering, 'plot_df')
        assert clustering.plot_df is not None
        assert 'x' in clustering.plot_df.columns
        assert 'y' in clustering.plot_df.columns
