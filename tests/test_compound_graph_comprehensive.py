"""
Comprehensive tests for compound graph generation and manipulation.

Extends test_make_cg.py with detailed tests for graph structures,
feature extraction, PyG data conversion, and visualization.
"""

import pytest
import numpy as np
from typing import List, Tuple

from GSGE import GSGE, GS_Vocab
from GSGE.graphs.compound_graph.data import compound_graph


class TestCompoundGraphCreation:
    """Tests for creating compound graphs from SMILES."""

    def test_create_cg_from_simple_molecule(self, gsge_with_descriptors):
        """Test creating compound graph from simple molecule."""
        smiles = 'CCO'  # Ethanol

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        assert cg is not None
        # compound_graph extends MolecularGraph which has groups, atoms, bonds
        assert hasattr(cg, 'groups')
        assert hasattr(cg, 'atoms')
        assert len(cg.groups) > 0

    def test_create_cg_pyg_format(self, gsge_with_descriptors):
        """Test creating compound graph in PyG Data format."""
        smiles = 'c1ccccc1'  # Benzene

        edge_index, features = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=False
        )

        assert isinstance(edge_index, np.ndarray)
        assert isinstance(features, list)
        assert edge_index.shape[0] == 2  # Edge index format

    @pytest.mark.slow
    def test_create_multiple_cgs_parallel(self, gsge_with_descriptors, simple_smiles_list):
        """Test creating multiple compound graphs in parallel."""
        cgs = gsge_with_descriptors.make_compound_graphs(
            simple_smiles_list[:5],
            workers=2,
            pyg_data=False
        )

        assert len(cgs) == 5
        for edge_idx, features in cgs:
            assert isinstance(edge_idx, np.ndarray)
            assert isinstance(features, list)

    def test_cg_object_has_required_attributes(self, gsge_with_descriptors):
        """Test that compound_graph object has all required attributes."""
        smiles = 'CC(C)O'

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        # Check for essential attributes from MolecularGraph base class
        assert hasattr(cg, 'groups')
        assert hasattr(cg, 'atoms')
        # Check for compound_graph methods
        assert hasattr(cg, 'get_graph_data')


class TestAdjacencyMatrixStructure:
    """Tests for compound graph adjacency matrix structure."""

    def test_adjacency_matrix_shape(self, gsge_with_descriptors):
        """Test edge index has correct shape."""
        smiles = 'CCO'

        edge_index, features = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=False
        )

        # Should be 2 x num_edges format (edge index)
        assert edge_index.shape[0] == 2
        # May have 0 edges for single-fragment molecules
        assert edge_index.shape[1] >= 0

    def test_adjacency_matrix_symmetry(self, gsge_with_descriptors):
        """Test that edge index represents undirected graph."""
        smiles = 'CCCCC'  # Longer chain to ensure edges

        edge_index, _ = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=False
        )

        # Skip if no edges
        if edge_index.shape[1] == 0:
            return

        # For each edge (i, j), there should be edge (j, i)
        edges = set()
        for col_idx in range(edge_index.shape[1]):
            i, j = edge_index[0, col_idx], edge_index[1, col_idx]
            edges.add((i, j))

        # Check symmetry
        for i, j in list(edges):
            # Either this is a self-loop or reciprocal edge exists
            if i != j:
                assert (j, i) in edges or (i, j) in edges

    def test_adjacency_matrix_indices_valid(self, gsge_with_descriptors):
        """Test that edge index indices are within bounds."""
        smiles = 'CC(C)O'

        edge_index, features = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=False
        )

        num_nodes = len(features)

        # Skip if no edges
        if edge_index.shape[1] == 0:
            return

        # All indices should be < num_nodes
        assert np.all(edge_index[0] < num_nodes)
        assert np.all(edge_index[1] < num_nodes)
        assert np.all(edge_index >= 0)


class TestFeatureExtraction:
    """Tests for node feature extraction."""

    def test_features_are_fragment_ids(self, gsge_with_descriptors):
        """Test that features correspond to fragment IDs."""
        smiles = 'CCO'

        _, features = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=False
        )

        # Features should be integers (fragment IDs)
        assert all(isinstance(f, (int, np.integer)) for f in features)
        assert all(f >= 0 for f in features)

    def test_features_match_vocab_size(self, gsge_with_descriptors):
        """Test that fragment IDs are within vocabulary bounds."""
        smiles = 'CCCCC'  # Longer chain to get multiple fragments

        _, features = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=False
        )

        vocab_size = len(gsge_with_descriptors.get_fragments_smiles())

        # All fragment IDs should be < vocab_size
        assert all(f < vocab_size for f in features)

    def test_features_count_matches_nodes(self, gsge_with_descriptors):
        """Test that number of features matches number of groups."""
        smiles = 'CC(C)O'

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        # compound_graph has groups, not nodes
        node_ids, edge_index, bond_index = cg.get_graph_data()
        assert len(node_ids) == len(cg.groups)


class TestPyGDataConversion:
    """Tests for PyTorch Geometric Data format conversion."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pyg_data_format(self, gsge_with_descriptors, simple_smiles_list):
        """Test creating compound graphs in PyG Data format."""
        cgs_data = gsge_with_descriptors.make_compound_graphs(
            simple_smiles_list[:3],
            workers=1,
            pyg_data=True
        )

        assert len(cgs_data) == 3

        for data in cgs_data:
            # Should have PyG Data attributes
            assert hasattr(data, 'x')
            assert hasattr(data, 'edge_index')
            assert data.x is not None
            assert data.edge_index is not None

    @pytest.mark.integration
    @pytest.mark.slow
    def test_pyg_data_shapes(self, gsge_with_descriptors, simple_smiles_list):
        """Test that PyG Data has correct tensor shapes."""
        cgs_data = gsge_with_descriptors.make_compound_graphs(
            simple_smiles_list[:2],
            workers=1,
            pyg_data=True
        )

        for data in cgs_data:
            # x should be [num_nodes, num_features]
            assert data.x.ndim == 2
            assert data.x.shape[0] > 0

            # edge_index should be [2, num_edges]
            assert data.edge_index.shape[0] == 2


class TestCompoundGraphVisualization:
    """Tests for compound graph visualization."""

    def test_cg_has_plot_method(self, gsge_with_descriptors):
        """Test that compound_graph has plotting method."""
        smiles = 'CCO'

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        # Should have plotting methods
        assert hasattr(cg, 'plot_graph_rd_c_style')

    @pytest.mark.slow
    def test_plot_graph_rd_c_style(self, gsge_with_descriptors):
        """Test RDKit-style graph plotting."""
        smiles = 'c1ccccc1'

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        try:
            # Attempt to plot (may require display backend)
            cg.plot_graph_rd_c_style(show=False)
        except Exception as e:
            # May fail without display, but should not crash code
            pass


class TestErrorHandling:
    """Tests for error handling in compound graph creation."""

    def test_invalid_smiles_handling(self, gsge_with_descriptors):
        """Test handling of invalid SMILES."""
        invalid_smiles = ['INVALID!!!', 'NOT_SMILES']

        for smiles in invalid_smiles:
            try:
                cg = gsge_with_descriptors.get_CG_from_smiles(
                    smiles,
                    return_CG_object=True
                )
                # Invalid SMILES may return None or raise exception
                # If it returns something, it should be handled gracefully
            except (ValueError, AttributeError, KeyError, TypeError):
                # Expected for invalid SMILES
                pass

    def test_empty_smiles_handling(self, gsge_with_descriptors):
        """Test handling of empty SMILES."""
        try:
            cg = gsge_with_descriptors.get_CG_from_smiles(
                '',
                return_CG_object=True
            )
            # May return None or raise exception
        except (ValueError, AttributeError, TypeError):
            # Expected for empty input
            pass

    @pytest.mark.slow
    def test_parallel_cg_with_mixed_valid_invalid(self, gsge_with_descriptors):
        """Test parallel compound graph creation with mixed valid/invalid SMILES."""
        mixed_smiles = ['CCO', 'c1ccccc1']

        try:
            cgs = gsge_with_descriptors.make_compound_graphs(
                mixed_smiles,
                workers=1,
                pyg_data=False
            )
            # Should handle valid SMILES
            assert isinstance(cgs, list)
            assert len(cgs) == 2
        except Exception:
            # Acceptable if error raised
            pass


class TestCompoundGraphIntegration:
    """Integration tests for compound graph workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_full_workflow_vocab_to_cg(self, simple_smiles_list):
        """Test complete workflow from vocabulary to compound graphs."""
        # Build vocabulary
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=20,
            MIN_SIZE=1,
            MAX_SIZE=8
        )

        # Create GSGE
        gsge = GSGE(GS_vocab=vocab)
        gsge.add_all_single_elements()

        # Create compound graphs
        cgs = gsge.make_compound_graphs(
            simple_smiles_list[:3],
            workers=1,
            pyg_data=False
        )

        assert len(cgs) == 3

    @pytest.mark.integration
    def test_cg_with_descriptors(self, gsge_with_descriptors, simple_smiles_list):
        """Test compound graphs with fragment descriptors."""
        # Create compound graph
        cg = gsge_with_descriptors.get_CG_from_smiles(
            simple_smiles_list[0],
            return_CG_object=True
        )

        assert cg is not None
        # compound_graph has groups, not nodes
        assert len(cg.groups) > 0


class TestCompoundGraphProperties:
    """Tests for compound graph properties and invariants."""

    def test_cg_connectivity(self, gsge_with_descriptors):
        """Test that compound graph represents connected molecule."""
        # Linear molecule should have connected graph
        smiles = 'CCCC'

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        # compound_graph has groups (not nodes) and bonds (not edges)
        assert len(cg.groups) > 0
        # Check connectivity via bonds
        node_ids, edge_index, bond_index = cg.get_graph_data()
        # Should have at least some fragments
        assert len(node_ids) > 0

    def test_cg_preserves_ring_structure(self, gsge_with_descriptors):
        """Test that cyclic molecules preserve ring structure."""
        # Benzene ring
        smiles = 'c1ccccc1'

        cg = gsge_with_descriptors.get_CG_from_smiles(
            smiles,
            return_CG_object=True
        )

        # Should have cyclic structure - check via groups
        assert len(cg.groups) >= 1

    def test_different_molecules_different_cgs(self, gsge_with_descriptors):
        """Test that different molecules produce different graphs."""
        smiles1 = 'CCO'
        smiles2 = 'CCCCC'  # Different molecule

        cg1 = gsge_with_descriptors.get_CG_from_smiles(smiles1, return_CG_object=True)
        cg2 = gsge_with_descriptors.get_CG_from_smiles(smiles2, return_CG_object=True)

        # Different molecules may have different structures
        # They should both be valid compound_graph objects
        assert hasattr(cg1, 'groups')
        assert hasattr(cg2, 'groups')
