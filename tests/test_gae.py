"""
Tests for Graph Autoencoder (GAE) training and inference.

Tests encoder/decoder architecture, training loop, loss computation,
checkpointing, and metrics tracking.
"""

import pytest
import torch
import tempfile
from pathlib import Path

from GSGE.graphs.fragment_graph.GAE import (
    GraphDecoder,
    GraphAutoencoderTrainer,
    MetricsTracker,
    compute_atom_loss,
    compute_edge_loss
)
from GSGE import GSGE_Corpus, GS_Vocab
from torch_geometric.nn.models.attentive_fp import AttentiveFP


class TestGraphDecoder:
    """Tests for GraphDecoder architecture."""

    def test_decoder_init(self):
        """Test GraphDecoder initialization."""
        decoder = GraphDecoder(latent_dim=32, hidden_dim=64)

        assert decoder.latent_dim == 32
        assert decoder.hidden_dim == 64
        assert decoder.num_node_types == 22
        assert decoder.max_num_nodes == 20

    def test_decoder_forward_shape(self):
        """Test GraphDecoder forward pass output shapes."""
        decoder = GraphDecoder(latent_dim=32, hidden_dim=64)

        # Create dummy latent embeddings
        batch_size = 8
        z = torch.randn(batch_size, 32)
        batch = torch.arange(batch_size).repeat_interleave(10)  # 10 nodes per graph

        # Forward pass
        atom_feats, num_atoms, edge_feats, num_edges = decoder(z, batch, verbose=False)

        # Check output shapes
        assert atom_feats.shape == (batch_size, 20, 22)  # [batch, max_nodes, num_types]
        assert num_atoms.shape == (batch_size,)
        assert edge_feats.shape[0] == batch_size
        assert num_edges.shape == (batch_size,)

    def test_decoder_shared_layers(self):
        """Test shared decoder layers."""
        decoder = GraphDecoder(latent_dim=32, hidden_dim=64)

        z = torch.randn(4, 32)
        z_hidden = decoder.shared_decoder(z, verbose=False)

        assert z_hidden.shape == (4, 64)  # Should match hidden_dim

    def test_decoder_node_edge_decoder(self):
        """Test node and edge decoder heads."""
        decoder = GraphDecoder(latent_dim=32, hidden_dim=64)

        batch_size = 4
        z = torch.randn(batch_size, 64)  # Already processed by shared layers
        batch = torch.arange(batch_size).repeat_interleave(5)

        atom_feats, num_atoms, edge_feats, num_edges = decoder.node_edge_decoder(
            z, batch, verbose=False
        )

        assert atom_feats.shape[0] == batch_size
        assert num_atoms.shape[0] == batch_size


class TestMetricsTracker:
    """Tests for MetricsTracker."""

    def test_metrics_tracker_init(self):
        """Test MetricsTracker initialization."""
        tracker = MetricsTracker()

        assert len(tracker.atom_preds) == 0
        assert len(tracker.edge_type_preds) == 0

    def test_add_batch(self):
        """Test adding batch predictions to tracker."""
        tracker = MetricsTracker()

        # Create dummy predictions
        atom_preds = torch.randint(0, 22, (10,)).numpy()
        atom_targets = torch.randint(0, 22, (10,)).numpy()
        node_count_preds = torch.randn(2).numpy()
        node_count_trues = torch.randint(5, 15, (2,)).numpy()

        adj_preds = torch.randint(0, 2, (20,)).numpy()
        adj_targets = torch.randint(0, 2, (20,)).numpy()
        type_preds = torch.randint(0, 5, (10,)).numpy()
        type_targets = torch.randint(0, 5, (10,)).numpy()
        edge_count_preds = torch.randn(2).numpy()
        edge_count_trues = torch.randint(5, 20, (2,)).numpy()

        atom_loss_outputs = (atom_preds, atom_targets, node_count_preds, node_count_trues)
        edge_loss_outputs = (adj_preds, adj_targets, type_preds, type_targets,
                            edge_count_preds, edge_count_trues)

        tracker.add_batch(atom_loss_outputs, edge_loss_outputs)

        assert len(tracker.atom_preds) == 1
        assert len(tracker.adj_preds) == 1

    def test_compute_metrics(self):
        """Test computing metrics from accumulated predictions."""
        tracker = MetricsTracker()

        # Add some dummy batches
        for _ in range(3):
            atom_preds = torch.randint(0, 22, (20,)).numpy()
            atom_targets = torch.randint(0, 22, (20,)).numpy()
            node_count_preds = torch.randn(4).numpy()
            node_count_trues = torch.randint(5, 15, (4,)).numpy()

            adj_preds = torch.randint(0, 2, (40,)).numpy()
            adj_targets = torch.randint(0, 2, (40,)).numpy()
            type_preds = torch.randint(0, 5, (20,)).numpy()
            type_targets = torch.randint(0, 5, (20,)).numpy()
            edge_count_preds = torch.randn(4).numpy()
            edge_count_trues = torch.randint(5, 20, (4,)).numpy()

            tracker.add_batch(
                (atom_preds, atom_targets, node_count_preds, node_count_trues),
                (adj_preds, adj_targets, type_preds, type_targets,
                 edge_count_preds, edge_count_trues)
            )

        metrics = tracker.compute_metrics()

        # Check all expected metrics are present
        assert 'atom_accuracy' in metrics
        assert 'atom_f1' in metrics
        assert 'adj_accuracy' in metrics
        assert 'edge_type_accuracy' in metrics
        assert 'atom_num_r2' in metrics
        assert 'edge_num_r2' in metrics

    def test_reset_tracker(self):
        """Test resetting tracker clears all data."""
        tracker = MetricsTracker()

        # Add some data
        tracker.atom_preds.append(torch.randn(10).numpy())
        tracker.atom_targets.append(torch.randn(10).numpy())

        # Reset
        tracker.reset()

        assert len(tracker.atom_preds) == 0
        assert len(tracker.atom_targets) == 0


class TestGAETraining:
    """Tests for GAE training workflow."""

    @pytest.mark.slow
    @pytest.mark.gae
    def test_train_one_epoch(self, mock_encoder_cpu, mock_decoder_cpu, minimal_vocab, temp_checkpoint_dir):
        """Test training GAE for one epoch."""
        # Create minimal corpus
        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=['CCO', 'CC(C)O', 'c1ccccc1'],
            convert=True,
            min_size=1,
            max_size=8
        )

        # Create trainer
        from GSGE.core_gsge import CoreGSGE
        train_loader, val_loader = CoreGSGE.load_and_prepare_data(
            GS_vocab=minimal_vocab,
            GSGE_corpus=corpus,
            batch_size=2,
            val_percentage=0.2,
            split_seed=42
        )

        optimizer = torch.optim.Adam(
            list(mock_encoder_cpu.parameters()) + list(mock_decoder_cpu.parameters()),
            lr=0.001
        )

        trainer = GraphAutoencoderTrainer(
            encoder=mock_encoder_cpu,
            decoder=mock_decoder_cpu,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            checkpoint_dir=str(temp_checkpoint_dir),
            device='cpu',
            batch_size=2
        )

        # Train for 1 epoch only
        trainer.train(num_epochs=1, checkpoint_interval=1)

        # Check checkpoint was created
        checkpoints = list(temp_checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        assert len(checkpoints) >= 1

    def test_checkpoint_save_load(self, mock_encoder_cpu, mock_decoder_cpu, temp_checkpoint_dir):
        """Test saving and loading checkpoint."""
        # Save checkpoint manually
        checkpoint_path = temp_checkpoint_dir / 'test_checkpoint.pth'
        torch.save({
            'epoch': 10,
            'encoder_state_dict': mock_encoder_cpu.state_dict(),
            'decoder_state_dict': mock_decoder_cpu.state_dict(),
            'train_loss': 1.5,
            'val_loss': 1.8
        }, checkpoint_path)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        assert checkpoint['epoch'] == 10
        assert 'encoder_state_dict' in checkpoint
        assert 'decoder_state_dict' in checkpoint


class TestLossFunctions:
    """Tests for loss computation functions."""

    def test_compute_atom_loss_shapes(self):
        """Test compute_atom_loss returns correct shapes."""
        batch_size = 4
        max_nodes = 20
        num_classes = 22

        # Create dummy predictions and targets
        decoded_node_logits = torch.randn(batch_size, max_nodes, num_classes)
        decoded_num_nodes = torch.randn(batch_size)
        node_targets = torch.randint(0, num_classes, (50,))  # Total nodes across batch
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3] * 5)  # Assign nodes to graphs

        loss, atom_preds, atom_targets, node_count_preds, node_count_trues = compute_atom_loss(
            decoded_node_logits,
            decoded_num_nodes,
            node_targets,
            batch,
            verbose=False
        )

        assert loss.item() >= 0  # Loss should be non-negative
        assert len(atom_preds) > 0
        assert len(atom_targets) > 0
        assert len(node_count_preds) == batch_size
        assert len(node_count_trues) == batch_size

    def test_compute_edge_loss_shapes(self):
        """Test compute_edge_loss returns correct shapes."""
        batch_size = 4
        max_triu_size = 190
        num_edge_types = 5

        # Create dummy predictions
        decoded_edge_logits = torch.randn(batch_size, max_triu_size, 1 + num_edge_types)
        decoded_num_edges = torch.randn(batch_size)

        # Create dummy edge data
        edge_index = torch.randint(0, 40, (2, 30))
        edge_attr = torch.randint(0, num_edge_types, (30, 1))
        batch = torch.arange(batch_size).repeat_interleave(10)

        loss, adj_preds, adj_targets, type_preds, type_targets, edge_count_preds, edge_count_trues = compute_edge_loss(
            decoded_edge_logits,
            decoded_num_edges,
            edge_index,
            edge_attr,
            batch,
            verbose=False
        )

        assert loss.item() >= 0
        assert len(adj_preds) > 0
        assert len(edge_count_preds) == batch_size


class TestGAEIntegration:
    """Integration tests for full GAE workflow."""

    @pytest.mark.integration
    @pytest.mark.slow
    def test_gsge_train_gae_integration(self, simple_smiles_list, temp_checkpoint_dir):
        """Test end-to-end GAE training through GSGE interface."""
        from GSGE import GSGE

        # Create vocab and corpus
        vocab = GS_Vocab()
        vocab.build_vocab(
            m_set=simple_smiles_list,
            convert=True,
            target=20,
            MIN_SIZE=1,
            MAX_SIZE=8
        )

        corpus = GSGE_Corpus()
        corpus.build_corpus(
            m_set=simple_smiles_list,
            convert=True,
            min_size=1,
            max_size=8
        )

        # Create GSGE with models
        encoder = AttentiveFP(
            in_channels=9,
            hidden_channels=32,
            out_channels=16,
            edge_dim=3,
            num_layers=1,
            num_timesteps=1
        )
        decoder = GraphDecoder(latent_dim=16, hidden_dim=32)

        gsge = GSGE(
            GS_vocab=vocab,
            GSGE_corpus=corpus,
            encoder=encoder,
            decoder=decoder
        )

        # Train for 1 epoch
        gsge.train_GSGE_Auto_Encoder(
            batch_size=2,
            num_epochs=1,
            checkpoint_interval=1,
            device='cpu',
            checkpoint_dir=str(temp_checkpoint_dir)
        )

        # Verify checkpoint exists
        checkpoints = list(temp_checkpoint_dir.glob('checkpoint_epoch_*.pth'))
        assert len(checkpoints) >= 1
