import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.models.attentive_fp import AttentiveFP
import numpy as np
import os
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, r2_score

#### SETTINGS
ATOM_TYPES = 22 
OTHER_TOKENS_TYPES = 0 
ATOM_MAX_NUM = 20 
ATOM_OUT_PUT_DIM = int((ATOM_TYPES + OTHER_TOKENS_TYPES) * ATOM_MAX_NUM)  # 440
NUM_EDGE_TYPES = 5 
EDGE_OUTPUT_DIM = int(((ATOM_MAX_NUM * (ATOM_MAX_NUM - 1)) / 2) * (NUM_EDGE_TYPES + 1))  # 1140

class GraphDecoder(nn.Module):
    """
    Graph Decoder for Fragment Graph Autoencoder.

    Reconstructs molecular fragment graph structure from latent embeddings,
    predicting atom features, bond features, node counts, and edge counts.
    Uses separate decoder heads for atom and edge reconstruction.

    The decoder reconstructs:
    - Atom types (multi-class classification, 22 atom types)
    - Number of atoms (regression)
    - Edge types (multi-class classification, 5 bond types)
    - Number of edges (regression)

    Architecture:
    - Shared layers: 2 linear layers (latent_dim → hidden_dim → hidden_dim)
    - Atom decoder: 2 heads (atom features and atom count)
    - Edge decoder: 2 heads (edge features and edge count)
    """

    def __init__(self, latent_dim:int=32, hidden_dim:int=64, num_node_types:int=int(ATOM_TYPES + OTHER_TOKENS_TYPES),
                 max_num_nodes:int=ATOM_MAX_NUM, num_edge_types:int=NUM_EDGE_TYPES):
        """
        Initialize GraphDecoder.

        Args:
            latent_dim: Dimension of latent embedding from encoder. Default is 32.
            hidden_dim: Dimension of hidden layers in decoder. Default is 64.
            num_node_types: Number of atom types to predict (22 atom types).
                Default is ATOM_TYPES + OTHER_TOKENS_TYPES.
            max_num_nodes: Maximum number of nodes per fragment. Default is 20.
            num_edge_types: Number of edge/bond types (5 types). Default is NUM_EDGE_TYPES.

        Note:
            - node_output_dim = num_node_types * max_num_nodes (440)
            - edge_output_dim = (max_num_nodes * (max_num_nodes - 1) / 2) * (num_edge_types + 1) (1140)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_node_types = num_node_types  # 22
        self.max_num_nodes = max_num_nodes    # 20
        self.num_edge_types = num_edge_types  # 5
        
        self.node_output_dim = int(num_node_types * max_num_nodes)  # 440
        self.edge_output_dim = int(((max_num_nodes * (max_num_nodes - 1)) / 2) * (num_edge_types + 1))  # 1140

        #### SHARED DECODER LAYERS
        self.shared_linear1 = Linear(self.latent_dim, self.hidden_dim)
        self.shared_linear2 = Linear(self.hidden_dim, hidden_dim)

        #### ATOM DECODERS
        self.atom_feature_decoder = Linear(self.hidden_dim, self.node_output_dim)  # [hidden_dim] -> [440]
        self.num_atom_decoder = Linear(self.hidden_dim, 1)                        # [hidden_dim] -> [1]

        #### EDGE DECODERS
        self.edge_attr_decoder = Linear(self.hidden_dim, self.edge_output_dim)    # [hidden_dim] -> [1140]
        self.num_edges_decoder = Linear(self.hidden_dim, 1)                       # [hidden_dim] -> [1]

    def forward(self, z: Tensor, batch: Tensor, verbose:bool=True):
        """
        Forward pass through decoder to reconstruct graph.

        Args:
            z: Latent embedding tensor of shape [batch_size, latent_dim].
            batch: Batch assignment tensor mapping nodes to graph indices.
                Shape: [total_nodes].
            verbose: If True, print tensor shapes during forward pass. Default is True.

        Returns:
            Tuple of (pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges):
                - pred_atom_features: Atom type logits [batch_size, max_num_nodes, num_node_types]
                - pred_num_atom: Predicted atom counts [batch_size]
                - pred_edge_attr: Edge type logits [batch_size, num_edge_pairs, num_edge_types + 1]
                - pred_num_edges: Predicted edge counts [batch_size]

        Example:
            >>> decoder = GraphDecoder(latent_dim=32, hidden_dim=64)
            >>> z = torch.randn(64, 32)  # batch_size=64, latent_dim=32
            >>> batch = torch.arange(64).repeat_interleave(10)  # 10 nodes per graph
            >>> atom_feats, num_atoms, edge_feats, num_edges = decoder(z, batch, verbose=False)
        """
        if verbose:
            print('\n__________DECODER_INPUT__________')
            print('z.shape     :', z.shape)    # e.g., [64, 32]
            print('batch.shape :', batch.shape)  # e.g., [658]
            print('____________________________\n')

        z = self.shared_decoder(z, verbose=verbose)  # [batch_size, hidden_dim]

        pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges = self.node_edge_decoder(z, batch, verbose=verbose)

        if verbose:
            print('\n__________DECODER_OUTPUT__________')
            print(f'Decoded Node Logits Shape : {pred_atom_features.shape}')  # [64, 20, 22]
            print(f'Decoded Num Node Shape : {pred_num_atom.shape}')          # [64]
            print(f'Decoded Edge Logits Shape : {pred_edge_attr.shape}')      # [64, 190, 6]
            print(f'Decoded Num Edge Shape : {pred_num_edges.shape}')         # [64]
            print('____________________________\n')

        return pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges
    
    def shared_decoder(self, z, verbose=True):
        """
        Apply shared decoder layers to latent embedding.

        Args:
            z: Latent embedding tensor [batch_size, latent_dim].
            verbose: If True, print output shape. Default is True.

        Returns:
            Hidden representation tensor [batch_size, hidden_dim].
        """
        z = self.shared_linear1(z)  # [batch_size, latent_dim] -> [batch_size, hidden_dim]
        z = self.shared_linear2(z)  # [batch_size, hidden_dim] -> [batch_size, hidden_dim]

        if verbose:
            print('\n__________DECODER_SHARED_LAYERS_OUTPUT___________')
            print('z.shape     :', z.shape)  # e.g., [64, 64]
            print('____________________________\n')

        return z
    
    def node_edge_decoder(self, z, batch, verbose=False):
        """
        Decode node and edge features from hidden representation.

        Args:
            z: Hidden representation tensor [batch_size, hidden_dim].
            batch: Batch assignment tensor [total_nodes].
            verbose: If True, print debugging information. Default is False.

        Returns:
            Tuple of (pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges):
                - pred_atom_features: [batch_size, max_num_nodes, num_node_types]
                - pred_num_atom: [batch_size]
                - pred_edge_attr: [batch_size, num_edge_pairs, num_edge_types + 1]
                - pred_num_edges: [batch_size]

        Note:
            All decoding happens in parallel across the batch for efficiency.
        """
        batch_size = len(batch.unique())
        assert z.shape[0] == batch_size, f"z should have shape [batch_size, hidden_dim], got {z.shape}"
        assert z.shape[1] == self.hidden_dim, f"z hidden dim {z.shape[1]} != {self.hidden_dim}"

        if verbose:
            print(f'Processing batch - z.shape: {z.shape}')

        # Atom decoding (parallel for all graphs)
        pred_atom_features = self.atom_feature_decoder(z).view(batch_size, self.max_num_nodes, self.num_node_types)  # [batch_size, 440] -> [batch_size, 20, 22]
        pred_num_atom = self.num_atom_decoder(z).squeeze(-1)  # [batch_size, 1] -> [batch_size]

        # Edge decoding (parallel for all graphs)
        pred_edge_attr = self.edge_attr_decoder(z).view(batch_size, int(self.edge_output_dim / (self.num_edge_types + 1)), 
                                                       self.num_edge_types + 1)  # [batch_size, 1140] -> [batch_size, 190, 6]
        pred_num_edges = self.num_edges_decoder(z).squeeze(-1)  # [batch_size, 1] -> [batch_size]

        return pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges

class GraphAutoencoderTrainer:
    """
    Graph Autoencoder Training Manager.

    Manages end-to-end training of Graph Autoencoder for molecular fragment
    embeddings, including checkpointing, validation, and metrics tracking.

    Features:
    - Automatic checkpointing at configurable intervals
    - Validation after each epoch
    - Comprehensive metrics tracking (atom/edge accuracy, F1, R²)
    - Resume training from checkpoint
    - GPU/CPU compatibility
    """

    def __init__(self, encoder, decoder, optimizer, train_loader, val_loader, checkpoint_dir, device='cuda',
                 batch_size=64, load_checkpoint_path=None):
        """
        Initialize GraphAutoencoderTrainer.

        Args:
            encoder: Encoder model (e.g., AttentiveFP). Encodes fragment graphs
                to latent embeddings.
            decoder: Decoder model (e.g., GraphDecoder). Reconstructs graphs
                from latent embeddings.
            optimizer: PyTorch optimizer for training (e.g., Adam).
            train_loader: PyTorch Geometric DataLoader for training data.
            val_loader: PyTorch Geometric DataLoader for validation data.
            checkpoint_dir: Directory path for saving model checkpoints.
            device: Device for training ('cuda' or 'cpu'). Default is 'cuda'.
            batch_size: Batch size for training. Default is 64.
            load_checkpoint_path: Path to checkpoint file for resuming training.
                If None, starts from scratch. Default is None.

        Example:
            >>> from torch_geometric.nn.models.attentive_fp import AttentiveFP
            >>> encoder = AttentiveFP(in_channels=9, hidden_channels=128,
            ...                       out_channels=32, edge_dim=3, num_layers=3)
            >>> decoder = GraphDecoder(latent_dim=32, hidden_dim=128)
            >>> optimizer = torch.optim.Adam(
            ...     list(encoder.parameters()) + list(decoder.parameters()),
            ...     lr=0.001
            ... )
            >>> trainer = GraphAutoencoderTrainer(
            ...     encoder, decoder, optimizer,
            ...     train_loader, val_loader,
            ...     checkpoint_dir='./checkpoints'
            ... )
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = checkpoint_dir
        self.batch_size = batch_size
        self.tracker = MetricsTracker()

        # Ensure checkpoint directory exists
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Load checkpoint if provided
        if load_checkpoint_path and os.path.exists(load_checkpoint_path):
            checkpoint = torch.load(load_checkpoint_path, map_location=self.device)
            self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.last_epoch = checkpoint['epoch']
            print(f"Resuming training from epoch {self.last_epoch + 1}")
        else:
            self.last_epoch = -1  # Start from 0 if no checkpoint
            print("Starting training from scratch")

    def train(self, num_epochs, checkpoint_interval=5):
        """
        Run training loop with validation and checkpointing.

        Trains encoder-decoder on training data, validates on validation data
        after each epoch, and saves checkpoints at specified intervals.

        Args:
            num_epochs: Total number of epochs to train.
            checkpoint_interval: Save checkpoint every N epochs. Default is 5.

        Example:
            >>> trainer.train(num_epochs=300, checkpoint_interval=10)
            Epoch [1/300] ------------------------------------- Train
              Train Atom Loss: 2.5432, Train Edge Loss: 3.1245, ...
              ...
            Checkpoint saved at ./checkpoints/checkpoint_epoch_10.pth

        Note:
            - Prints comprehensive metrics after each epoch
            - Automatically saves checkpoints containing:
                - Epoch number
                - Encoder state dict
                - Decoder state dict
                - Optimizer state dict
                - Training and validation losses
        """
        for epoch in range(self.last_epoch + 1, num_epochs + 1):
            # Training phase
            self.encoder.train()
            self.decoder.train()
            total_atom_loss = 0.0
            total_edge_loss = 0.0

            self.tracker.reset()

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                batch = batch.to(self.device)

                # Forward pass
                z = self.encoder(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges = self.decoder(z, batch.batch, verbose=False)

                # Compute losses
                atom_loss_outputs = compute_atom_loss(pred_atom_features, pred_num_atom, batch.x[:, 0], batch.batch, verbose=False)
                edge_loss_outputs = compute_edge_loss(pred_edge_attr, pred_num_edges, batch.edge_index, batch.edge_attr, batch.batch, verbose=False)
                total_loss = atom_loss_outputs[0] + edge_loss_outputs[0]

                # Backward pass and optimization
                total_loss.backward()
                self.optimizer.step()

                total_atom_loss += atom_loss_outputs[0].item()
                total_edge_loss += edge_loss_outputs[0].item()

                self.tracker.add_batch(atom_loss_outputs[1:], edge_loss_outputs[1:])

            self.tracker.print_metrics(epoch, num_epochs, total_atom_loss, total_edge_loss, phase='Train')

            # Validation phase
            self.encoder.eval()
            self.decoder.eval()
            val_atom_loss = 0.0
            val_edge_loss = 0.0

            self.tracker.reset()

            with torch.no_grad():
                for batch in self.val_loader:
                    batch = batch.to(self.device)

                    z = self.encoder(batch.x.float(), batch.edge_index, batch.edge_attr.float(), batch.batch)
                    pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges = self.decoder(z, batch.batch, verbose=False)

                    atom_loss_outputs = compute_atom_loss(pred_atom_features, pred_num_atom, batch.x[:, 0], batch.batch, verbose=False)
                    edge_loss_outputs = compute_edge_loss(pred_edge_attr, pred_num_edges, batch.edge_index, batch.edge_attr, batch.batch, verbose=False)

                    val_atom_loss += atom_loss_outputs[0].item()
                    val_edge_loss += edge_loss_outputs[0].item()

                    self.tracker.add_batch(atom_loss_outputs[1:], edge_loss_outputs[1:])

            self.tracker.print_metrics(epoch, num_epochs, val_atom_loss, val_edge_loss, phase='Val')

            # Checkpointing
            if epoch % checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': self.encoder.state_dict(),
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'train_loss': total_atom_loss + total_edge_loss,
                    'val_loss': val_atom_loss + val_edge_loss,
                }, checkpoint_path)
                print(f"Checkpoint saved at {checkpoint_path}")

def compute_atom_loss(
        decoded_node_logits,
        decoded_num_nodes,
        node_targets,
        batch,
        weight_nodes_features=1.0,
        weight_node_num_pred=1.0,
        verbose=False
    ):
    """
    Compute combined loss for atom classification and node count prediction.

    Calculates cross-entropy loss for atom type prediction and MSE loss for
    node count regression, returning weighted combined loss plus predictions
    and targets for metric calculation.

    Args:
        decoded_node_logits: Predicted atom type logits [batch_size, max_num_nodes, num_classes].
        decoded_num_nodes: Predicted node counts [batch_size].
        node_targets: True atom types [total_nodes].
        batch: Batch assignment tensor mapping nodes to graph indices [total_nodes].
        weight_nodes_features: Weight for atom classification loss. Default is 1.0.
        weight_node_num_pred: Weight for node count regression loss. Default is 1.0.
        verbose: If True, print per-graph debugging information. Default is False.

    Returns:
        Tuple of (total_loss, atom_preds, atom_targets, node_count_preds, node_count_trues):
            - total_loss (Tensor): Weighted average loss for batch.
            - atom_preds (np.ndarray): Predicted atom types (argmax of logits).
            - atom_targets (np.ndarray): True atom types.
            - node_count_preds (np.ndarray): Predicted node counts.
            - node_count_trues (np.ndarray): True node counts.

    Note:
        - Uses CrossEntropyLoss for atom classification (22 atom types)
        - Uses MSELoss for node count regression
        - Returns predictions/targets for later epoch-level metric calculation
    """
    
    # Initialize loss functions
    loss_fn = nn.CrossEntropyLoss()  # For atom classification
    node_num_loss_fn = nn.MSELoss()  # For node count regression

    batch_size = decoded_node_logits.size(0)
    total_loss = 0.0

    # Accumulators for predictions and targets
    all_atom_preds = []
    all_atom_targets = []
    true_num_nodes = []
    predicted_num_nodes = decoded_num_nodes.detach().cpu()

    # Iterate through each graph in the batch
    for graph_index in range(batch_size):
        # Get true number of nodes for the current graph
        num_nodes_in_graph = (batch == graph_index).sum().item()
        true_num_nodes.append(num_nodes_in_graph)

        if verbose:
            print(f'Graph {graph_index}, Number of Atoms: {num_nodes_in_graph}')

        # Extract predictions and targets for the current graph
        graph_logits = decoded_node_logits[graph_index, :num_nodes_in_graph, :]  # [num_nodes_in_graph, num_classes]
        graph_targets = node_targets[batch == graph_index]  # [num_nodes_in_graph]

        # Atom classification loss
        atom_loss = loss_fn(graph_logits, graph_targets.long())
        total_loss += atom_loss

        # Save predictions and true values for later metric calculation
        all_atom_preds.append(torch.argmax(graph_logits, dim=1).detach().cpu())
        all_atom_targets.append(graph_targets.detach().cpu())


    # Convert node counts to tensor and calculate MSE loss
    true_num_nodes = torch.tensor(true_num_nodes, dtype=torch.float32, device=decoded_num_nodes.device)
    node_count_loss = node_num_loss_fn(decoded_num_nodes, true_num_nodes)

    # Calculate total weighted loss
    total_loss = (total_loss * weight_nodes_features) + (node_count_loss * weight_node_num_pred)
    total_loss /= batch_size

    # Collect node count predictions and true values
    node_count_preds = predicted_num_nodes.numpy()
    node_count_trues = true_num_nodes.cpu().numpy()

    # Flatten predictions and targets for later processing
    all_atom_preds = torch.cat(all_atom_preds).numpy()
    all_atom_targets = torch.cat(all_atom_targets).numpy()

    # Return loss and all predictions/targets for metric calculation outside the function
    return total_loss, all_atom_preds, all_atom_targets, node_count_preds, node_count_trues

def get_upper_tri_flat_array_with_attrs(
    edge_index,
    edge_attr,
    batch,
    graph_id,
    edge_types_num=5,
    include_diagonal=False,
    max_edges=190
):
    """
    Convert graph adjacency matrix upper triangle to flattened representation.

    Extracts upper triangular part of adjacency matrix for a single graph,
    flattens it to fixed-size array with padding, and extracts edge type
    attributes. Uses column-major ordering for consistency.

    Args:
        edge_index: Edge connectivity tensor [2, num_edges] with source and
            target node indices.
        edge_attr: Edge attributes tensor [num_edges, num_attr]. First column
            contains edge type indices (0-4 for bond types).
        batch: Batch assignment tensor [num_nodes] mapping nodes to graph IDs.
        graph_id: Integer ID of graph to extract from batch.
        edge_types_num: Number of edge type classes. Default is 5 (bond types).
        include_diagonal: If True, include diagonal in upper triangle.
            Default is False.
        max_edges: Fixed size for output padding (190 for 20 nodes max).
            Default is 190.

    Returns:
        Tuple of (edge_exists, edge_attrs, mask):
            - edge_exists (Tensor): Binary adjacency [max_edges]. 1 if edge exists,
              0 otherwise.
            - edge_attrs (Tensor): One-hot encoded edge types [num_existing_edges, edge_types_num].
            - mask (Tensor): Valid entries mask [max_edges]. 1 for valid positions,
              0 for padding.

    Raises:
        ValueError: If graph has more nodes than max_edges supports.

    Note:
        - Uses column-major order for upper triangle indexing
        - Pads output to max_edges for batch processing consistency
        - Only returns attributes for edges that exist in the graph
    """
    # Step 1: Get nodes for the graph
    graph_mask = (batch == graph_id)
    graph_nodes = torch.where(graph_mask)[0]
    num_nodes = graph_nodes.size(0)
    
    if num_nodes == 0:
        # Return empty padded tensor and all-zero mask
        return (torch.zeros(max_edges, device=edge_index.device),
                torch.zeros(0, edge_types_num, device=edge_index.device),
                torch.zeros(max_edges, dtype=torch.bool, device=edge_index.device))
    
    # Step 2: Filter edges and attributes
    edge_mask = (batch[edge_index[0]] == graph_id) & (batch[edge_index[1]] == graph_id)
    graph_edges = edge_index[:, edge_mask]
    graph_edge_attrs = edge_attr[edge_mask][:, 0].long()

    # Step 3: Remap to local indices
    sorter = torch.argsort(graph_nodes)
    local_src = sorter[torch.searchsorted(graph_nodes[sorter], graph_edges[0])]
    local_tgt = sorter[torch.searchsorted(graph_nodes[sorter], graph_edges[1])]
    
    # Step 4: Compute upper triangular size
    triu_size = (num_nodes * (num_nodes - 1)) // 2 if not include_diagonal else (num_nodes * (num_nodes + 1)) // 2
    
    if triu_size > max_edges:
        raise ValueError(f"Graph with {num_nodes} nodes has triu_size {triu_size} > max_edges {max_edges}")
    
    # Step 5: Initialize output and mask
    output = torch.zeros(max_edges, 1 + edge_types_num, device=edge_index.device)
    mask = torch.zeros(max_edges, dtype=torch.bool, device=edge_index.device)
    mask[:triu_size] = 1  # Mark valid entries
    
    # Step 6: Map edges to upper triangular positions (column-major order)
    edge_to_triu_idx = {}
    for i, (s, t) in enumerate(zip(local_src, local_tgt)):
        if s > t:
            s, t = t, s  # Ensure (s, t) with s <= t for upper triangle

        # Column-major order indexing for upper triangle
        triu_idx = (t * (t - 1)) // 2 + s if not include_diagonal else (t * (t + 1)) // 2 + s
        edge_to_triu_idx[i] = triu_idx

    # Step 7: Fill output tensor (only up to triu_size)
    for edge_idx, triu_idx in edge_to_triu_idx.items():
        if triu_idx >= max_edges:
            continue  # Avoid out-of-bounds if unexpected

        # Binary adjacency
        output[triu_idx, 0] = 1  

        # One-hot encode edge type
        edge_attr_fixed = torch.zeros(edge_types_num, device=edge_attr.device)
        edge_type_idx = graph_edge_attrs[edge_idx].item()

        if 0 <= edge_type_idx < edge_types_num:
            edge_attr_fixed[edge_type_idx] = 1  # One-hot encoding for the valid class

        # Assign one-hot encoded edge type
        output[triu_idx, 1:1 + edge_types_num] = edge_attr_fixed

    # Split the output
    edge_exists = output[:, 0]  # Binary edge existence
    edge_attrs = output[:, 1:]  # Edge type one-hot vectors

    # Mask edge attributes where edges exist
    existing_edge_attrs = edge_attrs[edge_exists.bool()]

    return edge_exists, existing_edge_attrs, mask

def compute_edge_loss(
        decoded_edge_logits,
        decoded_num_edges,
        edge_index,
        edge_attr,
        batch,
        weight_adj=1.0,
        weight_attrs=1.0,
        weight_edges_num_pred=1.0,
        max_nodes=20,
        edge_attr_classes=NUM_EDGE_TYPES,
        include_diagonal=False,
        verbose=False,
        neg_sample_ratio=1.0
    ):
    """
    Compute combined loss for edge adjacency and edge type prediction.

    Calculates binary cross-entropy loss for edge existence (with negative sampling),
    cross-entropy loss for edge type classification, and MSE loss for edge count
    regression.

    Args:
        decoded_edge_logits: Predicted edge logits [batch_size, max_triu_size, 1 + edge_attr_classes].
            First column is adjacency logits, remaining are edge type logits.
        decoded_num_edges: Predicted edge counts [batch_size].
        edge_index: Edge connectivity tensor [2, total_edges].
        edge_attr: Edge attributes tensor [total_edges, num_attr].
        batch: Batch assignment tensor [total_nodes].
        weight_adj: Weight for adjacency loss. Default is 1.0.
        weight_attrs: Weight for edge type classification loss. Default is 1.0.
        weight_edges_num_pred: Weight for edge count regression loss. Default is 1.0.
        max_nodes: Maximum number of nodes per graph. Default is 20.
        edge_attr_classes: Number of edge type classes (5 bond types). Default is NUM_EDGE_TYPES.
        include_diagonal: If True, include diagonal in adjacency matrix. Default is False.
        verbose: If True, print debugging information. Default is False.
        neg_sample_ratio: Ratio of negative to positive edge samples for adjacency loss.
            Default is 1.0 (balanced sampling).

    Returns:
        Tuple of (total_loss, adj_preds, adj_targets, type_preds, type_targets,
                  edge_count_preds, edge_count_trues):
            - total_loss (Tensor): Weighted average loss for batch.
            - adj_preds (np.ndarray): Predicted adjacency (binary).
            - adj_targets (np.ndarray): True adjacency (binary).
            - type_preds (np.ndarray): Predicted edge types (argmax of logits).
            - type_targets (np.ndarray): True edge types.
            - edge_count_preds (np.ndarray): Predicted edge counts.
            - edge_count_trues (np.ndarray): True edge counts.

    Note:
        - Uses negative sampling to balance adjacency loss (most pairs are non-edges)
        - Edge type loss only computed for positive edges
        - Returns predictions/targets for later epoch-level metric calculation
    """
    adj_loss_fn = nn.BCEWithLogitsLoss()
    type_loss_fn = nn.CrossEntropyLoss()
    edges_num_loss_fn = nn.MSELoss()  # For node count regression

    batch_size = decoded_edge_logits.size(0)
    adj_loss = 0
    type_loss = 0

    # Collect true number of nodes per graph
    true_num_edges = []  

    # Accumulators for predictions and targets
    all_adj_preds = []
    all_adj_targets = []
    all_type_preds = []
    all_type_targets = []
    predicted_num_edges = decoded_num_edges.detach().cpu()

    for graph_index in range(batch_size):
        num_nodes_in_graph = (batch == graph_index).sum().item()
        if num_nodes_in_graph == 0:
            continue
        
        triu_size = (num_nodes_in_graph * (num_nodes_in_graph - 1)) // 2 if not include_diagonal else (num_nodes_in_graph * (num_nodes_in_graph + 1)) // 2
        edge_exists, existing_edge_attrs, mask = get_upper_tri_flat_array_with_attrs(
            edge_index, edge_attr, batch, graph_index, edge_attr_classes, include_diagonal
        )  # [triu_size, 2]
        

        graph_logits = decoded_edge_logits[graph_index, :triu_size, :1 + edge_attr_classes]  # [triu_size, 23]
        
        # if verbose:
        #     num_edges = (graph_targets[:, 0] == 1).sum().item()
        #     print(f'Graph {graph_index}, Nodes: {num_nodes_in_graph}, Triu Size: {triu_size}, Edges: {num_edges}')
        
        adj_logits = graph_logits[:, 0]  # [triu_size]
        adj_targets = edge_exists[mask].float()#graph_targets[:, 0].float()  # [triu_size]

        # Positive edges
        pos_edge_mask = adj_targets == 1
        num_pos_edges = pos_edge_mask.sum().item()

        if num_pos_edges > 0:
            # Negative sampling: randomly sample non-edges
            num_neg_samples = int(neg_sample_ratio * num_pos_edges)
            neg_candidate_mask = adj_targets == 0  # All non-edge positions in triu
            neg_indices = torch.where(neg_candidate_mask)[0]
            if len(neg_indices) > num_neg_samples:
                neg_sampled_indices = neg_indices[torch.randperm(len(neg_indices))[:num_neg_samples]]
            else:
                neg_sampled_indices = neg_indices
            
            # Combine positive and sampled negative examples for adjacency loss
            sampled_indices = torch.cat([torch.where(pos_edge_mask)[0], neg_sampled_indices])
            sampled_logits = adj_logits[sampled_indices]
            sampled_targets = adj_targets[sampled_indices]

            adj_loss += adj_loss_fn(sampled_logits, sampled_targets)

            # Store adjacency predictions and true values (still over full triu for consistency)
            all_adj_preds.append((torch.sigmoid(adj_logits) > 0.5).detach().cpu())
            all_adj_targets.append(adj_targets.detach().cpu())
            
            # Edge type loss (only for positive edges)
            type_logits = graph_logits[pos_edge_mask, 1:]  # [num_edges, 22]
            type_targets = existing_edge_attrs # graph_targets[pos_edge_mask, 1].long()  # [num_edges], integer labels (0-21)
            type_loss += type_loss_fn(type_logits, type_targets)

            # Store type predictions and true values
            all_type_preds.append(torch.argmax(type_logits, dim=1).detach().cpu())
            all_type_targets.append(type_targets.detach().cpu())

            true_num_edges.append(type_targets.size(0))
        else:
            # No positive edges, still store adjacency info
            all_adj_preds.append((torch.sigmoid(adj_logits) > 0.5).detach().cpu())
            all_adj_targets.append(adj_targets.detach().cpu())
            true_num_edges.append(0)

    # Convert edge counts to tensors and compute MSE loss
    true_num_nodes = torch.tensor(true_num_edges, dtype=torch.float32, device=decoded_num_edges.device)
    edges_count_loss = edges_num_loss_fn(decoded_num_edges, true_num_nodes)

    # Collect edge count predictions and true values
    edge_count_preds = np.array(predicted_num_edges)
    edge_count_trues = np.array(true_num_edges)

    # Flatten predictions and targets for later evaluation
    all_adj_preds = torch.cat(all_adj_preds).numpy()
    all_adj_targets = torch.cat(all_adj_targets).numpy()

    if all_type_preds:  # Avoid empty tensor issues if no edges exist
        all_type_preds = torch.cat(all_type_preds).numpy()
        all_type_targets = torch.cat(all_type_targets).numpy().argmax(1)
    else:
        all_type_preds = np.array([])
        all_type_targets = np.array([])

    # Calculate weighted total loss
    total_loss = (adj_loss * weight_adj) + (type_loss * weight_attrs) + (edges_count_loss * weight_edges_num_pred)
    total_loss /= batch_size

    return total_loss, all_adj_preds, all_adj_targets, all_type_preds, all_type_targets, edge_count_preds, edge_count_trues

class MetricsTracker:
    """
    Metrics Tracker for Graph Autoencoder Training.

    Accumulates predictions and targets across batches for epoch-level metric
    computation. Tracks atom classification, edge adjacency, edge type
    classification, and count predictions.

    Metrics computed:
    - Atom type accuracy, F1-score, balanced accuracy
    - Atom count R² score
    - Edge adjacency accuracy, F1-score, balanced accuracy
    - Edge type accuracy, F1-score, balanced accuracy
    - Edge count R² score
    """

    def __init__(self):
        """
        Initialize MetricsTracker with empty accumulators.

        Creates separate accumulators for atom-related metrics (atom types,
        atom counts) and edge-related metrics (adjacency, edge types, edge counts).
        """
        # Atom-related accumulators
        self.atom_preds = []
        self.atom_targets = []
        self.node_count_preds = []
        self.node_count_trues = []

        # Edge-related accumulators
        self.adj_preds = []
        self.adj_targets = []
        self.edge_type_preds = []
        self.edge_type_targets = []
        self.edge_count_preds = []
        self.edge_count_trues = []

    def add_batch(self, atom_loss_outputs, edge_loss_outputs):
        """
        Add predictions and targets from single batch to accumulators.

        Args:
            atom_loss_outputs: Tuple of (atom_preds, atom_targets, node_count_preds,
                node_count_trues) from compute_atom_loss().
            edge_loss_outputs: Tuple of (adj_preds, adj_targets, edge_type_preds,
                edge_type_targets, edge_count_preds, edge_count_trues) from
                compute_edge_loss().

        Note:
            Called after each batch during training/validation to accumulate
            predictions for epoch-level metrics.
        """
        # Unpack atom loss outputs
        atom_preds, atom_targets, node_count_preds, node_count_trues = atom_loss_outputs
        self.atom_preds.append(atom_preds)
        self.atom_targets.append(atom_targets)
        self.node_count_preds.append(node_count_preds)
        self.node_count_trues.append(node_count_trues)

        # Unpack edge loss outputs
        adj_preds, adj_targets, edge_type_preds, edge_type_targets, edge_count_preds, edge_count_trues = edge_loss_outputs
        self.adj_preds.append(adj_preds)
        self.adj_targets.append(adj_targets)
        self.edge_type_preds.append(edge_type_preds)
        self.edge_type_targets.append(edge_type_targets)
        self.edge_count_preds.append(edge_count_preds)
        self.edge_count_trues.append(edge_count_trues)

    def compute_metrics(self):
        """
        Compute epoch-level metrics from accumulated predictions.

        Concatenates all batch predictions and targets, then computes classification
        metrics (accuracy, F1, balanced accuracy) and regression metrics (R²).

        Returns:
            Dictionary with 11 metrics:
                - atom_accuracy: Atom type classification accuracy
                - atom_f1: Atom type macro F1-score
                - balanced_atom_acc: Atom type balanced accuracy
                - atom_num_r2: Atom count R² score
                - adj_accuracy: Edge adjacency accuracy
                - adj_f1: Edge adjacency macro F1-score
                - balanced_adj_acc: Edge adjacency balanced accuracy
                - edge_type_accuracy: Edge type classification accuracy
                - edge_type_f1: Edge type macro F1-score
                - balanced_edge_type_acc: Edge type balanced accuracy
                - edge_num_r2: Edge count R² score

        Note:
            Requires at least one batch added via add_batch() before calling.
        """
        # Flatten all predictions and targets
        epoch_atom_preds = np.concatenate(self.atom_preds)
        epoch_atom_targets = np.concatenate(self.atom_targets)
        epoch_node_count_preds = np.concatenate(self.node_count_preds)
        epoch_node_count_trues = np.concatenate(self.node_count_trues)

        epoch_adj_preds = np.concatenate(self.adj_preds)
        epoch_adj_targets = np.concatenate(self.adj_targets)
        epoch_edge_type_preds = np.concatenate(self.edge_type_preds)
        epoch_edge_type_targets = np.concatenate(self.edge_type_targets)
        epoch_edge_count_preds = np.concatenate(self.edge_count_preds)
        epoch_edge_count_trues = np.concatenate(self.edge_count_trues)

        # Atom-related metrics
        atom_accuracy = accuracy_score(epoch_atom_targets, epoch_atom_preds)
        atom_f1 = f1_score(epoch_atom_targets, epoch_atom_preds, average='macro')
        balanced_atom_acc = balanced_accuracy_score(epoch_atom_targets, epoch_atom_preds)
        atom_num_r2 = r2_score(epoch_node_count_trues, epoch_node_count_preds)

        # Edge-related metrics
        adj_accuracy = accuracy_score(epoch_adj_targets, epoch_adj_preds)
        adj_f1 = f1_score(epoch_adj_targets, epoch_adj_preds, average='macro')
        balanced_adj_acc = balanced_accuracy_score(epoch_adj_targets, epoch_adj_preds)

        edge_type_accuracy = accuracy_score(epoch_edge_type_targets, epoch_edge_type_preds)
        edge_type_f1 = f1_score(epoch_edge_type_targets, epoch_edge_type_preds, average='macro')
        balanced_edge_type_acc = balanced_accuracy_score(epoch_edge_type_targets, epoch_edge_type_preds)
        edge_num_r2 = r2_score(epoch_edge_count_trues, epoch_edge_count_preds)

        # Return as dictionary for easy access
        metrics = {
            'atom_accuracy': atom_accuracy,
            'atom_f1': atom_f1,
            'balanced_atom_acc': balanced_atom_acc,
            'atom_num_r2': atom_num_r2,
            'adj_accuracy': adj_accuracy,
            'adj_f1': adj_f1,
            'balanced_adj_acc': balanced_adj_acc,
            'edge_type_accuracy': edge_type_accuracy,
            'edge_type_f1': edge_type_f1,
            'balanced_edge_type_acc': balanced_edge_type_acc,
            'edge_num_r2': edge_num_r2
        }
        return metrics

    def reset(self):
        """
        Clear all accumulators for next epoch.

        Empties all prediction and target lists to prepare for accumulating
        data from the next epoch.
        """
        self.atom_preds.clear()
        self.atom_targets.clear()
        self.node_count_preds.clear()
        self.node_count_trues.clear()
        self.adj_preds.clear()
        self.adj_targets.clear()
        self.edge_type_preds.clear()
        self.edge_type_targets.clear()
        self.edge_count_preds.clear()
        self.edge_count_trues.clear()

    def print_metrics(self, epoch, num_epochs, total_atom_loss, total_edge_loss, phase='Train'):
        """
        Print comprehensive metrics for current epoch.

        Args:
            epoch: Current epoch number (0-indexed).
            num_epochs: Total number of epochs for training.
            total_atom_loss: Total atom loss for epoch.
            total_edge_loss: Total edge loss for epoch.
            phase: Training phase ('Train' or 'Val'). Default is 'Train'.

        Example output:
            Epoch [151/300] ------------------------------------- Train
              Train Atom Loss: 1.2345, Train Edge Loss: 2.3456, Train Total Loss: 3.5801
              Train atom_accuracy: 0.8523, Train atom_f1: 0.7845, ...
        """
        metrics = self.compute_metrics()
        print(f"Epoch [{epoch+1}/{num_epochs}] ------------------------------------- {phase}")
        print(f"  {phase} Atom Loss: {total_atom_loss:.4f}, {phase} Edge Loss: {total_edge_loss:.4f}, "
              f"{phase} Total Loss: {total_atom_loss + total_edge_loss:.4f}")
        print(f"  {phase} atom_accuracy: {metrics['atom_accuracy']:.4f}, {phase} atom_f1: {metrics['atom_f1']:.4f}, "
              f"{phase} balanced_atom_acc: {metrics['balanced_atom_acc']:.4f}, {phase} atom_num_r2: {metrics['atom_num_r2']:.4f}")
        print(f"  {phase} edge_type_accuracy: {metrics['edge_type_accuracy']:.4f}, {phase} edge_type_f1: {metrics['edge_type_f1']:.4f}, "
              f"{phase} balanced_edge_type_acc: {metrics['balanced_edge_type_acc']:.4f}, {phase} edge_num_r2: {metrics['edge_num_r2']:.4f}")
        print(f"  {phase} adj_accuracy: {metrics['adj_accuracy']:.4f}, {phase} adj_f1: {metrics['adj_f1']:.4f}, "
              f"{phase} balanced_adj_acc: {metrics['balanced_adj_acc']:.4f}\n")

# Example usage
if __name__ == "__main__":
    from GSGE.core_gsge import CoreGSGE
    from GSGE import get_use_examples_dir

    # Get use_examples directory
    examples_dir = get_use_examples_dir()
    if examples_dir is None:
        raise RuntimeError("Cannot find use_examples directory. Run from source checkout.")

    # Load in data
    vocab_path = examples_dir / '00_making_vocabs' / 'vocabs' / 'GS_vocab_v5'
    corpus_path = examples_dir / '00_making_vocabs' / 'vocabs' / 'GSGE_corpus_v5'
    batch_size = 64

    train_loader, val_loader = CoreGSGE.load_and_prepare_data(str(vocab_path), str(corpus_path), x_percent=0.2, seed=42, batch_size=batch_size)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Models
    encoder = AttentiveFP(in_channels=9, hidden_channels=128, out_channels=32, edge_dim=3, num_layers=3, num_timesteps=2)
    decoder = GraphDecoder(latent_dim=32, hidden_dim=128)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)

    # Paths
    gae_dir = examples_dir / '03_GAE' / 'v2'
    checkpoint_dir = str(gae_dir / 'model_checkpoints')
    load_checkpoint_path = str(gae_dir / 'model_checkpoints' / 'checkpoint_epoch_300.pth')

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
    trainer.train(num_epochs=301, checkpoint_interval=5)