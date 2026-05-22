if __name__ == "__main__":
    from GSGE.graphs.fragment_graph.GAE import GraphDecoder
    import time
    import torch

    device = 'cuda'
    decoder = GraphDecoder(latent_dim=32, hidden_dim=64).to(device)
    
    z = torch.randn(64, 32).to(device)  # batch_size=64, latent_dim=32
    batch = torch.cat([torch.full((10,), i, dtype=torch.long) for i in range(64)]).to(device)  # Dummy batch tensor

    # Start timing
    start_time = time.time()
    
    # Forward pass
    pred_atom_features, pred_num_atom, pred_edge_attr, pred_num_edges = decoder(z, batch)
    
    # End timing
    end_time = time.time()
    
    # Print the time taken
    print(f"Forward pass time: {end_time - start_time:.6f} seconds")