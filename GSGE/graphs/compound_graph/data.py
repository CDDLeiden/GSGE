import torch
from torch_geometric.data import Dataset, Data
import matplotlib.pyplot as plt
from group_selfies.group_encoder import MolecularGraph
from tqdm import tqdm
from rdkit import Chem
from concurrent.futures import ProcessPoolExecutor, as_completed
from rdkit.Chem import AllChem
import numpy as np
from collections import defaultdict
import networkx as nx
from group_selfies import GroupGrammar


class compound_graph(MolecularGraph):
    """
    A group-level molecular graph built on top of MolecularGraph (from group_selfies).

    This class extends MolecularGraph to support:
    - Extraction of edge indices and adjacency matrices
    - Canonical graph reordering based on DFS traversal
    - Visualization of group-level molecular graphs using NetworkX or RDKit-style layouts

    Key Methods:
    - get_graph_data(): Returns linearly reordered node and edge data
    - plot_graph(): Visualizes the graph with NetworkX
    - plot_graph_rd_c_style(): RDKit-style 2D layout visualization with group-aware coloring

    Note:
    Use `get_graph_data()` to obtain properly ordered node/edge data for consistent representation.
    """
    
    def _get_bond_index(self):
        return [
            list(bond.group_idx_dict)  # No need for .keys(), dict itself is iterable
            for bond in self.get_bonds()
            if bond.group_idx_dict  # More efficient than len(bond.group_idx_dict.keys()) > 0
        ]

    def _get_node_ids(self):
        return [int(group.name.rsplit('_', 1)[-1]) for group in self.groups]  # rsplit is more efficient

    def _get_adj_matrix(self, bond_index=None):
        if bond_index is None:
            bond_index = self._get_bond_index()

        num_nodes = len(self.groups)
        adj_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

        # Convert bonds to numpy array for vectorized indexing
        edges = np.array([(bond[i], bond[j]) for bond in bond_index for i in range(len(bond)) for j in range(i + 1, len(bond))], dtype=int)

        if edges.size == 0:  # Handle empty case
            return adj_matrix

        # Vectorized assignment for adjacency matrix
        adj_matrix[edges[:, 0], edges[:, 1]] = 1
        adj_matrix[edges[:, 1], edges[:, 0]] = 1  # Since it's undirected

        return adj_matrix

    def _get_edge_index(self):
        bond_index = self._get_bond_index()
        
        # Use list comprehension for better performance
        edges = [(bond[i], bond[j]) for bond in bond_index for i in range(len(bond)) for j in range(i + 1, len(bond))]

        if not edges:  # Handle empty case early
            return np.empty((2, 0), dtype=int)

        # Convert to NumPy array more efficiently
        source_nodes, target_nodes = zip(*edges)
        
        # Stack with reverse edges in one step
        edge_index = np.array([source_nodes + target_nodes, target_nodes + source_nodes], dtype=int)
        
        return edge_index
    
    def _reorder_graph_linearly(self):
        """
        Reorders the graph to linearly follow paths from degree-1 nodes if possible.
        Safely handles empty graphs and graphs with no bonds.
        """

        edge_array, node_id_list = self._get_edge_index(), self._get_node_ids()

        edge_array = np.array(edge_array)
        num_nodes = len(node_id_list)

        # ------------------------------------------------------------------
        # SAFETY: handle empty graph or no bonds
        # ------------------------------------------------------------------
        if num_nodes == 0:
            # Completely empty compound graph
            return [], np.zeros((2, 0), dtype=int), np.zeros((0, 2), dtype=int)

        if edge_array.size == 0:
            # Nodes exist, but no edges
            new_node_id_list = list(node_id_list)
            new_edge_array = np.zeros((2, 0), dtype=int)
            bond_index = np.zeros((0, 2), dtype=int)
            return new_node_id_list, new_edge_array, bond_index

        # ------------------------------------------------------------------
        # Normal case: graph has edges
        # ------------------------------------------------------------------

        graph = defaultdict(list)
        for u, v in zip(edge_array[0], edge_array[1]):
            graph[u].append(v)
            graph[v].append(u)  # undirected graph

        # Find starting node (prefer degree-1)
        degree_one_nodes = [node for node, neighbors in graph.items() if len(neighbors) == 1]
        start_node = degree_one_nodes[0] if degree_one_nodes else edge_array[0][0]

        visited = set()
        order = []

        def dfs(node):
            visited.add(node)
            order.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor)

        dfs(start_node)

        # Handle disconnected components
        for node in range(num_nodes):
            if node not in visited:
                dfs(node)

        # Build mappings
        old_to_new = {old: new for new, old in enumerate(order)}
        new_to_old = {new: old for old, new in old_to_new.items()}

        # Remap node types
        new_node_id_list = [node_id_list[new_to_old[i]] for i in range(len(order))]

        # Remap edges
        new_edge_array = np.array(
            [[old_to_new[u], old_to_new[v]] for u, v in zip(edge_array[0], edge_array[1])],
            dtype=int
        ).T

        bond_index = edge_index_to_bond_index(new_edge_array)

        return new_node_id_list, new_edge_array, bond_index
    
    def get_graph_data(self):
        return self._reorder_graph_linearly()

    def plot_graph(self, bond_index=None, node_ids=None):
        
        if bond_index is None and node_ids is None:
            node_ids, _, bond_index = self._reorder_graph_linearly()
        elif bond_index is None or node_ids is None:
            raise ValueError('Either bond_index and node_ids must both be None or both provided')

        adj_matrix = self._get_adj_matrix(bond_index)

        # Create a NetworkX graph object from the adjacency matrix
        G = nx.from_numpy_array(adj_matrix)
        
        labels = {i: f"{node_ids[i]} | {i}" for i in range(len(node_ids))}
        
        # Plot the graph with labels
        pos = nx.spring_layout(G)  # Positioning of nodes
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color="skyblue", font_size=10)
        plt.show()
    
    def plot_graph_rd_c_style(
        self,
        bond_index=None,
        node_ids=None,
        color_by_group=True,
        figsize=(8, 6),
        node_size=700,
        font_size=10,
        rotate=0,
        flip_x=False,
        flip_y=False,
        label_fmt="{group} | {idx}",
        cmap=plt.cm.tab20,
        edge_color="black",
        width=1.0,
        node_color_map=None
        ):
        
        if bond_index is None and node_ids is None:
            node_ids, _, bond_index = self._reorder_graph_linearly()
        elif bond_index is None or node_ids is None:
            raise ValueError('Either bond_index and node_ids must both be None or both provided')

        adj_matrix = self._get_adj_matrix(bond_index)
        G = nx.from_numpy_array(adj_matrix)
        
        labels = {i: label_fmt.format(group=node_ids[i], idx=i) for i in range(len(node_ids))}

        # Build RDKit molecule
        mol = Chem.RWMol()
        n_nodes = adj_matrix.shape[0]
        for _ in range(n_nodes):
            mol.AddAtom(Chem.Atom(6))  # Carbon
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if adj_matrix[i, j] > 0:
                    mol.AddBond(i, j, Chem.BondType.SINGLE)

        AllChem.Compute2DCoords(mol)
        conf = mol.GetConformer()
        pos = {i: (conf.GetAtomPosition(i).x, conf.GetAtomPosition(i).y) for i in range(n_nodes)}

        # Apply transformations
        for k in pos:
            x, y = pos[k]
            if rotate:
                theta = np.radians(rotate)
                x, y = x*np.cos(theta) - y*np.sin(theta), x*np.sin(theta) + y*np.cos(theta)
            if flip_x:
                x = -x
            if flip_y:
                y = -y
            pos[k] = (x, y)

        plt.figure(figsize=figsize)

        if node_color_map is not None:
            # Use externally provided color map: {fragment_id: (R, G, B, A)}
            node_colors = [node_color_map.get(nid, (0.7, 0.7, 0.7, 1.0)) for nid in node_ids]
            nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size,
                    node_color=node_colors, font_size=font_size,
                    edge_color=edge_color, width=width)
        elif color_by_group:
            group_ids = [int(label.split(' | ')[0]) for label in labels.values()]
            unique_groups = sorted(set(group_ids))
            group_to_color = {group: idx for idx, group in enumerate(unique_groups)}
            node_colors = [group_to_color[group_id] for group_id in group_ids]

            nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size,
                    node_color=node_colors, cmap=cmap, font_size=font_size,
                    edge_color=edge_color, width=width)
        else:
            nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size,
                    node_color="skyblue", font_size=font_size,
                    edge_color=edge_color, width=width)

        plt.show()

def edge_index_to_bond_index(edge_index):
    # edge_index is a 2 x N numpy array or list
    return [list(pair) for pair in zip(edge_index[0], edge_index[1])]

def reorder_graph_linearly(edge_array:np.array, node_id_list:list):
    """
    Reorders the graph to linearly follow paths from degree-1 nodes if possible.
    Traverses the graph using DFS-like logic for linear layout, producing:
    - Reordered node_id_list
    - Remapped edge list (with new indices)
    - Mapping from old to new indices
    """

    # Build adjacency list
    edge_array = np.array(edge_array)
    graph = defaultdict(list)
    for u, v in zip(edge_array[0], edge_array[1]):
        graph[u].append(v)
        graph[v].append(u)  # undirected graph

    # Find a good starting node (prefer one with degree 1)
    degree_one_nodes = [node for node, neighbors in graph.items() if len(neighbors) == 1]
    start_node = degree_one_nodes[0] if degree_one_nodes else edge_array[0][0]

    visited = set()
    order = []

    def dfs(node):
        visited.add(node)
        order.append(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                dfs(neighbor)

    dfs(start_node)

    # Handle disconnected components
    for node in range(len(node_id_list)):
        if node not in visited:
            dfs(node)

    # Build mapping
    old_to_new = {old: new for new, old in enumerate(order)}
    new_to_old = {new: old for old, new in old_to_new.items()}

    # Remap node types
    new_node_id_list = [node_id_list[new_to_old[i]] for i in range(len(order))]

    # Remap edges
    new_edge_array = np.array([
        [old_to_new[u], old_to_new[v]] for u, v in zip(edge_array[0], edge_array[1])
    ]).T

    bond_index = edge_index_to_bond_index(new_edge_array)

    return new_node_id_list, new_edge_array, bond_index

def smiles_to_group_graph(
    smiles:str, 
    group_grammar:GroupGrammar, 
    return_CG_object:bool=False
    ):

    mol = Chem.MolFromSmiles(smiles)
    groups = group_grammar.extract_groups(mol)
    Chem.Kekulize(mol, clearAromaticFlags=True)
    graph = compound_graph(mol=mol, groups=groups)

    if return_CG_object == True:
        return graph

    new_node_id_list, new_edge_array, bond_index = graph.get_graph_data()

    return new_edge_array, new_node_id_list

def parallel_(smiles_list:list, process_func, max_workers:None|int=None):
    """Parallel processing with real-time tqdm updates."""
    
    results = [None] * len(smiles_list)  # Preallocate for order consistency
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_func, smiles): i
            for i, smiles in enumerate(smiles_list)
        }

        with tqdm(total=len(smiles_list), desc="parallel_tokenize_batch") as pbar:
            for future in as_completed(futures):
                i = futures[future]
                results[i] = future.result()
                pbar.update(1)

    return results

def preprocess_graph(smiles:str, group_grammar:GroupGrammar):
    edge_index, node_ids = smiles_to_group_graph(smiles, group_grammar)
    return edge_index, node_ids

class PreprocessedGraphDataset(Dataset):
    
    """
    PyTorch Dataset for Preprocessed Graph Data

    This dataset wraps preprocessed graph structures (edge indices and node features),
    optionally applies a custom embedding layer, and provides labels for supervised tasks.

    Designed for use with PyTorch Geometric models.

    Parameters:
    - preprocessed_data (list of tuples): Each item is (edge_index, node_features)
    - y (list or array-like, optional): Labels corresponding to each graph
    - emb_layer (nn.Module, optional): Embedding layer to apply to node features
    """

    def __init__(self, preprocessed_data, y=None, emb_layer=None):
        
        super().__init__()
        self.preprocessed_data = preprocessed_data
        self.emb_layer = emb_layer
        self.y = y

    def __len__(self):
        return len(self.preprocessed_data)

    def __getitem__(self, idx):
        edge_index, node_features  = self.preprocessed_data[idx]

        # Convert edge_index to a PyTorch tensor
        edge_index = torch.tensor(edge_index,
                                dtype=torch.long)

        # Convert node_classes to a tensor (assuming these are features)
        node_features = torch.tensor(node_features,
                                    dtype=torch.float)##.unsqueeze(1)  # Shape: [16, 1]
        
        if self.emb_layer is not None:
            node_features = self.emb_layer(node_features)
        
        # Create the Data object
        return Data(
            x=node_features, 
            edge_index=edge_index, 
            y=torch.tensor(self.y[idx],dtype=torch.long)if self.y is not None else None)