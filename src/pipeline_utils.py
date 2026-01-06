import torch
from torch_geometric.data import Data
from torch_geometric.utils import negative_sampling
import dataset_utils
from dataset_utils import Graph

# TODO: hard negative sampling?
def sample_negative_edges(G: Graph, negative_sample_size: int, excluded_edges: torch.Tensor|None = None) -> Graph:
    """
    Generates negative edges from a graph, which can be given as directed or undirected, weighted or unweighted.

    Parameters:
    - G: tuple containing a PyTorch Geometric Data object representing the graph to sample from, a bool stating if the graph is directed, a bool stating if the graph is weighted
    - negative_sample_size : integer that defines how many negative edges to generate.
    - excluded_edges [optional]: tensor object containing the edges to exclude from the negative sampling

    Returns:
    - A dataset_utils.Graph object containing: a PyTorch Geometric Data object representing the negative edges, a bool stating if the graph is directed, a bool stating if the graph is weighted
    """
    device = G.graph_data.edge_index.device

    if excluded_edges is not None:
        excluded_edges = excluded_edges.to(device)
        positive_edges = torch.cat([G.graph_data.edge_index, excluded_edges], dim=1)
    else:
        positive_edges = G.graph_data.edge_index

    negative_edges_index = negative_sampling(positive_edges, G.graph_data.num_nodes, negative_sample_size, force_undirected=not G.is_directed, method='sparse')

    if G.is_weighted:
        random_indices = torch.randint(0, G.graph_data.edge_attr.size(0),(negative_edges_index.size(1),), device=device)
        weights = G.graph_data.edge_attr[random_indices]
    else:
        weights = torch.ones(negative_edges_index.size(1), device=device)

    negative_graph = Data(edge_index=negative_edges_index, edge_attr=weights, num_nodes=G.graph_data.num_nodes)

    # check if enough negative edges were found
    min_neg_edges = negative_sample_size * (1 if G.is_directed else 2)
    num_neg_edges = negative_edges_index.size(1) // (1 if G.is_directed else 2)
    if num_neg_edges < min_neg_edges:
        print(f"WARNING: cannot sample a sufficient number of negative edges for the size requested ({num_neg_edges}/{negative_sample_size}).")

    return Graph(negative_graph, G.is_directed, G.is_weighted)


def split_graph_data(G: Graph, val_ratio: float, test_ratio: float) -> tuple[Graph, Graph, Graph]:
    """
    Generates train, val, test splits of a graph, which can be given as directed or undirected, weighted or unweighted.

    Parameters:
    - G: tuple containing a PyTorch Geometric Data object representing the graph to sample from, a bool stating if the graph is directed, a bool stating if the graph is weighted
    - val_ratio: float > 0 defining the percentage of edges in the validation split
    - test_ratio: float > 0 defining the percentage of edges in the test split

    Returns:
    - A tuple containing three Graph objects
    """
    if G.is_directed:
        device = G.graph_data.edge_index.device
        graph_perm = torch.randperm(G.graph_data.edge_index.size(1), device=device)
    else:
        # removing reverse edges, will be re-added later by create_split
        mask = G.graph_data.edge_index[0] <= G.graph_data.edge_index[1]
        graph_perm = mask.nonzero(as_tuple=False).view(-1)
        graph_perm = graph_perm[torch.randperm(graph_perm.size(0), device=graph_perm.device)]

    val_size  = int(graph_perm.numel() * val_ratio)
    test_size = int(graph_perm.numel() * test_ratio)
    train_size = graph_perm.numel() - val_size - test_size
    if train_size <= 0:
        raise ValueError("Training size is 0")

    train_edges     = graph_perm[                   :train_size]
    val_edges       = graph_perm[train_size         :train_size+val_size]
    test_edges      = graph_perm[train_size+val_size:]
    train_val_edges = graph_perm[                   :train_size+val_size]

    train_split = create_split(G, train_edges)
    val_split   = create_split(G, val_edges)
    test_split  = create_split(G, test_edges)

    return (train_split, val_split, test_split)

# Data contains a key-value dictionary called store, which collects all tensors
# used to define an homogeneous graph (for both nodes and edges).
# The structure of store is defined as follows:
# items: |          x              |       edge_index       |        edge_attr        |   y (label)  |
#        |-------------------------|------------------------|-------------------------|--------------|
# value: |      node features      |   from-to connections  |      edge features      |  node labels |
# size:  |  [n_nodes x m_features] |      [2 x n_edges]     |  [n_edges x m_features] |   [n_nodes]  |
#
# All "value" entries are tensors.
# The goal of create_split is to find the value of these tensors for a given split, by taking a subset of indices tensor belonging to the graph to split
def create_split(G:Graph, index:torch.Tensor) -> Graph:
    """
    Generates a split from the PyG Data object.

    Parameters:
    - data: dataset_utils.Graph object representing the graph to split
    - index: Tensor containing the edge indices of the split

    Assumptions:
    - The weight is appended as edge_attr and there must be no manually-added key-value tensor in Data where the tensor isn't an edge feature but its size is still equal to the number of edges.
    - There must not be manually-added key-value tensors in Data where the tensor isn't a node feature but its size is still equal to the number of nodes

    Returns:
    - A dataset_utils.Graph object representing the graph with the split of edges
    """
    data = G.graph_data
    splitted_data = Data()
    index = index.to(G.graph_data.edge_index.device)

    # creating tensor for split of node attributes not in x, node labels and metadata
    num_nodes = data.num_nodes
    splitted_data.num_nodes = num_nodes
    for key, value in data.items():
        if torch.is_tensor(value) and value.size(0) == num_nodes and key not in ['edge_index', 'edge_attr']:
            splitted_data[key] = value

    # creating tensor for split of edge attributes
    num_edges = data.edge_index.size(1)
    for key,value in data.items():
        if torch.is_tensor(value) and value.size(0) == num_edges and key != 'edge_index':
            value = value[index]
            if not G.is_directed:
                value = torch.cat([value, value], dim=0)
            splitted_data[key] = value

    # creating tensor for split of edge connections from-to
    edge_index = data.edge_index[:, index]
    if not G.is_directed:
        edge_index = torch.cat([edge_index, edge_index.flip([0])], dim=1)
    splitted_data.edge_index = edge_index

    return Graph(splitted_data, G.is_directed, G.is_weighted)

# This function makes use of the same principles used for split_graph_data and create_split.
# See the above functions for more details
def merge_negative_edges(G_pos: Graph, G_neg: Graph):
    """
    Merges to a graph with positive edges one with negative edges.

    Parameters:
    - G_pos: dataset_utils.Graph object containing the graph of positive edges
    - G_neg: dataset_utils.Graph object containing the graph of negative edges

    Assumptions:
    - node attributes, node labels and additional metadata from the positive graph are used and metadata that may come from negative graph is ignored

    Returns:
    - A dataset_utils.Graph object representing the graph with merged and shuffled positive and negative edges
    - A torch.Tensor containing the corresponding labels for the edges of the merged graph
    """
    pos_data = G_pos.graph_data
    neg_data = G_neg.graph_data
    device = pos_data.edge_index.device

    merged_data = Data()

    # Adding node attributes, node labels, metadata from G_pos
    for key, value in pos_data.items():
        if torch.is_tensor(value) and value.size(0) == pos_data.num_nodes and key not in ['edge_index', 'edge_attr']:
            merged_data[key] = value

    # merge + shuffle of edge connections from-to
    merged_graph_edges = torch.cat([pos_data.edge_index, neg_data.edge_index], dim=1)
    perm = torch.randperm(merged_graph_edges.size(1), device=device)
    shuffled_edge_index = merged_graph_edges[:, perm]
    merged_data.edge_index = shuffled_edge_index

    # merge + shuffle of edge attributes
    if pos_data.edge_attr is not None:
        merged_graph_edge_attr = torch.cat([pos_data.edge_attr, neg_data.edge_attr], dim=0)
        shuffled_edge_attr = merged_graph_edge_attr[perm]
    else:
        shuffled_edge_attr = None
    merged_data.edge_attr  = shuffled_edge_attr

    # merge + shuffle of edge labels
    pos_labels = torch.ones(pos_data.edge_index.size(1), device=device)
    neg_labels = torch.zeros(neg_data.edge_index.size(1), device=device)
    merged_labels = torch.cat([pos_labels, neg_labels], dim=0)
    shuffled_edge_labels = merged_labels[perm]

    merged_graph = Graph(merged_data, G_pos.is_directed, G_pos.is_weighted)

    return (merged_graph, shuffled_edge_labels)


