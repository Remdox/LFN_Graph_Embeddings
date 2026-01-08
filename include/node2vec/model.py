import os
import numpy

from torch_geometric.data import Data

import networkx as nx
from node2vec import Node2Vec

def run_data(graph: Data) -> numpy.ndarray:
    """
    Trains the GraphSage model and produces node embeddings.

    Parameters:
    - graph: a PyTorch Geometric Data object.

    Returns:
    - A numpy ndarray of shape (num_nodes, 128).
    """

    # Convert PyG Data to NetworkX graph
    edge_index = graph.edge_index.numpy()
    edge_attr = graph.edge_attr.numpy()
    G_nx = nx.Graph()
    for i in range(edge_index.shape[1]):
        u = edge_index[0, i]
        v = edge_index[1, i]
        w = edge_attr[i]
        G_nx.add_edge(u, v, weight=w)

    # Create Node2Vec model
    node2vec = Node2Vec(
                        graph=G_nx, 
                        dimensions=128, 
                        walk_length=20, 
                        num_walks=10, 
                        p=1.0, 
                        q=1.0, 
                        weight_key='weight',
                        workers=os.cpu_count(), 
                        quiet=True
                        )

    # Fit model
    model = node2vec.fit(window=10, min_count=1, batch_words=4)

    print("Training completed!")

    # Get embeddings for all nodes
    embeddings = numpy.zeros((graph.num_nodes, 128))

    for node in G_nx.nodes():
        embeddings[node] = model.wv[str(node)]

    return embeddings
