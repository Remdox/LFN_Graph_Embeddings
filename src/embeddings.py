import sys
import torch
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod

sys.path.append('.')
from dataset_utils import Graph
from include.graphsage.model import run_data
from include.node2vec.model import run_data as run_data_node2vec

class Embedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train_embed(self, graph: Graph):
        pass

    @abstractmethod
    def get_node_embedding(self, node_id: int):
        pass

class Node2Vec(Embedding):
    def __init__(self):
        self.embedding_matrix = None

    def train_embed(self, graph):
        self.embedding_matrix = run_data_node2vec(graph)
        return self.embedding_matrix
    
    def get_node_embedding(self, node_id):
        return self.embedding_matrix[node_id]

class GraphSage(Embedding):
    def __init__(self):
        self.model = None

    def train_embed(self, graph):
        feat_data = self.compute_features(graph)
        self.model = run_data(graph.graph_data, feat_data)
        return self.model
    
    def get_node_embedding(self, node_id):
        self.model.eval()
        with torch.no_grad():
            node_t = torch.LongTensor([node_id])
            emb = self.model.embed(node_t)
            if emb.shape[0] == 128 and (len(emb.shape) == 1 or emb.shape[1] != 128):
                emb = emb.t()
        return emb.squeeze().cpu()
    
    def compute_features(self, graph):
        # Creation of features for each node
        num_nodes = graph.graph_data.num_nodes
        u_list = graph.graph_data.edge_index[0].numpy()
        v_list = graph.graph_data.edge_index[1].numpy()
        weights = graph.graph_data.edge_attr.numpy()
        feat_data = np.zeros((num_nodes, 3))

        df = pd.DataFrame({'u': u_list, 'v': v_list, 'weight': weights})
        weights_sum = df.groupby('u')['weight'].sum()
        weights_max = df.groupby('u')['weight'].max()
        num_neighbors = df.groupby('u')['v'].nunique()
        # Assigns to each node the following features: sum of weights of the corresponding edges, max weight of the corresponding edges and number of neighbors.
        for node_id in weights_sum.index:
            feat_data[node_id, 0] = weights_sum[node_id]
            feat_data[node_id, 1] = weights_max[node_id]
            feat_data[node_id, 2] = num_neighbors[node_id]
        # Z-score standardization of the features
        feat_data = (feat_data - feat_data.mean(axis=0)) / (feat_data.std(axis=0) + 1e-7)
        return feat_data
    
    def update_adjacency(self, edge_index):
        u_list = edge_index[0].numpy()
        v_list = edge_index[1].numpy()
        
        internal_adj = self.model.enc.adj_lists
        
        for u, v in zip(u_list, v_list):
            internal_adj[int(u)].add(int(v))
            internal_adj[int(v)].add(int(u))

        print("Adjacency lists updated!")

    def update_features(self, feat_data):
        with torch.no_grad():
            self.model.enc.base_model.features.weight.copy_(torch.FloatTensor(feat_data))

        print("Features updated!")