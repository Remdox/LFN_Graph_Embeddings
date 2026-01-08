import sys
import torch
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
        self.model = run_data(graph)
        return self.model
    
    def get_node_embedding(self, node_id):
        self.model.eval()
        with torch.no_grad():
            node_t = torch.LongTensor([node_id])
            emb = self.model.embed(node_t)
            if emb.shape[0] == 128 and (len(emb.shape) == 1 or emb.shape[1] != 128):
                emb = emb.t()
        return emb.cpu().numpy().flatten()
