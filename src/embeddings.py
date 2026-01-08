import sys
from abc import ABC, abstractmethod
sys.path.append('.')
from include.graphsage.model import run_data

class Embedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train_embed(self, graph: Graph):
        pass

    @abstractmethod
    def get_node_embedding(self, node_id: int):
        pass


class GraphSage(Embedding):
    def __init__(self, graph, directed, weighted):
        self.graph = graph
        self.directed = directed
        self.weighted = weighted
        self.embeddings = None

    def train_embed(self):
        self.embeddings = run_data(self.graph)
        return self.embeddings
    
    def get_node_embedding(self, node_id):
        return self.embeddings[node_id]
