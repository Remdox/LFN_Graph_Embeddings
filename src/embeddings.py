import sys
import torch
from abc import ABC, abstractmethod
sys.path.append('.')
from include.graphsage.model import run_data
from dataset_utils import *

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
