from abc import ABC, abstractmethod

class Embedding(ABC):
    def __init__(self, graph, directed: bool, weighted: bool):
        self.graph = graph
        self.directed = directed
        self.weighted = weighted

    @abstractmethod
    def train_embed(self):
        pass

    @abstractmethod
    def get_node_embedding(self, node_id: int):
        pass


class Model(ABC):
    def __init__(self):
        super().__init__()
        # TODO decidere gli attributi

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def predict(self):
        pass
