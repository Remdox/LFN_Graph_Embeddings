from abc import ABC, abstractmethod

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

class MLP(Model):
    def __init__(self):
        print("ok")

    def train(self):
        pass

    def predict(self):
        pass
