from abc import ABC, abstractmethod

from include.svm.model import SVM as SVMModel
from include.svm.model import train as train_svm
from include.svm.model import predict as predict_svm


class Model(ABC):
    def __init__(self):
        super().__init__()
        # TODO decidere gli attributi

    @abstractmethod
    def train(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SVM(Model):
    def __init__(self):
        self.model = None

    def train(self, X, y):
        self.model = SVMModel(X.shape[1])
        self.model = train_svm(self.model, X, y)

    def predict(self, X):
        return predict_svm(self.model, X)


class MLP(Model):
    def __init__(self):
        print("ok")

    def train(self, X, y):
        pass

    def predict(self, X):
        pass
