from abc import ABC, abstractmethod
from sklearn import svm

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
        self.model = svm.SVC(kernel="linear", C=1e4)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class MLP(Model):
    def __init__(self):
        print("ok")

    def train(self, X, y):
        pass

    def predict(self, X):
        pass
