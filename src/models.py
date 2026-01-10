from abc import ABC, abstractmethod

from include.svm.model import SVM as SVMModel
from include.svm.model import train as train_svm
from include.svm.model import predict as predict_svm
import torch
from torch.nn.functional import dropout
from torch.nn import Linear


class Model(ABC):
    def __init__(self):
        super().__init__()
        # TODO decidere gli attributi

    @abstractmethod
    def train_model(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SVM(Model):
    def __init__(self):
        self.model = None

    def train_model(self, X, y):
        self.model = SVMModel(X.shape[1])
        X = X.float()
        self.model = train_svm(self.model, X, y)

    def predict(self, X):
        X = X.float()
        return predict_svm(self.model, X)


class MLP(torch.nn.Module, Model):
    def __init__(self, input_dim=257, hidden_channels=16, lr=0.01, weight_decay=5e-4):
        super().__init__()
        self.lin1 = Linear(input_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.out = None

    def forward(self, X):
        X = self.lin1(X)
        X = X.relu()
        X = dropout(X, p=0.5, training=self.training)
        X = self.lin2(X)
        return X

    def train_model(self, X, Y):
        self.train()
        self.optimizer.zero_grad()
        self.out = self.forward(X)
        loss = self.criterion(self.out, Y.long())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X):
        self.eval()
        with torch.no_grad():
            out = self.forward(X)
            pred = out.argmax(dim=1)
            return pred
