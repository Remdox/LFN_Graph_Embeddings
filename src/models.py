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
        self.model = train_svm(self.model, X, y)

    def predict(self, X):
        return predict_svm(self.model, X)


class MLP(torch.nn.Module, Model):
    def __init__(self, input_dim:int =257, hidden_channels:int =16, lr:float =0.01, weight_decay:float =5e-4):
        super().__init__()
        self.lin1 = Linear(input_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.out = None
        num_epochs = 200
        patience = 20

    def forward(self, X:torch.Tensor):
        X = self.lin1(X)
        X = X.relu()
        X = dropout(X, p=0.5, training=self.training)
        X = self.lin2(X)
        return X

    def train_model(self, X:torch.Tensor, Y:torch.Tensor):
        self.train()
        self.optimizer.zero_grad()
        self.out = self.forward(X)
        loss = self.criterion(self.out, Y.long())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X:torch.Tensor):
        self.eval()
        with torch.no_grad():
            out = self.forward(X)
            return torch.softmax(out, dim=1)[:, 1]
