from abc import ABC, abstractmethod
import torch
from torch.nn.functional import dropout
from torch.nn import Linear
import xgboost as xgb

from include.svm.model import SVM as SVMModel
from include.svm.model import train as train_svm
from include.svm.model import predict as predict_svm

class Model(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def train_model(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass


class SVM(Model):
    def __init__(self, device):
        self.model = None
        self.device = device

    def train_model(self, X, y):
        X, y = X.to(self.device), y.to(self.device)
        self.model = SVMModel(X.shape[1], self.device)
        self.model = train_svm(self.model, X, y)

    def predict(self, X):
        return predict_svm(self.model, X)

class RandomForest(Model):
    def __init__(self, device):
        self.model = None
        self.device = device

    def train_model(self, X, y):
        self.model = xgb.XGBRFClassifier(n_estimators=100, max_depth=10, tree_method='hist', device=str(self.device), random_state=104)
        self.model.fit(X.detach(), y.detach())

    def predict(self, X):
        pred = self.model.predict(X.detach())
        return torch.as_tensor(pred, dtype=torch.float32, device=self.device)


class MLP(torch.nn.Module, Model):
    def __init__(self, device, input_dim:int =257, hidden_channels:int =16, lr:float =0.01, weight_decay:float =5e-4, num_epochs:int=200, patience:int=20):
        super().__init__()
        self.lin1 = Linear(input_dim, hidden_channels)
        self.lin2 = Linear(hidden_channels, 2)

        self.to(device)

        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, X:torch.Tensor):
        X = self.lin1(X)
        X = X.relu()
        X = dropout(X, p=0.5, training=self.training)
        X = self.lin2(X)
        return X

    def train_model(self, X:torch.Tensor, Y:torch.Tensor):
        X, Y = X.to(self.device), Y.to(self.device)
        self.train()
        self.optimizer.zero_grad()
        out = self.forward(X)
        loss = self.criterion(out, Y.long())
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, X:torch.Tensor):
        X = X.to(self.device)
        self.eval()
        with torch.no_grad():
            out = self.forward(X)
            return torch.softmax(out, dim=1)[:, 1]
