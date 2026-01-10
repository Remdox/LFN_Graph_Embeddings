import torch
import torch.nn as nn
import torch.optim as optim


class SVM(nn.Module):

    def __init__(self, input_size):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)


def hinge_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Computes the hinge loss.

    Parameters:
    - y_true: true labels (1 or 0)
    - y_pred: predicted scores

    Returns:
    - Hinge loss value
    """

    return torch.mean(torch.clamp(1 - y_true * y_pred, min=0))


def train(model: SVM, X: torch.Tensor, y: torch.Tensor) -> SVM:
    """
    Trains the SVM model using hinge loss and L2 regularization.
    
    Parameters:
    - model: SVM model to be trained
    - X: input features as a torch.Tensor
    - y: true labels as a torch.Tensor (1 or 0)

    Returns:
    - Trained SVM model
    """

    optimizer = optim.SGD(model.parameters(), lr=0.01)

    epochs = 200
    C = 0.01

    for epoch in range(epochs):
        
        model.train()
        optimizer.zero_grad()
        y_pred = model(X)
        loss = hinge_loss(y, y_pred) + C * torch.norm(model.linear.weight) ** 2
        loss.backward()
        optimizer.step()

    return model


def predict(model: SVM, X: torch.Tensor) -> torch.Tensor:
    """
    Predicts labels for input features.
    
    Parameters:
    - model: Trained SVM model
    - X: input features as a torch.Tensor

    Returns:
    - Predicted labels as a torch.Tensor (1 or 0)
    """
    model.eval()
    
    with torch.no_grad():
        y_pred = model(X)
        y_pred = torch.where(y_pred >= 0, torch.tensor(1.0), torch.tensor(0.0))
    
    return y_pred