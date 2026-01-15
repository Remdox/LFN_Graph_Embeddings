import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec

def run_data(graph: Data, patience:int =20) -> Node2Vec:
    """
    Trains the GraphSage model and produces node embeddings.

    Parameters:
    - graph: a PyTorch Geometric Data object.

    Returns:
    - A trained Node2Vec model.
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Node2Vec(
                        edge_index=graph.edge_index,
                        embedding_dim=128,
                        walk_length=20,
                        context_size=10,
                        walks_per_node=10,
                        num_negative_samples=1,
                        p=1.0,
                        q=1.0,
                        sparse=True,
                    ).to(device)

    loader = model.loader(batch_size=1024, shuffle=True, num_workers=2)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
    best_loss = float('inf')
    epochs_wout_improvement = 0

    for epoch in range(200):
        model.train()
        total_loss = 0
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            epochs_wout_improvement = 0
        else:
            epochs_wout_improvement += 1

        if epochs_wout_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
    print("ss")
    
    return model
