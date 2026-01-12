import torch
from torch_geometric.data import Data
from torch_geometric.nn.models import Node2Vec

def run_data(graph: Data) -> Node2Vec:
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

    loader = model.loader(batch_size=128, shuffle=True)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)

    for epoch in range(200):
        model.train()
        for pos_rw, neg_rw in loader:
            optimizer.zero_grad()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
    
    return model
