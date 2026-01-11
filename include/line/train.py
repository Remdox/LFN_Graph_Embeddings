import torch
import numpy as np
from .line import Line
from .utils import AliasSampler

def run_data(graph, total_dim=128, epochs=10, batch_size=4096, initial_lr=0.025):
    data = graph.graph_data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_nodes = data.num_nodes
    dim_per_order = total_dim // 2
    
    edge_index = data.edge_index.t()
    weights = data.edge_attr.numpy().flatten()
    
    # Positive edge sampler
    edge_sampler = AliasSampler(weights)
    # Using np.bincount to calculate node degrees
    node_degrees = np.bincount(data.edge_index[0].numpy(), weights=weights, minlength=num_nodes)
    # Negative sampler
    node_sampler = AliasSampler(np.power(node_degrees, 0.75) + 1e-10)

    all_embs = []
    num_batches = edge_index.size(0) // batch_size
    total_steps = num_batches * epochs

    # Trains order 1 and 2
    for order in [1, 2]:
        print(f"Training order {order}...")
        model = Line(num_nodes, dim_per_order, order).to(device)
        # Using SparseAdam instead of SGD
        optimizer = torch.optim.SparseAdam(model.parameters(), lr=initial_lr)
        
        step_idx = 0
        for epoch in range(epochs):
            # Samples edges and negatives for the epoch
            s_edges = edge_sampler.sample(num_batches * batch_size)
            s_negs = node_sampler.sample(num_batches * batch_size * 5)
            
            for b in range(num_batches):
                # Added linear learning rate decay
                lr = initial_lr * (1.0 - (step_idx / total_steps))
                lr = max(lr, initial_lr * 0.001)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                
                # Defines the range for the current batch
                start, end = b * batch_size, (b + 1) * batch_size
                idx_batch = s_edges[start:end]
                # Extraction of u and v nodes
                u = edge_index[idx_batch, 0]
                v = edge_index[idx_batch, 1]
                # Extraction of negatives
                negs = s_negs[start*5:end*5].view(batch_size, 5)

                optimizer.zero_grad()
                loss = model(u, v, negs)
                loss.backward()
                optimizer.step()
                
                step_idx += 1
        
        emb = model.get_embeddings()
        if emb.dim() == 1:
            emb = emb.unsqueeze(1)
        all_embs.append(emb)

    # Returns the embeddings matrix with embeddings of both orders
    final_tensor = torch.cat(all_embs, dim=1)
    
    print(f"Training completed!")
    return final_tensor