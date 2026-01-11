import torch
import torch.nn as nn
import torch.nn.functional as F

class Line(nn.Module):
    def __init__(self, num_nodes, dim, order=2):
        super().__init__()
        self.order = order
        # Added sparse=True to allow updating only the relevant rows of the embedding matrix, increasing velocity
        self.u_embeddings = nn.Embedding(num_nodes, dim, sparse=True)
        self.v_embeddings = nn.Embedding(num_nodes, dim, sparse=True)
        
        self.u_embeddings.weight.data.uniform_(-.5/dim, .5/dim)
        if order == 2:
            self.v_embeddings.weight.data.uniform_(-.5/dim, .5/dim)
        else:
            self.v_embeddings.weight.data.zero_()

    def forward(self, u_idx, v_idx, neg_idx):
        u = self.u_embeddings(u_idx)
        v = self.v_embeddings(v_idx) if self.order == 2 else self.u_embeddings(v_idx)
        vn = self.v_embeddings(neg_idx) if self.order == 2 else self.u_embeddings(neg_idx)
        
        pos_score = torch.sum(u * v, dim=1)
        pos_loss = F.logsigmoid(pos_score)

        # Using torch.bmm to increase velocity. It computes the dot products between u nodes and all vn in a single call
        neg_score = torch.bmm(vn, u.unsqueeze(2)).squeeze(2)
        neg_loss = F.logsigmoid(-neg_score).sum(dim=1)
        
        return -torch.mean(pos_loss + neg_loss)

    def get_embeddings(self):
        # Added to produce the normalized embeddings (with L2 normalization)
        self.eval()
        with torch.no_grad():
            w = self.u_embeddings.weight.data
            return F.normalize(w, p=2, dim=1)