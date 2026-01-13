import sys
import torch
from abc import ABC, abstractmethod

sys.path.append('.')
from dataset_utils import Graph
from include.graphsage.model import run_data
from include.node2vec.model import run_data as run_data_node2vec
from include.line.train import run_data as run_data_line

from torch.nn.functional import relu, softplus

class Embedding(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def train_embed(self, graph: Graph):
        pass

    @abstractmethod
    def get_node_embedding(self, node_id: int):
        pass

class Node2Vec(Embedding):
    def __init__(self, device):
        self.embedding_matrix = None
        self.model = None

    def train_embed(self, graph):
        self.model = run_data_node2vec(graph.graph_data)
        return self.model
    
    def get_node_embedding(self, node_id):
        self.model.eval()
        node_tensor = torch.as_tensor(node_id, dtype=torch.long)
        with torch.no_grad():
            embedding = self.model.embedding(node_tensor)
        return embedding
    
class LINE(Embedding):
    def __init__(self, device):
        super().__init__()
        self.embedding_matrix = None

    def train_embed(self, graph):
        self.embedding_matrix = run_data_line(graph)
        return self.embedding_matrix
    
    def get_node_embedding(self, node_id):
        return self.embedding_matrix[node_id]

class GraphSage(Embedding):
    def __init__(self, device):
        self.model = None
        self.device = device

    def train_embed(self, graph):
        feat_data = self.compute_features(graph)
        self.model = run_data(graph.graph_data, feat_data)
        return self.model
    
    def get_node_embedding(self, node_id):
        self.model.eval()
        with torch.no_grad():
            node_t = torch.LongTensor([node_id])
            emb = self.model.embed(node_t)
            if emb.shape[0] == 128 and (len(emb.shape) == 1 or emb.shape[1] != 128):
                emb = emb.t()
        return emb.squeeze()
    
    def compute_features(self, graph):
        # Creation of features for each node
        num_nodes = graph.graph_data.num_nodes
        u_list = graph.graph_data.edge_index[0]
        v_list = graph.graph_data.edge_index[1]
        weights = graph.graph_data.edge_attr
        feat_data = torch.zeros(num_nodes, 3)

        # Assigns to each node the following features: sum of weights of the corresponding edges, max weight of the corresponding edges and number of neighbors.
        feat_data[:, 0].index_add_(0, u_list, weights)
        feat_data[:, 1].scatter_reduce_(0, u_list, weights.squeeze(), reduce='amax', include_self=False)
        ones = torch.ones_like(weights)
        feat_data[:, 2].index_add_(0, u_list, ones)

        # Z-score standardization of the features
        mean = feat_data.mean(dim=0)
        std = feat_data.std(dim=0)
        feat_data = (feat_data - mean) / (std + 1e-7)
        return feat_data
    
    def update_adjacency(self, edge_index):
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        
        internal_adj = self.model.enc.adj_lists
        
        for u, v in zip(u_list, v_list):
            internal_adj[u].add(v)
            internal_adj[v].add(u)

        print("Adjacency lists updated!")

    def update_features(self, feat_data):
        with torch.no_grad():
            self.model.enc.base_model.features.weight.copy_(torch.FloatTensor(feat_data))

        print("Features updated!")

class DVNE(torch.nn.Module, Embedding):
    def __init__(self, device, input_dim:int=3, hidden_dim:int=64, latent_dim:int=128):
        super().__init__()
        self.encoder= torch.nn.Linear(input_dim, hidden_dim)
        self.mu_layer = torch.nn.Linear(hidden_dim, latent_dim)
        self.sigma_layer = torch.nn.Linear(hidden_dim, latent_dim)

        self.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001)

        self.anchors = None
        self.positives = None
        self.negatives = None
        self.weights = None

        self.mu = None
        self.sigma = None
        self.loss = None

    @property
    def device(self):
        return next(self.parameters()).device

    # structured_negative_sampling cannot be used because the positive edges are shuffled without their attributes, so you then have to recover the right order of weights
    # Here, instead, by fixing an anchor from a positive edges and picking negative edges at random
    # you keep control of the order and apply it also to the edge_attributes
    # As long as the graph is sparse, the amount of false negatives is extremely low (The highest is ~3.4%, corresponding to the smallest dataset we use)
    def set_sampled_triplets(self, G_embed, batch_size, excluded_edges:torch.Tensor|None = None):
        positive_edges = G_embed.graph_data.edge_index
        indices = torch.randint(0, positive_edges.size(1), (batch_size,))
        self.anchors = positive_edges[0, indices]
        self.positives = positive_edges[1, indices]
        self.negatives = torch.randint(0, G_embed.graph_data.num_nodes, (batch_size,))
        self.weights = G_embed.graph_data.edge_attr[indices]
        false_negatives_count = 0
        if excluded_edges is not None:
            positive_edges = torch.cat([G_embed.graph_data.edge_index.to(self.device), excluded_edges], dim=1)
        edge_set = set(zip(positive_edges[0].tolist(), positive_edges[1].tolist()))
        for i in range(batch_size):
            anchor = self.anchors[i].item()
            negative = self.negatives[i].item()
            if (anchor, negative) in edge_set:
                false_negatives_count += 1
        print(f"False negative in sample: {false_negatives_count} (ratio: {(false_negatives_count/batch_size)})")

    def compute_features(self, graph:Graph):
        """
        Manually computes features for nodes of a graph.
        The node features computed are:
        - sum of weights
        - maximum weight
        - number of neighbors

        Parameters:
        - graph: Graph object containing the graph with its nodes

        Returns:
        - A torch.tensor object [num_edges x 3] containing the computed features.
        """
        edge_index = graph.graph_data.edge_index.to(self.device)
        edge_attr = graph.graph_data.edge_attr.to(self.device)
        num_nodes = graph.graph_data.num_nodes

        attr = graph.graph_data.edge_attr

        weights_sum = torch.zeros(num_nodes, device=self.device)
        weights_sum.scatter_reduce_(0, edge_index[0], edge_attr, reduce='sum', include_self=False)

        weights_max = torch.zeros(num_nodes, device=self.device)
        weights_max.scatter_reduce_(0, edge_index[0], edge_attr, reduce='amax', include_self=False)

        num_neighbors = torch.zeros(num_nodes, device=self.device)
        counts = torch.ones_like(edge_index[0], dtype=torch.float, device=self.device)
        num_neighbors.scatter_reduce_(0, edge_index[0], counts, reduce='sum', include_self=False)

        features = torch.stack([weights_sum, weights_max, num_neighbors], dim=1)

        return features

    def train_embed(self, graph, epochs:int=100):
        dev = next(self.parameters()).device
        node_features = self.compute_features(graph).to(dev)
        for epoch in range(epochs):
            self.train()
            self.optimizer.zero_grad()
            self.mu, self.sigma = self.forward(node_features)
            self.loss = self.contrastive_wasserstein_loss(self.mu, self.sigma, self.positives, self.negatives)
            self.loss.backward()
            self.optimizer.step()
        self.eval()
        with torch.no_grad():
            self.mu = self.mu.detach()
            self.sigma = self.sigma.detach()

    def get_node_embedding(self, node_id):
        return self.mu[node_id]

    def get_node_sigma(self, node_id):
        return self.sigma[node_id]

    def forward(self, x):
        h = relu(self.encoder(x))
        mu = self.mu_layer(h)
        sigma = softplus(self.sigma_layer(h))
        return mu, sigma

    def gauss_wasserstein_dist(self, mu_i, sigma_i, mu_j, sigma_j):
        mu_dist = torch.sum((mu_i - mu_j) ** 2, dim=1)
        sigma_dist = torch.sum((sigma_i - sigma_j) ** 2, dim=1)
        return mu_dist + sigma_dist

    def contrastive_wasserstein_loss(self, mu, sigma, pos_idx, neg_idx, margin=1.0):
        mu_i, sigma_i = mu, sigma
        mu_pos, sigma_pos = mu[pos_idx], sigma[pos_idx]
        mu_neg, sigma_neg = mu[neg_idx], sigma[neg_idx]

        dist_positives = self.gauss_wasserstein_dist(mu_i, sigma_i, mu_pos, sigma_pos)
        dist_negatives = self.gauss_wasserstein_dist(mu_i, sigma_i, mu_neg, sigma_neg)

        loss = relu(margin + dist_positives - dist_negatives).mean()
        return loss
