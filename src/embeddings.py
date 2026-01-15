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
        self.device = device

    def train_embed(self, graph):
        self.model = run_data_node2vec(graph.graph_data)
        return self.model

    def get_node_embedding(self, node_id):
        return self.embedding_matrix[node_id]

    def compute_all_embeddings_in_batch(self, graph:Graph, batch_size=512):
        """
        Computes all embeddings of the graph and saves them in a tensor inside the object.

        Description of this implementation:
        - A tensor slice as long as batch_size is used to compute the embedding of a batch, which is
        then saved in RAM. The data is not moved to the VRAM until all embedding batches have been computed.
        This represents an effective solution which can reach good performance when using the gpu,
        while keeping a low VRAM usage.

        Parameters:
        - graph: Graph object containing the graph with its nodes
        - batch_size: the size of the batch. Higher means more gpu computation speed, but also more memory used for allocating the embeddings of each node inside the batch.
        """
        num_nodes = graph.graph_data.num_nodes
        all_node_indices = torch.arange(graph.graph_data.num_nodes, device=self.device)
        self.model.eval()
        batched_embs = []
        with torch.no_grad():
            for i in range(0, num_nodes, batch_size):
                batch_indices = all_node_indices[i : i + batch_size]
                batch_emb = self.model.embedding(batch_indices)
                batched_embs.append(batch_emb.cpu())

        self.embedding_matrix = torch.cat(batched_embs, dim=0).to(self.device)

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
        self.node_embeddings = None

    def train_embed(self, graph):
        feat_data = self.compute_features(graph).to(self.device)
        self.model = run_data(graph.graph_data, feat_data, self.device)
        return self.model

    def get_node_embedding(self, node_id):
        return self.node_embeddings[node_id]

    def compute_all_embeddings_in_batch(self, graph:Graph, batch_size=512):
        """
        Computes all embeddings of the graph and saves them in a tensor inside the object.

        Description of this implementation:
        - A tensor slice as long as batch_size is used to compute the embedding of a batch, which is
        then saved in RAM. The data is not moved to the VRAM until all embedding batches have been computed.
        This represents an effective solution which can reach good performance when using the gpu,
        while keeping a low VRAM usage.

        Parameters:
        - graph: Graph object containing the graph with its nodes
        - batch_size: the size of the batch. Higher means more gpu computation speed, but also more memory used for allocating the embeddings of each node inside the batch.
        """
        all_node_indices = torch.arange(graph.graph_data.num_nodes, device=graph.graph_data.edge_index.device)
        self.model.eval()
        batched_embs = []
        with torch.no_grad():
            for i in range(0, graph.graph_data.num_nodes, batch_size):
                batch_indices = all_node_indices[i : i + batch_size]
                batch_emb = self.model.embed(batch_indices).cpu()
                batched_embs.append(batch_emb)
        self.node_embeddings = torch.cat(batched_embs, dim=0).to(self.device)

    def compute_features(self, graph):
        # Creation of features for each node
        num_nodes = graph.graph_data.num_nodes
        u_list = graph.graph_data.edge_index[0]
        v_list = graph.graph_data.edge_index[1]
        weights = graph.graph_data.edge_attr
        feat_data = torch.zeros(num_nodes, 3, device=self.device)

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
            self.model.enc.base_model.features.weight.copy_(feat_data.to(self.device))

        print("Features updated!")

class DVNE(torch.nn.Module, Embedding):
    def __init__(self, device, input_dim:int=4, hidden_dim:int=64, latent_dim:int=128):
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
        self.epoch = 0

        self.eps = 1e-6
        self.beta = 0.01

    @property
    def device(self):
        return next(self.parameters()).device

    # structured_negative_sampling cannot be used because the positive edges are shuffled without their attributes, so you then have to recover the right order of weights
    # Here, instead, by fixing an anchor from a positive edges and picking negative edges at random
    # you keep control of the order and apply it also to the edge_attributes
    # As long as the graph is sparse, the amount of false negatives is extremely low
    def set_sampled_triplets(self, G_embed, batch_size, excluded_edges:torch.Tensor|None = None):
        """
        Samples triplets (anchors, positive nodes, negative nodes) from the graphs, as they are needed for the DVNE training.

        Assumption: the input graph is sparse, so that the amount of false negatives is low

        Implementation Description: fix the anchors, pick as positives the other end of their positive edges
        and pick as negatives the other end of their positive edges, but shuffled. This is an approximation
        that is way faster and simpler than an exact negative edge picking method and also better than just
        picking random nodes as negatives, because:
            - the memory for negatives is already allocated
            - the amount of possible false negatives is way lower

        """
        positive_edges = G_embed.graph_data.edge_index
        indices = torch.randint(0, positive_edges.size(1), (batch_size,))
        self.anchors = positive_edges[0, indices]
        self.positives = positive_edges[1, indices]
        self.weights = G_embed.graph_data.edge_attr[indices]
        self.negatives = torch.randint(0, G_embed.graph_data.num_nodes, (batch_size,), device=self.device)

        shuffle_idx = torch.randperm(batch_size)
        self.negatives = self.positives[shuffle_idx]

        self.count_false_negatives(G_embed, batch_size, excluded_edges)

    def count_false_negatives(self, G_embed, batch_size, excluded_edges:torch.Tensor|None = None):
        false_negatives_count = 0
        positive_edges = G_embed.graph_data.edge_index
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

        damp = 0.85
        pagerank = torch.ones(num_nodes, device=self.device) / num_nodes
        deg_out = num_neighbors + 1e-6 # evita divisioni per zero

        for _ in range(10):
            incoming = torch.zeros_like(pagerank)
            incoming.scatter_add_(0, edge_index[1], pagerank[edge_index[0]] / deg_out[edge_index[0]])
            pagerank = (1 - damp) / num_nodes + damp * incoming

        features = torch.stack([weights_sum, weights_max, num_neighbors, pagerank], dim=1)

        mean = features.mean(dim=0)
        std = features.std(dim=0) + 1e-6
        features = (features - mean) / std

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
            self.epoch = epoch
            if epoch % 10 == 0:
                print(f"Epoch {epoch} | Loss: {self.loss.item():.4f} | Sigma Mean: {self.sigma.mean().item():.4f}")
        self.eval()
        with torch.no_grad():
            self.mu, self.sigma = self.forward(node_features)
            self.mu = self.mu.detach()
            self.sigma = self.sigma.detach()

    def get_node_embedding(self, node_id):
        return self.mu[node_id]

    def get_node_sigma(self, node_id):
        return self.sigma[node_id]

    def forward(self, node_features):
        h = relu(self.encoder(node_features))
        mu = self.mu_layer(h)
        sigma = softplus(self.sigma_layer(h)) + self.eps
        return mu, sigma

    def gauss_wasserstein_dist(self, mu_a, sigma_a, mu_b, sigma_b):
        mu_dist    = torch.sum( (mu_a - mu_b)** 2, dim=1)
        sigma_dist = torch.sum( (sigma_a - sigma_b)** 2, dim=1)
        return (mu_dist+sigma_dist)

    def contrastive_wasserstein_loss(self, mu, sigma, pos_idx, neg_idx, margin=1.0):
        """
        Computes contrastive loss based on Wasserstein distances.

        The idea is to make the distance anchor<->positive neighbor to be smaller than
        the distance anchor<->negative neighbor

        parameters:
        - mu: mean vectors for all nodes
        - sigma: variance vectors for all nodes
        - mu_j: mean of the second distribution
        - sigma_j: standard deviation of the second distribution

        Returns:
        - a torch.Tensor representing the average loss for the current batch
        """
        mu_anchors, sigma_anchors = mu[self.anchors], sigma[self.anchors]
        mu_pos, sigma_pos = mu[pos_idx], sigma[pos_idx]
        mu_neg, sigma_neg = mu[neg_idx], sigma[neg_idx]

        dist_positives = self.gauss_wasserstein_dist(mu_anchors, sigma_anchors, mu_pos, sigma_pos)
        dist_negatives = self.gauss_wasserstein_dist(mu_anchors, sigma_anchors, mu_neg, sigma_neg)

        # if self.epoch % 10 == 0:
        #     print(f"  > Dist Pos (Mean): {dist_positives.mean().item():.4f}")
        #     print(f"  > Dist Neg (Mean): {dist_negatives.mean().item():.4f}")
        #     print(f"  > Active Constraints: {(dist_positives + margin > dist_negatives).float().mean().item()*100:.1f}%")
        # loss = relu(margin+dist_positives-dist_negatives).mean()
        # return loss

        log_dist_pos = torch.log1p(dist_positives)
        log_dist_neg = torch.log1p(dist_negatives)

        if self.epoch % 10 == 0:
            print(f"  > Dist Pos (Mean): {dist_positives.mean().item():.4f}")
            print(f"  > Dist Neg (Mean): {dist_negatives.mean().item():.4f}")
            print(f"  > Active Constraints: {(log_dist_pos + margin > log_dist_neg).float().mean().item()*100:.1f}%")
        # Adjust margin for log scale (e.g., margin=1.0 or 2.0)
        contrastive_loss = torch.relu(margin + log_dist_pos - log_dist_neg).mean()
        kl_loss = -0.5 * torch.sum(1 + torch.log(sigma.pow(2) + self.eps) - mu.pow(2) - sigma.pow(2), dim=1).mean()

        return contrastive_loss + (self.beta * kl_loss)
