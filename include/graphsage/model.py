import torch
import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable

import pandas as pd
import sys
import numpy as np
import random
from collections import defaultdict

from .encoders import Encoder
from .aggregators import MeanAggregator

"""
Simple supervised GraphSAGE model
"""

class SupervisedGraphSage(nn.Module):

    def __init__(self, num_classes, enc):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.xent = nn.CrossEntropyLoss()

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())
    
    def embed(self, nodes):
        return self.enc(nodes)

def load_data(graph, feat_data):
    """
    Converts the graph into the structures required for GraphSage.

    Parameter:
    - graph: a PyTorch Geometric Data object.

    Returns:
    - feat_data: normalized node features (sum and max of edge weights),
    - labels: binary labels based on the sum of weights of each node,
    - adj_lists: adjacency list for each pair of nodes.
    """
    num_nodes = graph.num_nodes
    u_list = graph.edge_index[0].numpy()
    v_list = graph.edge_index[1].numpy()
    
    # Creation of labels: label is 1 if the sum of all the weights of a specific node is greater than the median of the weights and is 0 otherwise
    labels = np.zeros((num_nodes, 1), dtype=np.int64)
    threshold = np.median(feat_data[:, 0])
    for i in range(num_nodes):
        labels[i] = 1 if feat_data[i, 0] > threshold else 0
    
    # Creation of the adjacency lists
    adj_lists = defaultdict(set)
    for u, v in zip(u_list, v_list):
        adj_lists[int(u)].add(int(v))
        adj_lists[int(v)].add(int(u))
        
    return labels, adj_lists

def run_data(graph, feat_data):
    """
    Loads the graph data, trains the GraphSage model and produces node embeddings.

    Parameter:
    - graph: a PyTorch Geometric Data object.

    Returns:
    - final_embeddings: Numpy array of shape (num_nodes, 128).
    """
    np.random.seed(1)
    random.seed(1)
    torch.manual_seed(1) # TODO capire se lasciare o no
    labels, adj_lists = load_data(graph, feat_data)
    num_nodes = feat_data.shape[0]
    num_feat = feat_data.shape[1]
    features = nn.Embedding(num_nodes, num_feat)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    agg1 = MeanAggregator(features, cuda=True)
    enc1 = Encoder(features, num_feat, 128, adj_lists, agg1, gcn=True, cuda=False)
    agg2 = MeanAggregator(lambda nodes : enc1(nodes).t(), cuda=False)
    enc2 = Encoder(lambda nodes : enc1(nodes), enc1.embed_dim, 128, adj_lists, agg2, base_model=enc1, gcn=True, cuda=False)
    enc1.num_samples = 10
    enc2.num_samples = 10

    graphsage = SupervisedGraphSage(2, enc2)

    optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, graphsage.parameters()), lr=0.01)
    
    train_nodes = list(adj_lists.keys())

    for batch in range(200):
        batch_nodes = random.sample(train_nodes, min(len(train_nodes), 1024))
        batch_labels = torch.LongTensor(labels[np.array(batch_nodes)])

        optimizer.zero_grad()
        loss = graphsage.loss(batch_nodes, batch_labels)
        loss.backward()
        optimizer.step()
        
        print(f"Batch {batch}, Loss: {loss.item()}")

    # Production of trained model
    print("Training completed!")
    graphsage.eval()
    return graphsage
