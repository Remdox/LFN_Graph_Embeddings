import torch
from torch_geometric.data import Data
from utils import parser
from dataclasses import dataclass

@dataclass(slots=True)
class Graph:
    graph_data: Data
    is_directed: bool
    is_weighted: bool

def dataset_as_graph(dataset: list[dict]) -> Data:
    """
    Converts a dataset from a list of edges to a graph.

    Parameters:
    - dataset: list of edges represented as dictionaries with keys 'u', 'v', and 'weight',
    - directed: boolean indicating if the graph is directed.

    Returns:
    - A PyTorch Geometric Data object representing the graph.
    """

    dataset_graph = Data()
    edge_index = [[], []]
    edge_attributes = list()

    for edge in dataset:
        edge_index[0].append(int(edge['u']))
        edge_index[1].append(int(edge['v']))
        edge_attributes.append(float(edge['weight']))

    dataset_graph.edge_index = torch.tensor(edge_index, dtype=torch.long)
    dataset_graph.edge_attr = torch.tensor(edge_attributes, dtype=torch.float)

    # if num_nodes is not fixed, PyG tries to compute the number of nodes using the size of column x,
    # but since there are no node features then x is not defined, and PyTorch has to fall back to the maximum edge_index it can find from the given graph, causing a warning.
    # Since the number of nodes is inferred by reading the edges of the dataset, it's not actually a  problem and we fix the number of nodes immediately.
    dataset_graph.num_nodes = int(dataset_graph.edge_index.max()) + 1

    return dataset_graph


def get_datasets() -> dict[str, tuple[Data, bool, bool]]:
    """
    Loads all processed datasets into a dictionary.

    Assumptions:
    - Datasets are already processed and stored in ./datasets/processed_datasets/
    
    Returns: 
    - A dictionary where keys are dataset names and values are tuples (graph, directed, weighted).
    """

    # Load dataset information
    datasets_csv = parser('./datasets/datasets_info.csv')

    datasets = dict()

    # Load processed dataset
    for dataset in datasets_csv:
        name = dataset['name']
        print(f'Loading dataset: {name}...')
        filename = f'./datasets/processed_datasets/{name}.csv'
        data = Graph(dataset_as_graph(parser(filename)), dataset['directed'] == 'True', dataset['weight'] != '')
        datasets[dataset['name']] = data

    return datasets


if __name__ == "__main__":
    datasets = get_datasets()
    for name, graph in datasets.items():
        print(f'Dataset: {name}, Directed: {graph.is_directed}, Weighted: {graph.is_weighted}, Num nodes: {graph.graph_data.num_nodes}, Num edges: {int(graph.graph_data.num_edges / (2 if not graph.is_directed else 1))}')
