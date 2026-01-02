import csv
import os
from utils import parser

def create_dataset_csv(name:str, edges:list[dict]) -> None:
    """
    Creates a dataset CSV file from the given edges.
    The CSV file contains columns: u, v, weight.
    The processed datasets are saved in ./datasets/processed_datasets/

    Assumptions: 
    - Each edge is represented as a list: [u, v, weight]

    Parameters:
    - name: name of the dataset,
    - edges: list of edges.
    """

    print(f'Creating processed dataset for {name}...')

    filename = f'./datasets/processed_datasets/{name}.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['u', 'v', 'weight'])
        for edge in edges:
            writer.writerow(edge)


def node_id_mapping(rows:list[dict], u_key:str, v_key:str) -> list[dict]:
    """
    Maps non-integer node IDs to integer IDs.
    
    Parameters:
    - rows: list of dictionaries representing edges,
    - u_key: key for from-node ID, 
    - v_key: key for to-node ID.

    Returns:
    - Updated list of dictionaries with integer node IDs.
    """

    node_ids = set()

    for row in rows:
        node_ids.add(row[u_key])
        node_ids.add(row[v_key])

    node_ids = list(node_ids)
    id_mapping = {old_id: str(node_ids.index(old_id)) for old_id in node_ids}

    for row in rows:
        row[u_key] = id_mapping[row[u_key]]
        row[v_key] = id_mapping[row[v_key]]

    return rows


def dataset_preprocessing() -> None:
    """
    Preprocesses datasets based on the information in datasets_info.csv
    The node IDs are mapped to integers starting from 0.
    If the graph is undirected, edges are duplicated in the opposite direction.
    If no weight is provided, a default weight of 1.0 is assigned to each edge.
    The processed datasets are saved in ./datasets/processed_datasets/

    Assumptions:
    - datasets_info.csv contains columns: name - name of the file containing the dataset, 
                                          u - from-node ID, 
                                          v - to-node ID, 
                                          weight - eventual numeric weight of the edge, 
                                          directed - if the graph is directed or not
    - The datasets to process are stored in ./datasets/original_datasets/
    - If the dataset starts with node 0, node ID mapping is not applied.
    """

    # Load dataset information
    datasets = parser('./datasets/datasets_info.csv')

    # Create processed_datasets directory if it doesn't exist
    if not os.path.exists("./datasets/processed_datasets/"):
        os.mkdir("./datasets/processed_datasets/")

    for dataset in datasets:

        name = dataset['name']
        print(f'Preprocessing dataset: {name}...')

        # Load original dataset
        dataset_path = f'./datasets/original_datasets/{name}.csv'
        rows = parser(dataset_path)

        # Map node IDs to integers starting from 0
        try:
            if int(rows[0][dataset['u']]) > 0:
                rows = node_id_mapping(rows, dataset['u'], dataset['v'])
        except ValueError:
            rows = node_id_mapping(rows, dataset['u'], dataset['v'])

        edges = list()

        for row in rows:

            # Create edge from u to v
            edges.append([row[dataset['u']], row[dataset['v']]])

            # Assign weight
            if dataset['weight'] != '':
                edges[-1].append(float(row[dataset['weight']]))
            else:
                edges[-1].append(1.0)

            # If the graph is undirected (without duplicated edges), create edge from v to u
            if dataset['directed'] == 'False':
                edges.append([row[dataset['v']], row[dataset['u']], edges[-1][2]])

        # Create processed dataset CSV
        create_dataset_csv(dataset['name'], edges)


if __name__ == "__main__":
    dataset_preprocessing()