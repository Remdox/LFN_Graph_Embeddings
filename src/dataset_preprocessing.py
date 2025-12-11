import csv

# Parses a CSV file and returns its content as a list of dictionaries
#
# Parameter: filename - path to the CSV file
def parser(filename:str) -> list[dict]:

    print(f'Parsing {filename}...')

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        rows = list()
        for row in reader:
            rows.append(row)

    return rows

# Creates a dataset CSV file from the given edges
# The CSV file contains columns: u, v, weight
# The processed datasets are saved in ./datasets/processed_datasets/
#
# Assumption: Each edge is represented as a list: [u, v, weight]
#
# Parameters: name - name of the dataset, 
#             edges - list of edges
def create_dataset_csv(name:str, edges:list[dict]) -> None:

    print(f'Creating processed dataset for {name}...')

    filename = f'./datasets/processed_datasets/{name}.csv'

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['u', 'v', 'weight'])
        for edge in edges:
            writer.writerow(edge)

# Maps non-integer node IDs to integer IDs
#
# Parameters: rows - list of dictionaries representing edges,
#             u_key - key for from-node ID,
#             v_key - key for to-node ID
def node_id_mapping(rows:list[dict], u_key:str, v_key:str) -> list[dict]:

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


# Preprocesses datasets based on the information in datasets_info.csv
# If a dataset contains non-integer node IDs, they are mapped to integers
# If the graph is undirected, edges are duplicated in the opposite direction
# If no weight is provided, a default weight of 1.0 is assigned to each edge
# The processed datasets are saved in ./datasets/processed_datasets/
#
# Assumption: datasets_info.csv contains columns: name - name of the file containing the dataset, 
#                                                 u - from-node ID, 
#                                                 v - to-node ID, 
#                                                 weight - eventual numeric weight of the edge, 
#                                                 directed - if the graph is directed or not
#             The datasets to process are stored in ./datasets/original_datasets/
def dataset_preprocessing() -> None:

    # Load dataset information
    datasets = parser('./datasets/datasets_info.csv')

    for dataset in datasets:

        name = dataset['name']
        print(f'Preprocessing dataset: {name}...')

        # Load original dataset
        dataset_path = f'./datasets/original_datasets/{name}.csv'
        rows = parser(dataset_path)
        
        # If node IDs are not integers, map them to integers
        try:
            int(rows[0][dataset['u']])
        except ValueError:
            rows = node_id_mapping(rows, dataset['u'], dataset['v'])

        edges = list()

        for row in rows:

            # Create edge from u to v
            edges.append([int(row[dataset['u']]), int(row[dataset['v']])])

            # Assign weight
            if dataset['weight'] != '':
                edges[-1].append(float(row[dataset['weight']]))
            else:
                edges[-1].append(1.0)

            # If the graph is undirected, create edge from v to u
            if dataset['directed'] == 'False':
                edges.append([int(row[dataset['v']]), int(row[dataset['u']]), edges[-1][2]])

        create_dataset_csv(dataset['name'], edges)


# Loads all processed datasets into a dictionary
#
# Assumption: Datasets are already processed and stored in ./datasets/processed_datasets/
#
# Returns: A dictionary where keys are dataset names and values are lists of edges
def get_datasets() -> dict:

    # Load dataset information
    datasets_csv = parser('./datasets/datasets_info.csv')

    datasets = dict()

    for dataset in datasets_csv:
        # Load processed dataset
        name = dataset['name']
        print(f'Loading dataset: {name}...')
        filename = f'./datasets/processed_datasets/{name}.csv'
        datasets[dataset['name']] = parser(filename)

    return datasets

#dataset_preprocessing()
#datasets = get_datasets()