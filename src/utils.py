import csv


def parser(filename:str) -> list[dict]:
    """
    Parses a CSV file and returns its content as a list of dictionaries.

    Parameters:
    - filename: path to the CSV file.

    Returns:
    - List of tuples associated with the CSV file.
    """

    print(f'Parsing {filename}...')

    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        rows = list()
        for row in reader:
            rows.append(row)

    return rows
