import csv
import time


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

def get_current_time_ms() -> int:
    """
    Returns the current time in milliseconds.

    Returns:
    - Current time in milliseconds.
    """

    return int(time.time() * 1000)

def elapsed_time_ms(start_time: int, end_time: int) -> int:
    """
    Calculates the elapsed time in milliseconds between two timestamps.

    Parameters:
    - start_time: start timestamp in milliseconds.
    - end_time: end timestamp in milliseconds.

    Returns:
    - Elapsed time in milliseconds.
    """
    
    return end_time - start_time