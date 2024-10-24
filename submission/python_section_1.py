from typing import Dict, List

import pandas as pd


def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    # Your code goes here.
    lst1 = []
    l = len(lst)
    for i in range(0, l, n):
        group = lst1[i:i + n]
        reversed_group = []
        for j in range(len(group) - 1, -1, -1):
            reversed_group.append(group[j])
        lst1.extend(reversed_group)
    return lst1


def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    # Your code here
    result = {}
    for string in strings:
        length = len(string)
        if length not in result:
            result[length] = []
        result[length].append(string)

    # Sort the dictionary by keys (lengths) in ascending order
    res_dict = dict(sorted(result.items()))
    return res_dict


def flatten_dict(nested_dict: Dict, sep: str = '.') -> Dict:
    """
    Flattens a nested dictionary into a single-level dictionary with dot notation for keys.
    
    :param nested_dict: The dictionary object to flatten
    :param sep: The separator to use between parent and child keys (defaults to '.')
    :return: A flattened dictionary
    """
    # Your code here
    items = []
    for k, v in input_dict.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, elem in enumerate(v):
                if isinstance(elem, dict):
                  items.extend(flatten_dict(elem, new_key + f"[{i}]", sep=sep).items())
                else:
                    items.append((new_key + f"[{i}]", elem))
        else:
            items.append((new_key, v))
    return dict(items)



import itertools
def unique_permutations(nums: List[int]) -> List[List[int]]:
    """
    Generate all unique permutations of a list that may contain duplicates.
    
    :param nums: List of integers (may contain duplicates)
    :return: List of unique permutations
    """
    # Your code here
    
    lst = [list(p) for p in itertools.permutations(nums)]
    unique = set(tuple(item) for item in lst)

    # Convert the tuples back to lists
    return [list(item) for item in unique]


import re
def find_all_dates(text: str) -> List[str]:
    """
    This function takes a string as input and returns a list of valid dates
    in 'dd-mm-yyyy', 'mm/dd/yyyy', or 'yyyy.mm.dd' format found in the string.
    
    Parameters:
    text (str): A string containing the dates in various formats.

    Returns:
    List[str]: A list of valid dates in the formats specified.
    """
    date_pattern = r"\d{2}-\d{2}-\d{4}|\d{2}/\d{2}/\d{4}|\d{4}\.\d{2}\.\d{2}"
    dates = re.findall(date_pattern, text)
    return dates




import polyline

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface given their latitude and longitude.
    The result is returned in meters.
    """
    # Radius of the Earth in meters
    R = 6371000  
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # distance in meters


def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
    """
    Converts a polyline string into a DataFrame with latitude, longitude, and distance between consecutive points.
    
    Args:
        polyline_str (str): The encoded polyline string.

    Returns:
        pd.DataFrame: A DataFrame containing latitude, longitude, and distance in meters.
    """

      # Decode the polyline string into a list of (latitude, longitude) tuples
    coordinates: List[Tuple[float, float]] = polyline.decode(polyline_str)
    
    # Create a DataFrame with latitude and longitude columns
    df = pd.DataFrame(coordinates, columns=['latitude', 'longitude'])
    
    # Initialize the distance column with 0 for the first point
    distances = [0]
    
    # Calculate the distance between consecutive points
    for i in range(1, len(coordinates)):
        lat1, lon1 = coordinates[i - 1]
        lat2, lon2 = coordinates[i]
        distance = haversine(lat1, lon1, lat2, lon2)
        distances.append(distance)
    
    # Add the distance column to the DataFrame
    df['distance'] = distances
    
    return df
    


import copy
def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
    """
    Rotate the given matrix by 90 degrees clockwise, then multiply each element 
    by the sum of its original row and column index before rotation.
    
    Args:
    - matrix (List[List[int]]): 2D list representing the matrix to be transformed.
    
    Returns:
    - List[List[int]]: A new 2D list representing the transformed matrix.
    """
    # Your code here
    n = len(matrix)
    
    # Rotate the matrix 90 degrees clockwise
    rotated_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            rotated_matrix[j][n - 1 - i] = matrix[i][j]
    
    # Calculate sums for each element in the rotated matrix
    transformed_matrix = copy.deepcopy(rotated_matrix)  # Create a copy to avoid modifying the original
    for i in range(n):
        for j in range(n):
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = 0
            for k in range(n):
                if k != i:
                    col_sum += rotated_matrix[k][j]
            transformed_matrix[i][j] = row_sum + col_sum
            
    return transformed_matrix



import pandas as pd
df = pd.read_csv('dataset-1.csv')
def time_check(df) -> pd.Series:
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    # Write your logic here
    df['startTime'] = pd.to_datetime(df['startTime'], format='%H:%M:%S').dt.time
    df['endTime'] = pd.to_datetime(df['endTime'], format='%H:%M:%S').dt.time
    grouped = df.groupby(['id', 'id_2'])

    # Initialize an empty dictionary to store results
    results_dict = {} 

    for name, group in grouped:
        if (group['startTime'].min() <= pd.to_datetime('00:00:00', format='%H:%M:%S').time() and
            group['endTime'].max() >= pd.to_datetime('23:59:59', format='%H:%M:%S').time() and
            set(group['startDay']).union(set(group['endDay'])) == set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])):
            results_dict[name] = False  # Store results in the dictionary
        else:
            results_dict[name] = True   # Store results in the dictionary

    # Convert the dictionary to a pandas Series 
    results = pd.Series(results_dict)

    return results

    
    




