import pandas as pd
import numpy as np


df = pd.read_csv('dataset-2.csv')
def calculate_distance_matrix(df)->pd.DataFrame():
    """
    Calculate a distance matrix based on the dataframe, df.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Distance matrix
    """
    # Write your logic here
    # Step 1: Extract all unique toll locations
    # Changed column names to 'id_start' and 'id_end' to match the DataFrame
    locations = pd.concat([df['id_start'], df['id_end']]).unique()  
    locations.sort()  # Sort locations for easier reading
    
    # Step 2: Initialize the distance matrix with infinity, except the diagonal (0 for same locations)
    distance_matrix = pd.DataFrame(np.inf, index=locations, columns=locations)
    np.fill_diagonal(distance_matrix.values, 0)
    
    # Step 3: Fill the matrix with the given distances
    # Changed column names to 'id_start', 'id_end', and 'distance' to match the DataFrame
    for _, row in df.iterrows():
        loc_A, loc_B, distance = row['id_start'], row['id_end'], row['distance'] 
        distance_matrix.loc[loc_A, loc_B] = distance
        distance_matrix.loc[loc_B, loc_A] = distance  # Ensure symmetry
    
    # Step 4: Apply the Floyd-Warshall algorithm for cumulative shortest paths
    for k in locations:
        for i in locations:
            for j in locations:
                # Update the distance with the shorter path
                distance_matrix.loc[i, j] = min(distance_matrix.loc[i, j],
                                                distance_matrix.loc[i, k] + distance_matrix.loc[k, j])
    
    return distance_matrix



def unroll_distance_matrix(df)->pd.DataFrame():
    """
    Unroll a distance matrix to a DataFrame in the style of the initial dataset.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Unrolled DataFrame containing columns 'id_start', 'id_end', and 'distance'.
    """
    # Write your logic here
     # List to hold the unrolled data
    unrolled_data = []

    # Iterate through the matrix by row (id_start) and column (id_end)
    for id_start in distance_matrix.index:
        for id_end in distance_matrix.columns:
            # Exclude diagonal elements (distance from a point to itself)
            if id_start != id_end:
                # Append the row data
                unrolled_data.append({
                    'id_start': id_start,
                    'id_end': id_end,
                    'distance': distance_matrix.loc[id_start, id_end]
                })
    
    # Convert the list of dictionaries to a DataFrame
    unrolled_df = pd.DataFrame(unrolled_data)

    return unrolled_df


def find_ids_within_ten_percentage_threshold(df, reference_id)->pd.DataFrame():
    """
    Find all IDs whose average distance lies within 10% of the average distance of the reference ID.

    Args:
        df (pandas.DataFrame)
        reference_id (int)

    Returns:
        pandas.DataFrame: DataFrame with IDs whose average distance is within the specified percentage threshold
                          of the reference ID's average distance.
    """
    # Write your logic here
    # Filter rows with the reference ID and calculate the average distance for the reference ID
    reference_df = df[df['id_start'] == reference_id]
    ref_avg_distance = reference_df['distance'].mean()
    
    # Set 10% threshold bounds
    lower_bound = 0.9 * ref_avg_distance
    upper_bound = 1.1 * ref_avg_distance
    
    # Group by 'id_start' and calculate average distance for each group
    grouped_df = df.groupby('id_start')['distance'].mean().reset_index()

    # Filter the IDs whose average distance is within the 10% threshold
    within_threshold_df = grouped_df[
        (grouped_df['distance'] >= lower_bound) & (grouped_df['distance'] <= upper_bound)
    ]
    
    return within_threshold_df



def calculate_toll_rate(df)->pd.DataFrame():
    """
    Calculate toll rates for each vehicle type based on the unrolled DataFrame.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame
    """
    # Wrie your logic here
    # Define rate coefficients for each vehicle type
    rates = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }
    
    # Calculate toll rates for each vehicle type and add to DataFrame
    for vehicle, rate in rates.items():
        df[vehicle] = df['distance'] * rate
    
    return df



    
