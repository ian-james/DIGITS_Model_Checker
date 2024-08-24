
import os
import pandas as pd
import argparse
import numpy as np

def split_combined_dataframes(df, category_column_name, timestamp_column_name):
    """Reads a combined dataframe and splits it into multiple dataframes based on the category column."""
    # count the values in the 'category' column and split the dataframe based on the result
    value_counts = df[category_column_name].value_counts()
    #dfs = [df[df[category_column_name] == value_count] for value_count in value_counts]
    dfs = df.sort_values(timestamp_column_name).groupby(category_column_name)
    #.sort_values(timestamp_column_name)
    # sort each dataframe by timestamp
    #sorted_dfs = [df.sort_values(timestamp_column_name) for df in dfs]
    #return sorted_dfs    
    return dfs

def is_excel(file_path):
    """Checks if a file is an Excel file."""
    file_extension = file_path.split('.')[-1]
    return file_extension in ['xls', 'xlsx']

def is_csv(file_path):
    """Checks if a file is a CSV file."""
    file_extension = file_path.split('.')[-1]
    return file_extension == 'csv'

def read_file_to_dataframe(file_path):
    """Reads a file to a pandas DataFrame based on file extension."""
    file_extension = file_path.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file_path,sep="\t")
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "File format not supported. Only .csv, .xls, and .xlsx files are supported.")
    return df


def compare_dataframes(df1, df2, threshold):
    """
    Compares each row of df1 and df2 to determine if significant positional differences occur
    (above the given threshold). Returns a new dataframe with a boolean result for each row.
    """
    # Ensure the input dataframes have the same shape
    if df1.shape != df2.shape:
        raise ValueError('Input dataframes must have the same shape')

    # Compute the absolute difference between each corresponding element in df1 and df2
    diff_df = np.abs(df1 - df2)

    # Create a boolean dataframe with the same shape as diff_df
    bool_df = pd.DataFrame(False, index=diff_df.index, columns=diff_df.columns)

    # Iterate over each row in diff_df and set the corresponding value in bool_df to True if
    # any element in the row exceeds the threshold
    for i, row in diff_df.iterrows():
        if (row > threshold).any():
            bool_df.loc[i, :] = True

    return bool_df


def compare_files(file_path_1, file_path_2,time='fps_count', exclude_columns=[]):
    """Compares two Excel files based on their 'fps_count' column."""
    df1 = read_file_to_dataframe(file_path_1)
    df2 = read_file_to_dataframe(file_path_2)

    df1 = df1.drop(exclude_columns, axis=1)
    df2 = df2.drop(exclude_columns, axis=1)

    # Check if the 'fps_count' column exists in both files
    if time not in df1.columns or time not in df2.columns:
        raise ValueError(
            "The 'fps_count' column is missing in one or both files.")

    # Compare the two files based on their 'fps_count' column
    missing_rows = []
    different_rows = []
    for index, row1 in df1.iterrows():
        fps_count = row1[time]
        matching_rows = df2.loc[df2[time] == fps_count]
        if len(matching_rows) == 0:
            missing_rows.append(fps_count)
        else:
            row2 = matching_rows.iloc[0]            
            if not row1.equals(row2):
                different_rows.append(time)

    return missing_rows, different_rows


def track_positional_changes(df, col_name, threshold, use_normalized_diff=False):
    """
    Compares two sequential DataFrame rows to determine if the normalized value has exceeded a specific threshold.
    
    Args:
    df (pd.DataFrame): The DataFrame to be processed.
    col_name (str): The name of the column to be compared.
    threshold (float): The threshold value for detecting positional changes.
    
    Returns:
    pd.DataFrame: The processed DataFrame with an additional column indicating whether the positional change exceeds the threshold.
    """
    # Calculate the difference between sequential values in the specified column
    diff = df[col_name].diff()

    # Normalize the difference using the maximum absolute value of the column
    if( use_normalized_diff):
        max_val = df[col_name].abs().max()
        norm_diff = diff / max_val        
        # Create a new column to track positional changes
        df['pos_change'] = norm_diff.abs() > threshold
    else:
        df['pos_change'] = diff.abs() > threshold

    return df

def track_all_positional_changes(df, threshold):
    """
    Compares two sequential DataFrame rows to determine if the value difference has exceeded a specific threshold for all columns.
    
    Args:
    df (pd.DataFrame): The DataFrame to be processed.
    threshold (float): The threshold value for detecting positional changes.
    
    Returns:
    pd.DataFrame: The processed DataFrame with additional columns indicating whether positional changes exceed the threshold for each column.
    """
    # Calculate the absolute difference between sequential values for each column
    abs_diff = df.diff().abs()

    # Create new columns to track positional changes for each column
    pos_changes = abs_diff > threshold
    pos_changes.columns = [col_name + '_pos_change' for col_name in df.columns]
    df = pd.concat([df, pos_changes], axis=1)

    return df

def detect_large_positional_changes(df, threshold):
    """
    Compares two sequential DataFrame rows to determine if the value difference has exceeded a specific threshold for any column.
    
    Args:
    df (pd.DataFrame): The DataFrame to be processed.
    threshold (float): The threshold value for detecting positional changes.
    
    Returns:
    pd.DataFrame: The processed DataFrame with additional columns indicating whether any positional changes exceed the threshold.
    """
    # Create an empty list to store the results
    results = []

    # Get the column names that contain numerical values
    numeric_cols = df.select_dtypes(include='number').columns.tolist()

    # Loop through each row and compare the values to the previous row
    for i in range(0, len(df)):
        prev_row = df.iloc[i - 1]
        curr_row = df.iloc[i]

        # Check for large positional changes in each numeric column
        for col in numeric_cols:
            if abs(prev_row[col] - curr_row[col]) > threshold:
                results.append(True)
                break
        else:
            results.append(False)

    # Add the results as a new column to the DataFrame
    df['large_positional_changes'] = results

    return df

def get_position_names(part="shoulder"):
    return [part + "_left_x", part + "_left_y", part + "_right_x", part + "_right_y"]

if __name__ == '__main__':
   
    ap = argparse.ArgumentParser()    
    ap.add_argument("-f","--file1", type=str, default="datasets/copy_test/output/csvs/r1-001-L-1-1_mediapipe_nh_1_md_0.8_mt_0.8_mp_0.8.csv", help='path to first input file')
    ap.add_argument("-s","--file2", type=str, default="datasets/copy_test/output/csvs/001-L-1-1_mediapipe_nh_1_md_0.8_mt_0.8_mp_0.8.csv", help='path to second input file')
    ap.add_argument("-t","--threshold", type=float, default=0.2, help='threshold value for positional changes')
        
    #ap.add_argument('column_names', nargs='+', type=str, default="", help='Column names to track for positional changes')    
   
    # Parse the arguments
    args = vars(ap.parse_args())    
    
    file1 = args['file1']
    file2 = args['file2']
    threshold = args['threshold']
    
    file1 = os.path.abspath(file1)        

    if(file1 != "" and file2 != ""):    
        missing_rows, different_rows = compare_files( file1, file2,"time","timestamp")
    else:
        df = read_file_to_dataframe(file1)
        dataframes = split_combined_dataframes(df,"time","timestamp")        
        
        for g in dataframes.groups:
            
            dfg = dataframes.get_group(g)
            #dfg = track_positional_changes(dfg,check_position, threshold, True)
            dfg = track_all_positional_changes(dfg, threshold)
            dfh = detect_large_positional_changes(dfg, threshold)
        
            df.to_csv("output.csv", index=False)
        
    
