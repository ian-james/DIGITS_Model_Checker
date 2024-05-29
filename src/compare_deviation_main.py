import pandas as pd
import numpy as np
import argparse

def add_moving_avg_std(df, value_column, window_size):
    """
    Adds moving average and standard deviation columns to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.
    value_column (str): Column name for the values to calculate the moving statistics.
    window_size (int): Window size for the rolling calculations.

    Returns:
    pd.DataFrame: DataFrame with added moving average and standard deviation columns.
    """
    df[f'{value_column}_moving_avg'] = df[value_column].rolling(window=window_size).mean()
    df[f'{value_column}_moving_std'] = df[value_column].rolling(window=window_size).std()
    return df

def find_biggest_deviation(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    time_column = 'time'

    # Select columns to analyze by excluding specified columns
    columns_to_analyze = [col for col in df.columns if col not in exclude_columns and col != time_column]
    
    # Calculate the deviations for each selected column
    # Cacluating the difference between consecutive rows
    deviations = df[columns_to_analyze].diff().abs()
    
    # Find the maximum deviation for each column
    max_deviation = deviations.max()
    
    # Find the frame number at which the maximum deviation occurs for each column
    max_deviation_frame = deviations.idxmax()
    
    # Combine the results into a DataFrame
    result = pd.DataFrame({
        'column': max_deviation.index,
        'max_deviation': max_deviation,
        'frame_of_max_deviation': df.loc[max_deviation_frame, time_column].values
    })

    return result

def calculate_deviations(df, exclude_columns=None):
    if exclude_columns is None:
        exclude_columns = []

    time_column = 'time'

    # Select columns to analyze by excluding specified columns
    columns_to_analyze = [col for col in df.columns if col not in exclude_columns and col != time_column]
    
    # Calculate the deviations for each selected column
    deviations = df[columns_to_analyze].diff().abs()
    
    # Add the frame column to the deviations DataFrame
    deviations[time_column] = df[time_column]

    return deviations

def main(input_file, output_file, exclude_columns):
    # Read the input file
    try:
        df = pd.read_csv(input_file, sep='\t')
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return    

    df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')))]

    df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')))]
    
    # Find the biggest deviations
    biggest_deviation_result = find_biggest_deviation(df, exclude_columns)
    
    # Calculate deviations at each time step
    deviations_at_each_step = calculate_deviations(df, exclude_columns)
    
    # Write the results to the output file
    with pd.ExcelWriter(output_file) as writer:
        biggest_deviation_result.to_excel(writer, sheet_name='Biggest_Deviations', index=False)
        deviations_at_each_step.to_excel(writer, sheet_name='Deviations_at_Time', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Find and calculate deviations in data columns.')
    parser.add_argument("-i","--input_file", type=str, help='Input CSV file')
    parser.add_argument("-o","--output_file", type=str, help='Output Excel file')
    parser.add_argument("-e", "--exclude_columns", type=str, nargs='*', default=["timestamp","handedness"], help='Columns to exclude from analysis')
        
    args = vars(parser.parse_args())

    print("The input file is ", args['input_file'])
    print("The output file is ", args['output_file'])
    
    main(args['input_file'], args['output_file'], args['exclude_columns'])
