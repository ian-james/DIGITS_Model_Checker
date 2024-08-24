# Write a main that takes a single csv file and split it into multiple csv files based on a specific column.

import os
import argparse
import pandas as pd

from file_utils import setupAnyCSV, clean_spaces


def main():

    # Setup Arguments
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple files based on a column value.")
    parser.add_argument("-f", "--input_file", type=str, default="datasets/Ethans_Evaluation/our_videos/others/r2_landmarkData.csv", help="Path to the input CSV file.")    
    parser.add_argument("-o", "--output_directory", type=str, default="datasets/Ethans_Evaluation/our_videos/others/", help="Directory to save the split files.")

    args = vars(parser.parse_args())

    input_file = args['input_file']    
    output_directory = args['output_directory']

    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    # Check if the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        #Read the CSV file
        df = setupAnyCSV(input_file,None)
        df = df.dropna(axis=1, how='all')


        # Strip whitespace and check if the last column contains only empty strings or whitespace
        # There appears to be an extra column with just spaces (fix for that)
        if df.iloc[:, -1].str.strip().eq('').all():
            df = df.iloc[:, :-1]  # Drop the last column

       
        # Use the filenames to split the data
        df_values = df.iloc[:,-1]

        unique_values = df_values.unique()

        # Count the total rows of df and the rows of each csv file to ensure the same number of rows
        total_rows = df.shape[0]
        if(total_rows == 0):
            print("Error: No rows found in the CSV file.")
            return

        # Separate the data based on the unique values
        for value in unique_values:
            # Filter the data based on the unique value
            filtered_data = df[df_values == value]

            #Check if filtered is actually unique at the last column
            last_column = filtered_data.iloc[:,-1]
            if last_column.nunique() > 1:
                print(f"Error: The last column is not unique for {value}")
                continue

            # Write the filtered data to a new CSV file
            # Remove the extension from the value
            value = value.split(".")[0]
            output_file = f"{value}.csv"

            # Trim output_file
            output_file = clean_spaces(output_file)
            output_path = os.path.join(output_directory, output_file)
            filtered_data.to_csv(output_path, index=False, header=False, sep="\t")


            total_rows -= filtered_data.shape[0]

        # Print a message to indicate that the process is complete

        if(total_rows == 0):
            print("Data split successfully")
        else:
            print(f"Error: {total_rows} rows were not processed, missing rows.")

    except Exception as e:
        print(f"Error Processing File: {e}")
        return


if __name__ == "__main__":
    main()