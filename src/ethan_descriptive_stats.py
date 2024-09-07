# Write a main functoin to combines csv files and calculates descriptive stas

#
import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path

from stats_functions import compute_statistics

from file_utils import setupAnyCSV
from convert_mediapipe_index import get_joint_name

def get_directory(file_path):

    # Check if the file_path is a directory and already exists.
    if os.path.isdir(file_path):
        return file_path

    if not file_path:
        raise ValueError("The file path must not be empty.")

    return os.path.dirname(os.path.abspath(file_path))

def main():
    parser = argparse.ArgumentParser(description="Create a video from a single image.")
    parser.add_argument("-d","--directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_51/nh_1_md_0.5_mt_0.5_mp_0.5/csvs/angles/", help="Path to the input csv.")
    #parser.add_argument("-d","--directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_50/csv_angles/", help="Path to the input csv.")
    parser.add_argument("-o","--out_filename", type=str, default="vs_combine_output.csv", help="Path and filename for the converted video.")
    #parser.add_argument("-s","--stats_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_50/csv_angles/test_stats/", help="Specific Path for stats files (optional).")
    parser.add_argument("-s","--stats_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_51/nh_1_md_0.5_mt_0.5_mp_0.5/csvs/angles/test_stats/", help="Specific Path for stats files (optional).")
    parser.add_argument("-m","--stats", type=str, default="all", help="Statistics to compute (var, std, mean, max, min)")
    parser.add_argument("-e","--eth", action='store_true', help="Calculate Ethan's descriptive, otherwise calculate mine")

    # Setup Arguments
    args = vars(parser.parse_args())
    directory = args['directory']
    if( directory == ""):
        print("Please provide a directory path.")
        return

    if( not os.path.exists(directory) ):
        print(f"Directory not found: {directory}")
        return

    # Create a dataframe to store the statistics
    settings = ['filename', 'nh', 'md','mt','mp']

    total_df = pd.DataFrame()

    # Using pathlib get the parent directory of this directory.
    odirectory = Path(directory).parent
    sdirectory  = args['stats_directory']
    use_ethans = args['eth']

    if( sdirectory == ""):
        sdirectory = odirectory

    if( not os.path.exists(sdirectory) ):
        os.makedirs(sdirectory)

    print(f"Input directory: {directory}")
    print(f"Output directory: {odirectory}")
    print(f"Stats directory: {sdirectory}")

    files = os.listdir(directory)

    for file_name in files:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            try:

                use_headers = None if use_ethans else 0
                df = setupAnyCSV(file_path,header=use_headers)

                item = {
                        'filename': file_name,
                        'nh': 1,
                        'md': 0.5,
                        'mt': 0.5,
                        'mp': 0.5,
                        'model': "mediapipe"
                }

                # Remove the first and last columns
                if(use_ethans):
                    df.drop(df.columns[[0,-1]], axis=1, inplace=True)

                df = df.select_dtypes(include=[float, int])
                df = df.apply(pd.to_numeric, errors='coerce')

                # Compute the statistics
                stats_df = compute_statistics(df, exclude_columns=[])
                sfile = os.path.join(sdirectory,f"stats_{file_name}")

                # Change the stas_file columns names to include the joint name
                if(use_ethans):
                    stats_df.columns = [get_joint_name(c) for c in stats_df.columns]
                else:
                    # For the number of columns use the index to get the joint name
                    column_len = len(stats_df.columns)
                    cols = [get_joint_name(i) for i in range(1,column_len+1)]
                    stats_df.columns = cols               
                    
                stats_df.to_csv(sfile, header=True, sep="\t")

                stat_var = args['stats']
                # Add the statistics to the item dataframe
                if( isinstance(stat_var,str)):
                    if( stat_var == "all"):
                        stat_var = stats_df.index.values
                    else:
                        stat_var = [stat_var]

                item_df = pd.DataFrame([item])

                # for s in stat_var:
                #     for column_name, value in stats_df.loc[s].items():
                #         item_df[s+"_"+column_name] = value
                new_columns = {f"{s}_{column_name}": stats_df.loc[s, column_name] for s in stat_var for column_name in stats_df.columns}

                # Directly create a DataFrame from the new columns dictionary
                new_columns_df = pd.DataFrame(new_columns, index=[0])

                # Concatenate with item_df (along columns) if item_df already exists
                item_df = pd.concat([item_df, new_columns_df], axis=1)

                total_df = pd.concat([total_df, item_df], axis=0,ignore_index=True)

            except:
                print(f"Error parsing the filename: {file_name}")

    if( total_df is None or len(total_df) == 0):
        print("No data to save.")
    else:
        # Save the file to the output directory
        if( use_ethans):            
            out_file = os.path.join(odirectory, args['out_filename'])
        else:
            out_file = os.path.join( Path(odirectory).parent, args['out_filename'])
        total_df.to_csv(out_file, index=False, header=True, sep="\t")

    print(f"Finished writing the statistics to {args['out_filename']}")


if __name__ == "__main__":
    main()
