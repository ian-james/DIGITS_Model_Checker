# Write a main functoin to combines csv files and calculates descriptive stas

#
import os
import argparse
import pandas as pd
import numpy as np

from stats_functions import compute_statistics


def main():
    parser = argparse.ArgumentParser(description="Create a video from a single image.")    
    parser.add_argument("-d","--directory", type=str, default="./output/nyu_tests/csvs/", help="Path to the input csv.") 
    parser.add_argument("-o","--out_filename", type=str, default="combine_output.csv", help="Path and filename for the converted video.")
     
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
    exclude_columns = ['filename','time','timestamp','handedness']
    total_df = pd.DataFrame()

    files = os.listdir(directory)    
    for file_name in files:
        file_path = os.path.join(directory, file_name)
        if os.path.isfile(file_path) and file_name.endswith('.csv'):
            try:
                # Parse the filename
                components = file_name.split('_')
                item = {
                    'filename': components[0] + "_" + components[1] + "_" + components[2],
                    'nh': components[4],
                    'md': components[6],
                    'mt': components[8],
                    'mp': components[10],
                    'model': components[11].split('.')[0]
                }

                df = pd.read_csv(file_path, sep='\t')
                item_df = pd.DataFrame([item])

                # Compute the statistics
                stats_df = compute_statistics(df, exclude_columns=exclude_columns)
                stats_df.to_csv(f"stats_{file_name}_.csv", header=True, sep="\t")

                print(stats_df.loc['var'].T)

                for column_name, value in stats_df.loc['var'].items():
                    item_df[column_name] = value
                print(item_df.columns)

                item_df.to_csv(f"item_stats_{file_name}_.csv", header=True, sep="\t")
        
                total_df = pd.concat([total_df, item_df], axis=0,ignore_index=True)   
                print(total_df.columns)

            except:
                print(f"Error parsing the filename: {file_name}")
                continue

    # Save the statistics to a new CSV file
    if( total_df is None or len(total_df) == 0):
        print("No data to save.")
    else:
        total_df.to_csv(args['out_filename'], index=False, header=True, sep="\t")

    print(f"Finished writing the statistics to {args['out_filename']}")


if __name__ == "__main__":
    main()