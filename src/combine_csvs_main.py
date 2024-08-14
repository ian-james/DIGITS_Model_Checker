# Write a main functoin to combines csv files and calculates descriptive stas

#
import os
import argparse
import pandas as pd
import numpy as np

from pathlib import Path

from stats_functions import compute_statistics

from file_utils import setupAnyCSV

def get_directory(file_path):
    
    # Check if the file_path is a directory and already exists.
    if os.path.isdir(file_path):
        return file_path
    
    if not file_path:
        raise ValueError("The file path must not be empty.")
    
    return os.path.dirname(os.path.abspath(file_path))

def main():
    parser = argparse.ArgumentParser(description="Create a video from a single image.")    
    parser.add_argument("-d","--directory", type=str, default="analysis/original/nh_1_md_0.8_mt_0.8_mp_0.8/roms/csvs", help="Path to the input csv.") 
    parser.add_argument("-o","--out_filename", type=str, default="vs_combine_output.csv", help="Path and filename for the converted video.")
    parser.add_argument("-s","--stats_directory", type=str, default="analysis/original/nh_1_md_0.8_mt_0.8_mp_0.8/roms/csvs/tests", help="Specific Path for stats files (optional).") 
    parser.add_argument("-m","--stats", type=str, default="all", help="Statistics to compute (var, std, mean, max, min)")
     
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

    # Using pathlib get the parent directory of this directory.
    odirectory = Path(directory).parent
    sdirectory  = args['stats_directory']
   
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
                # Parse the filename
                components = file_name.split('_')

                item = None
                try:
                    item = {
                        'filename': components[0] + "_" + components[1] + "_" + components[2],
                        'nh': components[3],
                        'md': components[5],
                        'mt': components[7],
                        'mp': os.path.splitext(components[9])[0],
                        'model': components[1]
                    }
                except Exception as e:
                    print(f"Error parsing the filename: {file_name}")
                    item = file_name
                    continue

                df = setupAnyCSV(file_path)

                df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')),errors='ignore')]

                df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')),errors='ignore')]

                item_df = pd.DataFrame([item])

                # Compute the statistics
                stats_df = compute_statistics(df, exclude_columns=exclude_columns)
                sfile = os.path.join(sdirectory,f"stats_{file_name}")                
                stats_df.to_csv(sfile, header=True, sep="\t")
                
                stat_var = args['stats']   
                # Add the statistics to the item dataframe
                if( isinstance(stat_var,str)):
                    if( stat_var == "all"):
                        stat_var = stats_df.index.values
                    else:
                        stat_var = [stat_var]

                for s in stat_var:
                    for column_name, value in stats_df.loc[s].items():
                        item_df[s+"_"+column_name] = value    

                # Combine the statistics with the item dataframe
                

                #item_df.to_csv(f"item_stats_{file_name}_.csv", header=True, sep="\t")
        
                total_df = pd.concat([total_df, item_df], axis=0,ignore_index=True)                   

            except:
                print(f"Error parsing the filename: {file_name}")
                continue

    # Save the statistics to a new CSV file
    if( total_df is None or len(total_df) == 0):
        print("No data to save.")
    else:
        # Save the file to the output directory
        out_file = os.path.join(odirectory, args['out_filename'])
        total_df.to_csv(out_file, index=False, header=True, sep="\t")

    print(f"Finished writing the statistics to {args['out_filename']}")


if __name__ == "__main__":
    main()
