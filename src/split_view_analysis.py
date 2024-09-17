# Open an dataframe in Pandas and separate by patient and view
# Files are format 001-L-1-7_mediapipe_nh_0.5_np_0.5_nt_0.5.csv
# Only focused on 001 - Patient, L - Left, 1 - View 1, 7 - 7th Pose

import os
import argparse
import pandas as pd
import numpy as np

from file_utils import setupAnyCSV

def group_by_view(df):

    df.sort_values(by='filename', inplace=True)

    # Create a column for the Patient ID, View, and Pose
    df['patient'] = df['filename'].str.split('-', expand=True)[0]    
    df['pose'] = df['filename'].str.split('-', expand=True)[3]

    # Create a new dataframe for each patient and pose
    df_groupby = df.groupby(['patient','pose'])
    df['row_id'] = df.groupby(['patient','pose']).cumcount() + 1    

    # For each entry combine all the views into a single row
    pivoted_df = df.pivot(index='patient', columns='row_id')

    # Flatten the MultiIndex columns
    pivoted_df.columns = [f'{col[0]}_{col[1]}' for col in pivoted_df.columns]

    # Reset index to get a clean DataFrame
    pivoted_df.reset_index(inplace=True)
    return pivoted_df   

def main():
    parser = argparse.ArgumentParser(description="Create a video from a single image.")    
    parser.add_argument("-f","--input_file", type=str, default="datasets/circumduction_test/output/all_cirumduction.csv", help="Path to the input csv.") 
    parser.add_argument("-o","--out_file", type=str, default="test_split.csv", help="Path and filename for the converted video.")
    
    # Setup Arguments
    args = vars(parser.parse_args())    
    
    input_file = args['input_file']

    if( not os.path.exists(input_file) ):
        print(f"File not found: {input_file}")
        return
    
    try:
        df = setupAnyCSV(input_file)
        df = group_by_view(df)
        df.to_csv(args['out_file'], index=False, header=True, sep="\t")
    except Exception as e:
        print(f"Error Processing File: {e}")
        return
        

if __name__ == '__main__':
    main()