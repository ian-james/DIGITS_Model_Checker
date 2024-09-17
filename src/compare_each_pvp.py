# This file takes two all angles files that have the patient number, hand, view, and pose 
# It compares each matching row based on the patient number, hand, view, and pose
# It then writes the comparison to a file

import os
import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind

from file_utils import setupAnyCSV, clean_spaces
from convert_mediapipe_index import get_joint_names, convert_real_finger_name_to_mediapipe_name_within_string

from pprint import pprint
import re

from calculation_helpers import natural_sort_key

from hand_file_builder import *


def setup_columns_names(df):
    try:
        df['filename'] = df['filename'].apply(clean_spaces)
    
        # Check if the first element of the df column has the correct file format.
        if( not check_if_file_is_in_lab_format(df['filename'].iloc[0]) ):
            df['filename'] = df['filename'].apply(change_filename_from_lab_numbers)

        # Remove Stats names from any of the columns names
        df.columns = [remove_stat_prefix_from_filename(col) for col in df.columns]
    

        # Change Finger Names to be the same
        df.columns = [convert_real_finger_name_to_mediapipe_name_within_string(col) for col in df.columns]
    
        # Sort the dataframes by the filename
        df = df.sort_values(by='filename', key=lambda col: col.map(natural_sort_key))
    except Exception as e:
        print(f"Error: {e}")

    return df
  

def find_patient_hand_view_pose(filename):


    digits = re.search(r'\d+', filename)
   
    # Get the location of the first digit
    if(digits):
        patient_number = digits[0]

        hand = get_keywords_from_filename(filename, get_hand_names())
        view = get_keywords_from_filename(filename, get_view_names())
        pose = get_keywords_from_filename(filename, get_pose_names())

        return build_patient_filename(str(int(patient_number)), hand, view, pose)
    
    return filename

def main():

    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")
    parser.add_argument("-f","--file", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/df_cleaned.csv", help="Path to the input CSV file.")    
    parser.add_argument("-m","--mfile", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/mdf_cleaned.csv", help="Path to the input CSV file.")

    # Example Groups
    # Setup Arguments
    args = vars(parser.parse_args())
    file_input = args['file']
    mfile = args['mfile']

    ignore_columns = ['filename', 'nh', 'md', 'mt', 'mp', 'model', 'basename', 'group']


    # Check the files and directories are created
    if( not os.path.exists(file_input) ):
        print(f"File not found: {file_input}")
        return

    if( not os.path.exists(mfile) ):
        print(f"File not found: {mfile}")
        return
    
    base_folder = os.path.dirname(file_input)

    # Make a diffs folder
    os.makedirs(os.path.join(base_folder, "diffs"), exist_ok=True)
   
    try:
            # Setup their comparison csv files, nominally, theirs is the first file and mine is the second.
        df = setupAnyCSV(file_input,0)
        mdf = setupAnyCSV(mfile,0)

        df = setup_columns_names(df)
        mdf = setup_columns_names(mdf)

        df['basename'] = df['filename'].apply(lambda x: find_patient_hand_view_pose(x))
        mdf['basename'] = mdf['filename'].apply(lambda x: find_patient_hand_view_pose(x))
        df['group'] = "one"
        mdf['group'] = "two"

        columns_to_process = df.columns.difference(ignore_columns)
        df[columns_to_process] = df[columns_to_process].apply(pd.to_numeric, errors='coerce')

        # Group each by the basename to compare the difference, absolute difference between the t-test
        # This will allow us to see the difference between the two files
        # We will then write the difference to a file

        # Group by the basename
        df_grouped = df.groupby('basename')
        mdf_grouped = mdf.groupby('basename')

        # Get the keys for each group
        df_keys = df_grouped.groups.keys()
        mdf_keys = mdf_grouped.groups.keys()

        # Get the intersection of the keys
        shared_keys = set(df_keys).intersection(set(mdf_keys))

        # Get the difference between the two files        
        # This will be the absolute difference between the two files
        # Write the code now.

        diff_df = pd.DataFrame()
        
        for key in shared_keys:
            # Get the group for each key
            df_group = df_grouped.get_group(key)
            print(df_group)
            mdf_group = mdf_grouped.get_group(key)
            print(mdf_group)

            # Print the Index MCP values
            print(f"Index MCP: {df_group['Index MCP'].values}")
            print(f"Index MCP: {mdf_group['Index MCP'].values}")

            df_group.to_csv(os.path.join(base_folder, "diffs", f"test_{key}_df.csv"), index=False)
            mdf_group.to_csv(os.path.join(base_folder,"diffs",f"test2_{key}_df.csv"), index=False)

            c = pd.concat([df_group,mdf_group], ignore_index=True)
            c.to_csv(os.path.join(base_folder, "diffs", f"concat_{key}_df.csv"), index=False)
        
            diff = (df_group[columns_to_process].copy().reset_index() - mdf_group[columns_to_process].copy().reset_index()).abs()
        
            if( not 'basename' in diff.columns ):
                diff.insert(0, 'basename', key)
            diff_df = pd.concat([diff_df, diff], ignore_index=True)
            
            # Save the difference to a file
            diff.to_csv(os.path.join(base_folder, "diffs", f"{key}_diff.csv"), index=False)

        # Save the difference to a file
        diff_df.to_csv(os.path.join(base_folder, f"all_diff.csv"), index=False)

    except Exception as e:
        print(f"Error: {e}")
        return
    
if __name__ == '__main__':
    main()