
#Steps
# Step1 ( Group of view files to average out)
# Take the file and group by filename ( But you need to account for slight filename differences)
# Take the average of that group and store it in a new dataframe
# Save that file of the averages of each file group

# Step2 ( Views of the hand )
# Go through the file names and group specific views that contain the names like LT_Palmer, RT_Palmer, LT_Rad, LT_UT, LT_UT, RT_UT, LT_MT, RT_MT, LT_MP, RT_MP
# Group by Palmer, Radial Oblique, Radial Side, Ulnar Oblique, Ulnar Side
# Save each each view as as separate files

# Step3 ( Compare the files  for finger and joint)

# Step
# TODO Later ()
# Go through the filenames and group by the action the user is taking EXT, FIST, IM (  more poses for our data )
# Compare them at pose level and save them as separate files

# TODO Goniometery files don't have the same column name order or column names 
# They follow mostly the same format but Index,Long, Ring, Little, Thumb is the order and the pose(extension,flexion)
# In Handle all angle files, we need to account for the slight differences in the filenames and convert to our standard names

#Step 1
import os
import pandas as pd
import numpy as np
import argparse
import re
from pathlib import Path
from scipy import stats
from scipy.stats import ttest_ind

from file_utils import setupAnyCSV, clean_spaces
from convert_mediapipe_index import get_joint_names

from pprint import pprint

from hand_file_builder import *

# stats = ['max', 'min', 'mean', 'median', 'std', 'var', sem]
# LT, RT
# Palmer, Rad_Obl, Rad_Side, Uln_Obl, Uln_Side,
# Fist, Ext, IM
# get_joint_names()
# LT_UT, LT_UT, RT_UT, LT_MT, RT_MT, LT_MP, RT_MP

def process_stats(df_rows, stats_fun, ignore_columns,out_directory):
    # the first filename from the row
    row_name = df_rows['filename'].iloc[0]

    # Strip the extension from the filename
    row_name = Path(row_name).stem

    # If the file ends with a number, then remove the number.
    # It indicates a replication from Sasha's data.
    row_name = re.sub(r'_\d+$', '', row_name)

    # Compute the stats for each column except the ignore columns
    rows = df_rows.drop(columns=ignore_columns, errors='ignore')

    # convert the columns to float
    rows.apply(pd.to_numeric, errors='coerce')

    # Compute the statistics for each column
    stats_df = rows.agg(stats_fun)

    # Add the filename as the first column in the table
    stats_df.insert(0, 'filename', row_name)                

    sfile = f"{clean_spaces(row_name)}_stats.csv"
    stats_df.to_csv( os.path.join(out_directory,sfile) , index=False, header=True, sep=',')
    return stats_df

def process_individual_files(stats_df, row_name, column_names, out_directory):
    for col_name in column_names:     
        df_cols = stats_df.filter(like=col_name, axis=1)
        if( len(df_cols) <= 0):
            print(f"Column name not found: {col_name}")
            continue

        # Save the file as a separate file
        sfile = f"{clean_spaces(row_name)}_{clean_spaces(col_name)}.csv"
        df_cols.to_csv(  os.path.join(out_directory,sfile) , index=False, header=True, sep=',')


def get_longest_shared_filename(filename1, filename2):
   
    all_keywords = get_all_keywords()
    patient_id = get_patient_id()

    # TODO IF NOT WORKING`
  

def main():

    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")
    parser.add_argument("-f","--file", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/formatted_goniometry_measures.csv", help="Path to the input CSV file.")    
    #parser.add_argument("-o","--out_filename", type=str, default="vs_combine_output.csv", help="Path and filename for the converted video.")
    parser.add_argument("-d","--directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test", help='Output directory to save the files')
    parser.add_argument('-i', '--input', type=str, default="filename", help='Default column name to group by')
    parser.add_argument("-m","--mfile", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/all_angles_combined.csv", help="Path to the input CSV file.")
    parser.add_argument("-b", "--both_name", type=str, default="filename", help="Column name to group by in both files")

    # A group of strings can be passed as a list
    parser.add_argument("-c","--column_name", nargs='+', type=str, default="", help='Column names to group together')
    parser.add_argument("-g", '--group_names', nargs='+', type=str, default="", help='Filenames or column names to group by')
    parser.add_argument("-s", "--stats", type=str, default="mean", help="Statistics to compute (var, std, mean, max, min)")

    # Add an argument to select the final stats to select
    parser.add_argument("-k", "--keep", type=str, default="max", help="Compute all the statistics")  

    # Example Groups
    # Setup Arguments
    args = vars(parser.parse_args())
    file_input = args['file']
    mfile = args['mfile']
    
    out_directory = args['directory']
    header_input = args['input']

    # Setup directory names for consistency
    DIR_VIEW_POSE = "view_pose"
    DIR_VIEW_POSE_ALL = "all"
    DIR_VIEW_POSE_COMPARE = "compare"
    DIR_THIER = "theirs"
    DIR_MINE = "mine"
    DIR_COMBINED = "combined"
    
    
    stats_fun = [args['stats']]
    #stats_fun = ['max', 'min', 'mean', 'median', 'std', sem]

    # Ignore the columns that are not numeric
    ignore_columns = ['filename', 'nh', 'md', 'mt', 'mp', 'model']

    # Check the files and directories are created
    if( not os.path.exists(file_input) ):
        print(f"File not found: {file_input}")
        return
    
    if( not os.path.exists(mfile) ):
        print(f"File not found: {mfile}")
        return

    # Make the output directory if it does not exist
    if( out_directory != "" and not os.path.exists(out_directory)):
        os.makedirs(out_directory)

    os.makedirs(os.path.join(out_directory, DIR_VIEW_POSE_COMPARE),exist_ok=True)
    os.makedirs(os.path.join(out_directory, DIR_COMBINED),exist_ok=True)

    for nd in [DIR_THIER, DIR_MINE]:
        os.makedirs(os.path.join(out_directory, nd),exist_ok=True)
        os.makedirs(os.path.join(out_directory, nd, DIR_VIEW_POSE),exist_ok=True)
        os.makedirs(os.path.join(out_directory, nd, DIR_VIEW_POSE_ALL),exist_ok=True)        

    # Try and load the CSV file
    try:
        # Setup their comparison csv files, nominally, theirs is the first file and mine is the second.
        df = setupAnyCSV(file_input,0)
        mdf = setupAnyCSV(mfile,0)

        # if the args['column_name'] is a file then load the names from the file
        if( args['column_name'] == ""):
            column_names = get_joint_names().values()
        elif( os.path.exists(args['column_name']) ):
            with open(args['column_name'], 'r') as f:
                column_names = f.readlines()
                column_names = [x.strip() for x in column_names]
        else:
            column_names = args['column_name']

        # if args['group_names'] is a file then load the names from the file
        if( args['group_names'] == ""):
            group_names =  build_nvp_group_names()
        elif( os.path.exists(args['group_names']) ):
            with open(args['group_names'], 'r') as f:
                group_names = f.readlines()
                group_names = [x.strip() for x in group_names]
        else:
            group_names = args['group_names']

        
        # Clean up the spaces in the filename columns
        df['filename'] = df['filename'].apply(clean_spaces)
        mdf['filename'] = mdf['filename'].apply(clean_spaces)

        # Check if the first element of the df column has the correct file format.
        if( not check_if_file_is_in_lab_format(df['filename'].iloc[0]) ):
            df['filename'] = df['filename'].apply(change_filename_from_lab_numbers)

        if( not check_if_file_is_in_lab_format(mdf['filename'].iloc[0]) ):
            mdf['filename'] = mdf['filename'].apply(change_filename_from_lab_numbers)

        # Sort dataframes by filename
        df = df.sort_values(by=['filename'])
        mdf = mdf.sort_values(by=['filename'])

        # Dataframe to hold all combined.
        all_combined_df = pd.DataFrame()

        # For each column name, group by that column name
        for row_name in group_names:

            # Get all the rows that contain that name
            df_rows = df[ df[header_input].str.contains(row_name, case=False, na=False)]
            mdf_rows = mdf[ mdf[header_input].str.contains(row_name, case=False, na=False)]

            pprint(df_rows['filename'])

            print(">>>>>>>>>>>>>>>>>>>>>>>>")

            pprint(df_rows['filename'])

            # If the column name is not found, then skip
            not_in_df = len(df_rows) <= 0
            not_in_mdf = len(mdf_rows) <= 0
            if( not_in_df or not_in_mdf):
                if( not_in_df and not_in_mdf):
                    print(f"Row not found in either file: {row_name}")
                elif( not_in_df):
                    print(f"Row not found in DF: {row_name}")
                else:
                    print(f"Row not found in MDF: {row_name}")
                continue

            # If the stats is not empty, then compute the stats
            if( args['stats'] != ""):                
                stats_df = process_stats(df_rows, stats_fun, ignore_columns, os.path.join(out_directory, DIR_THIER, DIR_VIEW_POSE))
                mstats_df = process_stats(mdf_rows, stats_fun, ignore_columns, os.path.join(out_directory, DIR_MINE, DIR_VIEW_POSE))

                # Select the columns that we want to save independentely
                # This could be the particular fingers or joints
                process_individual_files(stats_df, row_name, column_names, os.path.join(out_directory,DIR_THIER,DIR_VIEW_POSE_ALL))
                process_individual_files(mstats_df, row_name, column_names, os.path.join(out_directory,DIR_MINE,DIR_VIEW_POSE_ALL))

                # Concat the two dataframe and save them in the output directory
                combined_stat_df = pd.concat([stats_df, mstats_df], axis=0)
                combined_stat_df.to_csv( os.path.join(out_directory,  f"{row_name}_concated_stats.csv"), index=False, header=True, sep=',')

                # Subtract all columns that have matching column names, excluding the filename or ignore columns
                # remove the filename and ignore columns
                stats_df = stats_df.drop(columns=ignore_columns, errors='ignore')
                mstats_df = mstats_df.drop(columns=ignore_columns, errors='ignore')              

                # Subtract the two dataframes
                results = (stats_df - mstats_df).abs()

                # Save the comparison to a new file
                if( args['keep'] == "all"):
                    results_file = os.path.join(out_directory, DIR_VIEW_POSE_COMPARE,f"{clean_spaces(row_name)}_all_compare.csv")
                    results.to_csv( results_file , index=False, header=True, sep=',')
                else:
                    # Check which columns start with the keep value and save them
                    keep_columns = results.filter(like=args['keep'], axis=1)
                    keep_file = os.path.join(out_directory, DIR_VIEW_POSE_COMPARE,f"{clean_spaces(row_name)}_{args['keep']}_compare.csv")
                    keep_columns.to_csv(keep_file, index=False, header=True, sep=',')
                
                # add the filename to the start of the dataframe                
                # stats_df.insert(0, 'filename', "one")
                # mstats_df.insert(0, 'filename', "two")

                # Combine the two dataframes so that it creates two rows for each file
                combined_df = pd.concat([stats_df, mstats_df], axis=0, ignore_index=True)
                
                # Save the combined file, to the combined directory with the name of the row
                combine_file = os.path.join(out_directory, DIR_COMBINED, f"{clean_spaces(row_name)}_combined.csv")
                combined_df.to_csv( combine_file, index=False, header=True, sep=',')


                

    except FileNotFoundError:
        print(f"File not found: {file_input}")
        return

if __name__ == "__main__":
    main()

