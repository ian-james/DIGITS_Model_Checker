
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

#Step 1
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from file_utils import setupAnyCSV, clean_spaces
from convert_mediapipe_index import get_joint_names

# stats = ['max', 'min', 'mean', 'median', 'std', 'var', sem]
# LT, RT
# Palmer, Rad_Obl, Rad_Side, Uln_Obl, Uln_Side,
# Fist, Ext, IM
# get_joint_names()
# LT_UT, LT_UT, RT_UT, LT_MT, RT_MT, LT_MP, RT_MP

def build_filename(nh, view, pose):
    """Build a filename from the hand, name, view, and pose."""
    return f"{nh}_{view}_{pose}_"

def get_hand_names():
    """Return the hand names to group by."""
    return ['Lt', 'Rt']

def get_view_names():
    """Return the view names to group by."""
    return ['Palmar', 'Rad_Obl', 'Rad_Side', 'Uln_Obl', 'Uln_Side']

def get_pose_names():
    """Return the pose names to group by."""
    return ['Ext', 'Fist', 'IM']

def build_group_names():
    """Build the group names from the hand, view, and pose."""
    hand_names = get_hand_names()
    view_names = get_view_names()
    pose_names = get_pose_names()
    group_names = []
    for hand in hand_names:
        for view in view_names:
            for pose in pose_names:
                group_names.append(build_filename(hand, view, pose))
    return group_names

def process_stats(df_rows, stats_fun, ignore_columns,out_directory):
    # the first filename from the row
    row_name = df_rows['filename'].iloc[0]

    # strip off the last 6 characters to get the filename ( ex _1.csv )
    row_name = row_name[:-6]

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
        sfile = f"{clean_spaces(row_name)}{clean_spaces(col_name)}.csv"
        df_cols.to_csv(  os.path.join(out_directory,sfile) , index=False, header=True, sep=',')

def main():

    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")
    parser.add_argument("-f","--file", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_49/compare_angles/all_angles_combined.csv", help="Path to the input CSV file.")    
    parser.add_argument("-o","--out_filename", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_49/compare_angles/vs_combine_output.csv", help="Path and filename for the converted video.")
    parser.add_argument("-d","--directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_49/stats/", help='Output directory to save the files')
    parser.add_argument('-i', '--input', type=str, default="filename", help='Default column name to group by')
    parser.add_argument("-m","--mfile", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_49/compare_angles/my_all_angles_combined.csv", help="Path to the input CSV file.")

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
    out_filename = args['out_filename']
    out_directory = args['directory']
    header_input = args['input']
    mfile = args['mfile']

    DIR_VIEW_POSE = "view_pose"
    DIR_VIEW_POSE_ALL = "all"
    DIR_VIEW_POSE_COMPARE = "compare"
    DIR_THIER = "theirs"
    DIR_MINE = "mine"
    
    
    stats_fun = [args['stats']]
    #stats_fun = ['max', 'min', 'mean', 'median', 'std', sem]

    # Ignore the columns that are not numeric
    ignore_columns = ['filename', 'nh', 'md', 'mt', 'mp', 'model']

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

    for nd in [DIR_THIER, DIR_MINE]:
        os.makedirs(os.path.join(out_directory, nd),exist_ok=True)
        os.makedirs(os.path.join(out_directory, nd, DIR_VIEW_POSE),exist_ok=True)
        os.makedirs(os.path.join(out_directory, nd, DIR_VIEW_POSE_ALL),exist_ok=True)  

    # Try and load the CSV file
    try:
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
            group_names = build_group_names()
        elif( os.path.exists(args['group_names']) ):
            with open(args['group_names'], 'r') as f:
                group_names = f.readlines()
                group_names = [x.strip() for x in group_names]
        else:
            group_names = args['group_names']

        
        # Clean up the spaces in the filename columns
        df['filename'] = df['filename'].apply(clean_spaces)
        mdf['filename'] = mdf['filename'].apply(clean_spaces)

        # Then save the files as separate files
        # Run stats on the files if indicated.

        # For each column name, group by that column name
        for row_name in group_names:

            # Get all the rows that contain that name
            df_rows = df[ df[header_input].str.contains(row_name, case=False, na=False)]
            mdf_rows = mdf[ mdf[header_input].str.contains(row_name, case=False, na=False)]

            # If the column name is not found, then skip
            if( len(df_rows) <= 0 or len(mdf_rows) <= 0):
                print(f"Row not found: {row_name}")
                continue

            # If the stats is not empty, then compute the stats
            if( args['stats'] != ""):                
                stats_df = process_stats(df_rows, stats_fun, ignore_columns, os.path.join(out_directory, DIR_THIER, DIR_VIEW_POSE))
                mstats_df = process_stats(mdf_rows, stats_fun, ignore_columns, os.path.join(out_directory, DIR_MINE, DIR_VIEW_POSE))

                # Select the columns that we want to save independentely
                # This could be the particular fingers or joints
                process_individual_files(stats_df, row_name, column_names, os.path.join(out_directory,DIR_THIER,DIR_VIEW_POSE_ALL))
                process_individual_files(mstats_df, row_name, column_names, os.path.join(out_directory,DIR_MINE,DIR_VIEW_POSE_ALL))

                # Subtract all columns that have matching column names, excluding the filename or ignore columns
                # remove the filename and ignore columns
                stats_df = stats_df.drop(columns=ignore_columns, errors='ignore')
                mstats_df = mstats_df.drop(columns=ignore_columns, errors='ignore')
                results = (stats_df - mstats_df).abs()

               
              
                # Save the comparison to a new file
                if( args['keep'] == "all"):
                    results.to_csv( os.path.join(out_directory, DIR_VIEW_POSE_COMPARE,f"{clean_spaces(row_name)}all_compare.csv"), index=False, header=True, sep=',')
                else:
                    # Check which columns start with the keep value and save them
                    keep_columns = results.filter(like=args['keep'], axis=1)
                    keep_columns.to_csv( os.path.join(out_directory, DIR_VIEW_POSE_COMPARE,f"{clean_spaces(row_name)}{args['keep']}_compare.csv"), index=False, header=True, sep=',')
                

    except FileNotFoundError:
        print(f"File not found: {file_input}")
        return

if __name__ == "__main__":
    main()

