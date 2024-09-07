import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from file_utils import setupAnyCSV, clean_spaces

from convert_mediapipe_index import get_joint_names, get_just_joint_names
from pprint import pprint

# TODO: Pivot tables are basically the same idea as is algorithmic process.
# Perhaps that could be generalized to a function that takes in the desired order of columns and rows.


# stats = ['max', 'min', 'mean', 'median', 'std', 'var', sem]
# LT, RT
# Palmer, Rad_Obl, Rad_Side, Uln_Obl, Uln_Side,
# Fist, Ext, IM
# get_joint_names()
# LT_UT, LT_UT, RT_UT, LT_MT, RT_MT, LT_MP, RT_MP
from hand_file_builder import *


def finger_joint_order():
    """Return the finger joint order."""
    return  ["max_Thumb_CMC",
        "max_Thumb_MCP",
        "max_Thumb_IP",
        "max_Index_MCP",
        "max_Index_PIP",
        "max_Index_DIP",
        "max_Long_MCP",
        "max_Long_PIP",
        "max_Long_DIP",
        "max_Ring_MCP",
        "max_Ring_PIP",
        "max_Ring_DIP",
        "max_Small_MCP",
        "max_Small_PIP",
        "max_Small_DIP"
    ]


# Setup the table in the format we want.
def pivot_table_combine(df, out_directory):
    try:
        # Create a pivot table
        pivot_df = df.pivot_table(index=['finger','joint','view'],columns=['pose'],values=['median_abs_diff_combined','Q1','Q3','IQR','p_value'])

        # Flatten the MultiIndex columns
        pivot_df.columns = [f'{stat}_{pose}' for stat, pose in pivot_df.columns]

        # Reset index to turn the index back into columns
        pivot_df.reset_index(inplace=True)

        # Define the desired order of statistics
        desired_order = ['median_abs_diff_combined', 'Q1', 'Q3', 'IQR', 'p_value']
        desired_pose_order = ['Ext', 'Fist', 'IM']

        # Reorder the columns based on the desired order
        columns_in_order = [f'{stat}_{pose}' for pose in desired_pose_order for stat in desired_order]

        pivot_df = pivot_df[['finger', 'joint', 'view'] + columns_in_order]

        if( out_directory is not None):
            print("\nSaving pivoted DataFrame to CSV file...")
            out_file = os.path.join(out_directory, "final_pivot_table.csv")
            pivot_df.to_csv(out_file, index=False, header=True)
        else:
            print("\nPivoted DataFrame:")
            print(pivot_df)

    except Exception as e:
        print(f"Error reading file: {e}")
        return


def main():

    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")
    parser.add_argument("-d","--directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/combined/", help="Path to the input csv.")
    parser.add_argument("-o","--out_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/combined/table_setup/", help="Path to the output csv.")

    args = vars(parser.parse_args())

    # Check  if the input file exists
    directory = args['directory']
    out_directory = args['out_directory']

    # Check if the file_path is a directory and already exists.
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return

    # Files Examples
    # "final_Palmar_Ext.csv"
    # "quartiles_both_hands_Palmar_Ext.csv"
    # "ttest_combined_hands_Palmar_Fist.csv"
    os.makedirs(out_directory, exist_ok=True)

    total_df = pd.DataFrame()
    for file in os.listdir(directory):
        if file.startswith("final") and file.endswith(".csv"):
            print(f"Processing: {file}")

            try:
                df = setupAnyCSV(os.path.join(directory,file))

                q_file = file.replace("final", "quartiles_both_hands")
                q_df = setupAnyCSV(os.path.join(directory,q_file))

                t_file = file.replace("final", "ttest_combined_hands")
                t_df = setupAnyCSV(os.path.join(directory,t_file))

                # Get the keywords from the filename
                view, pose = get_view_pose_from_filename(file)

                # Create a column for the finger based on df['columns']
                df['joint'] = df['columns'].apply(get_joint_from_filename)
                df['finger'] = df['columns'].apply(get_finger_from_filename)
                df['view']=view
                df['pose']=pose

                col = df[['median_abs_diff_combined']]

                qcols =  q_df[["Q1","Q3","IQR"]]

                tcols = t_df[["p_value"]]

                ndf = pd.concat([df[['finger','joint','view','pose']],col,qcols,tcols],axis=1)
                ndf.to_csv(os.path.join(out_directory, f"final_{view}_{pose}.csv"), index=False)

                total_df = pd.concat([total_df,ndf],axis=0)
            except Exception as e:
                print(f"Error reading file: {e}")
                continue

    out_file = os.path.join(out_directory, "final_table.csv")
    total_df.to_csv(out_file, index=False, header=True)

    pivot_table_combine(total_df, out_directory)


def their_vs_mine_main():

    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")
    parser.add_argument("-d","--directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/combined/", help="Path to the input csv.")
    parser.add_argument("-o","--out_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/compare/table_setup/", help="Path to the output csv.")
    parser.add_argument("-c","--compare", type=str, default="mine", help="Compare Ethans to mine.")
    parser.add_argument("-t","--ttest_directory", type=str,
                        default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/combined/",
                        help="Path to the ttest (if not the default) only works with my comparison.")

    parser.add_argument("-q","--quartile_directory", type=str,
                        default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/compare/",)

    args = vars(parser.parse_args())

    # Check  if the input file exists
    directory = args['directory']
    out_directory = args['out_directory']
    ttest_directory = args['ttest_directory']
    quartile_directory = args['quartile_directory']

    # Check if the file_path is a directory and already exists.
    if not os.path.exists(directory):
        print(f"Final Directory not found: {directory}")
        return

    # Check if the file_path is a directory and already exists.
    if args['compare'] == "mine":
        if not os.path.exists(ttest_directory):
            print(f"Ttest Directory not found: {ttest_directory}")
            ttest_directory = directory
            print("Defaulting to the final directory: f{ttest_directory}")
        if( not os.path.exists(quartile_directory)):
            print(f"Quartile Directory not found: {quartile_directory}")
            quartile_directory = directory
            print("Defaulting to the final directory: f{quartile_directory}")

    # Files Examples
    # "final_Palmar_Ext.csv"
    # "quartiles_both_hands_Palmar_Ext.csv"
    # "ttest_combined_hands_Palmar_Fist.csv"
    os.makedirs(out_directory, exist_ok=True)
    os.makedirs(ttest_directory, exist_ok=True)

    total_df = pd.DataFrame()
    stats = ['mean_their_combined', 'mean_my_combined', 'mean_diff_combined', 'median_abs_combined_their', 'median_abs_combined_mine', 'median_abs_diff_combined']
    tstats = ["Q1","Q3","IQR"]
    pval = ["p_value"]
    iorder = ['finger','joint','view']
    pivot = ['pose']
    poses =  ['Ext', 'Fist', 'IM']

    for file in os.listdir(directory):
        if file.startswith("final") and file.endswith(".csv"):
            print(f"Processing: {file}")

            try:
                df = setupAnyCSV(os.path.join(directory,file))

                 # Get the keywords from the filename
                view, pose = get_view_pose_from_filename(file)
                # Create a column for the finger based on df['columns']
                df['joint'] = df['columns'].apply(get_joint_from_filename)
                df['finger'] = df['columns'].apply(get_finger_from_filename)
                df['view']=view
                df['pose']=pose

                # Example - quartiles_my_Lt_Palmar_Ext.csv
                my_qfile  = f"quartiles_my_{build_nvp_filename('combined', view, pose)}.csv"
                mq_df = setupAnyCSV(os.path.join(quartile_directory,my_qfile))
                mqcols =  mq_df[tstats]
                mqcols = mqcols.add_prefix("my_")

                # Example - quartiles_my_Lt_Palmar_Ext.csv
                my_tfile  = f"quartiles_their_{build_nvp_filename('combined', view, pose)}.csv"
                mt_df = setupAnyCSV(os.path.join(quartile_directory,my_tfile))
                mtcols =  mt_df[tstats]
                mtcols = mtcols.add_prefix("their_")

                # Example ttest_my_their_Rt_Palmar_Ext.csv
                t_file = f"ttest_{build_nvp_filename('combined_hands', view, pose)}.csv"
                t_df = setupAnyCSV(os.path.join(ttest_directory,t_file))                
                tcols = t_df[pval]

                col = df[stats]
                ndf = pd.concat([df[iorder + pivot], col, mqcols, mtcols, tcols], axis=1)
                ndf.to_csv(os.path.join(out_directory, f"final_compare_{view}_{pose}.csv"), index=False)

                total_df = pd.concat([total_df,ndf],axis=0)
            except Exception as e:
                print(f"Error reading file: {e}")
                continue

    out_file = os.path.join(out_directory, "final_comparison_table.csv")
    total_df.to_csv(out_file, index=False, header=True)

    # Add the prefixes to the tstats
    mstats = ["my_" + tstat for tstat in tstats]
    hsstats = ["their_" + tstat for tstat in tstats]
    tstats = mstats + hsstats + pval

    pivot_table_compare(total_df, out_directory, index=iorder, columns=pivot, values=stats+tstats+pval, desired_order=stats+tstats, pose_order=poses)

# Setup the table in the format we want.
# Index = ['finger','joint','view']
# Columns = ['pose']
# Values is the statistics
# desired_order = ['mean_their_combined', 'mean_my_combined', 'mean_diff_combined', 'median_abs_combined_their', 'median_abs_combined_mine', 'median_abs_diff_combined']
# pose_order = ['Ext', 'Fist', 'IM']
def pivot_table_compare(df, out_directory, index, columns, values, desired_order, pose_order):
    try:
        # Create a pivot table
        pivot_df = df.pivot_table(index=index,
                                  columns=columns,
                                  values=values
        )

        # Flatten the MultiIndex columns
        pivot_df.columns = [f'{stat}_{pose}' for stat, pose in pivot_df.columns]

        # Reset index to turn the index back into columns
        pivot_df.reset_index(inplace=True)

        # Define the desired order of statistics
        desired_order = values
        desired_pose_order = pose_order

        # Reorder the columns based on the desired order
        columns_in_order = [f'{stat}_{pose}' for pose in desired_pose_order for stat in desired_order]
        # Check Each of the desired columns are in the pivot table columns
        for col in columns_in_order:
            if col not in pivot_df.columns:
                print(f"Column not in the pivot table: {col}")

        pivot_df = pivot_df[index + columns_in_order]

        if( out_directory is not None):
            
            out_file = os.path.join(out_directory, "final_compare_pivot_table.csv")
            print(f"\nSaving pivoted DataFrame to CSV file... to {out_file}")
            pivot_df.to_csv(out_file, index=False, header=True)
        else:
            print("\nPivoted DataFrame:")
            print(pivot_df)

    except Exception as e:
        print(f"Error reading file: {e}")
        return

if __name__ == "__main__":
    #main()
    their_vs_mine_main()
