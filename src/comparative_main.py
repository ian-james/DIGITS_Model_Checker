

#import libraries
import os
import argparse
import logging

import numpy as np
import pandas as pd

from log import set_log_level

from stats_functions import compute_statistics
import matplotlib.pyplot as plt

def setup_arguments():
    
    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="./output/all_combined_files_analyzed_nyu.csv", help="Load a CSV file for evaluation.")

    ap.add_argument("-o", "--out_filename", type=str, default="./output/out_data.csv", help="Save to a CSV file.")

    return ap

# Given a CSV file, load the data and compute the statistics for each column
# Output a new CSV file with the statistics and the filename

def load_data_and_compute_statistics(filename, out_filename):
    # Load the data
    try:
        df = pd.read_csv(filename,sep='\t')
        if( df is None):
            return None
        
        # Add the filename before the statistics are setup
        df['filename'] = filename

        df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')),errors='ignore')]

        df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')),errors='ignore')]

        # Compute the statistics
        stats_df = compute_statistics(df, exclude_columns=['filename','time','timestamp','handedness','model','nh','md','mt','mp'])

        # Save the statistics to a new CSV file
        stats_df.to_csv(out_filename)
        return stats_df

    except Exception as e:
        logging.error(f"Failed to load the data and compute the statistics. {e}")
        print(f"Failed to load the data and compute the statistics. {e}")
        return None

def main():
    
    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")

    # Load the data and compute the statistics
    stats_df = load_data_and_compute_statistics(args['filename'], args['out_filename'])

    print(f"Finished writing the statistics to {args['out_filename']}")

    # Get the names of all the columns
    columns = stats_df.columns

    # Get the wrist columns
    wrist_columns = [col for col in columns if 'Wrist' in col]
    # Get the finger columns

    pinky_columns = [col for col in columns if 'Pinky' in col]
    ring_columns = [col for col in columns if 'Ring' in col]
    middle_columns = [col for col in columns if 'Middle' in col]
    index_columns = [col for col in columns if 'Index' in col]
    thumb_columns = [col for col in columns if 'Thumb' in col]

    # make a directory with the name of the input file and save the plots there
    out_dir = os.path.dirname(args['out_filename'])

    # Use in the put file name as part of the directory structure
    out_dir = os.path.join(out_dir, os.path.basename(args['filename']).split('.')[0])
    os.makedirs(out_dir, exist_ok=True)


    # for each group of fingers, get the mean of the variances
    for finger_columns in [wrist_columns, pinky_columns, ring_columns, middle_columns, index_columns, thumb_columns]:
        mean_of_variances = [stats_df[col]['var'] for col in finger_columns]
        print(f"Mean of Variances for {finger_columns}: {mean_of_variances}")
        # Get the variance of the variances
        variance_of_variances = np.var(mean_of_variances)
        print(f"Variance of Variances for {finger_columns}: {variance_of_variances}")
        finger_name = finger_columns[0].split()[0]
        
        # Plot the mean of variances
        plt.figure(figsize=(12, 6))
        plt.bar(finger_columns, mean_of_variances)
        plt.title(f'Mean of Variances for {finger_name}')
        plt.ylabel('Variance')
        plt.xticks(rotation=45)  # Rotate the x-axis labels by 45 degrees
        plt.tight_layout()  # Adjust layout to make room for the rotated labels
        #plt.show()

        # Save the plot to the same directory as the output file        
        plt.savefig(os.path.join(out_dir, f'mean_of_variances_{finger_name}.png'))
        plt.close()

        # Plot the variance of variances
        plt.figure(figsize=(12, 6))
        plt.bar(finger_columns, variance_of_variances)
        plt.title(f'Variance of Variances for {finger_name}')
        plt.ylabel('Variance')
        plt.xticks(rotation=45)  # Rotate the x-axis labels by 45 degrees
        plt.tight_layout()  # Adjust layout to make room for the rotated labels
        plt.savefig(os.path.join(out_dir, f'variance_of_variances_{finger_name}.png'))
        plt.close()

if __name__ == '__main__':
    main()
