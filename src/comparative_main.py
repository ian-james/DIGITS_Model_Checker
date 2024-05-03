

#import libraries
import os
import argparse
import logging

import pandas as pd

from log import set_log_level

import numpy as np
from scipy import stats

def setup_arguments():
    
    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="./output/test/hands_944_video_nh_1_md_0.4_mt_0.4_mp_0.4_mediapipe.csv", help="Load a CSV file for evaluation.")

    ap.add_argument("-o", "--out_filename", type=str, default="./output/out_data.csv", help="Save to a CSV file.")

    return ap


# Function to compute the statistics
# Compute the The standard error of the mean (SEM) is a measure of how much the sample mean is expected to vary from the true population mean.
def sem(x):
    return np.std(x, ddof=0) / np.sqrt(len(x))

def compute_statistics(df, exclude_columns=[]):

    if( df is None):
        return None

    # Exclude specific columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore') 

    # Compute the statistics for each column
    stats_fun = ['max', 'min', 'mean', 'median', 'std', sem] 
    stats_df = df.agg(stats_fun)        

    return stats_df

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

        # Compute the statistics
        stats_df = compute_statistics(df, exclude_columns=['filename','time','timestamp','handedness'])

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

if __name__ == '__main__':
    main()
