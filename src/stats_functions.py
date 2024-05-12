import numpy as np
import pandas as pd
from scipy import stats


# Function to compute the statistics
# Compute the The standard error of the mean (SEM) is a measure of how much the sample mean is expected to vary from the true population mean.
def sem(x):
    return np.std(x, ddof=0) / np.sqrt(len(x))

def var(data):
    variance = np.var(data)
    return variance

def compute_statistics(df, exclude_columns=[]):

    if( df is None):
        return None

    # Exclude specific columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore') 

    # Compute the statistics for each column
    stats_fun = ['max', 'min', 'mean', 'median', 'std', 'var'] #, sem] 
    stats_df = df.agg(stats_fun)        

    return stats_df