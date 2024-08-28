# This program takes multiple patient directories and and joints the similar files and headers into a single file.
import os
import pandas as pd
import numpy as np
import argparse
from pathlib import Path

from file_utils import setupAnyCSV, clean_spaces

from convert_mediapipe_index import get_joint_names

from scipy import stats
from scipy.stats import ttest_ind

from pprint import pprint

def print_group(grouped, group_label):
    # Check if the group label exists in the grouped object
    if group_label in grouped.groups:
        # Get the group and pretty-print it
        group = grouped.get_group(group_label)
        print(f"Group '{group_label}':")
        pprint(group.to_dict(orient='list'))  # Pretty-print the group as a dictionary
    else:
        print(f"Group '{group_label}' not found in the grouped object.")


def print_and_group(df, group_by_column, group_label):
    # Group by the specified column
    grouped = df.groupby(group_by_column)
    
    # Check if the group label exists
    if group_label in grouped.groups:
        # Get and print the group
        group = grouped.get_group(group_label)
        print(f"Group '{group_label}':")
        print(group)
    else:
        print(f"Group '{group_label}' not found in the DataFrame.")

# Sample DataFrame
data = {
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': [1, 2, 3, 4, 5, 6, 7, 8],
    'C': [9, 10, 11, 12, 13, 14, 15, 16]
}

# stats = ['max', 'min', 'mean', 'median', 'std', 'var', sem]
# LT, RT
# Palmer, Rad_Obl, Rad_Side, Uln_Obl, Uln_Side,
# Fist, Ext, IM
# get_joint_names()
# LT_UT, LT_UT, RT_UT, LT_MT, RT_MT, LT_MP, RT_MP

def build_filename(view, pose):
    """Build a filename from the hand, name, view, and pose."""
    return f"{view}_{pose}"

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
    view_names = get_view_names()
    pose_names = get_pose_names()
    group_names = []
    for view in view_names:
        for pose in pose_names:
            group_names.append(build_filename(view, pose))
    return group_names

def opposite_hand_name(filename):
    """Replace hand name with the opposite hand."""
    if 'Lt' in filename:
        return filename.replace('Lt', 'Rt')
    elif 'Rt' in filename:
        return filename.replace('Rt', 'Lt')
    else:
        return filename

def is_left_hand(filename):
    """Check if the filename is for the left hand."""
    return 'Lt' in filename

def remove_hand_name(filename):
    """Remove the hand name from the filename."""
    return filename.replace('Lt_', '').replace('Rt_', '')




def get_all_keywords():
    """Return all the keywords for the filenames."""
    # Split get_view_names because of _ in the names
    clean_views = [view.split('_') for view in get_view_names()]

    # Flatten the list
    clean_views = [item for sublist in clean_views for item in sublist]
    return get_hand_names() + clean_views + get_pose_names()

def keep_only_keywords( filename, keywords):
    # Remove any words not contained in the keywords list
    return '_'.join([word for word in filename.split('_') if word in keywords])

def get_all_files(directory):

    all_files = []
    # Check if the directory exists
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
    else:
        for root, dirs, files in os.walk(directory):
            # sort the files
            files.sort()
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))
    return all_files

def run_ttest(grp1, grp2):

    # Check the groups are the same size
    # print the sizes of the groups    
    if( grp1.shape[0] != grp2.shape[0]):
        print(f"Error: Groups are not the same size. Group 1: {grp1.shape[0]} Group 2: {grp2.shape[0]}")
        return   

    results = []  
    t_stat, p_value = ttest_ind(grp1,grp2)
    results.append({'group1':'A',
                    'group2':'B',
                    'T-statistic': t_stat,
                    'P-value': p_value})
    results_df = pd.DataFrame(results)
    return results_df

def run_quartiles(df):
    # Calculate the quartiles for each column
    results = []
    # all combined_df columns that are not the filename
    q = df.quantile([0.25, 0.75])
    for col in df.columns:
        q1 = q[col].iloc[0]
        q3 = q[col].iloc[1]
        iqr = q3 - q1
        results.append({'Column':col, 'Q1': q1, 'Q3': q3, 'IQR': iqr})

    return results


def save_results(results, out_directory, out_file):
    # If results is array of dictionaries, convert to a dataframe
    if( isinstance(results, list) and len(results) > 0) and isinstance(results[0], dict):
        results_df = pd.DataFrame(results)
    elif( isinstance(results, pd.DataFrame)):
        results_df = results
    else:
        print("Error: Results is not a list of dictionaries or a pandas dataframe.")
        return

    if( out_directory is None or out_file is None):
        return

    results_df.to_csv( os.path.join(out_directory, out_file), index=False, header=True, sep=',')

def make_default_output_directores(out_directory):
       # Check if the output directory exists, otherwise create it
    
    os.makedirs(out_directory, exist_ok=True)

    my_out_directory = os.path.join(out_directory, 'mine')
    their_out_directory = os.path.join(out_directory, 'theirs')
    combined_out_directory = os.path.join(out_directory, 'combined')
    compare_out_directory = os.path.join(out_directory, 'compare')
    raw_combined = os.path.join(out_directory, 'raw_combined')    
    raw_compare = os.path.join(out_directory, 'raw_compare')

    os.makedirs( my_out_directory, exist_ok=True)
    os.makedirs( their_out_directory, exist_ok=True)
    os.makedirs( combined_out_directory, exist_ok=True)
    os.makedirs( compare_out_directory, exist_ok=True)
    os.makedirs( raw_combined, exist_ok=True)
    os.makedirs(raw_compare, exist_ok=True)

    return my_out_directory, their_out_directory, combined_out_directory, compare_out_directory, raw_combined, raw_compare

def main():

    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")

    # parser a list of directories
    parser.add_argument("-i","--input_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/raw_combined/", help="Directory of combined files for final analysis")
    parser.add_argument("-d","--out_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/", help='Output directory to save the files')

    args = vars(parser.parse_args())
    out_directory = args['out_directory']
    input_directory = args['input_directory']

    # Check the input directory
    if not os.path.exists(input_directory):
        print(f"Input directory {input_directory} does not exist.")
        return

    # Setup the output directories
    my_out_directory, their_out_directory, combined_out_directory, compare_out_directory, raw_combined, raw_compare = make_default_output_directores(out_directory)

    # For each directory, find the csv files
   
    all_files = get_all_files(input_directory)

    for file in all_files:
        try:
            # Check if the file exists
            if not os.path.exists(file):
                print(f"File {file} does not exist.")
                continue

            if(not is_left_hand(file)):
                continue

            left_df = setupAnyCSV(file)
            left_df['hand'] = 'Lt'
            # Find the right filename in all_files array                
            right_df = setupAnyCSV(opposite_hand_name(file))
            right_df['hand'] = 'Rt'

            left_file = clean_spaces(os.path.basename(file))
            right_file = opposite_hand_name(left_file)
            hand_file = remove_hand_name(right_file)

            combined_df = pd.concat([left_df, right_df], axis=0, ignore_index=True)
            out_file = os.path.join(combined_out_directory, f"both_hands_{hand_file}")
            combined_df.to_csv(out_file, index=False, header=True)

            # Take the mean of each column in each of the data frames.
            results = []
            final_df = pd.DataFrame()

            groupby =  'filename' if('filename' in combined_df.columns) else 'Group_name'
            cols_test = [col for col in combined_df.columns if col not in [groupby, 'hand']]
            cols_test = [col for col in cols_test if col.startswith('max_')]

            #################################################################################################################
            ############################## T-TEST ###########################################################################
            # Get Groups
            left_grp = left_df.groupby(groupby)[cols_test]
            right_grp = right_df.groupby(groupby)[cols_test]

            my_left_grp = left_grp.get_group('two')[cols_test]
            my_right_grp = right_grp.get_group('two')[cols_test]

            their_left_grp = left_grp.get_group('one')[cols_test]
            their_right_grp = right_grp.get_group('one')[cols_test]

            combined_their_grp = combined_df.groupby(groupby).get_group('one')[cols_test]
            combined_mine_grp = combined_df.groupby(groupby).get_group('two')[cols_test]

            # Compare My Left side values to their Left side values
            results = run_ttest( their_left_grp, my_left_grp)
            save_results(results, compare_out_directory, f"ttest_my_their_left_hand_{left_file}")

            # Compare My Right side values to their Right side values
            results = run_ttest( their_right_grp, my_right_grp)
            save_results(results, compare_out_directory, f"ttest_my_their_right_hand_{right_file}")          

            # Compare Combined My side to Combined Their side
            results = run_ttest( combined_their_grp, combined_mine_grp)
            save_results(results, combined_out_directory, f"ttest_combined_hands{hand_file}")

            #################################################################################################################
            ############################## QUARTILES ########################################################################

            # Calculate the quartiles for my left_hand
            results = run_quartiles(my_left_grp)
            save_results(results, my_out_directory, f"quartiles_my_{left_file}")

            # Calculate the quartiles for my right_hand
            results = run_quartiles(my_right_grp)
            save_results(results, my_out_directory, f"quartiles_my_{right_file}")

            # Calculate the quartiles for their left_hand
            results = run_quartiles(their_left_grp)
            save_results(results, their_out_directory, f"quartiles_their_{left_file}")

            # Calculate the quartiles for their right_hand
            results = run_quartiles(their_right_grp)
            save_results(results, their_out_directory, f"quartiles_their_{right_file}")

            # Run quartiles on combined df
            results =  run_quartiles(combined_df[cols_test])
            save_results(results, combined_out_directory, f"quartiles_both_hands_{hand_file}")

            #################################################################################################################
            ############################## Mean #############################################################################

            # Calculate the mean for mine vs their for each hand            
            final_df['mean_their_left_hand'] = their_left_grp.mean()
            final_df['mean_their_right_hand'] = their_right_grp.mean()
            final_df['mean_my_left_hand'] =  my_left_grp.mean()
            final_df['mean_my_right_hand'] = my_right_grp.mean()

            # Calculate the difference between the two groups
            final_df['mean_diff_my_left_right'] = my_left_grp.mean() - my_right_grp.mean()

            # Calculate the difference between the combined two groups
            final_df['mean_diff_combined'] = combined_their_grp.mean() - combined_mine_grp.mean()

            #################################################################################################################
            ############################## Median of Absolute Differences####################################################
            # Calculate the median of the absolute differences between the two groups
            final_df['median_abs_diff_their_left_right'] = np.abs(their_left_grp - their_right_grp).median()

            # Calculate the media of the absolute differences between their two groups
            final_df['median_abs_diff_my_left_right'] = np.abs(my_left_grp - my_right_grp).median()

            # # Calculate the median of the absolute differences between the two groups
            # t = np.abs(combined_their_grp - combined_mine_grp).median()

            # # Check if either group has nan values
            # if( combined_their_grp.isnull().values.any()):
            #     print(f"Error: Their group has nan values. {t}")
            # if( combined_mine_grp.isnull().values.any()):
            #     print(f"Error: My group has nan values. {t}")
                
            # if( t.isnull().values.any()):
            #     print(f"Error: Combined group has nan values. {t}")
            
            # print(combined_their_grp)
            # print("********************")
            # print(combined_mine_grp)
            # print("********************")
            # z = np.abs(combined_their_grp.reset_index() - combined_mine_grp.reset_index())
            # print(z)
            # b = z.median()
            final_df['median_abs_diff_combined'] = np.abs(combined_their_grp.reset_index() - combined_mine_grp.reset_index()).median()
            #################################################################################################################
            ############################## Median of Absolute Differences####################################################

            out_file = os.path.join(combined_out_directory, f"final_{clean_spaces(hand_file)}.csv")
            final_df.to_csv(out_file, index=False, header=True)

        except Exception as e:
            print(f"Error {e}")                


def setup_intput_directory_defaults(input_directory, select_directories = False):
    if( input_directory is None):
        print("No input directories provided.")

        if( select_directories == True):            
            input_directory = [
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_49/compare_angles/stats/compare/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_50/compare_angles/stats/compare/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_51/compare_angles/stats/compare/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_52/compare_angles/stats/compare/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_53/compare_angles/stats/compare/",
                                ]
        else:            
            input_directory = [
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_49/compare_angles/stats/combined/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_50/compare_angles/stats/combined/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_51/compare_angles/stats/combined/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_52/compare_angles/stats/combined/",
                                "/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_53/compare_angles/stats/combined/",
                                ]
    return input_directory

def combine_all_input_files(input_directory, out_directory):
      # Create a dictionary, of dictionary, of pandas dataframes that append the dataframes of each filename
    #  directory -> filename -> dataframe
    all_dataframes = {}

    # For each directory, find the csv files
    all_files = []
    all_keywords = get_all_keywords()

    for directory in input_directory:

        all_files = []

        # Check if the directory exists
        if not os.path.exists(directory):
            print(f"Directory {directory} does not exist.")
            continue

        for root, dirs, files in os.walk(directory):
            # sort the files
            files.sort()
            for file in files:
                if file.endswith(".csv"):
                    all_files.append(os.path.join(root, file))

        # For each file, read the csv file and append the data to the dictionary
        for file in all_files:
            try:
                # Check if the file exists
                if not os.path.exists(file):
                    print(f"File {file} does not exist.")
                    continue

                df = setupAnyCSV(file)

                # Clean the spaces in the column names
                df.columns = [clean_spaces(col) for col in df.columns]
                # Get the filename without the extension
                f = os.path.splitext(os.path.basename(file))[0]

                # Remove any words that are not in the keywords list
                filename = keep_only_keywords(f, all_keywords)

                # Check if the filename is in the dictionary
                if filename in all_dataframes:
                    # Append but ignore index duplicate headers
                    all_dataframes[filename] = pd.concat([all_dataframes[filename], df], axis=0, ignore_index=True)
                else:
                    all_dataframes[filename] = df

            except Exception as e:
                print(f"Error reading file {file}. Error: {e}")

    

    # Write all files to the output directory
    for filename, df in all_dataframes.items():
        out_file = os.path.join(out_directory, f"{filename}.csv")
        # Change the column filename to Group_name
        df.rename(columns={'filename':'Group_name'}, inplace=True)
        df.to_csv(out_file, index=False, header=True)


def combine_files_main():
    # Setup Arguments
    # args should be file_input and out_filename
    parser = argparse.ArgumentParser(description="Handle all the angle files.")

    # parser a list of directories
    parser.add_argument("-i","--input_directory", type=str, nargs='+', help="List of directories to search for files.")
    parser.add_argument("-d","--out_directory", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/final/", help='Output directory to save the files')
    parser.add_argument("-s","--select_directories", action='store_true', help='Select the directories to use')

    args = vars(parser.parse_args())
    out_directory = args['out_directory']
    input_directory = args['input_directory']

    # Create the output directory
    
    raw_combined = os.path.join(out_directory, 'raw_combined')
    raw_compare = os.path.join(out_directory, 'raw_compare')
    os.makedirs(out_directory, exist_ok=True)
    os.makedirs(raw_compare, exist_ok=True)
    os.makedirs(raw_combined, exist_ok=True)

    # Setup the input directories
    input_directory = setup_intput_directory_defaults(input_directory,args['select_directories'])

    # Combine all the files
    combine_all_input_files(input_directory, raw_combined)
 

if __name__ == "__main__":
    #combine_files_main()
    main()
