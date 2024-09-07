# This program separates the Ethan's landmark angle file into individual files for each landmark
# The input file has a timestamp then all the landmark, another timestamp, and all the joint angles, then the filename
# The output will be placed into the same output directory with user indicate filenames.

# Import the necessary libraries
import os
import argparse
import pandas as pd
from file_utils import setupAnyCSV, clean_spaces
from convert_mediapipe_index import get_joint_name, get_all_landmarks_xyz_names



def main():
     # Setup Arguments
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple files based on a column value.")
    parser.add_argument("-f", "--input_file", type=str, default="datasets/Ethans_Evaluation/csv/001-L-1-1_landmarkData.csv", help="Path to the input CSV file.")
    parser.add_argument("-o", "--output_directory", type=str, default="datasets/Ethans_Evaluation/our_videos/ethan_split_files/", help="Directory to save the split files.")
    parser.add_argument("-l", "--out_landmark_file", type=str, default="", help="Output filename for the landmark files.")
    parser.add_argument("-a", "--out_angle_file", type=str, default="", help="Output filename for the angle files.")
    parser.add_argument("-c", "--add_column_headers", action='store_false', help="Add column headers to the output files.")
    parser.add_argument('-k', "--keep_filename", action='store_false', help="Keep the original filename in the output files.")

    args = vars(parser.parse_args())

    input_file = args['input_file']
    output_directory = args['output_directory']

    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    # set the output directory to the same directory as the input file
    if( args['output_directory'] == ""):
        output_directory = os.path.dirname(input_file)

    # Check if the output directory exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        df = setupAnyCSV(input_file,None)
        print(f"Processing file: {input_file}")

        # Strip whitespace and check if the last column contains only empty strings or whitespace
        # There appears to be an extra column with just spaces (fix for that)
        if df.iloc[:, -1].str.strip().eq('').all():
            df = df.iloc[:, :-1]  # Drop the last column

        # Check that the dataframe has the shape we expect, it should have 81 columns
        if df.shape[1] <= 80:
            print("Error: The dataframe does not have the expected number of columns.")
            return

        # Get just the base name of the file without the extension
        base_filename = os.path.splitext(os.path.basename(input_file))[0]

        # The file will be format with a timestamp then all the landmark data we want to separate the dataframe into two dataframes at that point.
        # The first dataframe will be the landmark data and the second dataframe will be the joint angle data.
        # split the dataframe at the 64th column
        landmark_df = df.iloc[:, :64]

        if(args['keep_filename']):
            landmark_df = pd.concat([landmark_df, df.iloc[:, -1]], axis=1)

        header = None


        # The landmark data will be saved to a file with the user provided filename.
        if(args['out_landmark_file'] == ""):
            args['out_landmark_file'] = os.path.join(output_directory, base_filename + "_landmarks.csv")

        if(args['add_column_headers']):
            cols = get_all_landmarks_xyz_names()
            cols = ['timestamp'] +  cols 
            if(args['keep_filename']):
                cols = cols + ['filename']

            # print all the columns one per line
            for col in cols:
                print(col)
            
            landmark_df.columns = cols
            header = landmark_df.columns            

        landmark_df.to_csv(args['out_landmark_file'], index=False, header=header)

        header = None
        # The joint angle data will be saved to a file with the user provided filename.
        angle_df = df.iloc[:, 65:]
        if(args['out_angle_file'] == ""):
            args['out_angle_file'] = os.path.join(output_directory, base_filename + "_angles.csv")
       
        # Subtract timestampe and filename from the column length        
        if(args['add_column_headers']):
            column_len = len(angle_df.columns)-2
            cols = ['timestamp'] +  [get_joint_name(i) for i in range(1,column_len+1)]
            if(args['keep_filename']):
                cols = cols + ['filename']
            angle_df.columns = cols
            header = angle_df.columns

        # Get the joint names from the first row of the dataframe
        angle_df.to_csv(args['out_angle_file'], index=False, header=header)


    except Exception as e:
        print(f"Error: {e}")
        return



if __name__ == "__main__":
    main()