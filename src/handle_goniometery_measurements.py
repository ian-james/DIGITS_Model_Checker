# This program cleans and processes the goniometry measurements from the dataset.
# These are the real-world measurements that were taken of the participants.
# The input file has a timestamp then all the landmark, another timestamp, and all the joint angles, then the filename
# The output will be placed into the same output directory with user indicate filenames.

# Import the necessary libraries
import os
import argparse
import pandas as pd
from file_utils import setupAnyCSV, clean_spaces
from convert_mediapipe_index import real_world_landmark_names, convert_real_finger_name_to_mediapipe_name_within_string, convert_real_joint_name_to_mediapipe_name_within_string
import re

from hand_file_builder import *

# ID	Index MP joint (flexion)	Index PIP joint (flexion)	Index DIP joint (flexion)	Long MP joint (flexion)	Long PIP joint (flexion)	Long DIP joint (flexion)	Ring MP joint (flexion)	Ring PIP joint (flexion)	Ring DIP joint (flexion)	Little MP joint (flexion)	Little PIP joint (flexion)	Little DIP joint (flexion)	Thumb MP joint (flexion)	Thumb IP joint (flexion)	Index MP joint (extension)	Index PIP joint (extension)	Index DIP joint (extension)	Long MP joint (extension)	Long PIP joint (extension)	Long DIP joint (extension)	Ring MP joint (extension)	Ring PIP joint (extension)	Ring DIP joint (extension)	Little MP joint (extension)	Little PIP joint (extension)	Little DIP joint (extension)	Thumb MP joint (extension)	Thumb IP joint (extension)	Thumb abduction	Thumb opposition	Thumb circumduction:

# Check if the column name matches any of the keywords
def check_keywords(column_name, keywords):
    return any(keyword.lower() in column_name.lower() for keyword in keywords)

def separate_row_by_pose(row, filename):
    pdf = row
    if(is_extension_file(filename)):
        # Select only the columns that are extension                    
        pdf = row.filter(regex='extension|Ext', axis=0)

    if(is_flexion_file(filename)):
        # Select only the columns that are flexion
        pdf = row.filter(regex='flexion|flex|Fist', axis=0)

    if(is_intrinsic_minus_file(filename)):
        # Select only the columns that are intrinsic minus
        pdf = row.filter(regex='intrinsic minus|intrinsic_minus|IM', axis=0)
    
    return pdf

# def process_goniometry_measurement_file(input_file, has_left_file, is_left=False):

#     try:
#         df = setupAnyCSV(input_file)

#        # Get the first column name
#         first_column = df.iloc[:, 0]  # Keeps the first column

#         df_ext = df.apply(lambda row: separate_row_by_pose(row, "extension"), axis=1)
#         cdf_ext = df_ext.copy()        

#         df_flex = df.apply(lambda row: separate_row_by_pose(row, "flexion"), axis=1)
#         cdf_flex = df_flex.copy()

        


def process_goniometry_measurement_file(input_file, has_left_file, is_left=False):
    try:
        df = setupAnyCSV(input_file)
        print(f"Processing file: {input_file}")
        print(f"Dataframe shape: {df.shape}")

        # Get the real world column and landmark names
        column_names = real_world_landmark_names()

        # Get the first column name
        first_column = df.iloc[:, 0]  # Keeps the first column

        # Filter columns where any keyword is found in the column name
        filtered_df = df.loc[:, df.columns.map(lambda col: check_keywords(col, column_names))]

        # After the fisrt column, the data is the real-world measurements
        # Combining the first column back with the filtered dataframe
        df = pd.concat([first_column, filtered_df.iloc[:, 1:]], axis=1)

        # Remove any patients that have no data
        df = df.dropna(subset=column_names[0])
        print(df.head())

        # Get setup for the file names based on the hand, view, and pose
        if( not has_left_file):
            hand_view_pose_group = build_nvp_group_names(include_im=False)
        else:
            # remove all the strings that have LT in them
            if( is_left):
                hand_view_pose_group = [x for x in build_nvp_group_names(False) if is_left_hand(x)]
            else:
                hand_view_pose_group = [x for x in build_nvp_group_names(False) if not is_left_hand(x)]

        # For each row in the table we are going to replace
        # Take the first column combine it with the hand_view_pose_group name
        # Then save the file to the output directory

        print(df.columns)
        second_half_columns = df.columns[1:].values.tolist()
        ndf = pd.DataFrame(columns=['id','filename', 'nh', 'md', 'mt', 'mp', 'model'])

        for index, row in df.iterrows():
            # Get the first column value
            first_column_value = row.iloc[0]

            # For each filename in the hand_view_pose_group
            # Combin the filename with the first column value

            #include the setup information as well
            for hvp in hand_view_pose_group:
                id =int(first_column_value),
                filename = f"DIGITS_CJ_{str(int(first_column_value))}_{hvp}.csv"
                print(f"Filename: {filename}")
                item = {
                        'id': id[0],
                        'filename': filename,
                        'nh': 1,
                        'md': 0.5,
                        'mt': 0.5,
                        'mp': 0.5,
                        'model': "real"
                }

                second_half_df = pd.DataFrame(row[1:]).T
              
                second_half_df = second_half_df.apply(lambda row: separate_row_by_pose(row, hvp), axis=1)
                second_half_columns = [re.sub(r'\(.*\)', '', x) for x in second_half_df.columns]
                # Remove the word 'joint' from the columns
                second_half_columns = [re.sub(r'joint', '', x).strip() for x in second_half_columns]
                 

                    # Change Fingernames to Mediapipe names for the fingers and then followed by the joints.
                second_half_columns = [convert_real_finger_name_to_mediapipe_name_within_string(col) for col in second_half_columns] 
                second_half_columns = [convert_real_joint_name_to_mediapipe_name_within_string(col) for col in  second_half_columns]
                second_half_df.columns = second_half_columns

                # Reorder the finger columns to match the mediapipe order

                mediapipe_order = list(get_joint_names().values())
                #Check if CMC mediapipe order is in the columns
                if not mediapipe_order[0] in second_half_df.columns:
                    # Not working with correct columns, so skip CMC                                        
                    mediapipe_order = mediapipe_order[1:]

                # Check if number of columns are equal
                if len(mediapipe_order) == len(second_half_df.columns):
                    second_half_df = second_half_df[mediapipe_order]
                else:
                    print(f"Columns do not match for {filename}")
                    print(f"Columns: {second_half_df.columns}")
                    print(f"Mediapipe Order: {mediapipe_order}")
            
                # Combine the first column with the second half
                item_df = pd.DataFrame([item])
                a = second_half_df.copy().reset_index(drop=True)
                sdf = pd.concat([item_df, a], axis=1)
               
            
                ndf = pd.concat([ndf, sdf], axis=0, ignore_index=True)
               
        ndf.dropna(axis=0)
        print(f"Final Dataframe shape: {df.shape}")
        return ndf

    except Exception as e:
        print(f"Error: {e}")
        raise

def main():
     # Setup Arguments
    parser = argparse.ArgumentParser(description="Split a CSV file into multiple files based on a column value.")
    #parser.add_argument("-i", "--input_file", type=str, default="datasets/uwo/goniometry_measured.csv", help="Path to the input CSV file.")
    parser.add_argument("-i", "--input_file", type=str, default="datasets/uwo/DIGITS_Right_Hand_Goniometry_Measurements.csv", help="Path to the input CSV file.")    
    parser.add_argument("-o", "--output_file", type=str, default="datasets/uwo/formatted_goniometry_measures.csv", help="Directory to save the split files.")
    parser.add_argument("-l", "--left_file", type=str, default="datasets/uwo/DIGITS_Left_Hand_Goniometry_Measurements.csv", help="Potentially load left and right separately, right is the default")
    #parser.add_argument("-l", "--left_file", type=str, default="", help="Potentially load left and right separately, right is the default")
    
    args = vars(parser.parse_args())

    input_file = args['input_file']
    output_file = args['output_file']
    left_file = args['left_file']
    left_file_exists = left_file != ""

    # Check if the file exists
    if not os.path.exists(input_file):
        print(f"File not found: {input_file}")
        return

    if( left_file != ""):
        if not os.path.exists(left_file):
            print(f"File not found: {left_file}")
            return

    try:
       
        df = process_goniometry_measurement_file(input_file, left_file_exists, is_left=False)
        #df.to_csv("datasets/uwo/right.csv", index=False)
        if( left_file != ""):
            left_df = process_goniometry_measurement_file(left_file, left_file_exists, is_left=True)
            #left_df.to_csv("datasets/uwo/left.csv", index=False)
            df = pd.concat([df, left_df], axis=0, ignore_index=True)            
            # Sort the dataframe by the index and filename
            # Sort by 'Index' and then by 'Filename' using natural sorting
            df = df.sort_values(by=['id','filename'])

        # Drop the index column and save
        df.drop(columns=['id'], inplace=True)
        print(f"Final Dataframe shape: {df.shape}")        
        df.to_csv(output_file, index=False)
        print(f"Saved file to {output_file}")

    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()