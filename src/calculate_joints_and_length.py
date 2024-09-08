# This program calculates the joint positions, angles, and lengths of the bones.
# It will take the mediapipe results and calculate the joint positions.
import os
import argparse
import logging
import math
from log import set_log_level
from pathlib import Path
import pandas as pd
import numpy as np

from pprint import pprint
from calculation_helpers import calculate_angle, calculate_angle_between_each_digit_joint

from convert_mediapipe_index import *

from file_utils import setupAnyCSV

import ast
def safe_literal_eval(val):
    try:
        return ast.literal_eval(val)
    except (ValueError, SyntaxError):
        return val

def apply_literal_eval_all(df):
    for col in df.columns:
        df[col] = df[col].apply(safe_literal_eval)
    return df

def setup_arguments():
    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/csvs/001-L-1-1_mediapipe_nh_1_md_0.5_mt_0.5_mp_0.5.csv", help="Load a CSV file that has been evaluated by mediapipe to produce landmark information.")

    ap.add_argument("-o", "--out_filename", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/angles/001-L-1-1_mediapipe_nh_1_md_0.5_mt_0.5_mp_0.5.csv", help="Save the digit lengths and angles to a file")

    # Add an option to save csv files separatly for lengths and angles
    ap.add_argument("-s", "--save_separately", action="store_false", help="Save the digit lengths and angles to separate files")

    return ap

# Calculate the distance between two points in 3D space.
def calculate_length( p1, p2):
    distance = math.dist(p1,p2)
    return distance

def calculate_digit_length(df, digit_tip_name, digit_base_name):
    if( df is None):
        return None
    if( digit_tip_name in df.columns and digit_base_name in df.columns):
        digit_tip = df[digit_tip_name]
        digit_base = df[digit_base_name]
        return calculate_length(digit_tip, digit_base)
    return None

# This function only works if the landmark is listed in a single Column ie [x,y,z]
# Rather than the X-Coordinate, Y-Coordinate, Z-Coordinate format.
def calculate_all_digit_lengths(df, digit_names, include_wrist=True):

    if( df is None):
        return None

    if( digit_names is None):
        digit_names = default_digit_tip_names()

    if( include_wrist):
        wrist = default_wrist_to_index_tip_names()
        digit_names.update(wrist)

    digit_lengths = {}
    for digit, names in digit_names.items():
        x1 = names[0]
        x2 = names[1]
        digit_length = df.apply( lambda x: calculate_length(x[x1], x[x2]), axis=1)
        digit_lengths[digit+"_Length"] = digit_length.reset_index(drop=True)

    return digit_lengths

# Calculate the min/max of the video, must provide the column names in full
# Index Finger MCP (X-coordinate)
def calculate_digit_range(df, digit_name):
    if( df is None):
        return None

    if( digit_name in df.columns):
        digit = df[digit_name]
        return [digit.min(), digit.max()]

    return None

## This assumes that the Coordinates are listed as X-Coordinate, Y-Coordinate, Z-Coordinate for each Joint
# If we have split each landmark into multiple columns
def calculate_full_range(df, digit_names, coordinate_names):
    if( df is None):
        return None

    if( digit_names is None):
        digit_names = default_digit_tip_names()

    if( coordinate_names is None):
        coordinate_names = default_coordinate_names()

    digit_ranges = {}
    for digit, names in digit_names.items():
        xlabel =  names[0] + coordinate_names["X"]
        ylabel =  names[0] + coordinate_names["Y"]
        zlabel =  names[0] + coordinate_names["Z"]
        digit_x_range = calculate_digit_range(df, xlabel)
        digit_y_range = calculate_digit_range(df, ylabel)
        digit_z_range = calculate_digit_range(df, zlabel)

        if( digit_x_range is not None ):
            digit_ranges[xlabel+"_Range_Min"] = digit_x_range[0]
            digit_ranges[xlabel+"_Range_Max"] = digit_x_range[1]

        if( digit_y_range is not None):
            digit_ranges[ylabel+"_Range_Min"] = digit_y_range[0]
            digit_ranges[ylabel+"_Range_Max"] = digit_y_range[1]

        if(digit_z_range is not None):
            digit_ranges[zlabel+"_Range_Min"] = digit_z_range[0]
            digit_ranges[zlabel+"_Range_Max"] = digit_z_range[1]

    return digit_ranges

def calculate_full_digit_range_df(df, column_name, coordinate_names=['x','y','z']):
    if( column_name in df.columns):

        # Convert the list of points to a numpy array for easy manipulation
        points_array = np.array(df[column_name].tolist())

        # Coordinates to process
        coordinates = coordinate_names
        max_points = {}
        min_points = {}

        # Loop through each coordinate to find the max and min points
        for i, coord in enumerate(coordinates):
            max_index = np.argmax(points_array[:, i])
            min_index = np.argmin(points_array[:, i])
            max_points[column_name+"_"+coord+"_max"] = df[column_name].iloc[max_index]
            min_points[column_name+"_"+coord+"_min"] = df[column_name].iloc[min_index]

        # # Display the min and max points dictionary
        # for key in max_points.keys():
        #     print(f"{key}: {max_points[key]}")

        # for key in min_points.keys():
        #     print(f"{key}: {min_points[key]}")

        return [min_points, max_points]


# Need to have the landmarks
def calculate_all_finger_angles(landmarks, add_wrist_to_indices=False, add_wrist = False):
    digit_indices = get_all_fingers_indices(add_wrist_to_indices, add_wrist)
    return {
        Digit.Thumb.name: calculate_angle_between_each_digit_joint(landmarks, digit_indices [Digit.Thumb.name]),
        Digit.Index.name: calculate_angle_between_each_digit_joint(landmarks,digit_indices  [Digit.Index.name]),
        Digit.Middle.name: calculate_angle_between_each_digit_joint(landmarks, digit_indices[Digit.Middle.name]),
        Digit.Ring.name:  calculate_angle_between_each_digit_joint(landmarks, digit_indices [Digit.Ring.name]),
        Digit.Pinky.name: calculate_angle_between_each_digit_joint(landmarks,digit_indices  [Digit.Pinky.name])
    }

def calculate_angle_between_digit_df(df, digit_name):
    if( df is None):
        return None

    # Get the column names for the digit
    ids = get_all_fingers_indices(True,True)[digit_name]

    # Make all the vectors
    dnames = []
    ndf = pd.DataFrame()
    for i in range(1,len(ids)):        
        name = get_landmark_name(ids[i-1])
        name2 = get_landmark_name(ids[i])

        d = "d_"+name+"_"+name2
        dnames.append(d)        
        ndf[d] = df.apply(lambda row: np.subtract(row[name], row[name2]), axis=1)

    # Calculate the angle between each vector
    
    for i in range(1,len(dnames)):        
        name = get_landmark_name(ids[i])        
        ndf[name] = ndf.apply(lambda x: calculate_angle(x[dnames[i-1]], x[dnames[i]]), axis=1)

    # Drop all vector columns
    
    for d in dnames:
        ndf.drop(columns=[d], inplace=True)

    return ndf

def calculate_all_finger_angle_df(df, digit_names):
    if( df is None):
        return None

    if( digit_names is None):
        digit_names = get_all_landmark_names()

    # For each digit, calculate the vector between each adjacvent point
    # Then calculate the angle between each vector
    rom_df = pd.DataFrame()
    for digit in Digit:
        if(digit == Digit.Wrist):
            continue
        ddf = calculate_angle_between_digit_df(df, digit.name)
        #ddf.to_csv(f"./output/{digit.name}_ROM.csv", index=False, sep=",")

        rom_df = pd.concat([rom_df,ddf], axis=1)

    #concatenate all the dataframes with just their header files
    return rom_df

def convert_csv_with_xyz_to_landmarks(df):

    df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')),errors='ignore')]
    df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')),errors='ignore')]

    landmarks = []
    for i in range(0,21):
        name = get_landmark_name(i)
        if( f"{name} (X-coordinate)" not in df.columns) or (f"{name} (Y-coordinate)" not in df.columns) or ( f"{name} (Z-coordinate)" not in df.columns):
            #print(f"Missing Landmark: {name}")
            continue

        df[name] = df.apply(lambda x: [x[f"{name} (X-coordinate)"], x[f"{name} (Y-coordinate)"], x[f"{name} (Z-coordinate)"]], axis=1)
        df.drop(columns=[f"{name} (X-coordinate)", f"{name} (Y-coordinate)", f"{name} (Z-coordinate)"], inplace=True)
    return df

def convert_csv_to_landmarks(df):
    landmarks = []
    for i in range(0,21):
        landmark = [df[f"{get_landmark_name(i)} (X-coordinate)"], df[f"{get_landmark_name(i)} (Y-coordinate)"], df[f"{get_landmark_name(i)} (Z-coordinate)"]]
        landmarks.append(landmark)
    return landmarks

def main_xyz_df(args, df):

    # Load the CSV file
    df = setupAnyCSV(args['filename'])  
    if( df is None):
        logging.error(f"Failed to load file: {args['filename']}")
        return

    try:
        df = convert_csv_with_xyz_to_landmarks(df)
        df = apply_literal_eval_all(df)
    except Exception as e:
        logging.error(f"Failed to convert the CSV file to landmarks: {e}")
        return

    try:

        # Calculate the digit lengths
        digit_lengths = calculate_all_digit_lengths(df, None )
        ldf = pd.DataFrame.from_dict(digit_lengths)        

        # Calculate the angles between each joint
        #fangles = calculate_angle_between_digit_df(df, Digit.Index.name, None)s
        fangles = calculate_all_finger_angle_df(df, None)
        fdf = pd.DataFrame.from_dict(fangles)
        
        # #Combine the two dataframes
        if( args['save_separately']):
            #  Split the out_file into directory and filename, then put the length file in the same directory            
            length_file = os.path.join(get_length_path(args['out_filename']), "length_"+os.path.basename(args['out_filename']))            
            ldf.to_csv(length_file, index=False, sep=",")
            fdf.to_csv(args['out_filename'], index=False, sep=",")
        else:
            combined_df = pd.concat([fdf,ldf], axis=1)
            combined_df.to_csv(args['out_filename'], index=False)
            print(f"Saved ROM to {args['out_filename']}")

    except Exception as e:
        logging.error(f"Failed to calculate the digit lengths and angles: {e}")
        return

def get_length_path(out_filename):
    return os.path.join(Path(os.path.dirname(out_filename)).parent, "length")
    

def main():
    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")

    # Check if the csv file exists
    if( not os.path.exists(args['filename'])):
        logging.error(f"File not found: {args['filename']}")
        return
    
    angle_directory =  os.path.join(os.path.dirname(args['out_filename']))
    os.makedirs(angle_directory, exist_ok=True)

    lenth_directory = get_length_path(args['out_filename'])
    os.makedirs(lenth_directory, exist_ok=True)

    main_xyz_df(args, None)


# ********************************************************************************************************************
if __name__ == "__main__":
    main()
