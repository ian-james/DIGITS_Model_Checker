# This program calculates the joint positions, angles, and lengths of the bones.
# It will take the mediapipe results and calculate the joint positions.
import os
import argparse
import logging
import math
from log import set_log_level
import pandas as pd
import numpy as np

from pprint import pprint
from calculation_helpers import calculate_angle, calculate_angle_between_each_digit_joint

from convert_mediapipe_index import *

def setup_arguments():


    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="./output/rgb_1_0000141_fps_30_mediapipe_nh_1_md_0.4_mt_0.4_mp_0.4.csv", help="Load a CSV file that has been evaluated by mediapipe to produce landmark information.")

    ap.add_argument("-o", "--out_filename", type=str, default="out_data.csv", help="Save the digit lengths and angles to a file")

    return ap

# The Headings are
# Wrist (X-coordinate)	Wrist (Y-coordinate)	Wrist (Z-coordinate)
# Thumb CMC (X-coordinate)	Thumb CMC (Y-coordinate)	Thumb CMC (Z-coordinate)
# Thumb MCP (Y-coordinate)	Thumb MCP (Z-coordinate)
#	Thumb IP (X-coordinate)	Thumb IP (Y-coordinate)	Thumb IP (Z-coordinate)
#	Thumb Tip (X-coordinate)	Thumb Tip (Y-coordinate)	Thumb Tip \(Z-coordinate\)
#	Index Finger MCP (X-coordinate)	Index Finger MCP (Y-coordinate)	Index Finger MCP (Z-coordinate)
#	Index Finger PIP (X-coordinate)	Index Finger PIP (Y-coordinate)	Index Finger PIP (Z-coordinate)
#	Index Finger DIP (X-coordinate)	Index Finger DIP (Y-coordinate)	Index Finger DIP (Z-coordinate)
#	Index Finger Tip (X-coordinate)	Index Finger Tip (Y-coordinate)	Index Finger Tip (Z-coordinate)
#	Middle Finger MCP (X-coordinate)	Middle Finger MCP (Y-coordinate)	Middle Finger MCP (Z-coordinate)
#	Middle Finger PIP (X-coordinate)	Middle Finger PIP (Y-coordinate)	Middle Finger PIP (Z-coordinate)
#	Middle Finger DIP (X-coordinate)	Middle Finger DIP (Y-coordinate)	Middle Finger DIP (Z-coordinate)
#	Middle Finger Tip (X-coordinate)	Middle Finger Tip (Y-coordinate)	Middle Finger Tip (Z-coordinate)
#	Ring Finger MCP (X-coordinate)	Ring Finger MCP (Y-coordinate)	Ring Finger MCP (Z-coordinate)
#	Ring Finger PIP (X-coordinate)	Ring Finger PIP (Y-coordinate)	Ring Finger PIP (Z-coordinate)
#	Ring Finger DIP (X-coordinate)	Ring Finger DIP (Y-coordinate)	Ring Finger DIP (Z-coordinate)
#	Ring Finger Tip (X-coordinate)	Ring Finger Tip (Y-coordinate)	Ring Finger Tip (Z-coordinate)
#	Pinky MCP (X-coordinate)	Pinky MCP (Y-coordinate)	Pinky MCP (Z-coordinate)
#	Pinky PIP (X-coordinate)	Pinky PIP (Y-coordinate)	Pinky PIP (Z-coordinate)
#	Pinky DIP (X-coordinate)	Pinky DIP (Y-coordinate)	Pinky DIP (Z-coordinate)
#	Pinky Tip (X-coordinate)	Pinky Tip (Y-coordinate)	Pinky Tip (Z-coordinate)

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
def calculate_all_digit_lengths(df, digit_names):

    if( df is None):
        return None

    if( digit_names is None):
        digit_names = default_digit_tip_names()

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



def calculate_all_finger_angles(landmarks, add_wrist_to_indices=False, add_wrist = False):
    digit_indices = get_all_fingers_indices(add_wrist_to_indices, add_wrist)
    return {
        Digit.Thumb.name: calculate_angle_between_each_digit_joint(landmarks, digit_indices [Digit.Thumb.name]),
        Digit.Index.name: calculate_angle_between_each_digit_joint(landmarks,digit_indices  [Digit.Index.name]),
        Digit.Middle.name: calculate_angle_between_each_digit_joint(landmarks, digit_indices[Digit.Middle.name]),
        Digit.Ring.name:  calculate_angle_between_each_digit_joint(landmarks, digit_indices [Digit.Ring.name]),
        Digit.Pinky.name: calculate_angle_between_each_digit_joint(landmarks,digit_indices  [Digit.Pinky.name])
    }

def calculate_angle_between_digit_df(df, digit_name, coordinate_names):
    if( df is None):
        return None

    if( coordinate_names is None):
        coordinate_names = default_coordinate_names()

    # Get the column names for the digit
    names = get_all_fingers_indices(True,True)[digit_name]

    # Make all the vectors
    dnames = []
    for i in range(0,len(names)-1):
        name = get_landmark_name(names[i])
        name2 = get_landmark_name(names[i+1])

        d = "d_"+name+"_"+name2
        dnames.append(d)
        df[d] = df.apply(lambda row: np.subtract(row[name], row[name2]), axis=1)

    # Calculate the angle between each vector
    angles = []
    for i in range(0,len(dnames)-1):
        df['ROM_'+name+"_"+name2] = df.apply(lambda x: calculate_angle(x[dnames[i]], x[dnames[i+1]]), axis=1)

    # Drop all vector columns
    for d in dnames:
        df.drop(columns=[d], inplace=True)

    return df

def calculate_all_finger_angle_df(df, digit_names, coordinate_names):
    if( df is None):
        return None

    # For each digit, calculate the vector between each adjacvent point
    # Then calculate the angle between each vector

    digits = get_all_fingers_indices(True, True)
    for digit, name in digit_names.items():
        df =  calculate_angle_between_digit_df(df, name, coordinate_names)

    return df


def calculate_all_finger_angles_csv(df, digit_names, coordinate_names):
    if( df is None):
        return None

    if( digit_names is None):
        digit_names = get_all_landmark_names()

    if( coordinate_names is None):
        coordinate_names = default_coordinate_names()

    # For each digit, calculate the vector between each adjacvent point
    # Then calculate the angle between each vector
    finger_angles = {}
    for digit, names in digit_names.items():
        xlabel =  names[0] + coordinate_names["X"]
        ylabel =  names[0] + coordinate_names["Y"]
        zlabel =  names[0] + coordinate_names["Z"]

        #Get the Next Column


def convert_csv_with_xyz_to_landmarks(df):

    df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')),errors='ignore')]
    df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')),errors='ignore')]

    landmarks = []
    for i in range(0,21):
        name = get_landmark_name(i)
        print(name)
        if( f"{name} (X-coordinate)" not in df.columns) or (f"{name} (Y-coordinate)" not in df.columns) or ( f"{name} (Z-coordinate)" not in df.columns):
            print(f"Missing Landmark: {name}")
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

    # Load the CSV file
    df = pd.read_csv(args['filename'],sep="\t")
    if( df is None):
        logging.error(f"Failed to load file: {args['filename']}")
        return

    df = convert_csv_with_xyz_to_landmarks(df)

    # Calculate the digit lengths
    digit_lengths = calculate_all_digit_lengths(df, None )
    logging.info(f"Digit Lengths: {digit_lengths}")

    ldf = pd.DataFrame.from_dict(digit_lengths)
    ldf.to_csv("./output/Digit_Lengths.csv", index=False)

    # Calculate the angles between each joint
    fangles = calculate_angle_between_digit_df(df, Digit.Index.name, None)

    # # Calculate the angles between each joint
    # finger_angles = calculate_all_finger_angles(landmarks, True)
    # Save the results to a new CSV file
    # adf = pd.DataFrame.from_dict(finger_angles, orient='index')
    fangles.to_csv("./output/Digit_ROM.csv", index=False)

    # #Combine the two dataframes
    # combined_df = pd.concat([ldf, adf], axis=1)
    # combined_df.to_csv("./output/Combined.csv", index=False)


#********************************************************************************************************************
# Test Section
def test_csv_to_landmarks():

    # Create a dummy dataframe, with columns matching the hand headers listed at the top of the file.
    # The values are arbitrary, but should be consistent with the expected values.
    titles = [ "Wrist (X-coordinate)","Wrist (Y-coordinate)","Wrist (Z-coordinate)",
"Thumb CMC (X-coordinate)","Thumb CMC (Y-coordinate)","Thumb CMC (Z-coordinate)",
"Thumb MCP (X-coordinate)","Thumb MCP (Y-coordinate)","Thumb MCP (Z-coordinate)",
"Thumb IP (X-coordinate)","Thumb IP (Y-coordinate)","Thumb IP (Z-coordinate)",
"Thumb Tip (X-coordinate)","Thumb Tip (Y-coordinate)","Thumb Tip (Z-coordinate)",
"Index Finger MCP (X-coordinate)","Index Finger MCP (Y-coordinate)","Index Finger MCP (Z-coordinate)",
"Index Finger PIP (X-coordinate)","Index Finger PIP (Y-coordinate)","Index Finger PIP (Z-coordinate)",
"Index Finger DIP (X-coordinate)","Index Finger DIP (Y-coordinate)","Index Finger DIP (Z-coordinate)",
"Index Finger Tip (X-coordinate)","Index Finger Tip (Y-coordinate)","Index Finger Tip (Z-coordinate)",
"Middle Finger MCP (X-coordinate)","Middle Finger MCP (Y-coordinate)","Middle Finger MCP (Z-coordinate)",
"Middle Finger PIP (X-coordinate)","Middle Finger PIP (Y-coordinate)","Middle Finger PIP (Z-coordinate)",
"Middle Finger DIP (X-coordinate)","Middle Finger DIP (Y-coordinate)","Middle Finger DIP (Z-coordinate)",
"Middle Finger Tip (X-coordinate)","Middle Finger Tip (Y-coordinate)","Middle Finger Tip (Z-coordinate)",
"Ring Finger MCP (X-coordinate)","Ring Finger MCP (Y-coordinate)","Ring Finger MCP (Z-coordinate)",
"Ring Finger PIP (X-coordinate)","Ring Finger PIP (Y-coordinate)","Ring Finger PIP (Z-coordinate)",
"Ring Finger DIP (X-coordinate)","Ring Finger DIP (Y-coordinate)","Ring Finger DIP (Z-coordinate)",
"Ring Finger Tip (X-coordinate)","Ring Finger Tip (Y-coordinate)","Ring Finger Tip (Z-coordinate)",
"Pinky MCP (X-coordinate)","Pinky MCP (Y-coordinate)","Pinky MCP (Z-coordinate)",
"Pinky PIP (X-coordinate)","Pinky PIP (Y-coordinate)","Pinky PIP (Z-coordinate)",
"Pinky DIP (X-coordinate)","Pinky DIP (Y-coordinate)","Pinky DIP (Z-coordinate)",
"Pinky Tip (X-coordinate)","Pinky Tip (Y-coordinate)","Pinky Tip (Z-coordinate)"]

    # Create an increasing array from 1 to the length of titles
    values = list(range(1,len(titles)+1))

    # Create a dataframe with the titles and values
    df = pd.DataFrame([values], columns=titles)

    # Print the dataframe titles and values
    print(df.columns)
    print(df.values)

    # Convert the dataframe to landmarks
    landmarks = convert_csv_to_landmarks(df)

    # Print the landmarks
    print(landmarks)

#********************************************************************************************************************
# Test Section

# write a couple of tests for the functions
# Test the calculate_length function
# Test the calculate_digit_length function

# Test Main
def test_main():
    ap = setup_arguments()
    args = vars(ap.parse_args())
    set_log_level(args['log'])
    logging.info("Starting TEST Program")

    # run the test functions with print statements
    test_calculate_length()
    #test_calculate_digit_length()
    #test_calculate_all_digit_lengths()

    test_calculate_full_digit_range_df()

    test_calculate_digit_range()
    test_calculate_full_range()

    print("All Done")

# Write a test function for calculate_full_range_df(...)
def test_calculate_full_digit_range_df():
    # Create a dummy dataframe, with columns matching the hand headers listed at the top of the file.
    # The values are arbitrary, but should be consistent with the expected values.
    titles = [ "Thumb Tip","Index Tip","Middle Tip"]
    df = pd.DataFrame([[[0,0,0],[1,1,1],[2,2,2]], [[1,1,1],[2,2,2],[3,3,3]], [[2,0,2],[0,3,0],[0,0,4]]], columns=titles)
    print(df)
    res = calculate_full_digit_range_df(df, "Thumb Tip")
    print(res)

    # Res in an array of two dictionaries
    # The first dictionary contains the min values for each coordinate
    # The second dictionary contains the max values for each coordinate
    assert res[0]["Thumb Tip_x_min"] == [0,0,0]
    assert res[0]["Thumb Tip_y_min"] == [0,0,0]
    assert res[0]["Thumb Tip_z_min"] == [0,0,0]
    assert res[1]["Thumb Tip_x_max"] == [2,0,2]
    assert res[1]["Thumb Tip_y_max"] == [1,1,1]
    assert res[1]["Thumb Tip_z_max"] == [2,0,2]
    



def test_calculate_length():
    p1 = [0,0,0]
    p2 = [1,1,1]
    assert calculate_length(p1,p2) == math.sqrt(3)

def test_calculate_digit_length():
    df = pd.DataFrame({
        "Thumb Tip (X-coordinate)":0,
        "Thumb Tip (Y-coordinate)":0,
        "Thumb Tip (Z-coordinate)":0,
        "Thumb CMC (X-coordinate)":1,
        "Thumb CMC (Y-coordinate)":1,
        "Thumb CMC (Z-coordinate)":1
    },index=[0])
    assert calculate_digit_length(df, "Thumb Tip (X-coordinate)", "Thumb CMC (X-coordinate)") == 1

def test_calculate_all_digit_lengths():
    df = pd.DataFrame({
        "Thumb Tip (X-coordinate)": 0,
        "Thumb Tip (Y-coordinate)": 0,
        "Thumb Tip (Z-coordinate)": 0,
        "Thumb CMC (X-coordinate)": 1,
        "Thumb CMC (Y-coordinate)": 1,
        "Thumb CMC (Z-coordinate)": 1,
        "Index Finger Tip (X-coordinate)": 0,
        "Index Finger Tip (Y-coordinate)": 0,
        "Index Finger Tip (Z-coordinate)": 0,
        "Index Finger MCP (X-coordinate)": 1,
        "Index Finger MCP (Y-coordinate)": 1,
        "Index Finger MCP (Z-coordinate)": 1,

    }, index=[0])
    digit_lengths = calculate_all_digit_lengths(df, None)
    assert digit_lengths["Thumb_Length"] == 1
    assert digit_lengths["Index_Length"] == 1


def test_calculate_digit_range():
    df = pd.DataFrame({
        "Thumb Tip (X-coordinate)": 0,
        "Thumb Tip (Y-coordinate)": 0,
        "Thumb Tip (Z-coordinate)": 0,
        "Thumb CMC (X-coordinate)": 1,
        "Thumb CMC (Y-coordinate)": 1,
        "Thumb CMC (Z-coordinate)": 1
    },index=[0])
    assert calculate_digit_range(df, "Thumb Tip (X-coordinate)") == [0,0]

def test_calculate_full_range():
    df = pd.DataFrame([{
        "Thumb Tip (X-coordinate)": 0,
        "Thumb Tip (Y-coordinate)": 0,
        "Thumb Tip (Z-coordinate)": 0,
        "Thumb CMC (X-coordinate)": 0,
        "Thumb CMC (Y-coordinate)": 0,
        "Thumb CMC (Z-coordinate)": 0,
        "Index Finger Tip (X-coordinate)": 0,
        "Index Finger Tip (Y-coordinate)": 0,
        "Index Finger Tip (Z-coordinate)": 0,
        "Index Finger MCP (X-coordinate)": 0,
        "Index Finger MCP (Y-coordinate)": 0,
        "Index Finger MCP (Z-coordinate)": 0
    },  {

        "Thumb Tip (X-coordinate)": 1,
        "Thumb Tip (Y-coordinate)": 1,
        "Thumb Tip (Z-coordinate)": 1,
        "Thumb CMC (X-coordinate)": 1,
        "Thumb CMC (Y-coordinate)": 1,
        "Thumb CMC (Z-coordinate)": 1,
        "Index Finger Tip (X-coordinate)": 1,
        "Index Finger Tip (Y-coordinate)": 1,
        "Index Finger Tip (Z-coordinate)": 1,
        "Index Finger MCP (X-coordinate)": 1,
        "Index Finger MCP (Y-coordinate)": 1,
        "Index Finger MCP (Z-coordinate)": 1
    }])

    digit_ranges = calculate_full_range(df, None, None)
    assert digit_ranges["Thumb Tip (X-coordinate)_Range_Min"] == 0
    assert digit_ranges["Thumb Tip (X-coordinate)_Range_Max"] == 1
    assert digit_ranges["Index Finger Tip (X-coordinate)_Range_Min"] == 0
    assert digit_ranges["Index Finger Tip (X-coordinate)_Range_Max"] == 1


def test_thumb_angles(landmarks, expected_angles): 
    
    angles = calculate_angle_between_each_digit_joint(landmarks, [0,1,2,3,4])
    
    #Check that the calculates angles are within accepted error range.
    for i in range(len(expected_angles)):
        print(f"i = {i}  Expected: {expected_angles[i]}, Calculated: {angles[i]}")
        assert abs(angles[i] - expected_angles[i]) < 1e-5, f"Expected {expected_angles}, but got {angles}"    
    
    print("All tests pass")

def test_thumb_angles_all_zero():
    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 1, 1],  # Thumb CMC
        [2, 2, 2],  # Thumb MCP
        [3, 3, 3],  # Thumb IP
        [4, 4, 4]  # Thumb Tip
    ]
    expected_angles = [0, 0, 0]
    test_thumb_angles(landmarks, expected_angles)

def test_thumb_angles_all_45():
    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 0, 0],  # Thumb CMC
        [2, 1, 0],  # Thumb MCP
        [3, 1, 1],  # Thumb IP
        [4, 2, 1]  # Thumb Tip
    ]
    expected_angles = [45, 60, 60]
    test_thumb_angles(landmarks, expected_angles)


def test_calculate_all_finger_angles():
    landmarks = [
        [0, 0, 0],  # Wrist
        [1, 0, 0],  # Thumb CMC
        [2, 1, 0],  # Thumb MCP
        [3, 1, 1],  # Thumb IP
        [4, 2, 1],  # Thumb Tip
        [5, 0, 0],  # Index Finger MCP
        [6, 1, 0],  # Index Finger PIP
        [7, 1, 1],  # Index Finger DIP
        [8, 2, 1],  # Index Finger Tip
        [9, 0, 0],  # Middle Finger MCP
        [10, 1, 0],  # Middle Finger PIP
        [11, 1, 1],  # Middle Finger DIP
        [12, 2, 1],  # Middle Finger Tip
        [13, 0, 0],  # Ring Finger MCP
        [14, 1, 0],  # Ring Finger PIP
        [15, 1, 1],  # Ring Finger DIP
        [16, 2, 1],  # Ring Finger Tip
        [17, 0, 0],  # Pinky MCP
        [18, 1, 0],  # Pinky PIP
        [19, 1, 1],  # Pinky DIP
        [20, 2, 1]  # Pinky Tip
    ]
    expected_angles = {
        Digit.Thumb.name: [45, 60, 60],
        Digit.Index.name: [45, 60, 60],
        Digit.Middle.name: [45, 60, 60],
        Digit.Ring.name: [45, 60, 60],
        Digit.Pinky.name: [45, 60, 60]
    }
    angles = calculate_all_finger_angles(landmarks,True,False)
    for digit in angles:
        for i in range(len(angles[digit])):
            assert abs(angles[digit][i] - expected_angles[digit][i]) < 1e-5, f"Expected {expected_angles}, but got {angles}"
    print("All tests pass")




# ********************************************************************************************************************
if __name__ == "__main__":
    main()
    #test_csv_to_landmarks()
    #test_main()