# This program calculates the joint positions, angles, and lengths of the bones.
# It will take the mediapipe results and calculate the joint positions.
import os
import argparse
import logging
import math
from log import set_log_level
import pandas as pd

from convert_mediapipe_index import *

def setup_arguments():
    
    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="", help="Load a CSV file that has been evaluated by mediapipe to produce landmark information.")

    ap.add_argument("-o", "--out_filename", type=str, default="./output/out_data.csv", help="Save to a CSV file.")

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

#
def calculate_all_digit_lengths(df, digit_names, coordinate_names):

    if( df is None):
        return None
    
    if( digit_names is None):
        digit_names = default_digit_tip_names()

    if( coordinate_names is None):
        coordinate_names = default_coordinate_names()

    digit_lengths = {}
    for digit, names in digit_names.items():     
        x1 = names[0] + coordinate_names["X"]
        x2 = names[1] + coordinate_names["X"]
        digit_x_length = calculate_digit_length(df, names[0] + coordinate_names["X"], names[1]+ coordinate_names["X"])
        digit_y_length = calculate_digit_length(df, names[0] + coordinate_names["Y"], names[1]+ coordinate_names["Y"])
        digit_z_length = calculate_digit_length(df, names[0] + coordinate_names["Z"], names[1]+ coordinate_names["Z"])
                
        if digit_x_length is not None and digit_y_length is not None and digit_z_length is not None:
            digit_lengths[digit+"_Length"] = max(digit_x_length, digit_y_length, digit_z_length)

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


def calculate_all_finger_angles(landmarks, add_wrist_to_indices=False, add_wrist = False):
    digit_indices = get_all_fingers_indices(add_wrist_to_indices, add_wrist)
    return {
        Digit.Thumb.name: calculate_angle_between_each_digit_joint(landmarks, digit_indices [Digit.Thumb.name]),
        Digit.Index.name: calculate_angle_between_each_digit_joint(landmarks,digit_indices  [Digit.Index.name]),
        Digit.Middle.name: calculate_angle_between_each_digit_joint(landmarks, digit_indices[Digit.Middle.name]),
        Digit.Ring.name:  calculate_angle_between_each_digit_joint(landmarks, digit_indices [Digit.Ring.name]),
        Digit.Pinky.name: calculate_angle_between_each_digit_joint(landmarks,digit_indices  [Digit.Pinky.name])
    }

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
    df = pd.read_csv(args['filename'])
    if( df is None):
        logging.error(f"Failed to load file: {args['filename']}")
        return
    
    # Calculate the digit lengths
    digit_lengths = calculate_all_digit_lengths(df, None, None)
    logging.info(f"Digit Lengths: {digit_lengths}")

    # If using a CSV file, we need to extract the landmark positions
    landmarks = convert_csv_to_landmarks(df)

    # Calculate the ROM for each digit
    digit_ranges = calculate_full_range(df, None, None)

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
    test_calculate_digit_length()
    test_calculate_all_digit_lengths()
    test_calculate_digit_range()
    test_calculate_full_range()

    print("All Done")

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
    digit_lengths = calculate_all_digit_lengths(df, None, None)
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



# ********************************************************************************************************************
if __name__ == "__main__":
    #main()
    test_csv_to_landmarks()
    ##test_main()