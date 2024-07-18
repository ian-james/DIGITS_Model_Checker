#import libraries
import os
import argparse
import logging
import cv2
from enum import Enum, auto

import mediapipe as mp
import pandas as pd

from VideoCap import VideoCap_Info

from log import set_log_level
from mediapipe_helpers import *


from file_utils import change_extension,create_directory, could_be_directory

from mediapipe_helpers import *

from calculate_joints_and_length import calculate_all_digit_lengths, calculate_angle_between_digit_df, convert_csv_with_xyz_to_landmarks

from fix_normalized_mediapipe import fix_normalized_mediapipe

from main import process_frame_from_cap, write_model_results_to_csv, make_output_filename

from convert_mediapipe_index import *

def get_mediapipe_settings(args):
    return [
        args['num_hands'],
        args['min_detection_confidence'],
        args['min_presense_confidence'],
        args['min_tracking_confidence']
    ]

def setup_arguments():

    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ##################### Debugging arguments.
    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-s", "--show_visual", action="store_true", help="Show Windows with visual information.")

    # Add an option to load a video file instead of a camera.
    # Default to 0 for the camera.
    #../videos/tests/quick_flexion_test.mp4
    ap.add_argument("-f", "--filename", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/small_sample_test/Videos/rgb_1_0000001_fps_30.mp4",
                    help="Load a video file instead of a camera.")



    # Add an option to load a video file instead of a camera.
    ap.add_argument("-fn", "--friendly_names", action="store_false", help="Use friendly names for the columns in the output file.")

    ap.add_argument("-r", "--resolution", action="store_false", help="Use Image Resolution for the output file.")

    ap.add_argument("-d","--remove_non_position_columns", action="store_false", help="Remove non position columns from the output file.")

    ap.add_argument("-u","--use_resolution", action="store_false", help="Use the resolution for the output file.")


      # Add an option to set the minimum detection confidence
    ap.add_argument('-md', '--min_detection_confidence', type=float, default=0.8,
                     help="Minimum detection confidence.")

    # Add an option to set the minimum tracking confidence
    ap.add_argument('-mt', '--min_tracking_confidence', type=float, default=0.8,
                     help="Minimum tracking confidence.")

    ap.add_argument("-mp", '--min_presense_confidence', type=float, default=0.5
                    , help="Minimum presence confidence.")

    # Add an option to set the number of hands to detect
    ap.add_argument('-nh', '--num_hands', type=int, default=2,
                     help="Number of hands to detect.")

    # Add an option to select a ML model (mediapipe or YOLO)
    ap.add_argument("-m", "--model", type=str, default="mediapipe",
                    help="Select the ML model to use. (mediapipe, yolo)")

    ##################### Output arguments.
    # Add the output file argument
    ap.add_argument("-o", "--output", type=str, default="",
                     help="Output file name")

    return ap



def write_results(df, args, resolution):
    if(args['filename'] == "" or args['filename'].isdigit()):
        args['filename'] = "saved_frame_data.png"

    if( could_be_directory(args['output']) ):
        create_directory(args['output'])
        args['output'] = os.path.join(args['output'],make_output_filename(args))

    if(args['output'] == ""):
        # set the output file to the same name as the input file with csv extension.
        args['output'] = change_extension( os.path.join("output/",make_output_filename(args)))

    write_model_results_to_csv(args['output'], df,True,True,True)
    logging.info("End of Program")

def main():
    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
    logging.info(f"Setup Arguments: {args}")

    # Testing args
    args['filename'] = "./media/Videos/Test_Clinical_Videos/CS-L-1-00.00.47.846-00.00.57.171-Opposition.mp4"

    # Testing workaround to not calcualte every time.
    
    df = None
    if not os.path.exists("./output/mediapipe_data_fixed.csv"):
        if os.path.exists("./output/mediapipe_data.csv"):
            df = pd.read_csv("./output/mediapipe_data.csv")
            res = (1920,1080)
        else:
            frame_processor = None
            # Setup Frame Processor
            if(args['model'] == "mediapipe"):
                hand_model = get_hand_model(*get_mediapipe_settings(args))
                frame_processor = FrameProcessor(hand_model)
            else:
                logging.error("Model not supported.")
                return

            try:
                with VideoCap_Info.with_no_info() as cap:
                    cap.setup_capture(args['filename'])
                    process_frame_from_cap(cap, frame_processor, args['show_visual'])
            except Exception as e:
                logging.error(f"Failed to read video or camera options. {e}")

            # Work with the Dataframe
            df = frame_processor.get_dataframe()
            df.to_csv("./output/mediapipe_data.csv")
        
            write_results(df, args, frame_processor.get_resolution())

            if(args['use_resolution']):            
                df = fix_normalized_mediapipe(df,frame_processor.get_resolution())
                df.to_csv("./output/mediapipe_data_fixed.csv")
    else:
        df = pd.read_csv("./output/mediapipe_data_fixed.csv")
    
    if(df is None):
        logging.error("Dataframe is None.")
        return
    
    # Check if the columns are split into x,y,z or contained as a single list in one
    if("Thumb Tip (X-coordinate)" in df.columns):
        df = convert_csv_with_xyz_to_landmarks(df)

    # Calculate Digit Length
    digit_lengths = calculate_all_digit_lengths(df, None )
    logging.info(f"Digit Lengths: {digit_lengths}")

    pd.DataFrame.from_dict(digit_lengths).to_csv("./output/digit_lengths.csv")

    # Calcualte the join ROM
    fangles = calculate_angle_between_digit_df(df, Digit.Index.name, None)
    fangles.to_csv("./output/finger_angles.csv")

    


if __name__ == '__main__':
    main()