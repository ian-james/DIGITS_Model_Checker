# This tool simply takes mediapipe files in multiple formats and makes sure they are not in normalized format.
# The csv file and the image or video file must match (or the proper resolution must be provided)
import argparse
import logging
from log import set_log_level

import cv2
import pandas as pd
import numpy as np

from VideoCap import VideoCap_Info
from file_utils import check_if_file_is_image

from convert_mediapipe_index import get_all_landmark_names

def setup_arguments():
    
    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="./output/mediapipe_data.csv", help="Load a CSV file that has been evaluated by mediapipe to produce landmark information.")

    ap.add_argument("-o", "--out_filename", type=str, default="./output/out_data.csv", help="Save to a CSV file.")

    ap.add_argument("-m", "--media", type=str, default="", help="Load an media file that has been evaluated by mediapipe to produce landmark information.")

    ap.add_argument("-w", "--width", type=int, default=640, help="Width of the media.")
    ap.add_argument("-r", "--height", type=int, default=480, help="Height of the media.")

    return ap

# Check if the series is normalized
def is_normalized(series):
    return series.max() <= 1 and series.min() >= 0

# Check if [x,y,z] array is normalized
def is_normalized_array(arr):
    return np.max(arr) <= 1 and np.min(arr) >= 0

def get_columns_with_str(df, str):
    return df.filter(regex='.*'+str)

#Z-coordinate is not normalized but included for completeness.
# Since we don't have a resolution for the actual depth of the image (or real distance)
def fix_normalized_mediapipe(df, resolution, xlbl="X-coordinate", ylbl="Y-coordinate",zlbl="Z-coordinate"):
    """
    This function will fix the normalized mediapipe data.
    """ 
    #Use a regularly expression to find columns that contain the xlbl and ylbl
    xcols = get_columns_with_str(df, xlbl)
    ycols = get_columns_with_str(df, ylbl)

    if(not xcols.empty and not ycols.empty):
        # Check if the xcols are normalized, if not multiply by the resolution
        for col in xcols.columns:
            if is_normalized(df[col]):
                df[col] = df[col] * resolution[0]

        # Check if the ycols are normalized, if not multiply by the resolution
        for col in ycols.columns:
            if is_normalized_array(df[col]):
                print("Hello")
                
    else:
        # Might be in list mode
        # Get the names of the digits
        digit_names = get_all_landmark_names()
        for digit_name in digit_names:
            if( digit_name in df.columns):
                print(digit_name)
                if is_normalized(df[digit_name]):
                    df[digit_name] = df[digit_name] * resolution[0]                

    return df




def main():
      
    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
       
    resolution = (0,0)

    # Check for width or height, if both 0. Then we need to load the media file.
    if(args['width'] != 0 and args['height'] != 0):
        resolution = (args['width'], args['height'])
    else:
        if(args['media'] != ""):
             # Run the program for an images
            if(check_if_file_is_image(args['filename'])):
                #load the image and the resolution
                image = cv2.imread(args['media'])
                resolution = (image.shape[1], image.shape[0])
            else:
                try:
                    with VideoCap_Info.with_no_info() as cap:
                        cap.setup_capture(args['filename'])
                        resolution = (cap.get_width(), cap.get_height())                
                except Exception as e:
                    logging.error(f"Failed to read video or camera options. {e}")
                    return
            
        else:
            logging.error("No Media file provided.")
            return
    
    # Load the CSV if it exists
    df = None
    if(args['filename'] != ""):        
        try:
            df = pd.read_csv(args['filename'],sep="\t")        
        except IOError as e:
            logging.error(f"Failed to load the CSV file. {e}")
            return
    else:
        logging.error("No CSV file provided.")
        return

    df = fix_normalized_mediapipe(df,resolution)  
  

    # Save the dataframe to a new file
    df.to_csv(args['out_filename'], index=False)

if __name__ == '__main__':
    main()