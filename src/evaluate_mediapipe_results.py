# This program takes a single csv file, that represents a mediaipipe landmarks results file.
# It then computes the statistics for each column in the file and outputs a new csv file with the statistics.
# It compares the Joint position movements of the wrist, pinky, ring, middle, index, and thumb.

# This should figure out the bone length and the different between key joint positions.
# For instance, along the same finger

# Lets say 100 frames for static image videos
# We can calculate the Bone1 length at each frame. So index length equals =5
# Does it flux in a meaning ful way

import argparse
import logging
from log import set_log_level


def setup_arguments():
    
    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-f", "--filename", type=str, default="", help="Load a CSV file that has been evaluated by mediapipe to produce landmark information.")

    ap.add_argument("-o", "--out_filename", type=str, default="./output/out_data.csv", help="Save to a CSV file.")

    return ap

def main():
      
    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")


if __name__ == "main":
    main()
