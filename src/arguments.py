import argparse

# ******************************************* Arguments Sections
def setup_arguments():

    # Initialize the argument parser
    ap = argparse.ArgumentParser()

    ##################### Debugging arguments.
    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-s", "--show_visual", action="store_true", help="Show Windows with visual information.")

    # Add an option to load a video file instead of a camera.
    #../videos/tests/quick_flexion_test.mp4
    ap.add_argument("-f", "--filename", type=str, default="",
                    help="Load a video file instead of a camera.")

    # Add an option to load a video file instead of a camera.
    #../videos/tests/quick_flexion_test.mp4
    ap.add_argument("-cd", "--collect_data", action="store_true", 
                    help="Collect Data from the video file.")

    # Add an option to load a video file instead of a camera.
    #../videos/tests/quick_flexion_test.mp4
    ap.add_argument("-ci", "--camera_id", type=int, default=4,
                    help="Load camera from the Idx.")

    # Add the debug mode for more verbose output in terminal.
    # Add an option to run Mediapipe without additional processing.
    ap.add_argument("-n", "--media", action="store_true",
                    help="Run Mediapipe without additional processing.")

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
    ap.add_argument("-o", "--output", type=str, default="saved_frame_data.csv",
                     help="Output file name")    

    # Add an option to record the video
    ap.add_argument("-r", "--record", type=str, default="record_video.mp4",
                    help="Record the video only")

    # Add time to the output file argument
    ap.add_argument("-t", "--timestamp", action="store_true",
                    help="Output append time to file name")

    return ap

