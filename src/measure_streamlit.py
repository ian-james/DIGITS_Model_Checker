import os
import streamlit as st
import time
import logging
from log import set_log_level
import argparse

import cv2
from fps_timer import FPS
from VideoCap import WindowName

from mediapipe_helpers import *

import project_settings as ps

def get_mediapipe_settings(args):
    return [
        args['num_hands'],
        args['min_detection_confidence'],
        args['min_presense_confidence'],
        args['min_tracking_confidence']
    ]

def setup_arguments():

    # Initialize the argument parserhttp://localhost:2000
    ap = argparse.ArgumentParser()

    ##################### Debugging arguments.
    ap.add_argument("-l", "--log", type=str, default="info", help="Set the logging level. (debug, info, warning, error, critical)")

    ap.add_argument("-s", "--show_visual", action="store_false", help="Show Windows with visual information.")

    # Add an option to load a video file instead of a camera.
    # Default to 0 for the camera.
    #../videos/tests/quick_flexion_test.mp4
    ap.add_argument("-i", "--input_file", type=str, default="./media/Output/test_in.png", help="Load an image file")

    # Add an option to name the output directory
    ap.add_argument("-o", "--output_file", type=str, default="./media/Output/test.png", help="Name of the output file.")


    # Add an option to select the model to use.
    ap.add_argument("-m", "--model", type=str, default="mediapipe", help="Select the model to use. (mediapipe, yolo)")

    # Add an option to select the number of hands to detect.
    ap.add_argument("-nh", "--num_hands", type=int, default=1, help="Number of hands to detect.")

    # Add an option to set the minimum detection confidence.
    ap.add_argument("-md", "--min_detection_confidence", type=float, default=0.5, help="Minimum detection confidence.")

    # Add an option to set the minimum tracking confidence.
    ap.add_argument("-mt", "--min_tracking_confidence", type=float, default=0.5, help="Minimum tracking confidence.")

    # Add an option to set the minimum presense confidence.
    ap.add_argument("-mp", "--min_presense_confidence", type=float, default=0.5, help="Minimum presense confidence.")

    # Calculate distance between landmarks (hands only)
    ap.add_argument("-d","--distance", action="store_false", help="Calculate the distance between the hand landmarks.")

    return ap

def process_frame_from_image(image_file, frames_to_run, frame_processor, show_visual=True):

    image = cv2.imread(image_file)
    limage = image
    
    fps = FPS()
    fps.start()

    for i in range(frames_to_run):
        
        if image is None:
            logging.error(f"Failed to read image: {image_file}")
            return

        fps.update()

        # Convert the image to RGB
        bimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the frame and calculate the distances
        results, rimage = frame_processor.process_frame(bimage,fps.total_num_frames)

        # Convert to BGR for display
        limage = cv2.cvtColor(rimage, cv2.COLOR_RGB2BGR)            

    # Finalize the Dataframe for processing and display
    frame_processor.finalize_dataframe()
    return limage


def change_filename(filepath, new_filename):
    # Get the directory and extension of the original file
    directory = os.path.dirname(filepath)
    extension = os.path.splitext(filepath)[1]

    # Create the new file path with the updated filename
    new_filepath = os.path.join(directory, new_filename + extension)
    return new_filepath


# Define a main that collects an image and analyzes it
def main():

    project_directory = ps.DIR_IMAGES

     # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
    logging.info(f"Setup Arguments: {args}")

    st.title("Measure Streamlit")
    st.write("This is a simple Streamlit app to demonstrate how to measure the performance of your Streamlit app.")
    st.write("The app will load an image and analyze it using the MediaPipe Hands model.")

    # Create a sidebar with options for logging level and number of hands
    st.sidebar.title("Options")    
    log_level = st.sidebar.selectbox("Select a logging level", ["debug", "info", "warning", "error", "critical"])
    args['log'] = log_level
    
    st.sidebar.subheader("MediaPipe Hands Settings")    
    num_hands = st.sidebar.slider("Number of hands", 1, 2)
    min_detection_confidence = st.sidebar.slider("Minimum detection confidence", 0.0, 1.0, 0.5)
    min_presense_confidence = st.sidebar.slider("Minimum presense confidence", 0.0, 1.0, 0.5)
    min_tracking_confidence = st.sidebar.slider("Minimum tracking confidence", 0.0, 1.0, 0.5)
    # Put a number box for the proximal phalanx size
    proximal_phalanx_size = st.sidebar.number_input("Proximal Phalanx Size in cm", value=3.0)

    
   
    # Load the image
    image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image:
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Analyze the image
        st.write("Analyzing the image...")

        frame_processor = None
        # Setup Frame Processor
        if(args['model'] == "mediapipe"):
            hand_model = get_hand_model(num_hands, min_detection_confidence, min_presense_confidence, min_tracking_confidence)
            frame_processor = HandDistance_Wrapper(hand_model, proximal_phalanx_size)

        # Measure the time taken to process the image
        start_time = time.time()

        # Process the image
        enhanced_image = process_frame_from_image( change_filename(project_directory,image.name), 1, frame_processor)

        end_time = time.time()
        process_time = end_time - start_time

        st.write(f"Time taken to process the image: {process_time:.2f} seconds")

        # Display the image with the hand landmarks drawn on it
        st.write("Displaying the image with the hand landmarks...")

        # Measure the time taken to display the image
        start_time = time.time()

        if(enhanced_image is not None):
            
            # Display the image with the hand landmarks
            try:
                st.image(enhanced_image, caption="Image with Hand Landmarks", use_column_width=True)
            except Exception as e:
                st.write(f"Error displaying the image: {e}")

            try:
                bimage = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)

                st.image(bimage, caption="Image with Hand Landmarks", use_column_width=True)
            except Exception as e:
                st.write(f"Error displaying theb image: {e}")   


        end_time = time.time()
        display_time = end_time - start_time

        st.write(f"Time taken to display the image: {display_time:.2f} seconds")

        # Display the landmarks in a table
        st.write("Displaying the hand landmarks...")

        st.write(frame_processor.get_dataframe())

        st.write("Displaying the hand distances...")
        st.write(frame_processor.get_distances())


        st.write("Analysis complete.")


if __name__ == '__main__':
    main()