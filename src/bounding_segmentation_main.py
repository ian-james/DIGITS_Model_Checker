# This Tool develops a bounding box and segmentation mask around the hand in a video. 
# The tool uses the MediaPipe Hands model to detect the hand landmarks and then uses the landmarks to create a bounding box and segmentation mask around the hand. 

import os
import argparse
import logging

from log import set_log_level
from mediapipe_helpers import *

from fps_timer import FPS
from VideoCap import VideoCap_Info, WindowName

from file_utils import check_if_file_is_image

# Setup input arguments
def setup_arguments():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("-m","--model", type=str, default="mediapipe", help="Model to use for segmentation. Default is mediapipe.")
    parser.add_argument("-l","--log", type=str, default="info", help="Logging level. Default is info.")    
    parser.add_argument("-i","--input_file", type=str, default="./media/Images/left_arm_above_head.png", help="Path to the input video file.")
    parser.add_argument("-o","--frames_output_dir", type=str, default="./media/Output/", help="Directory to save the extracted frames.")
    parser.add_argument("-px","--xpixels", type=int, default=256, help="Size of the extracted frames in pixels. Default is 256 pixels.")
    parser.add_argument("-py","--ypixels", type=int, default=256, help="Size of the extracted frames in pixels. Default is 256 pixels.")
    parser.add_argument("-s","--show_visual", action="store_false", help="Show visualizations of the extracted frames.")
    return parser


# Load a video file and extract frames
def process_frame_from_image(image_file, frame_processor, show_visual=True):

    # Setup Display
    if(show_visual):
        for window_name in WindowName:
            cv2.namedWindow(window_name.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name.name, 600,400)            

        cv2.moveWindow(WindowName.ORIGINAL.name,0,0)
        cv2.moveWindow(WindowName.LANDMARKS.name,0,400)
        cv2.moveWindow(WindowName.ENHANCED.name,600,0)
        cv2.moveWindow(WindowName.STATUS.name,600,400)    

    image = cv2.imread(image_file)
    if image is None:
        logging.error(f"Failed to read image: {image_file}")
        return

    if(show_visual):
        cv2.imshow(WindowName.ORIGINAL.name, image)
    bimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    if(show_visual):
        cv2.imshow(WindowName.ENHANCED.name, bimage)

    results, rimage = frame_processor.process_frame(bimage,1)
    if(show_visual):
        cv2.imshow(WindowName.LANDMARKS.name, rimage)
    limage = cv2.cvtColor(rimage, cv2.COLOR_RGB2BGR)

    if(show_visual):
        cv2.imshow(WindowName.STATUS.name, limage)

    if(show_visual):
        key = cv2.waitKey(1)

    frame_processor.finalize_dataframe()


def build_bounding_box(frame_processor):
    df = frame_processor.get_dataframe()
    # Get the minimum value from the all the dataframe columns marked with an trailing x    
    # Still have to select only those for X and Y
    x = df.select_dtypes(include=['float64']).min()
    mx = df.select_dtypes(include=['float64']).max()
    y = df.select_dtypes(include=['float64']).min()
    my = df.select_dtypes(include=['float64']).max()

    # Get the minimum value from the dataframe with columns marked with an trailing y
    boundingbox = {
        "x": df['x'].min(),
        "y": df['y'].min(),
        "width": df['x'].max() - df['x'].min(),
        "height": df['y'].max() - df['y'].min()
    
    }


def main():
    
    args = vars(setup_arguments().parse_args())

     # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
    logging.info(f"Setup Arguments: {args}")    

    frame_processor = None
    # Setup Frame Processor
    if(args['model'] == "mediapipe"):
        #detector = get_segmentation_model()
        #frame_processor = ImageSegmenter_Wrapper(detector)        
        detector = get_face_model() 
        frame_processor = FaceDetector_Wrapper(detector=detector)

      # Run the program for an images
    if(check_if_file_is_image(args['input_file'])):
        process_frame_from_image(args['input_file'], frame_processor, args['show_visual'])
        
    else:
        try:
            with VideoCap_Info.with_no_info() as cap:
                cap.setup_capture(args['input_file'])
               #process_frame_from_cap(cap, frame_processor)
        except Exception as e:
            logging.error(f"Failed to read video or camera options. {e}")


if __name__ == "__main__":
    main()