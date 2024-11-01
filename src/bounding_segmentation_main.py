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
from main import process_frame_from_cap

# Setup input arguments
def setup_arguments():
    parser = argparse.ArgumentParser(description="Extract frames from a video")

    # Logging level and visualization
    parser.add_argument("-l","--log", type=str, default="info", help="Logging level. Default is info.")
    parser.add_argument("-s","--show_visual", action="store_false", help="Show visualizations of the extracted frames.")

    # Model medaiapipe or Yolo  
    parser.add_argument("-m","--model", type=str, default="mediapipe", help="Model to use for segmentation. Default is mediapipe.")

    # Target to segment between face or hand    
    parser.add_argument("-t", "--target", type=str, default="hand", help="Target to segment. Default is hand.")
        
    # parser.add_argument("-i","--input_file", type=str, default="./media/Images/left_arm_above_head.png", help="Path to the input video file.")
    parser.add_argument("-i","--input_file", type=str, default="/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos_full/test/64-R-3.mov", help="Path to the input video file.")
    parser.add_argument("-o","--frames_output_dir", type=str, default="./media/Output/", help="Directory to save the extracted frames.")

    # Frame extraction options
    parser.add_argument("-px","--xpixels", type=int, default=256, help="Size of the extracted frames in pixels. Default is 256 pixels.")
    parser.add_argument("-py","--ypixels", type=int, default=256, help="Size of the extracted frames in pixels. Default is 256 pixels.")

    
    # Fill boudning box with segmentation mask
    parser.add_argument("-f","--fill", action="store_false", help="Fill the bounding box with the segmentation mask.")
    parser.add_argument("-c","--color", type=str, default="red", help="Color of the bounding box and segmentation mask. Default is red.")

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


def process_face_hands_frame_from_cap(cap, face_processor, hand_processor, show_visual=True):

    # Setup Display
    if(show_visual):
        for window_name in WindowName:
            cv2.namedWindow(window_name.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name.name, 600,400)

        cv2.moveWindow(WindowName.ORIGINAL.name,0,0)
        cv2.moveWindow(WindowName.LANDMARKS.name,0,400)
        cv2.moveWindow(WindowName.ENHANCED.name,600,0)
        cv2.moveWindow(WindowName.STATUS.name,600,400)

    # Start the FPS timer   
    fps = FPS()
    fps.start()

    # While the Video has frames, get each frame and display the original
    # Process the frame with both the face and hand processors
    # Determine the overlay of the face and hand landmarks
    # Generate a new image with the overlay
    # Display the new image

    while cap.is_opened():

        success, image = cap.get_frame()
        fps.update()

        if not success:
            if(cap.is_video()):
                logging.info("Finished the video.")
                break
            else:
                logging.info("Ignoring empty camera frame.")
                continue

        if(show_visual):            
            cv2.imshow(WindowName.ORIGINAL.name, image)
        bimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if(show_visual):
            cv2.imshow(WindowName.ENHANCED.name, bimage)


        # As expected, if the blur occurs before detection, the hand detection will fail.
        face_detection_results = face_processor.process_detection_results(bimage)
        hand_detection_results = hand_processor.process_detection_results(bimage)

        # Compare the bounding box of the faces to the bounding box of the hand.

        dresults, himage = face_processor.process_image_drawing_detections(bimage, fps.total_num_frames, face_detection_results, hand_detection_results)
        hresults, rimage = hand_processor.process_image_drawing(himage, fps.total_num_frames, hand_detection_results)
        
        if(show_visual):
            cv2.imshow(WindowName.LANDMARKS.name, rimage)
        limage = cv2.cvtColor(rimage, cv2.COLOR_RGB2BGR)

        if(show_visual):
            cv2.imshow(WindowName.STATUS.name, limage)

        if(show_visual):
            key = cv2.waitKey(1)
            if key == ord('p'):
                cv2.waitKey(5000)
            elif key & 0xFF == 27:
                break
            elif key == ord('q'):
                break
            elif key == ord('p'):
                cv2.waitKey()

    face_processor.finalize_dataframe()
    hand_processor.finalize_dataframe()

    if(show_visual):
        cv2.destroyAllWindows()

def main():
    
    args = vars(setup_arguments().parse_args())

     # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
    logging.info(f"Setup Arguments: {args}")    

    frame_processor = None
    # Setup Frame Processor
    if(args['model'] == "mediapipe"):
        
        detector = get_face_model() 
        frame_processor = FaceDetector_Wrapper(detector=detector)

        hand_detector = get_hand_model()
        hand_processor = FrameProcessor(detector=hand_detector)

      # Run the program for an images
    if(check_if_file_is_image(args['input_file'])):
        process_frame_from_image(args['input_file'], frame_processor, args['show_visual'])        
    else:
        try:
            
            with VideoCap_Info.with_no_info() as cap:
                cap.setup_capture(args['input_file'])
                if( hand_detector is not None):
                    process_face_hands_frame_from_cap(cap, frame_processor, hand_processor, args['show_visual'])
                else:
                    process_frame_from_cap(cap, frame_processor, show_visual=True)

        except Exception as e:

            logging.error(f"Failed to read video or camera options. {e}")


if __name__ == "__main__":
    main()