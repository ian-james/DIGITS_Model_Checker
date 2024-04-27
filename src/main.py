#import libraries
import os
import argparse
import logging
import cv2
from enum import Enum, auto

import mediapipe as mp
import pandas as pd

from fps_timer import FPS
from VideoCap import VideoCap_Info
from VideoCap import find_camera
from DisplayCap import DisplayCap_Info

from arguments import setup_arguments
from log import set_log_level
from mediapipe_helpers import *

from ultralytics import YOLO

class WindowName(Enum):
    ORIGINAL = auto()
    LANDMARKS = auto()
    ENHANCED = auto()
    STATUS = auto()

def handle_keyboard():
    # Allow some keyboard actions
    # p-Pause
    # esc-exit
    key = cv2.waitKey(1)
    if key == ord('p'):
        cv2.waitKey(5000)
    elif key & 0xFF == 27:
        return False
    elif key == ord('q'):
        return False
    elif key == ord('p'):
        cv2.waitKey()
        
    return True

def process_frame_from_cap(cap, frame_processor, show_visual=True):

    # Setup Display
    if(show_visual):
        for window_name in WindowName:
            cv2.namedWindow(window_name.name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name.name, 600,400)

        cv2.moveWindow(WindowName.ORIGINAL.name,0,0)
        cv2.moveWindow(WindowName.LANDMARKS.name,0,400)
        cv2.moveWindow(WindowName.ENHANCED.name,600,0)
        cv2.moveWindow(WindowName.STATUS.name,600,400)

    fps = FPS()
    fps.start()

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

        results, rimage = frame_processor.process_frame(bimage,fps.total_num_frames)
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

    frame_processor.finalize_dataframe()
    

def load_yolo():
    model_name = "yolov8n"

    # Model Names
    #yolov8n-pose.pt
    #yolov8s-pose.pt
    #yolov8m-pose.pt
    #yolov8l-pose.pt yolov8x-pose.pt yolov8x-pose-p6.pt

    # Create a new YOLO Model
    model = YOLO(f"{model_name}.yaml")

    # Load the model with pre-trained weights
    model = YOLO(f"{model_name}.pt")

    return

def get_yolo_settings(args):
    return [
        args['num_hands'],
        args['min_detection_confidence'],
        args['min_presense_confidence'],
        args['min_tracking_confidence']
    ]


def get_mediapipe_settings(args):    
    return [
        args['num_hands'],
        args['min_detection_confidence'],
        args['min_presense_confidence'],
        args['min_tracking_confidence']
    ]


def main():

    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
    logging.info(f"Setup Arguments: {args}")

    args['filename'] = "./media/Videos/hands_944_video.mp4"
    args['collect_data'] = True
    args['camera_id'] = 4  

    print( os.getcwd())

    frame_processor = None
    # Setup Frame Processor
    if(args['model'] == "mediapipe"):
        hand_model = get_hand_model(*get_mediapipe_settings(args))
        frame_processor = FrameProcessor(hand_model)
    elif(args['model'] == "yolo"):
        frame_processor = FrameProcessor(load_yolo())
    else:
        logging.error("Model not supported.")
        return    
    
    try:
        with VideoCap_Info.with_no_info() as cap:
            cap.setup_capture(args['filename'])
            process_frame_from_cap(cap, frame_processor)

            df = frame_processor.get_dataframe()

            if( df is not None):                
                filename = args['output']
                print(f"Writting to output file: {filename} with {df.shape[0]} records.")
                df.to_csv(filename, index=False, header=True, sep="\t")
            else:
                if(args['collect_data']):
                    logging.info("Data was unable to saved on the output file.")
                else:
                    logging.info("Data was not collected.")            

    except Exception as e:
        logging.error(f"Failed to read video or camera options. {e}")
    finally:
        cv2.destroyAllWindows()

    logging.info("End of Program")

if __name__ == '__main__':
    main()
