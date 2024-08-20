#import libraries
import os
import argparse
import logging
import cv2
from enum import Enum, auto

import mediapipe as mp
import pandas as pd

from fps_timer import FPS
from VideoCap import VideoCap_Info, WindowName
from VideoCap import find_camera
from DisplayCap import DisplayCap_Info

from arguments import setup_arguments
from log import set_log_level
from mediapipe_helpers import *

from convert_mediapipe_index import convert_all_columns_to_friendly_name
from file_utils import change_extension,create_directory, check_if_file_is_image

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

def process_frame_from_image(image_file, frames_to_run, frame_processor, show_visual=True):

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

    for i in range(frames_to_run):

        image = cv2.imread(image_file)
        if image is None:
            logging.error(f"Failed to read image: {image_file}")
            return

        fps.update()

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
            

    frame_processor.finalize_dataframe()


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

    if(show_visual):
        cv2.destroyAllWindows()

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


# Helper function
def write_model_results_to_csv(filename, df, collected_data = True, use_friendly_names = True, remove_non_position_columns = True   ):
    if( df is None or len(df) == 0 or df.shape[0] == 0):
        if(collected_data):
            logging.info("Data was unable to saved on the output file.")
        else:
            logging.info("Data was not collected.")
    else:
        print(f"Writting to output file: {filename} with {df.shape[0]} records.")

        if(use_friendly_names):
              # Convert the columns to user-friendly names
            df.columns = convert_all_columns_to_friendly_name(df, [])

        if( remove_non_position_columns):
            # Remove the presence and visibility columns
            df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')),errors='ignore')]
            df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')),errors='ignore')]

        print(filename)
        df.to_csv(filename, index=False, header=True, sep="\t")


def make_output_filename(args):
    # Combine the output directory, the input filename, and the parameters to create a new filename.
    # This will be used to save the results of the model.

    output_file = os.path.basename(args['filename']).split(".")[0]
    output_file = f"{output_file}_{args['model']}_nh_{args['num_hands']}_md_{args['min_detection_confidence']}_mt_{args['min_tracking_confidence']}_mp_{args['min_presense_confidence']}.csv"

    return output_file

def image_main(args, frame_processor):
    process_frame_from_image(args['filename'], args["extend_frames"], frame_processor, args['show_visual'])
    df = frame_processor.get_dataframe()

def main():

    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())

    # Setup Logging
    set_log_level(args['log'])
    logging.info("Starting Program")
    logging.info(f"Setup Arguments: {args}")

    # Testing Arguments
    # args['filename'] = "./media/Videos/hands_944_video.mp4"
    # args['collect_data'] = True
    # args['camera_id'] = 4

    frame_processor = None
    # Setup Frame Processor
    if(args['model'] == "mediapipe"):
        hand_model = get_hand_model(*get_mediapipe_settings(args))
        frame_processor = FrameProcessor(hand_model)
        #frame_processor = HandROM_Thumb_Wrapper(hand_model)
    else:
        logging.error("Model not supported.")
        
        return

    # Run the program for an images
    if(check_if_file_is_image(args['filename'])):
        image_main(args, frame_processor)
    else:
        try:
            with VideoCap_Info.with_no_info() as cap:
                cap.setup_capture(args['filename'])
                process_frame_from_cap(cap, frame_processor, args['show_visual'])

            # Create the directory and change the files names.
            create_directory(args['output'])
            args['output'] = os.path.join(args['output'],make_output_filename(args))

            if(args['output'] == ""):        
                # set the output file to the same name as the input file with csv extension.
                args['output'] = change_extension( os.path.join("output/",make_output_filename(args)))

            frame_processor.save_model_data(args)
        except Exception as e:
            logging.error(f"Failed to read video or camera options. {e}")
    logging.info("End of Program")

if __name__ == '__main__':
    main()
