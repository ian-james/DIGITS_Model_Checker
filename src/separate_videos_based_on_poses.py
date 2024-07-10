# Uses Mediapipe GestureRecognizer to separate videos based on poses
# Currently it does not separate the videos because the gesture recognizer is not reliable for most positions
# A new model could be re-trained with their ModelMaker tool
# Thus far videos are separated manually 
import os
import argparse
from pathlib import Path
import logging

import cv2
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from VideoCap import VideoCap_Info, WindowName

from fps_timer import FPS
from opencv_file_utils import open_recording_file

DESIRED_HEIGHT = 480
DESIRED_WIDTH = 480

def resize_and_show(image, width = DESIRED_WIDTH, height = DESIRED_HEIGHT):
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (width, math.floor(h/(w/width))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/height)), height))
  cv2.imshow(img)

def setup_output_file(input_file, gesture_name):
    file_path = Path(input_file)
    output_suffix = Path(input_file).suffix
    output_file = file_path.parent / f"{file_path.stem}_{gesture_name}{output_suffix}"
    return output_file

def separate_videos_based_on_poses(cap, frame_processor, output_video_path, show_visual=True):

    # Setup Display
    if(show_visual):
        cv2.namedWindow(WindowName.ORIGINAL.name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WindowName.ORIGINAL.name, 600,400)
        cv2.moveWindow(WindowName.ORIGINAL.name,0,0)

    frame_width = int(cap.width)
    frame_height = int(cap.height)
    output_video = open_recording_file(output_video_path, (frame_width, frame_height), cap.fps_rate)

    fps = FPS()
    fps.start()

    save_gestures = [ {
        "time":0,
        "gesture":"Start",
        "score":0
        }
    ]
    last_gesture = "Start"

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


        # Convert the image
        bimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)    
        mimage = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        # STEP 4: Recognize gestures in the input image.
        recognition_result = frame_processor.recognize(mimage)
        #recognition_result = frame_processor.recognize_for_videos(mimage,cap.get_fps())

        # STEP 5: Process the result. In this case, visualize it.
        all_gestures = recognition_result.gestures
        if(len(all_gestures) == 0):
            #print("No gestures found")         
            pass
        else:
            #for gesture in all_gestures:
            #    print(f"Gesture: {gesture[0].category_name} with score {gesture[0].score}")
            
            top_gesture = all_gestures[0][0]
            #print(f"Top gesture: {top_gesture.category_name} with score {top_gesture.score}")           

            if(top_gesture.category_name != last_gesture):
                print(f"Frame Number: {cap.frame_id}")            
                print(f"Gesture: {top_gesture.category_name} with score {top_gesture.score}")
                save_gestures.append({
                    "time":cap.frame_id,
                    "gesture":top_gesture.category_name,
                    "score":top_gesture.score
                })
                last_gesture = top_gesture.category_name

        cv2.imshow(WindowName.ORIGINAL.name, bimage)

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


    output_video.release()

    if(show_visual):
        cv2.destroyAllWindows()

def main():
    # STEP 1: Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Create a video from a single image.")
    parser.add_argument("-f","--filename", type=str, default="./media/Videos/Test_Clinical_Videos/CS-R-2.mov", help="Path to the input video file.")
    parser.add_argument("-v","--video_length", type=int, default=10, help="Store the video length in seconds.")
    parser.add_argument("-s","--suggest", action="store_false", help="Suggest timestamped positions and poses for evaluation.")
    parser.add_argument("-o","--output", type=str, default="./media/Videos/test.mp4", help="Default prefix for output files.")


    args = vars(parser.parse_args())
    print(args)

    video_file = args['filename']
    if( not os.path.exists(video_file)):
        print(f"Input video file not found: {video_file}")
        return

    setup_output_file = args['output']

    # STEP 2: Create an GestureRecognizer object.
    BaseOptions = mp.tasks.BaseOptions
    GestureRecognizer = mp.tasks.vision.GestureRecognizer
    GestureRecognizerOptions = mp.tasks.vision.GestureRecognizerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    base_options = python.BaseOptions(model_asset_path='./models/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    # options = GestureRecognizerOptions(
    # base_options=BaseOptions(model_asset_path='./models/gesture_recognizer.task'),
    # running_mode=VisionRunningMode.VIDEO)

    # STEP 3: Load the input image.
    try:
        with VideoCap_Info.with_no_info() as cap:
            cap.setup_capture(video_file)
            with GestureRecognizer.create_from_options(options) as recognizer:
                separate_videos_based_on_poses(cap, recognizer, setup_output_file, True)
    except Exception as e:
        logging.error(f"Failed to read video or camera options. {e}")



if __name__ == "__main__":
    main()
