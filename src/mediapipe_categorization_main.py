import os
import argparse
from pathlib import Path

import cv2
import math

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


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

def main():
    # STEP 1: Parse the command-line arguments.
    parser = argparse.ArgumentParser(description="Create a video from a single image.")    
    parser.add_argument("-i","--input_image", type=str, default="./media/Images/face-hand-orange.jpg", help="Path to the input video file.")        
    parser.add_argument("-s","--suggest", action="store_false", help="name of the output video.")
    parser.add_argument("-p","--print_all", action="store_true", help="Print all gestures and scores")    

    args = vars(parser.parse_args())
    #print(args)

    # STEP 2: Create an GestureRecognizer object.
    base_options = python.BaseOptions(model_asset_path='./models/gesture_recognizer.task')
    options = vision.GestureRecognizerOptions(base_options=base_options)
    recognizer = vision.GestureRecognizer.create_from_options(options)
    
    image_file = args['input_image']
    if( not os.path.exists(image_file)):
        print(f"Input video file not found: {image_file}")
        return

    # STEP 3: Load the input image.    
    image = mp.Image.create_from_file(image_file)

    # STEP 4: Recognize gestures in the input image.
    recognition_result = recognizer.recognize(image)

    # STEP 5: Process the result. In this case, visualize it.
    all_gestures = recognition_result.gestures
    if(len(all_gestures) == 0):
        print("No gestures found")
        return
    top_gesture = all_gestures[0][0]
    
    if(args['print_all']):
      for gesture in all_gestures:
          print(f"Gesture: {gesture[0].category_name} with score {gesture[0].score}")
    else:
       print(f"Top gesture: {top_gesture.category_name} with score {top_gesture.score}")

    # COPY THE image or video FILE TO A NEW NAME BASED ON THE GESTURE LABEL
    if(args['suggest']):        
        output_file =  setup_output_file(image_file, top_gesture.category_name)        
        print(f"{output_file}")
    else:
       print(f"{top_gesture.category_name}")
    
    
if __name__ == "__main__":
    main()
