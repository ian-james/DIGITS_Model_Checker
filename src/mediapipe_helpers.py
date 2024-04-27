import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from datetime import datetime, timezone

import cv2

def get_hand_model(num_hands=2, min_detection_confidence=0.5, min_presence_confidence=0.5, min_tracking_confidence=0.5):
    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, #vision.RunningMode.VIDEO,vision.RunningMode.LIVE_STREAM
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    vision.HandLandmarkerOptions(base_options=base_options,)               
    detector = vision.HandLandmarker.create_from_options(options)
    return detector

def setup_frame_dictionary(frame_id):
    return {'time': frame_id , 'timestamp': datetime.now(timezone.utc).isoformat()}        

def mediapipe_results_to_dataframe(mediapipe_results, existing_df=None):
    """
    Converts MediaPipe detector results into a pandas DataFrame or appends to an existing DataFrame.

    :param mediapipe_results: The results from a MediaPipe detector.
    :param existing_df: An existing pandas DataFrame to append to. If None, a new DataFrame is created.
    :return: A pandas DataFrame containing the MediaPipe results.
    """
    # Extract data from mediapipe_results. This will depend on the type of MediaPipe detector used.
    # For example, if you're using pose detection, you might extract the x, y coordinates and visibility of each landmark.
    # Here, I'll assume a simple case where each result is a list of (x, y) tuples for each landmark.

    data = []
    for result in mediapipe_results:
        # Flatten the result into a single list - this might change depending on your data structure.
        flattened_result = [coordinate for landmark in result for coordinate in (landmark.x, landmark.y)]
        data.append(flattened_result)

    # Convert the list of results to a DataFrame
    df = pd.DataFrame(data)

    # If an existing DataFrame is provided, append to it
    if existing_df is not None:
        df = existing_df.append(df, ignore_index=True)

    return df

def hand_landmarks_to_dataframe(detection_result,frame_id):
    """
    Converts MediaPipe Hand Landmarks and Handedness into a pandas DataFrame.

    :param hand_landmarks_list: List of hand landmarks from MediaPipe detection.
    :param handedness_list: List of handedness information corresponding to each hand.
    :return: A pandas DataFrame with landmarks and handedness.
    """
    if( detection_result is None):
        return pd.DataFrame()
    
    if( detection_result.hand_landmarks is None):
        return pd.DataFrame()

    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness    
    # Initialize a list to hold all the rows for the DataFrame
    rows = []

    if( len(hand_landmarks_list) == 0):
        return None
    
    if( len(handedness_list) == 0 ):
        return None

    
    # Iterate over hand landmarks and handedness
    for idx, hand_landmarks in enumerate(hand_landmarks_list):
        hand_landmarks = hand_landmarks_list[idx]        
        handedness = handedness_list[idx]
        
        row = {'time': frame_id , 'timestamp': datetime.now(timezone.utc).isoformat(),'handedness': handedness[0].category_name}        
      
        # Add each landmark's coordinates to the row
        for i, landmark in enumerate(hand_landmarks):
            
            n_location = landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)

            row[f'x_{i}'] = landmark.x
            row[f'y_{i}'] = landmark.y
            row[f'z_{i}'] = landmark.z
            row[f'presence_{i}'] = landmark.presence
            row[f'visibility_{i}'] = landmark.visibility

        rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(rows)
    return df



def draw_landmarks_on_image(rgb_image, detection_result):
  MARGIN = 10  # Setting an arbitrary margin for easier visualization.
  FONT_SIZE = 1
  FONT_THICKNESS = 1
  HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


class FrameProcessor:
    def __init__(self, detector):
        """
        Initialize the FrameProcessor with a detector function.

        :param detector: A function that takes an image frame and returns data about the frame.
        """
        self.detector = detector
        # Initialize an empty DataFrame to store data about each frame
        self.data = []
        self.enable_empty_frame_collection = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup code if needed
        pass

    def process_frame(self, frame, frame_id):
        """
        Process an image frame using the detector and store the results in the dataframe.

        :param frame: The image frame to process.
        :param frame_id: An identifier for the frame.
        """
        # Use the detector to process the frame
        try:
            converted_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)            
            detection_results = self.detector.detect(converted_frame)

            if detection_results.hand_landmarks != []:
                # STEP 5: Process the classification result. In this case, visualize it.
                enhanced_frame = draw_landmarks_on_image(converted_frame.numpy_view(), detection_results)                
                frame_data = hand_landmarks_to_dataframe(detection_results,frame_id)
                self.data.append(frame_data)
                return detection_results, enhanced_frame
                
            elif( self.enable_empty_frame_collection):            
                frame_data = setup_frame_dictionary(frame_id)
                self.data.append(frame_data)            

            return detection_results, converted_frame
            
                  
        except Exception as e:            
            print(f"An error occurred while processing the frame: {e}")            
        return None, frame, frame
        
    def finalize_dataframe(self):
        if(self.data):
          self.data = pd.concat(self.data, ignore_index= True)
        else:
           self.data = pd.DataFrame()
        
    def get_dataframe(self):
        return self.data