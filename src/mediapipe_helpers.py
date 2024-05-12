import numpy as np
import pandas as pd

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from datetime import datetime, timezone
import math
from typing import List, Tuple, Union

import cv2

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):
    # TODO: Draw coordinates even if it's outside of the image bounds.
    return None
  x_px = min(math.floor(normalized_x * image_width), image_width - 1)
  y_px = min(math.floor(normalized_y * image_height), image_height - 1)
  return x_px, y_px



# Create a image segmenter instance with the live stream mode: Callback
# Used only for video or live stream
def print_result(result: List[mp.Image], output_image: mp.Image, timestamp_ms: int):
    print('segmented masks size: {}'.format(len(result)))


def get_face_model():
    # STEP 2: Create an FaceDetector object.
    base_options = python.BaseOptions(model_asset_path='./models/blaze_face_short_range.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)
    return detector


# Image segmentation
def get_segmentation_model():
    base_options = python.BaseOptions(model_asset_path='./models/selfie_segmenter.tflite')
    
    ImageSegmenterOptions = vision.ImageSegmenterOptions
    ImageSegmenter = vision.ImageSegmenter
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = ImageSegmenterOptions(
        base_options=base_options,
        running_mode=VisionRunningMode.IMAGE, #VisionRunningMode.VIDEO, VisionRunningMode.LIVE_STREAM
        output_category_mask=True
    )
    detector = ImageSegmenter.create_from_options(options)
    return detector

def get_hand_model(num_hands=2, min_detection_confidence=0.5, min_presence_confidence=0.5, min_tracking_confidence=0.5):
    base_options = python.BaseOptions(model_asset_path='./models/hand_landmarker.task')
    options = vision.HandLandmarkerOptions(base_options=base_options,
        running_mode=vision.RunningMode.IMAGE, #vision.RunningMode.VIDEO,vision.RunningMode.LIVE_STREAM
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    #vision.HandLandmarkerOptions(base_options=base_options)               
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

    def check_results_exist(self, detection_results):
        return detection_results and detection_results.hand_landmarks != []
    
    def draw_on_image(self, frame, detection_results):
        return draw_landmarks_on_image(frame, detection_results)
    
    def get_df_frame(self, detection_results, frame_id):
        return hand_landmarks_to_dataframe(detection_results,frame_id)
    
    def detect(self, image):
        return self.detector.detect(image)

    def process_frame(self, frame, frame_id):
        """
        Process an image frame using the detector and store the results in the dataframe.

        :param frame: The image frame to process.
        :param frame_id: An identifier for the frame.
        """
        # Use the detector to process the frame
        try:
            converted_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)            
            detection_results = self.detect(converted_frame)

            if self.check_results_exist(detection_results):
                # STEP 5: Process the classification result. In this case, visualize it.
                enhanced_frame = self.draw_on_image(converted_frame.numpy_view(), detection_results)                
                frame_data = self.get_df_frame(detection_results,frame_id)
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
    
###########################################################################################################

class ImageSegmenter_Wrapper(FrameProcessor):
    def __init__(self, detector):
        self.detector = detector

    def detect(self, image):
        return self.detector.segment(image)
    
    def check_results_exist(self,detection_results):
        return detection_results is not None and detection_results.category_mask is not None
    
    def get_df_frame(self, detection_results, frame_id):
        if(detection_results is None):
            return pd.DataFrame()
                
        row = {'time': frame_id , 'timestamp': datetime.now(timezone.utc).isoformat() }
        df = pd.DataFrame(row)        
        return df
    
    def draw_on_image(self, frame, detection_results):
        BG_COLOR = (192, 192, 192) # gray
        MASK_COLOR = (255, 255, 255) # white
        category_mask = detection_results.category_mask
        
        # Generate solid color images for showing the output segmentation mask.
        image_data = frame
        fg_image = np.zeros(image_data.shape, dtype=np.uint8)
        fg_image[:] = MASK_COLOR
        bg_image = np.zeros(image_data.shape, dtype=np.uint8)
        bg_image[:] = BG_COLOR

        condition = np.stack((category_mask.numpy_view(),) * 3, axis=-1) > 0.2
        output_image = np.where(condition, fg_image, bg_image)
        return output_image
    
###########################################################################################################

class FaceDetector_Wrapper(FrameProcessor):
    def __init__(self, detector):
        FrameProcessor.__init__(self, detector)        

    def detect(self, image):
        return self.detector.detect(image)
    
    def check_results_exist(self,detection_results):
        return detection_results is not None and detection_results.detections is not None
    
    def get_df_frame(self, detection_results, frame_id):
        if(detection_results is None):
            return pd.DataFrame()
                
        row = {'time': frame_id , 'timestamp': datetime.now(timezone.utc).isoformat() }
        df = pd.DataFrame(row, index=[0])
        return df
    
    def draw_on_image_box_and_features(self, frame, detection_results):

        MARGIN = 10  # pixels
        ROW_SIZE = 10  # pixels
        FONT_SIZE = 1
        FONT_THICKNESS = 1
        TEXT_COLOR = (255, 0, 0)  # red

        annotated_image = frame.copy()
        height, width, _ = frame.shape

        for detection in detection_results.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            start_point = bbox.origin_x, bbox.origin_y
            end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
            cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

            # Draw keypoints
            for keypoint in detection.keypoints:
                keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                                width, height)
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

                # Draw label and score
                category = detection.categories[0]
                category_name = category.category_name
                category_name = '' if category_name is None else category_name
                probability = round(category.score, 2)
                result_text = category_name + ' (' + str(probability) + ')'
                text_location = (MARGIN + bbox.origin_x,
                                MARGIN + ROW_SIZE + bbox.origin_y)
                cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                            FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

        return annotated_image

    def draw_on_image(self, frame, detection_results):

        TEXT_COLOR = (0, 0, 0)

        annotated_image = frame.copy()
        height, width, _ = frame.shape  

        for detection in detection_results.detections:
            # Draw bounding_box
            bbox = detection.bounding_box
            annotated_image[bbox.origin_y:bbox.origin_y+bbox.height,bbox.origin_x:bbox.origin_x+bbox.width] = TEXT_COLOR


        return annotated_image

