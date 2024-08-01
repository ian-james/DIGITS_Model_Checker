import numpy as np
import pandas as pd

import logging

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.python.solutions.drawing_utils import DrawingSpec
from mediapipe.python.solutions.drawing_styles import _HAND_LANDMARK_STYLE, _RED
from mediapipe.framework.formats import landmark_pb2

from datetime import datetime, timezone
import math
from typing import List, Tuple, Union, Mapping

from convert_mediapipe_index import get_landmark_name, get_thumb_indices, get_thumb_tip_indices, convert_all_columns_to_friendly_name
from calculate_joints_and_length import calculate_all_digit_lengths, calculate_angle_between_digit_df
from calculation_helpers import subtract_lists, calculate_angle, calculate_angles2

import cv2

# Constant for the estimated hand size in centimeters.
# ROUGHT ESTIMATE

FINGER_TIP_CONNECTIONS = ((4, 8), (8, 12), (12,16),(16, 20))
_FINGER_TIP_STYLE = { FINGER_TIP_CONNECTIONS: DrawingSpec(color=_RED, thickness=2) }

def get_default_finger_tip_landmarks_style() -> Mapping[int, DrawingSpec]:
  """Returns the default hand landmarks drawing style.

  Returns:
      A mapping from each hand landmark to its default drawing spec.
  """
  styles = {}
  for k, v in _FINGER_TIP_STYLE.items():
    for landmark in k:
      styles[landmark] = v
  return styles

v = get_default_finger_tip_landmarks_style()

def get_landmark_pb( landmarks, idx):
    r = None
    for i, landmark in enumerate(landmarks):
        if(i == idx):
            #x,y,z = landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            r = [landmark.x, landmark.y, landmark.z]
            break
    return r

def get_landmark(landmarks, idx):

    # Add each landmark's coordinates to the row
    for i, landmark in enumerate(landmarks):
        print(i, landmark)
        if( i == idx ):
            return landmark
    return None

def _normalized_to_pixel_coordinates(
    normalized_x: float, normalized_y: float, image_width: int,
    image_height: int) -> Union[None, Tuple[int, int]]:
  """Converts normalized value pair to pixel coordinates."""

  # Checks if the float value is between 0 and 1.
  def is_valid_normalized_value(value: float) -> bool:
    return (value > 0 or math.isclose(0, value)) and (value < 1 or math.isclose(1, value))

  if not (is_valid_normalized_value(normalized_x) and
          is_valid_normalized_value(normalized_y)):

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

def hand_landmarks_to_dataframe(detection_result,frame_id, save_as_list = False, save_extra_columns = False, resolution = (1,1,1)):
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

            if(not save_as_list):
                row[f'x_{i}'] = landmark.x * resolution[0]
                row[f'y_{i}'] = landmark.y * resolution[1]
                row[f'z_{i}'] = landmark.z * resolution[2]
                if(save_extra_columns):
                    row[f'presence_{i}'] = landmark.presence
                    row[f'visibility_{i}'] = landmark.visibility
            else:
                row[f'{get_landmark_name(i)}'] = [landmark.x * resolution[0], landmark.y * resolution[1], landmark.z * resolution[2]]
                if(save_extra_columns):
                    row[f'presence_{i}'] = [landmark.presence]
                    row[f'visibility_{i}'] = [landmark.visibility]

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
        self.resolution = (1,1,1)
        self.save_as_list = False
        self.save_extra_columns = False

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
        return hand_landmarks_to_dataframe(detection_results,frame_id,self.save_as_list,self.save_extra_columns,self.resolution)

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
            if(self.resolution == (1,1,1)):
                self.resolution = (frame.shape[1],frame.shape[0],1)

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

            return detection_results, frame


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

    def get_resolution(self):
        return self.resolution

    def save_model_data(self, kwargs):

        filename = kwargs.get('output','output.csv')
        collected_data = kwargs.get('collected_data',True)
        use_friendly_names = kwargs.get('use_friendly_names',True)
        remove_non_position_columns = kwargs.get('remove_non_position_columns',True)
        sep = kwargs.get('sep',"\t")

        df = self.get_dataframe()

        if(df is None):
            raise Exception("Dataframe is None.")

        if(type(df) is not pd.DataFrame):
            raise Exception("Dataframe is not a pandas dataframe.")

        if( df is None or len(df) == 0 or df.shape[0] == 0):
            if(collected_data):
                logging.info("Data was unable to saved on the output file.")
            else:
                logging.info("Data was not collected.")

        if(use_friendly_names):
            df.columns = convert_all_columns_to_friendly_name(df, [])

        if( remove_non_position_columns):
            # Remove the presence and visibility columns
            df = df[df.columns.drop(list(df.filter(regex='^presence_\\d+')),errors='ignore')]
            df = df[df.columns.drop(list(df.filter(regex='^visibility_\\d+')),errors='ignore')]

        print(filename)
        df.to_csv(filename, index=False, header=True, sep=sep)
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

###########################################################################################################

class HandDistance_Wrapper(FrameProcessor):
    def __init__(self, detector, proximal_phalanx_size=3.0):
        FrameProcessor.__init__(self, detector)

        # Add Last Calcualted Distances
        self.distances = []

        # Add IDxs for the tips
        # 4,8,12,16, 20 Thumb, Index ....Pinky
        self.idxs = [4,8,12,16,20]

        # Colors for each finger line
        self.colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (0,255,255)]

        self.MY_PROXIMAL_PHALANX_SIZE = proximal_phalanx_size
        self.PIXEL_TO_CM_ESTIMATE = 0.018592443594320586


    def get_distances(self):
        return self.distances


    def draw_on_image(self, frame, detection_results):
        return draw_landmarks_on_image(frame, detection_results)

    def get_df_frame(self, detection_results, frame_id):
        return hand_landmarks_to_dataframe(detection_results,frame_id,self.save_as_list,self.save_extra_columns,self.resolution)


    def process_frame(self, frame, frame_id):
        detection_results, image =  super().process_frame(frame, frame_id)

        if detection_results is not None:

            tips = detection_results.hand_landmarks
            height, width, _ = frame.shape

            index_proximal_phalanx_size = self.calculate_distance( detection_results.hand_landmarks[0], [6,7], width, height)
            PIXEL_TO_CM_ESTIMATE = self.MY_PROXIMAL_PHALANX_SIZE / index_proximal_phalanx_size["Distance"].sum()

            index_size = self.calculate_distance( detection_results.hand_landmarks[0], [5,8], width, height)
            print(f"\n\nIndex Finger Size: {index_size['Distance'].sum()*PIXEL_TO_CM_ESTIMATE} cm\n\n")

            self.distances =  self.calculate_distance(detection_results.hand_landmarks[0],self.idxs,width,height)
            if "Distance" in self.distances:
                self.distances["Est CM Distance"] = self.distances["Distance"] * PIXEL_TO_CM_ESTIMATE
            image = self.draw_tip_distances(image,detection_results, self.distances)

        return detection_results, image

    # List should be sorted
    def get_tip_landmarks(self, landmarks, indices=[4, 8, 12, 16, 20 ]):
        idx = 0
        key_landmarks = []
        for i, landmark in enumerate(landmarks):
            if(idx >= len(indices)):
                break
            if(i == indices[idx]):
                print(f"Landmark {i} - {landmark} - {get_landmark_name(i)}")
                key_landmarks.append(landmark)
                idx += 1

        return key_landmarks

    def draw_tip_distances(self, image, detection_results, distances):

        hand_landmarks_list = detection_results.hand_landmarks

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Draw the hand landmarks.
            print(idx)
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image,
                hand_landmarks_proto,
                FINGER_TIP_CONNECTIONS,
                solutions.drawing_styles.get_default_hand_landmarks_style(),
                get_default_finger_tip_landmarks_style()
            )
        return image


    def calculate_distance(self, landmarks, idxs=[4,8,12,16,20], width = 1, height=1):
        """
        Calculate the distance between two landmarks.

        :param landmarks: A list of two landmarks, each containing x, y, and z coordinates.
        :return: The Euclidean distance between the two landmarks.
        """

        dists = []
        for i in range(0,len(idxs)-1):

            if idxs[i] >= len(landmarks) and idxs[i+1] >= len(landmarks):
                break

            idxi = idxs[i]
            idxj = idxs[i+1]

            x1, y1, z1 =[ landmarks[idxi].x *width , landmarks[idxi].y * height, landmarks[idxi].z ]
            x2, y2, z2 =[ landmarks[idxj].x *width, landmarks[idxj].y * height, landmarks[idxj].z ]

            # Calculate the Euclidean distance between the two landmarks
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            print(f"Distance between landmarks: {distance:.2f}")

            c = {"Connection A": get_landmark_name(idxi),"Connection B":  get_landmark_name(idxs[i+1]), "Distance":distance}
            dists.append(c)

        return pd.DataFrame(dists)


###########################################################################################################

class HandROM_Thumb_Wrapper(FrameProcessor):
    def __init__(self, detector):
        super().__init__(detector)

        # Colors for each finger line
        self.colors = [(0,0,255), (0,255,0), (255,0,0), (255,255,0), (0,255,255)]

        self.min_points = {}
        self.max_points = {}
        self.min_values = {}
        self.max_values = {}

        # Add Last Calcualted Distances
        self.angles = []
        self.diameter = []


    def get_df_frame(self, detection_results, frame_id):
        return hand_landmarks_to_dataframe(detection_results,frame_id,self.save_as_list,self.save_extra_columns,self.resolution)

    def process_frame(self, frame, frame_id):

        detection_results, image =  super().process_frame(frame, frame_id)

        if detection_results is not None:
            self.calculate_digit_range(detection_results)
            nimage = self.draw_thumb_ROM(image,detection_results)
            if( nimage is None):
                #return detection_results, image.numpy_view()
                return detection_results, image
            return detection_results, nimage

        return detection_results, image


    # This function isn't used
    # def calculate_tip_movement_digit_range(self,detection_results):

    #     if( detection_results is None or detection_results.hand_landmarks == []):
    #         return

    #     hand_landmarks_list = detection_results.hand_landmarks

    #     thumb_indicies = get_thumb_tip_indices()
    #     coordinates = ['x','y','z']

    #     # Write code that checks the hand_landmarks_list and returns the min and max points for the digit
    #     # Loop through the detected hands to visualize.
    #     for idx in range(len(thumb_indicies)):

    #         # Gets us the landmark as an array
    #         landmark = get_landmark_pb(hand_landmarks_list[0], thumb_indicies[idx])

    #         # Convert the list of points to a numpy array for easy manipulation
    #         # Each position should be in either [x,y,z] or ['x-coordinate','y-coordinate','z-coordinate']

    #         # Loop through each coordinate to find the max and min points
    #         # NOte this will have to change because this is checking local values not maximumums
    #         for i, coord in enumerate(coordinates):

    #             if((coord not in self.max_points) or landmark[i] > self.max_points[coord][i]):
    #                 self.max_points[coord] = landmark

    #             if( (coord not in self.min_points) or landmark[i] < self.min_points[coord][i]):
    #                 self.min_points[coord] = landmark

    def calculate_digit_range(self,detection_results):

        if( detection_results is None or detection_results.hand_landmarks == []):
            return

        hand_landmarks_list = detection_results.hand_landmarks

        # Select the first and last thumb indices
        thumb_indicies = get_thumb_indices()
        # These are the Thumb Tip and the Thumb Base
        thumb_idx = [thumb_indicies[0], thumb_indicies[-1]]
        coordinates = ['x','y','z']

        # Write code that checks the hand_landmarks_list and returns the min and max points for the digit
        # Loop through the detected hands to visualize.
                # Gets us the landmark as an array
        base_landmark = get_landmark_pb(hand_landmarks_list[0], thumb_idx[0])
        tip_landmark = get_landmark_pb(hand_landmarks_list[0], thumb_idx[1])
        landmark = subtract_lists(tip_landmark,base_landmark)

        # Convert the list of points to a numpy array for easy manipulation
        # Each position should be in either [x,y,z] or ['x-coordinate','y-coordinate','z-coordinate']

        # Loop through each coordinate to find the max and min points
        # NOte this will have to change because this is checking local values not maximumums
        update_min_max = False
        for i, coord in enumerate(coordinates):

            # the absolute value of the landmark
            self.diameter.append(landmark)
            # This version creates ones of max/min as static while the other is dynamic
            if(update_min_max):
                if((coord not in self.max_points) or landmark[i] > self.max_points[coord][i]):
                    self.max_points[coord] = tip_landmark

                if( (coord not in self.min_points) or landmark[i] < self.min_points[coord][i]):
                    self.min_points[coord] = tip_landmark
            else:
                #This version stores the max/min values to show the full ROM of the tip of the thumb.
                if((coord not in self.max_points) or landmark[i] > self.max_values[coord][i]):
                    self.max_values[coord] = landmark
                    self.max_points[coord] = tip_landmark

                if( (coord not in self.min_points) or landmark[i] < self.min_values[coord][i]):
                    self.min_values[coord] = landmark
                    self.min_points[coord] = tip_landmark

        # Calculate the angle between the min and max vector points             
        center = get_landmark_pb(hand_landmarks_list[0], thumb_idx[0])

        # Calculate the angle between the center and the min and max points
        v1 = subtract_lists(self.min_points['x'], center)
        v2 = subtract_lists(self.max_points['x'], center)
        a = calculate_angle(v1,v2)

        # if both vectors are two small, then we can't calculate the angle
        if( abs(v1[0]) < 0.0005 and abs(v1[1]) < 0.0005 and v2[0] < 0.0005 and abs(v2[1]) < 0.0005):
            a = 0   
        
        self.angles.append(a)

    def calculate_center(self):
        # Find the center of the circle
        if( 'x' in self.min_points and 'x' in self.max_points and 'y' in self.min_points and 'y' in self.max_points):
            min_x,max_x = self.min_points['x'], self.max_points['x']
            min_y,max_y = self.min_points['y'], self.max_points['y']

            center_x = (max_x[0] + min_x[0]) / 2
            center_y = (max_y[1] + min_y[1]) / 2

            return (center_x, center_y )
        return None


    # This function takes the Min/MAx points from a video and draw a circle around it.
    def draw_thumb_ROM(self, image, detection_results, include_min_max = True):

        # Draw a circle at the image points created by the min and max points
        if( image is None):
            return image

        if( 'x' in self.min_points and 'x' in self.max_points and 'y' in self.min_points and 'y' in self.max_points):
            min_x,max_x = self.min_points['x'], self.max_points['x']
            min_y,max_y = self.min_points['y'], self.max_points['y']

            # Draw a circle such that these points are on the circumference
            # Find the center of the circle
            center_x = (max_x[0] + min_x[0]) / 2
            center_y = (max_y[1] + min_y[1]) / 2

            # Find the radius of the circle
            radius = math.sqrt(((max_x[0] - center_x) *self.resolution[0] )** 2 + ((max_y[1] - center_y)*self.resolution[1] ) ** 2)

            try:
                # Draw the circle on the image
                x = center_x*self.resolution[0]
                y = center_y*self.resolution[1]
                cv2.circle(image, (int(x), int(y)), int(radius), (0, 255, 0), 2)

                if(include_min_max):
                    # Draw each min point in a different color
                    x = min_x[0]*self.resolution[0]
                    y = min_y[1]*self.resolution[1]
                    cv2.circle(image, (int(x), int(y)), 10, (255, 0, 0),1)

                    # Draw the max point in a different color
                    x = max_x[0]*self.resolution[0]
                    y = max_y[1]*self.resolution[1]
                    cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), 1)

                    # Calculate the angle between the min and max points
                    if( detection_results is None or detection_results.hand_landmarks == []):
                        return image

                    hand_landmarks_list = detection_results.hand_landmarks

                    # Select the first and last thumb indices
                    thumb_indicies = get_thumb_indices()
                    thumb_idx = thumb_indicies[0]
                    base_landmark = get_landmark_pb(hand_landmarks_list[0], thumb_idx)

                    # Calculate base landmark location

                    x = int(base_landmark[0]*self.resolution[0])
                    y = int(base_landmark[1]*self.resolution[1])

                    mx = int(abs(self.max_points['x'][0]*self.resolution[0]))
                    my = int(abs(self.max_points['x'][1]*self.resolution[1]))
                    
                    # Max Line - Pink
                    cv2.line(image, (int(x), int(y)), (mx,my), (255, 0, 255), 2)

                    #Min Line - Yellow
                    xmin = int(self.min_points['x'][0]*self.resolution[0])
                    ymin = int(self.min_points['x'][1]*self.resolution[1])
                    cv2.line(image, (int(x), int(y)), (xmin,ymin), (255, 255, 0), 2)
                    
                return image
            except Exception as e:
                print(f"Error drawing the circle: {e}")
                return None
        return None

    def save_model_data(self, kwargs):
        super().save_model_data(kwargs)

        filename = kwargs.get('output','output.csv')
        sep = kwargs.get('sep',"\t")

        angles_file = filename.replace(".csv","_angles.csv")
        if( self.angles):
            df_angles = pd.DataFrame(columns=['angle'],data=self.angles)
            df_angles.to_csv(angles_file, index=False, header=True, sep=sep)

        diameter_file = filename.replace(".csv","_diameter.csv")
        if( self.diameter):
            df_diameter = pd.DataFrame(columns=['x','y','z'], data=self.diameter)

            # Multiply each column by the resolution to get the actual values
            df_diameter['x'] = df_diameter['x'] * self.resolution[0]
            df_diameter['y'] = df_diameter['y'] * self.resolution[1]
            df_diameter['z'] = df_diameter['z'] * self.resolution[2]

            df_diameter.to_csv(diameter_file, index=False, header=True, sep=sep)

