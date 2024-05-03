
# import mediapipe as mp
# import numpy as np
# import pandas as pd

# from mediapipe import solutions
# from mediapipe.framework.formats import landmark_pb2

# from Capture_Info import hand_landmarks_to_dataframe

# import cv2

# def draw_landmarks_on_image(rgb_image, detection_result):
#   MARGIN = 10  # Setting an arbitrary margin for easier visualization.
#   FONT_SIZE = 1
#   FONT_THICKNESS = 1
#   HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

#   hand_landmarks_list = detection_result.hand_landmarks
#   handedness_list = detection_result.handedness
#   annotated_image = np.copy(rgb_image)

#   # Loop through the detected hands to visualize.
#   for idx in range(len(hand_landmarks_list)):
#     hand_landmarks = hand_landmarks_list[idx]
#     handedness = handedness_list[idx]

#     # Draw the hand landmarks.
#     hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
#     hand_landmarks_proto.landmark.extend([
#       landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
#     ])
#     solutions.drawing_utils.draw_landmarks(
#       annotated_image,
#       hand_landmarks_proto,
#       solutions.hands.HAND_CONNECTIONS,
#       solutions.drawing_styles.get_default_hand_landmarks_style(),
#       solutions.drawing_styles.get_default_hand_connections_style())

#     # Get the top left corner of the detected hand's bounding box.
#     height, width, _ = annotated_image.shape
#     x_coordinates = [landmark.x for landmark in hand_landmarks]
#     y_coordinates = [landmark.y for landmark in hand_landmarks]
#     text_x = int(min(x_coordinates) * width)
#     text_y = int(min(y_coordinates) * height) - MARGIN

#     # Draw handedness (left or right hand) on the image.
#     cv2.putText(annotated_image, f"{handedness[0].category_name}",
#                 (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
#                 FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

#   return annotated_image


# class FrameProcessor:
#     def __init__(self, detector):
#         """
#         Initialize the FrameProcessor with a detector function.

#         :param detector: A function that takes an image frame and returns data about the frame.
#         """
#         self.detector = detector
#         # Initialize an empty DataFrame to store data about each frame
#         self.data = []

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_val, exc_tb):
#         # Cleanup code if needed
#         pass

#     def process_frame(self, frame, frame_id):
#         """
#         Process an image frame using the detector and store the results in the dataframe.

#         :param frame: The image frame to process.
#         :param frame_id: An identifier for the frame.
#         """
#         # Use the detector to process the frame
#         try:
#           converted_frame = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
#           #converted_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#           detection_results = self.detector.detect(converted_frame)

#           if detection_results.hand_landmarks != []:
#           # STEP 5: Process the classification result. In this case, visualize it.
#               enhanced_frame = draw_landmarks_on_image(converted_frame.numpy_view(), detection_results)
#              # enhanced_frame = draw_landmarks_on_image(converted_frame, detection_result)
#               frame_data = hand_landmarks_to_dataframe(detection_results,frame_id)
#               self.data.append(frame_data)
              
#               return detection_results, enhanced_frame

#           return None, frame
                  
#         except Exception as e:            
#             print(f"An error occurred while processing the frame: {e}")            
#         return None, frame
        
#     def finalize_dataframe(self):
#         if(self.data):
#           self.data = pd.concat(self.data, ignore_index= True)
#         else:
#            self.data = pd.DataFrame()
        