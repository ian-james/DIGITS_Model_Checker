import pandas as pd
from mediapipe.framework.formats import landmark_pb2

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
        
        row = {'time': frame_id ,  'handedness': handedness[0].category_name}        
      
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