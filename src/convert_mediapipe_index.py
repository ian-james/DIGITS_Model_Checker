import pandas as pd
import logging
from enum import Enum, auto
from pprint import pprint


class Digit(Enum):
    Wrist = auto()
    Thumb = auto()
    Index = auto()
    Middle = auto()
    Ring = auto()
    Pinky = auto()


def default_digit_tip_names():
    return{
        "Thumb": ["Thumb Tip", "Thumb CMC"],
        "Index": ["Index Finger Tip", "Index Finger MCP"],
        "Middle": ["Middle Finger Tip", "Middle Finger MCP"],
        "Ring": ["Ring Finger Tip", "Ring Finger MCP"],
        "Pinky": ["Pinky Tip", "Pinky MCP"]
    }

def default_coordinate_names():
    return {
        "X": " (X-coordinate)",
        "Y": " (Y-coordinate)",
        "Z": " (Z-coordinate)"
    }

def get_landmark_name(index):
    landmarks = {
        0: "Wrist",
        1: "Thumb CMC",
        2: "Thumb MCP",
        3: "Thumb IP",
        4: "Thumb Tip",
        5: "Index Finger MCP",
        6: "Index Finger PIP",
        7: "Index Finger DIP",
        8: "Index Finger Tip",
        9: "Middle Finger MCP",
        10: "Middle Finger PIP",
        11: "Middle Finger DIP",
        12: "Middle Finger Tip",
        13: "Ring Finger MCP",
        14: "Ring Finger PIP",
        15: "Ring Finger DIP",
        16: "Ring Finger Tip",
        17: "Pinky MCP",
        18: "Pinky PIP",
        19: "Pinky DIP",
        20: "Pinky Tip"
    }
    return landmarks.get(index, "Unknown Landmark")


def get_all_landmark_names():
    return [get_landmark_name(i) for i in range(21)]

def convert_all_columns_to_friendly_name(df, exclude_columns):
    return [convert_to_friendly_name(col) if col not in exclude_columns else col for col in df.columns]

def convert_to_friendly_name(coordinate_str):
    """
    Converts a coordinate string of the form 'x_0', 'y_0', 'z_0', etc.,
    to a user-friendly version that describes the hand landmark.

    Args:
    coordinate_str (str): The coordinate string representing a MediaPipe hand landmark.

    Returns:
    str: A user-friendly description of the hand landmark.
    """
    # Extract the index and the coordinate (x, y, or z) from the string
    parts = coordinate_str.split('_')
    if len(parts) == 2 and parts[0] in ['x', 'y', 'z']:
        try:
            index = int(parts[1])
            landmark_name = get_landmark_name(index)
            coordinate = parts[0].upper()  # Convert 'x', 'y', 'z' to uppercase for readability
            return f"{landmark_name} ({coordinate}-coordinate)"
        except ValueError:
            # The part after the underscore wasn't an integer
            return coordinate_str
    else:
        return coordinate_str
    

def save_user_friendly(filename, df):
    if(df is not None):
        # Save the DataFrame to a CSV file
        df = convert_all_columns_to_friendly_name(df,[])
        df.to_csv(filename, index=False)    
        logging.info(f"Saved user friendly file to {filename}")
    else:
        logging.error(f"Failed to save user friendly file to {filename}")

# if __name__ == "__main__":
#     # Example usage
#     for coordinate_str in ["x_0", "y_4", "z_10"]:
#         friendly_name = convert_to_friendly_name(coordinate_str)
#         print(f"{coordinate_str} -> {friendly_name}")

#     # Example usage
#     df = pd.DataFrame(columns=["time","x_0", "y_4", "z_10"])
#     renames1 = convert_all_columns_to_friendly_name(df, ["time"])

#     renames = convert_all_columns_to_friendly_name(df,[])

def get_thumb_base_index():
    return [1]

def get_thumb_tip_indices():
    return [4]


def get_wrist_index():
    return [0]

def get_thumb_indices():
    return [1, 2, 3, 4]
    
def get_index_finger_indices():
    return [5, 6, 7, 8]

def get_middle_finger_indices():
    return [9, 10, 11, 12]

def get_ring_finger_indices():
    return [13, 14, 15, 16]

def get_pinky_finger_indices():
    return [17, 18, 19, 20]

def get_wrist_thumb_indices():
    return get_wrist_index() + get_thumb_indices()

def get_all_fingers_indices(add_wrist_to_indices=False, add_wrist = False):
    r= {
        Digit.Thumb.name:  get_wrist_index() + get_thumb_indices() if add_wrist_to_indices else get_thumb_indices(),
        Digit.Index.name:  get_wrist_index() + get_index_finger_indices() if add_wrist_to_indices else get_index_finger_indices(),
        Digit.Middle.name: get_wrist_index() + get_middle_finger_indices() if add_wrist_to_indices else get_middle_finger_indices(),
        Digit.Ring.name:   get_wrist_index() + get_ring_finger_indices() if add_wrist_to_indices else get_ring_finger_indices(),
        Digit.Pinky.name:  get_wrist_index() + get_pinky_finger_indices() if add_wrist_to_indices else get_pinky_finger_indices()
    }
    if add_wrist:
        r[Digit.Wrist.name] = get_wrist_index()
    return r

########################################################################################################
def test_main():
    test_index_finger_indices()
    test_all_fingers_indices()    

def test_index_finger_indices():
    expected_indices = [5, 6, 7, 8]
    indices = get_index_finger_indices()
    assert indices == expected_indices, f"Expected {expected_indices}, but got {indices}"
    print("All tests pass")


def test_all_fingers_indices():
    expected_indices = {
        #Digit.Wrist.name: [0],
        Digit.Thumb.name: [1, 2, 3, 4],
        Digit.Index.name: [5, 6, 7, 8],
        Digit.Middle.name: [9, 10, 11, 12],
        Digit.Ring.name: [13, 14, 15, 16],
        Digit.Pinky.name: [17, 18, 19, 20]
    }
    indices = get_all_fingers_indices()
    assert indices == expected_indices, f"Expected {expected_indices}, but got {indices}"
    print("All tests pass")



########################################################################################################
if __name__ == "__main__":
    test_main()
    
