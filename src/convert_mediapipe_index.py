import pandas as pd
import logging
from enum import Enum, auto
from pprint import pprint

import re
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


def default_wrist_to_index_tip_names():
    return {
        "Wrist": ["Wrist", "Index Finger Tip"]
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

def get_landmark_xyz_name(index):
    landmark_name = get_landmark_name(index)    
    return [f"{landmark_name} ({coordinate}-coordinate)" for coordinate in ["X", "Y", "Z"]]

def get_all_landmarks_xyz_names():    
    all_landmarks = []
    for i in range(21):
        all_landmarks += get_landmark_xyz_name(i)
    return all_landmarks
    
def real_world_landmark_names():
    return real_world_extension_names() + real_world_flexion_names()

def real_world_flexion_names():
    return ["Index MP joint (flexion)", "Index PIP joint (flexion)", "Index DIP joint (flexion)", "Long MP joint (flexion)", "Long PIP joint (flexion)", "Long DIP joint (flexion)", "Ring MP joint (flexion)", "Ring PIP joint (flexion)", "Ring DIP joint (flexion)", "Little MP joint (flexion)", "Little PIP joint (flexion)", "Little DIP joint (flexion)", "Thumb MP joint (flexion)", "Thumb IP joint (flexion)"]

def real_world_extension_names():
    return [ "Index MP joint (extension)", "Index PIP joint (extension)", "Index DIP joint (extension)", "Long MP joint (extension)", "Long PIP joint (extension)", "Long DIP joint (extension)", "Ring MP joint (extension)", "Ring PIP joint (extension)", "Ring DIP joint (extension)", "Little MP joint (extension)", "Little PIP joint (extension)", "Little DIP joint (extension)", "Thumb MP joint (extension)", "Thumb IP joint (extension)"]

def real_world_finger_order():
    return ["Index", "Long", "Ring", "Little", "Thumb"]

def mediapipe_finger_order():
    return ["Thumb", "Index", "Middle", "Ring", "Pinky"]

def replace_str(s, old, new):
    t = s.replace(old, new)
    return t

def convert_real_finger_name_to_mediapipe_name_within_string(real_name):
    # Find the finger name in the string
    for finger in real_world_finger_order():
        if finger in real_name:
            return replace_str(real_name,finger, convert_real_world_finger_name_to_sasha_name(finger))

    return real_name

def convert_real_world_finger_name_to_mediapipe_name(real_world_name):
    c = {
        "Index": "Index",
        "Long": "Middle",
        "Ring": "Ring",
        "Little": "Pinky",
        "Thumb": "Thumb"
    }
    return c.get(real_world_name, "Unknown Finger")

def mediapipe_index_order_in_real_world():
    return [1, 2, 3, 4, 0]
    
def convert_real_world_finger_name_to_mediapipe_index(real_world_name):
    creal = convert_real_world_finger_name_to_mediapipe_name(real_world_name)
    # Safely find the index of the finger in the mediapipe order
    return mediapipe_finger_order().index(creal)


def convert_real_world_to_mediapipe_index(real_world_name):

    # parts = real_world_name.split(" ")
    # if len(parts) == 4:
    #     finger, joint, _,  flexion  = parts
        
    #     finger_joint_idx = get_joint_index(f"{finger} {joint}") 

    pass

def convert_real_world_finger_name_to_sasha_name(real_world_name):
    c = {
        "Index": "Index",
        "Long": "Long",
        "Ring": "Ring",
        "Little": "Small",
        "Thumb": "Thumb"
    }
    return c.get(real_world_name, "Unknown Finger")

# Join names consistent with Sasha's work in thesis.
def get_joint_names():
    # Make a dictionary of the joint names
    # "Thumb CMC","Thumb MCP","Thump IP","Index MCP","Index PIP","Index DIP","Long MCP","Long PIP","Long DIP","Ring MCP","Ring PIP","Ring DIP","Small MCP","Small PIP","Small DIP"
    joints = {
        1: "Thumb CMC",
        2: "Thumb MCP",
        3: "Thumb IP",
        4: "Index MCP",
        5: "Index PIP",
        6: "Index DIP",
        7: "Long MCP",
        8: "Long PIP",
        9: "Long DIP",
        10: "Ring MCP",
        11: "Ring PIP",
        12: "Ring DIP",
        13: "Small MCP",
        14: "Small PIP",
        15: "Small DIP"
    }
    return joints

def get_digit_joint_names():
    return {
        Digit.Thumb.name: ["Thumb CMC", "Thumb MCP", "Thumb IP"],
        Digit.Index.name: ["Index MCP", "Index PIP", "Index DIP"],
        Digit.Middle.name: ["Long MCP", "Long PIP", "Long DIP"],
        Digit.Ring.name: ["Ring MCP", "Ring PIP", "Ring DIP"],
        Digit.Pinky.name: ["Small MCP", "Small PIP", "Small DIP"]
    }

def get_digit_joint_indices():
    return {
        Digit.Thumb.name: [1, 2, 3],
        Digit.Index.name: [4, 5, 6],
        Digit.Middle.name: [7, 8, 9],
        Digit.Ring.name: [10, 11, 12],
        Digit.Pinky.name: [13, 14, 15]
    }

def get_digit_joint_names_from_digit(digit):
    return get_digit_joint_names().get(digit, [])

def get_digit_joint_index_from_digit(digit):
    return get_digit_joint_indices().get(digit, [])

def reverse_joint_names():
    return {v: k for k, v in get_joint_names().items()}

def get_joint_index(joint_name):
    joints = reverse_joint_names()
    return joints.get(joint_name, -1)

def get_just_joint_names():
    return ["CMC","MCP","PIP","DIP","IP"]


def real_world_finger_joint_order():
    return ["MP", "PIP", "DIP"]

def real_world_thumb_joint_order():
    return ["MP", "IP"] #CMC is missing

def get_real_world_joint_to_mediapipe_mapping():
    return {
        "CMC": "CMC",
        "MP": "MCP",
        "MCP:": "MCP",
        "PIP": "PIP",
        "DIP": "DIP",
        "IP": "IP"
    }

def convert_real_world_joint_name_to_mediapipe_name(joint_name):
    c = get_real_world_joint_to_mediapipe_mapping()
    return c.get(joint_name, "Unknown Joint")

def convert_real_joint_name_to_mediapipe_name_within_string(real_name):
    # Find the finger name in the string
    keywords = get_real_world_joint_to_mediapipe_mapping().keys()

    pattern = r'\b(' + '|'.join(re.escape(word) for word in keywords) + r')\b'

    # Find all matches
    matches = re.findall(pattern, real_name)
    
    for joint in matches:
        return real_name.replace(joint, convert_real_world_joint_name_to_mediapipe_name(joint))

def get_joint_name(index):
    joints = get_joint_names()
    return joints.get(index, "Unknown Joint")

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
    
