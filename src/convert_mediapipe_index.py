import pandas as pd
import logging

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
