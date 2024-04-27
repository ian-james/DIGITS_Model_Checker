import os
import cv2
from file_utils import create_directory

def flip_image(image, should_flip):
    if (should_flip):
        image = cv2.flip(image, 1)
        should_flip = False
    return image, should_flip

def write_snapshot_image(filename, image):
    # Write the image to a file
    status = cv2.imwrite(filename, image)
    return status

def open_recording_file(record_file, frame_size, fps, location="../records/"):
    # Define the codec and create a VideoWriter object
    # Allow Recording only if the user previously specified a file name.
    out_record_media = None
    if (record_file != ""):
        absolute_path = os.path.abspath(location)
        create_directory(absolute_path)
        full_path = os.path.join(absolute_path, record_file)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_record_media = cv2.VideoWriter(full_path, fourcc, fps, frame_size)
    return out_record_media

def open_image(image_path):
    try:
        # Read the image file
        image = cv2.imread(image_path)

        if image is None:
            raise ValueError("Error reading image file.")

        return image

    except Exception as e:
        print(f"An error occurred opening the image: {image_path}, {str(e)}")
        return None
