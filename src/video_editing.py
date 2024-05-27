import cv2
import os
from moviepy.editor import VideoFileClip, concatenate_videoclips
import numpy as np
from pathlib import Path


# Check if a codec is supported by MoviePy (FFmpeg)
# Testing for only a small subset of codecs has been tested.
# You can add more codecs to the list if needed.
def is_codec_supported(codec):
    # List of common codecs supported by MoviePy (FFmpeg)
    supported_codecs = [
        'libx264',  # H.264 video codec
        #'libx265',  # H.265 video codec
        #'libvpx',   # VP8 video codec
        #'libvpx-vp9', # VP9 video codec
        #'libaom-av1', # AV1 video codec
        'mpeg4',    # MPEG-4 video codec
        #'libtheora' # Theora video codec
    ]

    # Check if the codec is in the list of supported codecs
    return codec.lower() in supported_codecs

# Extract frames from a video
def extract_frames(video_path, output_dir):
 
    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error loading video: {video_path}")
        return  # Exit if video not opened
    
    count = 0
    filestem = Path(video_path).stem
    # Extract frames
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Fix this to use the video name        
        n = f"{filestem}_frame_{count:06d}.jpg"
        cv2.imwrite(os.path.join(output_dir, f"{filestem}_frame_{count:06d}.jpg"), frame)
        count += 1
    
    cap.release()
    print(f"Extracted {count} frames into directory: {output_dir}")

# Convert video format, from one codec to another
def convert_video_format(input_video_path, output_video_path, codec="libx264"):
    clip = VideoFileClip(input_video_path)
    clip.write_videofile(output_video_path, codec=codec)

def cut_video_by_time(input_video_path, output_video_path, start_time, end_time, codec="libx264"):
    # Load the video file
    video = VideoFileClip(input_video_path)
    
    # Cut the video
    cut_video = video.subclip(start_time, end_time)
    
    # Write the resulting video to a file
    cut_video.write_videofile(output_video_path, codec=codec)
    
    # Close the video file to free resources
    video.close()
    cut_video.close()


# Cut a video by frame numbers
def cut_video_by_frame(input_video_path, output_video_path, start_frame, end_frame, codec="libx264"):
    # Load the video file
    video = VideoFileClip(input_video_path)
    
    # Calculate the start and end times based on frame numbers
    fps = video.fps  # Frames per second
    start_time = start_frame / fps
    end_time = end_frame / fps
    
    # Cut the video
    cut_video = video.subclip(start_time, end_time)
    
    # Write the resulting video to a file
    cut_video.write_videofile(output_video_path, codec=codec)
    
    # Close the video file to free resources
    video.close()
    cut_video.close()


# This function takes a single image and creates a video from it
def create_video_from_image(image_path, output_video_path, duration=5, codec="mp4v", fps=30):
    # Load the image
    image = cv2.imread(image_path)

    # Check if the image loaded properly
    if image is None:
        print(f"Error loading image: {image_path}")
        return
    
    height, width, _ = image.shape

    # Wrap this in a try-except block to catch any errors
    try:
        # Create a VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Calculate the number of frames needed
        num_frames = duration * fps

        # Write the image to the video file multiple times to create the video
        for _ in range(num_frames):
            out.write(image)

        # Release the VideoWriter object
        out.release()        
        print(f"Video created from image: {output_video_path}")
    except Exception as e:
        print(f"Error creating video from image: {e}")

# Repeat the last frame of a video multiple times
# This function takes an input video, repeats the last frame a specified number of times,
def repeat_last_frame(input_video_path, output_video_path, repeat_count):
    # Load the video
    video = VideoFileClip(input_video_path)
    
    # Get the last frame, which will be an ImageClip
    last_frame = video.to_ImageClip(duration=video.duration)
    
    # Set the duration for the last frame to repeat
    # Assume each repeated frame should last for 1 second (adjust duration as needed)
    last_frame_duration = 1  # in seconds
    last_frame = last_frame.set_duration(last_frame_duration)
    
    # Repeat the last frame
    repeated_last_frames = [last_frame] * repeat_count
    
    # Create a new clip that concatenates the original video with the repeated frames
    final_clip = concatenate_videoclips([video] + repeated_last_frames)
    
    # Write the result to a file
    final_clip.write_videofile(output_video_path, codec='libx264')
    
    # Close all clips to free up resources
    video.close()
    final_clip.close()


def are_images_equal(image1, image2):   

    # Check if both images are loaded properly
    if image1 is None or image2 is None:
        print("Error: One or both image paths are invalid.")
        return False

    # Check if the dimensions and channels are the same
    if image1.shape != image2.shape:
        print("The images have different sizes or channels.")
        return False

    # Check if the images are exactly the same
    if np.array_equal(image1, image2):
        return True
    else:
        return False