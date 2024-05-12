import os
import argparse

# User libraries
from video_editing import extract_frames, convert_video_format, is_codec_supported

# This is the main function that will be called when the script is run
# It will parse the command line arguments and call the appropriate functions
# to extract frames from a video and convert the video format
# The input video path, output frames directory, and output video path are required arguments
# The codec argument is optional and defaults to "libx264"
def main():
    parser = argparse.ArgumentParser(description="Extract frames from a video")
    parser.add_argument("-i","--input_video", type=str, help="Path to the input video file.")
    parser.add_argument("-o","--frames_output_dir", type=str, help="Directory to save the extracted frames.")    
    parser.add_argument("-c","--codec", type=str, default="libx264", help="Codec to use for video conversion. Default is 'libx264'.")

    args = vars(parser.parse_args())

    if( not is_codec_supported(args["codec"]) ):
        print(f"Codec not supported: {args['codec']}")
        return

    if(not os.path.exists(args["input_video"]) ):
        print(f"Input video file not found: {args['input_video']}")
        return

    # Make sure the output directory exists
    if not os.path.exists(args["frames_output_dir"]):
        os.makedirs(args["frames_output_dir"])

    # Extract frames
    extract_frames(args["input_video"], args["frames_output_dir"])    

if __name__ == "__main__":
    main()
