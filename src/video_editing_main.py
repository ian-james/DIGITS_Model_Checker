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
    parser = argparse.ArgumentParser(description="Extract frames from video and convert video format.")
    parser.add_argument("input_video_path", help="Path to the input video file.")
    parser.add_argument("frames_output_dir", help="Directory to save the extracted frames.")
    parser.add_argument("output_video_path", help="Path and filename for the converted video.")
    parser.add_argument("--codec", default="libx264", help="Codec to use for video conversion. Default is 'libx264'.")

    args = parser.parse_args()

    if not is_codec_supported(args.codec):
        print(f"Codec '{args.codec}' is not supported.")
        return

    if(not os.path.exists(args.input_video_path)):
        print(f"Input video file not found: {args.input_video_path}")
        return

    # Make sure the output directory exists
    if not os.path.exists(args.frames_output_dir):
        os.makedirs(args.frames_output_dir)

    # Extract frames
    extract_frames(args.input_video_path, args.frames_output_dir)

    # Convert video format
    convert_video_format(args.input_video_path, args.output_video_path, args.codec)

if __name__ == "__main__":
    main()
