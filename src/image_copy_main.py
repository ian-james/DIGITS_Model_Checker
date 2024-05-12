
import os
import argparse
from pathlib import Path

# User libraries
from video_editing import create_video_from_image

def main():
    parser = argparse.ArgumentParser(description="Create a video from a single image.")    
    parser.add_argument("-i","--input_image", type=str, default="input_frame.png", help="Path to the input video file.")    
    parser.add_argument("-o","--output_video", type=str, default="output_video.mp4", help="Path and filename for the converted video.")
    parser.add_argument("-c","--codec", type=str, default="mp4v", help="Codec to use for video conversion. Default is 'libx264'.")
    parser.add_argument("-d","--duration", type=int, default=5, help="Duration of the output video in seconds. Default is 5 seconds.")

    args = vars(parser.parse_args())
    print(args)

    # Check if the input video file exists    
    if( not os.path.exists(args['input_image']) ):
        print(f"Input video file not found: {args['input_image']}")
        return
    
    # If the output video is the empty string, then set it to the input filename with the extension changed to .mp4
    if( args['output_video'] == ""):
        args['output_video'] = Path(args['input_image']).stem + ".mp4"

    # Extract frames
    create_video_from_image(args['input_image'], args['output_video'],args['duration'], args['codec'])


if __name__ == "__main__":
    main()