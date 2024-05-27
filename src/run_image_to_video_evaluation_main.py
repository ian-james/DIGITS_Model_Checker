import argparse
import os
import random
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images to videos and run Mediapipe")
    parser.add_argument("-idir", "--images_directory", required=True, help="Directory containing images")
    parser.add_argument("-vdir", "--videos_output_directory", required=True, help="Directory for videos output")
    parser.add_argument("-cdir", "--csv_output_directory", required=True, help="Directory for CSV output")
    parser.add_argument("-s", "--shuffle", action="store_true", help="Shuffle files")
    parser.add_argument("-n", "--number_of_files", type=int, default=0, help="Number of files to process")
    parser.add_argument("-dbg", "--debug", action="store_true", help="Debug mode")

    # Image to video arguments
    parser.add_argument("-fps", "--frames_per_second", type=int, default=10, help="Frames per second for the video")
    parser.add_argument("-c", "--codec", type=str, default="mp4v", help="Codec to use for video conversion")
    parser.add_argument("-d", "--duration", type=int, default=5, help="Duration of the output video in seconds")    

    # Mediapipe arguments
    # -md 0.4 -mt 0.4 -mp 0.25 -nh 1
    parser.add_argument("-md", "--min_detection_confidence", type=float, default=0.8, help="Minimum detection confidence")
    parser.add_argument("-mt", "--min_tracking_confidence", type=float, default=0.8, help="Minimum tracking confidence")
    parser.add_argument("-mp", "--min_presense_confidence", type=float, default=0.5, help="Minimum presence confidence")
    parser.add_argument("-nh", "--num_hands", type=int, default=2, help="Number of hands to detect")

    # Option to run each part of the process as needed.
    parser.add_argument("-C", "--Convert", action="store_false", help="Convert the image to videos"  )  
    parser.add_argument("-M", "--Mediapipe", action="store_false", help="Run Mediapipe on the videos")
    parser.add_argument("-J", "--Join", action="store_false", help="Join all CSV files into one")

    return parser.parse_args()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_files(images_directory, shuffle, num_files):
    files = os.listdir(images_directory)
    if shuffle:
        random.shuffle(files)
    return files[:num_files]

def run_command(command, mode):
    if(mode == "debug"):
        print(f"Would Run command: {command}")
    else:
        subprocess.run(command, shell=True, check=True)

def main():
    args = parse_arguments()
    
    ensure_directory_exists(args.videos_output_directory)
    ensure_directory_exists(args.csv_output_directory)
    is_debug = args['debug']

    files_to_process = get_files(args.images_directory, args.shuffle, args.number_of_files)
    
    if(args.Convert):
        print("Generating videos from images...")
        for file in files_to_process:
            image_path = os.path.join(args.images_directory, file)
            video_path = os.path.join(args.videos_output_directory, f"{file}.mp4")
            # If the video already exists, skip it.
            if os.path.exists(video_path):
                print(f"Video already exists: {video_path}")
                continue
            command = f"python ./src/image_copy_main.py -o {video_path} -d {args['duration']} -i {image_path} -f{args['frames_per_second']} -c {args['codec']}"
            run_command(command,is_debug)

    if(args["m"]):
        print("Running Mediapipe on videos...")
        video_files = os.listdir(args.videos_output_directory)
        for video_file in video_files:
            video_path = os.path.join(args.videos_output_directory, video_file)
            media_args = f" -md {args['min_detection_confidence']} -mt {args['min_tracking_confidence']} -mp {args['min_presense_confidence']} -nh {args['num_hands']}"
            command = f"python ./src/main.py -o {args.csv_output_directory} -f {video_path} {media_args}"
            run_command(command, is_debug)

    if(args['Join']):
        print("Combining all CSV files...")
        command = f"python ./src/combine_csvs_main.py -d {args.csv_output_directory} -o {args.csv_output_directory}/all_combined_csvs.csv"
        run_command(command,is_debug)

    print("Process completed.")

if __name__ == "__main__":
    main()

