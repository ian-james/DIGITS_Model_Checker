import argparse
import os
import random
import subprocess

def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images to videos and run Mediapipe")
    parser.add_argument("-i", "--images_directory", required=True, help="Directory containing images")
    parser.add_argument("-v", "--videos_output_directory", required=True, help="Directory for videos output")
    parser.add_argument("-c", "--csv_output_directory", required=True, help="Directory for CSV output")
    parser.add_argument("-s", "--shuffle", action="store_true", help="Shuffle files")
    parser.add_argument("-n", "--number_of_files", type=int, default=200, help="Number of files to process")
    return parser.parse_args()

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_files(images_directory, shuffle, num_files):
    files = os.listdir(images_directory)
    if shuffle:
        random.shuffle(files)
    return files[:num_files]

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def main():
    args = parse_arguments()
    
    ensure_directory_exists(args.videos_output_directory)
    ensure_directory_exists(args.csv_output_directory)

    files_to_process = get_files(args.images_directory, args.shuffle, args.number_of_files)
    
    print("Generating videos from images...")
    for file in files_to_process:
        image_path = os.path.join(args.images_directory, file)
        video_path = os.path.join(args.videos_output_directory, f"{file}.mp4")
        command = f"python ./src/image_copy_main.py -o {video_path} -d 10 -i {image_path}"
        run_command(command)

    print("Running Mediapipe on videos...")
    video_files = os.listdir(args.videos_output_directory)
    for video_file in video_files:
        video_path = os.path.join(args.videos_output_directory, video_file)
        command = f"python ./src/main.py -o {args.csv_output_directory} -f {video_path} -md 0.4 -mt 0.4 -mp 0.25 -nh 1"
        run_command(command)

    print("Combining all CSV files...")
    command = f"python ./src/combine_csvs_main.py -d {args.csv_output_directory} -o {args.csv_output_directory}/all_combined_csvs.csv"
    run_command(command)

    print("Process completed.")

if __name__ == "__main__":
    main()



# Linux Commands
#ls -1 ~/Projects/Western/Western_Postdoc/Datasets/reduced/nyu_RGB/ | head -n 2 | xargs -t -I {} python ./src/image_copy_main.py  -o ./output/nyu_tests/videos/{}.mp4  -d 10  -i ~/Projects/Western/Western_Postdoc/Datasets/reduced/nyu_RGB/{}#

#ls -1 ./output/nyu_tests/videos/ | xargs -t -I {} python ./src/main.py -o ./output/nyu_tests/csvs/ -f ./output/nyu_tests/videos//{} -md 0.4 -mt 0.4 -mp 0.25 -nh 1
#ls -1 ./output/nyu_tests/videos/ | xargs -t -I {} python ./src/main.py -o ./output/nyu_tests/csvs/ -f ./output/nyu_tests/videos//{} -md 0.6 -mt 0.6 -mp 0.6 -nh 1
#ls -1 ./output/nyu_tests/videos/ | xargs -t -I {} python ./src/main.py -o ./output/nyu_tests/csvs/ -f ./output/nyu_tests/videos//{} -md 0.8 -mt 0.8 -mp 0.8 -nh 1