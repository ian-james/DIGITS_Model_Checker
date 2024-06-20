#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -i <images_directory> -o <videos_output_directory> [-n <number_of_files>] [-s]"
    echo "This program takes an image and converts it to a video."
    echo "  -i  Directory containing images"
    echo "  -o  Directory for videos output"
    echo "  -n  Number of files to process (default: 0(all)"
    echo "  -f  FPS (default: 30)"
    echo "  -s  Shuffle files"

    exit 1
}

# Default values
num_files=0
shuffle=false
fps=30
duration=10

# Parse command-line options
while getopts "i:o:d:n:f:sh" opt; do
    case "$opt" in
        i) images_directory=$OPTARG ;;
        o) videos_output_directory=$OPTARG ;;
        d) duration=$OPTARG ;;
        n) num_files=$OPTARG ;;
        s) shuffle=true ;;
	f) fps=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if mandatory arguments are provided
if [ -z "$images_directory" ] || [ -z "$videos_output_directory" ]; then
    usage
fi

# Ensure the images directory exists
if [ ! -d "$images_directory" ]; then
    echo "Error: Images directory '$images_directory' does not exist."
    exit 1
fi

# Ensure the videos output directory exists or create it
mkdir -p "$videos_output_directory"

# Determine the command for listing and selecting files
if [ "$num_files" -eq 0 ]; then
	file_list_command="ls -1 $images_directory" 
else
	if $shuffle; then
	    file_list_command="ls -1 $images_directory | shuf -n $num_files"
	else
	    file_list_command="ls -1 $images_directory | head -n $num_files"
	fi
fi

# Generate videos from images
echo "Generating videos from images..."
eval $file_list_command | xargs -t -I {} python ./src/image_copy_main.py -f "$fps" -o "$videos_output_directory{}.mp4" -d $duration -i "$images_directory{}"

echo "Process completed."

