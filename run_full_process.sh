#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <images_directory> <videos_output_directory> <csv_output_directory>"
    exit 1
fi

# Get the input parameters
images_directory=$1
videos_output_directory=$2
csv_output_directory=$3

# Make sure the output directories exist
mkdir -p "$videos_output_directory"
mkdir -p "$csv_output_directory"

# Generate videos from images
echo "Generating videos from images..."
ls -1 "$images_directory" | shuf -n 200 | xargs -t -I {} python ./src/image_copy_main.py -o "$videos_output_directory/{}.mp4" -d 10 -i "$images_directory/{}"

# Run Mediapipe
echo "Running Mediapipe on videos..."
ls -1 "$videos_output_directory" | xargs -t -I {} python ./src/main.py -o "$csv_output_directory" -f "$videos_output_directory/{}" -md 0.4 -mt 0.4 -mp 0.25 -nh 1

# Combine all CSVs
echo "Combining all CSV files..."
python ./src/combine_csvs_main.py -d "$csv_output_directory" -o "$csv_output_directory/all_combined_csvs.csv"

echo "Process completed."

