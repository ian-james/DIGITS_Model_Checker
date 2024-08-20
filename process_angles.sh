#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -d <output_directory> -o <output_directory>"
    echo "Combines all the CSV files into a single file for analysis."
    echo "  -i  Directory containing CSV files"
    echo "  -o  Output directory CSV"
    exit 1
}

# Parse command-line options
while getopts "i:o:h" opt; do
    case "$opt" in
        i) csv_directory=$OPTARG ;;
        o) output_directory=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if mandatory arguments are provided
if [ -z "$csv_directory" ] || [ -z "$output_directory" ]; then
    usage
fi

# Ensure the CSV directory exists
if [ ! -d "$csv_directory" ]; then
    echo "Error: CSV directory '$csv_directory' does not exist."
    exit 1
fi

# Ensure the output directory exists or create it
mkdir -p "$output_directory"

# Determine the command for listing and selecting files
file_list_command="ls -1 $csv_directory" 

# Combine CSV files with the specified Python script
echo "Calculate Angles CSV files..."
eval $file_list_command | xargs -t -I {} python ./src/calculate_joints_and_length.py -f  "$csv_directory{}" -o "$output_directory{}"

echo "Process completed."

