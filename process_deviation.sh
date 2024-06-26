#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -i <csv_files> -o <excel_output_directory> [-n <number_of_files>] [-s]"
    echo "This program takes a directory of mediapipe csv files and calculates the deviation between all frames in each file to an excel file."
    echo "It also calculates the maximum deviation and the frame it occurs."
    echo "  -i  Directory containing csvs"
    echo "  -o  Directory for excel output"
    echo "  -n  Number of files to process (default: 0(all)"

    exit 1
}

# Default values
num_files=0
shuffle=false

# Parse command-line options
while getopts "i:o:n:f:sh" opt; do
    case "$opt" in
        i) csvs_directory=$OPTARG ;;
        o) excel_output_directory=$OPTARG ;;
        n) num_files=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if mandatory arguments are provided
if [ -z "$csvs_directory" ] || [ -z "$excel_output_directory" ]; then
    usage
fi

# Ensure the csvs directory exists
if [ ! -d "$csvs_directory" ]; then
    echo "Error: No Mediapipe CSVS directory '$csvs_directory' does not exist."
    exit 1
fi

# Determine the command for listing and selecting files
if [ "$num_files" -eq 0 ]; then
	file_list_command="ls -1 $csvs_directory" 
else
	if $shuffle; then
	    file_list_command="ls -1 $csvs_directory | grep -E "*.csv$" | shuf -n $num_files"
	else
	    file_list_command="ls -1 $csvs_directory |  grep -E "*.csv$" |  head -n $num_files"
	fi
fi

# Ensure the excel output directory exists or create it
mkdir -p "$excel_output_directory"

# Generate excel from csvs
echo "Generating excel from csvs..."
eval $file_list_command | grep -E "\.csv$"  | xargs -t -I {} python ./src/compare_deviation_main.py -i "$csvs_directory{}" -o "$excel_output_directory{}" 

echo "Process completed."

