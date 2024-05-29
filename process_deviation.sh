#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -i <csv_files> -o <excel_output_directory> [-n <number_of_files>] [-s]"
    echo "This program takes an image and converts it to a video."
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
    echo "Error: Images directory '$csvs_directory' does not exist."
    exit 1
fi

# Ensure the excel output directory exists or create it
mkdir -p "$excel_output_directory"

# Determine the command for listing and selecting files
if [ "$num_files" -eq 0 ]; then
	file_list_command="ls -1 $csvs_directory" 
else
	if $shuffle; then
	    file_list_command="ls -1 $csvs_directory | shuf -n $num_files"
	else
	    file_list_command="ls -1 $csvs_directory | head -n $num_files"
	fi
fi

# Generate excel from csvs
echo "Generating excel from csvs..."
eval $file_list_command | xargs -t -I {} python ./src/compare_deviation_main.py -o "$excel_output_directory{}.xlsx"  -i "$csvs_directory/{}"

echo "Process completed."

