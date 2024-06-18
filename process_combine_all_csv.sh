#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -d <csv_directory> -o <output_filename>"
    echo "Combines all the CSV files into a single file for analysis."
    echo "  -d  Directory containing CSV files"
    echo "  -o  Output filename for the combined CSV"
    exit 1
}

# Parse command-line options
while getopts "d:o:h" opt; do
    case "$opt" in
        d) csv_directory=$OPTARG ;;
        o) output_filename=$OPTARG ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if mandatory arguments are provided
if [ -z "$csv_directory" ] || [ -z "$output_filename" ]; then
    usage
fi

# Ensure the CSV directory exists
if [ ! -d "$csv_directory" ]; then
    echo "Error: CSV directory '$csv_directory' does not exist."
    exit 1
fi

# Ensure the output directory exists or create it
output_directory=$(dirname "$output_filename")
mkdir -p "$output_directory"

# Combine CSV files with the specified Python script
echo "Combining CSV files..."
python ./src/combine_csvs_main.py -d "$csv_directory" -o "$output_filename"

echo "Process completed."

