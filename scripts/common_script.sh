#!/bin/bash

# Function to display usage information
usage() {
    echo "This program combines all the CSV files into a single file for analysis."
    echo "The program takes a directory containing CSV files represending landmarks, angles, or other data."
    echo 
    echo "Usage: $0 -i <input_directory> -o <output_directory> [-s <stats_directory>] [-m <stats>] [-p <python_script>]"
    echo "  -i  Directory containing CSV files"
    echo "  -o  Output directory for CSV output"
    echo "  -c  All combined CSV filename"
    echo "  -s  Stats directory (optional) if you want to save stats files to a different directory"
    echo "  -m  Stats (optional) - specify the stats file to save (e.g., max, min, mean)"
    echo "  -p  Python script to use (optional, default is combine_csvs_main.py)" 
    exit 1
}

# Default script to use
python_script="./src/combine_csvs_main.py"

# Parse command-line options
while getopts "i:o:c:s:m:p:h" opt; do
    case "$opt" in
        i) csv_directory=$OPTARG ;;
        o) output_directory=$OPTARG ;;
        c) all_combined_file=$OPTARG ;;
        s) stats_dir=$OPTARG ;;
        m) stats=$OPTARG ;;
        p) python_script=$OPTARG ;; # Specify Python script via argument
        h) usage ;;
        *) usage ;;
    esac
done

# Ensure the directory paths always end with a trailing slash
csv_directory="${csv_directory%/}/"
output_directory="${output_directory%/}/"
stats_dir="${stats_dir%/}/"

# Ensure mandatory arguments are provided
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

# Ensure the stats directory exists or create it
mkdir -p "$stats_dir"

# Set a default filename for output (modify as needed)
output_filename="${output_directory}/$all_combined_file"

# Run the Python script dynamically based on the script passed or default
echo "Combining CSV files using $python_script ..."
python "$python_script" -d "$csv_directory" -o "$output_filename" -s "$stats_dir" -m "$stats"

echo "Process completed."
