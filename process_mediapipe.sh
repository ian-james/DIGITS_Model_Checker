#!/bin/bash

# Function to display usage information
usage() {
    echo "Usage: $0 -i <videos_directory> -o <csv_output_directory> [-md <md_value>] [-mt <mt_value>] [-mp <mp_value>] [-nh <nh_value>]"
    echo "  -i  Directory containing videos"
    echo "  -o  Directory for CSV output"
    echo "  -n  Number of files to process (default: 200)"
    echo "  -md Minimum detection confidence (default: 0.4)"
    echo "  -mt Minimum tracking confidence (default: 0.4)"
    echo "  -mp Minimum pose confidence (default: 0.25)"
    echo "  -nh Number of hands (default: 1)"
    echo "  -s  Shuffle files"
    exit 1
}

# Default values
md_value=0.4
mt_value=0.4
mp_value=0.25
nh_value=1

num_files=200
shuffle=false

# Parse command-line options
while getopts "i:o:n:s:md:mt:mp:nh:h" opt; do
    case "$opt" in
        i) videos_directory=$OPTARG ;;
        o) csv_output_directory=$OPTARG ;;
        n) num_files=$OPTARG ;;
        md) md_value=$OPTARG ;;
        mt) mt_value=$OPTARG ;;
        mp) mp_value=$OPTARG ;;
        nh) nh_value=$OPTARG ;;
        s) shuffle=true ;;
        h) usage ;;
        *) usage ;;
    esac
done

# Check if mandatory arguments are provided
if [ -z "$videos_directory" ] || [ -z "$csv_output_directory" ]; then
    usage
fi

# Ensure the videos directory exists
if [ ! -d "$videos_directory" ]; then
    echo "Error: Videos directory '$videos_directory' does not exist."
    exit 1
fi

# Ensure the CSV output directory exists or create it
mkdir -p "$csv_output_directory"

# Determine the command for listing and selecting files
if [ "$num_files" -eq 0 ]; then
	file_list_command="ls -1 $videos_directory" 
else
	if $shuffle; then
	    file_list_command="ls -1 $videos_directory | shuf -n $num_files"
	else
	    file_list_command="ls -1 $videos_directory | head -n $num_files"
	fi
fi

echo $videos_directory

echo "\n\n"

echo $file_list_command
eval $file_list_command

# Process videos with the specified Python script
echo "Processing videos to generate CSV files..."
eval $file_list_command | xargs -t -I {} python ./src/main.py -o "$csv_output_directory" -f "$videos_directory{}" -md "$md_value" -mt "$mt_value" -mp "$mp_value" -nh "$nh_value"

echo "Process completed."

