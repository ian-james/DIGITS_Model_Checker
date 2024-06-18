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
md_value="0.5" 
mt_value="0.5"
mp_value="0.5"
nh_value=1

num_files=0
shuffle=false

# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i) videos_directory="$2"; shift ;;
        -o) csv_output_directory="$2"; shift ;;
        -n) num_files="$2"; shift ;;
        -md) md_value="$2"; shift ;;
        -mt) mt_value="$2"; shift ;;
        -mp) mp_value="$2"; shift ;;
        -nh) nh_value="$2"; shift ;;
        -s) shuffle=true ;;
        -h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done


echo $md_value
echo $mt_value
echo $mp_value

# Check if mandatory arguments are provided
if [ -z "$videos_directory" ] || [ -z "$csv_output_directory" ]; then
    echo "Error missing '$videos_directory'"
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


echo $file_list_command
eval $file_list_command

# Process videos with the specified Python script
echo "Processing videos to generate CSV files..."
eval $file_list_command | xargs -t -I {} python ./src/main.py -o "$csv_output_directory" -f "$videos_directory{}" -md $md_value -mt $mt_value -mp $mp_value -nh $nh_value

echo "Process completed."

