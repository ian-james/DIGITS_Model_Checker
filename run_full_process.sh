#!/bin/bash

# Check if the correct number of arguments is provided
# Function to display usage information
usage() {
    echo "Usage: $0 -i <videos_directory> -o <csv_output_directory> [-md <md_value>] [-mt <mt_value>] [-mp <mp_value>] [-nh <nh_value>]"
    echo "This program takes a directory of video(s) and produces a set of mediapipe csv file(s) of hand landmarks."
    echo "You can set specific parameters for mediapipe confidence"
    echo "You can also specify the number of files to hand and if to shuffle the order."
    echo "  -i  Directory containing videos"
    echo "  -o  Directory for CSV output"
    echo "  -a  all_combined csv filename "
    echo "  -n  Number of files to process (default: 0)"
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
nh_value="1"

num_files=0
shuffle=false

all_combined_file="all_combined.csv"


# Parse command-line options
# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i) videos_directory="$2"; shift ;;
        -o) csv_output_directory="$2"; shift ;;
        -n) num_files="$2"; shift ;;
        -a) all_combined_file="$2"; shift ;;
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
echo $nh_value
echo "HERE"

echo num_files
echo shuffle

# Make sure the output directories exist
# Check if mandatory arguments are provided
if [ -z "$videos_directory" ]; then
    echo "Error missing folder for input or output: '$videos_directory'"
    usage
    exit 1
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

# Ensure the CSV output directory exists or create it
# Settings dir is meant to distinguish runs and aligns with python make_output_folder
settings_dir='nh_'$nh_value'_md_'$md_value'_mt_'$mt_value'_mp_'$mp_value'/'

csvs_out=$csv_output_directory$settings_dir'csvs/'
mkdir -p "$csvs_out"

echo -e "\nStarting Mediapipe on video files."
./process_mediapipe.sh -n $num_files -i $videos_directory -o $csvs_out -nh $nh_value -md $md_value -mt $mt_value -mp $mp_value 

if [ $? -eq 0 ]; then
	echo "Process completed."
else
	echo "Process Failed"
        exit 1
fi


angles_dir=$csvs_out'angles/'
echo -e "Starting angles calculations in $angles_dir"
./process_angles.sh -i $csvs_out -o $angles_dir 

if [ $? -eq 0 ]; then
	echo "Process completed."
else
	echo "Process Failed"
        exit 1
fi

stats_dir=$csvs_out'stats/'
#echo $stats_dir
#echo $csvs_out

echo -e "\nCombine Mediapipe Results into stats files."
./process_combine_all_csv.sh -d $csvs_out  -s $stats_dir -o $all_combined_file

if [ $? -eq 0 ]; then
	echo "Process completed."
else
	echo "Process Failed"
        exit 1
fi


excel_out=$csv_output_directory$settings_dir'excel/'
echo -e "\nCalculate Deviation based on mediapipe results."
#echo ">> $excel_out"
./process_deviation.sh -i $csvs_out -o $excel_out

if [ $? -eq 0 ]; then
	echo "Process completed."
else
	echo "Process Failed"
        exit 1
fi




