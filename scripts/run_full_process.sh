#!/bin/bash

# Default values
md_value="0.5" 
mt_value="0.5"
mp_value="0.5"
nh_value="1"
all_combined_file="all_combined.csv"
all_angles_combined_file="all_angles_combined.csv"

# Run Sections
skip_mediapipe=false
skip_deviation=false
skip_angles_length=false
skip_combine_all_landmarks=false
skip_combine_all_angles=false
skip_thumb_circumduction=false
skip_flexion_extension_conversion=false

scripts_run=()

# Check if the correct number of arguments is provided
# Function to display usage information
usage() {
    echo "Usage: $0 -i <videos_directory> -o <csv_output_directory> [-md <md_value>] [-mt <mt_value>] [-mp <mp_value>] [-nh <nh_value>]"
    echo "This program takes a directory of video(s) and produces a set of mediapipe csv file(s) of hand landmarks."
    echo "You can set specific parameters for mediapipe confidence"
    echo "You can also specify the number of files to hand and if to shuffle the order."
    echo "  -i  Directory containing videos"
    echo "  -o  Directory for CSV output"
    echo "  -l  all_combined csv filename "
    echo "  -a  all_angles_combined csv filename "
    echo "  -md Minimum detection confidence (default: $md_value)"
    echo "  -mt Minimum tracking confidence (default: $mt_value)"
    echo "  -mp Minimum pose confidence (default: $mp_value)"
    echo "  -nh Number of hands (default: $nh_value)"
    echo " -skip_mediapipe"
    echo " -skip_deviation"
    echo " -skip_combine_all_landmarks"
    echo " -skip_combine_all_angles"
    echo " -skip_angles_length"
    echo " -skip_thumb_circumduction"
    echo " -skip_flexion_extension_conversion"
    exit 1
}

# Parse command-line options
# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i) videos_directory="$2"; shift ;;
        -o) csv_output_directory="$2"; shift ;;
        -l) all_combined_file="$2"; shift ;;
        -a) all_angles_combined_file ="$2"; shift ;;
        -md) md_value="$2"; shift ;;
        -mt) mt_value="$2"; shift ;;
        -mp) mp_value="$2"; shift ;;
        -nh) nh_value="$2"; shift ;;
	    --skip_mediapipe) skip_mediapipe=true ;;
        --skip_deviation) skip_deviation=true ;;
        --skip_combine_all_landmarks) skip_combine_all_landmarks=true ;;
        --skip_combine_all_angles) skip_combine_all_landmarks=true ;;
        --skip_angles_length) skip_angles_length=true ;;
        --skip_thumb_circumduction) skip_thumb_circumduction=true ;;
        --skip_flexion_extension_conversion) skip_flexion_extension_conversion=true ;;
        -h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

################### Step 1 #############################################################yy
# Setup/Check Default Values for upcoming scripts
# Build Directories as needed.

echo "Mediapipe Model Settings:"
echo "Model Detection: $md_value"
echo "Model Tracking: $mt_value"
echo "Model Pose: $mp_value"
echo "Number of Hands: $nh_value"

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

# Ensure the CSV output directory exists or create it
# Settings dir is meant to distinguish runs and aligns with python make_output_folder
settings_dir='nh_'$nh_value'_md_'$md_value'_mt_'$mt_value'_mp_'$mp_value'/'

# Setup the csv directory for the results
csvs_out=$csv_output_directory$settings_dir'csvs/'
mkdir -p "$csvs_out"

################### Step 2 #############################################################yy
# Run Mediapipe on a folder of video files and produce results
# Can change model settings through nh, md, mt, mp settings.
if $skip_mediapipe; then
    echo "Skipping Mediapipe evaluaton videos"
else
    scripts_run+=("Mediapipe")
    echo "################### Step 1 #############################################################"
    echo -e "\nStarting Mediapipe on video files."

    ./scripts/mediapipe/process_mediapipe.sh -i $videos_directory -o $csvs_out -nh $nh_value -md $md_value -mt $mt_value -mp $mp_value 

    if [ $? -eq 0 ]; then
        echo "Process completed."
    else
        echo "Process Failed"
        exit 1
    fi
fi

################### Step 3 #############################################################yy
# Process the angle data
angles_dir=$csvs_out'angles/'

if $skip_angles_length; then
    echo "Skipping calculating the angles"
else
    echo "################### Step 2 #############################################################"
    echo -e "Starting angles calculations in $angles_dir"

    scripts_run+=("Angles")
    ./scripts/workflow/process_angles.sh -i $csvs_out -o $angles_dir 

    if [ $? -eq 0 ]; then
        echo "Process completed."
    else
        echo "Process Failed"
            exit 1
    fi
fi

################### Step x #############################################################yy
# Calculate Thumb Circumduction
#

################### Step y #############################################################yy
# Flexion/Extension 

################### Step 4 #############################################################yy
# Create combination of landmark data combined file of all the videos and angles.
#
if $skip_combine_all_landmarks; then
    echo "Skipping combine all"
else
    echo "################### Step 3 #############################################################"
    echo -e "\nCombine Mediapipe Results into stats files."
    echo "Combining all the results file into a single file."

    scripts_run+=("Combine All")
    stats_dir=$csvs_out'stats/'
    mkdir -p "$stats_dir"

     ./scripts/workflow/process_combine_all_csv.sh -d $csvs_out  -s $stats_dir -o $all_combined_file
    # python ./src/ethan_descriptive_stats.py -d "$csvs_out" -o "$all_combined_file" -s "$stats_dir"

    if [ $? -eq 0 ]; then
	    echo "Process completed."
    else
	    echo "Process Failed"
           exit 1
    fi
fi

################### Step 4 #############################################################yy
# Create a combined file of all the videos and angles.
#
if $skip_combine_all_landmarks; then
    echo "Skipping combine all"
else
    echo "################### Step 4 #############################################################"
    echo -e "\nCombine Mediapipe Angles into a single file."

    scripts_run+=("Combine All Angles")
    stats_angles_dir=$csvs_out'angle_stats/'
    mkdir -p "$stats_angles_dir"

    python ./src/ethan_descriptive_stats.py -d "$angles_dir" -o "$all_angles_combined_file" -s "$stats_angles_dir"
    if [ $? -eq 0 ]; then
	    echo "Process completed."
    else
	    echo "Process Failed"
           exit 1
    fi
fi


################### Step 5 #############################################################yy
# Create a combined file of all the videos and angles.
# Check if the deviation flag is set
if $skip_deviation; then 
    echo "Skipping deviation"
else
    echo "################### Step 5 #############################################################"
    echo -e "\nCalculate Deviation based on mediapipe results."
    scripts_run+=("Deviation")
    excel_out=$csv_output_directory$settings_dir'excel/'
    
    #echo ">> $excel_out"
    ./scripts/workflow/process_deviation.sh -i $csvs_out -o $excel_out

    if [ $? -eq 0 ]; then
        echo "Process completed."
    else
        echo "Process Failed"
        exit 1
    fi
fi


################### Final Step #############################################################
echo "################### Scripts Run ###########################################################"
for script in "${scripts_run[@]}"; do
    echo "Script: $script"
done

################### Step y #############################################################yy
# Ensure all patients are calculate and a combined version for all_patients..etc 
#

################### Step y #############################################################yy
# 1)Move to another script to have comparisons between models 
# 2) Have a script to compare to goniometry
# 3) Script to summarize the final tables
# 4) Script to summarize which views are better or worse
# 	flexion + in model 1          flexion - in model 2
# 	pose + in model 2
# 	view - in model 1....etc








