#!/bin/bash
# This script converts Goniometry to a comparable method to Mediapipe.

usage() {
    echo "Usage: $0 -i <videos_directory> -o <csv_output_directory> -g <goniometry_dir> -p <processed_videos_dir>"
    echo "This program converts Goniometry files."

    echo "  -i  Directory containing videos"
    echo "  -o  Directory for CSV output"
    echo "  -g  Directory containing goniometry files"
    echo "  -p  Directory for processed videos"
    exit 1
}

# Initialize variables
videos_directory=""
csv_output_directory=""
goniometry_dir=""
processed_videos_dir=""

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i) videos_directory="$2"; shift ;;
        -o) csv_output_directory="$2"; shift ;;
        -g) goniometry_dir="$2"; shift ;;
        -p) processed_videos_dir="$2"; shift ;;
        -h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Validate required arguments
if [[ -z "$videos_directory" || -z "$csv_output_directory" || -z "$goniometry_dir" || -z "$processed_videos_dir" ]]; then
    echo "Missing required arguments."
    usage
fi

# Paths for goniometry files and output
right_hand_goniometry_file="$goniometry_dir/DIGITS_Right_Hand_Goniometry_Measurements.csv"
left_hand_goniometry_file="$goniometry_dir/DIGITS_Left_Hand_Goniometry_Measurements.csv"
formatted_goniometry_output="$csv_output_directory/formatted_goniometry_measures.csv"

# Paths for processed video files
combined_angles_file="$processed_videos_dir/all_angles_combined.csv"
handle_all_angles_dir="$processed_videos_dir/results"
cleaned_file1="$handle_all_angles_dir/df_cleaned.csv"
cleaned_file2="$handle_all_angles_dir/mdf_cleaned.csv"
combined_dir="$handle_all_angles_dir/combined/"
final_dir="$handle_all_angles_dir/test/final/"

# Take the left and right goniometry hand measurements CSV files, format, and convert to a single CSV file.
python3 ./src/handle_goniometery_measurements.py -i "$right_hand_goniometry_file" -l "$left_hand_goniometry_file" -o "$formatted_goniometry_output"

# Use the formatted goniometry measurements file for further analysis.
python3 ./src/handle_all_angle_files.py -f "$formatted_goniometry_output" -m "$combined_angles_file" -d "$handle_all_angles_dir/"

# Compare one group vs another
python3 ./src/compare_each_pvp.py -f "$cleaned_file1" -m -f "$cleaned_file2"

# Compare each patient for digit-pose-view-joint evaluation
python3 ./src/compare_all_patients.py -i "$combined_dir" -d "$final_dir"

# Develop comparison tables
python3 ./src/develop_comparison_tables.py -i "$combined_dir" -d "$final_dir"

echo "Finished Process"

