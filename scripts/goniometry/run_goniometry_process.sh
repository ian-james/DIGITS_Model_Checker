#!/bin/bash
# This Converts Goniometry to comparable method to Mediapipe.

usage() {
    echo "Usage: $0 -i <videos_directory> -o <csv_output_directory>"
    echo "This program converts Goniometry files."

    echo "  -i  Directory containing videos"
    echo "  -o  Directory for CSV output"
    exit 1
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i) videos_directory="$2"; shift ;;
        -o) csv_output_directory="$2"; shift ;;
        -h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

# Take the left and right goniometry hand measuremnts csv files and format and convert to a single csv file.
python3 ./src/handle_goniometery_measurements.py -i datasets/uwo/DIGITS_Right_Hand_Goniometry_Measurements.csv -l datasets/uwo/DIGITS_Left_Hand_Goniometry_Measurements.csv -o datasets/uwo/formatted_goniometry_measures.csv 

# formatted_goniometry_measurement file should be created.
python ./src/handle_all_angle_files.py -f ~/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/formatted_goniometry_measures.csv -m ~/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/all_angles_combined.csv -d ~/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/handle_all_angles/

# Compare one group vs another (one and two or mine theirs)
# Creates Mine/Theirs/Combined/Compare folders
python ./src/compare_each_pvp.py  -f "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/df_cleaned.csv" -m
 -f "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/mdf_cleaned.csv" 

# Compare each patient for digit-pose-view-joint evaluation
python ./src/compare_all_patients.py -i "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/combined/" -d "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/final/"


python ./src/develop_comparison_tables.py -i "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/combined/"  -d "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/final/"

echo "Finished Process"
