#!/bin/bash

# Default values
videos_directory=""
csv_output_directory=""
num_files=0
md_value=""
mt_value=""
mp_value=""
nh_value=""
shuffle=false

# Function to display usage
usage() {
    echo "Usage: $0 [-i videos_directory] [-o csv_output_directory] [-n num_files] [-s] [-md md_value] [-mt mt_value] [-mp mp_value] [-nh nh_value] [-h]"
    exit 1
}

# Parse command-line options
while getopts "i:o:n:md:mt:mp:nh:sh" opt; do
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

# Check for required parameters
if [ -z "$videos_directory" ] || [ -z "$csv_output_directory" ]; then
    echo "Error: Missing required parameters."
    usage
fi

# Your script logic here
echo "Videos Directory: $videos_directory"
echo "CSV Output Directory: $csv_output_directory"
echo "Number of Files: $num_files"
echo "MD Value: $md_value"
echo "MT Value: $mt_value"
echo "MP Value: $mp_value"
echo "NH Value: $nh_value"
echo "Shuffle: $shuffle"


