
#!/bin/bash
#
usage() {
    echo "Usage: $0 -i <videos_directory> -o <csv_output_directory> [-md <md_value>] [-mt <mt_value>] [-mp <mp_value>] [-nh <nh_value>]"
    echo "This program takes a directory of video(s) and produces a set of mediapipe csv file(s) of hand landmarks."

    echo "  -i  Directory containing csv"
    echo "  -o  Directory for CSV output"
    echo "  -a  all_combined csv filename "
    exit 1
}

# Define the parameters
csv_directory=""
output_directory=""
output_file="ROM_"

# Parse command-line options
# Parse command-line options
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -i) videos_directory="$2"; shift ;;
        -o) csv_output_directory="$2"; shift ;;
        -a) all_combined_file="$2"; shift ;;
        -h) usage ;;
        *) echo "Unknown option: $1"; usage ;;
    esac
    shift
done

if [ -z "$csv_directory" ]; then
    echo "Error missing folder for input or output: '$csv_directory'"
    usage
    exit 1
fi

# Ensure the CSV output directory exists or create it
mkdir -p "$csv_output_directory"

python ./src/main.py -o "$csv_output_directory" -f "$videos_directory{}" 

for file in *.csv
do
  # Check if the file exists to avoid any errors
  if [ -f "$file" ]; then
    echo "Processing $file..."
    # Run the Python script on the CSV file
    python ./src/calculate_joints_and_length.pyth 
  else
    echo "No CSV files found in the current directory."
  fi
done

if [ $? -eq 0 ]; then
	echo "Process completed."
else
	echo "Process Failed"
        exit 1
fi


