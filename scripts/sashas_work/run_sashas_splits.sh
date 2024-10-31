#!/bin/bash

# Go through each folder and split/ combine angles files if they exist.
# Split for Patient 49,50,51,52,53

output_directory="$HOME/Projects/Western/Western_Postdoc/Analysis/test/"

################### Step 1 #############################################################yy
# Sasha's work was partially put together, so some of the 48-53 have combined files.
# We split those so that we can look at individual views, poses

# Loop through patient numbers 49 to 53
for num in {48..53}; do
  new_dir="${output_directory}C_${num}/csv_angles/"

  # Create the new directory if it doesn't exist
  mkdir -p "$new_dir"

  # Split the angles CSV file for each patient
  python ./src/split_ethans_csv.py \
    -f "$HOME/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_C_${num}/DIGITS_C_${num}_DIGITS_Upload_Output_Trials_1-5.csv" \
    -o "$new_dir"
done

################### Step 2 #############################################################yy
# Sasha's work had put some of the patients together, so we split and extra those
# Split the CSV file that has all patients in one file
python ./src/split_ethans_csv.py \
  -f "$HOME/Projects/Western/Western_Postdoc/Datasets/Sasha_Datasets/DIGITS_Angle_DIGITS_C_48-53.csv" \
  -o "$output_directory"

################### Step 3 #############################################################yy
# Organizing the different split patient data into their own folders
# Patients folders are labeled C_<N>, where N is the number of the patient.
# A sub folder /csv_angles/ is produced for each patient to organize angle data vs other data.
# Organize files for patients 49 to 53
for num in {48..53}; do
  new_dir="${output_directory}C_${num}/csv_angles/"
	echo "Copying files to output directory: ${new_dir}" 

  # Make sure the directory exists (in case it was skipped earlier)
  mkdir -p "$new_dir"

  # Move the split CSV files to the new directory
  find "$output_directory" -maxdepth 1 -type f -name "*C_${num}*" -exec mv {} "$new_dir" \;

done

################### Step 4 #############################################################
# Calculate Stats for Ethans Angles
#
all_combined_file="all_angles_combined.csv"
for num in {48..53}; do
  new_dir="${output_directory}C_${num}/csv_angles/"
	stats_dir="${new_dir}stats/" 
	echo "Calculating Stats for angles folder ${new_dir} and adding stats to ${stats_dir}"	
	echo "Building a file $all_combined_file"
	python ./src/ethan_descriptive_stats.py -d "$new_dir" -o "$all_combined_file" -s "${new_dir}stats/" -e
done
