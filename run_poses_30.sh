#!/bin/bash

#closed fist
#finger_circle 

# Define the parameters
fps=30
#database="./datasets/poses_images_lib"
database="./datasets/nyu_test_short_videos/videos/fps_$fps"

annotation_file="all_fps_$fps_combined.csv"
nh_value="1"

for folder in "$database"/*; do
  if [ -d "$folder" ]; then

	#input_dir="$folder/videos/fps_30/"
	input_dir="$folder/"
	output_dir="$folder/analysis_fps_$fps/"
        echo $input_dir

	for md_value in 0.8 0.6 0.4
	do
		./run_full_process.sh -i "$input_dir" -o "$output_dir" -a "$annotation_file" -md "$md_value" -mt "$md_value" -mp "$md_value" -nh "$nh_value"
   		#echo " -i "$input_dir" -o "$output_dir" -a "$annotation_file" -md "$md_value" -mt "$md_value" -mp "$md_value" -nh "$nh_value" "
	done
  fi
done

# Loop through different parameters and run the process


#./run_full_process.sh -i datasets/poses_images_lib videos/fps_30/ -o ./datasets/poses_images_lib/closed_fist/analysis_fps_30/ -a all_fps_30_combined.csv -md 0.8 -mt 0.8 -mp 0.8 -nh 1; 

#./run_full_process.sh -i datasets/poses_images_lib/closed_fist/videos/fps_30/ -o ./datasets/poses_images_lib/closed_fist/analysis_fps_30/ -a all_fps_30_combined.csv -md 0.6 -mt 0.6 -mp 0.6 -nh 1;

#./run_full_process.sh -i datasets/poses_images_lib/closed_fist/videos/fps_30/ -o ./datasets/poses_images_lib/closed_fist/analysis_fps_30/ -a all_fps_30_combined.csv -md 0.6 -mt 0.6 -mp 0.6 -nh 1;

