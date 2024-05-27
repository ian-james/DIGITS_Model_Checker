# Create a video file, for the first 2  images in nyu_RGB

ls -1 ~/Projects/Western/Western_Postdoc/Datasets/reduced/nyu_RGB/ | head -n 2 | xargs -t -I {} python ./src/image_copy_main.py  -o ./output/nyu_tests/{}.mp4 -d ./output/nyu_tests/ -d


# Shuffle the values rather than select all
ls -1 ~/Projects/Western/Western_Postdoc/Datasets/reduced/nyu_RGB/ | shuf -n 200 | xargs -t -I {} python ./src/image_copy_main.py  -o ./output/nyu_tests/videos/{}.mp4  -d 10  -i ~/Projects/Western/Western_Postdoc/Datasets/reduced/nyu_RGB/{}

# Evaluate Mediapipe for each video with parameters
 ls -1 ./output/nyu_tests/videos/ | xargs -t -I {} python ./src/main.py -o ./output/nyu_tests/csvs/ -f ./output/nyu_tests/videos//{} -md 0.4 -mt 0.4 -mp 0.25 -nh 1


 # Combine CSVs from all the movies or frames
 ./src/combine_csvs_main.py -d <DIR> -o <Output_filename>


 # Calculate stats for dataframe
 comparative_main.py -i <input> -o <Output_filename>

 # Extract Frames
 ./video_editing_main.py


 ##############################
 # Visual Tools
 ## Streamlite_graph - Graph a Dataframe and look at individual columns

 ## Streamlit_capture_main - capture mediapipe with multiple mediums