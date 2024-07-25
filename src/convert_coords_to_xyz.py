import os
import argparse
import pandas as pd

from calculate_joints_and_length import convert_csv_to_landmarks, convert_csv_with_xyz_to_landmarks

def setup_arguments():
    # Allow an input and output files to be passed in as arguments    
       # Initialize the argument parser
    ap = argparse.ArgumentParser()
    #ap.add_argument("-f", "--filename", type=str, default="./output/IMG_20220430_180941_fps_60_mediapipe_nh_1_md_0.4_mt_0.4_mp_0.4.csv", help="Path to the input CSV file.")
    ap.add_argument("-f", "--filename", type=str, default="./output/rgb_1_0000001_fps_30_mediapipe_nh_2_md_0.8_mt_0.8_mp_0.5.csv", help="Path to the input CSV file.")
    ap.add_argument("-o", "--output", type=str, default="./output/output_landmarks.csv", help="Path to the output CSV file.")
    ap.add_argument("-c","--convert_to_coordinates", action="store_true", help="Convert the landmarks to XYZ coordinates.")
    ap.add_argument("-s","--separator", type=str, default="\t", help="Separator for the CSV file.")
    return ap

def main():
    
    # Setup Arguments
    ap = setup_arguments()
    args = vars(ap.parse_args())
    # Check if the input file exists
    if not os.path.exists(args["filename"]):
        print(f"File not found: {args['filename']}")
        return
    
    try:
        df = pd.read_csv(args["filename"],sep="\t")

        # Convert the CSV file to a list of landmarks
        landmarks = None
        if(args['convert_to_coordinates']):
            landmarks = convert_csv_to_landmarks(df)
        else:
            landmarks = convert_csv_with_xyz_to_landmarks(df)
        
        if(landmarks is None):
            print("No landmarks found.")
            return
    
        # Save the landmarks to a new CSV file
        landmarks.to_csv(args['output'], index=False)
        print("Landmarks saved to: ", args['output'])
    except Exception as e:
        print(f"Error processing file: {args['filename']}")
        print(e)

if __name__ == "__main__":
    main()