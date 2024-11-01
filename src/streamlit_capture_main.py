import os
from pathlib import Path
import base64
from pathlib import Path
from io import BytesIO

import streamlit as st
from moviepy.editor import VideoFileClip

import cv2
from opencv_file_utils import open_image

import numpy as np
import plotly.express as px

# MediaPipe Includes
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dataframe_actions import *
from file_utils import *

from stats_functions import sem

from Capture_Info import *

import matplotlib.pyplot as plt

from fps_timer import FPS
from VideoCap import VideoCap_Info

import convert_mediapipe_index as convert_mp

from mediapipe_helpers import *

from scipy import stats
import project_settings as ps

from streamlit_utils import save_uploadedfile,  display_download_buttons, download_image

# Calculate Angles
from calculate_joints_and_length import convert_csv_with_xyz_to_landmarks, calculate_all_finger_angle_df

import tempfile

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def change_filename(filepath, new_filename):
    # Get the directory and extension of the original file
    directory = os.path.dirname(filepath)
    extension = os.path.splitext(filepath)[1]

    # Create the new file path with the updated filename
    new_filepath = os.path.join(directory, new_filename + extension)
    return new_filepath

def convert_to_mp4(video_path, output_path, codec='libx264'):
    try:
        clip = VideoFileClip(video_path)
        clip.write_videofile(output_path, codec=codec)
        clip.close()
        return True
    except Exception as e:
        st.error(f"Failed to convert video file named {video_path} to {output_path} with codec {codec}. {e}")
    return False

def save_uploadedfile(uploadedfile, folder='tempDir'):
    location =  os.path.join(folder, uploadedfile.name)
    st.write("Saving File:{} to {}".format(uploadedfile.name, location))
    try:
        with open(location, "wb") as f:
            f.write(uploadedfile.getbuffer())
    except:
        return None, st.error(f"Failed to save the {uploadedfile.name}.")

    return location, st.success("Saved File:{} to tempDir".format(uploadedfile.name))

def set_state_option(key,value):
    if key not in st.session_state:
        st.session_state[key] = value
    else:
        st.session_state[key] = value


def get_state_option(key):
    if key not in st.session_state:
        return None
    else:
        return st.session_state[key]

def flip_video_state(key):
    set_state_option(key, not get_state_option(key))

def allow_download_button(file_path):
    with open(file_path, 'rb') as my_file:
        st.download_button(label='Download', data=my_file, file_name='filename.xlsx',
                       mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')

def convert_image_to_bytes(image, format='png'):
    """
    Converts an OpenCV image to a BytesIO object in the specified format.

    Args:
        image: The OpenCV image to be converted (numpy array).
        format: The desired file format ('png', 'jpg', 'jpeg', 'bmp').

    Returns:
        A BytesIO object containing the encoded image.
    """
    # Map file formats to their correct encoding string
    format = format.lower()
    valid_formats = {'png': '.png', 'jpg': '.jpg', 'jpeg': '.jpg', 'bmp': '.bmp', 'svg': '.svg'}

    if format not in valid_formats:
        raise ValueError(f"Unsupported format: {format}. Supported formats: {list(valid_formats.keys())}")

    # Encode the image to the specified format
    is_success, buffer = cv2.imencode(valid_formats[format], image)
    if not is_success:
        raise ValueError(f"Failed to encode image to {format.upper()} format.")

    # Convert the buffer to BytesIO for download
    return BytesIO(buffer)


def run_original_streamlit_video_mediapipe_main(cap, frame_processor):

    # Streamlit UI Options.
    original_placeholder = st.empty()
    frame_placeholder = st.empty()
    with st.expander("See Data Table"):
        datatable_placeholder = st.empty()

    fps_text = st.empty()

    fps = FPS()
    fps.start()

    while cap.is_opened():

        success, image = cap.get_frame()
        fps.update()

        if not success:
            if(cap.is_video()):
                logging.info("Finished the video.")
                break
            else:

                logging.info("Ignoring empty camera frame.")
                continue

        original_placeholder.image(image=image, caption="Original Image", channels="BGR")

        bimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results, rimage = frame_processor.process_frame(bimage,fps.total_num_frames)
        frame_placeholder.image(image=rimage, caption="Enhanced Image", channels="RGB")

    st.write("## FINISHED ANALYSIS")
    frame_processor.finalize_dataframe()
    datatable_placeholder.dataframe(frame_processor.get_dataframe(), hide_index=True)  # You can also use st.write(df)
    return frame_processor.get_dataframe()

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename, sep='\t')
    cols = convert_mp.convert_all_columns_to_friendly_name(df,[])
    df.columns = cols
    return df

def compute_statistics(df, exclude_columns=[]):

    if( df is None):
        return None

    # Exclude specific columns
    if exclude_columns:
        df = df.drop(columns=exclude_columns, errors='ignore')

    # Compute the statistics for each column
    stats_fun = ['max', 'min', 'mean', 'median', 'std'] #, sem]
    stats_df = df.agg(stats_fun)

    return stats_df


def anaysis_image(file_directory,filename, frame_processor):

    image = open_image(filename)
    if (image is not None):
        if Path(filename).exists():

            frame_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results, rimage = frame_processor.process_frame(frame_rgb,0)
            st.image(image=rimage, caption="Enhanced Image", channels="RGB")

            # Display the DataFrame
            st.write("## Frame Data")
            frame_processor.finalize_dataframe()

            df = frame_processor.get_dataframe()

            df.columns = convert_mp.convert_all_columns_to_friendly_name(frame_processor.get_dataframe(),['timestamp','time','handedness'])
            st.dataframe(df, hide_index=True)

            if(filename is not None):
                display_download_buttons(frame_processor.get_dataframe(), os.path.join(file_directory, Path(filename).stem))

            # Calculate the angles based on the landmarks
            df = convert_csv_with_xyz_to_landmarks(df)

            fangles = calculate_all_finger_angle_df(df, None)
            fdf = pd.DataFrame.from_dict(fangles)

            st.write("## Angle Data")
            st.dataframe(fdf, hide_index=True)

            # Download the angles files
            if(filename is not None):
                display_download_buttons(fdf, os.path.join(file_directory,"angles_", Path(filename).stem))
                st.write("## Download Images")
                st.download_button(label='Download Image', data=convert_image_to_bytes(image), file_name="image.png", mime="image/png")
                bimage = cv2.cvtColor(rimage, cv2.COLOR_RGB2BGR)
                st.download_button(label='Download Enhanced Image', data=convert_image_to_bytes(bimage), file_name="enhanced_image.png", mime="image/png")
        else:
            st.write(f"Video file is not open {filename}")


def main():
    project_directory = ps.DIR_MEDIA

    title = st.title("Mediapipe Evaluation Tool")
    # STEP 2: Create an HandLandmarker object.

    # If the frame processor was not previously loaded, then load it now.
    if 'frame_processor' not in st.session_state:
        frame_processor = FrameProcessor(get_hand_model())

    ################################################################################
    # MediaPipe Options
    mediapipe_expander = st.sidebar.expander("## User Options", expanded=True)
    mediapipe_expander.markdown("## Mediapipe Options")
    st.markdown("## Program Options")
    mode_src = st.selectbox("Select the mode", ['None', 'Video', 'Image', "Camera", "Camera Capture", "Graph Data"])
    ################################################################################
    # Debug Options
    debug_expander = st.sidebar.expander("## Developer Options", expanded=False)
    debug_expander.markdown("## Debugging Controls")

    debug_levels = ['debug', 'info', 'warning', 'error', 'critical']
    debug_expander.selectbox("Set the logging level", debug_levels)

    if (mode_src == 'Image'):
        title.title("Image Analysis")
        st.subheader("Analyse a single image.")
        st.divider()
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg",
                                                                       "png", "jpeg"])
        st.write(uploaded_file)
        if(uploaded_file):
            if uploaded_file is not None:
                # To read file as bytes:
                file_directory = os.path.join(project_directory, ps._SAVED_IMAGES)
                filename, result = save_uploadedfile(uploaded_file, file_directory)

                st.write(f"File saved: {filename}")
                if (result):
                    anaysis_image(project_directory, filename, frame_processor)

    elif (mode_src == 'Camera'):
        with VideoCap_Info.with_no_info() as cap:

            cap.setup_capture(-1)
            run_original_streamlit_video_mediapipe_main(cap, frame_processor)
        if(filename is not None):
            display_download_buttons(frame_processor.get_dataframe(), os.path.join(file_directory, Path(filename).stem))

    elif (mode_src == 'Video'):
        title.title("Video Analysis")
        st.subheader("Analyse a video file.")
        st.divider()
        # Upload the video and save it
        uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
        if (uploaded_file):
             # Create a temporary file to store the uploaded video
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())

            with VideoCap_Info.with_no_info() as cap:
                cap.setup_capture(tfile.name)
                run_original_streamlit_video_mediapipe_main(cap, frame_processor)
            if(filename is not None):
                display_download_buttons(frame_processor.get_dataframe(), os.path.join(file_directory, "video_analysis_", Path(uploaded_file.name).stem))


    elif (mode_src == 'Camera Capture'):

        title.title("Image Capture")
        st.subheader("Capture an image from your camera.")
        st.divider()
        img_file_buffer = st.camera_input("Camera")
        if(img_file_buffer):
            st.write(img_file_buffer)
            filename, result = save_uploadedfile(img_file_buffer, os.path.join(project_directory, ps._SAVED_IMAGES))
            if(result):
                image = open_image(filename)
                if (image is not None):
                    anaysis_image(project_directory, filename, frame_processor)

    if (mode_src == 'Graph Data'):

        # Streamlit application starts here

        st.title('Select File')
        uploaded_file = st.sidebar.file_uploader("Upload a dataframe file (csv)", type=["csv"])
        if(uploaded_file):
            # To read file as bytes:
            # file_directory = os.path.join(project_directory, "saved_data_frames")
            # filename, result = save_uploadedfile(uploaded_file, file_directory)

            st.write("### Data Preview")
            df = load_data(uploaded_file)
            st.expander("Show Data", expanded=False).write(df)
            cols = df.columns

            st.write("### Select Columns to Graph")
            options_x_axis = st.selectbox('Select columns to plot on the x-axis',cols, key="x_axis")
            options = st.multiselect('Select columns to plot',cols, max_selections=9, key="label_axis")

            if len(options) > 4:
                st.warning('Please select no more than 4 columns.')
                options = options[:4]  # Keep only the first 4 selections

            # Checkbox to indicate if y-axis should be changed
            yaxis_change = st.checkbox("Change y-axis range", value=False, key="yaxis_change")

            # Plotting the selected columns
            if( (len(options_x_axis) >= 1)  and (len(options) >= 1)):

                fig = px.line(df, x=options_x_axis, y=options, color_discrete_sequence=px.colors.qualitative.Plotly)
                # Draw a scatter plot of the options onto the figure
                fig.update_traces(mode='markers+lines')

                # Add a line for the mean and median, the two should be included in a legend
                for column in options:
                    fig.add_hline(y=df[column].mean(), line_dash="dot", line_color="green", annotation_text=f"Mean {column}", annotation_position="bottom right")
                    fig.add_hline(y=df[column].median(), line_dash="dot", line_color="red", annotation_text=f"Median {column}", annotation_position="top right")

                if(yaxis_change):
                    fig.update_yaxes(range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)

            # Compute the statistics
            stats_df = compute_statistics(df, exclude_columns=['handedness','timestamp'])
            if(stats_df is not None):
                st.write("### Statistics")
                st.dataframe(stats_df, use_container_width=True)
                file_directory = os.path.join(project_directory, ps._DF_STATS)
                display_download_buttons(stats_df, os.path.join(file_directory, Path(uploaded_file.name).stem))

if __name__ == '__main__':
    main()
