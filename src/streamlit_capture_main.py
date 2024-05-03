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

from Capture_Info import *

import matplotlib.pyplot as plt

from fps_timer import FPS
from VideoCap import VideoCap_Info

import convert_mediapipe_index as convert_mp


from mediapipe_helpers import *

from scipy import stats

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


def download_dataframe(df, file_name, file_format):

    if(df is None):
        st.write("No data to download.")
        return

    # Create a button to download the file
    output = BytesIO()
    if file_format == 'xlsx':
        df.to_excel(output,sheet_name="Sheet1", index=False, header=True)
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
    elif file_format == 'csv':
        df.to_csv(output, sep='\t', index=False)
        mime_type = 'text/csv'

    output.seek(0)

    if file_format == 'xlsx':
        ext = 'xlsx'
    elif file_format == 'csv':
        ext = 'csv'

    file_label = f'Download {file_format.upper()}'
    file_download = f'{file_name}.{ext}'
    b64 = base64.b64encode(output.read()).decode()

    st.markdown(
        f'<a href="data:file/{mime_type};base64,{b64}" download="{file_download}">{file_label}</a>',
        unsafe_allow_html=True
    )

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

def display_download_buttons(df, file_path):
    col1, col2 = st.columns([1, 1])
    with col1:
        download_dataframe(df, file_path, "csv")
    with col2:
        download_dataframe(df, file_path, 'xlsx')

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
        results, rimage = frame_processor.process_frame(image,fps.total_num_frames)
        if(results is not None and results.hand_landmarks != []):
            frame_placeholder.image(image=rimage, caption="Enhanced Image", channels="BGR")
        else:
            frame_placeholder.image(image=rimage.numpy_view(), caption="Enhanced Image", channels="BGR")
        
    
    st.write("## FINISHED ANALYSIS")
    frame_processor.finalize_dataframe()
    datatable_placeholder.dataframe(frame_processor.get_dataframe(), hide_index=True)  # You can also use st.write(df)
    return frame_processor.get_dataframe()


# def graph_df(df):
#     if(df is not None):

#         cols = convert_mp.convert_all_columns_to_friendly_name(df,[])
#         df.columns = cols

#         # Let the user select the columns to plot
#         options_x_axis = st.selectbox('Select columns to plot on the x-axis',cols, key="x_axis")

#         options = st.multiselect('Select columns to plot (max 4)',cols, key="label_axis")

#         if len(options) > 8:
#             st.warning('Please select no more than 4 columns.')
#             options = options[:8]  # Keep only the first 4 selections

#         # Checkbox to indicate if y-axis should be changed
#         yaxis_change = st.checkbox("Change y-axis range", value=False, key="yaxis_change")

#         # Plotting the selected columns
#         if( (len(options_x_axis) >= 1)  and (len(options) >= 1)):
#             #fig = px.scatter(df, x=options_x_axis, color_continuous_scale=px.colors.sequential.Viridis)
#             fig = px.line(df, x=options_x_axis, y=options, color_discrete_sequence=px.colors.qualitative.Plotly)
#             if(yaxis_change):
#                 fig.update_yaxes(range=[0, 1])
#             st.plotly_chart(fig)

#             draw_histogram_and_qq_plots(df[options], "norm", colors=px.colors.qualitative.Plotly)

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename, sep='\t')
    cols = convert_mp.convert_all_columns_to_friendly_name(df,[])
    df.columns = cols
    return df
    
# Function to compute the statistics
# Compute the The standard error of the mean (SEM) is a measure of how much the sample mean is expected to vary from the true population mean.
def sem(x):
    return np.std(x, ddof=0) / np.sqrt(len(x))

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
            st.dataframe(frame_processor.get_dataframe(), hide_index=True)
            if(filename is not None):
                display_download_buttons(frame_processor.get_dataframe(), os.path.join(file_directory, Path(filename).stem))
        else:
            st.write(f"Video file is not open {filename}")


def main():
    project_directory = "/home/jame/Projects/mediapipe_tools/media/"

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
        uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "png", "jpeg"])
        st.write(uploaded_file)
        if(uploaded_file):
            if uploaded_file is not None:
                # To read file as bytes:
                file_directory = os.path.join(project_directory, "saved_images")
                filename, result = save_uploadedfile(uploaded_file, file_directory)
                st.write(f"File saved: {filename}")
                if (result):
                    anaysis_image(project_directory, filename, frame_processor)

    elif (mode_src == 'Camera'):
        with VideoCap_Info.with_no_info() as cap:
            cap.setup_capture(4)
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
            file_directory = os.path.join(project_directory, "saved_videos")
            filename, result = save_uploadedfile(uploaded_file, file_directory)
            output_file = filename
            if Path(output_file).exists():
                st.write(f"Output file exists {output_file}")
                with VideoCap_Info.with_no_info() as cap:
                    cap.setup_capture(filename)
                    run_original_streamlit_video_mediapipe_main(cap, frame_processor)
                if(filename is not None):
                    display_download_buttons(frame_processor.get_dataframe(), os.path.join(file_directory, Path(filename).stem))
            else:
                st.write(f"Video file is not open {output_file}")

    elif (mode_src == 'Camera Capture'):

        title.title("Image Capture")
        st.subheader("Capture an image from your camera.")
        st.divider()
        img_file_buffer = st.camera_input("Camera")
        if(img_file_buffer):
            st.write(img_file_buffer)
            filename, result = save_uploadedfile(img_file_buffer, os.path.join(project_directory, "saved_images"))
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
                file_directory = os.path.join(project_directory, "saved_data_frames_stats")
                display_download_buttons(stats_df, os.path.join(file_directory, Path(uploaded_file.name).stem))







if __name__ == '__main__':
    main()
