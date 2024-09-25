import os
from pathlib import Path
import logging

import streamlit as st
import numpy as np
import plotly.express as px

from fps_timer import FPS
import matplotlib.pyplot as plt

import pandas as pd

import convert_mediapipe_index as convert_mp

from scipy import stats

from streamlit_utils import save_uploadedfile, download_dataframe, display_download_buttons

from mediapipe_helpers import *
from fps_timer import FPS

import project_settings as ps

def run_original_streamlit_video_mediapipe_main(cap, frame_processor):

    # Streamlit UI Options.
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

        results, rimage = frame_processor.process_frame(image,fps.total_num_frames)
        frame_placeholder.image(image=rimage, caption="Enhanced Image", channels="BGR")

    st.write("## FINISHED ANALYSIS")
    datatable_placeholder.dataframe(frame_processor.dataframe, hide_index=True)  # You can also use st.write(df)
    return frame_processor.get_dataframe()


def draw_histogram_and_qq_plots(data, distribution, colors):

    # Plot the histogram
    for i, column in enumerate(data.columns):
        data[column].hist(bins=20, color=colors[i % len(colors)], alpha=0.7, label=column)
        plt.legend()
        plt.show()

# def save_uploadedfile(uploadedfile, folder='tempDir'):
#     try:
#         with open(os.path.join(folder, uploadedfile.name), "wb") as f:
#             f.write(uploadedfile.getbuffer())
#         return os.path.join(folder, uploadedfile.name), True
#     except Exception as e:
#         return str(e), False

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
    stats_fun = ['max', 'min', 'mean', 'median', 'std', sem]
    stats_df = df.agg(stats_fun)

    return stats_df

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename, sep='\t')
    cols = convert_mp.convert_all_columns_to_friendly_name(df,[])
    df.columns = cols
    return df


def main():

    project_directory = ps.DIR_MEDIA

    st.set_page_config(page_title="Mediapipe Graphing Tool", page_icon=":bar_chart:", layout="wide")
    st.write("### This tool is designed to graph data from a CSV file.")
    ################################################################################
    # MediaPipe Options
    mediapipe_expander = st.sidebar.expander("## User Options", expanded=True)
    mediapipe_expander.markdown("## Mediapipe Options")
    st.markdown("## Program Options")
    mode_src = st.selectbox("Select the mode", ["None","Graph Data"])
    ################################################################################
    # Debug Options
    debug_expander = st.sidebar.expander("## Developer Options", expanded=False)
    debug_expander.markdown("## Debugging Controls")

    debug_levels = ['debug', 'info', 'warning', 'error', 'critical']
    debug_expander.selectbox("Set the logging level", debug_levels)


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
            stats_df = compute_statistics(df, exclude_columns=['handedness','timestamp','filename'])
            if(stats_df is not None):
                st.write("### Statistics")
                st.dataframe(stats_df, use_container_width=True)
                file_directory = os.path.join(project_directory, ps._DF_STATS)
                display_download_buttons(stats_df, os.path.join(file_directory, Path(uploaded_file.name).stem))


if __name__ == '__main__':
    main()

