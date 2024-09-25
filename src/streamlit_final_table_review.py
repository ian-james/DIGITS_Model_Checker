# This program demonstrates a streamlit UI to display a table of data that can be filtered and downloaded.

import os
import streamlit as st
import pandas as pd
import logging
from log import set_log_level
from io import BytesIO


import numpy as np
import plotly.express as px

from fps_timer import FPS
import matplotlib.pyplot as plt

from streamlit_utils import save_uploadedfile, download_dataframe, display_download_buttons
import project_settings as ps

import convert_mediapipe_index as convert_mp
from file_utils import setupAnyCSV

from stats_functions import compute_statistics


def page_setup():
    st.set_page_config(page_title="Mediapipe Analysis", page_icon=":bar_chart:", layout="wide")
    st.title("Mediapipe Analysis")
    st.write("### This tools is designed to analyze data from a CSV file.")
    

def setup_sidebar(project_directory):

    st.sidebar.title('Select File')
    uploaded_file = st.sidebar.file_uploader("Upload a dataframe file (csv)", type=["csv"], accept_multiple_files=False)
    
    mediapipe_expander = st.sidebar.expander("## User Options", expanded=True)
    mediapipe_expander.markdown("## Program Options")
    mode_src = mediapipe_expander.selectbox("Select the mode", ["None","Graph Data"])


    ################################################################################
    # Debug Options
    debug_expander = st.sidebar.expander("## Developer Options", expanded=False)
    debug_expander.markdown("## Debugging Controls")

    debug_levels = ['debug', 'info', 'warning', 'error', 'critical']
    debug = debug_expander.selectbox("Set the logging level", debug_levels)    

    return uploaded_file, mode_src, debug

@st.cache_data
def load_data(file_upload):
    try:
        df = pd.read_csv(file_upload,sep=",")
    except Exception as e:
        return None    
    return df


# Function to save plotly figure as PNG or SVG in memory
def save_plotly_figure(fig, format):
    # Create an in-memory buffer
    buffer = BytesIO()
    # Use the write_image method to save the figure to the buffer
    fig.write_image(buffer, format=format)
    # Move the buffer's cursor to the beginning
    buffer.seek(0)
    return buffer

def graph_df(df):
    
    
    if(df is None):
        st.write("### Error: Unable to read the file.")
        return

    st.expander("Show Data", expanded=False).write(df)
    cols = df.columns.tolist()

    # Streamlit UI
    st.write("### Select Columns to Graph")

    # Multiselect for x-axis columns
    options_x_axis = st.multiselect('Select columns to plot on the x-axis', cols, key="x_axis")

    # Multiselect for y-axis columns (limit to 4 selections)
    options = st.multiselect('Select columns to plot', cols, max_selections=9, key="label_axis")

    # Limit selection to 4 columns for both x and y axes
    if len(options_x_axis) > 4:
        st.warning('Please select no more than 4 x-axis columns.')
        options_x_axis = options_x_axis[:4]

    if len(options) > 4:
        st.warning('Please select no more than 4 y-axis columns.')
        options = options[:4]

    # Initialize df_filtered as df
    df_filtered = df.copy()

    # Range slider or multiselect for x-axis filtering
    if options_x_axis:
        for opt in options_x_axis:
            is_numeric = pd.api.types.is_numeric_dtype(df[opt])
            if is_numeric:
                min_range, max_range = st.slider(f'Select range for {opt} values',
                                                float(df[opt].min()),
                                                float(df[opt].max()),
                                                (df[opt].min(), df[opt].max()))

                # Filter the data based on the selected range
                df_filtered = df_filtered[(df_filtered[opt] >= min_range) & (df_filtered[opt] <= max_range)]
            else:
                # Create a multi-select box for non-numeric values
                unique_values = df[opt].unique()
                selected_values = st.multiselect(f"Select {opt} values", unique_values)

                # Filter the data based on the selected values
                if selected_values:
                    df_filtered = df_filtered[df_filtered[opt].isin(selected_values)]

    # Filter Y-axis columns (range or multiselect based on type)
    for opt in options:
        is_numeric = pd.api.types.is_numeric_dtype(df[opt])
        if is_numeric:
            min_range, max_range = st.slider(f'Select range for {opt} values',
                                            float(df[opt].min()),
                                            float(df[opt].max()),
                                            (df[opt].min(), df[opt].max()))

            # Filter the data based on the selected range
            df_filtered = df_filtered[(df_filtered[opt] >= min_range) & (df_filtered[opt] <= max_range)]
            
            
        else:
            # Create a multi-select box for non-numeric values
            unique_values = df[opt].unique()
            selected_values = st.multiselect(f"Select {opt} values", unique_values)

            # Filter the data based on the selected values
            if selected_values:
                df_filtered = df_filtered[df_filtered[opt].isin(selected_values)]
        
        # Select only columns that are in options
        df_filtered = df_filtered[options_x_axis + options]

    st.expander("Show Filtered Data", expanded=False).write(df_filtered)
    display_download_buttons(df_filtered, "filtered_data")

    # Checkbox to indicate if y-axis range should be changed
    yaxis_change = st.checkbox("Change y-axis range", value=False, key="yaxis_change")
    
    graph_type = st.selectbox('Select Graph Type', ['Bar', 'Line', 'Scatter', 'Histogram',
                                                    'Box'])

    # Plotting the selected columns
    if options_x_axis and options:
        # Create a new column that concatenates the selected x-axis column values
        df_filtered['Group'] = df_filtered[options_x_axis].astype(str).agg(', '.join, axis=1)

        # Group the data by the concatenated group values and calculate the mean
        df_grouped = df_filtered.groupby('Group')[options].mean().reset_index()
        
        # Define the figure based on the selected graph type
        if graph_type == 'Bar':
            fig = px.bar(df_grouped, 
                        x='Group',           # Use the concatenated group as x-axis
                        y=options,           # Y-axis columns
                        barmode='group',     # Group bars for different y-columns
                        labels={'Group': ', '.join(options_x_axis)})  # Dynamic label for x-axis

        elif graph_type == 'Line':
            fig = px.line(df_grouped, 
                        x='Group', 
                        y=options, 
                        labels={'Group': ', '.join(options_x_axis)})

        elif graph_type == 'Scatter':
            fig = px.scatter(df_grouped, 
                            x='Group', 
                            y=options, 
                            labels={'Group': ', '.join(options_x_axis)})

        elif graph_type == 'Histogram':
            fig = px.histogram(df_filtered, 
                            x='Group', 
                            y=options[0],   # Histograms typically work with single columns
                            labels={'Group': ', '.join(options_x_axis)})

        elif graph_type == 'Box':
            fig = px.box(df_filtered, 
                        x='Group', 
                        y=options[0],       # Box plots usually show one variable distribution
                        labels={'Group': ', '.join(options_x_axis)})

        # Update the layout for better visualization
        fig.update_layout(
            xaxis_title=", ".join(options_x_axis),  # Show all group columns on x-axis
            yaxis_title="Y Values",
            xaxis={'categoryorder':'total descending'},  # Order the categories if needed
        )       

        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        png_buffer = save_plotly_figure(fig, "png")
        st.download_button(
            label="Download plot as PNG",
            data=png_buffer,
            file_name="plot.png",
            mime="image/png"
        )
    
        svg_buffer = save_plotly_figure(fig, "svg")
        st.download_button(
            label="Download plot as SVG",
            data=svg_buffer,
            file_name="plot.svg",
            mime="image/svg+xml"
        )

def setup_statistics(df):
    stats_df = compute_statistics(df, exclude_columns=['handedness','timestamp','filename'])
    if(stats_df is not None):
        st.write("### Statistics")
        st.write(stats_df)

def main():

    project_directory = "/home/jame/Projects/Western/Western_Postdoc/Datasets/Processed_Videos/analysis/nh_1_md_0.5_mt_0.5_mp_0.5/test/"

    # Setup Logging
    set_log_level("info")
    logging.info("Starting Program")

    page_setup()

    uploaded_file, mode_src, debug  = setup_sidebar(project_directory)

    if(uploaded_file):
        file=uploaded_file
        #file = os.join(project_directory,file)
        st.write("### Data Preview")
        df = load_data(uploaded_file)
        
        graph_df(df)

if __name__ == '__main__':
    main()