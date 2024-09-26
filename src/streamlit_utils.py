import os
import base64
from pathlib import Path
from io import BytesIO

import streamlit as st
from moviepy.editor import VideoFileClip


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
    location = os.path.join(folder, uploadedfile.name)
    try:
        with open(location, "wb") as f:
            f.write(uploadedfile.getbuffer())
    except:
        return None, st.error(f"Failed to save the {uploadedfile.name}.")

    return location, st.success("Saved File:{} to tempDir".format(uploadedfile.name))


def download_dataframe(df, file_name, file_format):
    # Create a button to download the file
    output = BytesIO()

    if file_format == 'xlsx':
        df.to_excel(output,sheet_name="Sheet1", index=False, header=True)
        mime_type = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        ext = 'xlsx'
    elif file_format == 'csv':
        df.to_csv(output, sep='\t', index=False)
        mime_type = 'text/csv'
        ext = 'csv'

    output.seek(0)

    # Use Streamlit's download button for cleaner handling
    st.download_button(
        label=f"Download as {file_format.upper()}",
        data=output,
        file_name=f"{file_name}.{ext}",
        mime=mime_type
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
        
# Download an image file
# File Formats
#    # Change file_format to svg
#     if file_format == 'svg':
#         file_format = 'image/svg+xml'
#     elif file_format == 'png':
#         file_format = 'image/png'
#     elif file_format == 'jpeg':
#         file_format = 'image/jpeg'
def download_image(image_path, file_name, file_format='image/png'):
    with open(image_path, 'rb') as f:
        image_bytes = f.read()
 
    st.download_button(label='Download Image', data=image_bytes, file_name=file_name, mime=file_format)
    

def display_download_buttons(df, file_path):
    col1, col2 = st.columns([1, 1])
    with col1:
        download_dataframe(df, file_path, "csv")
    with col2:
        download_dataframe(df, file_path, 'xlsx')


def display_video_buttons():

    col1, col2, col3, col4 = st.columns([1, 1, 1,1])
    with col1:
        play_button = st.button("Play",key="play_btn",on_click=flip_video_state('play'))
    with col3:
        stop_button = st.button("Stop",key="stop_btn")

    return play_button, stop_button