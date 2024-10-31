# This program puts an interface on taking a goniometry measurement file 
#   and processing it for analysis.
# Streamlit interface is used to upload the file and then the file is processed.
# The processed data is displayed in a table and can be downloaded.

import os
import streamlit as st

from streamlit_utils import save_uploadedfile, display_download_buttons
from file_utils import setupAnyCSV

from handle_goniometery_measurements import  build_goniometry_dataframe

def page_setup():
    st.set_page_config(page_title="Goniometry Analysis", page_icon=":bar_chart:", layout="wide")
    st.title("Goniometry Analysis")
    st.write("### This tool takes either a single (combined) or two (right,left) goniometry files to develop a csv for comparison to a standard model file.")
    st.write("### Example) A model would be Mediapipe data for a specific patient.")

# Load from cache
@st.cache_data
def load_data(file_upload):
    df = setupAnyCSV(file_upload)
    return df

def main():
    page_setup()

    st.sidebar.title('Select Goniometry File (Right if multiple)')
    uploaded_file = st.sidebar.file_uploader("Upload a dataframe file (csv)", type=["csv"], accept_multiple_files=False,key="right_file")

    # Setup the csv file
    st.sidebar.title('Select Left Goniometry File (if exists)')
    uploaded_left_file = st.sidebar.file_uploader("Upload a dataframe file (csv)", type=["csv"], accept_multiple_files=False, key="left_file")

    # Setup a submit button to run the analysis
    submit_btn = st.sidebar.button("Submit")

    if submit_btn:
        if  uploaded_file:

            output_dir = "output"
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            # Output file
            output_file = os.path.join(output_dir, "result_" + uploaded_file.name)

            # Save the files
            left_location = ""
            right_location, _ = save_uploadedfile(uploaded_file, output_dir)
            if  uploaded_left_file:
                left_location, _ = save_uploadedfile(uploaded_left_file , output_dir)

            df = build_goniometry_dataframe(right_location, output_file, left_location, left_location == "")

            if  df is not None:
                st.write("### Data Preview")
                st.expander("Show Data", expanded=False).write(df)
                display_download_buttons(df, "Goniometry Data")
            else:
                st.write("### Error Processing the file")
        else:
            st.write("### No file uploaded.")

if __name__ == "__main__":
    main()
