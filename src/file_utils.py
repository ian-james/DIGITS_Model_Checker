import os
from enum import Enum
import time

import pandas as pd
import logging

# A function to clean spaces in a string and replace them with underscores
def clean_spaces(string):
    return string.strip().replace(" ", "_")

def copy_csv_to_excel(csv_file, excel_file):
    # Copy the CSV file to an Excel file
    try:
        df = pd.read_csv(csv_file, sep="\t")
        df.to_excel(excel_file, index=False, header=True)
    except Exception as e:
        logging.error("Error copying CSV to Excel: " + str(e))

def change_extension(file_path, extension="csv"):
    # Split the path into the name and extension

    base_name, _ = file_path.rsplit('.', 1)
    # Define the new extension
    new_file_path = f"{base_name}.{extension}"
    return new_file_path

def add_extension(filename, extension=".csv"):
    basename, ext = os.path.splitext(filename)
    if ext == extension:
        return filename
    else:
        return filename + extension

def get_file_path(output_file, location="../records/"):
    # This section manages the data collection.
    absolute_path = os.path.abspath(location)
    create_directory(absolute_path)
    full_path = os.path.join(absolute_path, output_file)
    return full_path


def create_directory(directory_path):
    """
    Creates a directory at the given path if it doesn't already exist.

    Parameters:
        directory_path (str): The path to the directory to create.

    Returns:
        None
    """
    if directory_path != '' and not os.path.exists(directory_path):
        os.makedirs(directory_path)


def could_be_directory(path):

    if( os.path.isdir(path)):
        return True

    # Normalize the path to use the correct separators and remove any trailing separators
    normalized_path = os.path.normpath(path)

    # Check if the path has any system-reserved characters (simplified example)
    invalid_chars = ['<', '>', ':', '"', '|', '?', '*']
    if any(char in path for char in invalid_chars):
        return False

    # Check for multiple segments in the path, which are more indicative of a directory structure
    segments = normalized_path.split(os.sep)

    # Return true if there are multiple segments, suggesting a directory-like structure
    return len(segments) > 1


def setup_fullpath_to_timestamp_output(output_filename, add_timestamp, directory="../records/"):
   # Write the DataFrame to an Excel file
    file_time = time.strftime("%Y_%m_%d-%H_%M_%S_")

    if(add_timestamp):
        ofile = file_time + output_filename
    else:
        ofile = output_filename

    path_to_file = get_file_path(ofile, directory)
    output_full_file = add_extension(path_to_file)

    return output_full_file, path_to_file

def check_if_file_is_image(filename):
    if(filename.endswith(".jpg") or filename.endswith(".png")):
        return True
    return False



def setupCSV2(filename, sep='\t',header=0):
    if( os.path.isfile(filename) ):
        try:
            f, fext = os.path.splitext(filename)
            if(fext == ".xlsx"):
                return pd.read_excel(filename)
            elif( fext == ".csv"):
                return  pd.read_csv(filename, sep=sep, encoding="ISO-8859-1",keep_default_na=True, header=header)
            elif( fext == ".tsv"):
                return  pd.read_csv(filename, sep=sep, encoding="ISO-8859-1",keep_default_na=True)
        except Exception as e:
            print(f"Error reading file CSV2: {e}")
    return None

def findSep(filename):
    try:
        with open(filename, "r") as testFile:
            lines = testFile.readlines()
            if( lines[0].count(',') >= 1):
                return ','
            return '\t'
    except Exception as e:
        print(f"Unknown separator: {e}")
    return None

def setupAnyCSV(filename, header=0):
    sep = findSep(filename)
    if(sep):
        return setupCSV2(filename,sep,header)
    return None

def is_excel(file_path):
    """Checks if a file is an Excel file."""
    file_extension = file_path.split('.')[-1]
    return file_extension in ['xls', 'xlsx']

def is_csv(file_path):
    """Checks if a file is a CSV file."""
    file_extension = file_path.split('.')[-1]
    return file_extension == 'csv'

def read_file_to_dataframe(file_path):
    """Reads a file to a pandas DataFrame based on file extension."""
    file_extension = file_path.split('.')[-1]
    if file_extension == 'csv':
        df = pd.read_csv(file_path,sep="\t")
    elif file_extension in ['xls', 'xlsx']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(
            "File format not supported. Only .csv, .xls, and .xlsx files are supported.")
    return df

