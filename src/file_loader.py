import os
import pandas as pd
import re
from galvani import BioLogic

TIME_COLUMN_NAME = 'Time (HH:mm:ss.SSS)'

def load_mpr(file_name):
    # Get the absolute path of the project's root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Construct the path to the data folder
    data_folder = os.path.join(project_root, 'Flow Battery Project', 'data')
    
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Load the Excel file
    xls = BioLogic.MPRfile(file_path)
    
    return xls

def load_excel(file_name):
    """
    Load an Excel file from the 'Flow Battery Project' data folder.

    This function constructs the full file path to an Excel file located in the 
    'Flow Battery Project' data folder and loads it into a pandas ExcelFile object.

    Parameters:
    file_name (str): Name of the Excel file to load.

    Returns:
    pd.ExcelFile: The loaded Excel file.
    """
    # Get the absolute path of the project's root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Construct the path to the data folder
    data_folder = os.path.join(project_root, 'Flow Battery Project', 'data')
    
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    # xls = pd.read_excel(file_path, sheet_name = sheet_name)
    
    return xls


def load_txt(file_name):
    """
    Load a text file from the 'Flow Battery Project' data folder.

    This function constructs the full file path to a text file located in the 
    'Flow Battery Project/data' folder and loads it into a pandas DataFrame object.

    Parameters:
    file_name (str): Name of the text file to load.

    Returns:
    pd.DataFrame: The loaded text file as a pandas DataFrame.
    """
    # Get the absolute path of the project's root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Construct the path to the data folder
    data_folder = os.path.join(project_root, 'Flow Battery Project', 'data')
    
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Ensure the path uses the correct format for the operating system
    file_path = os.path.normpath(file_path)
    
    # Check if the file exists before trying to read it
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    # Load the text file, skipping the initial metadata rows and parsing the columns
    df = pd.read_csv(file_path, sep=r'\s+', skiprows=5, header=None)
    df.columns = ["Time Step", "voltage_vp", "flow-time"]
    
    return df



def load_csv_files(file_names, sub_folder, project_folder, data='data'):
    """
    Load and process multiple CSV files from the specified subfolder within the 'Hydrogen Project' data folder.

    This function constructs the full file path for each CSV file in the specified
    subfolder, reads the files into DataFrames, processes them (e.g., renaming columns, 
    cleaning data), and returns a dictionary of processed DataFrames.

    Parameters:
    file_names (list): List of CSV file names to load.
    sub_folder (str): Subfolder within the 'Hydrogen Project' data folder where the files are located.

    Returns:
    dict: Dictionary with file names as keys (processed) and DataFrames as values.
    """
    # Get the absolute path of the project's root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Construct the path to the data folder
    data_folder = os.path.join(project_root, project_folder, data, sub_folder)
    
    # Dictionary to hold DataFrames
    dataframes = {}

    # Function to clean and rename the DataFrame columns
    def process_dataframe(df, file_name):
        # Rename the columns
        df.columns = [TIME_COLUMN_NAME, 'Channel', 'CH', 'unnamed2', 'V', 'unnamed3', 'A', 'unnamed4']
        # Remove the brackets in 'Channel' and TIME_COLUMN_NAME columns
        df['Channel'] = df['Channel'].str.strip('[]')
        df[TIME_COLUMN_NAME] = df[TIME_COLUMN_NAME].str.strip('()')
        # Discard the unnecessary columns
        df = df.drop(columns=['unnamed2', 'unnamed3', 'unnamed4'])
        # Below code is for identifying the 'Data Source' from the file name (optional)
        # Extract the relevant part of the file name for 'Data Source'
        # match = re.search(r'_N11507060127_(.*?)_', file_name)
        # if match:
        #     middle_part = match.group(1)
        # else:
        #     middle_part = "rapid_polarization"
        # middle_part = re.search(r'Polarization_(.*?) mpm_corr', file_name).group(1)
        # middle_part = re.search(r'Rapid (.*)', file_name).group()
        # df['Data Source'] = middle_part
        return df

    # Read each file into a DataFrame with error handling and process them
    for file in file_names:
        try:
            # Construct the full file path
            file_path = os.path.join(data_folder, file)
            # Extract a unique key for the dictionary
            key = file.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
            # Attempt to read the CSV file starting from the 5th row
            df = pd.read_csv(file_path, skiprows=5)
            # Process the DataFrame
            dataframes[key] = process_dataframe(df, file)
            # print(f"Successfully read and processed {file}")
        except pd.errors.ParserError as e:
            print(f"Error reading {file}: {e}")
        except ValueError as e:
            print(f"Error processing {file}: {e}")

    return dataframes
