import os
import pandas as pd
import re

def load_excel(file_name):
    """
    Load an Excel file from the data folder.

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
    
    return xls

def load_csv_files(file_names, sub_folder):
    """
    Load and process multiple CSV files from the specified subfolder.

    Parameters:
    file_names (list): List of CSV file names to load.
    sub_folder (str): Subfolder within the data folder where the files are located.

    Returns:
    dict: Dictionary with file names as keys and processed DataFrames as values.
    """
    # Get the absolute path of the project's root directory
    project_root = os.path.dirname(os.path.dirname(__file__))
    
    # Construct the path to the data folder
    data_folder = os.path.join(project_root, 'Hydrogen Project', 'data', sub_folder)
    
    # Dictionary to hold DataFrames
    dataframes = {}

    # Function to clean and rename the DataFrame columns
    def process_dataframe(df, file_name):
        # Rename the columns
        df.columns = ['Time (HH:mm:ss.SSS)', 'Channel', 'CH', 'unnamed2', 'V', 'unnamed3', 'A', 'unnamed4']
        # Remove the brackets in 'Channel' and 'Time (HH:mm:ss.SSS)' columns
        df['Channel'] = df['Channel'].str.strip('[]')
        df['Time (HH:mm:ss.SSS)'] = df['Time (HH:mm:ss.SSS)'].str.strip('()')
        # Discard the unnecessary columns
        df = df.drop(columns=['unnamed2', 'unnamed3', 'unnamed4'])
        # Extract the relevant part of the file name for 'Data Source'
        middle_part = re.search(r'Polarization_(.*?) mpm_corr', file_name).group(1)
        df['Data Source'] = middle_part
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