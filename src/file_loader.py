import os
import pandas as pd
import re
from galvani import BioLogic

try:
    import PSData
except ImportError:
    PSData = None  # we'll handle this gracefully later

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



def load_csv_files(file_names, sub_folder: str = '', data: str = 'data'):
    """
    Load multiple CSV files from the specified subfolder within the
    'Flow Battery Project' data folder.

    Parameters
    ----------
    file_names : list of str
        CSV file names to load (e.g. ['1.csv']).
    sub_folder : str, optional
        Optional subfolder inside the data folder, e.g. '', 'cycle1', 'raw'.
    data : str, optional
        Name of the data folder (default: 'data').

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping a cleaned file key to the loaded DataFrame.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))  # .../Battery_Analysis1

    # Base: .../Battery_Analysis1/Flow Battery Project/data
    data_folder = os.path.join(project_root, 'Flow Battery Project', data)

    # Optional deeper subfolder
    if sub_folder:
        data_folder = os.path.join(data_folder, sub_folder)

    dataframes = {}

    for file in file_names:
        try:
            file_path = os.path.join(data_folder, file)

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")

            # Adjust skiprows/header depending on how your CSV looks.
            # If it has a normal header row, remove skiprows or set skiprows=0.
            df = pd.read_csv(file_path)  # or pd.read_csv(file_path, skiprows=5)

            key = (
                file.replace(' ', '_')
                    .replace('.', '_')
                    .replace('(', '')
                    .replace(')', '')
            )
            dataframes[key] = df

        except pd.errors.ParserError as e:
            print(f"Error reading {file}: {e}")
        except ValueError as e:
            print(f"Error processing {file}: {e}")

    return dataframes

def load_csv(file_name: str, sub_folder: str = '', data: str = 'data') -> pd.DataFrame:
    """Convenience wrapper to load a single CSV file."""
    dfs = load_csv_files([file_name], sub_folder=sub_folder, data=data)
    key = (
        file_name.replace(' ', '_')
                 .replace('.', '_')
                 .replace('(', '')
                 .replace(')', '')
    )
    return dfs[key]


def load_psdata_as_table(file_name: str) -> pd.DataFrame:
    """
    Fallback: try to load .psdata as a plain text table for quick inspection.
    This may or may not work depending on format, but it's useful to peek.

    Returns a raw DataFrame.
    """
    project_root = os.path.dirname(os.path.dirname(__file__))
    data_folder = os.path.join(project_root, 'Flow Battery Project', 'data')
    file_path = os.path.join(data_folder, file_name)
    file_path = os.path.normpath(file_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")

    # Try whitespace-delimited first
    df = pd.read_csv(
        file_path,
        sep=r'\s+',
        comment='#',
        header=None,
        engine='python'
    )
    return df



# def load_psdata(file_name: str):
#     """
#     Load a .psdata file from the 'Flow Battery Project' data folder.

#     This is a generic loader: it reads the file as a whitespace-separated
#     table and returns a raw pandas DataFrame. You can inspect the first
#     few rows in a notebook and then decide how to rename columns, handle
#     units rows, etc.

#     Parameters
#     ----------
#     file_name : str
#         Name of the .psdata file (e.g. '2.psdata').

#     Returns
#     -------
#     pd.DataFrame
#         Raw data from the file.
#     """
#     project_root = os.path.dirname(os.path.dirname(__file__))
#     data_folder = os.path.join(project_root, 'Flow Battery Project', 'data')
#     file_path = os.path.join(data_folder, file_name)

#     file_path = os.path.normpath(file_path)

#     if not os.path.exists(file_path):
#         raise FileNotFoundError(f"The file {file_path} does not exist.")

#     # First attempt: whitespace-separated, ignore comment lines
#     df = pd.read_csv(
#         file_path,
#         sep=r'\s+',
#         comment='#',
#         header=None,   # assume no header; you’ll inspect and adjust
#         engine='python'
#     )

#     return df
