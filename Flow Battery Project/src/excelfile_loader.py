import os
import pandas as pd

def load_excel(file_name):
    """
    Load an Excel file from the data folder.

    Parameters:
    file_name (str): Name of the Excel file to load.

    Returns:
    pd.ExcelFile: The loaded Excel file.
    """
    # Get the absolute path of the data folder
    data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Construct the full file path
    file_path = os.path.join(data_folder, file_name)
    
    # Load the Excel file
    xls = pd.ExcelFile(file_path)
    
    return xls
