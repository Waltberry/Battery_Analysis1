import pytest
import pandas as pd
from src.file_loader import load_excel, load_csv_files

def test_load_excel():
    file_name = 'chronopotentiometry For CNF Paper.xlsx'
    
    # Assuming the Excel file exists in the data folder
    xls = load_excel(file_name)
    
    # Check if the file is loaded
    assert isinstance(xls, pd.ExcelFile), "Loaded object is not an ExcelFile instance"
    
    # Check if the file contains sheets
    assert len(xls.sheet_names) > 0, "No sheets found in the loaded Excel file"

def test_load_csv_files():
    file_names = [
    '9171_20240626_120 s_N11503200095_Polarization_High to low_80 deg C_4 mpm_corr (1).csv',
    '9171_20240626_120 s_N11503200095_Polarization_High to low_80 deg C_4 mpm_corr (2).csv'
    ]
    sub_folder = ''
    
    # Assuming the CSV files exist in the specified subfolder
    dataframes = load_csv_files(file_names, sub_folder)
    
    # Check if the returned object is a dictionary
    assert isinstance(dataframes, dict), "Returned object is not a dictionary"
    
    # Check if all specified files are present in the dictionary
    for file in file_names:
        key = file.replace(' ', '_').replace('.', '_').replace('(', '').replace(')', '')
        assert key in dataframes, f"File {file} not found in the dictionary"
        
        # Check if the DataFrame is not empty
        df = dataframes[key]
        assert not df.empty, f"DataFrame for {file} is empty"
        # Check if the required columns are present
        required_columns = ['Time (HH:mm:ss.SSS)', 'Channel', 'V', 'A', 'Data Source']
        for col in required_columns:
            assert col in df.columns, f"Column {col} is missing in DataFrame for {file}"

if __name__ == "__main__":
    pytest.main()
