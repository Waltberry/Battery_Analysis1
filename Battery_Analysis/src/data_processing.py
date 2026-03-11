import pandas as pd

def identify_charging_cycles(data, time_col, value_col):
    """
    Identify charging cycles from the given dataset.

    This function analyzes the given dataset to identify charging cycles based on
    the provided time and value columns. A charging cycle is defined as a sequence
    of increasing values starting from a value greater than zero and ending when
    the value stops increasing or decreases.

    Parameters:
    data (pd.DataFrame): The dataset containing the time and value columns.
    time_col (str): The name of the column representing time.
    value_col (str): The name of the column representing the values indicating charging status.

    Returns:
    list of list of tuple: A list where each element is a list of tuples representing
    a charging cycle. Each tuple contains (time, value) pairs.
    """
    # Initialize variables
    charging_cycles = []
    cycle = []
    in_charging = False
    
    # Iterate through the data to identify charging cycles
    for i in range(1, len(data)):
        current_value = data[value_col].iloc[i]
        previous_value = data[value_col].iloc[i-1]

        if current_value > 0 and previous_value <= 0:
            # Start of a new charging cycle
            cycle = [(data[time_col].iloc[i], current_value)]
            in_charging = True
        elif current_value > previous_value and in_charging:
            # Continue the charging cycle
            cycle.append((data[time_col].iloc[i], current_value))
        elif current_value <= previous_value and in_charging:
            # End of the charging cycle
            if cycle:
                charging_cycles.append(cycle)
            cycle = []
            in_charging = False

    # If the last segment is still a charging cycle, add it to the list
    if cycle:
        charging_cycles.append(cycle)

    return charging_cycles

def get_previous_segment(df, start_idx):
    """
    Collect the previous segment (up to 10 points) before a charging/discharging cycle starts.

    This function looks back from the start of a charging/discharging cycle to collect up to 10
    data points where the control value was negative or zero. These points help in
    identifying the beginning of the cycle, ensuring a comprehensive cycle analysis.

    Parameters:
    df (pd.DataFrame): The dataset containing the charging/discharging data.
    start_idx (int): The index where the charging/discharging cycle starts.

    Returns:
    list: A list of rows (as dictionaries) representing the previous negative segment.
    """
    segment = []
    
    # Determine the starting point (either 10 points before the cycle or the beginning of the data)
    start_time = df.index[max(0, start_idx - 10)]
    current_index = start_time
    
    # Iterate through the data from start_time to just before start_idx
    while current_index < df.index[start_idx]:
        segment.append(df.loc[current_index])  # Append each row to the segment
        # Get the next index, ensuring we don't go out of bounds
        next_index = df.index[df.index.get_loc(current_index) + 1] if (df.index.get_loc(current_index) + 1) < len(df.index) else None
        if next_index is None:
            break
        current_index = next_index  # Move to the next index
    
    return segment

def collect_charging_cycle(df, start_idx):
    """
    Collect the data points for a charging cycle.

    This function gathers all data points from the start of a charging cycle until the
    cycle ends (i.e., when the control value becomes non-positive). The cycle is defined
    by the control/mA column.

    Parameters:
    df (pd.DataFrame): The dataset containing the charging data.
    start_idx (int): The index where the charging cycle starts.

    Returns:
    list: A list of rows (as dictionaries) representing the charging cycle.
    int: The index where the charging cycle ends, to continue processing.
    """
    charging_cycle = []
    i = start_idx  # Start from the provided start index
    
    # Continue adding points to the charging cycle as long as control/mA > 0
    while i < len(df.index) and df.loc[df.index[i], 'control/mA'] > 0:
        charging_cycle.append(df.loc[df.index[i]])  # Add current row to the cycle
        i += 1  # Move to the next index
    
    return charging_cycle, i  # Return the collected cycle and the next index to process

def collect_discharging_cycle(df, start_idx):
    """
    Collect the data points for a discharging cycle.

    This function gathers all data points from the start of a discharging cycle until the
    cycle ends (i.e., when the control value becomes non-negative). The cycle is defined
    by the control/mA column.

    Parameters:
    df (pd.DataFrame): The dataset containing the discharging data.
    start_idx (int): The index where the discharging cycle starts.

    Returns:
    list: A list of rows (as dictionaries) representing the discharging cycle.
    int: The index where the discharging cycle ends, to continue processing.
    """
    discharging_cycle = []
    i = start_idx  # Start from the provided start index
    
    # Continue adding points to the discharging cycle as long as control/mA < 0
    while i < len(df.index) and df.loc[df.index[i], 'control/mA'] < 0:
        discharging_cycle.append(df.loc[df.index[i]])  # Add current row to the cycle
        i += 1  # Move to the next index
    
    return discharging_cycle, i  # Return the collected cycle and the next index to process


def find_charging_cycles(df):
    """
    Find and return all charging cycles from the given DataFrame.

    This function processes the entire dataset to identify and isolate charging cycles,
    which are sequences where the control/mA is positive. Each cycle includes a small
    segment of negative data points preceding the positive sequence to capture the
    transition into the charging phase.

    Parameters:
    df (pd.DataFrame): The dataset containing the charging data.

    Returns:
    list of pd.DataFrame: A list where each element is a DataFrame representing a charging cycle.
    """
    cycles = []  # List to hold all identified cycles
    i = 0  # Start index for iteration
    
    # Iterate through the entire DataFrame by index
    while i < len(df.index):
        control_value = df.loc[df.index[i], 'control/mA']  # Current control value
        
        if control_value > 0:  # If a positive control value is found, a cycle starts
            # Get the previous negative segment to prepend to the charging cycle
            previous_segment = get_previous_segment(df, i)
            # Collect the charging cycle and update the index to where the cycle ends
            current_cycle, i = collect_charging_cycle(df, i)
            # Combine the previous segment and the current cycle into one DataFrame and add to cycles
            cycles.append(pd.DataFrame(previous_segment + current_cycle))
        else:
            i += 1  # If not in a charging cycle, move to the next index
    
    return cycles  # Return the list of all identified charging cycles

def find_discharging_cycles(df):
    """
    Find and return all discharging cycles from the given DataFrame.

    This function processes the entire dataset to identify and isolate discharging cycles,
    which are sequences where the control/mA is negative. Each cycle includes a small
    segment of positive data points preceding the negative sequence to capture the
    transition into the discharging phase.

    Parameters:
    df (pd.DataFrame): The dataset containing the discharging data.

    Returns:
    list of pd.DataFrame: A list where each element is a DataFrame representing a discharging cycle.
    """
    cycles = []  # List to hold all identified cycles
    i = 0  # Start index for iteration
    
    # Iterate through the entire DataFrame by index
    while i < len(df.index):
        control_value = df.loc[df.index[i], 'control/mA']  # Current control value
        
        if control_value < 0:  # If a negative control value is found, a cycle starts
            # Get the previous positive segment to prepend to the discharging cycle
            previous_segment = get_previous_segment(df, i)
            # Collect the discharging cycle and update the index to where the cycle ends
            current_cycle, i = collect_discharging_cycle(df, i)
            # Combine the previous segment and the current cycle into one DataFrame and add to cycles
            cycles.append(pd.DataFrame(previous_segment + current_cycle))
        else:
            i += 1  # If not in a discharging cycle, move to the next index
            
    return cycles  # Return the list of all identified charging cycles
    


# def get_loops_of_3_cycles(df, cycle_length=1750):
#     """
#     Divide the data into chunks of 3 cycles (each of size `cycle_length` rows).
    
#     Parameters:
#     df (pd.DataFrame): The full dataset.
#     cycle_length (int): Number of rows in a single cycle.
    
#     Returns:
#     list of pd.DataFrame: List of DataFrames, each representing a group of 3 cycles.
#     """
#     group_size = 3 * cycle_length
#     loops = []

#     for start in range(0, len(df), group_size):
#         end = min(start + group_size, len(df))
#         loop = df.iloc[start:end]
#         if not loop.empty:
#             loops.append(loop)

#     return loops


def slice_df_by_index_range(df, step=1750):
    """
    Slice the DataFrame using index-based ranges (assumes index is numerical, like time in seconds).

    Parameters:
    df (pd.DataFrame): DataFrame with numerical index (e.g., time).
    step (int or float): Width of each slice along the index.

    Returns:
    list of pd.DataFrame: Each element is a DataFrame sliced over [start:end] index range.
    """
    start = df.index.min()
    end = df.index.max()
    chunks = []

    while start < end:
        stop = start + step
        chunk = df.loc[(df.index >= start) & (df.index < stop)]
        if not chunk.empty:
            chunks.append(chunk)
        start = stop

    return chunks




