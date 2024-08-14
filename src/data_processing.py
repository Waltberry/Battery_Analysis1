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

def find_charging_cycles(df):
    cycles = []
    current_cycle = []
    # Use df.index for accessing rows
    index = df.index
    i = 0
    while i < len(index):
        control_value = df.loc[index[i], 'control/mA']
        if control_value > 0:
            # Find the start of the cycle (including 10 points from the previous negative part)
            start_time = index[max(0, i-10)]
            current_index = start_time
            # Collect data from start to current index
            while current_index < index[i]:
                current_cycle.append(df.loc[current_index])
                # Move to the next index
                next_index = df.index[df.index.get_loc(current_index) + 1] if (df.index.get_loc(current_index) + 1) < len(df.index) else None
                if next_index is None:
                    break
                current_index = next_index
            while i < len(index) and df.loc[index[i], 'control/mA'] > 0:
                current_cycle.append(df.loc[index[i]])
                i += 1
            # End of the cycle, add it to cycles list
            cycles.append(pd.DataFrame(current_cycle))
            current_cycle = []
        else:
            i += 1
    return cycles

