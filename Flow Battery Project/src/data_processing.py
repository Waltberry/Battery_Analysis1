import pandas as pd

def identify_charging_cycles(data, time_col, value_col):
    # Initialize variables
    charging_cycles = []
    cycle = []
    in_charging = False
    
    # Iterate through the first_sheet_df to identify charging cycles
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