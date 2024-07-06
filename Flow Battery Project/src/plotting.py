import matplotlib.pyplot as plt

def plot_scatterplot(times, values, idx):
    """
    Plot data as a scatter plot.

    Parameters:
    times (list): List of time values.
    values (list): List of corresponding data values.
    idx (int): Index of the charging cycle.
    
    Returns:
    None
    """
    plt.figure()
    plt.scatter(times, values,label = f'Charging Cycle {idx+1}')
    plt.xlabel('Time (hours)')
    plt.ylabel('Ewe/mV.4')
    plt.title(f'Charging Cycle {idx+1}')
    plt.legend()
    plt.show()

def plot_fitted(times, values, y_fitted, idx):
    """
    Plot data and a fitted model.

    Parameters:
    times (list): List of time values.
    values (list): List of corresponding data values.
    y_fitted (list): Fitted values corresponding to times.
    idx (int): Index of the charging cycle.
    
    Returns:
    None
    """
    # Plot data and fitted model
    plt.figure()
    plt.scatter(times, values, label='Data')
    plt.plot(times, y_fitted, label='Fitted Model (Gen Exp)', color='red')
    plt.xlabel('Time (hours)')
    plt.ylabel('Ewe/mV.4')
    plt.title(f'Charging Cycle {idx+1}')
    plt.legend()
    plt.show()

def print_fitted_params(fitted_params, n_terms=3):
    """
    Print fitted parameters.

    Parameters:
    fitted_params (list): List of fitted parameters.
    n_terms (int, optional): Number of terms in the model. Defaults to 3.
    
    Returns:
    None
    """
    # Print fitted parameters
    param_labels = ['c1'] + [f'c{i+2}' for i in range(n_terms)] + [f'b{i+2}' for i in range(n_terms)]
    for label, param in zip(param_labels, fitted_params):
        print(f'{label} = {param:.4f}')