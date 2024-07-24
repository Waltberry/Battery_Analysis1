import matplotlib.pyplot as plt

def plot_scatterplot(times, values, idx=None, x_label='Time (hours)', y_label='Ewe/mV.4', title=None):
    """
    Plot data as a scatter plot for a given charging cycle.

    This function creates a scatter plot of the provided time and value data,
    with customizable x and y labels, and an optional title. If no title is provided,
    it defaults to 'Charging Cycle {idx+1}' if idx is given, otherwise '{y_label} vs {x_label}'.

    Parameters:
    times (list or array-like): List or array of time values.
    values (list or array-like): List or array of corresponding data values.
    idx (int, optional): Index of the charging cycle.
    x_label (str): Label for the x-axis. Defaults to 'Time (hours)'.
    y_label (str): Label for the y-axis. Defaults to 'Ewe/mV.4'.
    title (str, optional): Title for the plot. Defaults to None.

    Returns:
    None
    """
    plt.figure()
    plt.scatter(times, values, label=f'Charging Cycle {idx+1}' if idx is not None else '', alpha=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    if title:
        plt.title(title)
    else:
        plt.title(f'Charging Cycle {idx+1}' if idx is not None else f'{y_label} vs {x_label}')
    if idx is not None:
        plt.legend()
    plt.show()

def plot_scatterplot_on_ax(ax, times, values, idx=None, x_label='Time (hours)', y_label='Ewe/mV.4', title=None):
    """
    Plot data as a scatter plot on the provided axes for a given charging cycle.

    This function creates a scatter plot of the provided time and value data,
    with customizable x and y labels, and an optional title. If no title is provided,
    it defaults to 'Charging Cycle {idx+1}' if idx is given, otherwise '{y_label} vs {x_label}'.

    Parameters:
    ax (matplotlib.axes.Axes): The axes to plot on.
    times (list or array-like): List or array of time values.
    values (list or array-like): List or array of corresponding data values.
    idx (int, optional): Index of the charging cycle.
    x_label (str): Label for the x-axis. Defaults to 'Time (hours)'.
    y_label (str): Label for the y-axis. Defaults to 'Ewe/mV.4'.
    title (str, optional): Title for the plot. Defaults to None.

    Returns:
    None
    """
    ax.scatter(times, values, label=f'Charging Cycle {idx+1}' if idx is not None else '', alpha=0.5)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f'Charging Cycle {idx+1}' if idx is not None else f'{y_label} vs {x_label}')
    if idx is not None:
        ax.legend()


def plot_fitted(times, values, y_fitted, idx):
    """
    Plot data and a fitted model for a given charging cycle.

    This function creates a scatter plot of the data and overlays it with 
    the fitted model values.

    Parameters:
    times (list or array-like): List or array of time values.
    values (list or array-like): List or array of corresponding data values.
    y_fitted (list or array-like): List or array of fitted values corresponding to the times.
    idx (int): Index of the charging cycle.

    Returns:
    None
    """
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
    Print fitted parameters of the model.

    This function prints the fitted parameters for the model, with labels for
    each coefficient based on the number of terms.

    Parameters:
    fitted_params (list or array-like): List of fitted parameters.
    n_terms (int, optional): Number of terms in the model. Defaults to 3.

    Returns:
    None
    """
    param_labels = ['c1'] + [f'c{i+2}' for i in range(n_terms)] + [f'b{i+2}' for i in range(n_terms)]
    for label, param in zip(param_labels, fitted_params):
        print(f'{label} = {param:.4f}')

def plot_cost(costs, name='Cost'):
    """
    Plot the costs of each charging cycle.

    This function creates a line plot of the costs for each charging cycle, 
    with the option to specify a custom name for the cost.

    Parameters:
    costs (list or array-like): List or array of costs for each charging cycle.
    name (str, optional): Name to display in the plot title. Defaults to 'Cost'.

    Returns:
    None
    """
    plt.figure(figsize=(10, 6))
    plt.plot(costs, marker='o', linestyle='-')
    plt.title(f'{name} of Each Charging Cycle')
    plt.xlabel('Charging Cycle Index')
    plt.ylabel(name)
    plt.grid(True)
    plt.show()
