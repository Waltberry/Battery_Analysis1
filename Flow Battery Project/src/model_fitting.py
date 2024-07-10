import numpy as np
from scipy.optimize import curve_fit
from src.plotting import plot_fitted, print_fitted_params
from src.cost_function import compute_cost
from src.model import generalized_exponential_model

def fit_model(times, values, n_terms=2, idx=None):
    """
    Fits a generalized exponential model to the given data using curve fitting.

    Parameters:
    times : numpy array
        Array of time values.
    values : numpy array
        Array of corresponding data values to fit the model.
    n_terms : int, optional
        Number of exponential terms in the model (default is 2).
    idx : int, optional
        Index of the charging cycle (default is None).

    Returns:
    tuple
        Tuple containing:
        - numpy array: Original times array.
        - numpy array: Original values array.
        - numpy array: Fitted values of the model.
        - list: Fitted parameters of the model.
    """
    if len(times) < 2 or len(values) < 2:
        print(f"Not enough data points to fit model for Charging Cycle {idx+1}.")
        return times, values, None, None, False  # Indicate fitting was not successful
    
    valid_indices = np.isfinite(times) & np.isfinite(values)
    times = times[valid_indices]
    values = values[valid_indices]
    
    if len(times) < 2 or len(values) < 2:
        print(f"Not enough valid data points to fit model for Charging Cycle {idx+1} after removing NaNs and infinite values.")
        return times, values, None, None, False  # Indicate fitting was not successful 
    
    print(f"Fitting model for Charging Cycle {idx+1} with {len(times)} data points.")
    
    # Normalize the data to avoid overflow issues
    x_mean = np.mean(times)
    x_std = np.std(times)
    x_normalized = (times - x_mean) / x_std if x_std != 0 else times - x_mean

    y_mean = np.mean(values)
    y_std = np.std(values)
    y_normalized = (values - y_mean) / y_std if y_std != 0 else values - y_mean
    
    # Initial guesses for the parameters
    initial_guess = [0] + [1] * n_terms + [0.1] * n_terms
    
    # Define bounds for the parameters to avoid overflow issues
    lower_bounds = [-np.inf] + [-np.inf] * n_terms + [0] * n_terms
    upper_bounds = [np.inf] + [np.inf] * n_terms + [np.inf] * n_terms
    
    try:
        # Fit the model
        params, covariance = curve_fit(generalized_exponential_model, x_normalized, y_normalized, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=5000)
        fitted_params = params
        
        # Generate fitted values
        y_fitted_normalized = generalized_exponential_model(x_normalized, *fitted_params)
        
        
        y_fitted = y_fitted_normalized * y_std + y_mean
        
        return times, values, y_fitted, fitted_params, True  # Indicate fitting was successful
    except Exception as e:
        print(f"Could not fit model for Charging Cycle {idx+1}: {e}")
        return times, values, None, None, False  # Indicate fitting was not successful


def fit_and_plot_cycle(times, values, idx, n_terms=2):
    """
    Fits a generalized exponential model to the given data using fit_model,
    plots the data and the fitted model, and prints the fitted parameters.

    Parameters:
    times : numpy array
        Array of time values.
    values : numpy array
        Array of corresponding data values to fit the model.
    idx : int
        Index of the charging cycle.
    n_terms : int, optional
        Number of exponential terms in the model (default is 2).

    Returns:
    tuple
        Tuple containing:
        - numpy array: Original times array.
        - numpy array: Original values array.
        - numpy array: Fitted values of the model.
        - list: Fitted parameters of the model.
    """
    try:
        # Use the fit_model function to fit the model and get fitted values and parameters
        times, values, y_fitted, fitted_params, success = fit_model(times, values, n_terms, idx=idx)
        if success:
            # Plot the data and the fitted model
            plot_fitted(times, values, y_fitted, idx)
            # Print the fitted parameters
            print_fitted_params(fitted_params, n_terms)
            
            # Calculate and print the cost
            cost = compute_cost(values, y_fitted)
            print(f"Cost for Charging Cycle {idx+1}: {cost}")
        
        return times, values, y_fitted, fitted_params, success
    except Exception as e:
        print(f"Could not fit model for Charging Cycle {idx+1}: {e}")
        return times, values, None, None, False