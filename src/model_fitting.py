import numpy as np
from scipy.optimize import curve_fit
from src.plotting import plot_fitted, print_fitted_params
from src.cost_function import compute_cost, average_percent_error
from src.model import generalized_exponential_model

def fit_model(times, values, n_terms=1, idx=None):
    """
    Fits a generalized exponential model to the given data using curve fitting.

    Parameters:
    times : numpy array
        Array of time values.
    values : numpy array
        Array of corresponding data values to fit the model.
    n_terms : int, optional
        Number of exponential terms in the model.
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
        return times, values, None, None, False 
    
    valid_indices = np.isfinite(times) & np.isfinite(values)
    times = times[valid_indices]
    values = values[valid_indices]
    
    if len(times) < 2 or len(values) < 2:
        print(f"Not enough valid data points to fit model for Charging Cycle {idx+1} after removing NaNs and infinite values.")
        return times, values, None, None, False 
    
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
    # Define the lower and upper bounds for the parameters in the curve fitting process.
    # The generalized_exponential_model function has parameters in the following order:
    # [c1, c2, ..., c_n, b1, b2, ..., b_n]
    # where c's are the coefficients and b's are the exponents.

    # `lower_bounds` specifies the lower limits for these parameters.
    # The first parameter (the offset) can be any value, so its lower bound is -infinity.
    # The next `n_terms` parameters (the c coefficients) can also be any value, so their lower bounds are -infinity.
    # The last `n_terms` parameters (the b exponents) must be non-negative, so their lower bounds are 0.
    lower_bounds = [-np.inf] + [-np.inf] * n_terms + [0] * n_terms

    # `upper_bounds` specifies the upper limits for these parameters.
    # The first parameter (the offset) can be any value, so its upper bound is infinity.
    # The next `n_terms` parameters (the c coefficients) can also be any value, so their upper bounds are infinity.
    # The last `n_terms` parameters (the b exponents) can be any value, so their upper bounds are infinity.
    upper_bounds = [np.inf] + [np.inf] * n_terms + [np.inf] * n_terms

    # We use these bounds in the curve_fit function to constrain the optimization process.
    # `curve_fit` will try to find the optimal parameters within the specified bounds.
    
    '''lower_bounds:
    The first parameter is allowed to be any real number (-inf).
    The next n_terms parameters (c coefficients) are also allowed to be any real number (-inf).
    The last n_terms parameters (b exponents) are constrained to be non-negative (0).
    
    upper_bounds:
    All parameters are allowed to be any real number (inf), meaning there is no upper constraint.'''

    
    try:
        # Fit the model
        params, _ = curve_fit(generalized_exponential_model, x_normalized, y_normalized, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=5000)
        fitted_params = params
        
        # Generate fitted values
        y_fitted_normalized = generalized_exponential_model(x_normalized, *fitted_params)
        y_fitted = y_fitted_normalized * y_std + y_mean
        
        return times, values, y_fitted, fitted_params, True  # Indicate fitting was successful
    except Exception as e:
        print(f"Could not fit model for Charging Cycle {idx+1}: {e}")
        return times, values, None, None, False


def fit_and_plot_cycle(times, values, idx, n_terms=1):
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
            
            # Calculate and print the average percent error
            ape = average_percent_error(values, y_fitted)
            print(f"average percent error for Charging Cycle {idx+1}: {ape}")
        
        return times, values, y_fitted, fitted_params, success
    except Exception as e:
        print(f"Could not fit model for Charging Cycle {idx+1}: {e}")
        return times, values, None, None, False