import numpy as np
from scipy.optimize import curve_fit

def generalized_exponential_model(t, *params):
    """
    Generalized Exponential Model:
    
    Parameters:
    - t: Independent variable (time or another independent parameter)
    - params: Parameters of the model, formatted as [c1, c2, b2, c3, b3, ..., cn, bn]
             where ci and bi are coefficients for each exponential term
    
    Formula:
    y(t) = c1 + sum(ci * exp(-bi * t) for i in range(n_terms))
    
    Explanation:
    - c1 is the initial value or constant offset.
    - ci are coefficients for each exponential term.
    - bi are decay rates for each exponential term.
    - n_terms is the number of exponential terms, calculated as (len(params) - 1) // 2.
    
    Args:
    - t (float or array-like): Input variable (time or other independent variable).
    - *params (float): Parameters of the model, should be in the format described.
    
    Returns:
    - float or array-like: Value(s) of the model at given t.
    """
    # Assuming params = [c1, c2, b2, c3, b3, ..., cn, bn]
    n_terms = (len(params) - 1) // 2
    c1 = params[0]
    result = c1
    for i in range(n_terms):
        ci = params[1 + 2 * i]
        bi = params[2 + 2 * i]
        result += ci * np.exp(-bi * t)
    return result

def fit_model(times, values, n_terms=2):
    """
    Fits a generalized exponential model to the given data using curve fitting.

    Parameters:
    times : numpy array
        Array of time values.
    values : numpy array
        Array of corresponding data values to fit the model.
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
    # Normalize the data to avoid overflow issues
    x_mean = np.mean(times)
    x_std = np.std(times)
    x_normalized = (times - x_mean) / x_std

    y_mean = np.mean(values)
    y_std = np.std(values)
    y_normalized = (values - y_mean) / y_std
    
    # Initial guesses for the parameters
    initial_guess = [0] + [1] * n_terms + [0.1] * n_terms
    
    # Define bounds for the parameters to avoid overflow issues
    lower_bounds = [-np.inf] + [-np.inf] * n_terms + [0] * n_terms
    upper_bounds = [np.inf] + [np.inf] * n_terms + [np.inf] * n_terms
    
    # Fit the model
    params, covariance = curve_fit(generalized_exponential_model, x_normalized, y_normalized, p0=initial_guess, bounds=(lower_bounds, upper_bounds), maxfev=5000)
    fitted_params = params
    
    # Generate fitted values
    y_fitted_normalized = generalized_exponential_model(x_normalized, *fitted_params)
    
    # Convert fitted values back to original scale
    y_fitted = y_fitted_normalized * y_std + y_mean

    return times, values, y_fitted, fitted_params
