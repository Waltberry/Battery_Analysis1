import numpy as np
from scipy.optimize import curve_fit

def generalized_exponential_model(t, *params):
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
