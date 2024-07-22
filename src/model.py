import numpy as np
from scipy.stats import linregress

def discrete_time_system_identification(u, y, order):
    """
    Discrete Time System Identification using least squares method.

    This function identifies the parameters of a discrete-time system given input-output data
    using the least squares method. The system is assumed to be linear and represented by
    a difference equation of a specified order.

    Mathematical Expression:
    The system is modeled as:
    y[k] = -sum(a_i * y[k-i]) + sum(b_i * u[k-i]) + e[k]

    where:
    - a_i are the coefficients for the past outputs
    - b_i are the coefficients for the past inputs
    - e[k] is the error term

    In matrix form, we construct:
    Phi = [[u[k-1], u[k-2], ..., u[k-order], -y[k-1], -y[k-2], ..., -y[k-order], 1], ...]
    Y = [y[order], y[order+1], ..., y[n_samples-1]]

    The parameters theta are obtained by solving:
    theta = (Phi.T * Phi)^(-1) * Phi.T * Y

    Parameters:
    u : numpy array
        Array of input values.
    y : numpy array
        Array of output values.
    order : int
        Order of the system.

    Returns:
    theta : numpy array
        Estimated parameters of the system.
    Phi : numpy array
        The constructed data matrix.
    Y : numpy array
        The vector of output values aligned with Phi.
    """
    n_samples = len(u)
    
    # Construct the Phi matrix
    Phi = np.zeros((n_samples - order, 2 * order + 1))
    for i in range(order, n_samples):
        u_slice = u[i-order:i][::-1] if i-order >= 0 else np.zeros(order)
        y_slice = -y[i-1:i-order-1:-1] if i-1-(i-order-1) >= 0 else np.zeros(order)
        
        # Debug prints to verify the slices
        print(f"i: {i}, u_slice: {u_slice}, y_slice: {y_slice}")

        Phi[i - order, :order] = u_slice
        Phi[i - order, order:2*order] = y_slice
        Phi[i - order, -1] = 1
    
    # Construct the Y vector
    Y = y[order:]
    
    # Calculate theta using the normal equation
    theta = np.linalg.inv(Phi.T @ Phi) @ Phi.T @ Y
    
    return theta, Phi, Y


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
    n_terms = (len(params) - 1) // 2
    c1 = params[0]
    result = c1
    for i in range(n_terms):
        ci = params[1 + 2 * i]
        bi = params[2 + 2 * i]
        result += ci * np.exp(-bi * t)
    return result

def complexpoles_exponential_model(t, *params):
    """
    Generalized Exponential Model including complex poles:
    
    Parameters:
    - t: Independent variable (time or another independent parameter)
    - params: Parameters of the model, formatted as [c0, a1, b1, alpha1, beta1, a2, b2, alpha2, beta2, ...]
              where a, b, alpha, and beta are coefficients for each complex exponential term.
    
    Formula:
    y(t) = c0 + sum(2a * exp(-alpha * t) * cos(beta * t) + 2b * exp(-alpha * t) * sin(beta * t) for i in range(n_terms))
    
    Explanation:
    - c0 is the initial value or constant offset.
    - a is the amplitude coefficient for the cosine term.
    - b is the amplitude coefficient for the sine term.
    - alpha is the real part of the pole (decay rate).
    - beta is the imaginary part of the pole (oscillation frequency).
    - n_terms is the number of complex exponential terms, calculated as (len(params) - 1) // 4.
    
    Args:
    - t (float or array-like): Input variable (time or other independent variable).
    - *params (float): Parameters of the model, should be in the format described.
    
    Returns:
    - float or array-like: Value(s) of the model at given t.
    """
    n_terms = (len(params) - 1) // 4
    c0 = params[0]
    result = c0
    for i in range(n_terms):
        a = params[1 + 4 * i]
        b = params[2 + 4 * i]
        alpha = params[3 + 4 * i]
        beta = params[4 + 4 * i]
        result += 2 * a * np.exp(-alpha * t) * np.cos(beta * t)
        result += 2 * b * np.exp(-alpha * t) * np.sin(beta * t)
    return result

# Fit straight-line models
def fit_straight_line(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept

