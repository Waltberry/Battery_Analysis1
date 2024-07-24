import numpy as np
from scipy.stats import linregress

def arx_model(u, y, order):
    """
    Estimate the parameters of an ARX (AutoRegressive with eXogenous inputs) model.

    Mathematical Expression:
    The ARX model is given by:
    
        y(t) = a1*y(t-1) + a2*y(t-2) + ... + an*y(t-n) + b1*u(t-1) + b2*u(t-2) + ... + bn*u(t-n) + e(t)
    
    In matrix form:
    
        Y = Φθ + e
    
    where:
        - Y is the vector of observed output values.
        - Φ (phi) is the matrix of past input and output values.
        - θ is the vector of parameters [b1, b2, ..., bn, -a1, -a2, ..., -an].
        - e is the error vector.

    The Φ (phi) matrix is constructed as follows:
    
        Φ = | u[n-1]  u[n-2]  ...  u[n-order]  -y[n-1]  -y[n-2]  ...  -y[n-order] |
            | u[n]    u[n-1]  ...  u[n-order+1] -y[n]    -y[n-1] ...  -y[n-order+1]|
            | ...    ...      ...  ...          ...      ...     ...  ...          |
            | u[N-1]  u[N-2]  ...  u[N-order]   -y[N-1]  -y[N-2] ...  -y[N-order]  |

    The parameter vector θ is estimated using the normal equation:
    
        θ = (Φ.T @ Φ)^(-1) @ Φ.T @ Y
    theta = (phi.T * phi)^(-1) * phi.T * Y

    Parameters:
        u (numpy array): The input signal (exogenous input).
        y (numpy array): The output signal.
        order (int): The order of the ARX model.
    
    Returns:
        theta (numpy array): The estimated parameters of the ARX model.
        phi (numpy array): The matrix of past input and output values.
        Y (numpy array): The vector of observed output values starting from the given order.
    """    
    
    n_samples = len(u)
    
    # Construct the phi matrix
    phi = np.zeros((n_samples - order, 2 * order))
    for i in range(order, n_samples):
        u_slice = np.flip(u[i-order+1:i+1])
        y_slice = np.flip(-y[i-order:i]) 
        
        phi[i - order, :order] = u_slice
        phi[i - order, order:2*order] = y_slice
    
    # Construct the Y vector
    Y = y[order:]
    
    # Calculate theta using the normal equation
    theta = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    
    return theta, phi, Y



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


def fit_straight_line(x, y):
    """
    Fit a straight line to the data using linear regression.

    Parameters:
    x : numpy array
        Array of x values.
    y : numpy array
        Array of y values.

    Returns:
    tuple
        Tuple containing:
        - slope: Slope of the fitted line.
        - intercept: Intercept of the fitted line.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept
