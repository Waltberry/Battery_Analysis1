import numpy as np

def discrete_time_system_identification(u, y, order):
    """
    Discrete Time System Identification using least squares method.

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
    """
    n_samples = len(u)
    
    # Construct the Phi matrix
    Phi = np.zeros((n_samples - order, 2 * order + 1))
    for i in range(order, n_samples):
        u_slice = u[i-order:i][::-1]
        y_slice = -y[i-1:i-order-1:-1]
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
