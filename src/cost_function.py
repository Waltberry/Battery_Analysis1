import numpy as np

def compute_cost(y_actual, y_predicted):
    """
    Computes the mean squared error between the actual and predicted values.

    Parameters:
    y_actual : numpy array
        Actual output values.
    y_predicted : numpy array
        Predicted output values.

    Returns:
    float
        Mean squared error.
    """
    error = np.subtract(y_actual, y_predicted)
    squared_error = np.square(error)
    mse = np.mean(squared_error)
    return mse
