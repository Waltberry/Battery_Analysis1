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


def average_percent_error(actual, predicted):
    """
    Calculate the average percent error between actual and predicted values.
    
    Parameters:
    actual (array-like): Array of actual values.
    predicted (array-like): Array of predicted values.
    
    Returns:
    float: Average percent error.
    """
    # # Ensure the input arrays are numpy arrays
    # actual = np.array(actual)
    # predicted = np.array(predicted)
    
    # Calculate the absolute percent error for each point
    percent_errors = np.abs((actual - predicted) / actual) * 100
    
    # Calculate the average percent error
    average_percent_error = np.mean(percent_errors)
    
    return average_percent_error

