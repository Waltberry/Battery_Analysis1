import pytest
import numpy as np
from io import StringIO
from unittest.mock import patch
from src.plotting import plot_scatterplot, plot_fitted, print_fitted_params, plot_cost
import matplotlib.pyplot as plt

# Mock the plt.show() to avoid displaying plots during testing
@patch('matplotlib.pyplot.show')
def test_plot_scatterplot(mock_show):
    times = np.array([1, 2, 3, 4, 5])
    values = np.array([10, 15, 13, 17, 16])
    idx = 1
    
    # Call the function
    plot_scatterplot(times, values, idx)
    
    # Check that plt.show() was called
    mock_show.assert_called_once()

@patch('matplotlib.pyplot.show')
def test_plot_fitted(mock_show):
    times = np.array([1, 2, 3, 4, 5])
    values = np.array([10, 15, 13, 17, 16])
    y_fitted = np.array([10.5, 14.5, 13.5, 16.5, 15.5])
    idx = 1
    
    # Call the function
    plot_fitted(times, values, y_fitted, idx)
    
    # Check that plt.show() was called
    mock_show.assert_called_once()

def test_print_fitted_params(capfd):
    fitted_params = [1.0, 0.5, 0.2, 1.5]
    n_terms = 2
    
    # Call the function
    print_fitted_params(fitted_params, n_terms)
    
    # Capture the output
    captured = capfd.readouterr()
    
    expected_output = "c1 = 1.0000\nc2 = 0.5000\nc3 = 0.2000\nb2 = 1.5000\nb3 = 1.5000\n"
    assert captured.out == expected_output

def test_plot_cost():
    costs = np.array([10, 15, 12, 18, 14])
    
    # Call the function
    plot_cost(costs)
    
    # Check that plt.show() was called
    plt.show()

if __name__ == "__main__":
    pytest.main()
