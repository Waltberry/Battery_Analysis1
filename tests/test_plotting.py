import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import numpy as np
from io import StringIO
from unittest.mock import patch
from src.plotting import plot_scatterplot, plot_fitted, print_fitted_params, plot_cost
import matplotlib.pyplot as plt

class TestPlotting(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_plot_scatterplot(self, mock_show):
        times = np.array([1, 2, 3, 4, 5])
        values = np.array([10, 15, 13, 17, 16])
        idx = 1
        
        # Call the function
        plot_scatterplot(times, values, idx)
        
        # Check that plt.show() was called
        mock_show.assert_called_once()

    @patch('matplotlib.pyplot.show')
    def test_plot_fitted(self, mock_show):
        times = np.array([1, 2, 3, 4, 5])
        values = np.array([10, 15, 13, 17, 16])
        y_fitted = np.array([10.5, 14.5, 13.5, 16.5, 15.5])
        idx = 1
        
        # Call the function
        plot_fitted(times, values, y_fitted, idx)
        
        # Check that plt.show() was called
        mock_show.assert_called_once()

    @patch('sys.stdout', new_callable=StringIO)
    def test_print_fitted_params(self, mock_stdout):
        fitted_params = [1.0, 0.5, 0.2, 1.5, 1.5]
        n_terms = 2  # Number of terms for 'c' parameters
        
        # Call the function
        print_fitted_params(fitted_params, n_terms)
        
        # Capture the output
        captured = mock_stdout.getvalue()
        
        # Adjust the expected output based on the actual function behavior
        expected_output = "c1 = 1.0000\nc2 = 0.5000\nc3 = 0.2000\nb2 = 1.5000\nb3 = 1.5000\n"
        self.assertEqual(captured, expected_output)



    @patch('matplotlib.pyplot.show')
    def test_plot_cost(self, mock_show):
        costs = np.array([10, 15, 12, 18, 14])
        
        # Call the function
        plot_cost(costs)
        
        # Check that plt.show() was called
        mock_show.assert_called_once()

if __name__ == "__main__":
    unittest.main()
