import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from unittest.mock import patch
import unittest
import numpy as np
from src.model_fitting import fit_model, fit_and_plot_cycle

class TestModelFitting(unittest.TestCase):

    @patch('matplotlib.pyplot.show')
    def test_fit_model(self, mock_show):
        times = np.array([0, 1, 2, 3, 4, 5])
        values = np.array([1, 2, 3, 4, 3, 2])
        
        _, _, y_fitted, fitted_params, success = fit_model(times, values, n_terms=1, idx=0)
        
        # Check that the fitting process was successful
        self.assertTrue(success, "Model fitting was not successful")
        
        # Check that fitted parameters and fitted values are not None
        self.assertIsNotNone(fitted_params, "Fitted parameters should not be None")
        self.assertIsNotNone(y_fitted, "Fitted values should not be None")
        
        # Print the actual parameters for debugging
        print(f"Actual fitted parameters: {fitted_params}")

    @patch('matplotlib.pyplot.show')
    @patch('src.model_fitting.plot_fitted')  # Assuming plot_fitted is called in fit_and_plot_cycle
    def test_fit_and_plot_cycle(self, mock_plot_fitted, mock_show):
        times = np.array([0, 1, 2, 3, 4, 5])
        values = np.array([1, 2, 3, 4, 3, 2])
        
        _, _, y_fitted, fitted_params, success = fit_and_plot_cycle(times, values, idx=0, n_terms=1)
        
        # Check that the fitting process was successful
        self.assertTrue(success, "Model fitting was not successful")
        
        # Check that fitted parameters and fitted values are not None
        self.assertIsNotNone(fitted_params, "Fitted parameters should not be None")
        self.assertIsNotNone(y_fitted, "Fitted values should not be None")

if __name__ == "__main__":
    unittest.main()
