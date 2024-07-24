import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import numpy as np
from src.model_fitting import fit_model, fit_and_plot_cycle

class TestModelFitting(unittest.TestCase):

    def test_fit_model(self):
        times = np.array([0, 1, 2, 3, 4, 5])
        values = np.array([1, 2, 3, 4, 3, 2])
        
        expected_fitted_params = [0.5, 0.25, 1.6, 0.95]
        
        _, _, y_fitted, fitted_params, success = fit_model(times, values, n_terms=1, idx=0)
        
        self.assertTrue(success, "Model fitting was not successful")
        self.assertIsNotNone(fitted_params, "Fitted parameters should not be None")
        for expected, actual in zip(expected_fitted_params, fitted_params):
            self.assertAlmostEqual(expected, actual, delta=0.1, msg=f"Expected parameter {expected}, but got {actual}")

    def test_fit_and_plot_cycle(self):
        times = np.array([0, 1, 2, 3, 4, 5])
        values = np.array([1, 2, 3, 4, 3, 2])
        
        _, _, y_fitted, fitted_params, success = fit_and_plot_cycle(times, values, idx=0, n_terms=1)
        
        self.assertTrue(success, "Model fitting was not successful")
        self.assertIsNotNone(fitted_params, "Fitted parameters should not be None")
        self.assertIsNotNone(y_fitted, "Fitted values should not be None")

if __name__ == "__main__":
    unittest.main()
