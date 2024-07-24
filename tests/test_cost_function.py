import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import unittest
import numpy as np
import src.cost_function as cf

class TestCostFunction(unittest.TestCase):

    def test_compute_cost(self):
        y_actual = np.array([1, 2, 3, 4, 5])
        y_predicted = np.array([1.1, 1.9, 3.2, 3.8, 4.9])
        
        expected_mse = 0.022
        mse = cf.compute_cost(y_actual, y_predicted)
        
        self.assertAlmostEqual(mse, expected_mse, delta=0.001, msg=f"Expected MSE {expected_mse}, but got {mse}")

    def test_average_percent_error(self):
        actual = np.array([100, 200, 300, 400, 500])
        predicted = np.array([110, 190, 310, 390, 510])
        
        expected_avg_percent_error = 4.567  
        avg_percent_error = cf.average_percent_error(actual, predicted)
        
        self.assertAlmostEqual(avg_percent_error, expected_avg_percent_error, delta=0.1, msg=f"Expected average percent error {expected_avg_percent_error}, but got {avg_percent_error}")

if __name__ == "__main__":
    unittest.main()
