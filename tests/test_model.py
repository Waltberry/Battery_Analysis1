import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))


import unittest
import numpy as np
import scipy.signal as sp
from src.model import arx_model, generalized_exponential_model, complexpoles_exponential_model, fit_straight_line

class TestModels(unittest.TestCase):

    def test_arx_model(self):
        B = np.array([0.5, 0.25])
        A = np.array([1, 1.6, 0.95])
        N = 1000
        
        rng = np.random.default_rng(seed=42)  # Provide a seed for reproducibility
        u = rng.random(N)
        y = sp.lfilter(B, A, u)
        
        theta, _, _ = arx_model(u, y, 2)
        
        expected_theta = np.array([0.5, 0.25, 1.6, 0.95])
        self.assertTrue(np.allclose(theta, expected_theta, atol=0.01), f"Expected theta {expected_theta}, but got {theta}")

    def test_generalized_exponential_model(self):
        t = np.array([0, 1, 2, 3, 4, 5])
        params = [2, 1, 0.5, 0.5, 0.2]
        
        expected_result = np.array([3.5, 3.01589604, 2.70303946, 2.49753598, 2.35999977, 2.26602472])
        result = generalized_exponential_model(t, *params)
        
        self.assertTrue(np.allclose(result, expected_result, atol=0.01), f"Expected {expected_result}, but got {result}")

    def test_complexpoles_exponential_model(self):
        t = np.array([0, 1, 2, 3, 4, 5])
        params = [2, 1, 0.5, 0.5, 0.2, 0.5, 0.2, 0.3, 0.1]
        
        expected_result = np.array([5.0, 4.07608048, 3.40242257, 2.93077395, 2.60999605, 2.39637843])
        result = complexpoles_exponential_model(t, *params)
        
        self.assertTrue(np.allclose(result, expected_result, atol=0.01), f"Expected {expected_result}, but got {result}")

    def test_fit_straight_line(self):
        x = np.array([0, 1, 2, 3, 4, 5])
        y = np.array([0, 2, 4, 6, 8, 10])
        
        slope, intercept = fit_straight_line(x, y)
        
        expected_slope = 2
        expected_intercept = 0
        
        self.assertAlmostEqual(slope, expected_slope, delta=0.01, msg=f"Expected slope {expected_slope}, but got {slope}")
        self.assertAlmostEqual(intercept, expected_intercept, delta=0.01, msg=f"Expected intercept {expected_intercept}, but got {intercept}")

if __name__ == "__main__":
    unittest.main()
