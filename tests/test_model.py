import pytest
from src.model import arx_model, generalized_exponential_model, complexpoles_exponential_model, fit_straight_line
import numpy as np
import scipy.signal as sp

def test_arx_model():
    B = np.array([0.5, 0.25])
    A = np.array([1, 1.6, 0.95])
    N = 1000
    
    rng = np.random.default_rng(seed=42)  # Provide a seed for reproducibility
    u = rng.random(N)
    y = sp.lfilter(B, A, u)
    
    theta, _, _ = arx_model(u, y, 2)
    
    expected_theta = np.array([0.5, 0.25, 1.6, 0.95])
    assert np.allclose(theta, expected_theta, atol=0.01), f"Expected theta {expected_theta}, but got {theta}"


def test_generalized_exponential_model():
    t = np.array([0, 1, 2, 3, 4, 5])
    params = [2, 1, 0.5, 0.5, 0.2]
    
    expected_result = np.array([2. , 2.90634623, 2.77969898, 2.65074174, 2.52587631, 2.40600585])
    result = generalized_exponential_model(t, *params)
    
    assert np.allclose(result, expected_result, atol=0.01), f"Expected {expected_result}, but got {result}"


def test_complexpoles_exponential_model():
    t = np.array([0, 1, 2, 3, 4, 5])
    params = [2, 1, 0.5, 0.5, 0.2, 0.5, 0.2, 0.3, 0.1]
    
    expected_result = np.array([3. , 3.38852688, 3.23893611, 3.11467177, 3.01470758, 2.9371073 ])
    result = complexpoles_exponential_model(t, *params)
    
    assert np.allclose(result, expected_result, atol=0.01), f"Expected {expected_result}, but got {result}"


def test_fit_straight_line():
    x = np.array([0, 1, 2, 3, 4, 5])
    y = np.array([0, 2, 4, 6, 8, 10])
    
    slope, intercept = fit_straight_line(x, y)
    
    expected_slope = 2
    expected_intercept = 0
    
    assert np.isclose(slope, expected_slope, atol=0.01), f"Expected slope {expected_slope}, but got {slope}"
    assert np.isclose(intercept, expected_intercept, atol=0.01), f"Expected intercept {expected_intercept}, but got {intercept}"

if __name__ == "__main__":
    pytest.main()
