import pytest
from src.cost_function import compute_cost, average_percent_error
import numpy as np

def test_compute_cost():
    y_actual = np.array([1, 2, 3, 4, 5])
    y_predicted = np.array([1.1, 1.9, 3.2, 3.8, 4.9])
    
    expected_mse = 0.034
    mse = compute_cost(y_actual, y_predicted)
    
    assert np.isclose(mse, expected_mse, atol=0.001), f"Expected MSE {expected_mse}, but got {mse}"


def test_average_percent_error():
    actual = np.array([100, 200, 300, 400, 500])
    predicted = np.array([110, 190, 310, 390, 510])
    
    expected_avg_percent_error = 3.8  # Computed manually for the given values
    avg_percent_error = average_percent_error(actual, predicted)
    
    assert np.isclose(avg_percent_error, expected_avg_percent_error, atol=0.1), f"Expected average percent error {expected_avg_percent_error}, but got {avg_percent_error}"

if __name__ == "__main__":
    pytest.main()
