import pytest
from src.model import arx_model
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
    assert np.allclose(theta, expected_theta), f"Expected theta {expected_theta}, but got {theta}"

if __name__ == "__main__":
    pytest.main()
