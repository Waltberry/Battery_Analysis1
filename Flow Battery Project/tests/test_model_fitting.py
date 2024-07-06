import pytest
import numpy as np
from src.model_fitting import fit_model

def test_fit_model():
    times = np.array([0, 1, 2, 3, 4])
    values = np.array([0, 1, 2, 1, 0])
    times, values, y_fitted, params = fit_model(times, values)
    assert len(params) == 7  # Assuming n_terms=3
    assert len(y_fitted) == len(values)
