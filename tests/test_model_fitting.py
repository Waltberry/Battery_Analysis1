import pytest
import numpy as np
from src.model_fitting import fit_model, fit_and_plot_cycle


def test_fit_model():
    times = np.array([0, 1, 2, 3, 4, 5])
    values = np.array([1, 2, 3, 4, 3, 2])
    
    expected_fitted_params = [0.5, 0.25, 1.6, 0.95]
    
    _, _, y_fitted, fitted_params, success = fit_model(times, values, n_terms=1, idx=0)
    
    assert success == True, "Model fitting was not successful"
    assert fitted_params is not None, "Fitted parameters should not be None"
    assert np.allclose(fitted_params, expected_fitted_params, atol=0.1), f"Expected parameters {expected_fitted_params}, but got {fitted_params}"

def test_fit_and_plot_cycle():
    times = np.array([0, 1, 2, 3, 4, 5])
    values = np.array([1, 2, 3, 4, 3, 2])
    
    _, _, y_fitted, fitted_params, success = fit_and_plot_cycle(times, values, idx=0, n_terms=1)
    
    assert success == True, "Model fitting was not successful"
    assert fitted_params is not None, "Fitted parameters should not be None"
    assert y_fitted is not None, "Fitted values should not be None"

if __name__ == "__main__":
    pytest.main()
