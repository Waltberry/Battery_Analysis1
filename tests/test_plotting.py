import pytest
from src.plotting import plot_cycle, print_fitted_params

def test_plot_cycle():
    times = [0, 1, 2, 3, 4]
    values = [0, 1, 2, 1, 0]
    y_fitted = [0, 1, 2, 1, 0]
    plot_cycle(times, values, y_fitted, 1)

def test_print_fitted_params():
    params = [0, 1, 2, 3, 4, 5, 6]
    print_fitted_params(params, 3)
