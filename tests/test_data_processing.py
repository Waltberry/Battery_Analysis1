import pytest
import pandas as pd
from src.data_processing import identify_charging_cycles

def test_identify_charging_cycles():
    data = pd.DataFrame({
        'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'value': [0, 1, 2, 3, 2, 0, 1, 2, 3, 4]
    })

    expected_cycles = [
        [(2, 1), (3, 2), (4, 3)],  # First cycle from time 2 to 4
        [(7, 1), (8, 2), (9, 3), (10, 4)]  # Second cycle from time 7 to 10
    ]

    cycles = identify_charging_cycles(data, 'time', 'value')

    assert cycles == expected_cycles, f"Expected {expected_cycles}, but got {cycles}"

if __name__ == "__main__":
    pytest.main()
