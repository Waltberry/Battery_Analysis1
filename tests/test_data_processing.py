import pytest
import pandas as pd
from src.data_processing import identify_charging_cycles

def test_identify_charging_cycles():
    data = pd.DataFrame({
        'time/hours': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        'Ewe/mV.4': [0, 1, 2, 1, 0, 1, 2, 3, 2, 1]
    })
    cycles = identify_charging_cycles(data, 'time/hours', 'Ewe/mV.4')
    assert len(cycles) == 2
    assert cycles[0] == [(1, 1), (2, 2), (3, 1)]
    assert cycles[1] == [(5, 1), (6, 2), (7, 3), (8, 2), (9, 1)]
