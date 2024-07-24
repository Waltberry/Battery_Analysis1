
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.data_processing import identify_charging_cycles
import pandas as pd
import unittest

class TestDataProcessing(unittest.TestCase):

    def test_identify_charging_cycles(self):
        data = pd.DataFrame({
            'time': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'value': [0, 1, 2, 3, 2, 0, 1, 2, 3, 4]
        })

        expected_cycles = [
            [(2, 1), (3, 2), (4, 3)], 
            [(7, 1), (8, 2), (9, 3), (10, 4)]
        ]

        cycles = identify_charging_cycles(data, 'time', 'value')

        self.assertEqual(cycles, expected_cycles, f"Expected {expected_cycles}, but got {cycles}")

if __name__ == "__main__":
    unittest.main()
