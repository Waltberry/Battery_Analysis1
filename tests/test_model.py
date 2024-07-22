# import pytest
import pandas as pd
from src.model import discrete_time_system_identification
import numpy as np
import scipy.signal as sp

def test_discrete_time_system_identification(B=np.array([0.5, 0.25]), A =np.array([1, 1.6, 0.95]), N=1000):
    
    u = np.random.rand(N)
    y = sp.lfilter(B,A,u)
    theta, Phi, Y = discrete_time_system_identification(u, y, 3)
    # assert Y = 
    # assert theta = []
    # assert Phi =
    
    c = print(theta) 
    return c

call = test_discrete_time_system_identification()

print(call)

