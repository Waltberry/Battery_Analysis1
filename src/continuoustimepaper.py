import numpy as np
import scipy.signal as signal
import scipy.optimize as optimize
import control as ct
import matplotlib.pyplot as plt

N = 1000000
dt = 1/100
t = np.linspace(0,dt*N,N)

u_step = np.ones(N)
u_step[0:100] = np.zeros(100)

def model_response(params, t, u_step):
    R0, R1, R2, tau1, tau2 = params
    C1, C2 = tau1 / R1, tau2 / R2  # Compute C values
    
    num1, den1 = [R1], [R1 * C1, 1]
    num2, den2 = [R2], [R2 * C2, 1]
    num3, den3 = [R0], [1]
    
    H1 = ct.TransferFunction(num1, den1)
    H2 = ct.TransferFunction(num2, den2)
    H3 = ct.TransferFunction(num3, den3)
    H_model = H1 + H2 + H3
    
    tout, y_model = ct.forced_response(H_model, T=t, U=u)
    return y_model

def objective_function(params):
    y_model = model_response(params, t, u_step)
    return np.sum((y_model - yout_noisy) ** 2)
