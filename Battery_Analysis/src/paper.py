import numpy as np
import scipy.signal as signal
from scipy.linalg import lstsq

class BatteryParameterEstimator:
    def __init__(self, time, voltage, current):
        self.time = np.array(time)
        self.voltage = np.array(voltage)
        self.current = np.array(current)
        self.dt = np.mean(np.diff(self.time))
    
    def state_variable_filter(self, input_signal, alpha=0.01, order=2):
        """Apply a State Variable Filter (SVF) to smooth signal derivatives."""
        num, den = signal.butter(order, alpha, analog=False)
        return signal.filtfilt(num, den, input_signal)
    
    def construct_regression_matrix(self):
        """Construct the regression matrix for parameter estimation."""
        dv_dt = np.gradient(self.voltage, self.dt)
        dv2_dt2 = np.gradient(dv_dt, self.dt)
        
        u = np.column_stack([
            -dv_dt,   # First derivative
            -self.voltage,  # Voltage itself
            self.current,  # Current
            np.gradient(self.current, self.dt)  # Current derivative
        ])
        return u, dv2_dt2
    
    def least_squares_estimation(self):
        """Solve for battery parameters using Least Squares estimation."""
        u, y = self.construct_regression_matrix()
        params, _, _, _ = lstsq(u, y)
        return params
    
    def instrumental_variable_estimation(self):
        """Solve for battery parameters using Instrumental Variable estimation."""
        u, y = self.construct_regression_matrix()
        instrument = self.state_variable_filter(self.voltage)
        u_iv = np.column_stack([instrument, u[:, 1:]])  
        params, _, _, _ = lstsq(u_iv, y)
        return params
    
    def estimate_parameters(self):
        """Estimate battery parameters using both LS and IV methods."""
        params_ls = self.least_squares_estimation()
        params_iv = self.instrumental_variable_estimation()
        return params_ls, params_iv
