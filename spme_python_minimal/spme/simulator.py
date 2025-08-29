import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from .model import SPMe, SPMeParams

@dataclass
class SimulationResult:
    t: np.ndarray
    y: np.ndarray
    V: np.ndarray
    I: np.ndarray

def simulate_constant_current(I_app_A, t_end, spme: SPMe = None, y0=None):
    if spme is None:
        spme = SPMe(SPMeParams())
    g = spme.p.geom
    i_app = I_app_A / g.A_cell  # A/m^2

    if y0 is None:
        y0 = spme.initial_state()

    def rhs(t, y):
        return spme.rhs(t, y, i_app)

    sol = solve_ivp(rhs, [0.0, t_end], y0, method='BDF', max_step=t_end/500)
    V = np.array([spme.voltage(sol.y[:,k], i_app) for k in range(sol.y.shape[1])])
    I = np.full_like(V, I_app_A, dtype=float)
    return SimulationResult(sol.t, sol.y, V, I)
