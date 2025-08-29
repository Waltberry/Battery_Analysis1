from dataclasses import dataclass

@dataclass
class Geometry:
    # thicknesses [m]
    l_p: float = 75.6e-6
    l_s: float = 12e-6
    l_n: float = 85.2e-6
    # particle radii [m]
    R_p: float = 5.22e-6
    R_n: float = 5.86e-6
    # area [m^2]
    A_cell: float = 0.11

@dataclass
class PorosityBrugg:
    eps_p: float = 0.335
    eps_s: float = 0.47
    eps_n: float = 0.25
    brug_p: float = 2.43
    brug_s: float = 2.57
    brug_n: float = 2.91

@dataclass
class Transport:
    # electrolyte
    D_e: float = 1.0e-10     # bulk electrolyte diffusivity [m^2/s] (tunable)
    kappa: float = 1.0       # electrolyte conductivity [S/m] (tunable)
    t_plus: float = 0.363    # transference number [-]
    # solid
    sigma_p: float = 100.0   # positive solid conductivity [S/m]
    sigma_n: float = 100.0   # negative solid conductivity [S/m]
    D_s_p: float = 1.0e-14   # solid diffusion [m^2/s]
    D_s_n: float = 1.0e-14

@dataclass
class Kinetics:
    k_p: float = 1.0e-11     # reaction rate constant (m^2.5 / (mol^0.5 s))
    k_n: float = 1.0e-12
    alpha: float = 0.5

@dataclass
class Thermo:
    F: float = 96487.0
    R: float = 8.3145
    T: float = 298.15

@dataclass
class ElectrodeProps:
    c_s_max_p: float = 51765.0  # [mol/m^3]
    c_s_max_n: float = 29583.0  # [mol/m^3]
    c_e0: float = 1000.0        # nominal electrolyte concentration [mol/m^3]

@dataclass
class Discretization:
    N_x: int = 30  # number of electrolyte grid points (total across p|s|n)
