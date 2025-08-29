# SPMe (Single Particle Model with electrolyte) — Minimal Python

This is a teaching/reference implementation of a clean SPMe in Python. It includes:
- Parabolic-profile solid diffusion ODEs in each electrode
- 1D electrolyte diffusion with source terms in electrodes
- Butler–Volmer overpotentials and empirical OCPs (LG INR 21700 fits)
- Simple algebraic voltage expression with ohmic and concentration terms
- Scipy `solve_ivp` (BDF) time integration

See `examples/run_constant_current.py` to try a 1C discharge.
