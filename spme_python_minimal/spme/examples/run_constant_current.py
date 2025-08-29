import numpy as np
import matplotlib.pyplot as plt
from spme.model import SPMe, SPMeParams
from spme.simulator import simulate_constant_current

# 1C discharge for a 5 Ah cell
I_app = 5.0  # A
t_end = 3600 # s

spme = SPMe(SPMeParams())
res = simulate_constant_current(I_app, t_end, spme=spme)

plt.figure()
plt.plot(res.t/60, res.V)
plt.xlabel('Time [min]')
plt.ylabel('Voltage [V]')
plt.title('SPMe constant-current discharge')
plt.show()
