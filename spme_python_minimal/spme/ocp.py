import numpy as np

def U_p(theta):
    # LG INR21700 (NMC) positive OCP fit
    # theta in [0,1]
    return (-0.8090*theta + 4.4875
            - 0.0428*np.tanh(18.5138*(theta-0.5542))
            - 17.7326*np.tanh(15.7890*(theta-0.3117))
            + 17.5842*np.tanh(15.9308*(theta-0.3120)))

def U_n(theta):
    # LG INR21700 (graphite) negative OCP fit
    return (1.9793*np.exp(-39.3631*theta) + 0.2482
            - 0.0909*np.tanh(29.8538*(theta-0.1234))
            - 0.04478*np.tanh(14.9159*(theta-0.2769))
            - 0.0205*np.tanh(30.4444*(theta-0.6103)))
