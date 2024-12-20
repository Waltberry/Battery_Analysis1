import numpy as np
from scipy.stats import linregress
import scipy as sp
import src.sysid_util as sid 
import control as ct
import numpy as np
import matplotlib.pyplot as plt

def estimate_parameters(yy, uu, nf, nb, na, nc=0, nd=0, nk=0):
    """
    Estimate model parameters using ARX and Box-Jenkins models.

    This function processes input and output data arrays (yy and uu) to estimate model parameters.
    The model parameters for ARX and Box-Jenkins are calculated and stored for each cycle.

    Parameters:
    yy (list of np.array): List of output data arrays (each array corresponds to a cycle).
    uu (list of np.array): List of input data arrays (each array corresponds to a cycle).
    nf (int): Number of poles for the ARX model (default=2).
    nb (int): Number of zeros for the ARX model (default=2).
    nc (int): Number of poles for the noise model in Box-Jenkins (default=0).
    nd (int): Number of zeros for the noise model in Box-Jenkins (default=0).
    na (int): Number of poles for the AR model in Box-Jenkins (default=2).
    nk (int): Input delay for the ARX and Box-Jenkins models (default=0).

    Returns:
    tuple: Containing three lists:
        - theta_arx_list: List of ARX model parameters for each cycle.
        - theta_bj_list: List of Box-Jenkins model parameters for each cycle.
        - optimization_results_list: List of optimization results for each cycle.
    """
    
    # Initialize lists to store results
    theta_arx_list = []
    theta_bj_list = []
    optimization_results_list = []

    # Iterate through all arrays in yy and uu, skipping the first cycle
    for i in range(1, len(yy)):
        uu[i] = uu[i] - uu[i][0] * np.ones(len(uu[i]))
        yy[i] = yy[i] - yy[i][0] * np.ones(len(yy[i]))

        # Calculate the ARX model parameters
        n_arx = [nf, nb, nk]
        theta_arx = sid.V_arx_lin_reg(n_arx, yy[i], uu[i])
        
        # Store theta_arx in the list
        theta_arx_list.append(theta_arx)
        
        # Prepare initial guess for Box-Jenkins model
        theta_box_jenkins = np.concatenate((
            theta_arx[n_arx[0]:np.sum(n_arx)], 
            np.zeros(nc + nd), 
            theta_arx[0:n_arx[0]]
        ))

        # Define the structure for the Box-Jenkins model
        n_bj = [nb, nc, nd, nf, nk]
        
        # Perform optimization for Box-Jenkins model parameters
        optimization_results = sp.optimize.least_squares(
            sid.V_box_jenkins, 
            theta_box_jenkins, 
            jac=sid.jac_V_bj, 
            args=(n_bj, yy[i], uu[i])
        )
        
        # Store the optimization results in the list
        optimization_results_list.append(optimization_results)

    return theta_arx_list, theta_bj_list, optimization_results_list

def process_optimization_results(optimization_results_list):
    """
    Process the optimization results to extract x values and cost values.

    This function iterates through the provided optimization results, extracting
    the optimized parameters (x) and the associated cost for each result. It 
    prints each set of values and also returns lists of all x values and cost values.

    Parameters:
    optimization_results_list (list): A list of optimization result objects, 
                                      each containing .x and .cost attributes.

    Returns:
    tuple: Two lists containing the x values and the cost values from the optimization results.
    """
    x_values = []
    cost_values = []

    for result in optimization_results_list:
        x_values.append(result.x)
        cost_values.append(result.cost)

        # Print x and cost for each optimization result
        print("x:", result.x)
        print("cost:", result.cost)
        print()

    return x_values, cost_values

# Assuming optimization_results_list is already defined from the previous function
# x_values, cost_values = process_optimization_results(optimization_results_list)


def analyze_cost_values(cost_values, threshold=2):
    """
    Analyze the cost values to detect outliers and plot the cost over cycles.

    This function calculates the mean and standard deviation of the cost values,
    computes Z-scores to identify outliers, and plots the cost values over cycles.
    Outliers are highlighted in the plot.

    Parameters:
    cost_values (list): A list of cost values from the optimization results.
    threshold (float, optional): The Z-score threshold to identify outliers. Default is 2.

    Returns:
    list: A list of tuples containing the cycle number, cost value, and Z-score for each detected outlier.
    """
    # Calculate mean and standard deviation of the cost values
    mean_cost = np.mean(cost_values)
    std_cost = np.std(cost_values)

    # Calculate Z-scores
    z_scores = [(cost - mean_cost) / std_cost for cost in cost_values]

    # Identify outliers based on the Z-score threshold
    outliers = [(i + 1, cost_values[i], z_scores[i]) for i in range(len(z_scores)) if abs(z_scores[i]) > threshold]

    # Print detected outliers
    for cycle, cost, z_score in outliers:
        print(f"Outlier detected - Cycle: {cycle}, Cost: {cost:.2f}, Z-Score: {z_score:.2f}")

    # Plot cost over cycles
    plt.plot(range(1, len(cost_values) + 1), cost_values, marker='o', label='Cost')

    # Highlight outliers in the plot
    for cycle, cost, _ in outliers:
        plt.scatter(cycle, cost, color='red', label='Outlier' if cycle == outliers[0][0] else "")
    
    plt.xlabel('Cycle Number')
    plt.ylabel('Cost')
    plt.title('Cost over Cycles')
    plt.legend()
    plt.grid(True)
    plt.show()

    return outliers

# Example usage
# Assuming cost_values is already defined
# outliers = analyze_cost_values(cost_values)

def analyze_cycles(optimization_results_list, uu, yy, Ts=0.005, nb=2, nc=0, nd=0, nf=2, nk=0):
    """
    Analyze and simulate the system's response for each cycle using the optimized parameters.
    
    This function extracts the transfer functions for each cycle, simulates the system's response,
    and generates plots for measured vs. simulated outputs and input on two y-axes. It also prints
    the transfer function, zeros, and poles for each cycle and generates a Bode plot.
    
    Parameters:
    optimization_results_list (list): A list of optimization results containing optimized parameters.
    uu (list): A list of input (control/mA) arrays.
    yy (list): A list of output (voltage) arrays.
    Ts (float, optional): Sampling time. Default is 0.005.
    nb (int, optional): Number of numerator parameters for the Box-Jenkins model. Default is 2.
    nc (int, optional): Number of noise model parameters. Default is 0.
    nd (int, optional): Number of delay model parameters. Default is 0.
    nf (int, optional): Number of feedback model parameters. Default is 2.
    nk (int, optional): Number of time delays. Default is 0.
    """
    # Define the structure for the Box-Jenkins model
    n_bj = [nb, nc, nd, nf, nk]

    # Loop through the arrays in yy and uu, skipping the first one
    for i in range(1, len(yy)):
        # Extract the optimized parameters for the current cycle
        G_hat_box_jenkins, H_hat_box_jenkins = sid.theta_2_tf_box_jenkins(
            optimization_results_list[i-1].x, n_bj, Ts
        )
        
        # Print the transfer function
        print(f"Cycle {i}: G_hat_box_jenkins")
        print(G_hat_box_jenkins)
        
        # Print zeros and poles
        print(f"Cycle {i}: Zeros")
        print(G_hat_box_jenkins.zeros())
        print(f"Cycle {i}: Poles")
        print(G_hat_box_jenkins.poles())
        
        # Simulate the system's response using the identified transfer function
        tt, y_sim = ct.forced_response(G_hat_box_jenkins, U=uu[i])
        
        # Plot measured vs. simulated output and input on two y-axes
        fig, ax1 = plt.subplots()

        # Plot measured and simulated output on the left y-axis
        ax1.plot(yy[i], 'b-', label='Measured')
        ax1.plot(y_sim, 'g-', label='Simulated')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc='center right')  # Legend location: middle right

        # Create a second y-axis for the input (control/mA)
        ax2 = ax1.twinx()
        ax2.plot(uu[i], 'r-', label='Input (control/mA)')
        ax2.set_ylabel('Control/mA', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.legend(loc='lower right')  # Legend location: lower right

        # Add title
        plt.title(f'Measured vs. Simulated Outputs and Input for Cycle {i}')
        
        # Adjust layout to avoid overlapping labels
        fig.tight_layout()

        # Show the plot
        plt.show()

        # Bode plot for the identified transfer function
        ct.bode_plot(G_hat_box_jenkins)


def arx_model(u, y, order):
    """
    Estimate the parameters of an ARX (AutoRegressive with eXogenous inputs) model.

    Mathematical Expression:
    The ARX model is given by:
    
        y(t) = a1*y(t-1) + a2*y(t-2) + ... + an*y(t-n) + b1*u(t-1) + b2*u(t-2) + ... + bn*u(t-n) + e(t)
    
    In matrix form:
    
        Y = Φθ + e
    
    where:
        - Y is the vector of observed output values.
        - Φ (phi) is the matrix of past input and output values.
        - θ is the vector of parameters [b1, b2, ..., bn, -a1, -a2, ..., -an].
        - e is the error vector.

    The Φ (phi) matrix is constructed as follows:
    
        Φ = | u[n-1]  u[n-2]  ...  u[n-order]  -y[n-1]  -y[n-2]  ...  -y[n-order] |
            | u[n]    u[n-1]  ...  u[n-order+1] -y[n]    -y[n-1] ...  -y[n-order+1]|
            | ...    ...      ...  ...          ...      ...     ...  ...          |
            | u[N-1]  u[N-2]  ...  u[N-order]   -y[N-1]  -y[N-2] ...  -y[N-order]  |

    The parameter vector θ is estimated using the normal equation:
    
        θ = (Φ.T @ Φ)^(-1) @ Φ.T @ Y
    theta = (phi.T * phi)^(-1) * phi.T * Y

    Parameters:
        u (numpy array): The input signal (exogenous input).
        y (numpy array): The output signal.
        order (int): The order of the ARX model.
    
    Returns:
        theta (numpy array): The estimated parameters of the ARX model.
        phi (numpy array): The matrix of past input and output values.
        Y (numpy array): The vector of observed output values starting from the given order.
    """    
    
    n_samples = len(u)
    
    # Construct the phi matrix
    phi = np.zeros((n_samples - order, 2 * order))
    for i in range(order, n_samples):
        u_slice = np.flip(u[i-order+1:i+1])
        y_slice = np.flip(-y[i-order:i]) 
        
        phi[i - order, :order] = u_slice
        phi[i - order, order:2*order] = y_slice
    
    # Construct the Y vector
    Y = y[order:]
    
    # Calculate theta using the normal equation
    theta = np.linalg.inv(phi.T @ phi) @ phi.T @ Y
    
    return theta, phi, Y



def generalized_exponential_model(t, *params):
    """
    Generalized Exponential Model:
    
    Parameters:
    - t: Independent variable (time or another independent parameter)
    - params: Parameters of the model, formatted as [c1, c2, b2, c3, b3, ..., cn, bn]
             where ci and bi are coefficients for each exponential term
    
    Formula:
    y(t) = c1 + sum(ci * exp(-bi * t) for i in range(n_terms))
    
    Explanation:
    - c1 is the initial value or constant offset.
    - ci are coefficients for each exponential term.
    - bi are decay rates for each exponential term.
    - n_terms is the number of exponential terms, calculated as (len(params) - 1) // 2.
    
    Args:
    - t (float or array-like): Input variable (time or other independent variable).
    - *params (float): Parameters of the model, should be in the format described.
    
    Returns:
    - float or array-like: Value(s) of the model at given t.
    """
    n_terms = (len(params) - 1) // 2
    c1 = params[0]
    result = c1
    for i in range(n_terms):
        ci = params[1 + 2 * i]
        bi = params[2 + 2 * i]
        result += ci * np.exp(-bi * t)
    return result


def complexpoles_exponential_model(t, *params):
    """
    Generalized Exponential Model including complex poles:
    
    Parameters:
    - t: Independent variable (time or another independent parameter)
    - params: Parameters of the model, formatted as [c0, a1, b1, alpha1, beta1, a2, b2, alpha2, beta2, ...]
              where a, b, alpha, and beta are coefficients for each complex exponential term.
    
    Formula:
    y(t) = c0 + sum(2a * exp(-alpha * t) * cos(beta * t) + 2b * exp(-alpha * t) * sin(beta * t) for i in range(n_terms))
    
    Explanation:
    - c0 is the initial value or constant offset.
    - a is the amplitude coefficient for the cosine term.
    - b is the amplitude coefficient for the sine term.
    - alpha is the real part of the pole (decay rate).
    - beta is the imaginary part of the pole (oscillation frequency).
    - n_terms is the number of complex exponential terms, calculated as (len(params) - 1) // 4.
    
    Args:
    - t (float or array-like): Input variable (time or other independent variable).
    - *params (float): Parameters of the model, should be in the format described.
    
    Returns:
    - float or array-like: Value(s) of the model at given t.
    """
    n_terms = (len(params) - 1) // 4
    c0 = params[0]
    result = c0
    for i in range(n_terms):
        a = params[1 + 4 * i]
        b = params[2 + 4 * i]
        alpha = params[3 + 4 * i]
        beta = params[4 + 4 * i]
        result += 2 * a * np.exp(-alpha * t) * np.cos(beta * t)
        result += 2 * b * np.exp(-alpha * t) * np.sin(beta * t)
    return result


def fit_straight_line(x, y):
    """
    Fit a straight line to the data using linear regression.

    Parameters:
    x : numpy array
        Array of x values.
    y : numpy array
        Array of y values.

    Returns:
    tuple
        Tuple containing:
        - slope: Slope of the fitted line.
        - intercept: Intercept of the fitted line.
    """
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return slope, intercept
