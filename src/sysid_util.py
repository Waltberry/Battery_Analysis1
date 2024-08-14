import control as ct
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt


def theta_2_BCDF(theta, n):
    """
    Converts the parameter vector theta into coefficient arrays B, C, D, and F 
    for the Box-Jenkins model.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nc, nd, nf, nk].

    Returns:
    -------
    B : ndarray
        Coefficients for the B polynomial.
    C : ndarray
        Coefficients for the C polynomial.
    D : ndarray
        Coefficients for the D polynomial.
    F : ndarray
        Coefficients for the F polynomial.
    """
    nb, nc, nd, nf, nk = n

    # Extracting coefficients from theta
    theta_b = theta[0:nb]
    theta_c = np.concatenate(([1], theta[nb:nb + nc]))
    theta_d = np.concatenate(([1], theta[nb + nc:nb + nc + nd]))
    theta_f = np.concatenate(([1], theta[nb + nc + nd:nb + nc + nd + nf]))

    # Ensuring dimensions of B and F are consistent with nf
    if nf + 1 > nb:
        B = np.concatenate((theta_b, np.zeros(nf + 1 - nb)))
    elif nf + 1 == nb:
        B = theta_b
    else:
        raise ValueError('Must choose proper transfer function for plant model.')

    # Adding delay (nk) to F if nk > 0
    if nk > 0:
        F = np.concatenate((theta_f, np.zeros(nk)))
    else:
        F = theta_f 

    # Ensuring dimensions of C and D are consistent with nd
    if nd > nc:
        C = np.concatenate((theta_c, np.zeros(nd - nc)))
    elif nc == nd:
        C = theta_c
    else:
        raise ValueError('Must choose proper transfer function for noise model.')

    D = theta_d

    return B, C, D, F


def theta_2_tf_box_jenkins(theta, n, Ts):
    """
    Converts the parameter vector theta into transfer functions for the Box-Jenkins model.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nc, nd, nf, nk].
    Ts : float
        The sampling time.

    Returns:
    -------
    G_theta : TransferFunction
        Transfer function for the plant model.
    H_theta : TransferFunction
        Transfer function for the noise model.
    """
    B, C, D, F = theta_2_BCDF(theta, n)
    G_theta = ct.tf(B, F, Ts)
    H_theta = ct.tf(C, D, Ts)

    return G_theta, H_theta


def jac_V_bj(theta, n, y, u):
    """
    Computes the Jacobian of the cost function with respect to the Box-Jenkins model parameters.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nc, nd, nf, nk].
    y : ndarray
        The output data.
    u : ndarray
        The input data.

    Returns:
    -------
    depsilonTot : ndarray
        The Jacobian matrix.
    """
    N = y.shape[0]
    nb, nc, nd, nf, nk = n

    B, C, D, F = theta_2_BCDF(theta, n)

    G_theta = ct.tf(B, F, True)
    H_theta = ct.tf(C, D, True)
    
    # Compute y_hat (predicted output) using the Box-Jenkins model
    tt, y_hat_1 = ct.forced_response(G_theta/H_theta, U=u) 
    tt, y_hat_2 = ct.forced_response(1 - 1/H_theta, U=y)
    y_hat = y_hat_1 + y_hat_2
    epsilon = y - y_hat # Prediction error

    tt, y_hat_3 = ct.forced_response(G_theta, U=u) 
    e = y - y_hat_3
    
    # Calculate partial derivatives of epsilon with respect to B, C, D, and F
    depsilondB = np.empty((N,nb))
    for ii in range(nb):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nf))),F, True)
        #print(-d*P/H_theta)
        tt, depsilon = ct.forced_response(-d*P/H_theta,U=u)
        depsilondB[:,ii] = depsilon
        #dVdB[ii] = 2*(np.sum(epsilon * depsilon))

    depsilondC = np.empty((N,nc))
    for ii in range(nc):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii+1))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nc))),C, True)
        tt, depsilon = ct.forced_response(-d*P/H_theta,U=e)
        depsilondC[:,ii] = depsilon
        #dVdC[ii] = 2*(np.sum(epsilon * depsilon))
   
    depsilondD = np.empty((N,nd))
    for ii in range(nd):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii+1))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nc))),C, True)
        tt, depsilon = ct.forced_response(d*P,U=e)
        depsilondD[:,ii] = depsilon
        #dVdD[ii] = 2*(np.sum(epsilon * depsilon))
    
    depsilondF = np.empty((N,nf))
    for ii in range(nf):
        d = ct.tf(1,np.concatenate(([1],np.zeros(ii+1))), True)
        P = ct.tf(np.concatenate(([1],np.zeros(nf+nk))),F, True)
        tt, depsilon = ct.forced_response(d*P*G_theta/H_theta,U=u)
        depsilondF[:,ii] = depsilon
        #dVdF[ii] = 2*(np.sum(epsilon * depsilon))
        
    # Combine all partial derivatives   
    depsilonTot = np.concatenate((depsilondB, depsilondC, depsilondD, depsilondF),axis=1)
    return depsilonTot


def V_box_jenkins(theta, n, y, u):
    """
    Computes the prediction error for the Box-Jenkins model.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nc, nd, nf, nk].
    y : ndarray
        The output data.
    u : ndarray
        The input data.

    Returns:
    -------
    epsilon : ndarray
        The prediction error.
    """
    N = y.shape[0]
    y_hat = y_hat_box_jenkins(theta,n,y,u)
    epsilon = y - y_hat
    
    #return np.sum(epsilon**2)/N
    return epsilon


def y_hat_box_jenkins(theta, n, y, u):
    """
    Computes the predicted output y_hat for the Box-Jenkins model.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nc, nd, nf, nk].
    y : ndarray
        The output data.
    u : ndarray
        The input data.

    Returns:
    -------
    y_hat : ndarray
        The predicted output.
    """
    B,C,D,F = theta_2_BCDF(theta,n)
    G_theta = ct.tf(B, F, True)
    H_theta = ct.tf(C, D, True)    
    tt, y_hat_1 = ct.forced_response(G_theta/H_theta, U=u) 
    tt, y_hat_2 = ct.forced_response(1 - 1/H_theta, U=y)
    y_hat = y_hat_1 + y_hat_2
    
    return y_hat


def V_oe(theta, n, y, u):
    """
    Computes the cost function for the Output Error (OE) model.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nf].
    y : ndarray
        The output data.
    u : ndarray
        The input data.

    Returns:
    -------
    cost : float
        The cost function value.
    """
    theta_b = theta[0:n[0]]
    theta_f = np.concatenate(([1],theta[n[0]:n[0]+n[1]]))

    G_theta = ct.tf(theta_b, theta_f, True)
    tt, y_hat = ct.forced_response(G_theta, U=u) 
   
    epsilon = y - y_hat
    return np.sum(epsilon**2)


def theta_2_tf_oe(theta,n, Ts):
    """
    Converts the parameter vector theta into transfer functions for the Output Error (OE) model.

    Parameters:
    ----------
    theta : ndarray
        The parameter vector.
    n : list
        The list of model orders [nb, nf].
    Ts : float
        The sampling time.

    Returns:
    -------
    G_theta : TransferFunction
        Transfer function for the plant model.
    H_theta : TransferFunction
        Transfer function for the noise model (identity in OE model).
    """
    theta_b = theta[0:n[0]]
    theta_f = np.concatenate(([1],theta[n[0]:n[0]+n[1]]))

    G_theta = ct.tf(theta_b, theta_f, Ts)
    H_theta = ct.tf(1,1,Ts)

    return G_theta, H_theta


def y_hat_oe(theta, n, y, u):
    """
    Calculate the prediction error for an Output Error (OE) model.

    Parameters:
    ----------
    theta : np.ndarray
        Parameters of the OE model, consisting of numerator and denominator coefficients.
    n : tuple
        Model orders (n_b, n_f), where:
        n_b - order of the numerator polynomial.
        n_f - order of the denominator polynomial.
    y : np.ndarray
        Observed output data.
    u : np.ndarray
        Input data used to generate the output.

    Returns:
    -------
    epsilon : np.ndarray
        Prediction error between the observed and predicted output.
    """
    # Extracting numerator (B) and denominator (F) coefficients from theta
    theta_b = theta[0:n[0]]
    theta_f = np.concatenate(([1],theta[n[0]:n[0]+n[1]])) # Include leading 1 in denominator
    
    # Create transfer function model G_theta based on the theta coefficients
    G_theta = ct.tf(theta_b, theta_f, True)
    
    # Generate model output (y_hat) from input data (u)
    tt, y_hat = ct.forced_response(G_theta, U=u)
    
    # Calculate prediction error
    epsilon = y - y_hat

    return epsilon


def V_arx_lin_reg(n, y, u):
    """
    Perform linear regression to estimate ARX model parameters.

    Parameters:
    ----------
    n : tuple
        Model orders (n_a, n_b, n_k), where:
        n_a - order of the AR part.
        n_b - order of the X (input) part.
        n_k - input-output delay.
    y : np.ndarray
        Output data.
    u : np.ndarray
        Input data.

    Returns:
    -------
    theta : np.ndarray
        Estimated parameters of the ARX model.
    """
    
    #d = n[0]-n[1]+1
    d = n[2] + 1 # Input-output delay plus one
    t0 = np.sum(n)+d
    N = y.shape[0]
    
    # Constructing the regressor matrix (phi)
    phi = np.zeros((N-t0,n[0]+n[1]))
    for ii in range(N-t0):
        for jj in range(n[0]):
            phi[ii,jj] = -y[ii+t0-jj-1]
    
    for ii in range(N-t0):
        for jj in range(n[1]):
            phi[ii,jj+n[0]] = u[ii+t0-jj-d]
            
    # Solving for theta using the normal equation (least squares solution)
    theta = np.linalg.inv( phi.T @ phi ) @ (phi.T @ y[t0:N])
    
    return theta

def theta_2_tf_arx(theta,n,Ts):
    """
    Convert ARX model parameters to transfer functions.

    Parameters:
    ----------
    theta : np.ndarray
        ARX model parameters.
    n : tuple
        Model orders (n_a, n_b, n_k), where:
        n_a - order of the AR part.
        n_b - order of the X (input) part.
        n_k - input-output delay.
    Ts : float
        Sampling time.

    Returns:
    -------
    G_theta : control.TransferFunction
        Transfer function of the system (G).
    H_theta : control.TransferFunction
        Transfer function of the noise model (H).
    """
    na = n[0]
    nb = n[1]
    nk = n[2]
    
    # Constructing the numerator (B) and denominator (A) polynomials
    theta_a = np.concatenate(([1],theta[0:n[0]]))
    theta_b = theta[n[0]:n[0]+n[1]]
    
    # Adjusting the length of B and A if necessary
    if na+1 > nb:
        B = np.concatenate((theta_b,np.zeros(na+1-nb)))
    elif na+1==nb:
        B = theta_b
    else:
        print('Must choose proper transfer function for plant model.')
    
    
    # Account for delay by shifting the A polynomial
    if nk > 0:
        A = np.concatenate((theta_a,np.zeros(nk)))
    else:
        A = theta_a 
        
    
    # Noise model (assumed to be white noise)    
    C = np.zeros(na+1)
    C[0] = 1
    
    # Creating transfer functions for the system (G) and noise model (H)
    G_theta = ct.tf(B, A, Ts)
    H_theta = ct.tf(C, theta_a, Ts)
    return G_theta, H_theta



def cross_correlation_test(epsilon, u, tau=50):
    """
    Perform cross-correlation test between prediction error and input signal.

    Parameters:
    ----------
    epsilon : np.ndarray
        Prediction error.
    u : np.ndarray
        Input data.
    tau : int, optional
        Maximum lag to compute the cross-correlation, default is 50.

    Returns:
    -------
    None
    """
    N = u.shape[0]
    
    # Compute cross-correlation of epsilon with input u
    Reu = np.correlate(epsilon, u, 'full')
    Reu = Reu[N-tau:N+tau]  # Extract the relevant part of the cross-correlation
    
    # Compute bounds for significance testing
    Re = np.correlate(epsilon, epsilon, 'full')
    Ru = np.correlate(u, u, 'full')
    P = np.sum(Re * Ru)
    bound = np.sqrt(P / N) * 1.95  # 95% confidence bounds

    # Plotting the cross-correlation
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(-tau, tau), Reu)
    ax.plot(np.arange(-tau, tau), np.ones(2*tau) * bound, 'k:')
    ax.plot(np.arange(-tau, tau), -np.ones(2*tau) * bound, 'k:')
    ax.set_title('Cross-Correlation of Prediction Error')
    ax.set_xlabel('Lag (samples)')


def auto_correlation_test(epsilon, tau=50):
    """
    Perform auto-correlation test on the prediction error.

    Parameters:
    ----------
    epsilon : np.ndarray
        Prediction error.
    tau : int, optional
        Maximum lag to compute the auto-correlation, default is 50.

    Returns:
    -------
    None
    """
    N = epsilon.shape[0]
    
    # Compute auto-correlation of epsilon
    Re = np.correlate(epsilon, epsilon, 'full')
    Re_pos = Re[N:N+tau]  # Extract the positive lags

    # Bound for significance testing (95% confidence)
    bound_e = 1.95 / np.sqrt(N)

    # Plotting the auto-correlation
    fig, ax = plt.subplots(1)
    ax.plot(np.arange(1, tau + 1), Re_pos / Re[N - 1])
    ax.plot(np.arange(1, tau + 1), np.ones(tau) * bound_e, 'k:')
    ax.plot(np.arange(1, tau + 1), -np.ones(tau) * bound_e, 'k:')
    ax.set_title('Auto-Correlation of Prediction Error')
    ax.set_xlabel('Lag (samples)')
