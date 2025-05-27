import control as ct
import numpy as np
import scipy as sp  
import matplotlib.pyplot as plt


def validate_inputs(n, u, y, model_type="ARX"):
    """
    Validates the inputs for common conditions, dynamically adjusting checks 
    based on the model type.

    Parameters:
    ----------
    n : list or tuple
        Model orders. Structure depends on the model type:
        - ARX: (n_a, n_b, n_k)
        - Box-Jenkins: (n_b, n_c, n_d, n_f, n_k).
    u : ndarray
        Input data.
    y : ndarray
        Output data.
    model_type : str, optional
        Type of the model. Options are "ARX" (default) or "Box-Jenkins".

    Raises:
    -------
    ValueError
        If any of the conditions are violated.
    """
    # Check if n is a list/tuple of integers
    if not isinstance(n, (list, tuple)) or not all(isinstance(x, int) for x in n):
        raise ValueError("n must be a list or tuple of integers.")
    
    # Check if u and y have the same size
    if len(u) != len(y):
        raise ValueError("u and y must have the same size.")
    
    # Validate model-specific conditions
    if model_type == "ARX":
        if len(n) < 3:
            raise ValueError("ARX model requires (n_a, n_b, n_k).")
        na, nb, nk = n
        nb -= 1
        if nb > na:
            raise ValueError("In ARX, nb must not be greater than na.")

    elif model_type == "Box-Jenkins":
        # nb, nc, nd, nf, nk = n
        pass

    else:
        raise ValueError(f"Unsupported model type: {model_type}.")



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
    
    validate_inputs(n, np.array([]), np.array([]),"Box-Jenkins")
    nb, nc, nd, nf, nk = n
    
    # The following code above is equivalent to the commented code below
    # nb = n[0]
    # nc = n[1]
    # nd = n[2]
    # nf = n[3]
    # nk = n[4]

    # Extracting coefficients from theta
    theta_b = theta[0:nb]
    theta_c = np.concatenate(([1], theta[nb:nb + nc]))
    theta_d = np.concatenate(([1], theta[nb + nc:nb + nc + nd]))
    theta_f = np.concatenate(([1], theta[nb + nc + nd:nb + nc + nd + nf]))

    # Ensuring dimensions of B and F are consistent with nf
    if nf + 1 > nb:
        B = np.concatenate((theta_b, np.zeros(nf + 1 - nb)))
        F = theta_f
    elif nf + 1 == nb:
        B = theta_b
        F = theta_f
    else:
        B = theta_b
        F = np.concatenate((theta_f, np.zeros(nb-nf-1)))
        #raise ValueError('Must choose proper transfer function for plant model.')

    # Adding delay (nk) to F if nk > 0
    if nk > 0:
        F = np.concatenate((F, np.zeros(nk)))

    # Ensuring dimensions of C and D are consistent with nd
    if nd > nc:
        C = np.concatenate((theta_c, np.zeros(nd - nc)))
        D = theta_d
    elif nc == nd:
        C = theta_c
        D = theta_d
    else:
        C = theta_c
        D = np.concatenate(theta_d, np.zeros(nc-nd))
        #raise ValueError('Must choose proper transfer function for noise model.')

    return B, C, D, F


def theta_2_tf_box_jenkins(theta,n,Ts):
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
    validate_inputs(n, np.array([]), np.array([]),"Box-Jenkins")
    B,C,D,F = theta_2_BCDF(theta,n)
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
    validate_inputs(n, u,y,"Box-Jenkins")
    N = y.shape[0]
    nb, nc, nd, nf, nk = n
    
    # The following code above is equivalent to the commented code below
    # nb = n[0]
    # nc = n[1]
    # nd = n[2]
    # nf = n[3]
    # nk = n[4]

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
    validate_inputs(n, u,y,"Box-Jenkins")
    N = y.shape[0]
    y_hat = y_hat_box_jenkins(theta, n, y, u)
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
    validate_inputs(n, u, y,"Box-Jenkins")
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
    validate_inputs(n, u, y,"Box-Jenkins")
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
    validate_inputs(n, np.array([]), np.array([]), "Box-Jenkins")
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
    validate_inputs(n, u, y, "Box-Jenkins")
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
    validate_inputs(n, u, y, "ARX")

    # Extract orders and delay from the input tuple `n`
    na, nb, nk = n
    
    # The following code above is equivalent to the commented code below
    # na = n[0]
    # nb = n[1]
    # nk = n[2]

    t0 = np.maximum(na - 1, nb + nk - 1)
    N = y.shape[0]  

    if (t0 >= N) or (na+nb >= N):
        raise Exception('Number of parameters is too large for given data length')
    
    # Constructing the regressor matrix (phi)
    phi = np.zeros((N - t0, na + nb))
    
    for ii in range(N - t0):
        for jj in range(na):
            phi[ii, jj] = -y[ii + t0 - jj - 1]
            
    for ii in range(N - t0):
        for jj in range(nb):
            phi[ii, jj + na] = u[ii + t0 - jj - nk]

    # Solving for theta using the normal equation (least squares solution)
    theta = np.linalg.inv(phi.T @ phi) @ (phi.T @ y[t0:N])

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
    validate_inputs(n, np.array([]), np.array([]), "ARX")
    na, nb, nk = n
    # The following code above is equivalent to the commented code below
    # na = n[0]
    # nb = n[1]
    # nk = n[2]
    
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



def cross_correlation_test(epsilon,u, tau=50):
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
    Reu = np.correlate(epsilon,u,'full')
    
    Reu = Reu[N-tau:N+tau] # Extract the relevant part of the cross-correlation
    
    # Compute bounds for significance testing
    Re = np.correlate(epsilon,epsilon,'full')
    Ru = np.correlate(u,u,'full')
    P = np.sum(Re*Ru)
    bound = np.sqrt(P/N)*1.95 # 95% confidence bounds
    
    # Plotting the cross-correlation
    fig,ax = plt.subplots(1)
    ax.plot(np.arange(-tau,tau),Reu)
    ax.plot(np.arange(-tau,tau),np.ones(2*tau)*bound,'k:')
    ax.plot(np.arange(-tau,tau),-np.ones(2*tau)*bound,'k:')
    ax.set_title('Cross Correlation of Prediction Error')
    ax.set_xlabel('Lag (samples)')


    # Re = np.correlate(epsilon,epsilon,'full')


def auto_correlation_test(epsilon,tau = 50):
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
    Re = np.correlate(epsilon,epsilon,'full')
    Re_pos = Re[N:N+tau]
    
    # Bound for significance testing (95% confidence)
    bound_e = 1.95/np.sqrt(N)
    
    # Plotting the auto-correlation
    fig,ax = plt.subplots(1)
    ax.plot(np.arange(1,tau+1),Re_pos/Re[N-1])
    ax.plot(np.arange(1,tau+1),np.ones(tau)*bound_e,'k:')
    ax.plot(np.arange(1,tau+1),-np.ones(tau)*bound_e,'k:')
    ax.set_title('Auto Correlation of Prediction Error')
    ax.set_xlabel('Lag (samples)')


def FIR_estimates_GH(n, y, u):
    
    validate_inputs(n, u, y, "Box-Jenkins") # The Box-Jenkins is a placeholder for the FIR model

    na, nb, nk = n
    ng = nb+1
    nh = na+1

    theta = V_arx_lin_reg(n,y,u)
    
    # Separate estimated parameters for A and B polynomials
    A = -theta[0:na]
    B = theta[na:nb+na]

    rB = np.zeros(nh)
    cB = np.concatenate(([0],B,np.zeros(ng-nb-1)))

    rA = np.concatenate(([1], np.zeros(nh-1)))
    cA = np.concatenate(([1], -A, np.zeros(nh-na-1)))

    CB = sp.linalg.toeplitz(cB,r=rB)
    CA = sp.linalg.toeplitz(cA,r=rA)

    M = np.block([[np.eye(ng), -CB],[np.zeros((nh,ng)), CA]])
    V = np.concatenate((B,np.zeros(ng-nb),A,np.zeros(nh-na)))
    
    theta_gh = np.linalg.inv( M.T @ M ) @ (M.T @ V)

    g = np.concatenate((np.zeros(nk), theta_gh[0:ng-1]))
    h = np.concatenate(([1], theta_gh[ng:ng+nh-1]))

    return g, h


def tf_realization_GH(g,h,n):
    
    nb = n[0]
    nc = n[1]
    nd = n[2]
    na = n[3]
    nk = n[4]

    nh = h.shape[0]-1
    ng = g.shape[0]-nk
    
    # Create Toeplitz matrix for G transfer function realization
    Cg = np.array(sp.linalg.toeplitz(np.concatenate(([0],g[nk:nk+ng-1])),r=np.zeros(na)))
    Meye = np.concatenate((np.eye(nb), np.zeros((ng-nb,nb))),axis=0)
    M = np.concatenate((Meye,-Cg),axis=1)
    thetaBA = np.linalg.inv( M.T @ M ) @ (M.T @ g[nk:ng+nk] )
    
    # Create Toeplitz matrix for H transfer function realization
    if nc==0 and nd==0:
        thetaCD = []
    elif nc>0 and nd>0:
        Ch = np.array(sp.linalg.toeplitz(h[0:nh],r=np.concatenate(([1],np.zeros(nd-1)))))
        Meye = np.concatenate((np.eye(nc), np.zeros((nh-nc,nc))),axis=0)
        M = np.concatenate((Meye,-Ch),axis=1)
        thetaCD = np.linalg.inv( M.T @ M ) @ (M.T @ h[1:nh+1] )

    theta = np.concatenate((thetaBA[0:nb], thetaCD, thetaBA[nb:nb+na]))
    return theta


def get_initial_estimate_box_jenkins(n,n_high_order_approx, y,u):
    """
    Generate initial estimates for Box-Jenkins model parameters using high-order FIR approximation.

    Parameters:
    ----------
    n : tuple
        Model structure for Box-Jenkins as (nb, nc, nd, nf, nk).

    n_high_order_approx : tuple
        High-order approximation structure for ARX model as (na_ho, nb_ho).

    y : np.ndarray
        Output data for the system.

    u : np.ndarray
        Input data for the system.

    Returns:
    -------
    theta_init_bj : numpy.ndarray
        Initial parameter vector estimate for Box-Jenkins model.
    """
    validate_inputs(n, u, y, "Box-Jenkins") # The Box-Jenkins is a placeholder for the FIR model
    nb = n[0]
    nc = n[1]
    nd = n[2]
    nf = n[3]
    nk = n[4]

    na_ho = n_high_order_approx[0]
    nb_ho = n_high_order_approx[1]
    n_arx = [na_ho, nb_ho, nk] 

    g_imp_est, h_imp_est = FIR_estimates_GH(n_arx,y,u)

    theta_init_bj = tf_realization_GH(g_imp_est,h_imp_est,n)
    return theta_init_bj


def get_regression_matrix(w,t0,i1,i2):
    """
    Construct a regression matrix for linear regression using past values of data array `w`.

    Parameters:
    ----------
    w : np.ndarray
        Data array used to construct the regression matrix.

    t0 : int
        Starting index for data points to include in the matrix.

    i1 : int
        Starting index for regression term inclusion.

    i2 : int
        Ending index for regression term inclusion.

    Returns:
    -------
    phi : np.ndarray
        Regression matrix, where each row contains past values of `w` from index `t0`.
    """ 
    
    N = w.shape[0]
    phi = np.zeros((N-t0+i1,i2-i1))
    
    # Populate regression matrix with past values of `w`
    for ii in range(N-t0+i1):
        for jj in range(i1,i2):
            phi[ii,jj] = w[ii+t0-jj]   
    return phi