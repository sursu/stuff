import numpy as np
from scipy import stats
from scipy.optimize import minimize

import autograd.numpy as anp
from autograd import hessian, value_and_grad


y = np.random.normal(scale=[1,4], size=(100,2)).T.flatten()


def likelihood_AD(gamma):
    # Parameters
    p00, p11 = 1/(1+anp.exp(-gamma[:2]))
    mu       = gamma[2:4]
    sigma2   = anp.exp(gamma[4:])
    T        = len(y)
    
    # Transition Matrix
    P = np.array([[p00, 1-p11],[1-p00, p11]])
    
    # Bookkeeping
    global xi_10, xi_11, xi_1T
    xi_10 = np.zeros(shape=(T+1,2))    # Predictive
    xi_11 = np.zeros(shape=(T,2))      # Filtered
    lik   = np.full(T, fill_value=None)
    
    # Initialize to OLS estimates:
    A  = np.row_stack([np.identity(2) - P,np.ones(2)])
    xi_10[0] = anp.linalg.inv(A.T@A)@A.T@np.concatenate([np.zeros(2),[1]])
    
    # Forward filter recursion
    for t in range(T):
        # State densities
        eta        = stats.norm.pdf(y[t], mu, sigma2)

        # Likelihood
        lik[t]     = anp.log(eta@xi_10[t])

        # Filtering
        xi_11[t]   = (eta * xi_10[t])  / (eta@xi_10[t])

        # Prediction
        xi_10[t+1] = P@xi_11[t]

    # Return likelihood-value
    return -anp.sum(lik)



# Optimization
results = minimize(
                value_and_grad(likelihood_AD),
                x0=np.random.normal(size=6),
                jac=True,
                method='L-BFGS-B',
                options={'gtol': 1e-7, 'maxiter': 20000, 'disp': True}, 
            )
