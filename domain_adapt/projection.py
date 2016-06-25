import numpy as np
import scipy
import functools
import pdb
import kernels

NA = np.newaxis
norm_W = False
fix_W = True

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def logreg_loss(X, y, B):
    return np.log(1 + np.exp(-X.dot(B)*y)).sum()

def logreg_loss_grad(X, y, B):
    return (-y[:,NA]*X*sigmoid((-y[:,NA]*X).dot(B))[:,NA]).sum(axis=0)

def logreg(X, y):
    import sklearn.linear_model
    fitter = sklearn.linear_model.LogisticRegression(C=150,fit_intercept=False)
    fit = fitter.fit(X, y)
    #print dir(fit)
    return fit.coef_.flatten()
    
    fixed_logreg_loss = functools.partial(logreg_loss,X,y)
    fixed_logreg_loss_grad = functools.partial(logreg_loss_grad,X,y)
#    method = 'BFGS'
    method = 'CG'
    D = X.shape[1]
    B_0 = np.random.normal(size=D)
    ans = scipy.optimize.minimize(fixed_logreg_loss, B_0, jac=fixed_logreg_loss_grad, method=method)
    return ans['x']

# don't have to actually implement kernel version of logreg stuff - can just replace X with K and B with a

def U_to_B_opt(X, z, ker, U):
    ans = logreg(ker.projected_K(X,U),z)
    #print 'B_opt'
    #print ans
    return ans
    
def U_to_B_opt_grad(X, z, U, ker):

    # define some quantities
    X_U = ker.projected_K(X, U)
    B_opt = U_to_B_opt(X, z, ker, U)
    I = -z*X_U.dot(B_opt)

    # define L matrix    
    D = U.shape[0]
    K = U.shape[1]
    N = X_U.shape[1]
    L = np.zeros((N,N))
    for l in xrange(N):
        (sigmoid(I)*(1-sigmoid(I)))[:,NA]
        L[l,:] = ((sigmoid(I)*(1-sigmoid(I)))[:,NA]*X_U*X_U[:,l][:,NA]).sum(axis=0)

    # define C
    C = np.zeros(shape=(D,K,N))
    for i in xrange(D):
        for j in xrange(K):
            U_i_j_to_projected_K_ks_l_grad = ker.U_i_j_to_projected_K_ks_l_grad(X, U, i, j)
            C[i,j,:] = ((sigmoid(I)*(1-sigmoid(I)))[:,NA]*X_U*(U_i_j_to_projected_K_ks_l_grad.dot(B_opt))[:,NA]).sum(axis=0)
            #C[i,j,:] = ((sigmoid(I)*(1-sigmoid(I)))[:,NA]*X_U*(X[:,i][:,NA])*B_opt[j]).sum(axis=0)
            #C[i,j,j] -= (z*X[:,i]*sigmoid(I)).sum()
            C[i,j,:] -= (z[:,NA] * U_i_j_to_projected_K_ks_l_grad * sigmoid(I)[:,NA]).sum(axis=0)

    L_inv = np.linalg.inv(L)
#    print L_inv
    #print 'I'
    #print zip(I,z)
    #print 'sigmoid I'
    #print sigmoid(I)*(1-sigmoid(I))
    #print 'X_U'
    #print X_U
    J = np.zeros(C.shape)
    for i in xrange(D):
        for j in xrange(K):
            J[i,j,:] = L_inv.dot(-C[i,j,:])

    return J

def B_opt_U_to_W(r, X, B_opt, U):
    W = r * sigmoid(X.dot(U).dot(B_opt))
    if fix_W:
        return np.ones(X.shape[0])
    if not norm_W:
        return W
    else:
        return W / W.sum()

def U_to_W_grad(r, X, z, B_opt, U):
    D,K = U.shape
    N = X.shape[0]
#    B_opt = U_to_B_opt(X, z, U)
    W_pre = X.dot(U).dot(B_opt)
    U_to_W_pre_grad_direct = X.T[:,NA,:]*B_opt[NA,:,NA]
    B_opt_to_W_pre_grad = X.dot(U).T
    _U_to_B_opt_grad = U_to_B_opt_grad(X, z, U)
    U_to_W_pre_grad_indirect = np.zeros(shape=(D,K,N))
    for n in xrange(N):
        U_to_W_pre_grad_indirect[:,:,n] = (_U_to_B_opt_grad * B_opt_to_W_pre_grad[:,n][NA,NA,:]).sum(axis=2)
    U_to_W_pre_grad = U_to_W_pre_grad_direct + U_to_W_pre_grad_indirect
    _U_to_W_grad = U_to_W_pre_grad * (sigmoid(W_pre) * (1-sigmoid(W_pre)))[NA,NA,:]
    if fix_W:
        return np.zeros(shape=(D,K,N))
    if not norm_W:
        return _U_to_W_grad
    else:
        # redundant computation
        W = sigmoid(W_pre)
        W_sum = W.sum()
        W_to_norm_W_grad = (np.eye(N) - W[NA,:] / W_sum) / W_sum
        
        for i in xrange(D):
            for j in xrange(K):
                _U_to_W_grad[i,j,:] = _U_to_W_grad[i,j,:].dot(W_to_norm_W_grad)

        return _U_to_W_grad
        
def theta_U_to_Q(X, y, theta, U):
    # assume squared loss for now
    y_hat = X.dot(U).dot(theta)
    return (y_hat - y)**2

def theta_to_Q_grad(X, y, U, theta):
    return X.dot(U).T * (X.dot(U).dot(theta) - y).T[NA,:] * 2

def U_to_Q_grad(X, y, theta, U):
    return X.T[:,NA,:] * theta[NA,:,NA] * (X.dot(U).dot(theta) - y).T[NA,NA,:] * 2

def U_theta_to_L(r, X, y, z, U, theta):
    N = X.shape[0]
    Q = theta_U_to_Q(X, y, theta, U)
    B_opt = U_to_B_opt(X, z, U)
    #W = B_opt_U_to_W(r, X, B_opt, U)
    W = np.ones(N)
    L = W.dot(Q)
    print 'inside calculated L:', L
    return L

def U_theta_to_L_with_grad(r, X, y, z, U, theta):
    Q = theta_U_to_Q(X, y, theta, U)
    B_opt = U_to_B_opt(X, z, U)
    W = B_opt_U_to_W(r, X, B_opt, U)
    return W.dot(Q), (U_to_L_grad(r, X, y, theta, z, Q, B_opt, W, U),theta_to_L_grad(r, X, y, U, z, Q, W, theta))

def Q_to_L_grad(W, Q):
    return W

def W_to_L_grad(Q, W):
    return Q

def U_to_L_grad(r, X, y, theta, z, Q, B_opt, W, U):
    _U_to_Q_grad = U_to_Q_grad(X, y, theta, U)
    #Q = theta_U_to_Q(X, y, theta, U)
    #B_opt = U_to_B_opt(X, z, U)
    _U_to_W_grad = U_to_W_grad(r, X, z, B_opt, U)
    #W = B_opt_U_to_W(r, X, B_opt, U)
    _W_to_L_grad = W_to_L_grad(Q, W)
    _Q_to_L_grad = Q_to_L_grad(W, Q)

    U_to_L_grad_thru_Q = (_U_to_Q_grad * _Q_to_L_grad[NA,NA,:]).sum(axis=2)
    U_to_L_grad_thru_W = (_U_to_W_grad * _W_to_L_grad[NA,NA,:]).sum(axis=2)

    return U_to_L_grad_thru_Q + U_to_L_grad_thru_W

def U_to_L_grad_wrapper(r, X, y, theta, z, U):
    _U_to_Q_grad = U_to_Q_grad(X, y, theta, U)
    Q = theta_U_to_Q(X, y, theta, U)
    B_opt = U_to_B_opt(X, z, U)
    _U_to_W_grad = U_to_W_grad(r, X, z, B_opt, U)
    W = B_opt_U_to_W(r, X, B_opt, U)
    _W_to_L_grad = W_to_L_grad(Q, W)
    _Q_to_L_grad = Q_to_L_grad(W, Q)

    U_to_L_grad_thru_Q = (_U_to_Q_grad * _Q_to_L_grad[NA,NA,:]).sum(axis=2)
    U_to_L_grad_thru_W = (_U_to_W_grad * _W_to_L_grad[NA,NA,:]).sum(axis=2)

    return U_to_L_grad_thru_Q + U_to_L_grad_thru_W

def theta_to_L_grad(r, X, y, U, z, Q, W, theta):
    #Q = theta_U_to_Q(X, y, theta, U)
    #B_opt = U_to_B_opt(X, z, U)
    #W = B_opt_U_to_W(r, X, B_opt, U)
    return theta_to_Q_grad(X, y, U, theta).dot(Q_to_L_grad(W,Q))

def theta_to_L_grad_wrapper(r, X, y, U, z, theta):
    Q = theta_U_to_Q(X, y, theta, U)
    B_opt = U_to_B_opt(X, z, U)
    W = B_opt_U_to_W(r, X, B_opt, U)
    return theta_to_Q_grad(X, y, U, theta).dot(Q_to_L_grad(W,Q))


