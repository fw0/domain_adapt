import autograd.numpy as np
import autograd
import pdb
import sklearn.metrics.pairwise as pairwise
import pdb
import python_utils.python_utils.basic as basic
import functools, itertools
import matplotlib.pyplot as plt
import sklearn.linear_model

NA = np.newaxis

class nothing_kernel(object):

    def projected_K(self, X, U):
        return X.dot(U)

    def U_i_j_to_projected_K_ks_l_grad(self, X, U, i, j):
        # return N by L matrix
        N = X.shape[0]
        L = U.shape[1]
        ans = np.zeros((N, L))
        ans[:,j] = X[:,i]
        return ans

class rbf_kernel(object):

    def __init__(self, gamma):
        self.gamma = gamma
    
    def projected_K(self, X, U):
        X_U = X.dot(U)
        ans = pairwise.rbf_kernel(X_U, gamma=self.gamma/2.)
#        pdb.set_trace()
        return ans

    def U_i_j_to_projected_K_ks_l_grad(self, X, U, i, j):
        try:
            U_j = X.dot(U[:,j])
        except:
            pdb.set_trace()
        X_i = X[:,i]
        K = self.projected_K(X, U)
        ans = self.gamma * (X_i[NA,:] - X_i[:,NA]) * (U_j[NA,:] - U_j[:,NA]) * K
        return -ans

"""
new stuff for KMM, KDR
"""

def get_gaussian_K(sigma, xs1, xs2):

    if len(xs1.shape) == 1:
        xs1 = xs1.reshape((len(xs1),1))

    if len(xs2.shape) == 1:
        xs2 = xs2.reshape((len(xs2),1))
                
    diff = xs1[:,np.newaxis,:] - xs2[np.newaxis,:,:]
    norms = np.sum(diff * diff, axis=2)

    K = np.exp(-1. * norms / (2 * (sigma**2)))

    return K

def center_K(K):
    N = len(K)
    B = np.eye(N) - (np.dot(np.ones((N,1), dtype=float), np.ones((1,N), dtype=float)) / float(N))
    return np.dot(np.dot(B, K), B)

def get_KMM_ineq_constraints(num_train, B_max, eps):
    G_gt_0 = -np.eye(num_train)
    h_gt_0 = np.zeros(num_train)
    G_lt_B_max = np.eye(num_train)
    h_lt_B_max = np.ones(num_train) * B_max
    G_B_sum_lt = np.ones(num_train, dtype=float)
    h_B_sum_lt = (1+eps) * float(num_train) * np.ones(1)
    G_B_sum_gt = -np.ones(num_train, dtype=float)
    h_B_sum_gt = -(1-eps) * float(num_train) * np.ones(1)
    G = np.vstack((G_gt_0,G_lt_B_max,G_B_sum_lt,G_B_sum_gt))
    (h_gt_0,h_lt_B_max,h_B_sum_lt,h_B_sum_gt)
    h = np.hstack((h_gt_0,h_lt_B_max,h_B_sum_lt,h_B_sum_gt))    
    return G,h

def get_KMM_params(xs_train, xs_test, get_K):
    num_train = xs_train.shape[0]
    num_test = xs_test.shape[0]
    K = get_K(xs_train, xs_train)
    kappa = -(float(num_train)/num_test) * np.sum(get_K(xs_train, xs_test), axis=1)
    return K, kappa

def get_KMM_ws_given_P(xs_train, xs_test, get_K, B_max, eps, P):

    # project data
    us_train = np.dot(xs_train, P)
    us_test = np.dot(xs_test, P)

    return get_KMM_ws(B_max, eps, get_K, us_train, us_test)

def cvxopt_solver(P, q, G, h):
    import cvxopt
    from cvxopt import solvers
    solvers.options['show_progress'] = False
    ans = cvxopt.solvers.qp(\
                  cvxopt.matrix(P,tc='d'),\
                  cvxopt.matrix(q,tc='d'),\
                  cvxopt.matrix(G,tc='d'),\
                  cvxopt.matrix(h,tc='d'),\
#                  show_progress=False,\
                  )
    ans = np.array(ans['x'])[:,0]
    return ans

def get_KMM_ws(B_max, eps, get_K, xs_train, xs_test, cvxopt_solver=cvxopt_solver):
    
    # wrapper
    num_train = xs_train.shape[0]
    num_test = xs_test.shape[0]
    
    # define optimization problem
    K, kappa = get_KMM_params(xs_train, xs_test, get_K)
#    print 'kappa', kappa
    G,h = get_KMM_ineq_constraints(num_train, B_max, eps)
    
    # solve
    return cvxopt_solver(K, kappa, G, h)

def get_trace_from_ws_and_Ks(eps, Ky, Kx, ws=None):
    Gy = center_K(Ky)
    Gx = center_K(Kx)
    N = len(Kx)
    #print 'ws', ws
    if ws is None:
        ws = np.ones(N)
    ans = np.trace(np.dot(np.dot(np.diag(ws), Gy), np.linalg.inv(np.dot(np.diag(ws), Gx + float(N) * eps * np.eye(N)))))    
    return ans
    
#def get_trace_from_P(xs, ys, get_K, eps, P, ws):#=None):
#    # wrapper
#    us = np.dot(xs, P)
#    Ky = get_K(ys, ys)
#    Ku = get_K(us, us)
#    return get_trace_from_ws_and_Ks(eps, Ky, Ku, ws)

def ortho(P):
    ans = np.zeros(P.shape)
    x_dim, z_dim = P.shape
    for j in xrange(z_dim):
        temp = P[:,j] - np.dot(ans, np.dot(ans.T, P[:,j]))
        temp = temp / np.linalg.norm(temp)
        ans[:,j] = temp
    return ans

def get_tight_constraints(A, b, x):
    LHS = np.dot(A, x)
    assert (LHS < b).all()
    tight_eps = 0.0001
#    tight_eps = 0.0000000001
    tight = (b - LHS) < tight_eps
#    print 'num_tight:', np.sum(tight)
    A_tight = A[tight]
    b_tight = b[tight]
    return A_tight, b_tight

def get_dxopt_delta_p(lin_solver, df_dx, d_dp_df_dx, d_dx_df_dx, A, b, xopt, p, delta_p_direction):
    
    # f(x, p) should be convex
    x_len = A.shape[1]

    # get tight constraints
    A_tight, b_tight = get_tight_constraints(A, b, xopt)
    num_tight = A_tight.shape[0]

    # get d
    p_dim = len(delta_p_direction.shape)
    delta_p_direction_broadcasted = np.tile(delta_p_direction, tuple([x_len] + [1 for i in xrange(p_dim)]))
    d_top = -np.sum(d_dp_df_dx(p, xopt) * delta_p_direction_broadcasted, axis=tuple(range(1,1+p_dim)))
    d_bottom = np.zeros(num_tight)
    d = np.hstack((d_top,d_bottom))

    # get C
    C = np.vstack((np.hstack((d_dx_df_dx(xopt, p), -A_tight.T)), np.hstack((A_tight, np.zeros((num_tight, num_tight))))))

    # get deriv
    deriv = lin_solver(C, d)
    
#    print 'solver error:', np.linalg.norm(np.dot(C,deriv) - d)

    return deriv

def get_dxopt_dp(lin_solver, df_dx, d_dp_df_dx, d_dx_df_dx, A, b, xopt, p):

    ans = np.zeros(xopt.shape + p.shape)
    
    for index in np.ndindex(*p.shape):
        delta_p_direction = np.zeros(p.shape)
        delta_p_direction[index] = 1.
        temp = get_dxopt_delta_p(lin_solver, df_dx, d_dp_df_dx, d_dx_df_dx, A, b, xopt, p, delta_p_direction)
        ans[(slice(None),)+index] = temp[:len(xopt)]

    return ans

def get_dL_dp_thru_xopt(lin_solver, df_dx, d_dp_df_dx, d_dx_df_dx, dL_dxopt, A, b, xopt, p, L_args=None, f_args=None):
    # assumes L(x_opt), x_opt = argmin_x f(x,p) subject to Ax<=b
    # L_args is for arguments to L besides x_opt

    # first, get dL/dws to calculate the gradient at ws1
    if not L_args is None:
        pass
        #print 'L_args len:', len(L_args)
    else:
        print 'NONE'
    if L_args is None:
        dL_dxopt_anal_val1 = dL_dxopt(xopt) #
    else:
        dL_dxopt_anal_val1 = dL_dxopt(xopt, L_args)
    
    # get tight constraints
    A_tight, b_tight = get_tight_constraints(A, b, xopt)
    num_tight = A_tight.shape[0]

    # make C matrix
#    pdb.set_trace()
    if f_args is None:
        C_corner = d_dx_df_dx(xopt, p)
    else:
        C_corner = d_dx_df_dx(xopt, p, f_args)
    C = np.vstack((np.hstack((C_corner,-A_tight.T)), np.hstack((A_tight,np.zeros((num_tight,num_tight))))))
#    print 'C', C
#    print 'C rank', np.linalg.matrix_rank(C), C.shape
#    print 'C corner rank', np.linalg.matrix_rank(C_corner), C_corner.shape
    
    # make d vector
    d = np.hstack((dL_dxopt_anal_val1, np.zeros(num_tight)))

    # solve Cv=d for x
    v = lin_solver(C, d)
#    print 'v', v
#    print 'solver error:', np.linalg.norm(np.dot(C,v) - d)

    # make D
    if f_args is None:
        d_dp_df_dx_anal_val1 = d_dp_df_dx(p, xopt)
    else:
        d_dp_df_dx_anal_val1 = d_dp_df_dx(p, xopt, f_args)
    D = np.vstack((-d_dp_df_dx_anal_val1, np.zeros((num_tight,)+p.shape)))
#    print 'D', D[0:10]

    return np.sum(D.T * v[tuple([np.newaxis for i in xrange(len(p.shape))])+(slice(None),)], axis=-1).T

# stuff for input=P, output=L, intermediate variables are w*.  these fxns return actual values

def get_dobj_dP(xs_train, xs_test, Ky, SDR_get_K, KMM_get_K, dobj_dP_thru_Ku, lin_solver, cvxopt_solver, df_dws, d_dP_df_dws, d_dws_df_dws, dobj_dwsopt, A, b, P):
    # wrapper

    # calculate intermediate stuff
    us_train = np.dot(xs_train, P)
    us_test = np.dot(xs_test, P)
    KMM_Ku, kappau = get_KMM_params(us_train, us_test, KMM_get_K)
    wsopt = cvxopt_solver(KMM_Ku, kappau, A, b)

    # gradient thru wsopt
    SDR_Ku = SDR_get_K(us_train, us_train)
    dobj_dP_thru_wsopt_val = get_dL_dp_thru_xopt(lin_solver, df_dws, d_dP_df_dws, d_dws_df_dws, dobj_dwsopt, A, b, wsopt, P, L_args=(Ky, SDR_Ku)) # NVM.  changed so that f(x,p) can also accept optional args

    # gradient thru Ku
    dobj_dP_thru_Ku_val = dobj_dP_thru_Ku(P, wsopt)

    return dobj_dP_thru_wsopt_val + dobj_dP_thru_Ku_val

def get_obj(obj_from_ws_and_Ks, xs_train, xs_test, Ky, KMM_get_K, SDR_get_K, cvxopt_solver, A, b, P):

    # wrapper

    us_train = np.dot(xs_train, P)
    us_test = np.dot(xs_test, P)
    Ku, kappau = get_KMM_params(us_train, us_test, KMM_get_K)
    wsopt = cvxopt_solver(Ku, kappau, A, b)
    SDR_Ku = SDR_get_K(us_train, us_train)
    return obj_from_ws_and_Ks((Ky, SDR_Ku), wsopt)

# stuff for input=w*, input=P, intermediate variables are B*

def weighted_logreg_loss(reg, ws, (ys, xs), B):
    #print 'ws',ws.shape, 'ys',ys.shape, 'xs',xs.shape, 'B',B.shape
    return logloss(ys, np.dot(xs,B), ws) + (0.5 * reg * np.dot(B, B)) #/ float(len(ys))

def logloss(ys, ys_hat, ws=None):
    #print 'ws',ws.shape, 'ys',ys.shape, 'xs',xs.shape, 'B',B.shape
    if ws is None:
        return np.sum(np.log(1 + np.exp(-ys * ys_hat))) #+ (0.5 * reg * np.dot(B, B)) #/ float(len(ys))
    else:
        try:
            return np.sum(ws * np.log(1 + np.exp(-ys * ys_hat))) #+ (0.5 * reg * np.dot(B, B)) #/ float(len(ys))
        except:
            pdb.set_trace()
    
def weighted_logreg_get_Bopt(reg, ws, (ys, xs)):
    import sklearn.linear_model
    #fitter = sklearn.linear_model.LogisticRegression(penalty='l2', C=reg, fit_intercept=False)
    fitter = sklearn.linear_model.LogisticRegression(penalty='l2', C=1./reg, fit_intercept=False, solver='liblinear')
    fitter.fit(xs, ys.astype(int), sample_weight=ws)
#    print ys.max()
    #print fitter.coef_
    return fitter.coef_[0,:]

def Bopt_get_dobj_dwsopt_thru_Bopt(get_Bopt, dobj_dwsopt_thru_wsopt, lin_solver, cvxopt_solver, dg_dB, d_dwsopt_dg_dB, d_dB_dg_dB, dobj_dBopt, wsopt, (Ky, SDR_Ku)):
    # wrapper

    # calculate intermediate stuff
    Bopt = get_Bopt(wsopt, (Ky, SDR_Ku)) # IMPLEMENT

    # gradient through Bopt
    B_dim = SDR_Ku.shape[1]
    dobj_dwsopt_thru_Bopt_val = get_dL_dp_thru_xopt(lin_solver, dg_dB, d_dwsopt_dg_dB, d_dB_dg_dB, dobj_dBopt, np.zeros((0,B_dim)), np.zeros((0,)), Bopt, wsopt, L_args=(wsopt, (Ky, SDR_Ku)), f_args=(Ky, SDR_Ku))

    return dobj_dwsopt_thru_Bopt_val

def Bopt_get_dobj_dwsopt(get_Bopt, dobj_dwsopt_thru_wsopt, lin_solver, cvxopt_solver, dg_dB, d_dwsopt_dg_dB, d_dB_dg_dB, dobj_dBopt, wsopt, (Ky, SDR_Ku)):
    # wrapper

    # calculate intermediate stuff
    Bopt = get_Bopt(wsopt, (Ky, SDR_Ku)) # IMPLEMENT

    # gradient through Bopt
    B_dim = SDR_Ku.shape[1]
    dobj_dwsopt_thru_Bopt_val = get_dL_dp_thru_xopt(lin_solver, dg_dB, d_dwsopt_dg_dB, d_dB_dg_dB, dobj_dBopt, np.zeros((0,B_dim)), np.zeros((0,)), Bopt, wsopt, L_args=(wsopt, (Ky, SDR_Ku)), f_args=(Ky, SDR_Ku))

    # gradient through wsopt directly
    #dobj_dwsopt_thru_wsopt_val = dobj_dwsopt_thru_wsopt(wsopt, Bopt, L_args=(Ky, SDR_Ku), f_args=(Ky, SDR_Ku)) # CHANGE: note THREE arguments
    dobj_dwsopt_thru_wsopt_val = dobj_dwsopt_thru_wsopt(wsopt, Bopt, (Ky, SDR_Ku))
    
    return dobj_dwsopt_thru_Bopt_val + dobj_dwsopt_thru_wsopt_val

def Bopt_get_dobj_dP_thru_Bopt(Ky, SDR_get_K, get_Bopt, dobj_dP_thru_Ku, lin_solver, cvxopt_solver, dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, P, wsopt,  (SDR_Ky, SDR_Ku)):
    # wrapper

    # calculate intermediate stuff
    #us_train = np.dot(xs_train, P)
    #SDR_Ku = SDR_get_K(us_train, us_train)
    Bopt = get_Bopt(wsopt, (SDR_Ky, SDR_Ku))
    
    # gradient through Bopt
    B_dim = SDR_Ku.shape[1]
#    pdb.set_trace()
    dobj_dP_thru_Bopt_val = get_dL_dp_thru_xopt(lin_solver, dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, np.zeros((0,B_dim)), np.zeros((0,)), Bopt, P, L_args=(wsopt, (SDR_Ky, SDR_Ku)), f_args=(wsopt,SDR_Ky)) # dobj_dBopt should accept (wsopt, (Ky, SDR_Ku)), same as that used in get_dobj_dwsopt_thru_Bopt
    return dobj_dP_thru_Bopt_val
    
def Bopt_get_dobj_dP(xs_train, Ky, SDR_get_K, get_Bopt, dobj_dP_thru_Ku, lin_solver, cvxopt_solver, dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, P, wsopt):
    # wrapper

    # calculate intermediate stuff
    us_train = np.dot(xs_train, P)
    SDR_Ku = SDR_get_K(us_train, us_train)
    Bopt = get_Bopt(wsopt, (Ky, SDR_Ku))

    # gradient through Bopt
    B_dim = SDR_Ku.shape[1]
    dobj_dP_thru_Bopt_val = get_dL_dp_thru_xopt(lin_solver, dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, np.zeros((0,B_dim)), np.zeros((0,)), Bopt, P, L_args=(wsopt, (Ky, SDR_Ku)), f_args=(wsopt,Ky)) # dobj_dBopt should accept (wsopt, (Ky, SDR_Ku)), same as that used in get_dobj_dwsopt_thru_Bopt

    # gradient through P directly
    dobj_dP_thru_P_val = dobj_dP_thru_Ku(P, wsopt, Bopt) # has Bs_opt as argument compared to the original dobj_dP_thru_Ku.  should really pass in Ky as well, instead of using closure

    #print 'dobj_dP_thru_Bopt_val', dobj_dP_thru_Bopt_val
    #print 'dobj_dP_thru_P_val', dobj_dP_thru_P_val
    
    return dobj_dP_thru_Bopt_val + dobj_dP_thru_P_val

    
    
###

def ws_obj_f(xs_train, xs_test, get_K, ws, P):
    
    # create K, kappa, use them to return objective fxn value
    
    us_train = np.dot(xs_train, P)
    us_test = np.dot(xs_test, P)
    K, kappa = get_KMM_params(us_train, us_test, get_K)
    
    return np.dot(np.dot(ws.T, K), ws)/2. + np.dot(kappa, ws)

def ws_distance(ws):
    return np.dot(ws.T, ws) / (len(ws)**2)



# wrappers
#def get_obj_and_obj_gradient(KMM_get_K, B_max, KMM_eps, SDR_get_K, SDR_get_Ky, obj_from_ws_and_Ks, lin_solver, cvxopt_solver, xs_train, xs_test, ys_train):
#    pass

def get_obj_and_obj_gradient(KMM_get_K, B_max, KMM_eps, SDR_get_K, SDR_get_Ky, obj_from_ws_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku, lin_solver, cvxopt_solver, xs_train, xs_test, ys_train):
    # get_dobj_dP_thru_Ku accepts xs_train, SDR_get_K, Ky
    
    the_f = functools.partial(ws_obj_f, xs_train, xs_test, KMM_get_K)
    df_dws = autograd.jacobian(the_f) # ans dim: |x|
    def the_f_reverse(P,ws):
        return df_dws(ws,P)
    #d_dP_df_dws = autograd.jacobian(lambda P,ws: df_dws(ws,P)) # ans dim: |x| x |p|
    d_dP_df_dws = autograd.jacobian(the_f_reverse) # ans dim: |x| x |p|
    d_dws_df_dws = autograd.jacobian(autograd.jacobian(the_f))

    Ky = SDR_get_Ky(ys_train, ys_train)

    #dobj_dwsopt = autograd.jacobian(lambda wsopt, Ky, Ku: obj_from_ws_and_Ks(Ky, Ku, wsopt))

    #def obj_from_P_and_wsopt(P, wsopt):
    #    us_train = np.dot(xs_train, P)
    #    Ku = SDR_get_K(us_train, us_train)
    #    return obj_from_ws_and_Ks(Ky, Ku, wsopt)

    #dobj_dP_thru_Ku = autograd.jacobian(obj_from_P_and_wsopt)

    dobj_dP_thru_Ku = get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky)

    A, b = get_KMM_ineq_constraints(len(xs_train), B_max, KMM_eps)

    dobj_dP = functools.partial(get_dobj_dP, xs_train, xs_test, Ky, SDR_get_K, KMM_get_K, dobj_dP_thru_Ku, lin_solver, cvxopt_solver, df_dws, d_dP_df_dws, d_dws_df_dws, dobj_dwsopt, A, b)

    obj = functools.partial(get_obj, obj_from_ws_and_Ks, xs_train, xs_test, Ky, KMM_get_K, SDR_get_K, cvxopt_solver, A, b)

    return obj, dobj_dP

# diagnostic fxns

def plot_weights(xs_train, xs_test, KMM_get_K, B_max, KMM_eps, P):
    ws = get_KMM_ws_given_P(xs_train, xs_test, KMM_get_K, B_max, KMM_eps, P)
    fig, ax = plt.subplots()
    ax.scatter(ws, np.zeros(len(ws)), s=2.)
    ax.set_title('distribution distance: %.2f' % ws_distance(ws))
    ax.set_xlabel('weight')
    basic.display_fig_inline(fig)

def plot_K(xs1, xs2, get_K, P, title):
    K = get_K(np.dot(xs1, P), np.dot(xs2, P))
    fig, ax = plt.subplots()
    ax.hist(K.flatten(), bins=100)
    ax.set_title(title)
    basic.display_fig_inline(fig)

def gradient_check(obj, dobj_dP, P):
    delta = 0.0001
    dobj_dP_val = np.zeros(P.shape)
    P1 = P
    obj1 = obj(P1)
    for i in xrange(P.shape[0]):
        for j in xrange(P.shape[1]):
            delta_P_direction = np.zeros(P.shape, dtype=float)
            delta_P_direction[i,j] = 1.
            P2 = P1 + (delta * delta_P_direction)
            obj2 = obj(P2)
            dobj_dP_val[i,j] = (obj2 - obj1) / delta
    print 'numerical gradient:'
    print dobj_dP_val
    print 'analytical gradient:'
    print dobj_dP(P)

def plot_train_vs_test(xs_train, xs_test, P):
    us_train = np.dot(xs_train, P)
    us_test = np.dot(xs_test, P)
    fig, ax = plt.subplots()
    assert P.shape[1] <= 2
    if P.shape[1] == 1:
        ax.hist(us_train[:,0], color='r', label='train',alpha=0.7)
        ax.hist(us_test[:,0], color='b', label='test',alpha=0.7)
        #ax.scatter(us_train[:,0], np.zeros(len(us_train)), color='r', label='train')
        #ax.scatter(us_test[:,0], np.zeros(len(us_test)), color='b', label='test')
    elif P.shape[1] == 2:
        ax.scatter(us_train[:,0], us_train[:,1], color='r', label='train')
        ax.scatter(us_test[:,0], us_test[:,1], color='b', label='test')
    ax.legend()
    ax.set_title('projected train vs test features')
    basic.display_fig_inline(fig)

def plot_y_vs_u(xs_train, ys_train, P, xs_test=None, ys_test=None, xlim=None, ylim=None):
    s = 0.75
    alpha = 0.8
    us_train = np.dot(xs_train, P)
    if P.shape[1] == 1:
        fig, ax = plt.subplots()
        ax.scatter(us_train[:,0], ys_train, color='r', label='train', s=s, alpha=alpha)
        if not ys_test is None:
            us_test = np.dot(xs_test, P)
            ax.scatter(us_test[:,0], ys_test, color='b', label='test', s=s, alpha=alpha)
            ax.legend()
        ax.set_xlabel('u0')
        ax.set_ylabel('y')
    elif P.shape[1] == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(us_train[:,0], us_train[:,1],ys_train, color='r', label='train', s=s, alpha=alpha)
        if not ys_test is None:
            us_test = np.dot(xs_test, P)
            ax.scatter(us_test[:,0], us_test[:,1],ys_test, color='b', label='test', s=s, alpha=alpha)
        ax.set_xlabel('u0')
        ax.set_ylabel('u1')
        ax.set_zlabel('y')
    ax.set_title('y vs projected features')
    basic.display_fig_inline(fig)

def plot_opt_log(opt_log):

    fig,ax = plt.subplots()
    obj_vals = opt_log['iterations']['f(x)']
    ax.plot(range(len(obj_vals)), obj_vals)
    ax.set_title('obj vals')
    ax.set_xlabel('step')
    ax.set_ylabel('obj val')
    basic.display_fig_inline(fig)

    fig,ax = plt.subplots()
    grad_norms = opt_log['iterations']['gradnorm']
    ax.plot(range(len(grad_norms)), grad_norms)
    ax.set_title('grad norms')
    ax.set_xlabel('step')
    ax.set_ylabel('grad norm')
    basic.display_fig_inline(fig)

###### sklearn wrappers

from sklearn.base import TransformerMixin as TransformerMixin

class myTransformerMixin(TransformerMixin):

    def get_params(self, deep=True):
        return self.params

    def set_params(self, **kwargs):
        for (key,val) in kwargs.iteritems():
            self.params[key] = val
        return self

class predictProbaLogisticRegression(sklearn.linear_model.LogisticRegression):

    def predict(self, X):
        ps = self.predict_proba(X)[:,1]
        return np.log(ps / (1.-ps))
    
    
class shiftEstimator(object):#myTransformerMixin):

    def fit(self, shift_X_with_ws, shift_y, **fit_params):
        source_X, target_X, source_y, target_y, source_ws = shift_Xy_to_matrices(shift_X_with_ws, shift_y, weights=True) # WITH WEIGHTS
#        assert (fit_params['sample_weight'] == source_ws).all()
        super(shiftEstimator,self).fit(source_X, source_y, sample_weight=source_ws)
        return self
#        super(myLogisticRegression, self).fit(source_X, source_y, **fit_params)

    def source_predict(self, shift_X):
        source_X, target_X, source_y = shift_Xy_to_matrices(shift_X)
        return super(shiftEstimator,self).predict(source_X)
#        ps = super(myLogisticRegression,self).predict_proba(source_X)
#        return np.log(ps / (1. - ps))[:,1]
#        super(myLogisticRegression, self).predict(source_X, source_y)
    
    def predict(self, shift_X):
        # predicts target domain.  calling it ``predict'' assumes that a pipeline 
        source_X, target_X, source_y = shift_Xy_to_matrices(shift_X)
        return super(shiftEstimator,self).predict(target_X)
#        ps = sklearn.linear_model.LogisticRegression.predict_proba(self, target_X)
#        return np.log(ps / (1. - ps))[:,1]

class shiftLogisticRegression(shiftEstimator, predictProbaLogisticRegression):
    pass


def shift_Xy_to_matrices(shift_X, shift_y=None, weights=False):

    flatten = lambda ls: [l_i for l in ls for l_i in l]

    source_X_l = []
    target_X_l = []
    source_y_l = []
    try:
        for (source_X_elt, target_X_elt, source_y_elt) in zip(*zip(*shift_X)[0:3]):
            if len(target_X_elt) > 0:
                target_X_l.append(target_X_elt)
            source_X_l.append(source_X_elt)
            source_y_l.append(source_y_elt)
    except:
        pdb.set_trace()

    if shift_y is None:
        if not weights:
            return np.array(source_X_l), np.vstack(tuple(flatten(target_X_l))), np.array(source_y_l)
        else:
            assert len(iter(shift_X).next()) == 4
            return np.array(source_X_l), np.vstack(tuple(flatten(target_X_l))), np.array(source_y_l), np.array([shift_X_elt[-1] for shift_X_elt in shift_X])
    else:
        target_y_l = []
        for (source_X_elt, target_X_elt, source_y_elt), target_y_elt in itertools.izip(zip(*zip(*shift_X)[0:3]), shift_y):
            if len(target_X_elt) > 0:
                target_y_l.append(target_y_elt)
        if not weights:
            return np.array(source_X_l), np.vstack(tuple(flatten(target_X_l))), np.array(source_y_l), np.hstack(tuple(flatten(target_y_l)))
        else:
            return np.array(source_X_l), np.vstack(tuple(flatten(target_X_l))), np.array(source_y_l), np.hstack(tuple(flatten(target_y_l))), np.array([shift_X_elt[-1] for shift_X_elt in shift_X])

def matrices_to_shift_Xy(source_X, target_X, source_y, target_y=None):
    shift_X = []
    shift_y = []
    assert len(source_X) == len(source_y)
    per = float(len(target_X)) / len(source_X)
    old_int = 0
    this = []
    for i in xrange(len(source_X)):
        shift_X.append((source_X[i],list(target_X[int(i*per):int((i+1)*per),:]), source_y[i]))
    if target_y is None:
        return shift_X
    else:
        for i in xrange(len(source_X)):
            shift_y.append(target_y[int(i*per):int((i+1)*per)])
        return shift_X, shift_y

def add_weights_to_shift_X(ws, shift_X):
    return [tuple(list(shift_X_elt) + [w]) for (shift_X_elt,w) in itertools.izip(shift_X, ws)]

def plot_projection(P, feature_names):
    d = {}
    import pandas as pd
    for (i,v) in enumerate(P.T):
        idxs = np.argsort(-np.abs(v))
        print 'v_%d' % i
        print pd.Series(v, index=feature_names).iloc[idxs].iloc[0:10]

class projection_estimator(myTransformerMixin):

    def __init__(self, horse=None, feature_names=None):
        self.params = {'horse':horse, 'feature_names':feature_names}

    def fit(self, shift_X, shift_y):
        source_X, target_X, source_y = shift_Xy_to_matrices(shift_X)
        self.P = self.params['horse'](source_X, target_X, source_y)
        if not self.params['feature_names'] is None:
            plot_projection(self.P, self.params['feature_names'])
        return self

#    def transform(self, shift_X, shift_y):
    def transform(self, shift_X):
        source_X, target_X, source_y = shift_Xy_to_matrices(shift_X)
        return matrices_to_shift_Xy(np.dot(source_X,self.P), np.dot(target_X,self.P), source_y)
    
class weighted_estimator(myTransformerMixin):

    def __init__(self, weight_estimator=None, downstream_estimator=None):
        self.params = {'weight_estimator':weight_estimator, 'downstream_estimator':downstream_estimator}

    def fit(self, shift_X, shift_y):
        source_X, target_X, source_y = shift_Xy_to_matrices(shift_X)
        source_weights = self.params['weight_estimator'](source_X, target_X)
        # add weights to shift_X
        shift_X_with_ws = add_weights_to_shift_X(source_weights, shift_X)
        estimator = self.params['downstream_estimator']
        import sklearn
        from sklearn import grid_search
        estimator.fit(shift_X_with_ws, shift_y)
        
        #if isinstance(estimator, grid_search.BaseSearchCV):
        #    estimator.fit_params = {'sample_weight':source_weights}
        #    estimator.fit(shift_X_with_ws, shift_y)
        #else:
        #    estimator.fit(shift_X_with_ws, shift_y, sample_weight=source_weights)
        return self

    def predict(self, shift_X):
        # predicts the source.  when evaluating it in outer outer loop, should predict target
#        source_X, target_X, source_y = shift_Xy_to_matrices(shift_X)
        return self.params['downstream_estimator'].predict(shift_X)
        
# utility fxns

def mat_median(m):
    l = len(m)
    v = np.arange(l).reshape((l,1))
    return np.median(m[v != v.T])

def median_distance(xs1, xs2):
    if len(xs1.shape) == 1:
        diff = xs1[:,np.newaxis] - xs2[np.newaxis,:]
        norms = np.sum(diff * diff, axis=1) ** 0.5
    elif len(xs1.shape) == 2:
        diff = xs1[:,np.newaxis,:] - xs2[np.newaxis,:,:]
        norms = np.sum(diff * diff, axis=2) ** 0.5
    return mat_median(norms)

def weighted_lsqr_loss(ws, xs, ys):
    #ws = np.ones(len(xs))
    N = xs.shape[0]
    W = ws * np.eye(N)
    #ans2 = ys.T.dot(W).dot(np.eye(N)-xs.dot(np.linalg.inv(xs.T.dot(W).dot(xs))).dot(xs.T).dot(W)).dot(ys)
    temp1 = np.linalg.inv(np.dot(xs.T*ws,xs))
    temp2 = np.dot(np.dot(xs,temp1),xs.T*ws)
    #print 'ys',ys
    #print 'ws',ws
    #print ys*ws
    temp4 = ys*ws
    temp3 = temp4 - np.dot(temp4,temp2)
    ans = np.sum(np.dot(temp3,ys))
    #print ans2, ans, np_weighted_lsqr_loss(ws, xs, ys), np.linalg.inv(xs.T.dot(W).dot(xs)).shape, ws.shape, xs.shape, ys.shape
    return ans
