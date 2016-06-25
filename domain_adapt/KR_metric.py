import numpy as np
import scipy
import functools

#def log_k_f(A, x1, x2):
#    return -np.linalg.norm(A.dot(x1)-A.dot(x2))**2

def log_K_f(A, xs):
    return log_K_f_test(A, xs, xs)
    N = xs.shape[0]
    us = A.dot(xs.T).T
    ans = np.zeros((N,N))
    for i in xrange(N):
        ans[i,:] = -(np.linalg.norm(us - us[i,:],axis=1,ord=2)**2)
    return ans

def log_K_f_test(A, xs, xs_test):
    N = xs.shape[0]
    N_test = xs_test.shape[0]
    us = A.dot(xs.T).T
    us_test = A.dot(xs_test.T).T
    ans = np.zeros((N_test,N))
#    import pdb
#    pdb.set_trace()
    for i in xrange(N_test):
        ans[i,:] = -(np.linalg.norm(us - us_test[i,:],axis=1,ord=2)**2)
    return ans

#def ys_hat_f(K, ys):
#    top = K.dot(ys) - (np.diag(K)*ys)
#    bottom = K.sum(axis=1) - np.diag(K)
#    return top / bottom

def ys_hat_f_log_K_test(A, xs, ys, xs_test):

    log_K_test = log_K_f_test(A, xs, xs_test)
    
    top = log_K_test + np.log(ys)
#    np.fill_diagonal(top,-np.inf)
    
    bottom = log_K_test
#    log_K_diagonal = np.diag(log_K)
#    np.fill_diagonal(bottom,-np.inf)
    
    top_sums = scipy.misc.logsumexp(top,axis=1)
    bottom_sums = scipy.misc.logsumexp(bottom,axis=1)

    ans = np.exp(top_sums - bottom_sums)
#    np.fill_diagonal(log_K, log_K_diagonal)
    return ans

def ys_hat_f_log_K(A, xs, ys):

    log_K = log_K_f(A,xs)
    
    top = log_K + np.log(ys)
    np.fill_diagonal(top,-np.inf)
    
    bottom = log_K
    log_K_diagonal = np.diag(log_K)
    np.fill_diagonal(bottom,-np.inf)
    
    top_sums = scipy.misc.logsumexp(top,axis=1)
    bottom_sums = scipy.misc.logsumexp(bottom,axis=1)

    ans = np.exp(top_sums - bottom_sums)
    np.fill_diagonal(log_K, log_K_diagonal)
    return ans

def L(xs, ys, ws, A):
    N = len(xs)
    if ws is None:
        ws = np.ones(N)
#    log_K = log_K_f(A,xs)
    #K = np.exp(log_K)
    ys_hat_log_K = ys_hat_f_log_K(A,xs,ys)
#    import pdb
#    pdb.set_trace()
    return (ws * ((ys_hat_log_K - ys)**2)).sum()
    return np.linalg.norm(ys_hat_log_K - ys)**2

def L_grad(xs, ys, ws, A):
    N = len(xs)
    if ws is None:
        ws = np.ones(N)
    D = xs.shape[1]
    log_K = log_K_f(A,xs)
    K = np.exp(log_K)
    np.fill_diagonal(K,0)
    K = K / K.sum(axis=1)[:,np.newaxis]
    ys_hat = ys_hat_f_log_K(A,xs,ys)

    multiplier = np.zeros((D,D))
    for i in xrange(N):
        for j in xrange(N):
            multiplier += ws[i] * (ys_hat[i]-ys[i]) * (ys_hat[i]-ys[j]) * K[i,j] * np.outer(xs[i]-xs[j],xs[i]-xs[j])

    return 4*A.dot(multiplier)

def line_search(alpha_0, c, tau, f, t, x, grad):
    alpha = alpha_0
    m = np.linalg.norm(grad,ord=2)
    #print 'line search'
    #print f(x-(alpha*grad)),f(x),grad,x
    while f(x-(alpha*grad)) - f(x) > -c * alpha * m:
        #print f(x-(alpha*grad)),f(x),grad
        alpha = alpha * tau
    print 'alpha, change', alpha, f(x-(alpha*grad)) - f(x)
    return alpha

def grad_descent(step_f, stop_f, x_0, f, f_grad):
    x = x_0
    t = 0
    while True:
        grad = f_grad(x)
        alpha = step_f(t, x, grad)
        x_new = x - (alpha * grad)
        print 'x,f(x)', x_new,f(x_new)
        if stop_f(t, x, x_new):
            return x
        x = x_new
        t += 1

def update_A(mat_to_vec, vec_to_mat, L, L_grad, grad_descent, xs, ys, ws, A):

    mat_f = functools.partial(L,xs,ys,ws)
    vec_f = lambda vec_A: mat_f(vec_to_mat(vec_A))
    mat_f_grad = functools.partial(L_grad,xs,ys,ws)
    vec_f_grad = lambda vec_A: mat_to_vec(mat_f_grad(vec_to_mat(vec_A)))

    return grad_descent(A, vec_f, vec_f_grad)
        
def run(update_A, weight_f, A_0, ws_0, xs_train, xs_test, ys):
    A = A_0
    ws = ws_0
    for i in xrange(10):
        A = update_A(xs_train,ys,ws,A)
        ws = weight_f(xs_train.dot(A.T), xs_test.dot(A.T))
        print A
    return A
