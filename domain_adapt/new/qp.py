import numpy as np
import scipy
import cvxopt


def cvxopt_solver(P, q, G, h):

    from cvxopt import solvers
    solvers.options['show_progress'] = False
    ans = cvxopt.solvers.qp(\
                  cvxopt.matrix(P,tc='d'),\
                  cvxopt.matrix(q,tc='d'),\
                  cvxopt.matrix(G,tc='d'),\
                  cvxopt.matrix(h,tc='d'),\
#                  show_progress=False,\
                  )
    xs = np.array(ans['x'])[:,0]

    return xs


def primal_dual_qp(eps, eps_primal, eps_dual, G, h, l, u, A, b):
    """
    minimize_x x'(GG')x + h'x subject to l<x<u (element-wise) and Ax=b
    """

    # define some constants to reuse
    M, N = A.shape
    ones_N = np.ones(N)
    K = G.shape[1]
    eye_K = np.eye(K)
    alpha = 0.05
    beta = 0.5
    mu = 10.

    # define residual functions
    def get_r_dual(_x, _lam_lower, _lam_upper, _v):
        return (G.dot(G.T.dot(_x)) + h) + (-_lam_lower + _lam_upper) + A.T.dot(_v)

    def get_r_primal(_x):
        return A.dot(_x) - b

    def get_f_lower(_x):
        return -_x + l

    def get_f_upper(_x):
        return u - _x
    
    def get_r_slack_lower(_lam_lower, _x, _t):
        return - ((_lam_lower * get_f_lower(_x)) + (ones_N/t))

    def get_r_slack_upper(_lam_upper, _x, _t):
        return - ((_lam_upper * get_f_upper(_x)) + (ones_N/t))

    def get_r_norm(_x, _lam_lower, _lam_upper, _v, _t):
        return np.linalg.norm(get_r_dual(_x, _lam_lower, _lam_upper, _v)) + np.linalg.norm(get_r_primal(_x)) + np.linalg.norm(get_r_slack_lower(_lam_lower, _x, _t)) + np.linalg.norm(get_r_slack_upper(_lam_upper, _x, _t))
    
    # find feasible primal point
    x = np.random.uniform(l, u)

    # get feasible dual points
    lam_lower = np.random.uniform(0., 1., size=N)
    lam_upper = np.random.uniform(0., 1., size=N)
    v = np.random.uniform(-1.,1., size=M)

    # iterate until surrogate duality gap low and primal and dual residuals are low
    while True:

        # calculate gap and t
        f_lower = get_f_lower(_x)
        f_upper = get_f_upper(_x)
        gap = -(lam_lower.dot(f_lower) + lam_upper.dot(f_upper))
        t = mu * (M/gap)

        # calculate residuals
        r_dual = get_r_dual(x, lam_lower, lam_upper, v)
        r_primal = get_r_primal(_x)
        r_slack_lower = get_r_slack_lower(lam_lower, x, t)
        r_slack_upper = get_r_slack_upper(lam_upper, x, t)

        # eliminate d_lam to get augmented system with form
        # |Q + D_2, A^T| |d_x| = |y|
        # |A      , 0  | |d_v|   |z|
        D_2 = -((lam_lower / f_lower) + (lam_upper / f_upper))
        y = -(r_dual + (lam_lower + (ones_N / (t*f_lower))) - (lam_upper + (ones_N / (t*f_upper)))) # y_old = r_dual - ((r_slack_lower / f_lower) - (r_slack_upper / f_upper))
        z = -(r_primal)

        # eliminate d_x to get system
        # (A (Q+D_2)^-1 A^T) dv = -(z - A (Q+D_2)^-1 y) := w_2

        # first calculate RHS vector
        L_v_inv = np.linalg.cholesky(np.eye(K) + (G.T / D_2).dot(G))
        z_1 = y / D_2
        t_1 = scipy.linalg.solve_triangular(L_v_inv.T, scipy.linalg.solve_triangular(L_v_inv, G.T.dot(z_1), lower=True), lower=False)
        p = z_1 - (V.T / D_2).T.dot(t_1)
        w_2 = -(z - A.dot(p))
        
        # then calculate d_v
        V_hat = (A / D_2).dot(G).dot(L_v_inv.T)
        L = np.linalg.cholesky((A/D_2).dot(A.T))
        V_bar = scipy.linalg.solve_triangular(L, V_hat, lower=True)
        z_2 = scipy.linalg.solve_triangular(L, w_2, lower=True)
        t_2 = np.linalg.solve(eye_K - V_bar.T.dot(V_bar), V_bar.dot(z_2))
        d_v = scipy.linalg.solve_triangular(L.T, z_2 + V_bar.dot(t_2), lower=False)

        # solve for d_x
        y_3 = y - A.T.dot(d_v)
        z_3 = y_3 / D_2
        t_3 = scipy.linalg.solve_triangular(L_inv.T, scipy.linalg.solve_triangular(L_v_inv, V.T.dot(z_3), lower=True), lower=False)
        d_x = z_3 - (V.T / D_2).T.dot(t_3)

        # solve for d_lam
        d_lam_lower = -(lam_lower + ((ones_N/t) - (lam_lower*d_x)) / f_lower)
        d_lam_upper = -(lam_upper + ((ones_N/t) + (lam_upper*d_x)) / f_upper)

        # choose stepsize

        # first find largest s such that lam will still be positive
        d_lam_lower_neg = d_lam_lower < 0
        s_lam_lower = np.min((0 - lam_lower[d_lam_lower_neg]) / d_lam_lower[d_lam_lower_neg])
        d_lam_upper_neg = d_lam_upper < 0
        s_lam_upper = np.min((0 - lam_upper[d_lam_upper_neg]) / d_lam_upper[d_lam_upper_neg])
        s_lam = min(1., s_lam_lower, s_lam_upper)

        # further decrease s such that all ineq constraints are met
        d_x_pos = d_x > 0
        s_x_lower = np.min(-f_x_lower[d_x_pos] / d_x[d_x_pos])
        s_x_upper = np.min(-f_x_lower[d_x_pos] / d_x[d_x_pos])
        s_x = min(s_x_lower, s_x_upper)

        # now minimize the residual norm
        s = 0.99 * min(s_lam, s_x)
        r_norm = get_r_norm(x, lam_lower, lam_upper, v, t)
        while True:
            x_new, lam_lower_new, lam_upper_new, v_new, = x + (s * d_x), lam_lower + (s * d_lam_lower), lam_upper + (s * d_lam_upper), v + (s * d_v)
            if get_r_norm(x_new, lam_lower_new, lam_upper_new, v_new, t) <= (1. - (alpha * s)) * r_norm:
                break
            s *= beta

        x, lam_lower, lam_upper, v_new = x_new, lam_lower_new, lam_upper_new, v_new

        # compute surrogate gap
        gap = -(get_f_lower(x).dot(lam_lower) + get_f_upper(x).dot(lam_upper))
        if np.linalg.norm(get_r_primal(x)) < eps_primal and np.linalg.norm(get_r_dual(x, lam_lower, lam_upper, v)) < eps_dual and gap < eps:
            break

    return x, lam_lower, lam_upper, v_new
