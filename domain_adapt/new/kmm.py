import numpy as np
import qp
import utils

def get_kmm_Gh(w_max, eps, num_train, num_test):
    w_max = float(w_max)
    eps = float(eps)
    G_gt_0 = -np.eye(num_train)
    h_gt_0 = np.zeros(num_train)
    G_lt_B_max = np.eye(num_train)
    h_lt_B_max = np.ones(num_train) * w_max
    G_B_sum_lt = np.ones(num_train, dtype=float)
    h_B_sum_lt = (1+eps) * float(num_train) * np.ones(1)
    G_B_sum_gt = -np.ones(num_train, dtype=float)
    h_B_sum_gt = -(1-eps) * float(num_train) * np.ones(1)
    G = np.vstack((G_gt_0,G_lt_B_max,G_B_sum_lt,G_B_sum_gt))
    (h_gt_0,h_lt_B_max,h_B_sum_lt,h_B_sum_gt)
    h = np.hstack((h_gt_0,h_lt_B_max,h_B_sum_lt,h_B_sum_gt))
    return G, h

def get_cvxopt_KMM_ws(w_max, eps, K_train_train, K_train_test):

    # get kappa
    num_train = K_train_train.shape[0]
    num_test = K_train_test.shape[1]
    kappa = -(float(num_train)/num_test) * np.sum(K_train_test, axis=1)

    # get cvxopt constraints
    G,h = get_kmm_Gh(w_max, eps, num_train, num_test)

    # solve
    return qp.cvxopt_solver(K_train_train, kappa, G, h)


def get_cvxopt_KMM_ws_sigma_median_distance(w_max, eps, xs_train, xs_test):
    sigma = utils.median_distance(np.concatenate((xs_train, xs_test), axis=0), np.concatenate((xs_train, xs_test), axis=0))
    K_train_train = utils.get_gaussian_K(sigma, xs_train, xs_train)
    K_train_test = utils.get_gaussian_K(sigma, xs_train, xs_test)
    return get_cvxopt_KMM_ws(w_max, eps, K_train_train, K_train_test)
