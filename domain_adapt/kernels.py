import numpy as np
import pdb
import sklearn.metrics.pairwise as pairwise
import pdb

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
