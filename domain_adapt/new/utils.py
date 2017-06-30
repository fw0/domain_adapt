#import numpy as np
import autograd.numpy as np
import pandas as pd
from IPython.display import display_pretty, display_html
import python_utils.python_utils.basic as basic
import matplotlib.pyplot as plt
import pdb


def ortho(P):
    ans = np.zeros(P.shape)
    x_dim, z_dim = P.shape
    for j in xrange(z_dim):
        temp = P[:,j] - np.dot(ans, np.dot(ans.T, P[:,j]))
        temp = temp / np.linalg.norm(temp)
        ans[:,j] = temp
    return ans


def get_gaussian_K_helper(sigma, xs1, xs2):

    if False:
        return np.dot(xs1, xs2.T)

    diff = xs1[:,np.newaxis,:] - xs2[np.newaxis,:,:]
    norms = np.sum(diff * diff, axis=2)

    K = np.exp(-1. * norms / (2 * (sigma**2)))

    return K


def get_gaussian_K(sigma, xs1, xs2, nystrom=True):


    if len(xs1.shape) == 1:
        xs1 = xs1.reshape((len(xs1),1))

    if len(xs2.shape) == 1:
        xs2 = xs2.reshape((len(xs2),1))

    if not nystrom:
        return get_gaussian_K_helper(sigma, xs1, xs2)

    else:
                
#        print xs1.shape, xs2.shape
#        pdb.set_trace()
        try:
            assert (xs1 == xs2).all()
        except:
            pdb.set_trace()
        num_cols = 50
        total_num = len(xs1)# + len(xs2)
#        include_prob = num_cols / float(total_num)
#        include = np.random.uniform(size=total_num) < include_prob
#        included = xs1[include]
#        excluded = xs1[~include]
#        every = total_num / num_cols
        included = xs1[0:num_cols]
        excluded = xs1[num_cols:]
        W = get_gaussian_K_helper(sigma, included, included)
        K21 = get_gaussian_K_helper(sigma, excluded, included)
        C = np.concatenate((W, K21), axis=0)
        eps = 0.1
        W_inv = np.linalg.inv(W + (eps * np.eye(len(W))))
        return (C, W_inv, C.T)


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

class optimizer_logger(object):

    def __init__(self, f, df_dx, verbose=False, info_f=None):
        self.f, self.df_dx, self.verbose, self.info_f = f, df_dx, verbose, info_f
        self.l = []
        self.counter = 0

    def __call__(self, x):
        #print x
        val = self.f(x)
        info = {'f':val, 'grad_norm':np.linalg.norm(self.df_dx(x))}
#        print self.verbose, self.counter, self.f
#        print 'step', self.counter, val
        if self.verbose and ((self.counter % self.verbose) == 0):
            print 'step', self.counter, val
            import sys
            sys.stdout.flush()
            if not self.info_f is None:
                self.info_f(x)
            #print self.counter, info
        self.counter += 1
        self.l.append(info)

    def plot(self):
        df = pd.DataFrame(self.l)
        fig = plt.figure()
        f_ax = fig.add_subplot(2,1,1)
        f_ax.plot(df.index, df['f'])
        f_ax.set_ylabel('$f$')
#        f_ax.set_ylim((0,None))
        grad_norm_ax = fig.add_subplot(2,1,2)
        grad_norm_ax.plot(df.index, df['grad_norm'])
        grad_norm_ax.set_ylabel('$|df\_dx|$')
        grad_norm_ax.set_xlabel('iteration')
        grad_norm_ax.set_xlim((0,None))
        basic.display_fig_inline(fig)
        
    def display_df(self):
        display_html(pd.DataFrame(self.l).to_html(), raw=True)
