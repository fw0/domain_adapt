#import numpy as np
import autograd.numpy as np
import pandas as pd
from IPython.display import display_pretty, display_html
import python_utils.python_utils.basic as basic
import matplotlib.pyplot as plt

def ortho(P):
    ans = np.zeros(P.shape)
    x_dim, z_dim = P.shape
    for j in xrange(z_dim):
        temp = P[:,j] - np.dot(ans, np.dot(ans.T, P[:,j]))
        temp = temp / np.linalg.norm(temp)
        ans[:,j] = temp
    return ans

def get_gaussian_K(sigma, xs1, xs2):

    if len(xs1.shape) == 1:
        xs1 = xs1.reshape((len(xs1),1))

    if len(xs2.shape) == 1:
        xs2 = xs2.reshape((len(xs2),1))
                
    diff = xs1[:,np.newaxis,:] - xs2[np.newaxis,:,:]
    norms = np.sum(diff * diff, axis=2)

    K = np.exp(-1. * norms / (2 * (sigma**2)))

    return K

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
        if self.verbose and (self.counter % self.verbose) == 0:
            print 'step', self.counter, val
            if not self.info_f is None:
                self.info_f(x)
        self.counter += 1
        self.l.append({'f':val, 'grad_norm':np.linalg.norm(self.df_dx(x))})

    def plot(self):
        df = pd.DataFrame(self.l)
        fig = plt.figure()
        f_ax = fig.add_subplot(2,1,1)
        f_ax.plot(df.index, df['f'])
        f_ax.set_ylabel('$f$')
        f_ax.set_ylim((0,None))
        grad_norm_ax = fig.add_subplot(2,1,2)
        grad_norm_ax.plot(df.index, df['grad_norm'])
        grad_norm_ax.set_ylabel('$|df\_dx|$')
        grad_norm_ax.set_xlabel('iteration')
        grad_norm_ax.set_xlim((0,None))
        basic.display_fig_inline(fig)
        
    def display_df(self):
        display_html(pd.DataFrame(self.l).to_html(), raw=True)