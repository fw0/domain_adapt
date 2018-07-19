import autograd.numpy as np
import autograd
import copy, itertools, pdb
import kmm, qp, utils, optimizers
import scipy.optimize, scipy
#from scipy.optimize import optimize
#import autograd.scipy as scipy
import python_utils.python_utils.basic as basic
from autograd.core import primitive
import collections, pandas as pd
import time
#import autograd.numpy as np
from autograd.extend import primitive, defvjp


NA = np.newaxis

@primitive
def logsumexp(x):
    """Numerically stable log(sum(exp(x)))"""
    max_x = np.max(x)
    return max_x + np.log(np.sum(np.exp(x - max_x)))

def logsumexp_vjp(g, ans, vs, gvs, x):
    return np.full(x.shape, g) * np.exp(x - np.full(x.shape, ans))

defvjp(logsumexp, logsumexp_vjp)

def project(xs, B):
    return np.concatenate((np.dot(xs,B), np.ones((len(xs),1))), axis=1)

class fxn(object):

    def __init__(self, forward_prop=None, backward_prop=None, _val=None, _grad=None, _val_and_grad=None, _hessian_vector_product=None, _vector_jacobian_product=None, **kwargs):
        if not (forward_prop is None): self.forward_prop = forward_prop
        if not (backward_prop is None): self.backward_prop = backward_prop
        if not (_val is None): self._val = _val
        if not (_grad is None): self._grad = _grad
        if not (_val_and_grad is None): self._val_and_grad = _val_and_grad
        if not (_hessian_vector_product is None): self._hessian_vector_product = _hessian_vector_product
        if not (_vector_jacobian_product is None): self._vector_jacobian_product = _vector_jacobian_product

    def forward_prop(self, *args):
        """
        returns (temps, val)
        """
        return None, self._val(*args)

    def backward_prop(self, args, temps, val, care_argnums):
        """
        returns tuple of gradients of same length as args
        """
        try:
            return self._grad(*args, care_argnums=care_argnums)
        except NotImplementedError:
            pass
        raise NotImplementedError

    def _val(self, *args):
        """
        returns just fxn value, takes in single tuple of args
        """
        raise NotImplementedError

    def _grad(self, *args, **kwargs):
        """
        returns just grad, does not take in temps, takes in single tuple of args
        """
        raise NotImplementedError

    def _val_and_grad(self, *args, **kwargs):
        return self._val_and_grad_default(*args, **kwargs)
    
    def _val_and_grad_default(self, *args, **kwargs):
        """
        assumed to be more efficient than calling _val and _grad separately
        """
        care_argnums = kwargs.get('care_argnums')#, None)
        try:
            temps, val = self.forward_prop(*args)
            grad = self.backward_prop(args, temps, val, care_argnums)
            return val, grad
        except NotImplementedError:
            raise NotImplementedError
    
    def val(self, *args):
        try:
            temps, val = self.forward_prop(*args)
            if np.isnan(val).any():
                pdb.set_trace()
            return val
        except NotImplementedError:
            pass
        try:
            ans = self._val(*args)
            if np.isnan(ans).any():
                pdb.set_trace()
        except NotImplementedError:
            assert False
        try:
            val, grad = self._val_and_grad(*args)
            pdb.set_trace()
#            assert False
            return val
        except NotImplementedError:
            pass
        raise NotImplementedError

    def __call__(self, *args):
        return self.val(*args)

#    @basic.timeit('grad')
    def grad(self, *args, **kwargs):
        # fix: possibly put _val_and_grad first, so that my custom forward/backward passes are called b4 autograd, like in the case of quadratic_objective
        care_argnums = kwargs.get('care_argnums', range(len(args)))
        self.check_care_argnums(args, care_argnums)
        try:
            ans = self._grad(*args, care_argnums=care_argnums)
            if np.isnan(ans).any():
                pdb.set_trace()
#            print self, self._val
            return ans
        except NotImplementedError:
#            pdb.set_trace()
#            self._grad(*args, care_argnums=care_argnums)
            pass
        try:
            val, grad = self._val_and_grad(*args, care_argnums=care_argnums)
            if np.isnan(grad).any():
                pdb.set_trace()
#            print self, self._val
            return grad
        except NotImplementedError:
            pdb.set_trace()
            val, grad = self._val_and_grad(*args, care_argnums=care_argnums)
            pass
        pdb.set_trace()
        raise NotImplementedError

    def val_and_grad(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums', range(len(args)))
        self.check_care_argnums(args, care_argnums)
        try:
            ans = self._val_and_grad(*args, care_argnums=care_argnums)
#            print self, self._val
            return ans
        except NotImplementedError:
            pass
        try:
            ans = self.val(*args), self.grad(*args, care_argnums=care_argnums)
#            print self, self._val
            return ans
        except NotImplementedError:
            pass
        raise NotImplementedError

    def grad_check(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums', range(len(args)))
        self.check_care_argnums(args, care_argnums)
        delta = 0.01
        #delta = 0.00001
        tol = 1.
        val, anal_grad = basic.timeit('actual grad')(self.val_and_grad)(*args, care_argnums=care_argnums)
        try:
            val_shape = val.shape
        except AttributeError:
            val_shape = ()
        anal_grad = (anal_grad,) if len(care_argnums) == 1 else anal_grad # special
#        print anal_grad
#        print len(anal_grad), len(care_argnums)
        assert len(anal_grad) == len(care_argnums)
        for (i, a_anal_grad) in enumerate(anal_grad):
            a_arg = args[care_argnums[i]]
            a_check_grad = np.zeros(val_shape+a_arg.shape)
            for index in np.ndindex(*a_arg.shape):
                old = a_arg[index]
                a_arg[index] += delta
                if len(index) == len(a_check_grad.shape):
                    a_check_grad[index] = (self.val(*args) - val) / delta
                else:
                    a_check_grad[(slice(None),)+index] = (self.val(*args) - val) / delta
                a_arg[index] = old
            print 'anal_grad', a_anal_grad.shape
            print a_anal_grad
            print 'numerical_grad', a_check_grad.shape
            print a_check_grad
            #print 'zip'
            #print zip(a_anal_grad, a_check_grad)
            print 'error', np.linalg.norm(a_anal_grad - a_check_grad)
            #assert np.linalg.norm(a_anal_grad - a_check_grad) < tol
            #pdb.set_trace()

    def _hessian_vector_product(self, v, p_argnum, x_argnum, *args):
        raise NotImplementedError

    def hessian_vector_product(self, v, p_argnum, x_argnum, *args):
        return self._hessian_vector_product(self, v, p_argnum, x_argnum, *args)

    def _vector_jacobian_product(self, v, x_argnum, *args):
        def dotted(*_args):
            return np.dot(v, self._val(*_args))
        return fxn.autograd_fxn(_val=dotted).grad(*args, care_argnums=(x_argnum,))

#        np.sum(v[(slice(None),)+tuple([np.newaxis for i in xrange(len(p.shape))])] * self._val(*args)
        
    def vector_jacobian_product(self, v, x_argnum, *args):
        return self._vector_jacobian_product(v, x_argnum, *args)

    def check_care_argnums(self, args, care_argnums):
        return True

    @classmethod
    def autograd_fxn(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
        inst._grad = get_autograd_grad(inst.val)
        inst._val_and_grad = get_autograd_val_and_grad(inst.val)
        return inst

    @classmethod
    def wrap_primitive(cls, f):
        wrapped = primitive(f)
#        def fxn_vjps(argnum, g, ans, vs, gvs, *args):
#            return np.dot(g, f.grad(*args, care_argnums=(argnum,))) # fix: f assumed to output 1d vector
        max_argnum = 20
        def grad(argnum, ans, *args):
            return lambda g: np.dot(g, f.grad(*args, care_argnums=(argnum,)))
        import functools
        defvjp(wrapped, *[functools.partial(grad, argnum) for argnum in xrange(max_argnum)])
#        wrapped.defvjps(fxn_vjps, range(max_argnum))
        return wrapped

    def __repr__(self):
        return str(self.__class__) + str(self.__dict__)
    
    @property
    def __name__(self):
        return repr(self)
        
def get_autograd_horse(num_args, care_argnums, f):

    _nocare_argnums = [i for i in range(num_args) if (not i in care_argnums)]
        
    def horse(_care_args, _nocare_args):
        _args = [None for i in xrange(num_args)]
        if len(care_argnums) == 1:
            _args[care_argnums[0]] = _care_args
        else:
            for i in xrange(len(care_argnums)):#(i,_arg) in zip(care_argnums, _care_args):
                _args[care_argnums[i]] = _care_args[i]
        for i in xrange(len(_nocare_argnums)):#(i,_arg) in zip(_nocare_argnums, _nocare_args):
            _args[_nocare_argnums[i]] = _nocare_args[i]
        return f(*_args)

    return horse


def get_autograd_grad(f):

    def grad(*args, **kwargs):
        care_argnums = kwargs.get('care_argnums')#, None)
        horse = get_autograd_horse(len(args), care_argnums, f)
        care_args = [args[i] for i in care_argnums] if len(care_argnums) != 1 else args[care_argnums[0]] # special case
        nocare_args = [args[i] for i in xrange(len(args)) if not (i in care_argnums)]
#        pdb.set_trace()
        ans = autograd.jacobian(horse)(care_args, nocare_args)
#        ans = autograd.grad(horse)(care_args, nocare_args)
        return ans
        #return ans if len(ans) != 1 else ans[0]

    return grad


def get_autograd_val_and_grad(f):

    def val_and_grad(*args, **kwargs):
        care_argnums = kwargs.get('care_argnums')#, None)
        horse = get_autograd_horse(len(args), care_argnums, f)
        care_args = [args[i] for i in care_argnums] if len(care_argnums) != 1 else args[care_argnums[0]] # special case
        nocare_args = [args[i] for i in xrange(len(args)) if not (i in care_argnums)]
        val, grad = autograd.value_and_grad(horse)(care_args, nocare_args)
        return val, grad
        #return val, grad if len(grad) != 1 else grad[0]

    return val_and_grad

def weighted_lsqr_loss(B, xs, ys, ws, c, add_reg=False):
    b_opt = weighted_lsqr_b_opt(B, xs, ys, ws, c)
    return weighted_squared_loss_given_b_opt(B, xs, ys, ws, b_opt, c, add_reg)
#    N = xs.shape[0]
#    W = ws * np.eye(N)
#    us = np.dot(xs, B)
#    ys_hat = np.dot(us, b_opt)
#    error = ys - ys_hat
#    return np.dot(ws, error * error) + (np.dot(b_opt, b_opt) * c)
#    return np.sum(np.dot(ys*ws - np.dot(ys*ws,np.dot(np.dot(us,np.linalg.inv(np.dot(us.T*ws,us))),us.T*ws)),ys))


def weighted_lsqr_b_opt(B, xs, ys, ws, c):
#    print ws.shape, xs.shape, ys.shape
    us = project(xs, B)
    ys_prime = ys * (ws**0.5)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    us_prime = us * ((ws**0.5)[:,np.newaxis])
#    pdb.set_trace()
    #c = 0.01 # fix
    I_minus = np.diag(np.concatenate((np.ones(us.shape[1]-1), np.zeros(1))))
    try:
#        b_opt_lsqr = np.linalg.solve(np.dot(us_prime.T, us_prime) + len(xs)*c*np.eye(us_prime.shape[1]), np.dot(us_prime.T, ys_prime))
        b_opt_lsqr = np.linalg.solve(np.dot(us_prime.T, us_prime) + len(xs)*c*I_minus, np.dot(us_prime.T, ys_prime))
    except:
        pdb.set_trace()
#    print ws
    #b_opt = np.dot(np.dot(np.linalg.inv(np.dot(us_prime.T, us_prime) + len(xs)*c*np.eye(us_prime.shape[1])), us_prime.T), ys_prime) # fix: get rid of inverse, solve linear system instead
    #print b_opt_lsqr, b_opt
    #try:
    #    assert np.linalg.norm(b_opt_lsqr-b_opt) < 0.001
    #except:
    #    pdb.set_trace()
    #pdb.set_trace()
    return b_opt_lsqr

def ll_given_b_opt_ll(b_opt_ll, losses, xs_train, B):
    error = np.dot(project(xs_train,B), b_opt_ll) - losses
#    error = np.dot(xs_train, np.dot(B, b_opt_ll)) - losses
    return np.dot(error, error) / len(losses)
    

def unweighted_lsqr_b_opt(B, xs, ys, c):
    us = project(xs, B)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    #print xs.shape
    I_minus = np.diag(np.concatenate((np.ones(us.shape[1]-1), np.zeros(1))))
    b_opt_lsqr = basic.timeit('solve')(np.linalg.solve)(np.dot(us.T, us) + len(xs)*c*I_minus, np.dot(us.T, ys))
#    print b_opt_lsqr
    return b_opt_lsqr


def weighted_lsqr_b_opt_given_b_logreg(B, xs_train, xs_test, ys_train, b_logreg, c_lsqr, sigma, scale_sigma, max_ratio=5.):

#    scale_sigma = False

    ws = b_to_logreg_ratios_wrapper(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.)

    us = project(xs_train, B)
    ys_prime = ys_train * (ws**0.5)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    us_prime = us * ((ws**0.5)[:,np.newaxis])
#    pdb.set_trace()
    #c = 0.01 # fix

    b_opt_lsqr = np.linalg.solve(np.dot(us_prime.T, us_prime) + len(xs_train)*c_lsqr*np.eye(us_prime.shape[1]), np.dot(us_prime.T, ys_prime))
    b_opt = np.dot(np.dot(np.linalg.inv(np.dot(us_prime.T, us_prime) + len(xs_train)*c_lsqr*np.eye(us_prime.shape[1])), us_prime.T), ys_prime) # fix: get rid of inverse, solve linear system instead
    assert np.linalg.norm(b_opt_lsqr-b_opt) < 0.001
    return b_opt_lsqr


def weighted_squared_loss_given_b_opt(B, xs, ys, ws, b_opt, c, add_reg=False):
    us = project(xs, B)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    ys_hat = np.dot(us, b_opt)
    error = ys - ys_hat
    if add_reg:
        return (np.dot(ws, error * error) / len(xs)) + (np.dot(b_opt, b_opt) * c)
    else:
        return np.dot(ws, error * error) / len(xs)


def weighted_squared_loss_given_b_opt_and_b_logreg(B, xs_train, xs_test, ys_train, b_logreg, b_opt, c_lsqr, sigma, scale_sigma, max_ratio=5., add_reg=False):

#    scale_sigma = False

    ws = b_to_logreg_ratios_wrapper(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.)

    return weighted_squared_loss_given_b_opt(B, xs_train, ys_train, ws, b_opt, c_lsqr, add_reg)

    us = project(xs_train, B)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    ys_hat = np.dot(us, b_opt)
    error = ys_train - ys_hat
    return np.dot(ws, error * error) + (np.dot(b_opt, b_opt) * c_lsqr)



def weight_reg_given_b_logreg(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.):
    ws = b_to_logreg_ratios_wrapper(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.)
    ans = np.dot(ws.T, ws) / len(xs_train)
    #print 'weight_reg', ans
    return ans


def weighted_squared_loss_given_B(B, xs, ys, c, ws=None):
    # B is assumed to be 1-d projection
    assert False
    assert len(B.shape) == 1
    ws = np.ones(len(xs)) if ws is None else ws
    ys_hat = np.dot(xs, B)
    error = ys - ys_hat
    ans = np.dot(ws, error * error) + (c*np.dot(B,B))
    #preds = np.dot(xs, B)
    #print np.dot(ws, error * error),  (c*np.dot(B,B)), c, np.max(preds) , np.min(preds), 'ok'
#    ans = np.dot(ws, error * error) + (c*np.sum(np.absolute(B)))
#    pdb.set_trace()
    return ans


def b_opt_to_squared_losses(B, xs, ys, b_opt):
    us = project(xs, B)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    ys_hat = np.dot(us, b_opt)
    error = ys - ys_hat
    return error * error

def B_to_squared_losses(B, xs, ys):
    assert len(B.shape) == 1
    ys_hat = np.dot(xs, B)
    error = ys - ys_hat
    return error * error


def weighted_lsqr_loss_given_f(B, xs, ys, f, ws, c):
    assert False
    diff = np.dot(xs, np.dot(B, f)) - ys
    return np.dot(ws, diff * diff) + (c * np.dot(f,f))


def b_to_logreg_ratios(b, xs_train, xs_test, sigma, B, max_ratio=5.):
    scale_sigma = False
    return b_to_logreg_ratios_wrapper(b, xs_train, xs_test, sigma, scale_sigma, B, max_ratio)


def b_to_logreg_ratios_scale_sigma(b, xs_train, xs_test, sigma, B, max_ratio=5.):
    scale_sigma = True
    return b_to_logreg_ratios_wrapper(b, xs_train, xs_test, sigma, scale_sigma, B, max_ratio)


def b_to_logreg_ratios_wrapper(b, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.):
    if scale_sigma:
        assert len(B.shape) == 1
        sigma = sigma / np.linalg.norm(B)
    us_train = project(xs_train, B)
    us_test = project(xs_test, B)
    us = np.concatenate((us_train, us_test))
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))

    K = utils.get_gaussian_K(sigma, us, us)
    logits = nystrom_dot(K, b) # fix nystrom

    logits_train = logits[:len(xs_train)]
    ps_train = 1 / (1+np.exp(-logits_train))
    #return ps_train
    ratios_train = ps_train / (1.-ps_train)
    #ratio_max = 5.
#    ratio_max = 2.
    ratios_train = np.minimum(ratios_train, np.ones(len(ratios_train)) * max_ratio)
#    print max(ratios_train), 'max pre'
    ratios_train = (ratios_train / np.sum(ratios_train)) * len(xs_train)
#    print max(ratios_train), 'max post'
    #print b, 'b'
    #print np.sum(b), 'b_sum'
    #print ratios_train, 'ratios_train'
    return ratios_train


def lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio):
    us_train = project(xs_train, B)
    us_basis = project(xs_basis, B)
    train_basis_vals = utils.get_gaussian_K(sigma, us_train, us_basis, nystrom=False)
    ans = np.dot(train_basis_vals, lsif_alpha)
    ans = np.maximum(0.01, ans)
    #ans = np.maximum(0, ans)
    ans = (ans / np.sum(ans)) * len(xs_train)
    if False: # fix?
        ans = np.minimum(max_ratio, ans)
        ans = (ans / np.sum(ans)) * len(xs_train)
        ans = np.minimum(max_ratio, ans)
        ans = (ans / np.sum(ans)) * len(xs_train)
#    print ans[0:10], 'ws'
    if (np.isnan(ans)).any():
        pdb.set_trace()
#    print ans.sum(), 'sum', len(xs_train)
#    print ans, lsif_alpha
    return ans


def weighted_squared_loss_given_b_opt_and_lsif_alpha(B, xs_train, xs_basis, ys_train, lsif_alpha, b_opt, c_lsqr, sigma, max_ratio=5., add_reg=False):

#    scale_sigma = False
    ws = lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)
#    ws = b_to_logreg_ratios_wrapper(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.)

    return weighted_squared_loss_given_b_opt(B, xs_train, ys_train, ws, b_opt, c_lsqr, add_reg)


def weighted_lsqr_b_opt_given_lsif_alpha(B, xs_train, xs_basis, ys_train, lsif_alpha, c_lsqr, sigma, max_ratio=5.):

#    scale_sigma = False

    ws = lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)
#    ws = b_to_logreg_ratios_wrapper(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.)

    us = project(xs_train, B)
    ys_prime = ys_train * (ws**0.5)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    us_prime = us * ((ws**0.5)[:,np.newaxis])
#    pdb.set_trace()
    #c = 0.01 # fix
    b_opt_lsqr = np.linalg.solve(np.dot(us_prime.T, us_prime) + len(xs_train)*c_lsqr*np.eye(us_prime.shape[1]), np.dot(us_prime.T, ys_prime))
    b_opt = np.dot(np.dot(np.linalg.inv(np.dot(us_prime.T, us_prime) + len(xs_train)*c_lsqr*np.eye(us_prime.shape[1])), us_prime.T), ys_prime) # fix: get rid of inverse, solve linear system instead
    #print b_opt_lsqr, b_opt
    try:
        assert np.linalg.norm(b_opt-b_opt_lsqr) < 0.001
    except:
        print sigma
        pdb.set_trace()
    #print b_opt_lsqr, b_opt
#    pdb.set_trace()
    return b_opt_lsqr


def squared_losses(xs, ys, b):
    errors = ys - np.dot(xs, b)
    return errors * errors
    print errors.shape, 'gg'
    return np.dot(errors, errors)

def logistic_regression_losses(xs, ys, b):
    logits = np.dot(xs, b)
    try:
        losses = np.log(1 + np.exp(-ys * logits))
    except:
        pdb.set_trace()
#    print losses.shape, 'gg'
    return losses

def logistic_regression_loss(B, xs, ys, ws, b, c, add_reg):
    us = project(xs, B)
    losses = logistic_regression_losses(us, ys, b)
#    logits = np.dot(us, b)
#    losses = np.log(1 + np.exp(-ys * logits))
#    print c, b, 'hh'
    return np.mean(ws * losses, axis=0) + (np.dot(b[:-1],b[:-1])*c) # fixed

def logistic_regression_objective_helper(B, xs, ys, ws, c, b):
    #print B
    #print B.shape
    #print 'asdfgg'
#    pdb.set_trace()
    return logistic_regression_loss(B, xs, ys, ws, b, c, True)

def weight_reg_given_lsif_alpha(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio):
    ws = lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)
    return np.dot(ws, ws)


def expected_conditional_PE_dist(full_ws, ws, pseudocount):
    N = len(full_ws)
    assert len(full_ws) == len(ws)
#    pseudocount = 1.
    #ws = ws + pseudocount
    #ws = ws / N
#    print full_ws, 'full'
#    print ws, 'not full'
#    print zip(full_ws, ws)
#    print np.sum(((1./N) * (full_ws**2) / ws), axis=0) - 1., 'distance'
#    pdb.set_trace()
    ans = np.sum(((1./N) * ((full_ws**2)+pseudocount) / (ws+pseudocount)), axis=0) - 1.
#    print ans, 'CE'
#    pdb.set_trace()
    return ans

def weight_reg(ws):
    return np.dot(ws, ws) / len(ws)


def nystrom_dot(K, b):
    if isinstance(K, tuple):
        logits = b
        for M in reversed(K):
#            print M.shape, logits.shape
            logits = np.dot(M, logits)
    else:
        logits = np.dot(K, b) # fix nystrom
    return logits


def logreg_loss(xs, ys, b, ws=None): # fix nystrom
    assert ys.max() == 1 and ys.min() == -1
    if isinstance(xs, tuple):
        num_data = xs[0].shape[0]
    else:
        num_data = xs.shape[0]
    ws = np.ones(num_data) if ws is None else ws

    K = xs
    logits = nystrom_dot(xs, b)
#    pdb.set_trace()

    losses = np.log(1 + np.exp(-ys * logits))
#    pdb.set_trace()
    assert len(losses.shape) == 1
    ans = np.mean(ws * losses, axis=0)
#    pdb.set_trace()
    return ans



    


class objective(fxn):

    def __init__(self, _arg_shape=None, *args, **kwargs):
        if not (_arg_shape is None):
            self._arg_shape = _arg_shape
        fxn.__init__(self, *args, **kwargs)
        
    
    def arg_shape(self, *args):
        return self._arg_shape(*args)
#        raise NotImplementedError


class logistic_regression_objective(objective):

    def _val(self, B, xs_train, ys_train, ws_train, c_pred, b):
        return logistic_regression_objective_helper(B, xs_train, ys_train, ws_train, c_pred, b)

    def _grad(self, *args, **kwargs):
        return fxn._grad(self, *args, **kwargs)
    
    def arg_shape(self, B, xs_train, ys_train, ws_train, c_pred):
        return (B.shape[1],)

class logreg_ratio_objective(objective):

    def __init__(self, *args, **kwargs):
        self.scale_sigma = kwargs.get('scale_sigma', False)
        objective.__init__(self, *args, **kwargs)
    
    def arg_shape(self, xs_train, xs_test, sigma, B, logreg_c):
        #pdb.set_trace()
        #return (B.shape[1],)
        return (xs_train.shape[0] + xs_test.shape[0],)
        #return B.shape[1]
    
#    @basic.timeit
    def _grad(self, *args, **kwargs):
        return fxn._grad(*args, **kwargs)

#    @basic.timeit
    def _val(self, xs_train, xs_test, sigma, B, logreg_c, b):
        if self.scale_sigma:
#            print B
            assert len(B.shape) == 1
            sigma = sigma / np.linalg.norm(B)
#        pdb.set_trace()
        us_train = project(xs_train, B)
        us_test = project(xs_test, B)
        us = np.concatenate((us_train, us_test))
        if len(us.shape) == 1:
            us = us.reshape((len(us),1))
        K = utils.get_gaussian_K(sigma, us, us)
        #zs = np.hstack((np.ones(len(xs_train)), np.zeros(len(xs_test))))
        zs = np.hstack((-np.ones(len(xs_train)), np.ones(len(xs_test))))
#        C = 1.
#        print 1 / (1. + np.exp(-np.dot(K, b))), 'ps'
        return logreg_loss(K, zs, b) + (logreg_c * np.dot(b,b)) # fix nystrom
#        return np.sum(np.log(1 + np.exp(-zs * np.dot(K, b))), axis=0) + (C * np.dot(b,b))

class quad_objective(fxn):
    """
    assumes last argument is the one optimized over
    """
    def Pq(self, *args):
        raise NotImplementedError
        
    def forward_prop(self, *args):
        P, q = self.Pq(*args[0:-1]) # hard
        x = args[-1] # hard
        try:
            val = (0.5 * np.dot(x.T, nystrom_dot(P, x))) + np.dot(q, x)
        except:
            pdb.set_trace()
        return (P,q), val
    
    def backward_prop(self, args, (P,q), val, care_argnums):
        assert care_argnums == (len(args)-1,) # hard
        x = args[-1] # hard
        return np.dot(P, x) + q

    def _val_and_grad(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums')
        if care_argnums == (len(args)-1,): # hard
            return self._val_and_grad_default(*args, **kwargs)
        else:
            return self._val_and_grad_horse(*args, **kwargs)

    @classmethod
    def autograd_quad_objective(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
#        inst._grad_horse = get_autograd_grad(inst.val)
        inst._val_and_grad_horse = get_autograd_val_and_grad(inst.val)
        return inst
            

class new_kmm_objective(quad_objective):
    
    def _Pq(self, xs_train, xs_test, sigma, B, w_max, eps, nystrom):
        us_train = project(xs_train, B)
        us_test = project(xs_test, B)
        if len(us_train.shape) == 1:
            us_train = us_train.reshape((len(us_train),1))
        if len(us_test.shape) == 1:
            us_test = us_test.reshape((len(us_test),1))

        num_train = len(us_train)
        num_test = len(us_test)
        K_train_train = utils.get_gaussian_K(sigma, us_train, us_train, nystrom=nystrom)
        K_train_test = utils.get_gaussian_K(sigma, us_train, us_test, nystrom=nystrom)
#        kappa = -(float(num_train)/num_test) * np.sum(K_train_test, axis=1)
        kappa = -(float(num_train)/num_test) * nystrom_dot(K_train_test, np.ones(num_test))
        return K_train_train, kappa

    def Pq(self, xs_train, xs_test, sigma, B, w_max, eps):
        return self._Pq(xs_train, xs_test, sigma, B, w_max, eps, nystrom=False)


class nystrom_kmm_objective(new_kmm_objective):
  
    def Pq(self, xs_train, xs_test, sigma, B, w_max, eps):
        # calls nystrom getter, multiplies
        (C,W_inv,C_T), kappa = self._Pq(xs_train, xs_test, sigma, B, w_max, eps, nystrom=True)
        return np.dot(np.dot(C,W_inv), C_T), kappa

def N_eff(ws):
    return ws.sum()**2 / np.dot(ws, ws)        


def get_least_squares_ws(sigma, c_lsif, xs_train, xs_test, max_ratio=5.):
    xs_basis = xs_test[0:100]
    B = np.eye(xs_train.shape[1])
    lsif_alpha = least_squares_lsif_alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, 5., True)
    ws_train = lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)
    return ws_train


def sklearn_ridge_fitter(alphas=None, cheat=False, use_train_oos=False):

    def fitter(xs_train, xs_test, ys_train, ys_test=None, xs_train_oos=None, xs_test_oos=None, ys_train_oos=None, ys_test_oos=None):

        if use_train_oos:
            xs_train = np.concatenate((xs_train, xs_train_oos), axis=0)
            ys_train = np.concatenate((ys_train, ys_train_oos))

        from sklearn.linear_model import RidgeCV
        _fitter = RidgeCV(alphas=alphas)
        if not cheat:
            _fitter.fit(xs_train, ys_train)
#            print len(xs_train), 'no cheat'
        else:
            _fitter.fit(xs_test, ys_test)
#            print len(xs_test), 'cheat'
        predictor = lambda x: _fitter.predict([x])[0]
        predictor.coef_ = _fitter.coef_
        return predictor

    return fitter


def sklearn_ridge_logreg_fitter(alphas=None, cheat=False, use_train_oos=False):

    def fitter(xs_train, xs_test, ys_train, ys_test=None, xs_train_oos=None, xs_test_oos=None, ys_train_oos=None, ys_test_oos=None):

        if use_train_oos:
            xs_train = np.concatenate((xs_train, xs_train_oos), axis=0)
            ys_train = np.concatenate((ys_train, ys_train_oos))

        from sklearn.linear_model import LogisticRegressionCV
        _fitter = LogisticRegressionCV(Cs=alphas)
#        pdb.set_trace()
        if not cheat:
            _fitter.fit(xs_train, (ys_train > 0).astype(int))
            #print len(xs_train), 'no cheat'
        else:
            _fitter.fit(xs_test, (ys_test > 0).astype(int))
            #print len(xs_test), 'cheat'
        predictor = lambda x: np.log(_fitter.predict_proba([x])[0][1] / (1. -  _fitter.predict_proba([x])[0][1]))
#        predictor = lambda x: float(_fitter.predict([x])[0] > 0.5)
        predictor.coef_ = _fitter.coef_
        return predictor

    return fitter



def SIR_directions(xs, ys, method='sir'):
    from rpy2.robjects.packages import importr
    import rpy2.robjects as ro, string
    r = ro.r
    r('library(dr)')
    # create df
    N,K = xs.shape
    #pdb.set_trace()
    df_cmd = 'df=data.frame(matrix(c(%s),nrow=%d,ncol=%d))' % (string.join(map(str,xs.reshape(N*K)),sep=','),N,K)
    #print df_cmd
    import rpy2.robjects.numpy2ri as rpyn
    r(df_cmd)
    r('df$Y=c(%s)' % string.join(map(str,ys),sep=','))
    #print r('dim(df)')
    # call SIR
    rel = 'Y~%s' % string.join(map(lambda i: 'X%d'%i, range(1,K+1)), sep='+')
    #print rel
    bases = r('dr(%s, data=df, method=\"%s\")$evectors' % (rel,method))
    return rpyn.ri2py(bases)

    
def KDE(xs, B, sigma, xs_eval):
    # estimates are off by a multiplicative constant
    num_basis = 100
    us = np.dot(xs, B)
    us_basis = us[0:num_basis,:]
    us_eval = np.dot(xs_eval, B)
    return np.sum(utils.get_gaussian_K_helper(sigma, us_eval, us_basis), axis=1)

def KDE_ws(xs_train, xs_test, sigma, B, c_lsif, max_ratio, switch=False):
    if not switch:
        N_train = len(xs_train)
        train_densities = KDE(xs_train, B, sigma, xs_train)
        test_densities = KDE(xs_test, B, sigma, xs_train)
        train_densities = train_densities + c_lsif # view c_lsif as pseudocount here
        test_densities = test_densities + c_lsif
        ratios = test_densities / train_densities
        ratios = N_train * ratios / np.sum(ratios)
        if True: # fix?
            ratios = np.minimum(max_ratio, ratios)
            ratios = N_train * ratios / np.sum(ratios)
            
        return ratios
    else:
        N_test = len(xs_test)
        train_densities = KDE(xs_train, B, sigma, xs_test)
        test_densities = KDE(xs_test, B, sigma, xs_test)
        train_densities = train_densities + c_lsif # view c_lsif as pseudocount here
        test_densities = test_densities + c_lsif
        ratios = test_densities / train_densities
        ratios = N_test * ratios / np.sum(ratios)
        if True: # fix?
            ratios = np.minimum(max_ratio, ratios)
            ratios = N_test * ratios / np.sum(ratios)
        return ratios

def KDE_ws_oos(xs_train, xs_test, sigma, B, c_lsif, max_ratio, xs_oos):
    N_oos = len(xs_oos)
    train_densities = KDE(xs_train, B, sigma, xs_oos)
    test_densities = KDE(xs_test, B, sigma, xs_oos)
    train_densities = train_densities + c_lsif # view c_lsif as pseudocount here
    test_densities = test_densities + c_lsif
    ratios = test_densities / train_densities
    ratios = N_oos * ratios / np.sum(ratios)
    if True: # fix?
        ratios = np.minimum(max_ratio, ratios)
        ratios = N_oos * ratios / np.sum(ratios)
            
    return ratios
#def KLIEP_objective_given_ws(ws_train, ws_test):
#    return np.log(ws_test) / len(ws_test)

def LSIF_H_h(xs_train, xs_test, xs_basis, sigma):
#    sigma = utils.median_distance(np.concatenate((xs_train, xs_test), axis=0), np.concatenate((xs_train, xs_test), axis=0))
#    print sigma
#    pdb.set_trace()
    train_basis_vals = utils.get_gaussian_K(sigma, xs_basis, xs_train, nystrom=False)
    H = np.sum(train_basis_vals[:,np.newaxis,:] * train_basis_vals[np.newaxis,:,:], axis=2)
    h = np.sum(utils.get_gaussian_K(sigma, xs_test, xs_basis, nystrom=False), axis=0)
    return H, h


def least_squares_lsif_alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg):
    #    print c_lsif, sigma
    us_train = project(xs_train, B)
    us_test = project(xs_test, B)
    us_basis = project(xs_basis, B)
    H, h = LSIF_H_h(us_train, us_test, us_basis, sigma)
    if add_reg:
        H_hat = (H / len(us_train)) + (np.eye(len(H)) * c_lsif)
    else:
        H_hat = (H / len(us_train))
    h_hat = h / len(us_test)
    try:
        lsif_alpha = np.linalg.solve(H_hat, h_hat)
    except:
        print c_lsif, sigma
        pdb.set_trace()
    if (np.isnan(lsif_alpha)).any():
        pdb.set_trace()
#    print lsif_alpha
#    print 'sigma, c_lsif', sigma, c_lsif
#    print lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)[0:10]
    return lsif_alpha

def kliep_objective(xs_train, xs_test, alpha):
    N_train, N_test = len(xs_train), len(xs_test)
#    print 'test', (- ( np.sum(np.dot(xs_test, alpha) / N_test)))
#    print 'train', logsumexp(np.dot(xs_train, alpha))
    return (- ( np.sum(np.dot(xs_test, alpha) / N_test))) + (np.log(1./N_train) + logsumexp(np.dot(xs_train, alpha)))

def loglinear_alpha_given_x(xs_train, xs_test, c):

    N_train, N_test = len(xs_train), len(xs_test)
    D = xs_train.shape[1]
    assert xs_train.shape[1] == xs_test.shape[1]
    
    #print c
    
    def objective(alpha):
#        print alpha
#        print project(xs_test, alpha) / N_test
#        print np.log(1./N_train)
#        print project(xs_train, alpha)
#        pdb.set_trace()
#        print logsumexp(project(xs_train, alpha))
#        print - ( (project(xs_test, alpha) / N_test) - (np.log(1./N_train) + logsumexp(project(xs_train, alpha))) ), 'all'
#        pdb.set_trace()
        return kliep_objective(xs_train, xs_test, alpha) + (c * np.dot(alpha,alpha))
#        return (- ( np.sum(project(xs_test, alpha) / N_test) - (np.log(1./N_train) + logsumexp(project(xs_train, alpha))) )) + (c * np.dot(alpha,alpha))

    dobjective_dalpha = autograd.jacobian(objective)
    #import scipy
    alpha_fit = scipy.optimize.minimize(fun=objective, x0=np.zeros(D), jac=dobjective_dalpha)['x']
    return alpha_fit

def loglinear_ratios_given_x(xs_train, xs_test, c):
    N_train, N_test = len(xs_train), len(xs_test)
    alpha_fit = loglinear_alpha_given_x(xs_train, xs_test, c)
    unnormalized_ws_train = np.exp(np.dot(xs_train, alpha_fit))
    ws_train = N_train * unnormalized_ws_train / np.sum(unnormalized_ws_train)

    return ws_train



        


class LSIF_objective(quad_objective):

    def Pq(self, xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg):
        us_train = project(xs_train, B)
        us_test = project(xs_test, B)
        us_basis = project(xs_basis, B)
        H, h = LSIF_H_h(us_train, us_test, us_basis, sigma)
        #print H
#        print c_lsif, 'c_lsif'
#        print H[0:3,0:10]
        eps = 0.001
        try:
            if add_reg:
                return (H / len(us_train)) + (np.eye(len(us_basis)) * (c_lsif + eps)), -h / len(us_test)
            else:
                return (H / len(us_train)), -h / len(us_test)
        except:
            print add_reg
            pdb.set_trace()


class single_arg_LSIF_objective(quad_objective):

    def Pq(self, xs_train, xs_test, log_sigma_c_lsif, B, xs_basis, max_ratio, add_reg):
        sigma = np.exp(np.dot(log_sigma_c_lsif, np.array([1.,0.])))
        c_lsif = np.exp(np.dot(log_sigma_c_lsif, np.array([0.,1.])))
        return LSIF_objective.Pq(self, xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg)
        
    
class kmm_objective(quad_objective):
    
    def Pq(self, xs_train, xs_test, sigma, B):
        us_train = project(xs_train, B)
        us_test = project(xs_test, B)
        if len(us_train.shape) == 1:
            us_train = us_train.reshape((len(us_train),1))
        if len(us_test.shape) == 1:
            us_test = us_test.reshape((len(us_test),1))

        num_train = len(us_train)
        num_test = len(us_test)
        K_train_train = utils.get_gaussian_K(sigma, us_train, us_train)
        K_train_test = utils.get_gaussian_K(sigma, us_train, us_test)
        kappa = -(float(num_train)/num_test) * np.sum(K_train_test, axis=1)
        return K_train_train, kappa


class dopt_objective_dx(fxn):

    def __init__(self, objective):
        self.objective = objective

#    @basic.timeit
    def _val(self, *args):
        num_args = len(args)
        return self.objective.grad(*args, care_argnums=(num_args-1,))
    


def autograd_hessian_vector_product(inst, v, p_argnum, x_argnum, *args):
#    print 'HVP', v.shape
    def dotted(*_args):
        return np.dot(inst.grad(*_args, care_argnums=(x_argnum,)), v)
    return fxn.autograd_fxn(_val=dotted).grad(*args, care_argnums=(p_argnum,))


def finite_difference_hessian_vector_product(inst, v, p_argnum, x_argnum, *args):
    print 'HVP'
    import copy
    eps = .001
    new_args = copy.deepcopy(list(args))
    new_args[x_argnum] += eps * v
#    pdb.set_trace()
    return (inst.grad(*new_args, care_argnums=(p_argnum,)) + inst.grad(*args, care_argnums=(p_argnum,))) / eps


def get_tight_constraints(A, b, x):
    verbose = False
    LHS = np.dot(A, x)
    #print LHS, b
    #print zip(b-LHS, LHS < b)
#    print LHS < b
    #print x.min(), x.max()
    assert ( (LHS-0.01) < b).all()
    tight_eps = 0.01
    tight = (b - LHS) < tight_eps
    if verbose: print 'num_tight:', np.sum(tight)
    return A[tight], b[tight]
        

def get_dx_opt_dp_direct(lin_solver, d_dp_df_dx_val, d_dx_df_dx_val, ineq_val, dineq_dx_opt_val, dineq_dp_val):

    # create G matrix
    G = np.concatenate((np.concatenate((d_dx_df_dx_val, dineq_dx_opt_val.T), axis=1), np.concatenate((f_dual_val[:,np.newaxis]*dineq_dx_opt_val, np.diag(ineq_val)), axis=1)), axis=0)
    H = np.concatenate((d_dp_df_dx_val, dineq_dp_val), axis=0)
    ans_pre = lin_solver(G, H)
    x_len = len(d_dx_df_dx_val)
    p_dim = len(d_dp_df_dx_val.shape) - 1
    return ans[(slice(0,x_len),) + ((slice(None),) * p_dim)]

def get_dx_opt_delta_p(lin_solver, d_dp_df_dx_val, P, G, h, x_opt, p, delta_p_direction):

    verbose = False
    
    # f(x, p) should be convex
    x_len = G.shape[1]

    # get tight constraints
    G_tight, h_tight = get_tight_constraints(G, h, x_opt)
    num_tight = G_tight.shape[0]

    # get d
    p_dim = len(delta_p_direction.shape)
    delta_p_direction_broadcasted = np.tile(delta_p_direction, tuple([x_len] + [1 for i in xrange(p_dim)]))
    d_top = -np.sum(d_dp_df_dx_val * delta_p_direction_broadcasted, axis=tuple(range(1,1+p_dim)))
    d_bottom = np.zeros(num_tight)
    d = np.hstack((d_top,d_bottom))

    # get C
    C = np.vstack((np.hstack((P, G_tight.T)), np.hstack((G_tight, np.zeros((num_tight, num_tight))))))

    # get deriv
    deriv = lin_solver(C, d)
    if verbose: print 'solver error:', np.linalg.norm(np.dot(C,deriv) - d)

    return deriv


class ineq_constraints(fxn):
    pass


class Gh_ineq_constraints(ineq_constraints):

    def __init__(self, _get_Gh, **kwargs):
        self._get_Gh = _get_Gh
        ineq_constraints.__init__(self, **kwargs)
#        self._G, self._h = _G, _h

    def get_Gh(self, *args):
        return self._get_Gh(*args)
#        return self.G(*args), self.h(*args)
    
    def _val(self, *args):
        # accepts 1 more argument than get_Gh
        G,h = self.get_Gh(*args[:-1])
        x = args[-1]
        return np.dot(G,x) - h
#        return self.G(*args) - self.h(*args)


class constant_Gh_ineq_constraints(Gh_ineq_constraints):

    def _vector_jacobian_product(self, v, x_argnum, *args):
        return np.zeros(args[x_argnum].shape)
    

class unconstrained_ineq_constraints(constant_Gh_ineq_constraints):

    def __init__(self, *args, **kwargs):
        fxn.__init__(self, *args, **kwargs)
#        self.num_vars = num_vars

    def _val(self, *args):
        return np.zeros((0,))
#        ans = np.zeros((len(args[-1]),0))
#        pdb.set_trace()
#        return ans

    def _grad(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums')
        return np.zeros((0,)+args[care_argnums[0]].shape)
        
#    def _get_Gh(self, *args):
#        x_len = args
#        return np.zeros(shape=(0,x_len)), np.zeros(shape=(0,))
#        return np.zeros(shape=(0,self.num_args)), np.zeros(shape=(0,))
    
    
class opt(fxn):

    def __init__(self, lin_solver, objective, dobjective_dx, ineq_constraints=None, direct=True):
        self.lin_solver, self.objective, self.dobjective_dx, self.direct = lin_solver, objective, dobjective_dx, direct
        if not (ineq_constraints is None):
            self.ineq_constraints = ineq_constraints

    def backward_prop(self, args, _, val, care_argnums):
        num_args = len(args)
        self.check_care_argnums(args, care_argnums) # special
        #p = args[-1]
        p = args[care_argnums[0]]
        #G,h = self.ineq_constraints.get_Gh(*args)
        #d_dp_df_dx_val = self.dobjective_dx.grad(*(args+(val,)), care_argnums=(num_args-1,)) # hardcoded - should be able to take grad wrt to other arguments

        if self.direct:
            d_dp_df_dx_val = self.dobjective_dx.grad(*(args+(val,)), care_argnums=care_argnums) # hardcoded - should be able to take grad wrt to other arguments
            d_dx_df_dx_val = self.dobjective_dx.grad(*(args+(val,)), care_argnums=(num_args,))

            ineq_val = self.ineq_constraints(*(args+(val,)))
            dineq_dx_opt_val = self.ineq_constraints.grad(*(args+(val,)), care_argnums=(num_args,))
            dineq_dp_val = self.ineq_constraints.grad(*(args+(val,)), care_argnums=care_argnums)

            return get_dx_opt_dp_direct(lin_solver, d_dp_df_dx_val, d_dx_df_dx_val, ineq_val, dineq_dx_opt_val, dineq_dp_val)
        else:
            raise NotImplementedError
        
        ans = np.zeros(val.shape + p.shape)
    
        for index in np.ndindex(*p.shape):
            delta_p_direction = np.zeros(p.shape)
            delta_p_direction[index] = 1.
            temp = get_dx_opt_delta_p(self.lin_solver, d_dp_df_dx_val, d_dx_df_dx_val, G, h, val, p, delta_p_direction)
            ans[(slice(None),)+index] = temp[:len(val)]

        return ans

    def check_care_argnums(self, args, care_argnums):
        assert len(care_argnums) == 1
        #assert care_argnums == (len(args)-1,)


        
class cvx_opt(opt):
    """
    unconstrained
    """
#    def __init__(self, lin_solver, objective, dobjective_dx, optimizer=optimizers.scipy_minimize_optimizer(options={'maxiter':10})):
#    def __init__(self, lin_solver, objective, dobjective_dx, optimizer=optimizers.scipy_minimize_optimizer()): # FIX
    def __init__(self, lin_solver, objective, dobjective_dx, optimizer=optimizers.scipy_minimize_optimizer(options={'disp':False,'maxiter':1}), warm_start=True): # FIX
        self.optimizer, self.warm_start = optimizer, warm_start
        self.old_x_opt = {}
        opt.__init__(self, lin_solver, objective, dobjective_dx)

    @property
    def ineq_constraints(self):
#        pdb.set_trace()
        return unconstrained_ineq_constraints()#.autograd_fxn()
#        return unconstrained_ineq_constraints(self.objective.arg_shape(*args)[0])
#        return np.zeros(shape=(0,self.objective.arg_shape(*args)[0])), np.zeros(shape=(0,))
    
    def forward_prop(self, *args):
        f = lambda x: self.objective(*(args + (x,)))
        df_dx = lambda x: self.dobjective_dx(*(args + (x,)))
        if not self.warm_start:
            x0=np.random.normal(size=self.objective.arg_shape(*args))
        else:
            try:
                x0 = self.old_x_opt[self.objective.arg_shape(*args)]
            except KeyError:
                x0 = np.random.normal(size=self.objective.arg_shape(*args))
        logger = utils.optimizer_logger(f, df_dx)
#        pdb.set_trace()
        try:
            val = self.optimizer.optimize(f, df_dx, x0)
        except:
            pdb.set_trace()
        if self.warm_start:
            self.old_x_opt[self.objective.arg_shape(*args)] = val
#        ans = scipy.optimize.minimize(fun=f, x0=np.random.normal(size=self.objective.arg_shape(*args)), jac=df_dx, method='L-BFGS-B', options={'disp':10}, callback=logger)
        #logger.display_df()
        #logger.plot()
#        print ans['x'], 'hhhhhhhh'
#        print ans
        return (None,(None,None)), val
#        return None, val
        
        
class full_quad_opt(opt):
    """
    assumes objective fxn being maximized is quadratic and parameterized by the full quadratic term matrix
    cvxopt us used to maximize the objective
    """
    def __init__(self, lin_solver, objective, dobjective_dx, ineq_constraints, warm_start=True):
        self.warm_start = warm_start
        opt.__init__(self, lin_solver, objective, dobjective_dx, ineq_constraints)

    def forward_prop(self, *args):
        # fix: add warm_start
        P, q = self.objective.Pq(*args) # fix: if cholesky/nystrom, this will return the factorized matrix, solver will accept these
        G, h = self.ineq_constraints.get_Gh(*args)
        #print 'G'
        #print G[-4:,:]
        #print 'h'
        #print h[-4:]
#        pdb.set_trace()
        if self.warm_start:
            try:
                init_x, init_ineq_dual_val = self.old_x, self.old_ineq_dual_val
                if len(init_x) != G.shape[1]:
                    raise ValueError
#                pdb.set_trace()
                from cvxopt import matrix
                try:
                    primal_val, eq_dual_val, ineq_dual_val = qp.cvxopt_solver(P, q, G, h, initvals={'x':matrix(init_x), 'z':matrix(init_ineq_dual_val)})
                except:
#                    print P
                    pdb.set_trace()
            except (AttributeError, ValueError):
                primal_val, eq_dual_val, ineq_dual_val = qp.cvxopt_solver(P, q, G, h)
            self.old_x, self.old_ineq_dual_val = primal_val, ineq_dual_val
        else:
            primal_val, eq_dual_val, ineq_dual_val = qp.cvxopt_solver(P, q, G, h)
        return ((P,q,G,h),(eq_dual_val,ineq_dual_val)), primal_val # fix: can add dual optimal variables here


def kmm_get_Gh(xs_train, xs_test, sigma, B, w_max, eps):
    return kmm.get_kmm_Gh(w_max, eps, len(xs_train), len(xs_test))


def LSIF_get_Gh(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg):
    num_bases = len(xs_basis)
    G_pre, h_pre = -1. * np.eye(num_bases), np.zeros(num_bases)
#    return G_pre, h_pre
    us_train = project(xs_train, B)
    us_basis = project(xs_basis, B)
    train_basis_vals = utils.get_gaussian_K(sigma, us_train, us_basis, nystrom=False)
#    print train_basis_vals[0:5,0:20]
#    pdb.set_trace()
    eps = 0.001
    summed_basis_vals = np.sum(train_basis_vals, axis=0)
    #print 'summed'
    #print summed_basis_vals
    G_post = np.array([summed_basis_vals, -summed_basis_vals])
    #G_post = np.array([summed_basis_vals])
    #G_post = np.array([-summed_basis_vals])
    h_post = np.array([len(us_train) + eps, -(len(us_train) - eps)])
    #h_post = np.array([(len(us_train) + eps),])
    #h_post = np.array([-(len(us_train) + eps),])
    #print 'G_post'
    #print G_post
    return np.concatenate((G_pre,G_post), axis=0), np.concatenate((h_pre,h_post))

        
class full_ws_opt_given_B(full_quad_opt):

    def __init__(self, w_max, eps, lin_solver, kmm_objective, dkmm_objective_dws):
        self.w_max, self.eps = w_max, eps
        self.lin_solver, self.objective, self.dobjective_dx = lin_solver, kmm_objective, dkmm_objective_dws
    
    def get_Gh(self, xs_train, xs_test, sigma, B):
        return kmm.get_kmm_Gh(self.w_max, self.eps, len(xs_train), len(xs_test))

    
def get_dg_dp_thru_x_opt(lin_solver, d_dp_df_dx_val, dg_dx_opt_val, P, G, h, x_opt, p):

    verbose = False
    
    # assumes L(x_opt), x_opt = argmin_x f(x,p) subject to Ax<=b
    
    # get tight constraints
    G_tight, h_tight = get_tight_constraints(G, h, x_opt)
    num_tight = G_tight.shape[0]

    # make C matrix
    C = np.vstack((np.hstack((P,G_tight.T)), np.hstack((G_tight,np.zeros((num_tight,num_tight))))))
#    print np.linalg.eigvals(P)
#    pdb.set_trace()
#    print np.linalg.eigvals(C)
#    pdb.set_trace()
    
    # make d vector
    d = np.hstack((dg_dx_opt_val, np.zeros(num_tight)))

    # solve Cv=d for x
    v = lin_solver(C, d)
    if verbose: print 'solver error:', np.linalg.norm(np.dot(C,v) - d)
#    pdb.set_trace()

    # make D
    D = np.vstack((-d_dp_df_dx_val, np.zeros((num_tight,)+p.shape)))

    return np.sum(D.T * v[tuple([np.newaxis for i in xrange(len(p.shape))])+(slice(None),)], axis=-1).T



@basic.timeit('thru')
#def get_dg_dp_thru_x_opt_new(lin_solver, objective_hessian_vector_product, dg_dx_opt_val, G, h, f_args, f_val, f_dual_val, care_argnum_in_f_args, g_args, Gh_vector_jacobian_product):
def get_dg_dp_thru_x_opt_new(lin_solver, objective_hessian_vector_product, dg_dx_opt_val, ineq_val, dineq_dx_opt_val, f_args, f_val, f_dual_val, care_argnum_in_f_args, g_args, ineq_vector_jacobian_product):

    #print f_val

    #print 'ineq_val'
    #print ineq_val
    #print 'f_dual_val'
    #print f_dual_val
    #print 'dineq_dx_opt_val'
    #print dineq_dx_opt_val[-4:,:]
#    pdb.set_trace()
    
    p = f_args[care_argnum_in_f_args]

    verbose = False
    
    # assumes L(x_opt), x_opt = argmin_x f(x,p) subject to Ax<=b
    
    # get tight constraints
    #G_tight, h_tight = get_tight_constraints(G, h, f_val)
    #num_tight = G_tight.shape[0]

    # make C matrix
    #C = np.vstack((np.hstack((P,G_tight.T)), np.hstack((G_tight,np.zeros((num_tight,num_tight))))))

    # make function that computes C*v
    x_argnum_in_f_args = len(f_args)
    C_size = (len(f_val) + ineq_val.shape[0])
    #print dineq_dx_opt_val
    #print f_dual_val
    def Cu_f(u):
        assert len(u) == C_size
        u_pre, u_post = u[0:len(f_val)], u[len(f_val):]
        Cu_pre = objective_hessian_vector_product(u_pre, x_argnum_in_f_args, x_argnum_in_f_args, *(f_args+(f_val,)))
#        Cu_pre += np.dot(G_tight.T, u_post)
        #Cu_pre += np.dot(G.T, u_post)
        if len(u_post) > 0:
            Cu_pre += np.dot((f_dual_val[:,np.newaxis] * dineq_dx_opt_val).T, u_post)
        #Cu_post = np.dot(G_tight, u_pre)
#        Cu_post = np.dot(f_dual_val[:,np.newaxis] * dineq_dx_opt_val, u_pre) # new
            Cu_post = np.dot(dineq_dx_opt_val, u_pre) # new
        #Cu_post += (np.dot(G,f_val) - h) * u_post # new
            Cu_post += ineq_val * u_post # new
            Cu = np.concatenate((Cu_pre, Cu_post), axis=0)
        else:
            Cu = Cu_pre
#        print Cu
#        pdb.set_trace()
        return Cu
        
    from scipy.sparse.linalg import LinearOperator
    C = LinearOperator((C_size, C_size), matvec=Cu_f)
        

#    print np.linalg.eigvals(P)
#    pdb.set_trace()
#    print np.linalg.eigvals(C)
#    pdb.set_trace()
    
    # make d vector
    #d = np.hstack((dg_dx_opt_val, np.zeros(num_tight)))
    d = np.hstack((dg_dx_opt_val, np.zeros(ineq_val.shape[0]))) 

    # solve Cv=d for x
    v = lin_solver(C, d)
    if verbose: print 'solver error:', np.linalg.norm(C(v) - d)
#    pdb.set_trace()

    # calculate D.T * v
    v_pre = v[0:len(f_val)]
    v_post = v[len(f_val):]
    ans1 = (-objective_hessian_vector_product(v_pre, care_argnum_in_f_args, x_argnum_in_f_args, *(f_args+(f_val,))))
    ans2 = -ineq_vector_jacobian_product(v_post*f_dual_val, care_argnum_in_f_args, *(f_args+(f_val,)))
    #print 'ans1'
    #print ans1
    #print 'ans2'
    #print ans2
#    if not (Gh_vector_jacobian_product is None):
#        ans += Gh_vector_jacobian_product(v_post, care_argnum_in_f_args, *f_args)
    return ans1 + ans2


    # make D
    #D = np.vstack((-d_dp_df_dx_val, np.zeros((num_tight,)+p.shape)))
    D = np.vstack((-d_dp_df_dx_val, np.zeros((G.shape[0],)+p.shape))) # new

    return np.sum(D.T * v[tuple([np.newaxis for i in xrange(len(p.shape))])+(slice(None),)], axis=-1).T


class sum(fxn):

    def __init__(self, fs, fs_argnums, weights=None):
        self.weights = np.ones(len(fs)) if weights is None else weights
        self.fs, self.fs_argnums = fs, fs_argnums
        assert len(self.fs) == len(self.fs_argnums)
        assert len(self.weights) == len(self.fs)

    def forward_prop(self, *args):
        ans = 0.
        f_vals = []
        for (f,f_argnums,w) in itertools.izip(self.fs, self.fs_argnums, self.weights):
            if w != 0:
                try:
                    f_args = [args[i] for i in f_argnums]
                except:
                    pdb.set_trace()

                f_val = f(*f_args)
                f_vals.append(f_val)
                ans += w * f_val
            else:
                f_vals.append(0.)
        #print f_vals, 'sum', self.fs
        return f_vals, ans

    def backward_prop(self, args, f_vals, val, care_argnums):
        ans = 0.
#        print [f._val for f in self.fs], care_argnums
#        pdb.set_trace()
        for (f,f_argnums,w) in itertools.izip(self.fs, self.fs_argnums, self.weights):
            f_args = [args[i] for i in f_argnums]
            try:
                if w != 0:
                    ans += w * f.grad(*f_args, care_argnums=[list(f_argnums).index(care_argnum) for care_argnum in care_argnums])
            except ValueError:
                pass
        return ans

class product(fxn):

    def __init__(self, g, h, g_argnums, h_argnums):
        self.g, self.h, self.g_argnums, self.h_argnums = g, h, g_argnums, h_argnums

    def g_args(self, *args):
        return tuple([args[i] for i in self.g_argnums])

    def h_args(self, *args):
        return tuple([args[i] for i in self.h_argnums])

    def forward_prop(self, *args):
        g_args = self.g_args(*args)
        h_args = self.h_args(*args)
        g_val = self.g.val(*g_args)
        h_val = self.h.val(*h_args)
        return (g_val,h_val), g_val * h_val

    def backward_prop(self, args, (g_val,h_val), val, care_argnums):
        g_args = self.g_args(*args)
        h_args = self.h_args(*args)
        x = args[care_argnums[0]]
        #assert len(g_val.shape) == 1
        #assert len(h_val.shape) == 1
        assert g_val.shape == h_val.shape
        try:
            care_argnum_in_g_args = list(self.g_argnums).index(care_argnums[0])
            dg_dx_val = self.g.grad(*g_args, care_argnums=(care_argnum_in_g_args,))
            dproduct_dg_val = h_val[((slice(None),)*len(g_val.shape)) + ((np.newaxis,)*(len(x.shape)))] * dg_dx_val
        except ValueError:
            dproduct_dg_val = np.zeros(g_val.shape + x.shape)
        try:
            care_argnum_in_h_args = list(self.h_argnums).index(care_argnums[0])
            dh_dx_val = self.h.grad(*h_args, care_argnums=(care_argnum_in_h_args,))
            dproduct_dh_val = g_val[((slice(None),)*len(h_val.shape)) + ((np.newaxis,)*(len(x.shape)))] * dh_dx_val 
        except ValueError:
            dproduct_dh_val = np.zeros(h_val.shape + x.shape)
#        print (dg_dx_val * h_val) + (dh_dx_val * g_val), 'asdf'
        return dproduct_dg_val + dproduct_dh_val

    def check_care_argnums(self, args, care_argnums):
        assert len(care_argnums) == 1
        

class two_step(fxn):
    """
    h then g
    g_val_h_argnum is position in h_args to insert g_val
    """
    def __init__(self, g, h, g_argnums, h_argnums, g_val_h_argnum=None):
        self.g, self.h, self.g_argnums, self.h_argnums = g, h, g_argnums, h_argnums
        self.g_val_h_argnum = len(self.h_argnums) if g_val_h_argnum is None else g_val_h_argnum

    def g_args(self, *args):

        def get(key):
            if isinstance(key, tuple):
                i, f = key
                return f(args[i])
            else:
                return args[key]

        return tuple(map(get, self.g_argnums))
        
#        try:
#            return tuple([args[i] for i in self.g_argnums])
#        except:
#            pdb.set_trace()

    def h_args(self, g_val, *args):

        def get(key):
            if isinstance(key, tuple):
                i, f = key
                return f(args[i])
            else:
                return args[key]

        l = map(get, self.h_argnums)

        def pos(key):
            if isinstance(key, tuple):
                i, f = key
                return i
            else:
                return key

        try:
            sorted_keys = sorted(self.g_val_h_argnum, key=pos, reverse=True)
        except TypeError:
            sorted_keys = [self.g_val_h_argnum]

        def g_val_key(key):
            if isinstance(key,tuple):
                pos, f = key
                return f(g_val)
            else:
                return g_val
            
        for key in sorted_keys:
            l.insert(pos(key), g_val_key(key))

        return tuple(l)
        
#        l = [args[i] for i in self.h_argnums]
#        l.insert(self.g_val_h_argnum, g_val)
#        return tuple(l)
    
    def forward_prop(self, *args):
        g_args = self.g_args(*args)
#        try:
        g_stuff, g_val = self.g.forward_prop(*g_args)
#        except:
#            pdb.set_trace()
        try:
            h_args = self.h_args(g_val, *args)
        except:
            pdb.set_trace()
        h_stuff, h_val = self.h.forward_prop(*h_args)
        return (g_val, (g_stuff, h_stuff)), h_val

    def two_step_grad(self, args, (g_val,(g_stuff,h_stuff)), h_val, care_argnums):
        pdb.set_trace()
        assert False
        p = args[care_argnums[0]]
        if care_argnums[0] in self.g_argnums:
            care_argnum_in_g_args = list(self.g_argnums).index(care_argnums[0])
            g_args = self.g_args(*args)
            h_args = self.h_args(g_val, *args)
            dh_dg_val = self.h.grad(*h_args, care_argnums=(self.g_val_h_argnum,)) # hard
            dg_dp_val = self.g.grad(*g_args, care_argnums=(care_argnum_in_g_args,))
            # assume g_val has dimension 1
            dh_dg_dg_dp_val = np.sum(dh_dg_val[((slice(None),) * len(dh_dg_val.shape)) + ((np.newaxis,) * (len(dg_dp_val.shape)-1))] * dg_dp_val[((np.newaxis,) * (len(dh_dg_val.shape)-1)) + ((slice(None),) * len(dg_dp_val.shape))], axis=len(dh_dg_val.shape)-1)
            #print dh_dg_dg_dp_val.shape
#            dh_dg_dg_dp_val = np.dot(dh_dg_val, dg_dp_val)
        else:
            dh_dg_dg_dp_val = np.zeros(p.shape)

        return dh_dg_dg_dp_val
    
    def backward_prop(self, args, (g_val,(g_stuff,h_stuff)), h_val, care_argnums):
        
        self.check_care_argnums(args, care_argnums) # special
        p = args[care_argnums[0]]
        dh_dg_dg_dp_val = self.two_step_grad(args, (g_val,(g_stuff,h_stuff)), h_val, care_argnums)
#        pdb.set_trace()
        if care_argnums[0] in self.h_argnums:
            care_argnum_in_h_args = list(self.h_argnums).index(care_argnums[0])
            if care_argnum_in_h_args >= self.g_val_h_argnum:
                care_argnum_in_h_args += 1
            h_args = self.h_args(g_val, *args)
#            print self.h, 'hhhh'
            dh_dp_val = basic.timeit('direct two_step grad')(self.h.grad)(*h_args, care_argnums=(care_argnum_in_h_args,))
        else:
            dh_dp_val = np.zeros(p.shape)#        dg_dp_val = self.g.grad(*(g_args+(x_opt,)), care_argnums=(self.g_p_argnum,))

        p = args[care_argnums[0]]
#        print p, care_argnums
        #print dh_dg_dg_dp_val, 'gg3'
        #print dh_dp_val, 'gg2'
#        pdb.set_trace()
        
        return dh_dg_dg_dp_val + dh_dp_val

    def check_care_argnums(self, args, care_argnums):
        assert len(care_argnums) == 1
    

def one(f, g_args, g_val, care_argnum_in_g_args):
    return f(*(g_args+(g_val,)), care_argnums=(care_argnum_in_g_args,))

def two(f, h_args, g_val_h_argnum):
    return f(*h_args, care_argnums=(g_val_h_argnum,)) # hard

def three(f, g_args, g_val):
    return f(*(g_args+(g_val,)), care_argnums=(len(g_args),)) # hard



class g_thru_f_opt(two_step):
    """
    assumes x_opt is last argument to g (not included in g_argnums)
    OBSOLETE: gradient can only be taken wrt last argument.  last argument is same for this fxn as well as quad_opt fxn
    OBSOLETE: g_p_argnum is a position in g_args
    """
#    def __init__(self, full_quad_opt, g, quad_argnums, g_argnums):
#        self.full_quad_opt, self.g, self.quad_argnums, self.g_argnums = full_quad_opt, g, quad_argnums, g_argnums

#    def quad_args(self, *args):
#        return tuple([args[i] for i in self.quad_argnums])

#    def g_args(self, *args):
#        return tuple([args[i] for i in self.g_argnums])
    
#    def forward_prop(self, *args):
#        quad_args = self.quad_args(*args)
#        g_args = self.g_args(*args)
#        _, x_opt = self.full_quad_opt.forward_prop(*quad_args)
#        _, g_val = self.g.forward_prop(*(g_args+(x_opt,)))
#        return (x_opt,), g_val

    @basic.timeit('old two_step_grad')
    def two_step_grad(self, args, (g_val,), h_val, care_argnums):
        p = args[care_argnums[0]]
        if care_argnums[0] in self.g_argnums:
            care_argnum_in_g_args = list(self.g_argnums).index(care_argnums[0])
            g_args = self.g_args(*args)
            h_args = self.h_args(g_val, *args)
    #        d_dp_df_dx_val = self.full_quad_opt.dobjective_dx.grad(*(quad_args+(x_opt,)), care_argnums=(len(quad_args)-1,))


#            d_dp_df_dx_val = self.g.dobjective_dx.grad(*(g_args+(g_val,)), care_argnums=(care_argnum_in_g_args,))
#            print basic.timeit(self.g.dobjective_dx.val)(*(g_args+(g_val,))).shape
#            print basic.timeit(self.g.dobjective_dx.grad)(*(g_args+(g_val,)), care_argnums=(care_argnum_in_g_args,)).shape
            d_dp_df_dx_val = one(self.g.dobjective_dx.grad, g_args, g_val, care_argnum_in_g_args)
#            pdb.set_trace()

#            dh_dxopt_val = self.h.grad(*h_args, care_argnums=(self.g_val_h_argnum,)) # hard
            dh_dxopt_val = basic.timeit('dloss_dxopt')(two)(self.h.grad, h_args, self.g_val_h_argnum) # hard

#            d_dx_df_dx_val = self.g.dobjective_dx.grad(*(g_args+(g_val,)), care_argnums=(len(g_args),)) # hard
            d_dx_df_dx_val = three(self.g.dobjective_dx.grad, g_args, g_val)
            
            #P, q = self.full_quad_opt.objective.Pq(*quad_args)
            G, h = self.g.get_Gh(*g_args)
            dh_dg_dg_dp_val = basic.timeit('old get_dg_dp_thru_x_opt')(get_dg_dp_thru_x_opt)(self.g.lin_solver, d_dp_df_dx_val, dh_dxopt_val, d_dx_df_dx_val, G, h, g_val, p)
        else:
            dh_dg_dg_dp_val = np.zeros(p.shape)

        return dh_dg_dg_dp_val
    

        

    def check_care_argnums(self, args, care_argnums):
        assert len(care_argnums) == 1
#        assert care_argnums == (len(args)-1,)



class g_thru_f_opt_new(two_step):
    """
    assumes x_opt is last argument to g (not included in g_argnums)
    OBSOLETE: gradient can only be taken wrt last argument.  last argument is same for this fxn as well as quad_opt fxn
    OBSOLETE: g_p_argnum is a position in g_args
    """

    @basic.timeit('new two_step_grad')
    def two_step_grad(self, args, (g_val,(g_stuff,h_stuff)), h_val, care_argnums):
        _, (g_eq_dual_val,g_ineq_dual_val) = g_stuff
        p = args[care_argnums[0]]
        if care_argnums[0] in self.g_argnums:
            care_argnum_in_g_args = list(self.g_argnums).index(care_argnums[0])
            g_args = self.g_args(*args)
            h_args = self.h_args(g_val, *args)
    #        d_dp_df_dx_val = self.full_quad_opt.dobjective_dx.grad(*(quad_args+(x_opt,)), care_argnums=(len(quad_args)-1,))


####            d_dp_df_dx_val = self.g.dobjective_dx.grad(*(g_args+(g_val,)), care_argnums=(care_argnum_in_g_args,))
#            print basic.timeit(self.g.dobjective_dx.val)(*(g_args+(g_val,))).shape
#            print basic.timeit(self.g.dobjective_dx.grad)(*(g_args+(g_val,)), care_argnums=(care_argnum_in_g_args,)).shape
#            d_dp_df_dx_val = one(self.g.dobjective_dx.grad, g_args, g_val, care_argnum_in_g_args)
#            pdb.set_trace()

            dh_dxopt_val = basic.timeit('dloss_dxopt')(self.h.grad)(*h_args, care_argnums=(self.g_val_h_argnum,)) # hard
#            dh_dxopt_val = two(self.h.grad, h_args, self.g_val_h_argnum) # hard

####            d_dx_df_dx_val = self.g.dobjective_dx.grad(*(g_args+(g_val,)), care_argnums=(len(g_args),)) # hard
#            d_dx_df_dx_val = three(self.g.dobjective_dx.grad, g_args, g_val)
            
            #P, q = self.full_quad_opt.objective.Pq(*quad_args)
            #G, h = self.g.get_Gh(*g_args)
            ineq_val = self.g.ineq_constraints(*(g_args+(g_val,)))
            dineq_dx_opt_val = self.g.ineq_constraints.grad(*(g_args+(g_val,)), care_argnums=(len(g_args),))
            dh_dg_dg_dp_val = basic.timeit('get_dg_dp_thru_x_opt_new')(get_dg_dp_thru_x_opt_new)(self.g.lin_solver, self.g.objective.hessian_vector_product, dh_dxopt_val, ineq_val, dineq_dx_opt_val, g_args, g_val, g_ineq_dual_val, care_argnum_in_g_args, h_args, self.g.ineq_constraints.vector_jacobian_product)
        else:
            dh_dg_dg_dp_val = np.zeros(p.shape)

        return dh_dg_dg_dp_val
    

        

    def check_care_argnums(self, args, care_argnums):
        assert len(care_argnums) == 1
#        assert care_argnums == (len(args)-1,)



class weighted_lsqr_loss_fxn(two_step):

    def check_care_argnums(self, (B, xs, ys, ws), care_argnums):
        pass

    
class weighted_lsqr_loss_loss_fxn(two_step):

    def check_care_argnums(self, (B, xs, ys, ws, b_opt), care_argnums):
        pass


class upper_bound(product):

    def check_care_argnums(self, (B, xs, ys, ws, b_opt, ws_full), care_argnums):
        pass


class sklearn_linear_regression(object):

    def __init__(self, cs):
        self.cs = cs

    def __call__(self, xs_train, xs_test, ys_train, ys_test=None):
        from sklearn import linear_model
        reg = linear_model.RidgeCV(alphas=1./self.cs)
        reg.fit(xs_train, ys_train)

        predictor = lambda x: reg.predict([x])[0]

        return predictor


class sklearn_kernel_regression(object):

    def __init__(self, c):
        self.c = c

    def __call__(self, xs_train, xs_test, ys_train, ys_test=None):
        from sklearn.kernel_ridge import KernelRidge
        reg = KernelRidge(kernel='rbf',gamma=1./self.c)
        reg.fit(xs_train, ys_train)

        predictor = lambda x: reg.predict([x])[0]

        return predictor

class subsample_cv_getter(object):
    
    def __init__(self, is_prop_train, is_prop_test):
        self.is_prop_train, self.is_prop_test = is_prop_train, is_prop_test

    def __call__(self, i, xs_train, xs_test, ys_train, ys_test=None, data_tag=None):
        assert False
        N_train, N_test = len(xs_train), len(xs_test)
        state = np.random.get_state()
        np.random.seed(i)

        num_is_train = int(self.is_prop_train * N_train)
        is_keep_train = np.concatenate((np.ones(num_is_train).astype(bool), np.zeros(N_train-num_is_train).astype(bool)))
        np.random.shuffle(is_keep_train)
        oos_keep_train = np.logical_not(is_keep_train)

        num_is_test = int(self.is_prop_test * N_test)
        is_keep_test = np.concatenate((np.ones(num_is_test).astype(bool), np.zeros(N_test-num_is_test).astype(bool)))
        np.random.shuffle(is_keep_test)
        oos_keep_test = np.logical_not(is_keep_test)
        np.random.set_state(state)
        if isinstance(xs, np.ndarray):
            ans = xs_train[is_keep_train], xs_test[is_keep_test], ys_train[is_keep_train], ys_test[is_keep_test], xs_train[oos_keep_train], xs[oos_keep_test], ys_train[oos_keep_train], ys_test[oos_keep_test]
        else:
            ans = xs_train.iloc[is_keep_train], xs_test.iloc[is_keep_test], ys_train.iloc[is_keep_train], ys_test.iloc[is_keep_test], xs_train.iloc[oos_keep_train], xs.iloc[oos_keep_test], ys_train.iloc[oos_keep_train], ys_test.iloc[oos_keep_test]
        if data_tag is None:
            return ans
        else:
            return ans, '(%s_subsample_is_prop_train_%.2f_is_prop_test_%.2ftrial_%d)' % (data_tag, self.is_prop_train, self.is_prop_test, i)

class kfold_cv_getter(object):

    def __init__(self, num_folds):
        self.num_folds = num_folds

    def __call__(self, i, xs_train, xs_test, ys_train, ys_test=None, data_tag=None):
        assert False
        state = np.random.get_state()
        np.random.seed(42)
        from sklearn.model_selection import StratifiedKFold, KFold
        if (not (balance_col is None)) or to_balance:
            s = StratifiedKFold(n_splits=self.num_folds, shuffle=False)
            if not (balance_col is None):
                assert len(zs.shape) == 2
                if isinstance(zs, np.ndarray):
                    balance = zs[:,balance_col]
                else:
                    balance = zs[balance_col]
            else:
                balance = zs
            is_idxs, oos_idxs = s.split(xs, balance)[i]
        else:
            s = KFold(n_splits=self.num_folds, shuffle=False)
            is_idxs, oos_idxs = list(s.split(xs, ys))[i]
        np.random.set_state(state)
        if isinstance(xs, np.ndarray):
            ans = xs[is_idxs], ys[is_idxs], all_zs[is_idxs], xs[oos_idxs], ys[oos_idxs], all_zs[oos_idxs]
        else:
            ans = xs.iloc[is_idxs], ys.iloc[is_idxs], all_zs.iloc[is_idxs], xs.iloc[oos_idxs], ys.iloc[oos_idxs], all_zs.iloc[oos_idxs]
        if data_tag is None:
            return ans
        else:
            return ans, '(%s_fold_%d_%d)' % (data_tag, i, self.num_folds)

class cfr_predictor(object):

    def __init__(self, loss, dim, imb_fun, p_alpha, p_lambda, nonlin, use_test_cov, train_path, xs_train, xs_test, ys_train, ys_test, data_tag, info):
        self.loss, self.dim, self.imb_fun, self.p_alpha, self.p_lambda, self.nonlin, self.use_test_cov, self.train_path  = loss, dim, imb_fun, p_alpha, p_lambda, nonlin, use_test_cov, train_path
        self.xs_train, self.xs_test, self.ys_train, self.ys_test, self.data_tag, self.info = xs_train, xs_test, ys_train, ys_test, data_tag, info

    def __repr__(self):
        return 'loss=%s_dim=%d_imb_fun=%s_p_alpha=%.8f_p_lambda=%.8f_nonlin=%s' % (self.loss, self.dim, self.imb_fun, self.p_alpha, self.p_lambda, self.nonlin)

    def predict(self, xs, use_test_cov=None):

        # possibly add xs to xs_test
        if not (use_test_cov is None):
            use = use_test_cov
        else:
            use = self.use_test_cov
        if use:
            xs_test = np.concatenate((self.xs_test, xs), axis=0)
        else:
            xs_test = self.xs_test

        # create temp folder
        import python_utils.python_utils.caching as caching
        import os
        if self.data_tag is None:
            temp_folder = '%s/%d/%s' % (caching.cache_folder, id(xs_train), repr(self))
        else:
            temp_folder = '%s/%s/%s' % (caching.cache_folder, self.data_tag, repr(self))
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        # put data into folder
        datadir = '%s/%s/' % (temp_folder, 'data')
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        dataform = 'input.npz'
        garbage_val = 42.
        x = np.concatenate((self.xs_train, xs_test), axis=0)
        yf = np.concatenate((self.ys_train, garbage_val * np.ones(len(xs_test))), axis=0)
        t = np.concatenate((np.zeros(len(self.xs_train)), np.ones(len(xs_test))), axis=0)
        np.savez('%s/%s' % (datadir, dataform), x=np.expand_dims(x,-1), yf=np.expand_dims(yf,-1), t=np.expand_dims(t,-1))

        data_test = 'to_predict.npz'
        x_test = xs
        t_test = np.zeros(len(xs))
        yf_test = garbage_val * np.ones(len(xs))
        np.savez('%s/%s' % (datadir, data_test), x=np.expand_dims(x_test,-1), t=np.expand_dims(t_test,-1), yf=np.expand_dims(yf_test,-1))

        # create output folder
        outdir = '%s/%s' % (temp_folder,'results')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
#        pdb.set_trace()

        # create argument string and run
#        config_d = {'loss':self.loss, 'imb_fun':self.imb_fun, 'p_alpha':self.p_alpha, 'p_lambda':self.p_lambda, 'nonlin':self.nonlin, 'datadir':'\"%s\"' % datadir, 'dataform':'\"%s\"' %dataform, 'data_test':'\"%s\"' % data_test, 'outdir':'\"%s\"' % outdir}
        config_d = {'loss':self.loss, 'imb_fun':self.imb_fun, 'p_alpha':self.p_alpha, 'p_lambda':self.p_lambda, 'nonlin':self.nonlin, 'datadir':'%s' % datadir, 'dataform':'%s' %dataform, 'data_test':'%s' % data_test, 'outdir':'%s' % outdir}
        if self.dim == -1:
            config_d['varsel'] = 1
        else:
            config_d['varsel'] = 0
            config_d['dim_in'] = self.dim
        flags = ' '.join('--%s %s' % (k,v) for (k,v) in config_d.iteritems())

        import CFR.cfrnet.cfr_net_train as cfr_net_train
        argv = [''] + flags.split(' ')
        import sys
        sys.argv = argv
        print argv
        cfr_net_train.my_run()

        if False:
            import subprocess
            python_path = '/home/fultonw/anaconda2/bin/python2.7'
            cmd = "%s %s %s" % (python_path, self.train_path, flags)
            print cmd
            cmd = [python_path, self.train_path, flags]
        #        print cmd
            subprocess.call(cmd, shell=False)
        #        subprocess.call(cmd, shell=True)
            import time
            time.sleep(1)

        # retrieve predictions
        import os
        result_paths = [os.path.join(outdir, o) for o in os.listdir(outdir) if os.path.isdir(os.path.join(outdir,o)) and o[0:7] == 'out_dir']
        print result_paths
        print [os.path.join(outdir, o) for o in os.listdir(outdir) if os.path.isdir(os.path.join(outdir,o))]
#        pdb.set_trace()
        assert len(result_paths) == 1
        if len(result_paths) == 0:
            return np.nan * np.zeros(len(xs))
        result_path = result_paths[0]
        train_out_path = '%s/%s' % (result_path, 'result.npz')
        test_out_path = '%s/%s' % (result_path, 'result.test.npz')
        d_train = np.load(train_out_path)
        d_test = np.load(test_out_path)
        return d_test['pred'][:,0,0,-1]
        
class cfr_fitter(object):

    def __init__(self, loss, dim, imb_fun, p_alpha, p_lambda, nonlin, use_test_cov, info=None):
        self.loss = {'square':'l2','abs':'l1','logistic':'log'}[loss]
        self.dim, self.imb_fun, self.p_alpha, self.p_lambda, self.nonlin, self.use_test_cov = dim, imb_fun, p_alpha, p_lambda, nonlin, use_test_cov
        self.train_path = '/home/fultonw/modules/CFR/cfrnet/cfr_net_train.py'
        self.info = {} if info is None else info

    def fit(self, xs_train, xs_test, ys_train, ys_test, data_tag=None):
#        pdb.set_trace()
        return cfr_predictor(self.loss, self.dim, self.imb_fun, self.p_alpha, self.p_lambda, self.nonlin, self.use_test_cov, self.train_path, xs_train, xs_test, ys_train, ys_test, data_tag, self.info)

class cov_shift_fitter(object):

    def __init__(self, raw_fitter, use_test_cov, info=None):
        self.raw_fitter, self.use_test_cov = raw_fitter, use_test_cov
        self.info = {} if info is None else info

    def fit(self, xs_train, xs_test, ys_train, ys_test, data_tag=None):
        return cov_shift_predictor(self.raw_fitter, self.use_test_cov, xs_train, xs_test, ys_train, ys_test, data_tag, self.info)

class cov_shift_predictor(object):

    def __init__(self, raw_fitter, use_test_cov, xs_train, xs_test, ys_train, ys_test, data_tag, info):
        self.raw_fitter, self.use_test_cov, self.xs_train, self.xs_test, self.ys_train, self.ys_test, self.data_tag, self.info = raw_fitter, use_test_cov, xs_train, xs_test, ys_train, ys_test, data_tag, info
        self.raw_predictor = None

    def predict(self, xs, use_test_cov=None):
        if not (use_test_cov is None):
            use = use_test_cov
        else:
            use = self.use_test_cov
        if use:
            xs_test = np.concatenate((self.xs_test, xs), axis=0)
        else:
            xs_test = self.xs_test
        if self.raw_predictor is None:
            self.raw_predictor = self.raw_fitter(self.xs_train, xs_test, self.ys_train, self.ys_test)
            self.info['N_eff'] = self.raw_predictor.N_eff
            self.info['B'] = self.raw_predictor.B
            self.info['b'] = self.raw_predictor.b
            self.info['get_ws'] = self.raw_predictor.get_ws
#        self.info['raw_predictor'] = self.raw_predictor
        return np.array(map(self.raw_predictor, xs))

    def get_ws(self, xs):
        return self.raw_predictor.get_ws(xs)

class subgroup_kfold_cv_getter(object):

    def __init__(self, num_folds):
        self.num_folds = num_folds

    def __str__(self):
        return 'subgroup_%d_fold' % self.num_folds

    def __call__(self, i, xs_train, xs_test, ys_train, ys_test, data_tag=None):

        import subgroup
        get_row = subgroup.get_get_row(xs_train)
        concat = subgroup.get_concat(xs_train)

        state = np.random.get_state()
        np.random.seed(42)

        from sklearn.model_selection import KFold
        s = KFold(n_splits=self.num_folds, shuffle=False)

        N_train, N_test = len(xs_train), len(xs_test)
        assert (get_row(xs_train,slice(-N_test,None)) == xs_test).all()
        zero_xs, zero_ys = get_row(xs_train,slice(0,-N_test)), get_row(ys_train,slice(0,-N_test))
        one_xs, one_ys = xs_test, ys_test
        zero_is_idxs, zero_oos_idxs = list(s.split(zero_xs, zero_ys))[i % self.num_folds]
        one_is_idxs, one_oos_idxs = list(s.split(one_xs, one_ys))[i % self.num_folds]
        zero_is_xs, zero_is_ys = get_row(zero_xs,zero_is_idxs), get_row(zero_ys,zero_is_idxs)
        one_is_xs, one_is_ys = get_row(one_xs,one_is_idxs), get_row(one_ys,one_is_idxs)
        zero_oos_xs, zero_oos_ys = get_row(zero_xs,zero_oos_idxs), get_row(zero_ys,zero_oos_idxs)
        one_oos_xs, one_oos_ys = get_row(one_xs,one_oos_idxs), get_row(one_ys,one_oos_idxs)

        xs_train_is, ys_train_is = concat((zero_is_xs, one_is_xs), axis=0), concat((zero_is_ys, one_is_ys), axis=0)
        xs_test_is, ys_test_is = one_is_xs, one_is_ys
        xs_train_oos, ys_train_oos = concat((zero_oos_xs, one_oos_xs), axis=0), concat((zero_oos_ys, one_oos_ys), axis=0)
        xs_test_oos, ys_test_oos = one_oos_xs, one_oos_ys

        ans = xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos

        np.random.set_state(state)

        if data_tag is None:
            return ans
        else:
            return ans, '(%s_subgroup_kfold_%d_%d)' % (data_tag, i, self.num_folds)

class kfold_cv_getter(object):

    def __init__(self, num_folds):
        self.num_folds = num_folds

    def __str__(self):
        return '%d_fold' % self.num_folds

    def __call__(self, i, xs_train, xs_test, ys_train, ys_test, data_tag=None):

        import subgroup
        get_row = subgroup.get_get_row(xs_train)
        concat = subgroup.get_concat(xs_train)

        state = np.random.get_state()
        np.random.seed(42)

        from sklearn.model_selection import KFold
        s = KFold(n_splits=self.num_folds, shuffle=False)

        N_train, N_test = len(xs_train), len(xs_test)
        train_is_idxs, train_oos_idxs = list(s.split(xs_train, ys_train))[i % self.num_folds]
        xs_train_is, xs_train_oos = get_row(xs_train, train_is_idxs), get_row(xs_train, train_oos_idxs)
        ys_train_is, ys_train_oos = get_row(ys_train, train_is_idxs), get_row(ys_train, train_oos_idxs)

        test_is_idxs, test_oos_idxs = list(s.split(xs_test, ys_test))[i % self.num_folds]
        xs_test_is, xs_test_oos = get_row(xs_test, test_is_idxs), get_row(xs_test, test_oos_idxs)
        ys_test_is, ys_test_oos = get_row(ys_test, test_is_idxs), get_row(ys_test, test_oos_idxs)

        ans = xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos

        np.random.set_state(state)

        if data_tag is None:
            return ans
        else:
            return ans, '(%s_kfold_%d_%d)' % (data_tag, i, self.num_folds)

class sample_cv_getter(object):

    def __init__(self, prop_train_is, prop_test_is):
        self.prop_train_is, self.prop_test_is = prop_train_is, prop_test_is

    def __call__(self, i, xs_train, xs_test, ys_train, ys_test, data_tag=None):

        import subgroup
        get_row = subgroup.get_get_row(xs_train)
        concat = subgroup.get_concat(xs_train)

        state = np.random.get_state()
        np.random.seed(i)

        train_idxs = np.arange(len(xs_train))
        test_idxs = np.arange(len(xs_test))
        np.random.shuffle(train_idxs)
        np.random.shuffle(test_idxs)

        xs_train = get_row(xs_train, train_idxs)
        ys_train = get_row(ys_train, train_idxs)
        xs_test = get_row(xs_test, test_idxs)
        ys_test = get_row(ys_test, test_idxs)

        N_train_is = int(self.prop_train_is * len(xs_train))
        N_test_is = int(self.prop_test_is * len(xs_test))
        xs_train_is = get_row(xs_train, slice(0,N_train_is))
        xs_train_oos = get_row(xs_train, slice(N_train_is,None))
        xs_test_is = get_row(xs_test, slice(0,N_test_is))
        xs_test_oos = get_row(xs_test, slice(N_test_is,None))
        ys_train_is = get_row(ys_train, slice(0,N_train_is))
        ys_train_oos = get_row(ys_train, slice(N_train_is,None))
        ys_test_is = get_row(ys_test, slice(0,N_test_is))
        ys_test_oos = get_row(ys_test, slice(N_test_is,None))

        ans = xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos

        if data_tag is None:
            return ans
        else:
            return ans, '%s_%.2f_%.2f_%d' % (data_tag, self.prop_train_is, self.prop_test_is, i)

    def __repr__(self):
        return 'sample_cv_%.2f_%.2f' % (self.prop_train_is, self.prop_test_is)

class subgroup_one_oos_fitting_eval_getter(object):

    def __call__(self, xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos, data_tag=None):
    
        import subgroup
        get_row = subgroup.get_get_row(xs_train_is)
        concat = subgroup.get_concat(xs_train_is)

        N_test_is = len(xs_test_is)
        zero_xs_is, zero_ys_is = get_row(xs_train_is, slice(0,-N_test_is)), get_row(ys_train_is, slice(0,-N_test_is))
        one_xs_is, one_ys_is = xs_test_is, ys_test_is

        N_test_oos = len(xs_test_oos)
        zero_xs_oos, zero_ys_oos = get_row(xs_train_oos, slice(0,-N_test_oos)), get_row(ys_train_oos, slice(0,-N_test_oos))
        one_xs_oos, one_ys_oos = xs_test_oos, ys_test_oos

        xs_train_fitting, ys_train_fitting = concat((zero_xs_is,zero_xs_oos,one_xs_is), axis=0), concat((zero_ys_is,zero_ys_oos,one_ys_is), axis=0)
        xs_test_fitting, ys_test_fitting = one_xs_is, one_ys_is

        xs_eval, ys_eval = one_xs_oos, one_ys_oos

        ans = xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval

        if data_tag is None:
            return ans
        else:
            return ans, '(%s_one_oos)' % data_tag

    def __str__(self):
        return 'subgroup_one_oos'

class train_oos_fitting_eval_getter(object):

    def __call__(self, xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos, data_tag=None):
    
        import subgroup
        get_row = subgroup.get_get_row(xs_train_is)
        concat = subgroup.get_concat(xs_train_is)

        xs_train_fitting, ys_train_fitting = xs_train_is, ys_train_is
        xs_test_fitting, ys_test_fitting = concat((xs_test_is,xs_test_oos), axis=0), concat((ys_test_is,ys_test_oos), axis=0)
        xs_eval, ys_eval = xs_train_oos, ys_train_oos


        ans = xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval

        if data_tag is None:
            return ans
        else:
            return ans, '(%s_train_oos)' % data_tag

    def __str__(self):
        return 'train_oos'

class test_oos_fitting_eval_getter(object):

    def __call__(self, xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos, data_tag=None):
    
        import subgroup
        get_row = subgroup.get_get_row(xs_train_is)
        concat = subgroup.get_concat(xs_train_is)

        xs_train_fitting, ys_train_fitting = concat((xs_train_is,xs_train_oos), axis=0), concat((ys_train_is, ys_train_oos), axis=0)
        xs_test_fitting, ys_test_fitting = xs_test_is, ys_test_is
        xs_eval, ys_eval = xs_test_oos, ys_test_oos

        ans = xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval

        if data_tag is None:
            return ans
        else:
            return ans, '(%s_test_oos)' % data_tag

    def __str__(self):
        return 'train_oos'

class iden_fitting_eval_getter(object):

    def __call__(self, xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos, data_tag=None):
        ans = xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_test_is, ys_test_is
        if data_tag is None:
            return ans
        else:
            return ans, '(%s_iden)' % data_tag

    def __str__(self):
        return 'iden'

class imputed_test_oos_fitting_eval_getter(object):

    def __init__(self, use_train_is_impute, use_train_oos_training):
        self.use_train_is_impute, self.use_train_oos_training = use_train_is_impute, use_train_oos_training

    def __call__(self, xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos, data_tag=None):
    
        import subgroup
        get_row = subgroup.get_get_row(xs_train_is)
        concat = subgroup.get_concat(xs_train_is)

        if self.use_train_oos_training:
            xs_train_fitting, ys_train_fitting = concat((xs_train_is,xs_train_oos), axis=0), concat((ys_train_is, ys_train_oos), axis=0)
        else:
            xs_train_fitting, ys_train_fitting = xs_train_is, ys_train_is
        xs_test_fitting, ys_test_fitting = xs_test_is, ys_test_is
        xs_eval = xs_test_oos

        if self.use_train_is_impute:
            xs_impute, ys_impute = concat((xs_train_is, xs_train_oos), axis=0), concat((ys_train_is, ys_train_oos), axis=0)
        else:
            xs_impute, ys_impute = xs_train_oos, ys_train_oos

        try:
            dists = np.sum((xs_eval[:,np.newaxis,:] - xs_impute[np.newaxis,:,:])**2, axis=2)
        except TypeError:
            dists = np.sum((xs_eval.values[:,np.newaxis,:] - xs_impute.values[np.newaxis,:,:])**2, axis=2)
        ys_eval = get_row(ys_impute, np.argmin(dists, axis=1))

        ans = xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval

        if data_tag is None:
            return ans
        else:
            return ans, '(%s_test_oos)' % data_tag

    def __str__(self):
        return 'imputed_test_oos_utii=%d_utot=%d' % (self.use_train_is_impute, self.use_train_oos_training)

def eval_predictions_reader(path):
    # fix
#    pdb.set_trace()
    df = pd.read_csv(path, index_col=0, header=0)
    if df.shape[1] == 1:
        return df.iloc[:,0].values
#        return df['ys_hat'].values
    elif df.shape[1] == 2:
        return df['ys_hat'].values, df['ws'].values
#    return np.array(pd.Series.from_csv(path))

def eval_predictions_writer(ans, path):
    if isinstance(ans, tuple):
        assert len(ans) == 2
        ys_hat, ws = ans
        pd.DataFrame({'ys_hat':ys_hat, 'ws':ws}).to_csv(path)
    else:
        ys_hat = ans
        pd.DataFrame({'ys_hat':ys_hat}).to_csv(path)
#        pd.Series(ys_hat).to_csv(path)

def eval_predictions_get_path(identifier, fitter_info, xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval, get_eval_ws, data_tag):
    fitter_name, fitter = fitter_info
    import python_utils.python_utils.caching as caching
    return '%s/%s_%s_get_eval_ws=%s' % (caching.cache_folder, fitter_name, data_tag, get_eval_ws)

eval_predictions_suffix = 'csv'

def eval_predictions(fitter_info, xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval, get_eval_ws, data_tag): # fix, add default None argument to data_tag?
#    assert False
    fitter_name, fitter = fitter_info
    predictor = fitter.fit(xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, data_tag)
    ys_eval_hat = predictor.predict(xs_eval)
    if not get_eval_ws:
        return ys_eval_hat
    else:
        return ys_eval_hat, predictor.get_ws(xs_eval)

def accuracy(ys_hat, ys, ws=None):
    if ws is None:
        ws = np.ones(len(ys))
    return (((ys_hat > 0) == (ys > 0)).astype(float) * ws).sum() / float(len(ys))

def zero_one_loss(ys_hat, ys, ws=None):
    return 1. - accuracy(ys_hat, ys, ws)

from sklearn.metrics import roc_auc_score
def auroc(ys_hat, ys, ws=None):
    try:
        if ws is None:
            return roc_auc_score((ys > 0).astype(int), ys_hat)
        else:
            return roc_auc_score((ys > 0).astype(int), ys_hat, sample_weight=ws)
    except:
        return np.nan

def one_minus_auroc(ys_hat, ys, ws=None):
    return 1. - auroc(ys_hat, ys, ws)

def squared_error(ys_hat, ys, ws=None):
    if ws is None:
        ws = np.ones(len(ys))
    return np.mean(((ys_hat - ys)**2) * ws)

def abs_error(ys_hat, ys, ws=None):
    if ws is None:
        ws = np.ones(len(ys))
    return np.mean(np.abs((ys_hat - ys)) * ws)

class cv_cov_shift_fitter(object):

    def __init__(self, fitter_infos, cv_getter, fitting_eval_getter, num_folds, cache, recompute, metric, get_eval_ws=False):
        self.fitter_infos, self.cv_getter, self.fitting_eval_getter, self.num_folds, self.cache, self.recompute, self.metric, self.get_eval_ws = fitter_infos, cv_getter, fitting_eval_getter, num_folds, cache, recompute, metric, get_eval_ws

    def fit(self, xs_train, xs_test, ys_train, ys_test, data_tag=None):

        if self.cache:
#            pdb.set_trace()
            assert not (data_tag is None)
            original_data_tag = data_tag
            import python_utils.python_utils.caching as caching
            f = caching.switched_decorator(eval_predictions, True, self.recompute, eval_predictions_reader, eval_predictions_writer, eval_predictions_get_path, eval_predictions_suffix)
        else:
            f = eval_predictions

        cv_start_time = time.time()

        d = collections.defaultdict(list)

        for fitter_info in self.fitter_infos:

            fitter_name, fitter = fitter_info

            for i in xrange(self.num_folds):

                if self.cache:
                    (xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos), data_tag = self.cv_getter(i, xs_train, xs_test, ys_train, ys_test, data_tag=original_data_tag)
                    (xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval), data_tag  = self.fitting_eval_getter(xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos, data_tag)                
                else:
                    xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos = self.cv_getter(i, xs_train, xs_test, ys_train, ys_test)
                    xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval  = self.get_fitting_and_eval_data(xs_train_is, xs_test_is, ys_train_is, ys_test_is, xs_train_oos, xs_test_oos, ys_train_oos, ys_test_oos)
#                pdb.set_trace()
                if self.get_eval_ws:
                    ys_eval_hat, ws_eval_hat = f(fitter_info, xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval, True, data_tag)
                    d[fitter_name].append(self.metric(ys_eval_hat, ys_eval, ws_eval_hat))
                else:
#                    pdb.set_trace()
                    ys_eval_hat = f(fitter_info, xs_train_fitting, xs_test_fitting, ys_train_fitting, ys_test_fitting, xs_eval, ys_eval, False, data_tag)
#                    pdb.set_trace()
                    d[fitter_name].append(self.metric(ys_eval_hat, ys_eval))

        cv_end_time = time.time()

        import pandas as pd
        mean_series = pd.DataFrame(d).mean(axis=0)
        std_series = pd.DataFrame(d).std(axis=0)
        chosen_series = (mean_series == mean_series.min()).astype(float)
        best_fitter_name = mean_series.argmin()
        best_fitter = dict(self.fitter_infos)[best_fitter_name]

        final_start_time = time.time()
#        pdb.set_trace()
        self.best_predictor = best_fitter.fit(xs_train, xs_test, ys_train, ys_test, original_data_tag)
        final_end_time = time.time()

        self.info = dict(
            [(('mean_metric',fitter_name),val) for (fitter_name,val) in mean_series.iteritems()] +
            [(('std_metric',fitter_name),val) for (fitter_name,val) in std_series.iteritems()] +
            [(('chosen',fitter_name),val) for (fitter_name,val) in chosen_series.iteritems()]
            )

        self.info['cv_time'] = cv_end_time - cv_start_time
        self.info['cv_final_fit_time'] = final_end_time - final_start_time
        self.info['train_size'] = len(xs_train)
        self.info['test_size'] = len(xs_test)

        print 'beg'
#        print pd.DataFrame(d)
        print mean_series
#        print best_fitter_name, best_fitter
        print 'best'

        return self

    def get_ws(self, xs):
        return self.best_predictor.get_ws(xs)

    def predict(self, xs):
        start_time = time.time()
        ys_hat = self.best_predictor.predict(xs)
        end_time = time.time()
        self.info.update(self.best_predictor.info)
        self.info['cv_predict_time'] = end_time - start_time
        self.info['predict_size'] = len(xs)
        return ys_hat

def VI(loss, xs, ys, predictor, num_trials=1):
    if loss == 'logistic':
        def total_loss(xs, ys, predictor):
            logits = predictor.predict(xs)
            print 'logits', logits
            return np.mean(np.log(1 + np.exp(-ys * logits)))
    elif loss == 'square':
        def total_loss(xs, ys, predictor):
            ys_hat = predictor.predict(xs)
            return np.mean((ys - ys_hat) ** 2)

    D = xs.shape[1]
    in_order = np.arange(D)
    original_loss = total_loss(xs, ys, predictor)
    import copy

    b = np.dot(predictor.info['B'],predictor.info['b'][:-1])
    import python_utils.python_utils.caching as caching
    caching.fig_archiver.log_text('b', b)

    VIss = []
    for i in xrange(num_trials):
        VIs = np.zeros(D)
        shuffled_order = np.random.shuffle(np.arange(D))
        for j in xrange(xs.shape[1]):
            original_col = copy.deepcopy(xs[:,j])
#            print 'orig_col', original_col
            shuffled_col = copy.deepcopy(xs[:,j])
            np.random.shuffle(shuffled_col)
#            print 'shuf_col', shuffled_col
            xs[:,j] = shuffled_col
            VIs[j] = total_loss(xs, ys, predictor) - original_loss
#            print total_loss(xs, ys, predictor), original_loss, b[j]
            xs[:,j] = original_col
#            print total_loss(xs, ys, predictor)
        VIss.append(VIs)
    return np.mean(VIss, axis=0)
