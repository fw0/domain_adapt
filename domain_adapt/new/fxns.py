import autograd.numpy as np
import autograd
import copy, itertools, pdb
import kmm, qp, utils, optimizers
import scipy, scipy.optimize
import python_utils.python_utils.basic as basic
from autograd.core import primitive

NA = np.newaxis


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
            pass
        try:
            val, grad = self._val_and_grad(*args)
            assert False
            return val
        except NotImplementedError:
            pass
        raise NotImplementedError

    def __call__(self, *args):
        return self.val(*args)

    @basic.timeit('grad')
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
        def fxn_vjps(argnum, g, ans, vs, gvs, *args):
            return np.dot(g, f.grad(*args, care_argnums=(argnum,))) # fix: f assumed to output 1d vector
        max_argnum = 20
        wrapped.defvjps(fxn_vjps, range(max_argnum))
        return wrapped
    
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
    us = np.dot(xs, B)
    ys_prime = ys * (ws**0.5)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    us_prime = us * ((ws**0.5)[:,np.newaxis])
#    pdb.set_trace()
    #c = 0.01 # fix
    try:
        b_opt_lsqr = np.linalg.solve(np.dot(us_prime.T, us_prime) + len(xs)*c*np.eye(us_prime.shape[1]), np.dot(us_prime.T, ys_prime))
    except:
        pdb.set_trace()
#    print ws
    b_opt = np.dot(np.dot(np.linalg.inv(np.dot(us_prime.T, us_prime) + len(xs)*c*np.eye(us_prime.shape[1])), us_prime.T), ys_prime) # fix: get rid of inverse, solve linear system instead
    #print b_opt_lsqr, b_opt
    try:
        assert np.linalg.norm(b_opt_lsqr-b_opt) < 0.001
    except:
        pdb.set_trace()
    #pdb.set_trace()
    return b_opt


def weighted_lsqr_b_opt_given_b_logreg(B, xs_train, xs_test, ys_train, b_logreg, c_lsqr, sigma, scale_sigma, max_ratio=5.):

#    scale_sigma = False

    ws = b_to_logreg_ratios_wrapper(b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio=5.)

    us = np.dot(xs_train, B)
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
    us = np.dot(xs, B)
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

    us = np.dot(xs_train, B)
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
    us = np.dot(xs, B)
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
    us_train = np.dot(xs_train, B)
    us_test = np.dot(xs_test, B)
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
    us_train = np.dot(xs_train, B)
    us_basis = np.dot(xs_basis, B)
    train_basis_vals = utils.get_gaussian_K(sigma, us_train, us_basis, nystrom=False)
    ans = np.dot(train_basis_vals, lsif_alpha)
    ans = np.maximum(0.01, ans)
    #ans = np.maximum(0, ans)
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

    us = np.dot(xs_train, B)
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


def weight_reg_given_lsif_alpha(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio):
    ws = lsif_alpha_to_ratios(lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)
    return np.dot(ws, ws)


def expected_conditional_PE_dist(full_ws, ws):
    N = len(full_ws)
    assert len(full_ws) == len(ws)
    return np.sum(((1./N) * (full_ws**2) / ws), axis=0) - 1.

def weight_reg(ws):
    return np.dot(ws, ws)


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
        us_train = np.dot(xs_train, B)
        us_test = np.dot(xs_test, B)
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
        us_train = np.dot(xs_train, B)
        us_test = np.dot(xs_test, B)
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
    us_train = np.dot(xs_train, B)
    us_test = np.dot(xs_test, B)
    us_basis = np.dot(xs_basis, B)
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

class LSIF_objective(quad_objective):

    def Pq(self, xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg):
        us_train = np.dot(xs_train, B)
        us_test = np.dot(xs_test, B)
        us_basis = np.dot(xs_basis, B)
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
        us_train = np.dot(xs_train, B)
        us_test = np.dot(xs_test, B)
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

    def __init__(self, num_vars):
        self.num_vars = num_vars

    def _val(self, *args):
        return np.zeros((len(args[-1]),0))
        
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
        opt.__init__(self, lin_solver, objective, dobjective_dx)

    @property
    def ineq_constraints(self):
        return unconstrained_ineq_constraints(self.objective.arg_shape(*args)[0])
#        return np.zeros(shape=(0,self.objective.arg_shape(*args)[0])), np.zeros(shape=(0,))
    
    def forward_prop(self, *args):
        f = lambda x: self.objective.val(*(args + (x,)))
        df_dx = lambda x: self.dobjective_dx.val(*(args + (x,)))
        if not self.warm_start:
            x0=np.random.normal(size=self.objective.arg_shape(*args))
        else:
            try:
                x0 = self.old_x_opt
            except AttributeError:
                x0 = np.random.normal(size=self.objective.arg_shape(*args))
        logger = utils.optimizer_logger(f, df_dx)
        val = self.optimizer.optimize(f, df_dx, x0)
        if self.warm_start:
            self.old_x_opt = val
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
#                pdb.set_trace()
                from cvxopt import matrix
                try:
                    primal_val, eq_dual_val, ineq_dual_val = qp.cvxopt_solver(P, q, G, h, initvals={'x':matrix(init_x), 'z':matrix(init_ineq_dual_val)})
                except:
                    print P
                    pdb.set_trace()
            except AttributeError:
                primal_val, eq_dual_val, ineq_dual_val = qp.cvxopt_solver(P, q, G, h)
            self.old_x, self.old_ineq_dual_val = primal_val, ineq_dual_val
        else:
            primal_val, eq_dual_val, ineq_dual_val = qp.cvxopt_solver(P, q, G, h)
        return ((P,q,G,h),(eq_dual_val,ineq_dual_val)), primal_val # fix: can add dual optimal variables here


def kmm_get_Gh(xs_train, xs_test, sigma, B, w_max, eps):
    return kmm.get_kmm_Gh(w_max, eps, len(xs_train), len(xs_test))


def LSIF_get_Gh(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio):
    num_bases = len(xs_basis)
    G_pre, h_pre = -1. * np.eye(num_bases), np.zeros(num_bases)
#    return G_pre, h_pre
    us_train = np.dot(xs_train, B)
    us_basis = np.dot(xs_basis, B)
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
        Cu_pre += np.dot((f_dual_val[:,np.newaxis] * dineq_dx_opt_val).T, u_post)
        #Cu_post = np.dot(G_tight, u_pre)
#        Cu_post = np.dot(f_dual_val[:,np.newaxis] * dineq_dx_opt_val, u_pre) # new
        Cu_post = np.dot(dineq_dx_opt_val, u_pre) # new
        #Cu_post += (np.dot(G,f_val) - h) * u_post # new
        Cu_post += ineq_val * u_post # new
        Cu = np.concatenate((Cu_pre, Cu_post), axis=0)
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
            f_args = [args[i] for i in f_argnums]
            f_val = f.val(*f_args)
            f_vals.append(f_val)
            ans += w * f_val
        return f_vals, ans

    def backward_prop(self, args, f_vals, val, care_argnums):
        ans = 0.
#        print [f._val for f in self.fs], care_argnums
#        pdb.set_trace()
        for (f,f_argnums,w) in itertools.izip(self.fs, self.fs_argnums, self.weights):
            f_args = [args[i] for i in f_argnums]
            try:
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

        return map(get, self.g_argnums)
        
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
#            pdb.set_trace()
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
