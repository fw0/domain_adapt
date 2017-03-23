import autograd.numpy as np
import autograd
import copy, itertools, pdb
import kmm, qp, utils, optimizers
import scipy, scipy.optimize

NA = np.newaxis


class fxn(object):

    def __init__(self, forward_prop=None, backward_prop=None, _val=None, _grad=None, _val_and_grad=None):
        if not (forward_prop is None): self.forward_prop = forward_prop
        if not (backward_prop is None): self.backward_prop = backward_prop
        if not (_val is None): self._val = _val
        if not (_grad is None): self._grad = _grad
        if not (_val_and_grad is None): self._val_and_grad = _val_and_grad

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
            return val
        except NotImplementedError:
            pass
        try:
            return self._val(*args)
        except NotImplementedError:
            pass
        try:
            val, grad = self._val_and_grad(*args)
            return grad
        except NotImplementedError:
            pass
        raise NotImplementedError

    def __call__(self, *args):
        return self.val(*args)

    def grad(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums', range(len(args)))
        self.check_care_argnums(args, care_argnums)
        try:
            return self._grad(*args, care_argnums=care_argnums)
        except NotImplementedError:
            pass
        try:
            val, grad = self._val_and_grad(*args, care_argnums=care_argnums)
            return grad
        except NotImplementedError:
            pass
        raise NotImplementedError

    def val_and_grad(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums', range(len(args)))
        self.check_care_argnums(args, care_argnums)
        try:
            return self._val_and_grad(*args, care_argnums=care_argnums)
        except NotImplementedError:
            pass
        try:
            return self.val(*args), self.grad(*args, care_argnums=care_argnums)
        except NotImplementedError:
            pass
        raise NotImplementedError

    def grad_check(self, *args, **kwargs):
        care_argnums = kwargs.get('care_argnums', range(len(args)))
        self.check_care_argnums(args, care_argnums)
        #delta = 0.01
        delta = 0.001
        tol = 1.
        val, anal_grad = self.val_and_grad(*args, care_argnums=care_argnums)
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

    def check_care_argnums(self, args, care_argnums):
        return True

    @classmethod
    def autograd_fxn(cls, *args, **kwargs):
        inst = cls(*args, **kwargs)
        inst._grad = get_autograd_grad(inst.val)
        #inst._val_and_grad = get_autograd_val_and_grad(inst.val)
        return inst


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
        ans = autograd.jacobian(horse)(care_args, nocare_args)
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

def weighted_lsqr_loss(B, xs, ys, ws, c):
    b_opt = weighted_lsqr_b_opt(B, xs, ys, ws, c)
    return weighted_squared_loss_given_b_opt(B, xs, ys, ws, b_opt, c)
#    N = xs.shape[0]
#    W = ws * np.eye(N)
#    us = np.dot(xs, B)
#    ys_hat = np.dot(us, b_opt)
#    error = ys - ys_hat
#    return np.dot(ws, error * error) + (np.dot(b_opt, b_opt) * c)
#    return np.sum(np.dot(ys*ws - np.dot(ys*ws,np.dot(np.dot(us,np.linalg.inv(np.dot(us.T*ws,us))),us.T*ws)),ys))


def weighted_lsqr_b_opt(B, xs, ys, ws, c):
    us = np.dot(xs, B)
    ys_prime = ys * (ws**0.5)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    us_prime = us * ((ws**0.5)[:,np.newaxis])
#    pdb.set_trace()
    #c = 0.01 # fix
    b_opt = np.dot(np.dot(np.linalg.inv(np.dot(us_prime.T, us_prime) + c*np.eye(us_prime.shape[1])), us_prime.T), ys_prime) # fix: get rid of inverse, solve linear system instead
    return b_opt

def weighted_squared_loss_given_b_opt(B, xs, ys, ws, b_opt, c):
    us = np.dot(xs, B)
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    ys_hat = np.dot(us, b_opt)
    error = ys - ys_hat
    return np.dot(ws, error * error) + (np.dot(b_opt, b_opt) * c)


def weighted_squared_loss_given_B(B, xs, ys, c, ws=None):
    # B is assumed to be 1-d projection
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
    ys_hat = np.dot(us, b_opt)
    error = ys - ys_hat
    return error * error

def B_to_squared_losses(B, xs, ys):
    assert len(B.shape) == 1
    ys_hat = np.dot(xs, B)
    error = ys - ys_hat
    return error * error


def weighted_lsqr_loss_given_f(B, xs, ys, f, ws, c):
    diff = np.dot(xs, np.dot(B, f)) - ys
    return np.dot(ws, diff * diff) + (c * np.dot(f,f))


def b_to_logreg_ratios(b_logreg, xs_train, xs_test, sigma, B):
    scale_sigma = False
    if scale_sigma:
        assert len(B.shape) == 1
        sigma = sigma / np.linalg.norm(B)
    us_train = np.dot(xs_train, B)
    us_test = np.dot(xs_test, B)
    us = np.concatenate((us_train, us_test))
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    K = utils.get_gaussian_K(sigma, us, us)
    logits = np.dot(K, b_logreg)
    logits_train = logits[:len(xs_train)]
    ps_train = 1 / (1+np.exp(-logits_train))
    #return ps_train
    ratios_train = ps_train / (1.-ps_train)
    ratios_train = (ratios_train / np.sum(ratios_train)) * len(xs_train)
    #print b, 'b'
    #print np.sum(b), 'b_sum'
    #print ratios_train, 'ratios_train'
    return ratios_train


def b_to_logreg_ratios_scale_sigma(b, xs_train, xs_test, sigma, B):
    scale_sigma = True
    if scale_sigma:
        assert len(B.shape) == 1
        sigma = sigma / np.linalg.norm(B)
    us_train = np.dot(xs_train, B)
    us_test = np.dot(xs_test, B)
    us = np.concatenate((us_train, us_test))
    if len(us.shape) == 1:
        us = us.reshape((len(us),1))
    K = utils.get_gaussian_K(sigma, us, us)
    logits = np.dot(K, b)
    logits_train = logits[:len(xs_train)]
    ps_train = 1 / (1+np.exp(-logits_train))
    #return ps_train
    ratios_train = ps_train / (1.-ps_train)
    ratios_train = (ratios_train / np.sum(ratios_train)) * len(xs_train)
    #print b, 'b'
    #print np.sum(b), 'b_sum'
    #print ratios_train, 'ratios_train'
    return ratios_train


def expected_conditional_PE_dist(full_ws, ws):
    N = len(full_ws)
    assert len(full_ws) == len(ws)
    return np.sum(((1./N) * (full_ws**2) / ws), axis=0) - 1.

def weight_reg(ws):
    return np.dot(ws, ws)


def logreg_loss(xs, ys, b, ws=None):
    assert ys.max() == 1 and ys.min() == -1
    ws = np.ones(len(xs)) if ws is None else ws
    losses = np.log(1 + np.exp(-ys * np.dot(xs, b)))
    assert len(losses.shape) == 1
    return np.sum(ws * losses, axis=0)


class objective(fxn):

    def arg_shape(self):
        raise NotImplementedError


class logreg_ratio_objective(objective):

    def __init__(self, *args, **kwargs):
        self.scale_sigma = kwargs.get('scale_sigma', False)
#        objective.__init__(self, *args, **kwargs)
    
    def arg_shape(self, xs_train, xs_test, sigma, B, logreg_c):
        #pdb.set_trace()
        #return (B.shape[1],)
        return (xs_train.shape[0] + xs_test.shape[0],)
        #return B.shape[1]
    
    def _val(self, xs_train, xs_test, sigma, B, logreg_c, b):
        if self.scale_sigma:
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
        return logreg_loss(K, zs, b) + (logreg_c * np.dot(b,b))
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
        val = (0.5 * np.dot(x.T, np.dot(P, x))) + np.dot(q, x)
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

    def _val(self, *args):
        num_args = len(args)
        return self.objective.grad(*args, care_argnums=(num_args-1,))
        

def get_tight_constraints(A, b, x):
    LHS = np.dot(A, x)
    assert (LHS < b).all()
    tight_eps = 0.01
    tight = (b - LHS) < tight_eps
    print 'num_tight:', np.sum(tight)
    return A[tight], b[tight]
        

def get_dx_opt_delta_p(lin_solver, d_dp_df_dx_val, P, G, h, x_opt, p, delta_p_direction):
    
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
    print 'solver error:', np.linalg.norm(np.dot(C,deriv) - d)

    return deriv
        

class opt(fxn):

    def __init__(self, lin_solver, objective, dobjective_dx):
        self.lin_solver, self.objective, self.dobjective_dx = lin_solver, objective, dobjective_dx

    def get_Gh(self, *args):
        raise NotImplementedError

    def backward_prop(self, args, _, val, care_argnums):
        num_args = len(args)
        self.check_care_argnums(args, care_argnums) # special
        #p = args[-1]
        p = args[care_argnums[0]]
        G,h = self.get_Gh(*args)
        #d_dp_df_dx_val = self.dobjective_dx.grad(*(args+(val,)), care_argnums=(num_args-1,)) # hardcoded - should be able to take grad wrt to other arguments
        d_dp_df_dx_val = self.dobjective_dx.grad(*(args+(val,)), care_argnums=care_argnums) # hardcoded - should be able to take grad wrt to other arguments
        d_dx_df_dx_val = self.dobjective_dx.grad(*(args+(val,)), care_argnums=(num_args,))
        
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

    def __init__(self, lin_solver, objective, dobjective_dx, optimizer=optimizers.scipy_minimize_optimizer()):
        self.optimizer = optimizer
        opt.__init__(self, lin_solver, objective, dobjective_dx)

    def get_Gh(self, *args):
        return np.zeros(shape=(0,self.objective.arg_shape(*args)[0])), np.zeros(shape=(0,))
    
    def forward_prop(self, *args):
        f = lambda x: self.objective.val(*(args + (x,)))
        df_dx = lambda x: self.dobjective_dx.val(*(args + (x,)))
        x0=np.random.normal(size=self.objective.arg_shape(*args))
        logger = utils.optimizer_logger(f, df_dx)
        val = self.optimizer.optimize(f, df_dx, x0)
#        ans = scipy.optimize.minimize(fun=f, x0=np.random.normal(size=self.objective.arg_shape(*args)), jac=df_dx, method='L-BFGS-B', options={'disp':10}, callback=logger)
        #logger.display_df()
        #logger.plot()
#        print ans['x'], 'hhhhhhhh'
#        print ans
        return None, val
        
        
class full_quad_opt(opt):
    """
    assumes objective fxn being maximized is quadratic and parameterized by the full quadratic term matrix
    cvxopt us used to maximize the objective
    """
    def forward_prop(self, *args):
        P, q = self.objective.Pq(*args)
        G, h = self.get_Gh(*args)
        return (P,q,G,h), qp.cvxopt_solver(P, q, G, h)

    
class full_ws_opt_given_B(full_quad_opt):

    def __init__(self, w_max, eps, lin_solver, kmm_objective, dkmm_objective_dws):
        self.w_max, self.eps = w_max, eps
        self.lin_solver, self.objective, self.dobjective_dx = lin_solver, kmm_objective, dkmm_objective_dws
    
    def get_Gh(self, xs_train, xs_test, sigma, B):
        return kmm.get_kmm_Gh(self.w_max, self.eps, len(xs_train), len(xs_test))

    
def get_dg_dp_thru_x_opt(lin_solver, d_dp_df_dx_val, dg_dx_opt_val, P, G, h, x_opt, p):
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
    print 'solver error:', np.linalg.norm(np.dot(C,v) - d)

    # make D
    D = np.vstack((-d_dp_df_dx_val, np.zeros((num_tight,)+p.shape)))

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
        return tuple([args[i] for i in self.g_argnums])

    def h_args(self, g_val, *args):
        l = [args[i] for i in self.h_argnums]
        l.insert(self.g_val_h_argnum, g_val)
        return tuple(l)
    
    def forward_prop(self, *args):
        g_args = self.g_args(*args)
        _, g_val = self.g.forward_prop(*g_args)
        h_args = self.h_args(g_val, *args)
        _, h_val = self.h.forward_prop(*h_args)
        return (g_val,), h_val

    def two_step_grad(self, args, (g_val,), h_val, care_argnums):
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
    
    def backward_prop(self, args, (g_val,), h_val, care_argnums):
        
        self.check_care_argnums(args, care_argnums) # special
        p = args[care_argnums[0]]
        dh_dg_dg_dp_val = self.two_step_grad(args, (g_val,), h_val, care_argnums)
#        pdb.set_trace()
        if care_argnums[0] in self.h_argnums:
            care_argnum_in_h_args = list(self.h_argnums).index(care_argnums[0])
            if care_argnum_in_h_args >= self.g_val_h_argnum:
                care_argnum_in_h_args += 1
            h_args = self.h_args(g_val, *args)
            dh_dp_val = self.h.grad(*h_args, care_argnums=(care_argnum_in_h_args,))
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

    def two_step_grad(self, args, (g_val,), h_val, care_argnums):
        p = args[care_argnums[0]]
        if care_argnums[0] in self.g_argnums:
            care_argnum_in_g_args = list(self.g_argnums).index(care_argnums[0])
            g_args = self.g_args(*args)
            h_args = self.h_args(g_val, *args)
    #        d_dp_df_dx_val = self.full_quad_opt.dobjective_dx.grad(*(quad_args+(x_opt,)), care_argnums=(len(quad_args)-1,))
            d_dp_df_dx_val = self.g.dobjective_dx.grad(*(g_args+(g_val,)), care_argnums=(care_argnum_in_g_args,))
            dh_dxopt_val = self.h.grad(*h_args, care_argnums=(self.g_val_h_argnum,)) # hard
            d_dx_df_dx_val = self.g.dobjective_dx.grad(*(g_args+(g_val,)), care_argnums=(len(g_args),)) # hard
            
            #P, q = self.full_quad_opt.objective.Pq(*quad_args)
            G, h = self.g.get_Gh(*g_args)
            dh_dg_dg_dp_val = get_dg_dp_thru_x_opt(self.g.lin_solver, d_dp_df_dx_val, dh_dxopt_val, d_dx_df_dx_val, G, h, g_val, p)
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
