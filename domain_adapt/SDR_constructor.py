import kernels, functools
import autograd
import autograd.numpy as np
import pymanopt, pdb
import scipy
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic

def get_unsupervised_obj_and_gradients():

    # define objective fxn
    #tradeoff = 0.1
    def obj_from_wsopt_and_Ks((Ky, Ku), wsopt):  
        return kernels.ws_distance(wsopt)

    # define necessary gradients or their constructors
    dobj_dwsopt = autograd.jacobian(lambda wsopt, (Ky, Ku): obj_from_wsopt_and_Ks((Ky, Ku), wsopt))

    def get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky):
    
        def obj_from_P_and_wsopt(P, wsopt):
            us_train = np.dot(xs_train, P)
            Ku = SDR_get_K(us_train, us_train)
            return obj_from_wsopt_and_Ks((Ky, Ku), wsopt)

        dobj_dP_thru_Ku = autograd.jacobian(obj_from_P_and_wsopt)
    
        return dobj_dP_thru_Ku

    return obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku

def get_reduce_bias_obj_and_gradients(SDR_tradeoff, SDR_eps, reg, loss):
    # should return 3 gradients.  obj is SDR term + weighted loss

    def obj_from_wsopt_and_Ks_and_B_helper(_SDR_tradeoff, _SDR_eps, _reg, (Ky, Ku), (y, u), wsopt, B):
#        pdb.set_trace()
        return (_SDR_tradeoff * kernels.get_trace_from_ws_and_Ks(_SDR_eps, Ky, Ku)) + loss(y, np.dot(u, B), wsopt) + (reg * np.dot(B.T, B))

    obj_from_wsopt_and_Ks_and_B = functools.partial(obj_from_wsopt_and_Ks_and_B_helper, SDR_tradeoff, SDR_eps, reg)

    dobj_dwsopt = autograd.jacobian(lambda wsopt, (Ky,Ku,y,u,B): obj_from_wsopt_and_Ks_and_B((Ky,Ku), (y,u), wsopt, B))

    def get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky, y): # NEW has y as argument
    
        def obj_from_P_and_wsopt_and_B(P, wsopt, B):
            us_train = np.dot(xs_train, P)
            Ku = SDR_get_K(us_train, us_train)
            return obj_from_wsopt_and_Ks_and_B((Ky, Ku), (y, us_train), wsopt, B)

        dobj_dP_thru_Ku = autograd.jacobian(obj_from_P_and_wsopt_and_B)
    
        return dobj_dP_thru_Ku

    dobj_dB = autograd.jacobian(lambda B, (Ky,Ku), (y,u), wsopt: obj_from_wsopt_and_Ks_and_B((Ky,Ku), (y,u), wsopt, B))

    return obj_from_wsopt_and_Ks_and_B, dobj_dwsopt, get_dobj_dP_thru_Ku, dobj_dB

def get_SDR_least_squares_obj_and_gradients(SDR_tradeoff, SDR_eps, dist_tradeoff, reg, weighted=True):

    # define objective fxn
    #tradeoff = 0.1
    def obj_from_wsopt_and_Ks_helper(_SDR_tradeoff, _SDR_eps, _dist_tradeoff, _reg, (Ky, Ku), (y, u), wsopt):  
        if weighted:
            pdb.set_trace()
            return kernels.weighted_lsqr_loss(wsopt, Ku, Ky)
#            return (_SDR_tradeoff * kernels.get_trace_from_ws_and_Ks(_SDR_eps, Ky, Ku)) + (_dist_tradeoff * kernels.ws_distance(wsopt)) + kernels.weighted_lsqr_loss(wsopt, Ku, Ky)
        else:
            return (_SDR_tradeoff * kernels.get_trace_from_ws_and_Ks(_SDR_eps, Ky, Ku)) + (_dist_tradeoff * kernels.ws_distance(wsopt)) + kernels.weighted_lsqr_loss(np.ones(len(Ku), Ku, Ky))


    obj_from_wsopt_and_Ks = functools.partial(obj_from_wsopt_and_Ks_helper, SDR_tradeoff, SDR_eps, dist_tradeoff, reg)

#    if weighted:
#        obj_from_wsopt_and_Ks = lambda (Ky, Ku), wsopt:  kernels.ws_distance(wsopt) + (tradeoff * kernels.weighted_lsqr_loss(wsopt, Ku, Ky))
#        obj_from_wsopt_and_Ks = lambda (Ky, Ku), wsopt: kernels.ws_distance(wsopt) + (tradeoff * kernels.weighted_lsqr_loss(wsopt, Ku, Ky))
#    else:
#        obj_from_wsopt_and_Ks = lambda (Ky, Ku), wsopt: kernels.ws_distance(wsopt) + (tradeoff * kernels.weighted_lsqr_loss(np.ones(len(Ku)), Ku, Ky))

    # define necessary gradients or their constructors
    dobj_dwsopt = autograd.jacobian(lambda wsopt, (Ky, Ku,y,u): obj_from_wsopt_and_Ks((Ky, Ku), (y,u), wsopt))

    def get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky, y):
    
        def obj_from_P_and_wsopt(P, wsopt):
            us_train = np.dot(xs_train, P)
            Ku = SDR_get_K(us_train, us_train)
            return obj_from_wsopt_and_Ks((Ky, Ku), (y,us_train), wsopt)

        dobj_dP_thru_Ku = autograd.jacobian(obj_from_P_and_wsopt)
    
        return dobj_dP_thru_Ku

    return obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku

def get_least_squares_obj_and_gradients(tradeoff, reg, weighted=True):

    # define objective fxn
    #tradeoff = 0.1
    def obj_from_wsopt_and_Ks_helper(_tradeoff, (Ky, Ku), wsopt):  
        if weighted:
#            pdb.set_trace()
            return kernels.weighted_lsqr_loss(wsopt, Ku, Ky)
#            return kernels.ws_distance(wsopt) + (_tradeoff * kernels.weighted_lsqr_loss(wsopt, Ku, Ky))
        else:
            return kernels.ws_distance(wsopt) + (_tradeoff * kernels.weighted_lsqr_loss(np.ones(len(Ku)), Ku, Ky))

    obj_from_wsopt_and_Ks = functools.partial(obj_from_wsopt_and_Ks_helper, tradeoff)

#    if weighted:
#        obj_from_wsopt_and_Ks = lambda (Ky, Ku), wsopt:  kernels.ws_distance(wsopt) + (tradeoff * kernels.weighted_lsqr_loss(wsopt, Ku, Ky))
#        obj_from_wsopt_and_Ks = lambda (Ky, Ku), wsopt: kernels.ws_distance(wsopt) + (tradeoff * kernels.weighted_lsqr_loss(wsopt, Ku, Ky))
#    else:
#        obj_from_wsopt_and_Ks = lambda (Ky, Ku), wsopt: kernels.ws_distance(wsopt) + (tradeoff * kernels.weighted_lsqr_loss(np.ones(len(Ku)), Ku, Ky))

    # define necessary gradients or their constructors
    dobj_dwsopt = autograd.jacobian(lambda wsopt, (Ky, Ku): obj_from_wsopt_and_Ks((Ky, Ku), wsopt))

    def get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky):
    
        def obj_from_P_and_wsopt(P, wsopt):
            us_train = np.dot(xs_train, P)
            Ku = SDR_get_K(us_train, us_train)
            return obj_from_wsopt_and_Ks((Ky, Ku), wsopt)

        dobj_dP_thru_Ku = autograd.jacobian(obj_from_P_and_wsopt)
    
        return dobj_dP_thru_Ku

    return obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku


def get_SDR_logreg_obj_and_gradients(lin_solver, cvxopt_solver, SDR_tradeoff, SDR_eps, dist_tradeoff, reg, weighted=True):

    # define objective fxn, which is the objective Bopt optimizes (Bobj) + tradeoff * ws_distance

    #reg =  .1
#    Bobj_from_Ks_and_wsopt_and_Bopt = functools.partial(kernels.weighted_logreg_loss, reg)
    Bobj_from_Ks_and_wsopt_and_Bopt = lambda wsopt, (Ky, SDR_Ku), (y,u), Bopt: kernels.logloss(y, np.dot(u, Bopt), wsopt) + (0.5 * reg * np.dot(Bopt, Bopt))

    def obj_from_Ks_and_wsopt_and_Bopt_helper(_Bobj_from_Ks_and_wsopt_and_Bopt, _SDR_tradeoff, _SDR_eps, _dist_tradeoff, _reg, (Ky, SDR_Ku), (y,u), wsopt, Bopt):
        if not weighted:
            return (_SDR_tradeoff * kernels.get_trace_from_ws_and_Ks(_SDR_eps, Ky, SDR_Ku)) + (_dist_tradeoff * kernels.ws_distance(wsopt)) + _Bobj_from_Ks_and_wsopt_and_Bopt(wsopt,(Ky, SDR_Ku), (y,u), Bopt)
        else:
            return (_SDR_tradeoff * kernels.get_trace_from_ws_and_Ks(_SDR_eps, Ky, SDR_Ku, wsopt)) + (_dist_tradeoff * kernels.ws_distance(wsopt)) + _Bobj_from_Ks_and_wsopt_and_Bopt(wsopt,(Ky, SDR_Ku), (y,u), Bopt)


    #tradeoff = 1.
    obj_from_Ks_and_wsopt_and_Bopt = functools.partial(obj_from_Ks_and_wsopt_and_Bopt_helper, Bobj_from_Ks_and_wsopt_and_Bopt, SDR_tradeoff, SDR_eps, dist_tradeoff, reg)
#    obj_from_Ks_and_wsopt_and_Bopt = lambda (Ky, SDR_Ku), wsopt, Bopt: 1. * Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_Ku), Bopt) #+ tradeoff * kernels.ws_distance(wsopt)

    #get_Bopt_alternate = functools.partial(kernels.weighted_logreg_get_Bopt, reg)
    def get_Bopt(ws, (Ky,Ku), (ys, us)):
        loss = lambda B: Bobj_from_Ks_and_wsopt_and_Bopt(ws, (Ky,Ku), (ys, us), B)
        grad = autograd.grad(loss)
        ans = scipy.optimize.minimize(loss, np.random.normal(size=us.shape[1]),jac=grad)
        return ans['x']
    
    def obj_from_wsopt_and_Ks_helper(_obj_from_Ks_and_wsopt_and_Bopt, (Ky, SDR_Ku), (y,u), wsopt):
        return _obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), (y,u), wsopt, get_Bopt(wsopt, (Ky, SDR_Ku), (y, u)))

#    obj_from_wsopt_and_Ks = lambda (Ky, SDR_Ku), wsopt: obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), wsopt, get_Bopt(wsopt, (Ky, SDR_Ku)))
    obj_from_wsopt_and_Ks = functools.partial(obj_from_wsopt_and_Ks_helper, obj_from_Ks_and_wsopt_and_Bopt)

    # define dobj_dwsopt

    # below terms depend on Bobj
    the_g = lambda B, wsopt, (Ky, SDR_Ku, y,u): Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_Ku), (y,u), B)
    dg_dB = autograd.jacobian(the_g) # ans dim: |x|
    d_dwsopt_dg_dB = autograd.jacobian(lambda wsopt,B,(Ky, SDR_Ku,y,u): dg_dB(B,wsopt,(Ky, SDR_Ku,y,u))) # ans dim: |x| x |p|.  DIFF below
    d_dB_dg_dB = autograd.jacobian(autograd.jacobian(the_g))

    def asdf(Bopt, (wsopt, ((Ky, SDR_Ku),(y,u)))): 
        return obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), (y,u), wsopt, Bopt)
    dobj_dBopt = autograd.jacobian(asdf)
#    dobj_dBopt = autograd.jacobian(lambda Bopt, (wsopt, (Ky, SDR_Ku,y,u)): obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), (y,u), wsopt, Bopt))
#    dobj_dBopt = autograd.jacobian(lambda Bopt, (wsopt, (Ky, SDR_Ku,y,u)): obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), (y,u), wsopt, Bopt))

    def obj_from_wsopt_and_Bopt_and_Ks(wsopt, Bopt, (Ky, SDR_Ku),(y,u)):
        return obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), (y,u), wsopt, Bopt)

    dobj_dwsopt_thru_wsopt = autograd.jacobian(obj_from_wsopt_and_Bopt_and_Ks)

    use_yu = True
    dobj_dwsopt = functools.partial(kernels.Bopt_get_dobj_dwsopt, use_yu, get_Bopt, dobj_dwsopt_thru_wsopt, lin_solver, cvxopt_solver, dg_dB, d_dwsopt_dg_dB, d_dB_dg_dB, dobj_dBopt)#, wsopt, (Ky, SDR_Ku)) FIX?

    # define constructor for dobj_dP_thru_Ku

    def get_dobj_dP_thru_Ku_stuff(xs_train, SDR_get_K, Ky, y):

        def the_g2(B, P, (wsopt,Ky,y)):
            us_train = np.dot(xs_train, P)
            return Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_get_K(us_train,us_train)),(y,us_train), B)
    
        dg_dB = autograd.jacobian(the_g2)
        d_dP_dg_dB = autograd.jacobian(lambda P, B, (wsopt,Ky,y): dg_dB(B, P, (wsopt,Ky,y)))
    
        d_dB_dg_dB = autograd.jacobian(autograd.jacobian(the_g2))
    
        dobj_dP_thru_Ku = autograd.jacobian(lambda P, wsopt, Bopt: obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_get_K(np.dot(xs_train,P),np.dot(xs_train,P))),(y,np.dot(xs_train,P)), wsopt, Bopt))
    
        return dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, dobj_dP_thru_Ku

    def get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky, y):
    
        dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, dobj_dP_thru_Ku = get_dobj_dP_thru_Ku_stuff(xs_train, SDR_get_K, Ky, y)
    
        use_yu = True
        return functools.partial(kernels.Bopt_get_dobj_dP, use_yu, xs_train, Ky, y, SDR_get_K, get_Bopt, dobj_dP_thru_Ku, lin_solver, cvxopt_solver, dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt)#, P, wsopt)

    return obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku


def get_logreg_obj_and_gradients(lin_solver, cvxopt_solver, tradeoff, reg, weighted=True):

    # define objective fxn, which is the objective Bopt optimizes (Bobj) + tradeoff * ws_distance

    #reg =  .1
#    Bobj_from_Ks_and_wsopt_and_Bopt = functools.partial(kernels.weighted_logreg_loss, reg)
    Bobj_from_Ks_and_wsopt_and_Bopt = lambda wsopt, (Ky, SDR_Ku), Bopt: kernels.logloss(Ky, np.dot(SDR_Ku, Bopt), wsopt) + (0.5 * reg * np.dot(B, B))

    def obj_from_Ks_and_wsopt_and_Bopt_helper(_Bobj_from_Ks_and_wsopt_and_Bopt, _tradeoff, (Ky, SDR_Ku), wsopt, Bopt):
        if not weighted:
            wsopt = np.ones(len(wsopt), dtype=float)
        return 1. * _Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_Ku), Bopt) + _tradeoff * kernels.ws_distance(wsopt)

    #tradeoff = 1.
    obj_from_Ks_and_wsopt_and_Bopt = functools.partial(obj_from_Ks_and_wsopt_and_Bopt_helper, Bobj_from_Ks_and_wsopt_and_Bopt, tradeoff)
#    obj_from_Ks_and_wsopt_and_Bopt = lambda (Ky, SDR_Ku), wsopt, Bopt: 1. * Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_Ku), Bopt) #+ tradeoff * kernels.ws_distance(wsopt)

    #get_Bopt_alternate = functools.partial(kernels.weighted_logreg_get_Bopt, reg)
    def get_Bopt(ws, (ys, xs)):
        loss = lambda B: Bobj_from_Ks_and_wsopt_and_Bopt(ws, (ys, xs), B)
        grad = autograd.grad(loss)
        ans = scipy.optimize.minimize(loss, np.random.normal(size=xs.shape[1]),jac=grad)
        return ans['x']
    
    def obj_from_wsopt_and_Ks_helper(_obj_from_Ks_and_wsopt_and_Bopt, (Ky, SDR_Ku), wsopt):
        return _obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), wsopt, get_Bopt(wsopt, (Ky, SDR_Ku)))

#    obj_from_wsopt_and_Ks = lambda (Ky, SDR_Ku), wsopt: obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), wsopt, get_Bopt(wsopt, (Ky, SDR_Ku)))
    obj_from_wsopt_and_Ks = functools.partial(obj_from_wsopt_and_Ks_helper, obj_from_Ks_and_wsopt_and_Bopt)

    # define dobj_dwsopt

    # below terms depend on Bobj
    the_g = lambda B, wsopt, (Ky, SDR_Ku): Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_Ku), B)
    dg_dB = autograd.jacobian(the_g) # ans dim: |x|
    d_dwsopt_dg_dB = autograd.jacobian(lambda wsopt,B,(Ky, SDR_Ku): dg_dB(B,wsopt,(Ky, SDR_Ku))) # ans dim: |x| x |p|.  DIFF below
    d_dB_dg_dB = autograd.jacobian(autograd.jacobian(the_g))

    dobj_dBopt = autograd.jacobian(lambda Bopt, (wsopt, (Ky, SDR_Ku)): obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), wsopt, Bopt)) # FIX? should it be Bobj, not obj?

    def obj_from_wsopt_and_Bopt_and_Ks(wsopt, Bopt, (Ky, SDR_Ku)):
        return obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_Ku), wsopt, Bopt)

    dobj_dwsopt_thru_wsopt = autograd.jacobian(obj_from_wsopt_and_Bopt_and_Ks)

    use_yu = False
    dobj_dwsopt = functools.partial(kernels.Bopt_get_dobj_dwsopt, use_yu, get_Bopt, dobj_dwsopt_thru_wsopt, lin_solver, cvxopt_solver, dg_dB, d_dwsopt_dg_dB, d_dB_dg_dB, dobj_dBopt)#, wsopt, (Ky, SDR_Ku))

    # define constructor for dobj_dP_thru_Ku

    def get_dobj_dP_thru_Ku_stuff(xs_train, SDR_get_K, Ky):

        def the_g2(B, P, (wsopt,Ky)):
            us_train = np.dot(xs_train, P)
            return Bobj_from_Ks_and_wsopt_and_Bopt(wsopt, (Ky, SDR_get_K(us_train,us_train)), B)
    
        dg_dB = autograd.jacobian(the_g2)
        d_dP_dg_dB = autograd.jacobian(lambda P, B, (wsopt,Ky): dg_dB(B, P, (wsopt,Ky)))
    
        d_dB_dg_dB = autograd.jacobian(autograd.jacobian(the_g2))
    
        dobj_dP_thru_Ku = autograd.jacobian(lambda P, wsopt, Bopt: obj_from_Ks_and_wsopt_and_Bopt((Ky, SDR_get_K(np.dot(xs_train,P),np.dot(xs_train,P))), wsopt, Bopt))
    
        return dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, dobj_dP_thru_Ku

 def get_dobj_dP_thru_Ku(xs_train, SDR_get_K, Ky):
    
        dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt, dobj_dP_thru_Ku = get_dobj_dP_thru_Ku_stuff(xs_train, SDR_get_K, Ky)
    
        return functools.partial(kernels.Bopt_get_dobj_dP, xs_train, Ky, SDR_get_K, get_Bopt, dobj_dP_thru_Ku, lin_solver, cvxopt_solver, dg_dB, d_dP_dg_dB, d_dB_dg_dB, dobj_dBopt)#, P, wsopt)

    return obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku

def get_SDR_horse(desired_dim, num_trials, num_anneals, lin_solver, cvxopt_solver, B_max, KMM_eps, get_KMM_get_K, get_SDR_get_K, get_SDR_get_Ky, obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku, xs_train, xs_test, ys_train, ys_test=None, use_yu=False):


    # define stuff for plotting
    x_low, x_high = -2, 4
    y_low, y_high = -2, 4
    
    # define stuff for KMM
    #B_max = 10.
    #KMM_eps = 0.01

    #sigma = 0.5
    #KMM_get_K = functools.partial(kernels.get_gaussian_K, sigma)
    #get_KMM_get_K = lambda P,step: KMM_get_K
    #get_KMM_get_K = lambda P,step: functools.partial(kernels.get_gaussian_K, 1.*median_distance(np.dot(np.vstack((xs_train,xs_test)),P),np.dot(np.vstack((xs_train,xs_test)),P)))


    # define linear system solvers
#    lin_solver = lambda A,b: scipy.sparse.linalg.lsmr(A,b)[0]
    #lin_solver = lambda A,b: np.linalg.lstsq(A, b)[0]
    #lin_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]

    # define qp solver
#    cvxopt_solver = kernels.cvxopt_solver

    # define fxn that takes in stuff that changes over iterations
    get_obj_and_dobj_dP = lambda KMM_get_K, SDR_get_K, SDR_get_Ky: kernels.get_obj_and_obj_gradient(KMM_get_K, B_max, KMM_eps, SDR_get_K, SDR_get_Ky, obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku, lin_solver, cvxopt_solver, xs_train, xs_test, ys_train, use_yu=use_yu)

    #
    def plot_optimize(optimize_info):
        kernels.plot_opt_log(optimize_info)

    # define optimizer
    def optimize(plot_optimize, manifold, solver, obj, dobj_dP, P_init):
        from pymanopt import Problem
        problem = pymanopt.Problem(manifold=manifold, cost=obj, egrad=dobj_dP, verbosity=50)
        P_fit, optimize_info = solver.solve(problem, x=P_init)
        print 'plot_optimize'
        plot_optimize(optimize_info)
        return P_fit, optimize_info
    from pymanopt.solvers import SteepestDescent
    from pymanopt.manifolds import Stiefel
    solver = pymanopt.solvers.SteepestDescent(logverbosity=2)
    x_dim = xs_train.shape[1]
    manifold = pymanopt.manifolds.Stiefel(x_dim, desired_dim)
    plot_opt_info = lambda opt_log: None
    optimizer = functools.partial(optimize, plot_optimize, manifold, solver)

    #
    def plot_anneal_step_state(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P):
        #kernels.plot_weights(xs_train, xs_test, KMM_get_K, B_max, KMM_eps, P)
        #kernels.plot_K(xs_train, xs_train, KMM_get_K, P, 'KMM train train')
        #kernels.plot_K(xs_train, xs_test, KMM_get_K, P, 'KMM train test')
        #kernels.plot_K(xs_train, xs_train, SDR_get_K, P, 'SDR train train')
        kernels.gradient_check(obj, dobj_dP, P)
        #kernels.plot_train_vs_test(xs_train, xs_test, P)
        #print 'obj_val:', obj(P)
        pass
        #kernels.plot_y_vs_u(xs_train, ys_train, P, xs_test, ys_test, (x_low,x_high), (y_low,y_high))
    
    def plot_anneal_step(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P, opt_log):
        plot_anneal_step_state(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P)
        print 'annealing step P:', opt_log['final_values']['x'], opt_log['final_values']['f(x)']

    # 
    def plot_anneal(anneal_info):
        obj_val, anneal_step_infos = anneal_info
        fig, ax = plt.subplots()
        ax.scatter(range(len(anneal_step_infos)), [anneal_step_info[2].args[0] for anneal_step_info in anneal_step_infos])
        ax.set_title('KMM sigma')
        ax.set_xlabel('annealing step')
        ax.set_ylabel('sigma')
        basic.display_fig_inline(fig)

    # 
    def anneal(plot_anneal, plot_anneal_step, plot_anneal_step_state, num_anneals, optimizer, get_KMM_get_K, get_SDR_get_K, get_SDR_get_Ky, get_obj_and_dobj_dP, P):
        anneal_step_infos = [] # store the things that change
        for step in xrange(num_anneals):
            KMM_get_K = get_KMM_get_K(P,step, xs_train, xs_test)
            SDR_get_K = get_SDR_get_K(P,step, xs_train, xs_test)
            SDR_get_Ky = get_SDR_get_Ky(P,step, ys_train, ys_train)
            obj, dobj_dP = get_obj_and_dobj_dP(KMM_get_K, SDR_get_K, SDR_get_Ky)
        
            print 'plot_anneal_step_state'
            plot_anneal_step_state(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P)
        
            P, opt_log = optimizer(obj, dobj_dP, P)
            anneal_step_info = obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P, opt_log
            plot_anneal_step(*anneal_step_info)
            anneal_step_infos.append(anneal_step_info)
        anneal_info = (obj(P), anneal_step_infos)
        print 'plot_anneal'
        plot_anneal(anneal_info)
        return P, anneal_info
#    num_anneals = 1
    annealer = functools.partial(anneal, plot_anneal, plot_anneal_step, plot_anneal_step_state, num_anneals, optimizer, get_KMM_get_K, get_SDR_get_K, get_SDR_get_Ky, get_obj_and_dobj_dP)

    #
    def plot_multiples(multiples_info):
        obj_vals = []
        best_obj_val = None
        best_anneal_step_info = None
        for anneal_info in multiples_info:
            obj_val, anneal_step_infos = anneal_info
            obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P, opt_log = anneal_step_infos[-1]
            print 'this anneal obj_val:', obj_val, 'P:', P
            kernels.plot_y_vs_u(xs_train, ys_train, P, xs_test, ys_test, (x_low,x_high), (y_low,y_high))
            obj_vals.append(obj_val)
            if obj_val > best_obj_val:
                best_obj_val = obj_val
                best_anneal_step_info = anneal_step_infos[-1]
        #plot_anneal_step(*best_anneal_step_info)
        print 'obj_vals', obj_vals

    #
    def multiples(plot_multiples, num_trials, annealer, P_shape):
        trial_infos = []
        best_P = None
        best_obj_val = None
        for i in xrange(num_trials):
            P_init = kernels.ortho(np.random.normal(size=P_shape))
            P, anneal_info = annealer(P_init)
            trial_infos.append(anneal_info)
            (obj_val, anneal_step_infos) = anneal_info
            if best_P is None or obj_val < best_obj_val:
                best_P = P
                best_obj_val = obj_val
        multiples_info = trial_infos
        plot_multiples(multiples_info)
        return best_P


    return multiples(plot_multiples, num_trials, annealer, (x_dim,desired_dim))

import python_utils.python_utils.caching as caching

#@caching.default_read_fxn_decorator()
#@caching.default_write_fxn_decorator()
def get_noplot_SDR_horse(desired_dim, num_trials, num_anneals, lin_solver, cvxopt_solver, B_max, KMM_eps, get_KMM_get_K, get_SDR_get_K, get_SDR_get_Ky, obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku, xs_train, xs_test, ys_train, ys_test=None, use_yu=False):

    # define stuff for plotting
    x_low, x_high = -2, 4
    y_low, y_high = -2, 4
    
    # define stuff for KMM
    #B_max = 10.
    #KMM_eps = 0.01

    #sigma = 0.5
    #KMM_get_K = functools.partial(kernels.get_gaussian_K, sigma)
    #get_KMM_get_K = lambda P,step: KMM_get_K
    #get_KMM_get_K = lambda P,step: functools.partial(kernels.get_gaussian_K, 1.*median_distance(np.dot(np.vstack((xs_train,xs_test)),P),np.dot(np.vstack((xs_train,xs_test)),P)))


    # define linear system solvers
#    lin_solver = lambda A,b: scipy.sparse.linalg.lsmr(A,b)[0]
    #lin_solver = lambda A,b: np.linalg.lstsq(A, b)[0]
    #lin_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]

    # define qp solver
#    cvxopt_solver = kernels.cvxopt_solver

    # define fxn that takes in stuff that changes over iterations
    get_obj_and_dobj_dP = lambda KMM_get_K, SDR_get_K, SDR_get_Ky: kernels.get_obj_and_obj_gradient(KMM_get_K, B_max, KMM_eps, SDR_get_K, SDR_get_Ky, obj_from_wsopt_and_Ks, dobj_dwsopt, get_dobj_dP_thru_Ku, lin_solver, cvxopt_solver, xs_train, xs_test, ys_train, use_yu=use_yu)

    #
    def plot_optimize(optimize_info):
        kernels.plot_opt_log(optimize_info)

    # define optimizer
    def optimize(plot_optimize, manifold, solver, obj, dobj_dP, P_init):
        from pymanopt import Problem
        problem = pymanopt.Problem(manifold=manifold, cost=obj, egrad=dobj_dP, verbosity=0)
        P_fit, optimize_info = solver.solve(problem, x=P_init)
        #print 'plot_optimize'
        #plot_optimize(optimize_info)
        return P_fit, optimize_info
    from pymanopt.solvers import SteepestDescent
    from pymanopt.manifolds import Stiefel
    solver = pymanopt.solvers.SteepestDescent(logverbosity=2)
    x_dim = xs_train.shape[1]
    manifold = pymanopt.manifolds.Stiefel(x_dim, desired_dim)
    plot_opt_info = lambda opt_log: None
    optimizer = functools.partial(optimize, plot_optimize, manifold, solver)

    #
    def plot_anneal_step_state(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P):
        kernels.plot_weights(xs_train, xs_test, KMM_get_K, B_max, KMM_eps, P)
        kernels.plot_K(xs_train, xs_train, KMM_get_K, P, 'KMM train train')
        kernels.plot_K(xs_train, xs_test, KMM_get_K, P, 'KMM train test')
        kernels.plot_K(xs_train, xs_train, SDR_get_K, P, 'SDR train train')
        kernels.gradient_check(obj, dobj_dP, P)
        kernels.plot_train_vs_test(xs_train, xs_test, P)
        print 'obj_val:', obj(P)
        kernels.plot_y_vs_u(xs_train, ys_train, P, xs_test, ys_test, (x_low,x_high), (y_low,y_high))
    
    def plot_anneal_step(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P, opt_log):
        plot_anneal_step_state(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P)
        print 'annealing step P:', opt_log['final_values']['x'], opt_log['final_values']['f(x)']

    # 
    def plot_anneal(anneal_info):
        obj_val, anneal_step_infos = anneal_info
        fig, ax = plt.subplots()
        ax.scatter(range(len(anneal_step_infos)), [anneal_step_info[2].args[0] for anneal_step_info in anneal_step_infos])
        ax.set_title('KMM sigma')
        ax.set_xlabel('annealing step')
        ax.set_ylabel('sigma')
        basic.display_fig_inline(fig)

    # 
    def anneal(plot_anneal, plot_anneal_step, num_anneals, optimizer, get_KMM_get_K, get_SDR_get_K, get_SDR_get_Ky, get_obj_and_dobj_dP, P):
        anneal_step_infos = [] # store the things that change
        for step in xrange(num_anneals):
            KMM_get_K = get_KMM_get_K(P,step, xs_train, xs_test)
            SDR_get_K = get_SDR_get_K(P,step, xs_train, xs_test)
            SDR_get_Ky = get_SDR_get_Ky(P,step, ys_train, ys_train)
            obj, dobj_dP = get_obj_and_dobj_dP(KMM_get_K, SDR_get_K, SDR_get_Ky)
        
            #print 'plot_anneal_step_state'
            #plot_anneal_step_state(obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P)
        
            P, opt_log = optimizer(obj, dobj_dP, P)
            anneal_step_info = obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P, opt_log
            #plot_anneal_step(*anneal_step_info)
            anneal_step_infos.append(anneal_step_info)
        anneal_info = (obj(P), anneal_step_infos)
        #print 'plot_anneal'
        #plot_anneal(anneal_info)
        return P, anneal_info
#    num_anneals = 1
    annealer = functools.partial(anneal, plot_anneal, plot_anneal_step, num_anneals, optimizer, get_KMM_get_K, get_SDR_get_K, get_SDR_get_Ky, get_obj_and_dobj_dP)

    #
    def plot_multiples(multiples_info):
        obj_vals = []
        best_obj_val = None
        best_anneal_step_info = None
        for anneal_info in multiples_info:
            obj_val, anneal_step_infos = anneal_info
            obj, dobj_dP, KMM_get_K, SDR_get_K, SDR_get_Ky, P, opt_log = anneal_step_infos[-1]
            print 'this anneal obj_val:', obj_val, 'P:', P
            kernels.plot_y_vs_u(xs_train, ys_train, P, xs_test, ys_test, (x_low,x_high), (y_low,y_high))
            obj_vals.append(obj_val)
            if obj_val > best_obj_val:
                best_obj_val = obj_val
                best_anneal_step_info = anneal_step_infos[-1]
        #plot_anneal_step(*best_anneal_step_info)
        print 'obj_vals', obj_vals

    #
    def multiples(plot_multiples, num_trials, annealer, P_shape):
        trial_infos = []
        best_P = None
        best_obj_val = None
        for i in xrange(num_trials):
            P_init = kernels.ortho(np.random.normal(size=P_shape))
            P, anneal_info = annealer(P_init)
            trial_infos.append(anneal_info)
            (obj_val, anneal_step_infos) = anneal_info
            if best_P is None or obj_val < best_obj_val:
                best_P = P
                best_obj_val = obj_val
        multiples_info = trial_infos
        #plot_multiples(multiples_info)
        return best_P


    return multiples(plot_multiples, num_trials, annealer, (x_dim,desired_dim))
