import numpy as np
import pymanopt
import scipy.optimize
import pdb, itertools
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import utils
        
class pymanopt_optimizer(object):

    def __init__(self, **kwargs):
        self.options = kwargs
    
    def optimize(self, objective, dobjective_dB, B_init):
        from pymanopt.solvers import SteepestDescent
        from pymanopt.manifolds import Stiefel
        solver = pymanopt.solvers.SteepestDescent(**self.options)
        manifold = pymanopt.manifolds.Stiefel(B_init.shape[0], B_init.shape[1])
        from pymanopt import Problem
        problem = pymanopt.Problem(manifold=manifold, cost=objective, egrad=dobjective_dB, verbosity=2)
        B_fit, self.opt_log = solver.solve(problem, x=B_init)
        return B_fit

    def plot_objective(self, opt_log):
        print self.final_x(opt_log)
        
        fig,ax = plt.subplots()
        obj_vals = opt_log['iterations']['f(x)']
        ax.plot(range(len(obj_vals)), obj_vals)
        ax.set_title('obj vals')
        ax.set_xlabel('step')
        ax.set_ylabel('obj val')
        basic.display_fig_inline(fig)

        import pprint
        pp = pprint.PrettyPrinter()
        #pp.pprint(opt_log)

    def final_objective(self, opt_log):
        return opt_log['final_values']['f(x)']

    def final_x(self, opt_log):
        return opt_log['final_values']['x']

class linesearch_optimizer(object):

    def __init__(self, maxiter=1000):
        self.maxiter = maxiter

    def plot_objective(self, opt_log):
        fig,ax = plt.subplots()
        obj_vals = opt_log
        ax.plot(range(len(obj_vals)), obj_vals)
        ax.set_title('obj vals')
        ax.set_xlabel('step')
        ax.set_ylabel('obj val')
        basic.display_fig_inline(fig)
        
    def optimize(self, objective, dobjective_dx, x_init):
        f_xs = [objective(x_init)]
        x = x_init
        for i in xrange(self.maxiter):
            print i
            grad = dobjective_dx(x)
            print 'outside grad', grad
            x = x + (scipy.optimize.minimize_scalar(lambda alpha: objective(x + (alpha*grad)))['x'] * grad)
            f_xs.append(objective(x))
        self.opt_log = f_xs
        return x

    def final_objective(self, opt_log):
        return opt_log[-1]


class scipy_minimize_optimizer(object):

    def __init__(self, method=None, options={}, verbose=False, info_f=None):
        self.method, self.options, self.verbose, self.info_f = method, options, verbose, info_f

    def optimize(self, objective, dobjective_dx, x_init, bounds=None):
#        print self.verbose
#        pdb.set_trace()
#        print self.verbose, 'gggg'
        logger = utils.optimizer_logger(objective, dobjective_dx, self.verbose, self.info_f)
#        logger = utils.optimizer_logger(objective, dobjective_dx, self.verbose, self.info_f)
        logger.l.append({'f':objective(x_init), 'grad_norm':np.linalg.norm(dobjective_dx(x_init))})
        ans = scipy.optimize.minimize(fun=objective, x0=x_init, jac=dobjective_dx, method=self.method, options=self.options, callback=logger, bounds=bounds)
        self.opt_log = {'logger':logger, 'optimize_result':ans}
        return ans['x']

    def plot_objective(self, opt_log):
        print self.final_x(opt_log)
        opt_log['logger'].plot()

    def print_info(self, opt_log):
        print self.opt_log['optimize_result']

    def final_objective(self, opt_log):
        return opt_log['optimize_result']['fun']

    def final_x(self, opt_log):
        return opt_log['optimize_result']['x']
        
class B_f_optimizer(object):
    """
    does coordinate ascent, not "simultaneous" steps
    """
    def __init__(self, stiefel_optimizer, unconstrained_optimizer, tol=0.01, maxiter=1000):
        self.stiefel_optimizer, self.unconstrained_optimizer = stiefel_optimizer, unconstrained_optimizer
        self.tol, self.maxiter = tol, maxiter

    def optimize(self, objective, dobjective_dB, dobjective_df, B_init, f_init):
        B, f = B_init, f_init
        B_objective_vals = []
        f_objective_vals = [objective(B,f)]
        xs = []
        for i in xrange(self.maxiter):
            B = self.stiefel_optimizer.optimize(lambda B: objective(B,f), lambda B: dobjective_dB(B,f), B)
            B_objective_vals.append(self.stiefel_optimizer.final_objective(self.stiefel_optimizer.opt_log))
            f = self.unconstrained_optimizer.optimize(lambda f: objective(B,f), lambda f: dobjective_df(B,f), f)
            f_objective_vals.append(self.unconstrained_optimizer.final_objective(self.unconstrained_optimizer.opt_log))
            xs.append((B,f))
        self.opt_log = {'B_objective_vals':B_objective_vals, 'f_objective_vals':f_objective_vals}
        return B,f

    def plot_objective(self, opt_log):
        fig,ax = plt.subplots()
        B_color = 'red'
        B_label = 'B'
        f_color = 'blue'
        f_label = 'f'
        for (i,(B_objective_val,f_objective_val)) in enumerate(itertools.izip(opt_log['B_objective_vals'], opt_log['f_objective_vals'][1:])):
            if i == 0:
                ax.plot([i+0.5,i+1], [B_objective_val,f_objective_val], color=f_color, label=f_label)
            else:
                ax.plot([i+0.5,i+1], [B_objective_val,f_objective_val], color=f_color)
        for (i,(f_objective_val,B_objective_val)) in enumerate(itertools.izip(opt_log['f_objective_vals'][:-1], opt_log['B_objective_vals'])):
            if i == 0:
                ax.plot([i,i+0.5], [f_objective_val,B_objective_val], color=B_color, label=B_label)
            else:
                ax.plot([i,i+0.5], [f_objective_val,B_objective_val], color=B_color)
        ax.set_title('obj vals')
        ax.set_xlabel('step')
        ax.set_ylabel('obj val')
        ax.legend()
        basic.display_fig_inline(fig)

    def final_objective(self, opt_log):
        return opt_log['f_objective_vals'][-1]
    
    
class multiple_optimizer(object):

    def __init__(self, horse, num_tries, num_args):
        self.horse, self.num_tries, self.num_args = horse, num_tries, num_args

    def optimize(self, *args):
        opt_args, init_fs = args[0:(-self.num_args)], args[(-self.num_args):]
        xs = []
        f_xs = []
        opt_logs = []
        for i in xrange(self.num_tries):
            init_vals = tuple([init_f() for init_f in init_fs])
            x = self.horse.optimize(*(opt_args+init_vals))
            xs.append(x)
            opt_logs.append(self.horse.opt_log)
            f_x = self.horse.final_objective(self.horse.opt_log)
            f_xs.append(f_x)
        best_try = np.argmin(f_xs)
        assert np.min(f_xs) == f_xs[best_try]
        self.opt_log = {'opt_logs':opt_logs, 'final_objective':f_xs[best_try]}
        #print zip(xs, f_xs)
        return xs[best_try]

    def final_objective(self, opt_log):
        return opt_log['final_objective':]

    def plot_objective(self, opt_log):
        for (i,_opt_log) in enumerate(opt_log['opt_logs']):
            print 'iteration', i
            self.horse.plot_objective(_opt_log)
            
