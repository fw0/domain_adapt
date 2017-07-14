import numpy as np
import pymanopt
import scipy.optimize
import pdb, itertools
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import utils


class optimizer(object):

    def __call__(self, *args, **kwargs):
        return self.optimize(*args, **kwargs)

        
class pymanopt_optimizer(optimizer):

    def __init__(self, problem_verbosity=1, **kwargs):
        self.options, self.problem_verbosity = kwargs, problem_verbosity
    
    def optimize(self, objective, dobjective_dB, B_init):
        from pymanopt.solvers import SteepestDescent
        from pymanopt.manifolds import Stiefel
        solver = pymanopt.solvers.SteepestDescent(**self.options)
        #solver = pymanopt.solvers.ConjugateGradient(**self.options)
        manifold = pymanopt.manifolds.Stiefel(B_init.shape[0], B_init.shape[1])
        from pymanopt import Problem
        problem = pymanopt.Problem(manifold=manifold, cost=objective, egrad=dobjective_dB, verbosity=self.problem_verbosity)
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

class linesearch_optimizer(optimizer):

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


class scalar_fxn_optimizer(optimizer):

    def __init__(self, method, options={}, init_window_width=100.):
        self.options, self.method, self.init_window_width = options, method, init_window_width

    def optimize(self, objective, x_init):
        if self.method == 'golden' or self.method == 'brent':
            bracket = (0.01, x_init+self.init_window_width)
#            bracket = (x_init-self.init_window_width, x_init+self.init_window_width)
            result = scipy.optimize.minimize_scalar(fun=objective, bracket=bracket, method=self.method, options=self.options)
        elif self.method == 'bounded':
#            bounds = (x_init-self.init_window_width, x_init+self.init_window_width)
            eps = 0.1 # fix
            bounds = (eps, x_init+self.init_window_width)
            result = scipy.optimize.minimize_scalar(fun=objective, bounds=bounds, method=self.method, options=self.options)
        #print result
        #print map(objective, np.linspace(0,1,10))
#        pdb.set_trace()
        return result['x']


class scipy_minimize_optimizer(optimizer):

    def __init__(self, method=None, options={}, verbose=False, info_f=None):
        self.method, self.options, self.verbose, self.info_f = method, options, verbose, info_f
#        self.method = 'cg'

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
        
class B_f_optimizer(optimizer):
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
    
    
class multiple_optimizer(optimizer):

    def __init__(self, horse, num_tries, num_args):
        self.horse, self.num_tries, self.num_args = horse, num_tries, num_args

    def optimize(self, *args):
        opt_args, init_fs = args[0:(-self.num_args)], args[(-self.num_args):]
        xs = []
        f_xs = []
        opt_logs = []
        for i in xrange(self.num_tries):
            print 'try', i
            init_vals = tuple([init_f() for init_f in init_fs])
            #print init_vals
            x = self.horse.optimize(*(opt_args+init_vals))
            xs.append(x)
            opt_logs.append(self.horse.opt_log)
            f_x = self.horse.final_objective(self.horse.opt_log)
            f_xs.append(f_x)
        best_try = np.nanargmin(f_xs)
        assert np.nanmin(f_xs) == f_xs[best_try]
        self.opt_log = {'opt_logs':opt_logs, 'final_objective':f_xs[best_try]}
        #print zip(xs, f_xs)
        return xs[best_try]

    def final_objective(self, opt_log):
        return opt_log['final_objective':]

    def plot_objective(self, opt_log):
        for (i,_opt_log) in enumerate(opt_log['opt_logs']):
            print 'iteration', i
            self.horse.plot_objective(_opt_log)
            

class grid_search_optimizer(optimizer):

    def optimize(self, objective, ranges):
        d = []
        for args in itertools.product(*ranges):
#            print args, 'grid', ranges
            d.append((args, objective(*args)))
        best_args, best_val = min(d, key=lambda (args, val): val)
        #print d
        import pandas as pd
        from IPython.display import display_pretty, display_html
#        display_html(pd.DataFrame.from_records(d).to_html(), raw=True)
        print best_args, best_val, 'best'
        self.opt_log = ((best_args, best_val), d)
        return best_args
        
    def final_objective(self, ((best_args, best_val), d)):
        return best_args

    def plot_objective(self,  ((best_args, best_val), d)):
        print d
            
class get_stuff_optimizer(optimizer):

    def __init__(self, horse, get_stuff_fs, out_f):
        self.horse, self.get_stuff_fs, self.out_f = horse, get_stuff_fs, out_f

    def optimize(self, *args):
        stuffs = [get_stuff_f(*args) for get_stuff_f in self.get_stuff_fs]
        horse_result = self.horse(*stuffs)
        return self.out_f(args, stuffs, horse_result)


class many_optimizer(optimizer):

    def __init__(self, optimizers, num_cycles=5):
        self.optimizers, self.num_cycles = optimizers, num_cycles

    def final_objective(self, opt_log):
        return opt_log['final_objective']

    def optimize(self, objective, *args):
        import copy
        opt_log = []
#        while True:
        for i in xrange(self.num_cycles):
#            print args
            old_args = copy.deepcopy(args)
            for optimizer in self.optimizers:
                #print 'result before', args
                args = optimizer(*args)
                #print 'result after', args
#            if converged(old_args, args):
#                break
        self.opt_log = {'final_objective':objective(*args)}
        return args
