import fxns, optimizers, autograd, autograd.numpy as np, scipy
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching
import domain_adapt.domain_adapt.SDR_data as data
import utils
import itertools

def lsif_ratio_fitter(which_loss, num_basis, tradeoff_weight_reg, B_init_f_getter=None, learn_weights=True, learn_projection=True, max_ratio=5, unconstrained=False, lsif_least_squares=True, c_pred_use_test=False, num_folds_c_pred=3, plot_b_info=None, num_tries=1, unconstrained_scipy_minimize_method=None, unconstrained_scipy_minimize_options={}, unconstrained_scipy_minimize_info_f=lambda x: None, unconstrained_scipy_minimize_verbose=1, quad_opt_warm_start=True, pymanopt_options={'logverbosity':2, 'maxiter':100}, linesearch_method='brent', linesearch_options={}, linesearch_init_window_width=100, c_lsif_sigma_grad_warm_start=False, c_lsif_sigma_grad_scipy_minimize_method=None, c_lsif_sigma_grad_scipy_minimize_options={'maxiter':1}, c_lsif_sigma_grad_scipy_minimize_info_f=lambda x: None, c_lsif_sigma_grad_scipy_minimize_verbose=1, c_pred_line_search=True, c_pred_grad_warm_start=False, c_pred_grad_scipy_minimize_method=None, c_pred_grad_scipy_minimize_options={}, c_pred_grad_scipy_minimize_info_f=lambda x: None, c_pred_grad_scipy_minimize_verbose=1, c_lsif_sigma_grid_search=True, num_folds_lsif=3, c_lsif_sigma_grid_search_c_lsif_range=None, c_lsif_sigma_grid_search_sigma_range=None, c_lsif_sigma_grid_search_sigma_percentiles=None, many_optimizer_num_cycles=1):
    """
    unconstrained options are for 1d unconstrained projections
    pymanopt options is for manifold optimization
    linesearch options is for c_pred optimization
    c_lsif_sigma_grad options is for gradient descent optimization of c_lsif and sigma
    c_pred_grad options is for gradient descent optimization of c_pred
    """

    if c_lsif_sigma_grid_search_c_lsif_range is None:
        c_lsif_sigma_grid_search_c_lsif_range = 10**(np.arange(-3,2).astype(float))

    if c_lsif_sigma_grid_search_sigma_range is None:
        c_lsif_sigma_grid_search_sigma_range = np.linspace(0.1,10,4)

    if c_lsif_sigma_grid_search_sigma_percentiles is None:
        #c_lsif_sigma_grid_search_sigma_percentiles = np.array([20.,35.,50.,65.,80.])
        c_lsif_sigma_grid_search_sigma_percentiles = np.array([25.,50.,75.])

    
#    @basic.do_cprofile('cumtime')
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        # define constants

        xs_basis = xs_test[0:num_basis]
        add_reg_pred = False # for evaluating loss during training
        add_reg_pred_oos = False
        add_reg_lsif = True # for training
        add_reg_lsif_oos = False

        if not learn_projection:
            assert B_init_f_getter == None

        # define objective fxn

        if which_loss == 'square':
            b_opt_given_ws = fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt) # B, xs_train, ys_train, ws_train, c_pred
            weighted_loss_given_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt) # B, xs_train, ys_train, ws_train, b_opt, c_pred, add_reg_pred
            weighted_loss_given_ws = fxns.two_step.autograd_fxn( # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred
                g=b_opt_given_ws, # B, xs_train, ys_train, ws_train, c_pred
                h=weighted_loss_given_b_opt, # B, xs_train, ys_train, ws_train, b_opt, c_pred, add_reg_pred
                g_argnums=(0,1,2,3,4),
                h_argnums=(0,1,2,3,4,5),
                g_val_h_argnum=4
                )
        elif which_loss == 'logistic':
            raise NotImplementedError

        objective_given_ws = fxns.sum.autograd_fxn( # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred
            fs=[
                weighted_loss_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg), # ws_train
                ],
            fs_argnums=[
                (0,1,2,3,4,5),
                (3,),
                ],
            weights=[
                1.,
                tradeoff_weight_reg,
                ]
            )

        objective_given_alpha = fxns.two_step.autograd_fxn( # B, xs_train, ys_train, c_pred, add_reg_pred, lsif_alpha, xs_basis, sigma, max_ratio
            g=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
            h=objective_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred
            g_argnums=(5,1,6,7,0,8),
            h_argnums=(0,1,2,3,4),
            g_val_h_argnum=3,
            )
            
        if not lsif_least_squares:
            lsif_objective = fxns.LSIF_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
            dlsif_objective_dalpha = fxns.dopt_objective_dx.autograd_fxn(lsif_objective)
            gmres_solver = lambda A,b: scipy.sparse.linalg.gmres(A, b, maxiter=10)[0]
            alpha_given_B = fxns.full_quad_opt( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                lin_solver=gmres_solver,
                objective=lsif_objective,
                dobjective_dx=dlsif_objective_dalpha,
                ineq_constraints=fxns.Gh_ineq_constraints.autograd_fxn(_get_Gh=fxns.LSIF_get_Gh),
                warm_start=quad_opt_warm_start
                )
            objective_given_B_constructor = lambda *args, **kwargs: fxns.fxn.wrap_primitive(fxns.g_thru_f_opt_new(*args, **kwargs))
        else:
            alpha_given_B = fxns.fxn.autograd_fxn(_val=fxns.least_squares_lsif_alpha_given_B) # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            objective_given_B_constructor = lambda *args, **kwargs: fxns.fxn.wrap_primitive(fxns.two_step.autograd_fxn(*args, **kwargs))
            
        objective_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred
            g=alpha_given_B, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=objective_given_alpha, # B, xs_train, ys_train, c_pred, add_reg_pred, lsif_alpha, xs_basis, sigma, max_ratio
            g_argnums=(0,1,2,3,4,5,6,7),
            h_argnums=(3,0,8,9,10,5,2,6),
            g_val_h_argnum=5
        )

        # define c_pred hyperparameter optimizer

        # assumes b_opt_given_ws and weighted_loss_given_b_opt have the same signatures regardless of loss function
        weighted_loss_oos_given_ws_oos = fxns.two_step.autograd_fxn( # B, xs_train_is, ys_train_is, ws_train_is, c_pred, add_reg_pred_oos, xs_train_oos, ys_train_oos, ws_train_oos
            g=b_opt_given_ws, # B, xs_train_is, ys_train_is, ws_train_is, c_pred
            h=weighted_loss_given_b_opt, # B, xs_train_oos, ys_train_oos, ws_train_oos, b_opt, c_pred, add_reg_pred_oos
            g_argnums=(0,1,2,3,4),
            h_argnums=(0,6,7,8,4,5),
            g_val_h_argnum=4
            )

        if not c_pred_use_test:
        
            def weighted_loss_cv_given_ws(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos=None, ys_test_oos=None):
                # xs_test_oos and ys_test_oos are not actually used
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=num_folds_c_pred)
                losses = 0.
                for (is_idx, oos_idx) in kf.split(xs_train):
                    xs_train_is = xs_train[is_idx]
                    xs_train_oos = xs_train[oos_idx]
                    try:
                        ys_train_is = ys_train[is_idx]
                    except:
                        pdb.set_trace()
                    ys_train_oos = ys_train[oos_idx]
                    ws_train_is = ws_train[is_idx]
                    ws_train_oos = ws_train[oos_idx]
                    fold_loss = weighted_loss_oos_given_ws_oos(B, xs_train_is, ys_train_is, ws_train_is, c_pred, add_reg_pred_oos, xs_train_oos, ys_train_oos, ws_train_oos) # treating as regular function
                    losses += fold_loss * len(is_idx)
                return losses / len(xs_train)

            test_loss_given_ws = fxns.fxn.autograd_fxn(_val=weighted_loss_cv_given_ws)

        elif c_pred_use_test:

            def _test_loss_given_ws(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos):
                ws_test = np.ones(len(xs_test))
                return weighted_loss_oos_given_ws_oos(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, xs_test_oos, ys_test_oos, ws_test)

            test_loss_given_ws = fxns.fxn.autograd_fxn(_val=_test_loss_given_ws)

        test_loss_given_alphas = fxns.two_step.autograd_fxn( # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos
            g=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
            h=test_loss_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos
            g_argnums=(0,1,2,3,4,5),
            h_argnums=(4,1,6,7,8,9,10,11),
            g_val_h_argnum=3
            )

        test_loss_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos
            g=alpha_given_B, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=test_loss_given_alphas, # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos
            g_argnums=(0,1,2,3,4,5,6,7),
            h_argnums=(0,5,2,3,6,8,9,10,11,12,13),
            g_val_h_argnum=0
            )

        if c_pred_use_test and num_folds_c_pred != 0:
            def _final_test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test):
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=num_folds_c_pred)
                test_losses = 0.
                for ((train_is_idx, train_oos_idx), (test_is_idx, test_oos_idx)) in itertools.izip(kf.split(xs_train), kf.split(xs_test)):
                    xs_train_is = xs_train[train_is_idx]
                    xs_train_oos = xs_train[train_oos_idx]
                    xs_test_is = xs_test[test_is_idx]
                    xs_test_oos = xs_test[test_oos_idx]
                    ys_train_is = ys_train[train_is_idx]
                    ys_test_oos = ys_test[test_oos_idx]
                    fold_test_loss = objective_given_B_constructor(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train_is, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos)
                    test_losses += fold_test_loss * len(test_oos_idx)
                return test_losses / len(ys_test)

        else:

            def _final_test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test):
                return test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test, ys_test)
            
        final_test_loss_given_B = fxns.fxn.autograd_fxn(_val=_final_test_loss_given_B)

        def c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, B):
            return lambda _c_pred: final_test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, _c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test)

        def c_pred_optimizer_get_dobjective_dc_pred(c_lsif, sigma, c_pred, B):
            return lambda _c_pred: final_test_loss_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, _c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test, care_argnums=(9,))
            
        if c_pred_line_search:

            def c_pred_optimizer_get_c_pred(c_lsif, sigma, c_pred, B):
                print 'c_pred'
                return c_pred
            
            def c_pred_optimizer_out_f((c_lsif, sigma, c_pred, B), (objective, _c_pred), new_c_pred):
                return (c_lsif, sigma, new_c_pred, B)
        
            c_pred_horse_optimizer = optimizers.scalar_fxn_optimizer(method=linesearch_method, options=linesearch_options, init_window_width=linesearch_init_window_width)
            c_pred_optimizer = optimizers.get_stuff_optimizer(c_pred_horse_optimizer, (c_pred_optimizer_get_objective, c_pred_optimizer_get_c_pred), c_pred_optimizer_out_f)

        else:

            def log_c_pred_optimizer_get_log_c_pred(c_lsif, sigma, c_pred, B):
                print 'c_pred'
                return np.log(c_pred)
            
            def log_c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, B):
                horse = c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, B)
                return lambda log_c_pred: horse(np.exp(log_c_pred))

            def log_c_pred_optimizer_get_dobjective_dlog_c_pred(c_lsif, sigma, c_pred, B):
                horse = fxns.fxn.autograd_fxn(_val=log_c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, B))
                return lambda log_c_pred: horse.grad(log_c_pred, care_argnums=(0,))

            def log_c_pred_optimizer_out_f((c_lsif, sigma, c_pred, B), (objective, dobjective_dlog_c_pred, _log_c_pred), new_log_c_pred):
                return (c_lsif, sigma, np.exp(new_log_c_pred), B)

            c_pred_horse_optimizer = optimizers.scipy_minimize_optimizer(method=c_pred_grad_scipy_minimize_method, verbose=c_pred_grad_scipy_minimize_verbose, info_f=c_pred_grad_scipy_minimize_info_f, options=c_pred_grad_scipy_minimize_options)
            c_pred_optimizer = optimizers.get_stuff_optimizer(c_pred_horse_optimizer, (log_c_pred_optimizer_get_objective, log_c_pred_optimizer_get_dobjective_dlog_c_pred, log_c_pred_optimizer_get_log_c_pred), log_c_pred_optimizer_out_f)

        # define c_lsif and sigma hyperparameter optimizer

        lsif_objective_oos = objective_given_B_constructor( # xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, xs_train_oos, xs_test_oos, add_reg_lsif_oos
            g=alpha_given_B, # xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=fxns.LSIF_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product), # xs_train_oos, xs_test_oos, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif_oos, alpha
            g_argnums=(0,1,2,3,4,5,6,7),
            h_argnums=(8,9,2,3,4,5,6,10),
            g_val_h_argnum=11
            )

        def _weighted_lsif_objective_cv_given_B(xs_train, xs_test, c_lsif, B, sigma, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_oos, num_folds_lsif):
            from sklearn.model_selection import KFold
            import itertools
            kf = KFold(n_splits=num_folds_lsif)
            objectives = 0.
            for ((train_is_idx, train_oos_idx), (test_is_idx, test_oos_idx)) in itertools.izip(kf.split(xs_train), kf.split(xs_test)):
                xs_train_is = xs_train[train_is_idx]
                xs_train_oos = xs_train[train_oos_idx]
                xs_test_is = xs_test[test_is_idx]
                xs_test_oos = xs_test[test_oos_idx]
                alpha = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif)
                #grad = alpha_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, care_argnums=(3,))
                ws_train = fxns.lsif_alpha_to_ratios(alpha, xs_train, xs_basis, sigma, B, max_ratio)
#                print ws_train, 'ws_train'
#                fold_objective = lsif_objective_oos(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_test_is[0:num_basis], max_ratio, add_reg_lsif, xs_train_oos, xs_test_oos, add_reg_lsif_oos)
                fold_objective = lsif_objective_oos(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, xs_train_oos, xs_test_oos, add_reg_lsif_oos)
                objectives += fold_objective * len(train_oos_idx)
            return objectives / len(xs_train)

        weighted_lsif_objective_cv_given_B = fxns.fxn.autograd_fxn(_val=_weighted_lsif_objective_cv_given_B)

        if not c_lsif_sigma_grid_search:
#        if True:

            def log_c_lsif_sigma_optimizer_get_log_c_lsif_sigma(c_lsif, sigma, c_pred, B):
                print 'lsif, sigma'
                return np.log(np.array([c_lsif, sigma]))
            
            def log_c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_pred, B):
                def horse(log_c_lsif_sigma):
#                    print log_c_lsif_sigma, np.exp(log_c_lsif_sigma), 'inside'
                    alphas = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif), 'inside2'
                    return weighted_lsif_objective_cv_given_B(xs_train, xs_test, np.exp(log_c_lsif_sigma[0]), B, np.exp(log_c_lsif_sigma[1]), xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_oos, num_folds_lsif)
                return horse

            def log_c_lsif_sigma_optimizer_get_dobjective_dlog_c_lsif_sigma(c_lsif, sigma, c_pred, B):
                horse = fxns.fxn.autograd_fxn(_val=log_c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_pred, B))
                def wrapped(log_c_lsif_sigma):
                    try:
                        return horse.grad(log_c_lsif_sigma, care_argnums=(0,))
                    except:
                        alpha = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif)
                        grad = alpha_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, care_argnums=(3,))
                        ws_train = fxns.lsif_alpha_to_ratios(alpha, xs_train, xs_basis, sigma, B, max_ratio)
                        grad1 = objective_given_ws.grad(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, care_argnums=(0,))
                        print grad1
                        grad2 = objective_given_ws.grad(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, care_argnums=(3,))
                        print grad2
                        pdb.set_trace()
                return wrapped

            def log_c_lsif_sigma_optimizer_out_f((c_lsif, sigma, c_pred, B), (objective, dobjective_dlog_c_lsif_sigma, _log_c_lsif_sigma), new_log_c_lsif_sigma):
#                pdb.set_trace()
                return (np.exp(new_log_c_lsif_sigma[0]), np.exp(new_log_c_lsif_sigma[1]), c_pred, B)

            grad_c_lsif_sigma_horse_optimizer = optimizers.scipy_minimize_optimizer(method=c_lsif_sigma_grad_scipy_minimize_method, verbose=c_lsif_sigma_grad_scipy_minimize_verbose, info_f=c_lsif_sigma_grad_scipy_minimize_info_f, options=c_lsif_sigma_grad_scipy_minimize_options)
            grad_c_lsif_sigma_optimizer = optimizers.get_stuff_optimizer(grad_c_lsif_sigma_horse_optimizer, (log_c_lsif_sigma_optimizer_get_objective, log_c_lsif_sigma_optimizer_get_dobjective_dlog_c_lsif_sigma, log_c_lsif_sigma_optimizer_get_log_c_lsif_sigma), log_c_lsif_sigma_optimizer_out_f)

        else:
#        if True:

            def c_lsif_sigma_optimizer_get_c_lsif_sigma_ranges(c_lsif, sigma, c_pred, B):
                print 'c_lsif, sigma'
                us_test = np.dot(xs_test, B)
                center = np.mean(us_test, axis=0)
                diff = us_test - center
                dists = np.sum(diff * diff, axis=1) ** 0.5
                sigma_range = np.percentile(dists, c_lsif_sigma_grid_search_sigma_percentiles)
                print sigma_range, 'sigma_range'
#                print center, 'center'
#                pdb.set_trace()
                return (c_lsif_sigma_grid_search_c_lsif_range, sigma_range)
#                return (c_lsif_sigma_grid_search_c_lsif_range, c_lsif_sigma_grid_search_sigma_range)

            def c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_pred, B):
                print c_lsif, sigma
#                pdb.set_trace()
                return lambda _c_lsif, _sigma: weighted_lsif_objective_cv_given_B(xs_train, xs_test, _c_lsif, B, _sigma, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_oos, num_folds_lsif)

            def c_lsif_sigma_optimizer_out_f((c_lsif, sigma, c_pred, B), (objective, ranges), (new_c_lsif, new_sigma)):
                return (new_c_lsif, new_sigma, c_pred, B)

            c_lsif_sigma_horse_optimizer = optimizers.grid_search_optimizer()
            c_lsif_sigma_optimizer = optimizers.get_stuff_optimizer(c_lsif_sigma_horse_optimizer, (c_lsif_sigma_optimizer_get_objective, c_lsif_sigma_optimizer_get_c_lsif_sigma_ranges), c_lsif_sigma_optimizer_out_f)

        # make B optimizer

        def B_optimizer_get_B(c_lsif, sigma, c_pred, B):
            print 'B'
            if False:
                alpha = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif)
                grad = alpha_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, care_argnums=(3,))
                ws_train = fxns.lsif_alpha_to_ratios(alpha, xs_train, xs_basis, sigma, B, max_ratio)
                grad1 = objective_given_ws.grad(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, care_argnums=(0,))
                print grad1
                grad2 = objective_given_ws.grad(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, care_argnums=(3,))
                print grad2
                pdb.set_trace()
                grad3 = objective_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, care_argnums=(3,))
            return B
            
        def B_optimizer_get_objective(c_lsif, sigma, c_pred, B):
            objective = lambda B: objective_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred)
            return objective

        def B_optimizer_get_dobjective_dB(c_lsif, sigma, c_pred, B):
            dobjective_dB = lambda B: objective_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, care_argnums=(3,))
            return dobjective_dB

        def B_optimizer_out_f((c_lsif, sigma, c_pred, B), (objective, dobjective_dB, _B), new_B):
            return (c_lsif, sigma, c_pred, new_B)

        if unconstrained:
            B_horse_optimizer = optimizers.scipy_minimize_optimizer(method=unconstrained_scipy_minimize_method, verbose=unconstrained_scipy_minimize_verbose, info_f=unconstrained_scipy_minimize_info_f, options=unconstrained_scipy_minimize_options)
        else:
            B_horse_optimizer = optimizers.pymanopt_optimizer(**pymanopt_options)

        B_optimizer = optimizers.get_stuff_optimizer(B_horse_optimizer, (B_optimizer_get_objective, B_optimizer_get_dobjective_dB, B_optimizer_get_B), B_optimizer_out_f)

        # make actual optimizer
        
#        horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, grad_c_lsif_sigma_optimizer))#, B_optimizer))
#        horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, B_optimizer))
#        horse_optimizer = optimizers.many_optimizer((grad_c_lsif_sigma_optimizer, B_optimizer))
        if (not learn_projection) and (not learn_weights):
            assert many_optimizer_num_cycles == 1
            horse_optimizer = optimizers.many_optimizer((c_pred_optimizer,), num_cycles=many_optimizer_num_cycles)
        elif (not learn_projection) and learn_weights:
            assert many_optimizer_num_cycles == 1
            horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_pred_optimizer,), num_cycles=many_optimizer_num_cycles)
        elif learn_projection and learn_weights:
            horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_pred_optimizer, B_optimizer), num_cycles=many_optimizer_num_cycles)
        else:
            assert False
#        horse_optimizer = optimizers.many_optimizer((B_optimizer,))
#        horse_optimizer = optimizers.many_optimizer((c_pred_optimizer, ))
        optimizer = optimizers.multiple_optimizer(horse_optimizer, num_tries=num_tries, num_args=4)

        # do debugging stuff
        
        for i in range(3):
            if False:
                final_fxn.grad_check(ys_train, xs_train, xs_test, ws_full_train, sigma_init_f(), B_init_f(), c_lsqr_loss, c_lsqr_loss_eval, c_lsqr_init_f(), c_lsif_init_f(), max_ratio, add_reg, xs_basis, add_reg_lsif, care_argnums=(5,))

        # do the optimizing
        partialled_objective_given_B = lambda c_lsif, sigma, c_pred, B: objective_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred)
                
        c_lsif_init_f = lambda: 100000
        sigma_init_f = lambda: 100000
        c_pred_init_f = lambda: 1.
        if learn_projection:
            B_init_f = B_init_f_getter(xs_train, ys_train, xs_test)
        else:
            B_init_f = lambda : np.eye(xs_train.shape[1])

        c_lsif_fit, sigma_fit, c_pred_fit, B_fit = optimizer.optimize(partialled_objective_given_B, c_lsif_init_f, sigma_init_f, c_pred_init_f, B_init_f)
        
        # create predictor

        ws_given_B = fxns.two_step( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            g=alpha_given_B, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
            g_argnums=(0,1,2,3,4,5,6,7),
            h_argnums=(0,5,2,3,6),
            g_val_h_argnum=0
            )

        ws_train = ws_given_B(xs_train, xs_test, sigma_fit, B_fit, c_lsif_fit, xs_basis, max_ratio, add_reg_lsif)

        print 'N_eff:', ws_train.sum()**2 / np.dot(ws_train, ws_train)

        b_opt_fit = b_opt_given_ws(B_fit, xs_train, ys_train, ws_train, c_pred_fit)

        if unconstrained:
            b_predictor = B_fit
        else:
            b_predictor = np.dot(B_fit, b_opt_fit)
         
        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, b_predictor))

        if not plot_b_info is None:
            plot_b_info(xs_train, xs_test, ys_train, ys_test, b_predictor / np.linalg.norm(b_predictor), ws_train, predictor)

        return predictor

    fitter.tradeoff_weight_reg = tradeoff_weight_reg
    
    return fitter
