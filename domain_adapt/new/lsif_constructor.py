import fxns, optimizers, autograd, autograd.numpy as np, scipy
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching
import domain_adapt.domain_adapt.SDR_data as data
import utils
import itertools

def project(xs, B):
    return np.concatenate((np.dot(xs,B), np.ones((len(xs),1))), axis=1)

def lsif_ratio_fitter(which_loss, num_basis=100, tradeoff_weight_reg=0., u_dim=None, pseudo=1., which_B_init='random_projection', use_initial_sigma_range=False, uniform_full_weights=False, weighted_cheating_init=False, hardcoded_B_getter=None, learn_weights=True, learn_projection=False, no_projection=False, max_ratio=5, unconstrained=False, lsif_least_squares=True, KDE_ratio=False, c_pred_use_test=False, num_folds_c_pred=3, plot_b_info=None, num_tries=1, unconstrained_scipy_minimize_method=None, unconstrained_scipy_minimize_options={}, unconstrained_scipy_minimize_info_f=lambda x: None, unconstrained_scipy_minimize_verbose=1, quad_opt_warm_start=True, pymanopt_options={'logverbosity':2, 'maxiter':100}, linesearch_method='brent', linesearch_options={}, linesearch_init_window_width=100, c_lsif_sigma_grad_warm_start=False, c_lsif_sigma_grad_scipy_minimize_method=None, c_lsif_sigma_grad_scipy_minimize_options={'maxiter':1}, c_lsif_sigma_grad_scipy_minimize_info_f=lambda x: None, c_lsif_sigma_grad_scipy_minimize_verbose=1, c_pred_line_search=False, c_pred_grid_search=True, c_pred_grad_warm_start=False, c_pred_grad_scipy_minimize_method=None, c_pred_grad_scipy_minimize_options={}, c_pred_grid_search_c_pred_range=None, c_pred_grad_scipy_minimize_info_f=lambda x: None, c_pred_grad_scipy_minimize_verbose=1, c_lsif_sigma_grid_search=True, num_folds_lsif=3, c_lsif_sigma_grid_search_c_lsif_range=None, c_lsif_sigma_grid_search_sigma_range=None, c_lsif_sigma_grid_search_sigma_percentiles=None, many_optimizer_num_cycles=1, b_pred_warm_start=True, b_pred_scipy_minimize_method=None, b_pred_scipy_minimize_options={'maxiter':100}, b_pred_scipy_minimize_info_f=lambda x: None, b_pred_scipy_minimize_verbose=0, c_ll_grid_search_c_ll_range=None, num_folds_c_ll=3, tradeoff_UB=0., _c_ll=100., _c_full=None, num_folds_c_full=3, c_full_grid_search_c_full_range=None, last_pass=False, use_train_oos=False, add_reg_pred=False, weight_loss=False):
    """
    unconstrained options are for 1d unconstrained projections
    pymanopt options is for manifold optimization
    linesearch options is for c_pred optimization
    c_lsif_sigma_grad options is for gradient descent optimization of c_lsif and sigma
    c_pred_grad options is for gradient descent optimization of c_pred
    random_projection is the dim of u, if used
    """

    if not weight_loss:
        assert uniform_full_weights
    
    if c_lsif_sigma_grid_search_c_lsif_range is None:
        c_lsif_sigma_grid_search_c_lsif_range = 10**(np.arange(-4,4).astype(float))
#        c_lsif_sigma_grid_search_c_lsif_range = 10**(np.arange(-3,2).astype(float))

    if c_lsif_sigma_grid_search_sigma_range is None:
        c_lsif_sigma_grid_search_sigma_range = np.linspace(0.1,10,4)

    if c_lsif_sigma_grid_search_sigma_percentiles is None:
        #c_lsif_sigma_grid_search_sigma_percentiles = np.array([20.,35.,50.,65.,80.])
#        c_lsif_sigma_grid_search_sigma_percentiles = np.array([10.,25.])#,50.])#,75.])
#        c_lsif_sigma_grid_search_sigma_percentiles = np.array([25.,50.,75.])
#        c_lsif_sigma_grid_search_sigma_percentiles = np.linspace(10.,90.,9)
        c_lsif_sigma_grid_search_sigma_percentiles = np.linspace(.1,99.9,9)

    if c_pred_grid_search_c_pred_range is None:
        c_pred_grid_search_c_pred_range = 10**(np.arange(-4,2).astype(float))

    if c_ll_grid_search_c_ll_range is None:
        c_ll_grid_search_c_ll_range = 10**(np.arange(-3,2).astype(float))

    if c_full_grid_search_c_full_range is None:
        c_full_grid_search_c_full_range = 10**(np.arange(-2,3).astype(float))
        c_full_grid_search_c_full_range = 10**(np.arange(-4,4).astype(float))

    #lsif_ratio_fitter.c_ll = c_ll
    #lsif_ratio_fitter.c_full = c_full
    
#    @basic.do_cprofile('cumtime')
    def fitter(xs_train, xs_test, ys_train, ys_test=None, xs_train_oos=None, xs_test_oos=None, ys_train_oos=None, ys_test_oos=None):

#        true_B = np.zeros((xs_train.shape[1],2))
#        true_B[0,0] = 1.
#        true_B[1,1] = 1.
        print 'fitting:', fitter.__dict__

        if use_train_oos:
            xs_train = np.concatenate((xs_train, xs_train_oos), axis=0)
            ys_train = np.concatenate((ys_train, ys_train_oos))

#        print 'tradeoff_weight_reg: %.5f' % tradeoff_weight_reg
#        print 'tradeoff_UB: %.5f' % tradeoff_UB

        # define constants

        xs_basis = xs_test[0:num_basis]
#        add_reg_pred = False # for evaluating loss during training
        add_reg_pred_oos = False
        add_reg_lsif = True # for training
        add_reg_lsif_oos = False

#        if not learn_projection:
#            assert B_init_f_getter == None

        # calculate full weights
        #c_full = lsif_ratio_fitter.c_full

#        print lsif_ratio_fitter.c_full
            
        if uniform_full_weights:
            ws_full_train = np.ones(len(xs_train), dtype=float)
        elif _c_full is None:
            
            def kliep_objective_oos(_c_full):
                from sklearn.model_selection import KFold
                kf = KFold(n_splits=num_folds_c_full)
                losses = 0.
                for ((train_is_idx, train_oos_idx), (test_is_idx, test_oos_idx)) in itertools.izip(kf.split(xs_train), kf.split(xs_test)):
                    xs_train_is = xs_train[train_is_idx]
                    xs_train_oos = xs_train[train_oos_idx]
                    xs_test_is = xs_test[test_is_idx]
                    xs_test_oos = xs_test[test_oos_idx]
                    alpha_fit = fxns.loglinear_alpha_given_x(xs_train_is, xs_test_is, _c_full)
                    fold_test_loss = fxns.kliep_objective(xs_train_oos, xs_test_oos, alpha_fit)
                    losses += fold_test_loss
                return losses

            full_N_effs = np.array([fxns.N_eff(fxns.loglinear_ratios_given_x(xs_train, xs_test, __c_full)) for __c_full in c_full_grid_search_c_full_range])
#            N_eff_threshold = 0.1 * len(xs_train)
            N_eff_threshold = 0.02 * len(xs_train)
            above_threshold = full_N_effs > N_eff_threshold
            print full_N_effs, 'N_effs'
            above_threshold_c_full_range = c_full_grid_search_c_full_range[above_threshold]

            if len(above_threshold_c_full_range) == 0:
                ws_full_train = np.ones(len(xs_train), dtype=float)
            else:
                c_full_optimizer = optimizers.grid_search_optimizer()
                c_full, = c_full_optimizer.optimize(kliep_objective_oos, (above_threshold_c_full_range,))
                ws_full_train = fxns.loglinear_ratios_given_x(xs_train, xs_test, c_full)
#                pdb.set_trace()

        else:

            c_full = _c_full
            ws_full_train = fxns.loglinear_ratios_given_x(xs_train, xs_test, c_full)
#        print ws_full_train

        # define objective fxn

        # define predictive loss

        if which_loss == 'square':
            b_opt_given_ws = fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt) # B, xs_train, ys_train, ws_train, c_pred
            weighted_loss_given_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt) # B, xs_train, ys_train, ws_train, b_opt, c_pred, add_reg_pred
            weighted_loss_and_ll_given_ws_constructor = fxns.two_step.autograd_fxn
            pred_loss = fxns.squared_losses
        elif which_loss == 'logistic':
            logistic_regression_objective = fxns.objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product, _val=fxns.logistic_regression_objective_helper, _arg_shape=lambda *args: args[0].shape[1]+1) # fix if not adding in intercept term into us
            cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
            b_opt_given_ws = fxns.cvx_opt( # B, xs_train, ys_train, ws_train, c_pred
                lin_solver=cg_solver,
                objective=logistic_regression_objective,
                dobjective_dx=fxns.dopt_objective_dx.autograd_fxn(logistic_regression_objective),
                optimizer=optimizers.scipy_minimize_optimizer(method=b_pred_scipy_minimize_method, verbose=b_pred_scipy_minimize_verbose, info_f=b_pred_scipy_minimize_info_f, options=b_pred_scipy_minimize_options),
                warm_start=b_pred_warm_start
                )
            weighted_loss_given_b_opt = fxns.fxn.autograd_fxn(_val=fxns.logistic_regression_loss)
            weighted_loss_and_ll_given_ws_constructor = lambda *args, **kwargs: fxns.fxn.wrap_primitive(fxns.g_thru_f_opt_new(*args, **kwargs))
            pred_loss = fxns.logistic_regression_losses

        # define loss loss (ll)

        ll_given_losses = fxns.two_step( # losses, B, c_ll, xs_train
            g=fxns.fxn(_val=fxns.unweighted_lsqr_b_opt), # B, xs_train, losses, c_ll
            h=fxns.fxn(_val=fxns.ll_given_b_opt_ll), # b_opt_ll, losses, xs_train, B (use this instead of squared loss fxn to avoid specifying add_reg and uniform weights)
            g_argnums=(1,3,0,2),
            h_argnums=(0,3,1),
            g_val_h_argnum=0
            )

        def losses(B, xs_train, ys_train, b_opt):
            #print xs_train.shape
            return pred_loss(project(xs_train, B), ys_train, b_opt)
        
        ll_given_b_opt = fxns.two_step.autograd_fxn( # b_opt, B, c_ll, xs_train, ys_train
            g=fxns.fxn(_val=losses), # B, xs_train, ys_train, b_opt
            h=ll_given_losses, # losses, B, c_ll, xs_train
            g_argnums=(1,3,4,0),
            h_argnums=(1,2,3),
            g_val_h_argnum=0,
            )

        conditional_PE_given_b_opt = fxns.fxn(_val=fxns.expected_conditional_PE_dist) # ws_full_train, ws_train

        UB_given_b_opt = fxns.product.autograd_fxn( # b_opt, B, c_ll, xs_train, ys_train, ws_train, ws_full_train, pseudo
                g=conditional_PE_given_b_opt, # ws_full_train, ws_train, pseudo
                h=ll_given_b_opt, # b_opt, B, c_ll, xs_train, ys_train
                g_argnums=(5,6,7),
                h_argnums=(0,1,2,3,4)
                )
        
        # combine prediction loss and loss loss, which are given b_opt

        weighted_loss_and_ll_given_b_opt = fxns.sum( # B, xs_train, ys_train, ws_train, b_opt, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
            fs=[
                weighted_loss_given_b_opt, # B, xs_train, ys_train, ws_train, b_opt, c_pred, add_reg_pred
                UB_given_b_opt, # b_opt, B, c_ll, xs_train, ys_train, ws_train, ws_full_train, pseudo
                ],
            fs_argnums=[
                (0,1,2,3,4,5,6),
                (4,0,7,1,2,3,8,9),
                ],
            weights=[
                1.,
                tradeoff_UB,
                ]
            )
                

        weighted_loss_and_ll_given_ws = weighted_loss_and_ll_given_ws_constructor( # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
            g=b_opt_given_ws, # B, xs_train, ys_train, ws_train, c_pred
            h=weighted_loss_and_ll_given_b_opt, # B, xs_train, ys_train, ws_train, b_opt, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
            g_argnums=(0,1,2,3,4),
            h_argnums=(0,1,2,3,4,5,6,7,8),
            g_val_h_argnum=4
            )
        
        if weight_loss:
            loss_argnums = (0,1,2,3,4,5,6,7,8)
        else:
            loss_argnums = (0,1,2,7,4,5,6,7,8)
        objective_given_ws = fxns.sum(# B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
            fs=[
                weighted_loss_and_ll_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg), # ws_train
                ],
            fs_argnums=[
#                (0,1,2,3,4,5,6,7,8),
                loss_argnums,
                (3,),
                ],
            weights=[
                1.,
                tradeoff_weight_reg,
                ]
            )

#        print objective_given_ws.grad(np.eye(xs_train.shape[1]), xs_train, ys_train, np.ones(len(xs_train)), 2., True, care_argnums=(0,))

#        asdf = lambda B: objective_given_ws.val(B, xs_train, ys_train, np.ones(len(xs_train)), 2., True)
#        asdf_grad = autograd.grad(asdf)
#        asdf_grad(np.eye(xs_train.shape[1]))
#        pdb.set_trace()


        if not KDE_ratio:

            objective_given_alpha = fxns.two_step.autograd_fxn( # B, xs_train, ys_train, c_pred, add_reg_pred, lsif_alpha, xs_basis, sigma, max_ratio, c_ll, ws_full_train, pseudo
                g=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
                h=objective_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
                g_argnums=(5,1,6,7,0,8),
                h_argnums=(0,1,2,3,4,9,10,11),
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
    #            objective_given_B_constructor = lambda *args, **kwargs: fxns.fxn.wrap_primitive(fxns.two_step.autograd_fxn(*args, **kwargs))
                objective_given_B_constructor = lambda *args, **kwargs: fxns.two_step.autograd_fxn(*args, **kwargs)

            objective_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
                g=alpha_given_B, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                h=objective_given_alpha, # B, xs_train, ys_train, c_pred, add_reg_pred, lsif_alpha, xs_basis, sigma, max_ratio, c_ll, ws_full_train, pseudo
                g_argnums=(0,1,2,3,4,5,6,7),
                h_argnums=(3,0,8,9,10,5,2,6,11,12,13),
                g_val_h_argnum=5
            )

        else:

            ws_given_B = fxns.fxn(_val=fxns.KDE_ws)

            objective_given_B_constructor = lambda *args, **kwargs: fxns.two_step.autograd_fxn(*args, **kwargs)
            
            objective_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
                g=ws_given_B, # xs_train, xs_test, sigma, B, c_lsif, max_ratio
                h=objective_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo
                g_argnums=(0,1,2,3,4,6),
                h_argnums=(3,0,8,9,10,11,12,13),
                g_val_h_argnum=3
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
                ans = losses / len(xs_train)
                #pdb.set_trace() ppp
#                print ans, 'pred'
                return ans

            test_loss_given_ws = fxns.fxn.autograd_fxn(_val=weighted_loss_cv_given_ws)

        elif c_pred_use_test:

            def _test_loss_given_ws(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos):
                ws_test = np.ones(len(xs_test))
                return weighted_loss_oos_given_ws_oos(B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, xs_test_oos, ys_test_oos, ws_test)

            test_loss_given_ws = fxns.fxn.autograd_fxn(_val=_test_loss_given_ws)

        if not KDE_ratio:

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

        else:

            test_loss_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos
                g=ws_given_B, # xs_train, xs_test, sigma, B, c_lsif, max_ratio
                h=test_loss_given_ws, # B, xs_train, ys_train, ws_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos
                g_argnums=(0,1,2,3,4,6),
                h_argnums=(3,0,8,9,10,11,12,13),
                g_val_h_argnum=3
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
                    #fold_test_loss = objective_given_B_constructor(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train_is, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos) # fix
                    fold_test_loss = test_loss_given_B(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train_is, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test_oos, ys_test_oos) # fix
                    test_losses += fold_test_loss * len(test_oos_idx)
                return test_losses / len(ys_test)

        else:

            def _final_test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test):
                return test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred_oos, num_folds_c_pred, xs_test, ys_test)
            
        final_test_loss_given_B = fxns.fxn.autograd_fxn(_val=_final_test_loss_given_B)

        def c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B):
            return lambda _c_pred: final_test_loss_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, _c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test)

        def c_pred_optimizer_get_dobjective_dc_pred(c_lsif, sigma, c_pred, c_ll, B):
            return lambda _c_pred: final_test_loss_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, _c_pred, add_reg_pred_oos, num_folds_c_pred, ys_test, care_argnums=(9,))
            
        if c_pred_line_search:

            def c_pred_optimizer_get_c_pred(c_lsif, sigma, c_pred, c_ll, B):
#                print 'c_pred'
                return c_pred
            
            def c_pred_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, _c_pred), new_c_pred):
                return (c_lsif, sigma, new_c_pred, c_ll, B)
        
            c_pred_horse_optimizer = optimizers.scalar_fxn_optimizer(method=linesearch_method, options=linesearch_options, init_window_width=linesearch_init_window_width)
            c_pred_optimizer = optimizers.get_stuff_optimizer(c_pred_horse_optimizer, (c_pred_optimizer_get_objective, c_pred_optimizer_get_c_pred), c_pred_optimizer_out_f)

        elif c_pred_grid_search:

            def c_pred_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, ranges), (new_c_pred,)):
                return (c_lsif, sigma, new_c_pred, c_ll, B)

            def c_pred_optimizer_get_c_pred_ranges(c_lsif, sigma, c_pred, c_ll, B):
                return (c_pred_grid_search_c_pred_range,)
            
            c_pred_horse_optimizer = optimizers.grid_search_optimizer()
            c_pred_optimizer = optimizers.get_stuff_optimizer(c_pred_horse_optimizer, (c_pred_optimizer_get_objective, c_pred_optimizer_get_c_pred_ranges), c_pred_optimizer_out_f)

        else:

            def log_c_pred_optimizer_get_log_c_pred(c_lsif, sigma, c_pred, c_ll, B):
#                print 'c_pred'
                return np.log(c_pred)
            
            def log_c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B):
                horse = c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B)
                return lambda log_c_pred: horse(np.exp(log_c_pred))

            def log_c_pred_optimizer_get_dobjective_dlog_c_pred(c_lsif, sigma, c_pred, c_ll, B):
                horse = fxns.fxn.autograd_fxn(_val=log_c_pred_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B))
                return lambda log_c_pred: horse.grad(log_c_pred, care_argnums=(0,))

            def log_c_pred_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, dobjective_dlog_c_pred, _log_c_pred), new_log_c_pred):
                return (c_lsif, sigma, np.exp(new_log_c_pred), c_ll, B)

            c_pred_horse_optimizer = optimizers.scipy_minimize_optimizer(method=c_pred_grad_scipy_minimize_method, verbose=c_pred_grad_scipy_minimize_verbose, info_f=c_pred_grad_scipy_minimize_info_f, options=c_pred_grad_scipy_minimize_options)
            c_pred_optimizer = optimizers.get_stuff_optimizer(c_pred_horse_optimizer, (log_c_pred_optimizer_get_objective, log_c_pred_optimizer_get_dobjective_dlog_c_pred, log_c_pred_optimizer_get_log_c_pred), log_c_pred_optimizer_out_f)

        # define c_lsif and sigma hyperparameter optimizer

        if not KDE_ratio:

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
                    this_xs_basis = xs_test_is[0:100]
#                    pdb.set_trace()
#                    alpha = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif)
                    alpha = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, this_xs_basis, max_ratio, add_reg_lsif)
                    #grad = alpha_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, care_argnums=(3,))
                    #ws_train = fxns.lsif_alpha_to_ratios(alpha, xs_train, xs_basis, sigma, B, max_ratio)
    #                print ws_train, 'ws_train'
    #                fold_objective = lsif_objective_oos(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_test_is[0:num_basis], max_ratio, add_reg_lsif, xs_train_oos, xs_test_oos, add_reg_lsif_oos)
#                    fold_objective = lsif_objective_oos(xs_train_is, xs_test_is, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, xs_train_oos, xs_test_oos, add_reg_lsif_oos)
                    fold_objective = lsif_objective_oos(xs_train_is, xs_test_is, sigma, B, c_lsif, this_xs_basis, max_ratio, add_reg_lsif, xs_train_oos, xs_test_oos, add_reg_lsif_oos)
                    objectives += fold_objective * len(train_oos_idx)
                return objectives / len(xs_train)

            #weighted_lsif_objective_cv_given_B = fxns.fxn.autograd_fxn(_val=_weighted_lsif_objective_cv_given_B)

        else:

            def _weighted_lsif_objective_cv_given_B(xs_train, xs_test, c_lsif, B, sigma, max_ratio):
                #ws_test = 1. / fxns.KDE_ws(xs_test, xs_train, sigma, B, c_lsif, max_ratio) # switched
                ws_test = fxns.KDE_ws(xs_train, xs_test, sigma, B, c_lsif, max_ratio, switch=True)
#                pdb.set_trace()
                return np.mean(np.log(ws_test))

        weighted_lsif_objective_cv_given_B = fxns.fxn.autograd_fxn(_val=_weighted_lsif_objective_cv_given_B)
                

        if not c_lsif_sigma_grid_search:

            def log_c_lsif_sigma_optimizer_get_log_c_lsif_sigma(c_lsif, sigma, c_pred, c_ll, B):
#                print 'lsif, sigma'
                return np.log(np.array([c_lsif, sigma]))
            
            def log_c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B):
                def horse(log_c_lsif_sigma):
#                    print log_c_lsif_sigma, np.exp(log_c_lsif_sigma), 'inside'
                    if not KDE_ratio:
                        alphas = alpha_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif), 'inside2'
                        return weighted_lsif_objective_cv_given_B(xs_train, xs_test, np.exp(log_c_lsif_sigma[0]), B, np.exp(log_c_lsif_sigma[1]), xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_oos, num_folds_lsif)
                    else:
                        return weighted_lsif_objective_cv_given_B(xs_train, xs_test, np.exp(log_c_lsif_sigma[0]), B, np.exp(log_c_lsif_sigma[1]), max_ratio)
                        
                return horse

            def log_c_lsif_sigma_optimizer_get_dobjective_dlog_c_lsif_sigma(c_lsif, sigma, c_pred, c_ll, B):
                horse = fxns.fxn.autograd_fxn(_val=log_c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B))
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

            def log_c_lsif_sigma_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, dobjective_dlog_c_lsif_sigma, _log_c_lsif_sigma), new_log_c_lsif_sigma):
                return (np.exp(new_log_c_lsif_sigma[0]), np.exp(new_log_c_lsif_sigma[1]), c_pred, B)

            grad_c_lsif_sigma_horse_optimizer = optimizers.scipy_minimize_optimizer(method=c_lsif_sigma_grad_scipy_minimize_method, verbose=c_lsif_sigma_grad_scipy_minimize_verbose, info_f=c_lsif_sigma_grad_scipy_minimize_info_f, options=c_lsif_sigma_grad_scipy_minimize_options)
            grad_c_lsif_sigma_optimizer = optimizers.get_stuff_optimizer(grad_c_lsif_sigma_horse_optimizer, (log_c_lsif_sigma_optimizer_get_objective, log_c_lsif_sigma_optimizer_get_dobjective_dlog_c_lsif_sigma, log_c_lsif_sigma_optimizer_get_log_c_lsif_sigma), log_c_lsif_sigma_optimizer_out_f)

        else:

            initial_sigma_range = []

            def c_lsif_sigma_optimizer_get_c_lsif_sigma_ranges(c_lsif, sigma, c_pred, c_ll, B):
#                print 'c_lsif, sigma'
                if use_initial_sigma_range and (not (initial_sigma_range == [])):
                    return (c_lsif_sigma_grid_search_c_lsif_range, initial_sigma_range)
                else:
                    us_test = np.dot(xs_test, B)
                    us_train = np.dot(xs_train, B)
                    all_us = np.concatenate((us_test,us_train), axis=0)
                    center = np.mean(all_us, axis=0)
                    diff = all_us - center
                    dists = np.sum(diff * diff, axis=1) ** 0.5
#                print np.mean(dists), 'mean dist'
                    sigma_range = np.percentile(dists, c_lsif_sigma_grid_search_sigma_percentiles)
                    for sigma in sigma_range:
                        initial_sigma_range.append(sigma)
#                sigma_range = np.linspace(3,20,20)
#                print sigma_range, 'sigma_range'
#                print center, 'center'
#                pdb.set_trace()
                    return (c_lsif_sigma_grid_search_c_lsif_range, initial_sigma_range)
#                return (c_lsif_sigma_grid_search_c_lsif_range, c_lsif_sigma_grid_search_sigma_range)

            def c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B):
#                print c_lsif, sigma
#                pdb.set_trace()
                if not KDE_ratio:
                    return lambda _c_lsif, _sigma: weighted_lsif_objective_cv_given_B(xs_train, xs_test, _c_lsif, B, _sigma, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_oos, num_folds_lsif)
                else:
                    return lambda _c_lsif, _sigma: weighted_lsif_objective_cv_given_B(xs_train, xs_test, _c_lsif, B, _sigma, max_ratio)

            def c_lsif_sigma_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, ranges), (new_c_lsif, new_sigma)):
#                return (0.001, 1.1, c_pred, c_ll, B)
                return (new_c_lsif, new_sigma, c_pred, c_ll, B)

            c_lsif_sigma_horse_optimizer = optimizers.grid_search_optimizer()
            c_lsif_sigma_optimizer = optimizers.get_stuff_optimizer(c_lsif_sigma_horse_optimizer, (c_lsif_sigma_optimizer_get_objective, c_lsif_sigma_optimizer_get_c_lsif_sigma_ranges), c_lsif_sigma_optimizer_out_f)

        # make c_ll optimizer

        losses_given_ws = fxns.two_step( # B, xs_train, ys_train, ws_train, c_pred
            g=b_opt_given_ws, # B, xs_train, ys_train, ws_train, c_pred
            h=fxns.fxn(_val=losses), # B, xs_train, ys_train, b_opt
            g_argnums=(0,1,2,3,4),
            h_argnums=(0,1,2),
            g_val_h_argnum=3,
            )

        if not KDE_ratio:
        
            losses_given_alpha = fxns.two_step( # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio, c_pred, ys_train
                g=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
                h=losses_given_ws, # B, xs_train, ys_train, ws_train, c_pred
                g_argnums=(0,1,2,3,4,5),
                h_argnums=(4,1,7,6),
                g_val_h_argnum=3
                )

            losses_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, c_pred, ys_train
                g=alpha_given_B, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                h=losses_given_alpha, # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio, c_pred, ys_train
                g_argnums=(0,1,2,3,4,5,6,7),
                h_argnums=(0,5,2,3,6,8,9),
                g_val_h_argnum=0
                )

        else:

            losses_given_B = objective_given_B_constructor( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, c_pred, ys_train
                g=ws_given_B, # xs_train, xs_test, sigma, B, c_lsif, max_ratio
                h=losses_given_ws, # B, xs_train, ys_train, ws_train, c_pred
                g_argnums=(0,1,2,3,4,6),
                h_argnums=(3,0,9,8),
                g_val_h_argnum=3
                )

        ll_oos_given_losses = fxns.two_step( # losses, B, c_ll, xs_train, losses_oos, xs_train_oos
            g=fxns.fxn(_val=fxns.unweighted_lsqr_b_opt), # B, xs_train, losses, c_ll
            h=fxns.fxn(_val=fxns.ll_given_b_opt_ll), # b_opt_ll, losses_oos, xs_train_oos, B
            g_argnums=(1,3,0,2),
            h_argnums=(4,5,1),
            g_val_h_argnum=0
            )
        
        def ll_oos_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, num_folds_c_ll, c_ll):
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=num_folds_c_ll)
            losses = losses_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, c_pred, ys_train)
            
            test_losses = 0.
            for (train_is_idx, train_oos_idx) in kf.split(xs_train):
                xs_train_is = xs_train[train_is_idx]
                xs_train_oos = xs_train[train_oos_idx]
                losses_is = losses[train_is_idx]
                losses_oos = losses[train_oos_idx]
                test_losses += ll_oos_given_losses(losses_is, B, c_ll, xs_train_is, losses_oos, xs_train_oos)
#            print test_losses, 'll' # ppp
#            pdb.set_trace()
            return test_losses

        def c_ll_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B):
            return lambda _c_ll: ll_oos_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, num_folds_c_ll, _c_ll)
                                                       
        def c_ll_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, ranges), (new_c_ll,)):
            return (c_lsif, sigma, c_pred, new_c_ll, B)

        def c_ll_optimizer_get_c_ll_ranges(c_lsif, sigma, c_pred, c_ll, B):
            return (c_ll_grid_search_c_ll_range,)

        c_ll_horse_optimizer = optimizers.grid_search_optimizer()
        c_ll_optimizer = optimizers.get_stuff_optimizer(c_ll_horse_optimizer, (c_ll_optimizer_get_objective, c_ll_optimizer_get_c_ll_ranges), c_ll_optimizer_out_f)

        # make B optimizer

        def B_optimizer_get_B(c_lsif, sigma, c_pred, c_ll, B):
#            print 'B'
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
            
        def B_optimizer_get_objective(c_lsif, sigma, c_pred, c_ll, B):
            objective = lambda B: objective_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo)
            return objective

        def B_optimizer_get_dobjective_dB(c_lsif, sigma, c_pred, c_ll, B):
            dobjective_dB = lambda B: objective_given_B.grad(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo, care_argnums=(3,))
            return dobjective_dB

        def B_optimizer_out_f((c_lsif, sigma, c_pred, c_ll, B), (objective, dobjective_dB, _B), new_B):
            return (c_lsif, sigma, c_pred, c_ll, new_B)

        if unconstrained:
            B_horse_optimizer = optimizers.scipy_minimize_optimizer(method=unconstrained_scipy_minimize_method, verbose=unconstrained_scipy_minimize_verbose, info_f=unconstrained_scipy_minimize_info_f, options=unconstrained_scipy_minimize_options)
        else:
            B_horse_optimizer = optimizers.pymanopt_optimizer(**pymanopt_options)

        B_optimizer = optimizers.get_stuff_optimizer(B_horse_optimizer, (B_optimizer_get_objective, B_optimizer_get_dobjective_dB, B_optimizer_get_B), B_optimizer_out_f)

        # make actual optimizer

        coord_optimizers = []
        if learn_weights:
            coord_optimizers.append(c_lsif_sigma_optimizer)
        coord_optimizers.append(c_pred_optimizer)
        if _c_ll is None:
            assert learn_projection #or (not no_projection)
            coord_optimizers.append(c_ll_optimizer)
        if learn_projection:
            coord_optimizers.append(B_optimizer)
        else:
            assert many_optimizer_num_cycles == 1

#        if (not learn_projection) and (not learn_weights) and (not random_projection):
#            assert many_optimizer_num_cycles == 1
#            horse_optimizer = optimizers.many_optimizer((c_pred_optimizer,), num_cycles=many_optimizer_num_cycles)
#        elif (not learn_projection) and learn_weights and (not random_projection):
#            assert many_optimizer_num_cycles == 1
#            horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_pred_optimizer,), num_cycles=many_optimizer_num_cycles)
#        elif learn_projection and learn_weights:
#            assert not random_projection
#            if lsif_ratio_fitter.c_ll is None:
#                horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_pred_optimizer, c_ll_optimizer, B_optimizer), num_cycles=many_optimizer_num_cycles)
#            else:
#                horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_pred_optimizer, B_optimizer), num_cycles=many_optimizer_num_cycles)
#        elif random_projection:
#            assert (not learn_projection)
#            assert many_optimizer_num_cycles == 1
#            if learn_weights:
#                horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_pred_optimizer,), num_cycles=many_optimizer_num_cycles)
#            else:
#                horse_optimizer = optimizers.many_optimizer((c_pred_optimizer,), num_cycles=many_optimizer_num_cycles)
#        else:
#            assert False

        if last_pass:
            horse_optimizer = optimizers.many_optimizer(coord_optimizers, num_cycles=many_optimizer_num_cycles, last_round_start=0, last_round_end=-1)
        else:
            horse_optimizer = optimizers.many_optimizer(coord_optimizers, num_cycles=many_optimizer_num_cycles)
        optimizer = optimizers.multiple_optimizer(horse_optimizer, num_tries=num_tries, num_args=5)


        # do the optimizing
        partialled_objective_given_B = lambda c_lsif, sigma, c_pred, c_ll, B: objective_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif, ys_train, c_pred, add_reg_pred, c_ll, ws_full_train, pseudo)

        # make sure ws_given_B is defined for the sake of smart initializing
        if not KDE_ratio:
        
            ws_given_B = fxns.two_step( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                g=alpha_given_B, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                h=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
                g_argnums=(0,1,2,3,4,5,6,7),
                h_argnums=(0,5,2,3,6),
                g_val_h_argnum=0
                )


        if not learn_weights:
            c_lsif_init_f = lambda: 100000
            sigma_init_f = lambda: 100000
        else:
            c_lsif_init_f = lambda: np.exp(np.random.uniform())
            sigma_init_f = lambda: np.exp(np.random.uniform())
        c_pred_init_f = lambda: np.random.uniform(0.1,2)
#        pdb.set_trace()
        if not no_projection:
            # will project to lower dim space
#            B_init_f = lambda: true_B
            #B_init_f = B_init_f_getter(xs_train, ys_train, xs_test)
            if which_B_init == 'random_projection':
                #assert B_init_f_getter is None
                B_init_f = lambda: utils.ortho(np.random.normal(size=(xs_train.shape[1], u_dim)))
                
            elif which_B_init in ['smart', 'unweighted', 'weighted']:

                class supervised_initializer(object):
                        
                    def __init__(self, unweighted_first=True, weighted_cheating=False):

                        self.unweighted_first, self.weighted_cheating = unweighted_first, weighted_cheating
                        
                        if not weighted_cheating_init:
                            c_lsif, sigma, c_pred, c_ll, B = None, None, None, None, np.eye(xs_train.shape[1])
                            c_lsif, sigma, c_pred, c_ll, B = c_lsif_sigma_optimizer(c_lsif, sigma, c_pred, c_ll, B)
                            c_lsif, sigma, c_pred, c_ll, B = c_pred_optimizer(c_lsif, sigma, c_pred, c_ll, B)
                            if not KDE_ratio:
                                ws_train = ws_given_B(xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif)
                            else:
                                ws_train = ws_given_B(xs_train, xs_test, sigma, B, c_lsif, max_ratio)
                            weighted_b_opt = b_opt_given_ws(B, xs_train, ys_train, ws_train, c_pred)
                            self.weighted_b_opt = weighted_b_opt[:-1]
                        else:
                            if which_loss == 'square':
                                cheating_fitter = fxns.sklearn_ridge_fitter(alphas=10**np.linspace(-3,3,7), cheat=True)
                                self.weighted_b_opt = cheating_fitter(xs_train, xs_test, ys_train, ys_test).coef_
                            elif which_loss == 'logistic':
                                cheating_fitter = fxns.sklearn_ridge_logreg_fitter(alphas=10**np.linspace(-3,3,7), cheat=True)
                                self.weighted_b_opt = cheating_fitter(xs_train, xs_test, ys_train, ys_test).coef_

                        c_lsif, sigma, c_pred, c_ll, B = 100000, 100000, None, None, np.eye(xs_train.shape[1])
                        c_lsif, sigma, c_pred, c_ll, B = c_pred_optimizer(c_lsif, sigma, c_pred, c_ll, B)
                        ws_train = np.ones(len(xs_train), dtype=float)
                        unweighted_b_opt = b_opt_given_ws(B, xs_train, ys_train, ws_train, c_pred)
                        self.unweighted_b_opt = unweighted_b_opt[:-1]

                        self.counter = 0

                    def __call__(self):
                        B = np.random.normal(size=(xs_train.shape[1], u_dim))
                        if not (u_dim == 1 and self.counter >= 2):
                            if self.counter % 3 == 0:
                                if self.unweighted_first:
                                    B[:,0] = self.unweighted_b_opt
                                else:
                                    B[:,0] = self.weighted_b_opt
                            elif self.counter % 3 == 1:
                                if self.unweighted_first:
                                    B[:,0] = self.weighted_b_opt
                                else:
                                    B[:,0] = self.unweighted_b_opt
                            elif self.counter % 3 == 2:
                                B[:,0] = self.unweighted_b_opt
                        self.counter += 1
                        return utils.ortho(B)

                if which_B_init in ['smart', 'unweighted']:
                    B_init_f = supervised_initializer(unweighted_first=True)
                elif which_B_init == 'weighted':
                    B_init_f = supervised_initializer(unweighted_first=False)
                else:
                    assert False

                #B_init_f = lambda: true_B
            elif which_B_init in ['sir','save','phdres']:
                SIR_B = fxns.SIR_directions(xs_train, ys_train, method=which_B_init)
#                print SIR_B[:,0]
#                pdb.set_trace()
                B_init_f = lambda: SIR_B[:,0:u_dim]
#                pdb.set_trace()
            elif which_B_init == 'hardcoded':
                B_init_f = lambda: hardcoded_B_getter(xs_train, ys_train, xs_test)
            else:
                assert False
        elif no_projection:
            assert u_dim is None
            B_init_f = lambda : np.eye(xs_train.shape[1])
        #if learn_projection or no_pro:
        #    c_ll_init_f = lambda: None
        #else:
        c_ll_init_f = lambda: 100.
            
        for i in range(3):
            if False:
                objective_given_B.grad_check(xs_train, xs_test, 2., B_init_f(), 1., xs_basis, max_ratio, add_reg_lsif, ys_train, 1., add_reg_pred, c_ll, ws_full_train, pseudo, care_argnums=(3,))
#                final_fxn.grad_check(ys_train, xs_train, xs_test, ws_full_train, sigma_init_f(), B_init_f(), c_lsqr_loss, c_lsqr_loss_eval, c_lsqr_init_f(), c_lsif_init_f(), max_ratio, add_reg, xs_basis, add_reg_lsif, care_argnums=(5,))
        #assert False

        c_lsif_fit, sigma_fit, c_pred_fit, c_ll_fit, B_fit = optimizer.optimize(partialled_objective_given_B, c_lsif_init_f, sigma_init_f, c_pred_init_f, c_ll_init_f, B_init_f)

#        print B_fit
        
#        B_fit = np.array([[1.] + list(np.zeros(xs_train.shape[1]-1))]).T
#        B_fit = true_B
        
        # create predictor

        if not KDE_ratio:
        
            ws_train = ws_given_B(xs_train, xs_test, sigma_fit, B_fit, c_lsif_fit, xs_basis, max_ratio, add_reg_lsif)

        else:

            ws_train = ws_given_B(xs_train, xs_test, sigma_fit, B_fit, c_lsif_fit, max_ratio)
    

        b_opt_fit = b_opt_given_ws(B_fit, xs_train, ys_train, ws_train, c_pred_fit)

        if unconstrained:
            assert False
            b_predictor = B_fit
        else:
#            b_predictor = np.dot(B_fit, b_opt_fit)        
            predictor = fxns.fxn(_val = lambda x: np.dot(project(np.array([x]), B_fit), b_opt_fit)[0])
            predictor.B = B_fit
            predictor.b = b_opt_fit
            N_eff = fxns.N_eff(ws_train)
            print 'N_eff:', N_eff, np.sum(1./ws_train)
            predictor.N_eff = N_eff
#            predictor.

            if not KDE_ratio:
                alpha_fit = alpha_given_B(xs_train, xs_test, sigma_fit, B_fit, c_lsif_fit, xs_basis, max_ratio, add_reg_lsif)
                predictor.get_ws = lambda xs: fxns.lsif_alpha_to_ratios(alpha_fit, xs, xs_basis, sigma_fit, B_fit, max_ratio)
            else:
                predictor.get_ws = lambda xs: fxns.KDE_ws_oos(xs_train, xs_test, sigma_fit, B_fit, c_lsif_fit, max_ratio, xs)

        if not plot_b_info is None:
            plot_b_info(xs_train, xs_test, ys_train, ys_test, ws_train, predictor)
            #plot_b_info(xs_train, xs_test, ys_train, ys_test, b_predictor / np.linalg.norm(b_predictor), ws_train, predictor)

#        pdb.set_trace()
            
        return predictor

    fitter.tradeoff_weight_reg = tradeoff_weight_reg
    fitter.tradeoff_UB = tradeoff_UB
    fitter.pseudo = pseudo
    
    return fitter

