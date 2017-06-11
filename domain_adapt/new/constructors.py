import fxns, optimizers, autograd, autograd.numpy as np, scipy
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching
import domain_adapt.domain_adapt.SDR_data as data
    

def plot_b_info(xs_train, xs_test, ys_train, ys_test, b, ratios, predictor):

    s = 1.75
    alpha = 0.75
    fig, ax = plt.subplots()
    data.data.plot_us(b, xs_train, xs_test, ax, normed=True)
    ratio_ax = ax.twinx()
    ratio_ax.scatter(np.dot(xs_train, b), ratios, alpha=alpha, s=s, label='estimated ratios')
    ratio_ax.set_ylabel('estimated ratios')
    ratio_ax.legend()
    caching.fig_archiver.archive_fig(fig)
    #basic.display_fig_inline(fig)

    ys_train_hat = np.array([predictor(x) for x in xs_train])
    train_error = ys_train - ys_train_hat
    train_loss = np.dot(train_error,train_error)
    
    if not (ys_test is None):
        ys_test_hat = np.array([predictor(x) for x in xs_test])
        test_error = ys_test - ys_test_hat
        test_loss = np.dot(test_error,test_error)
    try:
        caching.fig_archiver.fig_text(['train_loss: %.2f' % train_loss, 'test_loss: %.2f' % test_loss])
    except NameError:
        caching.fig_archiver.fig_text(['train_loss: %.2f' % train_loss])
        
    print 'b_norm:', np.linalg.norm(b)

    fig, ax2 = plt.subplots()
    ax2.set_xlim(ax.get_xlim())
    ax2.scatter(np.dot(xs_train, b), ys_train, label='true train', color='r', alpha=alpha, s=s)
    if not (ys_test is None):
        ax2.scatter(np.dot(xs_test, b), ys_test, label='true test', color='b', s=s)
    ax2.scatter(np.dot(xs_train, b), ys_train_hat, marker='x', label='predicted train', color='orange', alpha=alpha, s=s)
    if not (ys_test is None):
        ax2.scatter(np.dot(xs_test,b), ys_test_hat, marker='x', label='predicted test', color='c', alpha=alpha, s=s)
    ax2.set_xlabel('B\'x')
    ax2.set_ylabel('y')
    ax2.legend()
    caching.fig_archiver.archive_fig(fig)
    #basic.display_fig_inline(fig)
        
    fig, ax = plt.subplots()
    ax.scatter(ys_test, ys_test_hat, s=s, alpha=alpha)
    ax.set_xlabel('true ys_test')
    ax.set_ylabel('predicted ys_test')
    caching.fig_archiver.archive_fig(fig)
    #basic.display_fig_inline(fig)




def logreg_ratios(c_logreg, max_ratio, xs_train, xs_test):
    logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=False)
    dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
    cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
    b_ratio_opt_given_B_fxn = fxns.cvx_opt(# xs_train, xs_test, sigma, B, c_logreg
        lin_solver=cg_solver, 
        objective=logreg_ratio_objective, 
        dobjective_dx=dlogreg_ratio_objective_db
    )

    logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                                   g=b_ratio_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                                   h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                                   g_argnums=(0,1,2,3,4),
                                                   h_argnums=(0,1,2,5,3,6),
                                                   g_val_h_argnum=0,
                                                   )

    import utils
    sigma = utils.median_distance(np.concatenate((xs_train, xs_test), axis=0), np.concatenate((xs_train, xs_test), axis=0))
    B = np.eye(xs_train.shape[1])
    scale_sigma = False
    ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio)

    return ws_train


def baseline_fitter(c_logreg, c_lsqr, sigma, scale_sigma=False, max_ratio=5., plot_b_info=None):

    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        b_ratio_opt_to_b_reg_opt = fxns.two_step( # b_ratio_opt, xs_train, xs_test, ys_train, sigma, B, c_lsqr, scale_sigma, max_ratio
            g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_ratio_opt, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ys_train, ws_train, c_lsqr
            g_argnums=(0,1,2,4,7,5,8),
            h_argnums=(5,1,3,6),
            g_val_h_argnum=3
        )

        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=False)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        b_ratio_opt_given_B_fxn = fxns.cvx_opt(# xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver, 
            objective=logreg_ratio_objective, 
            dobjective_dx=dlogreg_ratio_objective_db
        )

        b_ratio_reg_opt_fxn = fxns.two_step( # xs_train, xs_test, ys_train, sigma, B, c_logreg, c_lsqr, scale_sigma, max_ratio
            g=b_ratio_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
            h=b_ratio_opt_to_b_reg_opt, # b_ratio_opt, xs_train, xs_test, ys_train, sigma, B, c_lsqr, scale_sigma, max_ratio
            g_argnums=(0,1,3,4,5),
            h_argnums=(0,1,2,3,4,6,7,8),
            g_val_h_argnum=0
        )

        x_dim = xs_train.shape[1]
        B = np.eye(x_dim)

        b_fit = b_ratio_reg_opt_fxn.val(xs_train, xs_test, ys_train, sigma, B, c_logreg, c_lsqr, scale_sigma, max_ratio)

        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, b_fit))

        if not plot_b_info is None:
            logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                                   g=b_ratio_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                                   h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                                   g_argnums=(0,1,2,3,4),
                                                   h_argnums=(0,1,2,5,3,6),
                                                   g_val_h_argnum=0,
                                                   )
            ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio)
            true_b = np.zeros(len(b_fit))
            true_b[0] = 1.
            plot_b_info(xs_train, xs_test, ys_train, ys_test, true_b, ws_train, predictor)
#            plot_b_info(xs_train, xs_test, ys_train, ys_test, b_fit / np.linalg.norm(b_fit), ws_train, predictor)

        return predictor

    return fitter



def logreg_ratio_UB_fitter(c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f_getter, unconstrained, c_lsqr_loss, c_lsqr_loss_eval, scale_sigma=True, plot_b_info=None, max_ratio=5., ws_full_train_f = lambda xs_train, xs_test: np.ones(len(xs_train)), num_tries=1, scipy_minimize_options={}, scipy_minimize_info_f=lambda x: None, scipy_minimize_verbose=1, pymanopt_options={'verbose':2, 'maxiter':100}):

#def logreg_ratio_UB_fitter(c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):
    if not unconstrained:
        scale_sigma = False

    if UB_reg == 0.:
            pass
            #assert c_lsqr_loss == c_lsqr_loss_eval == 0.

    @basic.do_cprofile()
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        weighted_lsqr_loss_loss_fxn = fxns.two_step(# B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr_loss, c_lsqr_loss_eval
            g=fxns.fxn.autograd_fxn(_val=fxns.b_opt_to_squared_losses), # B, xs_train, ys_train, b_lsqr
            h=fxns.two_step( # B, xs_train, ls_train, ws_train, c_lsqr_loss, c_lsqr_loss_eval
                g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ls_train, ws_train, c_lsqr_loss
                h=fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt), # B, xs_train, ls_train, ws_train, b_lsqr_loss, c_lsqr_loss_eval
                g_argnums=(0,1,2,3,4),
                h_argnums=(0,1,2,3,5),
                g_val_h_argnum=4,
            ),
            g_argnums=(0,1,2,4),
            h_argnums=(0,1,3,5,6),
            g_val_h_argnum=2
        )

        upper_bound_fxn = fxns.product( # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
            g=fxns.fxn.autograd_fxn(_val=fxns.expected_conditional_PE_dist), # ws_full_train, ws_train
            h=weighted_lsqr_loss_loss_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr_loss, c_lsqr_loss_eval
            g_argnums=(5,3),
            h_argnums=(0,1,2,3,4,6,7)
        )

        objective_given_ws_and_b_opt = fxns.sum(# B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr
            fs=[
                fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt), # B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr
                upper_bound_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg), # ws_train
               ],
            fs_argnums=[
                (0,1,2,3,4,8),
                (0,1,2,3,4,5,6,7),
                (3,)
            ],
            weights=[
                1.,
                UB_reg,
                weight_reg,
            ],
        )

        objective_given_ws = fxns.two_step(# B, xs_train, ys_train, ws_train, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr
            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ys_train, ws_train, c_lsqr
            h=objective_given_ws_and_b_opt, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr
            g_argnums=(0,1,2,3,7),
            h_argnums=(0,1,2,3,4,5,6,7),
            g_val_h_argnum=4,
        )

        objective_given_b_logreg = fxns.two_step(# b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, scale_sigma, max_ratio
            g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper),# b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=objective_given_ws,# B, xs_train, ys_train, ws_train, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr
            g_argnums=(0,1,2,5,10,6,11),
            h_argnums=(6,1,3,4,7,8,9),
            g_val_h_argnum=3,
        )

        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        b_opt_given_B_fxn = fxns.cvx_opt( # xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver,
            objective=logreg_ratio_objective,
            dobjective_dx=dlogreg_ratio_objective_db
            )

        final_fxn = fxns.g_thru_f_opt(# ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio
            g=b_opt_given_B_fxn,# xs_train, xs_test, sigma, B, c_logreg
            h=objective_given_b_logreg, # b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, scale_sigma, max_ratio
            g_argnums=(1,2,4,5,9),
            h_argnums=(1,2,0,3,4,5,6,7,8,10,11),
            g_val_h_argnum=0
        )

        ws_full_train = ws_full_train_f(xs_train, xs_test)

        objective = lambda B: final_fxn.val(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio)
        dobjective_dB = lambda B: final_fxn.grad(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, care_argnums=(5,))

        if unconstrained:
            horse_optimizer = optimizers.scipy_minimize_optimizer(verbose=scipy_minimize_verbose, info_f=scipy_minimize_info_f, options=scipy_minimize_options)
        else:
            horse_optimizer = optimizers.pymanopt_optimizer(**pymanopt_options)
        
        optimizer = optimizers.multiple_optimizer(horse_optimizer, num_tries=num_tries, num_args=1)

        B_init_f = B_init_f_getter(xs_train, ys_train, xs_test)

        B_fit = optimizer.optimize(objective, dobjective_dB, B_init_f)
        
        logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                               g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                               h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                               g_argnums=(0,1,2,3,4),
                                               h_argnums=(0,1,2,5,3,6),
                                               g_val_h_argnum=0,
                                               )
        ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B_fit, c_logreg, scale_sigma, max_ratio)

        if unconstrained:
            b_predictor = B_fit
        else:
            b_fit = fxns.weighted_lsqr_b_opt(B_fit, xs_train, ys_train, ws_train, c_lsqr)
            b_predictor = np.dot(B_fit, b_fit)
            
        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, b_predictor))

        if not plot_b_info is None:
            plot_b_info(xs_train, xs_test, ys_train, ys_test, b_predictor / np.linalg.norm(b_predictor), ws_train, predictor)

        return predictor

    return fitter







"""
below is old stuff, do not use
"""


def logreg_ratio_fitter(c_lsqr, c_logreg, weight_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):

    return UB_fitter(c_lsqr, c_logreg, weight_reg, 0., sigma, B_init_f, False, 0., 0., scale_sigma=False, plot_b_info=plot_b_info, max_ratio=max_ratio, num_tries=num_tries, pymanopt_options=pymanopt_options)

"""
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        scale_sigma = False
        
        weighted_lsqr_loss_fxn = fxns.two_step( # B, xs_train, xs_test, ys_train, sigma, b_logreg, c_lsqr, scale_sigma, max_ratio
            g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=fxns.sum( # B, xs_train, ys_train, c_lsqr, ws_train
                fs=[
                    fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_loss), # B, xs_train, ys_train, ws_train, c_lsqr
                    fxns.fxn.autograd_fxn(_val=fxns.weight_reg), # ws_train
                    ],
                fs_argnums=[
                    (0,1,2,4,3),
                    (4,)
                    ],
                weights=[
                    1.,
                    tradeoff,
                    ],
                ),
            g_argnums=(5,1,2,4,7,0,8),
            h_argnums=(0,1,3,6),
            g_val_h_argnum=4,
        )

        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        b_opt_given_B_fxn = fxns.cvx_opt( # xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver,
            objective=logreg_ratio_objective,
            dobjective_dx=dlogreg_ratio_objective_db
            )

        weighted_lsqr_loss_thru_ws_opt_fxn = fxns.g_thru_f_opt( # ys_train, xs_train, xs_test, sigma, B, c_logreg, c_lsqr, scale_sigma, max_ratio
            g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
            h=weighted_lsqr_loss_fxn, # B, xs_train, xs_test, ys_train, sigma, b_logreg, c_lsqr, scale_sigma, max_ratio
            g_argnums=(1,2,3,4,5), 
            h_argnums=(4,1,2,0,3,6,7,8), 
            g_val_h_argnum=5
            )

        objective = lambda B: weighted_lsqr_loss_thru_ws_opt_fxn.val(ys_train, xs_train, xs_test, sigma, B, c_logreg, c_lsqr, scale_sigma, max_ratio)
        dobjective_dB = lambda B: weighted_lsqr_loss_thru_ws_opt_fxn.grad(ys_train, xs_train, xs_test, sigma, B, c_logreg, c_lsqr, scale_sigma, max_ratio, care_argnums=(4,))

        optimizer = optimizers.multiple_optimizer(optimizers.pymanopt_optimizer(**pymanopt_options), num_tries=num_tries, num_args=1)

        B_fit = optimizer.optimize(objective, dobjective_dB, B_init_f)
        
        logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                               g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                               h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                               g_argnums=(0,1,2,3,4),
                                               h_argnums=(0,1,2,5,3,6),
                                               g_val_h_argnum=0,
                                               )
        ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B_fit, c_logreg, scale_sigma, max_ratio)

        b_fit = fxns.weighted_lsqr_b_opt(B_fit, xs_train, ys_train, ws_train, c_lsqr)

        predictor_b_fit = np.dot(B_fit, b_fit)
        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, predictor_b_fit))

        if not plot_b_info is None:
            plot_b_info(xs_train, xs_test, ys_train, ys_test, predictor_b_fit / np.linalg.norm(predictor_b_fit), ws_train, predictor)

        return predictor

    return fitter
"""

def logreg_ratio_1d_unconstrained_fitter(c_B, c_logreg, weight_reg, sigma, B_init_f, plot_b_info=None, scale_sigma=True, max_ratio=5., num_tries=1, scipy_minimize_options={}, scipy_minimize_info_f=lambda x: None, scipy_minimize_verbose=False):
    """
    returns fxn that makes predictions
    """
    return UB_fitter(c_B, c_logreg, weight_reg, 0., sigma, B_init_f, True, 0., 0., scale_sigma=scale_sigma, plot_b_info=plot_b_info, max_ratio=max_ratio, num_tries=num_tries, scipy_minimize_options=scipy_minimize_options, scipy_minimize_info_f=scipy_minimize_info_f, scipy_minimize_verbose=scipy_minimize_verbose)

"""
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        weighted_lsqr_loss_fxn = fxns.two_step( # B, xs_train, xs_test, ys_train, sigma, b_logreg, c_B, scale_sigma, max_ratio
            g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=fxns.sum( # B, xs_train, ys_train, c_B, ws_train
                fs=[
                    fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_B), # B, xs_train, ys_train, c_B, ws_train
                    fxns.fxn.autograd_fxn(_val=fxns.weight_reg), # ws_train
                    ],
                fs_argnums=[
                    (0,1,2,3,4),
                    (4,)
                    ],
                weights=[
                    1.,
                    tradeoff,
                    ],
                ),
            g_argnums=(5,1,2,4,7,0,8),
            h_argnums=(0,1,3,6),
            g_val_h_argnum=4,
        )

        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        b_opt_given_B_fxn = fxns.cvx_opt( # xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver,
            objective=logreg_ratio_objective,
            dobjective_dx=dlogreg_ratio_objective_db
        )

        weighted_lsqr_loss_thru_ws_opt_fxn = fxns.g_thru_f_opt( # ys_train, xs_train, xs_test, sigma, B, c_logreg, c_B, scale_sigma, max_ratio
            g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
            h=weighted_lsqr_loss_fxn, # B, xs_train, xs_test, ys_train, sigma, b_logreg, c_B, scale_sigma, max_ratio
            g_argnums=(1,2,3,4,5), 
            h_argnums=(4,1,2,0,3,6,7,8), 
            g_val_h_argnum=5
            )

        objective = lambda B: weighted_lsqr_loss_thru_ws_opt_fxn.val(ys_train, xs_train, xs_test, sigma, B, c_logreg, c_B, scale_sigma, max_ratio)
        dobjective_dB = lambda B: weighted_lsqr_loss_thru_ws_opt_fxn.grad(ys_train, xs_train, xs_test, sigma, B, c_logreg, c_B, scale_sigma, max_ratio, care_argnums=(4,))

        optimizer = optimizers.multiple_optimizer(optimizers.scipy_minimize_optimizer(verbose=5, info_f=info_f, options=options), num_tries=num_tries, num_args=1)

        x_dim = xs_train.shape[1]
        #_B_init_f = (lambda: np.random.normal(size=x_dim)) if B_init_f is None else B_init_f
        B_fit = optimizer.optimize(objective, dobjective_dB, _B_init_f)

        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, B_fit))

        if not (plot_b_info is None):
            logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                                   g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                                   h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                                   g_argnums=(0,1,2,3,4),
                                                   h_argnums=(0,1,2,5,3,6),
                                                   g_val_h_argnum=0,
                                                   )
            ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B_fit, c_logreg, scale_sigma, max_ratio)
            plot_b_info(xs_train, xs_test, ys_train, ys_test, B_fit / np.linalg.norm(B_fit), ws_train, predictor)
        
        return predictor

    return fitter
"""
