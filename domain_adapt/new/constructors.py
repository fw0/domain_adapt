import fxns, optimizers, autograd, autograd.numpy as np, scipy
import pdb
import matplotlib.pyplot as plt
import python_utils.python_utils.basic as basic
import python_utils.python_utils.caching as caching
import domain_adapt.domain_adapt.SDR_data as data
import utils
import itertools


def simple_plot_b_info(xs_train, xs_test, ys_train, ys_test, ratios, predictor):

    s = 1.75
    alpha = 0.75
    train_color = 'red'
    test_color = 'blue'
    ratio_color = 'black'
    
    fig, ax = plt.subplots()

    predicted_ys_train = map(predictor, xs_train)
    predicted_ys_test = map(predictor, xs_test)

    ax.scatter(predicted_ys_train, ys_train, alpha=alpha, s=s, color=train_color, label='train')
    try:
        ax.scatter(predicted_ys_test, ys_test, alpha=alpha, s=s, color=test_color, label='test')
    except:
        pdb.set_trace()
    ax.legend()
    ax.set_ylabel('true y')
    ax.set_xlabel('predicted y')
    low_predicted = np.min((np.min(predicted_ys_train),np.min(predicted_ys_test)))
    high_predicted = np.max((np.max(predicted_ys_train),np.max(predicted_ys_test)))
    low_actual = np.min((np.min(ys_train),np.min(ys_test)))
    high_actual = np.max((np.max(ys_train),np.max(ys_test)))
    low = low_predicted
    high = high_predicted
#    low = np.min((low_predicted,low_actual))
#    high = np.max((high_actual,high_predicted))
    ax.plot((low,high), (low,high), ls="--", c=".3")
    ratio_ax = ax.twinx()
    ratio_ax.scatter(map(predictor, xs_train), ratios, alpha=alpha+0.25, s=s+7, marker='x', color=ratio_color)

#    fig.suptitle(str(predictor))


    
    #print predictor
    
    basic.display_fig_inline(fig)
    error = ys_test-np.array(map(predictor, xs_test))
    print np.mean(error*error), 'squared error'
    print np.mean(np.abs(predicted_ys_test-ys_test)), 'abs error'
    pdb.set_trace()

def plot_b_info(xs_train, xs_test, ys_train, ys_test, b, ratios, predictor):

    s = 1.75
    alpha = 0.75
    fig, ax = plt.subplots()
    data.data.plot_us(b, xs_train, xs_test, ax, normed=True)
    
    ratio_ax = ax.twinx()
    ratio_ax.scatter(np.dot(xs_train, b), ratios, alpha=alpha, s=s, label='estimated ratios')
    ratio_ax.set_ylabel('estimated ratios')
    effective_sample_size = (len(xs_train)**2) / (np.linalg.norm(ratios)**2)
    ratio_ax.set_title('N_eff: %.2f / %d' % (effective_sample_size, len(xs_train)))
    ratio_ax.legend()
    caching.fig_archiver.archive_fig(fig)
    #basic.display_fig_inline(fig)

    ys_train_hat = np.array([predictor(x) for x in xs_train])
    train_error = ys_train - ys_train_hat
    train_loss = np.dot(train_error,train_error) / len(train_error)
    
    if not (ys_test is None):
        ys_test_hat = np.array([predictor(x) for x in xs_test])
        test_error = ys_test - ys_test_hat
        test_loss = np.dot(test_error,test_error) / len(test_error)
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



def logreg_ratio_UB_fitter(c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f_getter, unconstrained, c_lsqr_loss, c_lsqr_loss_eval, scale_sigma=True, plot_b_info=None, max_ratio=5., ws_full_train_f = lambda xs_train, xs_test: np.ones(len(xs_train)), num_tries=1, scipy_minimize_method=None, scipy_minimize_options={}, scipy_minimize_info_f=lambda x: None, scipy_minimize_verbose=1, pymanopt_options={'verbose':2, 'maxiter':100}):

#def logreg_ratio_UB_fitter(c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):
    if not unconstrained:
        scale_sigma = False

    if UB_reg == 0.:
        pass
            #assert c_lsqr_loss == c_lsqr_loss_eval == 0.

    @basic.do_cprofile('cumtime')
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
            dobjective_dx=dlogreg_ratio_objective_db,
#            optimizer = scipy.optimize.minimize(options={'maxiter':1})
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


        
        if True:
            for i in xrange(10):
                #            objective(B_init_f_getter(xs_train, ys_train, xs_test)())
                basic.timeit(dobjective_dB)(B_init_f_getter(xs_train, ys_train, xs_test)())
            assert False


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


def no_ws_logreg_ratio_UB_fitter(c_lsqr, c_logreg, weight_reg, sigma, B_init_f_getter, unconstrained, add_reg=False, UB_reg=0., c_lsqr_loss=0., c_lsqr_loss_eval=0., scale_sigma=True, plot_b_info=None, max_ratio=5., ws_full_train_f = lambda xs_train, xs_test: np.ones(len(xs_train)), num_tries=1, unconstrained_scipy_minimize_method=None, unconstrained_scipy_minimize_options={}, unconstrained_scipy_minimize_info_f=lambda x: None, unconstrained_scipy_minimize_verbose=1, cvx_opt_warm_start=True, cvx_opt_scipy_minimize_method=None, cvx_opt_scipy_minimize_options={}, cvx_opt_scipy_minimize_info_f=lambda x: None, cvx_opt_scipy_minimize_verbose=1, pymanopt_options={'verbose':2, 'maxiter':100}):

#def logreg_ratio_UB_fitter(c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):
    if not unconstrained:
        scale_sigma = False

    if UB_reg == 0.:
        pass
            #assert c_lsqr_loss == c_lsqr_loss_eval == 0.

#    @basic.do_cprofile('cumtime')
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

#        weighted_lsqr_loss_loss_fxn = fxns.two_step(# B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr_loss, c_lsqr_loss_eval
#            g=fxns.fxn.autograd_fxn(_val=fxns.b_opt_to_squared_losses), # B, xs_train, ys_train, b_lsqr
#            h=fxns.two_step( # B, xs_train, ls_train, ws_train, c_lsqr_loss, c_lsqr_loss_eval
#                g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ls_train, ws_train, c_lsqr_loss
#                h=fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt), # B, xs_train, ls_train, ws_train, b_lsqr_loss, c_lsqr_loss_eval
#                g_argnums=(0,1,2,3,4),
#                h_argnums=(0,1,2,3,5),
#                g_val_h_argnum=4,
#            ),
#            g_argnums=(0,1,2,4),
#            h_argnums=(0,1,3,5,6),
#            g_val_h_argnum=2
#        )

#        upper_bound_fxn = fxns.product( # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
#            g=fxns.fxn.autograd_fxn(_val=fxns.expected_conditional_PE_dist), # ws_full_train, ws_train
#            h=weighted_lsqr_loss_loss_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr_loss, c_lsqr_loss_eval
#            g_argnums=(5,3),
#            h_argnums=(0,1,2,3,4,6,7)
#        )

        objective_given_b_logreg_and_b_opt = fxns.sum(# B, xs_train, xs_test, ys_train, b_logreg, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, sigma, scale_sigma, max_ratio, add_reg
            fs=[
                fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt_and_b_logreg), # B, xs_train, xs_test, ys_train, b_logreg, b_lsqr, c_lsqr, sigma, scale_sigma, max_ratio, add_reg
#                upper_bound_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg_given_b_logreg), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
               ],
            fs_argnums=[
                (0,1,2,3,4,5,9,10,11,12,13),
#                (0,1,2,3,4,5,6,7),
                (4,1,2,10,11,0,12)
            ],
            weights=[
                1.,
#                UB_reg,
                weight_reg,
            ],
        )

#        objective_given_ws = fxns.two_step(# B, xs_train, ys_train, ws_train, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr
#            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ys_train, ws_train, c_lsqr
#            h=objective_given_ws_and_b_opt, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr
#            g_argnums=(0,1,2,3,7),
#            h_argnums=(0,1,2,3,4,5,6,7),
#            g_val_h_argnum=4,
#        )

        objective_given_b_logreg = fxns.two_step(# b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, scale_sigma, max_ratio, add_reg
            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt_given_b_logreg), #B, xs_train, xs_test, ys_train, b_logreg, c_lsqr, sigma, scale_sigma, max_ratio
#g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper),# b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=objective_given_b_logreg_and_b_opt,# B, xs_train, xs_test, ys_train, b_logreg, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, sigma, scale_sigma, max_ratio, add_reg
            g_argnums=(6,1,2,3,0,9,5,10,11),
            h_argnums=(6,1,2,3,0,4,7,8,9,5,10,11,12),
            g_val_h_argnum=5,
        )

        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.finite_difference_hessian_vector_product)
        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        
        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        #dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        b_opt_given_B_fxn = fxns.cvx_opt( # xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver,
            objective=logreg_ratio_objective,
            dobjective_dx=dlogreg_ratio_objective_db,
            optimizer=optimizers.scipy_minimize_optimizer(method=cvx_opt_scipy_minimize_method, verbose=cvx_opt_scipy_minimize_verbose, info_f=cvx_opt_scipy_minimize_info_f, options=cvx_opt_scipy_minimize_options),
            warm_start=cvx_opt_warm_start
#            optimizer = scipy.optimize.minimize(options={'maxiter':100}) FIX
            )

        final_fxn = fxns.g_thru_f_opt_new(# ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg
            g=b_opt_given_B_fxn,# xs_train, xs_test, sigma, B, c_logreg
            h=objective_given_b_logreg, # b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, scale_sigma, max_ratio, add_reg
            g_argnums=(1,2,4,5,9),
            h_argnums=(1,2,0,3,4,5,6,7,8,10,11,12),
            g_val_h_argnum=0
        )

        ws_full_train = ws_full_train_f(xs_train, xs_test)

        objective = lambda B: final_fxn.val(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg)
        dobjective_dB = lambda B: final_fxn.grad(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg, care_argnums=(5,))


        
        if False:
            for i in xrange(10):
                print i, 'step'
                #            objective(B_init_f_getter(xs_train, ys_train, xs_test)())
                basic.timeit('test grad')(dobjective_dB)(B_init_f_getter(xs_train, ys_train, xs_test)())
            assert False


        if unconstrained:
            horse_optimizer = optimizers.scipy_minimize_optimizer(method=unconstrained_scipy_minimize_method, verbose=unconstrained_scipy_minimize_verbose, info_f=unconstrained_scipy_minimize_info_f, options=unconstrained_scipy_minimize_options)
        else:
            print pymanopt_options
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


def no_c_lsqr_no_ws_logreg_ratio_UB_fitter(c_logreg, weight_reg, sigma, B_init_f_getter, unconstrained, add_reg=False, UB_reg=0., c_lsqr_loss=0., c_lsqr_loss_eval=0., scale_sigma=True, plot_b_info=None, max_ratio=5., ws_full_train_f = lambda xs_train, xs_test: np.ones(len(xs_train)), num_tries=1, unconstrained_scipy_minimize_method=None, unconstrained_scipy_minimize_options={}, unconstrained_scipy_minimize_info_f=lambda x: None, unconstrained_scipy_minimize_verbose=1, cvx_opt_warm_start=True, cvx_opt_scipy_minimize_method=None, cvx_opt_scipy_minimize_options={}, cvx_opt_scipy_minimize_info_f=lambda x: None, cvx_opt_scipy_minimize_verbose=1, pymanopt_options={'verbose':2, 'maxiter':100}, linesearch_method='brent', linesearch_options={}, linesearch_init_window_width=100):

#def logreg_ratio_UB_fitter(c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):
    if not unconstrained:
        scale_sigma = False

    if UB_reg == 0.:
        pass


#    @basic.do_cprofile('cumtime')
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        # every: will be given ws_train and b_opt if kmm
        objective_given_b_logreg_and_b_opt = fxns.sum(# B, xs_train, xs_test, ys_train, b_logreg, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, sigma, scale_sigma, max_ratio, add_reg
            fs=[
                fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt_and_b_logreg), # B, xs_train, xs_test, ys_train, b_logreg, b_lsqr, c_lsqr, sigma, scale_sigma, max_ratio, add_reg
#                upper_bound_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg_given_b_logreg), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
               ],
            fs_argnums=[
                (0,1,2,3,4,5,9,10,11,12,13),
#                (0,1,2,3,4,5,6,7),
                (4,1,2,10,11,0,12)
            ],
            weights=[
                1.,
#                UB_reg,
                weight_reg,
            ],
        )

        # every: g will give optimal b for regression, and this will be a g_opt_thru_f fxn, with g being a cvx_opt node
        objective_given_b_logreg = fxns.two_step(# b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, scale_sigma, max_ratio, add_reg
            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt_given_b_logreg), #B, xs_train, xs_test, ys_train, b_logreg, c_lsqr, sigma, scale_sigma, max_ratio
#g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper),# b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=objective_given_b_logreg_and_b_opt,# B, xs_train, xs_test, ys_train, b_logreg, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, sigma, scale_sigma, max_ratio, add_reg
            g_argnums=(6,1,2,3,0,9,5,10,11),
            h_argnums=(6,1,2,3,0,4,7,8,9,5,10,11,12),
            g_val_h_argnum=5,
        )

        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.finite_difference_hessian_vector_product)
        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        
        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        #dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        b_opt_given_B_fxn = fxns.cvx_opt( # xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver,
            objective=logreg_ratio_objective,
            dobjective_dx=dlogreg_ratio_objective_db,
            optimizer=optimizers.scipy_minimize_optimizer(method=cvx_opt_scipy_minimize_method, verbose=cvx_opt_scipy_minimize_verbose, info_f=cvx_opt_scipy_minimize_info_f, options=cvx_opt_scipy_minimize_options),
            warm_start=cvx_opt_warm_start
#            optimizer = scipy.optimize.minimize(options={'maxiter':100}) FIX
            )

        final_fxn = fxns.g_thru_f_opt_new(# ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg
            g=b_opt_given_B_fxn,# xs_train, xs_test, sigma, B, c_logreg
            h=objective_given_b_logreg, # b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, scale_sigma, max_ratio, add_reg
            g_argnums=(1,2,4,5,9),
            h_argnums=(1,2,0,3,4,5,6,7,8,10,11,12),
            g_val_h_argnum=0
        )

        ws_full_train = ws_full_train_f(xs_train, xs_test)

        def objective(sigma, c_lsqr, B):
            return final_fxn(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg)

        # create B optimizer, which takes in c_lsqr and B

        def B_optimizer_get_objective(sigma, c_lsqr, B):
            objective = lambda B: final_fxn.val(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg)
            return objective

        def B_optimizer_get_dobjective_dB(sigma, c_lsqr, B):
            dobjective_dB = lambda B: final_fxn.grad(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, scale_sigma, max_ratio, add_reg, care_argnums=(5,))
            return dobjective_dB

        def B_optimizer_get_B(sigma, c_lsqr, B):
            return B

        def B_optimizer_out_f((sigma, c_lsqr, B), (objective, dobjective_dB, _B), new_B):
            return (sigma, c_lsqr, new_B)

        if unconstrained:
            B_horse_optimizer = optimizers.scipy_minimize_optimizer(method=unconstrained_scipy_minimize_method, verbose=unconstrained_scipy_minimize_verbose, info_f=unconstrained_scipy_minimize_info_f, options=unconstrained_scipy_minimize_options)
        else:
            B_horse_optimizer = optimizers.pymanopt_optimizer(**pymanopt_options)

        B_optimizer = optimizers.get_stuff_optimizer(B_horse_optimizer, (B_optimizer_get_objective, B_optimizer_get_dobjective_dB, B_optimizer_get_B), B_optimizer_out_f) # takes in c_lsqr and B, outputs new c_lsqr and B


        # create sigma optimizer (more like updater), takes in sigma, c_lsqr, B
        sigma_sample = 20
        xs_train_sigma = xs_train[0:sigma_sample]
        xs_test_sigma = xs_test[0:sigma_sample]
        xs_sigma = np.concatenate((xs_train_sigma, xs_test_sigma), axis=0)

        def update_sigma(sigma, c_lsqr, B):
            new_sigma = utils.median_distance(np.dot(xs_sigma,B), np.dot(xs_sigma, B))
            return new_sigma, c_lsqr, B

        # create c_lsqr optimizer, which takes in c_lsqr and B.  

        # first create objective 

        # every: this was necessary bc previous fxns never explicitly outputted ws_train.  for kmm this fxn will be unnecessary, just uses ws_train_given_B
        B_to_ws_train = fxns.two_step( # xs_train, xs_test, sigma, scale_sigma, B, max_ratio, c_logreg
            g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
            h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            g_argnums=(0,1,2,4,6),
            h_argnums=(0,1,2,3,4,5),
            g_val_h_argnum=0
        )

        ws_test = np.ones(shape=len(xs_test))

        # every: probably necessary since 1. objective does not involve ws penalty, and 2. penalty is on test losses, so have to reassemble the g and h here
        ws_train_to_test_loss = fxns.two_step( # B, xs_train, ys_train, ws_train, c_lsqr, xs_test, ys_test, ws_test, add_reg
            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ys_train, ws_train, c_lsqr
            h=fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt), # B, xs_test, ys_test, ws_test, b_opt, c_lsqr, add_reg
            g_argnums=(0,1,2,3,4),
            h_argnums=(0,5,6,7,4,8),
            g_val_h_argnum=4
        )

        # every: necessary since g,h have to be assembled together differently than before, due to loss being on test
        B_to_test_loss = fxns.two_step( # B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, scale_sigma, max_ratio, c_logreg
            g=B_to_ws_train, # xs_train, xs_test, sigma, scale_sigma, B, max_ratio, c_logreg
            h=ws_train_to_test_loss, # B, xs_train, ys_train, ws_train, c_lsqr, xs_test, ys_test, ws_test, add_reg
            g_argnums=(1,4,8,9,0,10,11),
            h_argnums=(0,1,2,3,4,5,6,7),
            g_val_h_argnum=3
        )


        use_ys_test = False
        num_folds = 2

        def c_lsqr_optimizer_get_objective(sigma, c_lsqr, B):
            if use_ys_test:
                objective = lambda c_lsqr: B_to_test_loss(B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, scale_sigma, max_ratio, c_logreg)
                return objective
            else:
                def cv_test_loss(c_lsqr):
                    ws_train = B_to_ws_train(xs_train, xs_test, sigma, scale_sigma, B, max_ratio, c_logreg)
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=num_folds)
                    test_losses = 0.
                    for (train_idx, test_idx) in kf.split(xs_train):
                        xs_train_train = xs_train[train_idx]
                        xs_train_test = xs_train[test_idx]
                        ys_train_train = ys_train[train_idx]
                        ys_train_test = ys_train[test_idx]
                        ws_train_train = ws_train[train_idx]
                        ws_train_test = ws_train[test_idx]
                        fold_test_loss = ws_train_to_test_loss(B, xs_train_train, ys_train_train, ws_train_train, c_lsqr, xs_train_test, ys_train_test, ws_train_test, add_reg)
                        test_losses += fold_test_loss * len(train_idx)
                    return test_losses / len(xs_train)
                return cv_test_loss
            
        def c_lsqr_optimizer_get_c_lsqr(sigma, c_lsqr, B):
            return c_lsqr

        def c_lsqr_optimizer_out_f((sigma, c_lsqr, B), (objective, _c_lsqr), new_c_lsqr):
            return (sigma, new_c_lsqr, B)

        
        c_lsqr_horse_optimizer = optimizers.scalar_fxn_optimizer(method=linesearch_method, options=linesearch_options, init_window_width=linesearch_init_window_width)
        c_lsqr_optimizer = optimizers.get_stuff_optimizer(c_lsqr_horse_optimizer, (c_lsqr_optimizer_get_objective, c_lsqr_optimizer_get_c_lsqr), c_lsqr_optimizer_out_f) # takes in c_lsqr and B, outputs new c_lsqr and B

        # create optimizer and optimize
#        horse_optimizer = optimizers.many_optimizer((update_sigma, B_optimizer, c_lsqr_optimizer))
        horse_optimizer = optimizers.many_optimizer((c_lsqr_optimizer, B_optimizer))
#        horse_optimizer = optimizers.many_optimizer((B_optimizer,))
        
        optimizer = optimizers.multiple_optimizer(horse_optimizer, num_tries=num_tries, num_args=3)

        sigma_init_f = lambda: 1.

        B_init_f = B_init_f_getter(xs_train, ys_train, xs_test)

        c_lsqr_init_f = lambda: 1.

#        B_c_lsqr_init_f = lambda: B_init_f(), c_lsqr_init_f()

        sigma_fit, c_lsqr_fit, B_fit = optimizer.optimize(objective, sigma_init_f, c_lsqr_init_f, B_init_f)
        
        # create predictor

#        """
#        # fix: this fxn is same as B_to_ws_train
#        logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
#                                               g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
#                                               h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
#                                               g_argnums=(0,1,2,3,4),
#                                               h_argnums=(0,1,2,5,3,6),
#                                               g_val_h_argnum=0,
#                                               )
#        """
#        ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B_fit, c_logreg, scale_sigma, max_ratio)
        ws_train = B_to_ws_train(xs_train, xs_test, sigma_fit, scale_sigma, B_fit, max_ratio, c_logreg)

        if unconstrained:
            b_predictor = B_fit
        else:
            b_fit = fxns.weighted_lsqr_b_opt(B_fit, xs_train, ys_train, ws_train, c_lsqr_fit)
            b_predictor = np.dot(B_fit, b_fit)
            
        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, b_predictor))

        if not plot_b_info is None:
            plot_b_info(xs_train, xs_test, ys_train, ys_test, b_predictor / np.linalg.norm(b_predictor), ws_train, predictor)

        return predictor

    return fitter


def no_c_lsqr_kmm_ratio_UB_fitter(w_max, eps, weight_reg, B_init_f_getter, unconstrained, add_reg=False, UB_reg=0., c_lsqr_loss=0., c_lsqr_loss_eval=0.,plot_b_info=None, ws_full_train_f = lambda xs_train, xs_test: np.ones(len(xs_train)), num_tries=1, unconstrained_scipy_minimize_method=None, unconstrained_scipy_minimize_options={}, unconstrained_scipy_minimize_info_f=lambda x: None, unconstrained_scipy_minimize_verbose=1, quad_opt_warm_start=True, pymanopt_options={'verbose':2, 'maxiter':100}, linesearch_method='brent', linesearch_options={}, linesearch_init_window_width=100):

#def logreg_ratio_UB_fitter(c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):
    if not unconstrained:
        scale_sigma = False

    if UB_reg == 0.:
        pass


#    @basic.do_cprofile('cumtime')
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        # option
        weighted_loss_given_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt) # B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr, add_reg
        
        # every: will be given ws_train and b_opt if kmm
        objective_given_ws_and_b_opt = fxns.sum(# B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr, add_reg
            fs=[
                weighted_loss_given_b_opt, # B, xs_train, ys_train, ws_train, b_lsqr, c_lsqr, add_reg
                #upper_bound_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg), # ws_train
               ],
            fs_argnums=[
                (0,1,2,3,4,6,7),
#                (0,1,2,3,4,5,6,7),
                (3,)
            ],
            weights=[
                1.,
#                UB_reg,
                weight_reg,
            ],
        )

        # option
        weighted_loss_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt) # B, xs_train, ys_train, ws_train, c_lsqr)
        
        # every: g will give optimal b for regression, and this will be a g_opt_thru_f fxn, with g being a cvx_opt node
        objective_given_b_opt = fxns.two_step(# B, xs_train, ys_train, ws_train, ws_full_train, c_lsqr, add_reg
            g=weighted_loss_b_opt, # B, xs_train, ys_train, ws_train, c_lsqr)
#            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt_given_b_logreg), #B, xs_train, xs_test, ys_train, b_logreg, c_lsqr, sigma, scale_sigma, max_ratio
#g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper),# b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
            h=objective_given_ws_and_b_opt, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr, add_reg
            g_argnums=(0,1,2,3,5),
            h_argnums=(0,1,2,3,4,5,6),
            g_val_h_argnum=4,
        )

        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.finite_difference_hessian_vector_product)
        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
        kmm_objective = fxns.nystrom_kmm_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
#        kmm_objective = fxns.new_kmm_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
        dkmm_objective_dws = fxns.dopt_objective_dx.autograd_fxn(kmm_objective)
        
        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        #dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        @basic.timeit('gmres_solver')
        def gmres_solver(A,b):
            asdf = scipy.sparse.linalg.gmres(A, b, maxiter=10)
#            print asdf[1]
#            pdb.set_trace()
            return asdf[0]
        #lin_solver = cg_solver
        bicg_solver = lambda A,b: scipy.sparse.linalg.gmres(A, b)[0]
#        lin_solver = gmres_solver
        lin_solver = bicg_solver
        ws_given_B_fxn = fxns.full_quad_opt( # xs_train, xs_test, sigma, B, w_max, eps
            lin_solver=lin_solver,
            objective=kmm_objective,
            dobjective_dx=dkmm_objective_dws,
            ineq_constraints=fxns.Gh_ineq_constraints.autograd_fxn(fxns.kmm_get_Gh),
#            get_Gh=fxns.kmm_get_Gh,
#            optimizer=optimizers.scipy_minimize_optimizer(method=cvx_opt_scipy_minimize_method, verbose=cvx_opt_scipy_minimize_verbose, info_f=cvx_opt_scipy_minimize_info_f, options=cvx_opt_scipy_minimize_options),
            warm_start=quad_opt_warm_start
#            optimizer = scipy.optimize.minimize(options={'maxiter':100}) FIX
            )

        final_fxn = fxns.g_thru_f_opt_new(# B, xs_train, xs_test, ys_train, ws_full_train, c_lsqr, sigma, w_max, eps, add_reg
            g=ws_given_B_fxn,# xs_train, xs_test, sigma, B, w_max, eps
            h=objective_given_b_opt, # B, xs_train, ys_train, ws_train, ws_full_train, c_lsqr, add_reg
            g_argnums=(1,2,6,0,7,8),
            h_argnums=(0,1,3,4,5,9),
            g_val_h_argnum=3,
        )

        ws_full_train = ws_full_train_f(xs_train, xs_test)

        def objective(sigma, c_lsqr, B):
            return final_fxn(B, xs_train, xs_test, ys_train, ws_full_train, c_lsqr, sigma, w_max, eps, add_reg)

        # create B optimizer, which takes in c_lsqr and B

        def B_optimizer_get_objective(sigma, c_lsqr, B):
            objective = lambda B: final_fxn.val(B, xs_train, xs_test, ys_train, ws_full_train, c_lsqr, sigma, w_max, eps, add_reg)
            return objective

        def B_optimizer_get_dobjective_dB(sigma, c_lsqr, B):
            dobjective_dB = lambda B: final_fxn.grad(B, xs_train, xs_test, ys_train, ws_full_train, c_lsqr, sigma, w_max, eps, add_reg, care_argnums=(0,))
            return dobjective_dB

        def B_optimizer_get_B(sigma, c_lsqr, B):
            return B

        def B_optimizer_out_f((sigma, c_lsqr, B), (objective, dobjective_dB, _B), new_B):
            return (sigma, c_lsqr, new_B)

        if unconstrained:
            B_horse_optimizer = optimizers.scipy_minimize_optimizer(method=unconstrained_scipy_minimize_method, verbose=unconstrained_scipy_minimize_verbose, info_f=unconstrained_scipy_minimize_info_f, options=unconstrained_scipy_minimize_options)
        else:
            B_horse_optimizer = optimizers.pymanopt_optimizer(**pymanopt_options)

        B_optimizer = optimizers.get_stuff_optimizer(B_horse_optimizer, (B_optimizer_get_objective, B_optimizer_get_dobjective_dB, B_optimizer_get_B), B_optimizer_out_f) # takes in c_lsqr and B, outputs new c_lsqr and B


        # create sigma optimizer (more like updater), takes in sigma, c_lsqr, B
        sigma_sample = 20
        xs_train_sigma = xs_train[0:sigma_sample]
        xs_test_sigma = xs_test[0:sigma_sample]
        xs_sigma = np.concatenate((xs_train_sigma, xs_test_sigma), axis=0)

        def update_sigma(sigma, c_lsqr, B):
            new_sigma = utils.median_distance(np.dot(xs_sigma,B), np.dot(xs_sigma, B))
            return new_sigma, c_lsqr, B

        # create c_lsqr optimizer, which takes in c_lsqr and B.  

        # first create objective 

        # every: this was necessary bc previous fxns never explicitly outputted ws_train.  for kmm this fxn will be unnecessary, just uses ws_train_given_B
        #B_to_ws_train = fxns.two_step( # xs_train, xs_test, sigma, scale_sigma, B, max_ratio, c_logreg
        #    g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
        #    h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
        #    g_argnums=(0,1,2,4,6),
        #    h_argnums=(0,1,2,3,4,5),
        #    g_val_h_argnum=0
        #)

        ws_test = np.ones(shape=len(xs_test))

        # every: probably necessary since 1. objective does not involve ws penalty, and 2. penalty is on test losses, so have to reassemble the g and h here
        ws_train_to_test_loss = fxns.two_step( # B, xs_train, ys_train, ws_train, c_lsqr, xs_test, ys_test, ws_test, add_reg
            g=weighted_loss_b_opt, # B, xs_train, ys_train, ws_train, c_lsqr 
            h=weighted_loss_given_b_opt, # B, xs_test, ys_test, ws_test, b_opt, c_lsqr, add_reg
            g_argnums=(0,1,2,3,4),
            h_argnums=(0,5,6,7,4,8),
            g_val_h_argnum=4
        )

        # every: necessary since g,h have to be assembled together differently than before, due to loss being on test
        B_to_test_loss = fxns.two_step( # B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, w_max, eps
            g=ws_given_B_fxn, # xs_train, xs_test, sigma, B, w_max, eps
#            g=B_to_ws_train, # xs_train, xs_test, sigma, scale_sigma, B, max_ratio, c_logreg
            h=ws_train_to_test_loss, # B, xs_train, ys_train, ws_train, c_lsqr, xs_test, ys_test, ws_test, add_reg
            g_argnums=(1,4,8,0,9,10),
            h_argnums=(0,1,2,3,4,5,6,7),
            g_val_h_argnum=3
        )


        use_ys_test = False
        num_folds = 2

        def c_lsqr_optimizer_get_objective(sigma, c_lsqr, B):
            if use_ys_test:
                objective = lambda c_lsqr: B_to_test_loss(B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, w_max, eps)
                return objective
            else:
                def cv_test_loss(c_lsqr):
                    ws_train = ws_given_B_fxn(xs_train, xs_test, sigma, B, w_max, eps)
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=num_folds)
                    test_losses = 0.
                    for (train_idx, test_idx) in kf.split(xs_train):
                        xs_train_train = xs_train[train_idx]
                        xs_train_test = xs_train[test_idx]
                        ys_train_train = ys_train[train_idx]
                        ys_train_test = ys_train[test_idx]
                        ws_train_train = ws_train[train_idx]
                        ws_train_test = ws_train[test_idx]
                        fold_test_loss = ws_train_to_test_loss(B, xs_train_train, ys_train_train, ws_train_train, c_lsqr, xs_train_test, ys_train_test, ws_train_test, add_reg)
                        test_losses += fold_test_loss * len(train_idx)
                    return test_losses / len(xs_train)
                return cv_test_loss
            
        def c_lsqr_optimizer_get_c_lsqr(sigma, c_lsqr, B):
            return c_lsqr

        def c_lsqr_optimizer_out_f((sigma, c_lsqr, B), (objective, _c_lsqr), new_c_lsqr):
            return (sigma, new_c_lsqr, B)

        
        c_lsqr_horse_optimizer = optimizers.scalar_fxn_optimizer(method=linesearch_method, options=linesearch_options, init_window_width=linesearch_init_window_width)
        c_lsqr_optimizer = optimizers.get_stuff_optimizer(c_lsqr_horse_optimizer, (c_lsqr_optimizer_get_objective, c_lsqr_optimizer_get_c_lsqr), c_lsqr_optimizer_out_f) # takes in c_lsqr and B, outputs new c_lsqr and B

        # create optimizer and optimize
#        horse_optimizer = optimizers.many_optimizer((update_sigma, B_optimizer, c_lsqr_optimizer))
#        horse_optimizer = optimizers.many_optimizer((c_lsqr_optimizer, B_optimizer))
#        horse_optimizer = optimizers.many_optimizer((B_optimizer,))
        horse_optimizer = optimizers.many_optimizer((update_sigma, B_optimizer))
        
        optimizer = optimizers.multiple_optimizer(horse_optimizer, num_tries=num_tries, num_args=3)

        sigma_init_f = lambda: 1.

        B_init_f = B_init_f_getter(xs_train, ys_train, xs_test)

        c_lsqr_init_f = lambda: 1.

#        B_c_lsqr_init_f = lambda: B_init_f(), c_lsqr_init_f()

        basic.do_cprofile('cumtime')(final_fxn.grad_check)(B_init_f(), xs_train, xs_test, ys_train, ws_full_train, c_lsqr_init_f(), sigma_init_f(), w_max, eps, add_reg, care_argnums=(0,))

        pdb.set_trace()

        sigma_fit, c_lsqr_fit, B_fit = optimizer.optimize(objective, sigma_init_f, c_lsqr_init_f, B_init_f)
        
        # create predictor

        """
        # fix: this fxn is same as B_to_ws_train
        logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                               g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                               h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                               g_argnums=(0,1,2,3,4),
                                               h_argnums=(0,1,2,5,3,6),
                                               g_val_h_argnum=0,
                                               )
        """
#        ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B_fit, c_logreg, scale_sigma, max_ratio)
#        ws_train = B_to_ws_train(xs_train, xs_test, sigma, scale_sigma, B_fit, max_ratio, c_logreg)
        ws_train = ws_given_B_fxn(xs_train, xs_test, sigma, B_fit, w_max, eps)
        if unconstrained:
            b_predictor = B_fit
        else:
            b_fit = weighted_loss_b_opt(B_fit, xs_train, ys_train, ws_train, c_lsqr_fit)
#            b_fit = fxns.weighted_lsqr_b_opt(B_fit, xs_train, ys_train, ws_train, c_lsqr_fit)
            b_predictor = np.dot(B_fit, b_fit)
            
        predictor = fxns.fxn(_val = lambda xs: np.dot(xs, b_predictor))

        if not plot_b_info is None:
            plot_b_info(xs_train, xs_test, ys_train, ys_test, b_predictor / np.linalg.norm(b_predictor), ws_train, predictor)

        return predictor

    return fitter


def no_c_lsqr_lsif_ratio_UB_fitter(num_basis, c_lsif, max_ratio, weight_reg, B_init_f_getter, unconstrained, add_reg=False, UB_reg=0., c_lsqr_loss=0., c_lsqr_loss_eval=0.,plot_b_info=None, ws_full_train_f = lambda xs_train, xs_test: np.ones(len(xs_train)), num_tries=1, unconstrained_scipy_minimize_method=None, unconstrained_scipy_minimize_options={}, unconstrained_scipy_minimize_info_f=lambda x: None, unconstrained_scipy_minimize_verbose=1, quad_opt_warm_start=True, pymanopt_options={'verbose':2, 'maxiter':100}, linesearch_method='brent', linesearch_options={}, linesearch_init_window_width=100, cvx_opt_warm_start=False, cvx_opt_scipy_minimize_method=None, cvx_opt_scipy_minimize_options={}, cvx_opt_scipy_minimize_info_f=lambda x: None, cvx_opt_scipy_minimize_verbose=1, ):

#def logreg_ratio_UB_fitter(c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_logreg, weight_reg, UB_reg, sigma, B_init_f, max_ratio=5., num_tries=1, pymanopt_options={'verbose':2, 'maxiter':100}, plot_b_info=None):
    if not unconstrained:
        scale_sigma = False

    if UB_reg == 0.:
        pass


#    @basic.do_cprofile('cumtime')
    def fitter(xs_train, xs_test, ys_train, ys_test=None):

        add_reg_lsif = True
        
        # option
        weighted_loss_given_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt_and_lsif_alpha) # B, xs_train, xs_basis, ys_train, lsif_alpha, b_opt, c_lsqr, sigma, max_ratio, add_reg
        
        # every: will be given ws_train and b_opt if kmm
        objective_given_lsif_alpha_and_b_opt = fxns.sum(# B, xs_train, xs_basis, ys_train, lsif_alpha, b_opt, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, sigma, max_ratio, add_reg
            fs=[
                weighted_loss_given_b_opt, # B, xs_train, xs_basis, ys_train, lsif_alpha, b_opt, c_lsqr, sigma, max_ratio, add_reg
                #upper_bound_fxn, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval
                fxns.fxn.autograd_fxn(_val=fxns.weight_reg_given_lsif_alpha), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio
               ],
            fs_argnums=[
                (0,1,2,3,4,5,9,10,11,12),
#                (0,1,2,3,4,5,6,7),
                (4,1,2,10,0,11)
            ],
            weights=[
                1.,
#                UB_reg,
                weight_reg,
            ],
        )

        # option
        weighted_loss_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt_given_lsif_alpha) # B, xs_train, xs_basis, ys_train, lsif_alpha, c_lsqr, sigma, max_ratio
        weighted_loss_given_b_opt = fxns.fxn.autograd_fxn(_val=fxns.weighted_squared_loss_given_b_opt) # B, xs_test, ys_test, ws_test, b_opt, c_lsqr, add_reg
        
        # every: g will give optimal b for regression, and this will be a g_opt_thru_f fxn, with g being a cvx_opt node
        #objective_given_b_opt = fxns.two_step(# B, xs_train, ys_train, ws_train, ws_full_train, c_lsqr, add_reg
#            g=weighted_loss_b_opt, # B, xs_train, ys_train, ws_train, c_lsqr)
#            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt_given_b_logreg), #B, xs_train, xs_test, ys_train, b_logreg, c_lsqr, sigma, scale_sigma, max_ratio
#g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper),# b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
#            h=objective_given_ws_and_b_opt, # B, xs_train, ys_train, ws_train, b_lsqr, ws_full_train, c_lsqr, add_reg
#            g_argnums=(0,1,2,3,5),
#            h_argnums=(0,1,2,3,4,5,6),
#            g_val_h_argnum=4,
#        )

#        objective_given_lsif_alphas = fxns.two_step( # lsif_alpha, xs_train, xs_basis, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, max_ratio, add_reg
        objective_given_lsif_alphas = fxns.two_step.autograd_fxn( # lsif_alpha, xs_train, xs_basis, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, max_ratio, add_reg
            g=weighted_loss_b_opt, # B, xs_train, xs_basis, ys_train, lsif_alpha, c_lsqr, sigma, max_ratio
            h=objective_given_lsif_alpha_and_b_opt, # B, xs_train, xs_basis, ys_train, lsif_alpha, b_opt, ws_full_train, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, sigma, max_ratio, add_reg
            g_argnums=(6,1,2,3,0,9,5,10),
            h_argnums=(6,1,2,3,0,4,7,8,9,5,10,11),
            g_val_h_argnum=5,
        )

        lsif_least_squares = True

        lsif_objective = fxns.LSIF_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
        
        if not lsif_least_squares:
        
            #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.finite_difference_hessian_vector_product)
            #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
            #kmm_objective = fxns.nystrom_kmm_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)

    #        kmm_objective = fxns.new_kmm_objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product)
            dlsif_objective_dalpha = fxns.dopt_objective_dx.autograd_fxn(lsif_objective)

            #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
            #dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
            cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
            @basic.timeit('gmres_solver')
            def gmres_solver(A,b):
                asdf = scipy.sparse.linalg.gmres(A, b, maxiter=10)
    #            print asdf[1]
    #            pdb.set_trace()
                return asdf[0]
            #lin_solver = cg_solver
            bicg_solver = lambda A,b: scipy.sparse.linalg.gmres(A, b)[0]
            lin_solver = gmres_solver
    #        lin_solver = bicg_solver
            lsif_alpha_given_B_fxn = fxns.fxn.wrap_primitive(fxns.full_quad_opt( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                lin_solver=lin_solver,
                objective=lsif_objective,
                dobjective_dx=dlsif_objective_dalpha,
                ineq_constraints=fxns.Gh_ineq_constraints.autograd_fxn(_get_Gh=fxns.LSIF_get_Gh),
    #            get_Gh=fxns.LSIF_get_Gh(),
    #            optimizer=optimizers.scipy_minimize_optimizer(method=cvx_opt_scipy_minimize_method, verbose=cvx_opt_scipy_minimize_verbose, info_f=cvx_opt_scipy_minimize_info_f, options=cvx_opt_scipy_minimize_options),
                warm_start=quad_opt_warm_start
    #            optimizer = scipy.optimize.minimize(options={'maxiter':100}) FIX
                ))

        else:
            lsif_alpha_given_B_fxn = fxns.fxn.autograd_fxn(_val=fxns.least_squares_lsif_alpha_given_B) # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif

        if not lsif_least_squares:    
        
            final_fxn = fxns.g_thru_f_opt_new(# ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_lsif, max_ratio, add_reg, xs_basis, add_reg_lsif
                g=lsif_alpha_given_B_fxn, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                h=objective_given_lsif_alphas, # lsif_alpha, xs_train, xs_basis, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, max_ratio, add_reg
    #            h=objective_given_b_logreg, # b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, max_ratio, add_reg
                g_argnums=(1,2,4,5,9,12,10,13),
                h_argnums=(1,12,0,3,4,5,6,7,8,10,11),
                g_val_h_argnum=0
            )
        else:
#            final_fxn = fxns.two_step(# ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_lsif, max_ratio, add_reg, xs_basis
            final_fxn = fxns.two_step.autograd_fxn(# ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_lsif, max_ratio, add_reg, xs_basis, add_reg_lsif
                g=lsif_alpha_given_B_fxn, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
                h=objective_given_lsif_alphas, # lsif_alpha, xs_train, xs_basis, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, max_ratio, add_reg
    #            h=objective_given_b_logreg, # b_logreg, xs_train, xs_test, ys_train, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, max_ratio, add_reg
                g_argnums=(1,2,4,5,9,12,10,13),
                h_argnums=(1,12,0,3,4,5,6,7,8,10,11),
                g_val_h_argnum=0
            )

        ws_full_train = ws_full_train_f(xs_train, xs_test)

        xs_basis = xs_test[0:num_basis]

        def objective(c_lsif, sigma, c_lsqr, B):
            return final_fxn(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_lsif, max_ratio, add_reg, xs_basis, add_reg_lsif)

        # create B optimizer, which takes in c_lsqr and B

        def B_optimizer_get_objective(c_lsif, sigma, c_lsqr, B):
            objective = lambda B: final_fxn.val(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_lsif, max_ratio, add_reg, xs_basis, add_reg_lsif)
            return objective

        def B_optimizer_get_dobjective_dB(c_lsif, sigma, c_lsqr, B):
            dobjective_dB = lambda B: final_fxn.grad(ys_train, xs_train, xs_test, ws_full_train, sigma, B, c_lsqr_loss, c_lsqr_loss_eval, c_lsqr, c_lsif, max_ratio, add_reg, xs_basis, add_reg_lsif, care_argnums=(5,))
            return dobjective_dB

        def B_optimizer_get_B(c_lsif, sigma, c_lsqr, B):
            return B

        def B_optimizer_out_f((c_lsif, sigma, c_lsqr, B), (objective, dobjective_dB, _B), new_B):
            return (c_lsif, sigma, c_lsqr, new_B)

        if unconstrained:
            B_horse_optimizer = optimizers.scipy_minimize_optimizer(method=unconstrained_scipy_minimize_method, verbose=unconstrained_scipy_minimize_verbose, info_f=unconstrained_scipy_minimize_info_f, options=unconstrained_scipy_minimize_options)
        else:
            B_horse_optimizer = optimizers.pymanopt_optimizer(**pymanopt_options)

        B_optimizer = optimizers.get_stuff_optimizer(B_horse_optimizer, (B_optimizer_get_objective, B_optimizer_get_dobjective_dB, B_optimizer_get_B), B_optimizer_out_f) # takes in c_lsqr and B, outputs new c_lsqr and B


        # create sigma optimizer (more like updater), takes in sigma, c_lsqr, B
        sigma_sample = 20
        xs_train_sigma = xs_train[0:sigma_sample]
        xs_test_sigma = xs_test[0:sigma_sample]
        xs_sigma = np.concatenate((xs_train_sigma, xs_test_sigma), axis=0)

        def update_sigma(c_lsif, sigma, c_lsqr, B):
            new_sigma = utils.median_distance(np.dot(xs_sigma,B), np.dot(xs_sigma, B))
            return c_lsif, new_sigma, c_lsqr, B

        # create c_lsqr optimizer, which takes in c_lsqr and B.  

        # first create objective 

        # every: this was necessary bc previous fxns never explicitly outputted ws_train.  for kmm this fxn will be unnecessary, just uses ws_train_given_B
        #B_to_ws_train = fxns.two_step( # xs_train, xs_test, sigma, B, max_ratio, c_lsif, xs_basis
        B_to_ws_train = fxns.two_step.autograd_fxn( # xs_train, xs_test, sigma, B, max_ratio, c_lsif, xs_basis, add_reg_lsif
            g=lsif_alpha_given_B_fxn, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=fxns.fxn.autograd_fxn(_val=fxns.lsif_alpha_to_ratios), # lsif_alpha, xs_train, xs_basis, sigma, B, max_ratio)
            g_argnums=(0,1,2,3,5,6,4,7),
            h_argnums=(0,6,2,3,4),
            g_val_h_argnum=0
        )

        ws_test = np.ones(shape=len(xs_test))

        # every: probably necessary since 1. objective does not involve ws penalty, and 2. penalty is on test losses, so have to reassemble the g and h here
        ws_train_to_test_loss = fxns.two_step( # B, xs_train, ys_train, ws_train, c_lsqr, xs_test, ys_test, ws_test, add_reg
#            g=weighted_loss_b_opt, # B, xs_train, ys_train, ws_train, c_lsqr
            g=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, xs_basis, ys_train, lsif_alpha, c_lsqr, sigma, max_ratio
            h=weighted_loss_given_b_opt, # B, xs_test, ys_test, ws_test, b_opt, c_lsqr, add_reg
            g_argnums=(0,1,2,3,4),
            h_argnums=(0,5,6,7,4,8),
            g_val_h_argnum=4
        )

        # every: necessary since g,h have to be assembled together differently than before, due to loss being on test
        #B_to_test_loss = fxns.two_step( # B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, max_ratio, c_lsif, xs_basis
        B_to_test_loss = fxns.two_step.autograd_fxn( # B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, max_ratio, c_lsif, xs_basis, add_reg_lsif
            g=B_to_ws_train, # xs_train, xs_test, sigma, B, max_ratio, c_lsif, xs_basis, add_reg_lsif
            h=ws_train_to_test_loss, # B, xs_train, ys_train, ws_train, c_lsqr, xs_test, ys_test, ws_test, add_reg
            g_argnums=(1,4,8,0,9,10,11,12),
            h_argnums=(0,1,2,3,4,5,6,7),
            g_val_h_argnum=3
        )


        use_ys_test = False
        num_folds = 2

        def c_lsqr_optimizer_get_objective(c_lsif, sigma, c_lsqr, B):
            if use_ys_test:
                objective = lambda c_lsqr: B_to_test_loss(B, xs_train, ys_train, c_lsqr, xs_test, ys_test, ws_test, add_reg, sigma, max_ratio, c_lsif, xs_basis)
                return objective
            else:
                def cv_test_loss(c_lsqr):
                    ws_train = B_to_ws_train(xs_train, xs_test, sigma, B, max_ratio, c_lsif, xs_basis, add_reg_lsif)
                    from sklearn.model_selection import KFold
                    kf = KFold(n_splits=num_folds)
                    test_losses = 0.
                    for (train_idx, test_idx) in kf.split(xs_train):
                        xs_train_train = xs_train[train_idx]
                        xs_train_test = xs_train[test_idx]
                        ys_train_train = ys_train[train_idx]
                        ys_train_test = ys_train[test_idx]
                        ws_train_train = ws_train[train_idx]
                        ws_train_test = ws_train[test_idx]
                        fold_test_loss = ws_train_to_test_loss(B, xs_train_train, ys_train_train, ws_train_train, c_lsqr, xs_train_test, ys_train_test, ws_train_test, add_reg)
                        test_losses += fold_test_loss * len(train_idx)
                    return test_losses / len(xs_train)
                return cv_test_loss
            
        def c_lsqr_optimizer_get_c_lsqr(c_lsif, sigma, c_lsqr, B):
            return c_lsqr

        def c_lsqr_optimizer_out_f((c_lsif, sigma, c_lsqr, B), (objective, _c_lsqr), new_c_lsqr):
            return (c_lsif, sigma, new_c_lsqr, B)

        
        c_lsqr_horse_optimizer = optimizers.scalar_fxn_optimizer(method=linesearch_method, options=linesearch_options, init_window_width=linesearch_init_window_width)
        c_lsqr_optimizer = optimizers.get_stuff_optimizer(c_lsqr_horse_optimizer, (c_lsqr_optimizer_get_objective, c_lsqr_optimizer_get_c_lsqr), c_lsqr_optimizer_out_f) # takes in c_lsqr and B, outputs new c_lsqr and B


        # create c_lsif optimizer

        add_reg_lsif_test = False
        
        #test_lsif_objective = fxns.two_step( # xs_train_train, xs_train_test, sigma, B, c_lsif, xs_basis, max_ratio, xs_test_train, xs_test_test
        test_lsif_objective = fxns.two_step.autograd_fxn( # xs_train_train, xs_train_test, sigma, B, c_lsif, xs_basis, max_ratio, xs_test_train, xs_test_test, add_reg_lsif, add_reg_lsif_test
            g=lsif_alpha_given_B_fxn, # xs_train_train, xs_train_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=lsif_objective, # xs_test_train, xs_test_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif_test, lsif_alpha
            g_argnums=(0,1,2,3,4,5,6,9),
            h_argnums=(7,8,2,3,4,5,6,10),
            g_val_h_argnum=8
            )

        lsif_num_folds = 2

#        def c_lsif_optimizer_get_objective(_c_lsif, sigma, c_lsqr, B):
#        def cv_lsif_objective(c_lsif, sigma, c_lsqr, B):
        def cv_lsif_objective(c_lsif, sigma, c_lsqr, B, xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds):
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=lsif_num_folds)
            test_lsif_objectives = 0.
            for (train_train_idx, train_test_idx), (test_train_idx, test_test_idx) in itertools.izip(kf.split(xs_train), kf.split(xs_test)):
                xs_train_train = xs_train[train_train_idx]
                xs_train_test = xs_train[train_test_idx]
                xs_test_train = xs_test[test_train_idx]
                xs_test_test = xs_test[test_test_idx]
                fold_lsif_objective = test_lsif_objective(xs_train_train, xs_train_test, sigma, B, c_lsif, xs_basis, max_ratio, xs_test_train, xs_test_test, add_reg_lsif, add_reg_lsif_test)
                test_lsif_objectives += fold_lsif_objective
            return test_lsif_objectives / lsif_num_folds
#            return cv_lsif_objective

        def c_lsif_optimizer_get_objective(c_lsif, sigma, c_lsqr, B):
            return (lambda _c_lsif: cv_lsif_objective(_c_lsif, sigma, c_lsqr, B, xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds))

        def c_lsif_optimizer_get_c_lsif(c_lsif, sigma, c_lsqr, B):
            return c_lsif

        def c_lsif_optimizer_out_f((c_lsif, sigma, c_lsqr, B), (objective, _c_lsif), new_c_lsif):
            return (new_c_lsif, sigma, c_lsqr, B)

        
        c_lsif_horse_optimizer = optimizers.scalar_fxn_optimizer(method=linesearch_method, options=linesearch_options, init_window_width=linesearch_init_window_width)
        c_lsif_optimizer = optimizers.get_stuff_optimizer(c_lsif_horse_optimizer, (c_lsif_optimizer_get_objective, c_lsif_optimizer_get_c_lsif), c_lsif_optimizer_out_f)

        # create grid_search c_lsif and sigma optimizer
        
        def c_lsif_sigma_optimizer_get_objective(c_lsif, sigma, c_lsqr, B):
            return (lambda _c_lsif, _sigma: cv_lsif_objective(_c_lsif, _sigma, c_lsqr, B, xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds))

        eps = 0.1
        c_lsif_range = np.linspace(0,2.5,15) + eps
        c_sigma_range = np.linspace(0,10,11) + eps
        
        def c_lsif_sigma_optimizer_get_c_lsif_sigma_ranges(c_lsif, sigma, c_lsqr, B):
            return c_lsif_range, c_sigma_range

        def c_lsif_sigma_optimizer_out_f((c_lsif, sigma, c_lsqr, B), (objective, ranges), (new_c_lsif, new_sigma)):
            #print 'B B'
            #print B
            return (new_c_lsif, new_sigma, c_lsqr, B)

        c_lsif_sigma_horse_optimizer = optimizers.grid_search_optimizer()
        c_lsif_sigma_optimizer = optimizers.get_stuff_optimizer(c_lsif_sigma_horse_optimizer, (c_lsif_sigma_optimizer_get_objective, c_lsif_sigma_optimizer_get_c_lsif_sigma_ranges), c_lsif_sigma_optimizer_out_f)

        # create gradient descent log_c_lsif_sigma optimizer
        def cv_log_c_lsif_sigma_objective_horse(c_lsqr, B, xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds, log_c_lsif_sigma):
            c_lsif = np.exp(log_c_lsif_sigma[0])
            sigma = np.exp(log_c_lsif_sigma[1])
            return cv_lsif_objective(c_lsif, sigma, c_lsqr, B, xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds)

        cv_log_c_lsif_sigma_objective = fxns.objective.autograd_fxn(_hessian_vector_product=fxns.autograd_hessian_vector_product, _val=cv_log_c_lsif_sigma_objective_horse, _arg_shape=lambda *args: (2,))
        dcv_log_c_lsif_sigma_objective_dstuff = fxns.dopt_objective_dx.autograd_fxn(cv_log_c_lsif_sigma_objective)
        
        #logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=scale_sigma)
        #dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        log_c_lsif_sigma_opt_fxn = fxns.cvx_opt( # c_lsqr, B, xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds
            lin_solver=cg_solver,
            objective=cv_log_c_lsif_sigma_objective,
            dobjective_dx=dcv_log_c_lsif_sigma_objective_dstuff,
            optimizer=optimizers.scipy_minimize_optimizer(method=cvx_opt_scipy_minimize_method, verbose=cvx_opt_scipy_minimize_verbose, info_f=cvx_opt_scipy_minimize_info_f, options=cvx_opt_scipy_minimize_options),
            warm_start=cvx_opt_warm_start
#            optimizer = scipy.optimize.minimize(options={'maxiter':100}) FIX
            )        
        
        # create optimizer and optimize
#        horse_optimizer = optimizers.many_optimizer((update_sigma, B_optimizer, c_lsqr_optimizer))
        horse_optimizer = optimizers.many_optimizer((c_lsqr_optimizer, B_optimizer))
#        horse_optimizer = optimizers.many_optimizer((update_sigma, c_lsif_optimizer, B_optimizer,))
#        horse_optimizer = optimizers.many_optimizer((,))
#        horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_lsqr_optimizer, B_optimizer))
#        horse_optimizer = optimizers.many_optimizer((c_lsif_sigma_optimizer, c_lsqr_optimizer))

        
        optimizer = optimizers.multiple_optimizer(horse_optimizer, num_tries=num_tries, num_args=4)

        sigma_init_f = lambda: 5.

        B_init_f = B_init_f_getter(xs_train, ys_train, xs_test)

        c_lsqr_init_f = lambda: 1.

        #c_lsif_init_f = lambda: 1
        c_lsif_init_f = lambda: 1.01 * len(xs_train)
                

#        B_c_lsqr_init_f = lambda: B_init_f(), c_lsqr_init_f()

        if False:
            for i in xrange(10):
                print ' '
                ans = np.exp(log_c_lsif_sigma_opt_fxn(np.random.random(), B_init_f(), xs_train, xs_test, xs_basis, max_ratio, add_reg_lsif, add_reg_lsif_test, lsif_num_folds))
                print 'ans', ans

            #pdb.set_trace()

        B_random = B_init_f()
        us_train = np.dot(xs_train, B_random)
        us_test = np.dot(xs_test, B_random)
        sigma = utils.median_distance(np.concatenate((us_train, us_test), axis=0), np.concatenate((us_train, us_test), axis=0))
        print 'sigma', sigma
#        pdb.set_trace()
        
        for i in range(3):
            if False:
                final_fxn.grad_check(ys_train, xs_train, xs_test, ws_full_train, sigma_init_f(), B_init_f(), c_lsqr_loss, c_lsqr_loss_eval, c_lsqr_init_f(), c_lsif_init_f(), max_ratio, add_reg, xs_basis, add_reg_lsif, care_argnums=(5,))
        #basic.do_cprofile('cumtime')(final_fxn.grad_check)(ys_train, xs_train, xs_test, ws_full_train, sigma_init_f(), B_init_f(), c_lsqr_loss, c_lsqr_loss_eval, c_lsqr_init_f(), c_lsif, max_ratio, add_reg, xs_basis, care_argnums=(5,))

        #pdb.set_trace()

        c_lsif_fit, sigma_fit, c_lsqr_fit, B_fit = optimizer.optimize(objective, c_lsif_init_f, sigma_init_f, c_lsqr_init_f, B_init_f)
        
        # create predictor

        """
        # fix: this fxn is same as B_to_ws_train
        logreg_ratios_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_logreg, scale_sigma, max_ratio
                                               g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
                                               h=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_wrapper), # b_logreg, xs_train, xs_test, sigma, scale_sigma, B, max_ratio
                                               g_argnums=(0,1,2,3,4),
                                               h_argnums=(0,1,2,5,3,6),
                                               g_val_h_argnum=0,
                                               )
        """
#        ws_train = logreg_ratios_fxn.val(xs_train, xs_test, sigma, B_fit, c_logreg, scale_sigma, max_ratio)
#        ws_train = B_to_ws_train(xs_train, xs_test, sigma, scale_sigma, B_fit, max_ratio, c_logreg)
        ws_train = B_to_ws_train(xs_train, xs_test, sigma_fit, B_fit, max_ratio, c_lsif, xs_basis, add_reg_lsif)
#        print ws_train
#        print 'ws_train sum', np.sum(ws_train)

        b_opt_fxn = fxns.two_step( # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, ys_train, c_lsqr, add_reg_lsif
            g=lsif_alpha_given_B_fxn, # xs_train, xs_test, sigma, B, c_lsif, xs_basis, max_ratio, add_reg_lsif
            h=weighted_loss_b_opt, # B, xs_train, xs_basis, ys_train, lsif_alpha, c_lsqr, sigma, max_ratio
            g_argnums=(0,1,2,3,4,5,6,9),
            h_argnums=(3,0,5,7,8,2,6),
            g_val_h_argnum=4
            )
        
#        pdb.set_trace()
        if unconstrained:
            b_predictor = B_fit
        else: 
            b_fit = b_opt_fxn(xs_train, xs_test, sigma_fit, B_fit, c_lsif, xs_basis, max_ratio, ys_train, c_lsqr_fit, add_reg_lsif)
#            b_fit = fxns.weighted_lsqr_b_opt(B_fit, xs_train, ys_train, ws_train, c_lsqr_fit)
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
