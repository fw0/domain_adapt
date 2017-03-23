import fxns, optimizers, autograd, autograd.numpy as np, scipy
import pdb

def logreg_ratio_1d_unconstrained_fitter(c_B, c_logreg, tradeoff, sigma, info_f=lambda x: None, verbose=False, num_tries=1, B_init_f=None, options={}):
    """
    returns fxn that makes predictions
    """

    def fitter(xs_train, xs_test, ys_train):
        weighted_lsqr_loss_fxn = fxns.two_step( # B, xs_train, xs_test, ys_train, sigma, b_logreg, c_B
            g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios_scale_sigma), # b_logreg, xs_train, xs_test, sigma, B
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
            g_argnums=(5,1,2,4,0),
            h_argnums=(0,1,3,6),
            g_val_h_argnum=4,
        )

        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=True)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        lin_solver = cg_solver
        b_opt_given_B_fxn = fxns.cvx_opt( # xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver,
            objective=logreg_ratio_objective,
            dobjective_dx=dlogreg_ratio_objective_db
        )

        weighted_lsqr_loss_thru_ws_opt_fxn = fxns.g_thru_f_opt( # ys_train, xs_train, xs_test, sigma, B, c_logreg, c_B
            g=b_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
            h=weighted_lsqr_loss_fxn, # B, xs_train, xs_test, ys_train, sigma, b_logreg, c_B
            g_argnums=(1,2,3,4,5), 
            h_argnums=(4,1,2,0,3,6), 
            g_val_h_argnum=5
        )

        objective = lambda B: weighted_lsqr_loss_thru_ws_opt_fxn.val(ys_train, xs_train, xs_test, sigma, B, c_logreg, c_B)
        dobjective_dB = lambda B: weighted_lsqr_loss_thru_ws_opt_fxn.grad(ys_train, xs_train, xs_test, sigma, B, c_logreg, c_B, care_argnums=(4,))

        optimizer = optimizers.multiple_optimizer(optimizers.scipy_minimize_optimizer(verbose=5, info_f=info_f, options=options), num_tries=num_tries, num_args=1)

        x_dim = xs_train.shape[1]
        _B_init_f = lambda: np.random.normal(size=x_dim) if B_init_f is None else B_init_f
        B_fit = optimizer.optimize(objective, dobjective_dB, _B_init_f)

        return fxns.fxn(_val = lambda xs: np.dot(xs, B_fit))

    return fitter


def baseline_fitter(c_logreg, c_lsqr, sigma):

    def fitter(xs_train, xs_test, ys_train):

        logreg_ratio_objective = fxns.logreg_ratio_objective.autograd_fxn(scale_sigma=False)
        dlogreg_ratio_objective_db = fxns.dopt_objective_dx.autograd_fxn(logreg_ratio_objective)
        cg_solver = lambda A,b: scipy.sparse.linalg.cg(A, b)[0]
        b_ratio_opt_given_B_fxn = fxns.cvx_opt(# xs_train, xs_test, sigma, B, c_logreg
            lin_solver=cg_solver, 
            objective=logreg_ratio_objective, 
            dobjective_dx=dlogreg_ratio_objective_db
        )

        b_ratio_opt_to_b_reg_opt = fxns.two_step( # b_ratio_opt, xs_train, xs_test, ys_train, sigma, B, c_lsqr
            g=fxns.fxn.autograd_fxn(_val=fxns.b_to_logreg_ratios), # b_ratio_opt, xs_train, xs_test, sigma, B
            h=fxns.fxn.autograd_fxn(_val=fxns.weighted_lsqr_b_opt), # B, xs_train, ys_train, ws_train, c_lsqr
            g_argnums=(0,1,2,4,5),
            h_argnums=(5,1,3,6),
            g_val_h_argnum=3
        )

        b_ratio_reg_opt_fxn = fxns.two_step( # xs_train, xs_test, ys_train, sigma, B, c_logreg, c_lsqr
            g=b_ratio_opt_given_B_fxn, # xs_train, xs_test, sigma, B, c_logreg
            h=b_ratio_opt_to_b_reg_opt, # b_ratio_opt, xs_train, xs_test, ys_train, sigma, B, c_lsqr
            g_argnums=(0,1,3,4,5),
            h_argnums=(0,1,2,3,4,6),
            g_val_h_argnum=0
        )

        x_dim = xs_train.shape[1]
        B = np.eye(x_dim)

        b_fit = b_ratio_reg_opt_fxn.val(xs_train, xs_test, ys_train, sigma, B, c_logreg, c_lsqr)

        return fxns.fxn(_val = lambda xs: np.dot(xs, b_fit))

    return fitter
