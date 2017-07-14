import numpy as np
import itertools, pdb
import python_utils.python_utils.caching as caching

def get_oos_loss(which_loss, num_folds, use_test, fitter, xs_train, xs_test, ys_train, ys_test=None):
    
    if which_loss == 'square':
        loss = lambda predicted_ys, ys: np.sum((ys-predicted_ys)**2)
    elif which_loss == '0-1':
        loss = lambda predicted_ys, ys: np.sum((2 * ((predicted_ys > 0.5).astype(int) - 0.5)) == ys)
        
    if num_folds > 0:
        losses = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=num_folds)
        for ((train_is_idx, train_oos_idx), (test_is_idx, test_oos_idx)) in itertools.izip(kf.split(xs_train), kf.split(xs_test)):
            xs_train_is = xs_train[train_is_idx]
            xs_train_oos = xs_train[train_oos_idx]
            xs_test_is = xs_test[test_is_idx]
            xs_test_oos = xs_test[test_oos_idx]
            ys_train_is = ys_train[train_is_idx]
            ys_train_oos = ys_train[train_oos_idx]
            forward_predictor = fitter(xs_train_is, xs_test_is, ys_train_is)
            predicted_ys_test_oos = np.array(map(forward_predictor, xs_test_oos))
            if use_test:
                ys_test_is = ys_test[test_is_idx]
                ys_test_oos = ys_test[test_oos_idx]
                losses.append(loss(predicted_ys_test_oos, ys_test_oos))
            else:
                backward_predictor = fitter(xs_test_oos, xs_train_oos, predicted_ys_test_oos)
                predicted_ys_train_is = np.array(map(backward_predictor, xs_train_is))
                losses.append(loss(predicted_ys_train_is, ys_train_is))
        return losses
    else:
        assert use_test
        predictor = fitter(xs_train, xs_test, ys_train)
        predicted_ys_test = np.array(map(predictor, xs_test))
        return [loss(predicted_ys_test, ys_test)]


class cv_fitter(object):

    def __init__(self, fitter_list, which_loss, num_folds, use_test):
        self.fitter_list, self.which_loss, self.num_folds, self.use_test = fitter_list, which_loss, num_folds, use_test

    def __call__(self, xs_train, xs_test, ys_train, ys_test=None):
        # assumes individual fitters do not use ys_test
        d = []
        for fitter in self.fitter_list:
            losses = get_oos_loss(self.which_loss, self.num_folds, self.use_test, fitter, xs_train, xs_test, ys_train, ys_test)
            if self.num_folds == 0:
                d.append((fitter, losses[0]))
            else:
                d.append((fitter, (np.mean(losses),np.std(losses),losses)))
        if self.num_folds == 0:
            best_fitter = min(d, key=lambda (fitter, mean): mean)[0]
        else:
            best_fitter = min(d, key=lambda (fitter, (mean, std, vals)): mean)[0]
        caching.fig_archiver.log_text(d)
        caching.fig_archiver.log_text('best:', best_fitter.__dict__)
        self.opt_log = d
        return best_fitter(xs_train, xs_test, ys_train)
