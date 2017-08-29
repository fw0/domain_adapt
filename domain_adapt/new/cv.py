import numpy as np
import itertools, pdb
import python_utils.python_utils.caching as caching
import copy

def get_oos_loss(which_loss, num_folds, use_test, fitter, xs_train, xs_test, ys_train, ys_test=None):
    
    if which_loss == 'square':
        loss = lambda predicted_ys, ys: np.mean((ys-predicted_ys)**2)
    elif which_loss == '0-1':
        loss = lambda predicted_ys, ys: np.mean((2 * ((predicted_ys > 0.5).astype(int) - 0.5)) == ys)
    elif which_loss == 'abs':
        loss = lambda predicted_ys, ys: np.mean(np.abs(ys-predicted_ys))
        
    if num_folds > 0:
        losses = []
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=num_folds)
        for (i,((train_is_idx, train_oos_idx), (test_is_idx, test_oos_idx))) in enumerate(itertools.izip(kf.split(xs_train), kf.split(xs_test))):
            print 'fold', i
            xs_train_is = xs_train[train_is_idx]
            xs_train_oos = xs_train[train_oos_idx]
            xs_test_is = xs_test[test_is_idx]
            xs_test_oos = xs_test[test_oos_idx]
            ys_train_is = ys_train[train_is_idx]
            ys_train_oos = ys_train[train_oos_idx]
            ys_test_is = ys_test[test_is_idx]
            ys_test_oos = ys_test[test_oos_idx]
            forward_predictor = fitter(xs_train_is, xs_test_is, ys_train_is, ys_test_is)
            predicted_ys_test_oos = np.array(map(forward_predictor, xs_test_oos))
            if use_test:
                #ys_test_is = ys_test[test_is_idx]
                #ys_test_oos = ys_test[test_oos_idx]
                losses.append(loss(predicted_ys_test_oos, ys_test_oos))
            else:
                backward_predictor = fitter(xs_test_oos, xs_train_oos, predicted_ys_test_oos)
                predicted_ys_train_is = np.array(map(backward_predictor, xs_train_is))
                result = loss(predicted_ys_train_is, ys_train_is)
                print 'result', result
                losses.append(result)
        print 'oos losses', losses
        #pdb.set_trace()
        return losses
    else:
        assert use_test
        predictor = fitter(xs_train, xs_test, ys_train, ys_test)
        predicted_ys_test = np.array(map(predictor, xs_test))
        #pdb.set_trace()
        return [loss(predicted_ys_test, ys_test)]



def cv_helper(which_loss, num_folds, use_test, fitter_constructor, arg_dicts, xs_train, xs_test, ys_train, ys_test=None):
    d = []
    for arg_dict in arg_dicts:
        fitter = fitter_constructor(**arg_dict)
        losses = get_oos_loss(which_loss, num_folds, use_test, fitter, xs_train, xs_test, ys_train, ys_test)
        if num_folds == 0:
            d.append((arg_dict, losses[0]))
        else:
            d.append((arg_dict, (np.mean(losses),np.std(losses),losses)))
    if num_folds == 0:
        best_arg_dict = min(d, key=lambda (arg_dict, mean): mean)[0]
    else:
        best_arg_dict = min(d, key=lambda (arg_dict, (mean, std, vals)): mean)[0]
    caching.fig_archiver.log_text(d)
    caching.fig_archiver.log_text('best:', best_arg_dict)
#    self.opt_log = d
    print 'best fitter'
    return best_arg_dict
#    best_fitter = fitter_constructor(best_arg)
#    return best_fitter(xs_train, xs_test, ys_train, ys_test)


class cv_fitter(object):

    def __init__(self, fitter_constructor, arg_dicts, which_loss, num_folds, use_test):
        self.fitter_constructor, self.arg_dicts, self.which_loss, self.num_folds, self.use_test = fitter_constructor, arg_dicts, which_loss, num_folds, use_test

    def __call__(self, xs_train, xs_test, ys_train, ys_test=None):
        # assumes individual fitters do not use ys_test

        best_arg_dict = cv_helper(self.which_loss, self.num_folds, self.use_test, self.fitter_constructor, self.arg_dicts, xs_train, xs_test, ys_train, ys_test)
        best_fitter = self.fitter_constructor(**best_arg_dict)
        return best_fitter(xs_train, xs_test, ys_train, ys_test)


class cheap_cv_fitter(object):
    
    def __init__(self, fitter_constructor, arg_range_dict, default_dict, arg_order, which_loss, num_folds, use_test):
        self.fitter_constructor, self.arg_range_dict, self.default_dict, self.arg_order, self.which_loss, self.num_folds, self.use_test = fitter_constructor, arg_range_dict, default_dict, arg_order, which_loss, num_folds, use_test

    def __call__(self, xs_train, xs_test, ys_train, ys_test=None):
        ans = copy.deepcopy(self.default_dict)
        for arg_to_change in self.arg_order:
            round_arg_dicts = []
            for possible in self.arg_range_dict[arg_to_change]:
                new = copy.deepcopy(ans)
                new[arg_to_change] = possible
                round_arg_dicts.append(new)
            round_best_arg_dict = cv_helper(self.which_loss, self.num_folds, self.use_test, self.fitter_constructor, round_arg_dicts, xs_train, xs_test, ys_train, ys_test)
            best_arg_to_change = round_best_arg_dict[arg_to_change]
            ans[arg_to_change] = best_arg_to_change
        best_fitter = self.fitter_constructor(**ans)
        return best_fitter(xs_train, xs_test, ys_train, ys_test)
            

#class cv_fitter(object):

#    def __init__(self, fitter_list, which_loss, num_folds, use_test):
#        self.fitter_list, self.which_loss, self.num_folds, self.use_test = fitter_list, which_loss, num_folds, use_test

#    def __call__(self, xs_train, xs_test, ys_train, ys_test=None):
        # assumes individual fitters do not use ys_test
#        d = []
#        for fitter in self.fitter_list:
#            pdb.set_trace()
#            losses = get_oos_loss(self.which_loss, self.num_folds, self.use_test, fitter, xs_train, xs_test, ys_train, ys_test)
#            if self.num_folds == 0:
#                d.append((fitter, losses[0]))
#            else:
#                d.append((fitter, (np.mean(losses),np.std(losses),losses)))
#        if self.num_folds == 0:
#            best_fitter = min(d, key=lambda (fitter, mean): mean)[0]
#        else:
#            best_fitter = min(d, key=lambda (fitter, (mean, std, vals)): mean)[0]
#        caching.fig_archiver.log_text(d)
#        caching.fig_archiver.log_text('best:', best_fitter.__dict__)
#        self.opt_log = d
#        print 'best fitter'
#        return best_fitter(xs_train, xs_test, ys_train, ys_test)

