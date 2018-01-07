import numpy as np
import pandas as pd
import copy, itertools

class all_fitter(object):

    def __init__(self, horse, horse_0_1=True):
        self.horse, self.horse_0_1 = horse, horse_0_1

    def fit(self, xs, ys, zs):
        """
        adds zs as feature to xs
        """
        if self.horse_0_1:
            ys = (ys + 1) / 2.
        actual_xs = np.concatenate((xs, np.expand_dims(zs, 1)), axis=1)
        horse = copy.deepcopy(self.horse)
        predictor = horse.fit(actual_xs, ys)
        info = {'one_ESS':len(xs)}
        return all_predictor(predictor, info)

class all_predictor(object):

    def __init__(self, predictor, info):
        self.predictor, self.info = predictor, info

    def predict(self, xs, zs):
        actual_xs = np.concatenate((xs, np.expand_dims(zs, 1)), axis=1)
        self.predictor.predict(actual_xs)

class subgroup_fitter(object):

    def __init__(self, horse, just_ones=False, horse_0_1=True):
        self.horse, self.just_ones, self.horse_0_1 = horse, just_ones, horse_0_1

    def fit(self, xs, ys, zs):
        if self.horse_0_1:
            ys = (ys + 1) / 2.
        info = {}
        if not self.just_ones:
            zero_xs, zero_ys = xs[zs==0,:], ys[zs==1]
            zero_horse = copy.deepcopy(self.horse)
            zero_predictor = zero_horse.fit(zero_xs, zero_ys)
            info['zero_ESS'] = len(zero_xs)
        else:
            zero_horse = None
        one_xs, one_ys = xs[zs==1,:], ys[zs==1]
        one_horse = copy.deepcopy(self.horse)
        one_predictor = one_horse.fit(one_xs, one_ys)
        info['one_ESS'] = len(one_xs)
        return subgroup_predictor(zero_predictor, one_predictor, info)

class subgroup_predictor(object):

    def __init__(self, zero_predictor, one_predictor, info):
        self.zero_predictor, self.one_predictor, self.info = zero_predictor, one_predictor, info
        
    def predict(self, xs, zs):
        if self.zero_predictor is None:
            assert np.sum(zs) == len(zs)
        ans = []
        for (x, z) in itertools.izip(xs, zs):
            if z == 0:
                ans.append(self.zero_predictor.predict([x])[0])
            elif z == 1:
                ans.append(self.one_predictor.predict([x])[0])
        return np.array(ans)
        
class cov_shift_fitter(object):
    """
    for now, only can make predictions for zs==1
    """
    def __init__(self, horse, use_test_cov=True, just_ones=False, horse_0_1=True):
        self.horse, self.just_ones, self.horse_0_1 = horse, just_ones, horse_0_1

    def fit(self, xs, ys, zs):
        if self.horse_0_1:
            ys = (ys + 1) / 2.
        self.xs, self.ys, self.zs = xs, ys, zs
        horse = copy.deepcopy(horse)
        return cov_shift_predictor(xs, ys, zs, horse, self.just_ones)

class cov_shift_predictor(object):

    def __init__(self, xs, ys, zs, horse, just_ones):
        self.xs, self.ys, self.zs, self.horse, self.just_ones = xs, ys, zs, horse, just_ones
        
    def predict(self, xs, zs):
        info = {}
        if not self.just_ones:
            if use_test_cov:
                zero_test_xs = np.concatenate((self.xs[self.zs==0,:], xs[zs==0,:]), axis=0)
            else:
                zero_test_xs = self.xs[self.zs==0,:]
            zero_horse = copy.deepcopy(horse)
            zero_predictor = zero_horse.fit(self.xs, self.ys, zero_test_xs)
            info['zero_ESS'] = zero_predictor.ESS # fix
        else:
            zero_horse = None
            assert np.sum(zs) == len(zs)

        if use_test_cov:
            one_test_xs = np.concatenate((self.xs[self.zs==1,:], xs[zs==1,:]), axis=0)
        else:
            one_test_xs = self.xs[self.zs==1,:]
        one_horse = copy.deepcopy(horse)
        one_predictor = one_horse.fit(self.xs, self.ys, one_test_xs)
        info['one_ESS'] = one_predictor.ESS # fix
            
        ans = []
        for (x, z) in itertools.izip(xs, zs):
            if z == 0:
                ans.append(zero_predictor.predict([x])[0])
            elif z == 1:
                ans.append(one_predictor.predict([x])[0])
        return np.array(ans)

class subsample_cv_getter(object):
    
    def __init__(self, train_prop):
        self.train_prop = train_prop

    def __call__(self, xs, ys, all_zs, i):
        N = len(xs)
        state = np.random.get_state()
        np.random.seed(i)
        num_keep = int(self.train_prop * N)
        keep_train = np.random.shuffle(np.concatenate((np.ones(num_keep).astype(bool), np.zeros(N-num_keep).astype(bool)), axis=0))
        keep_test = np.logical_not(keep_train)
        np.random.set_state(state)
        return xs.iloc[keep_train], ys.iloc[keep_train], all_zs.iloc[keep_train], xs.iloc[keep_test], ys.iloc[keep_test], all_zs.iloc[keep_test]
