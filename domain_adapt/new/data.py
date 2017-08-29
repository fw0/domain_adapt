import numpy as np
import pdb


def v_data(num_train, num_test, useful_dim, same_dim, diff_dim, alpha=0.2):

    low = -1.
    mid = 0.
    high = 1.

    xs1_low_prob_train = 0.99
    xs1_low_prob_test = 0.0

    xs1_train = np.array([np.random.uniform(low=low, high=mid, size=useful_dim) if np.random.uniform() < xs1_low_prob_train else np.random.uniform(low=mid, high=high, size=useful_dim) for i in xrange(num_train)])
    xs1_test = np.array([np.random.uniform(low=low, high=mid, size=useful_dim) if np.random.uniform() < xs1_low_prob_test else np.random.uniform(low=mid, high=high, size=useful_dim) for i in xrange(num_test)])

    xs2_low_prob_train = 0.5
    xs2_low_prob_test = 0.0

    xs2_train = np.array([np.random.uniform(low=low, high=mid, size=useful_dim) if np.random.uniform() < xs2_low_prob_train else np.random.uniform(low=mid, high=high, size=useful_dim) for i in xrange(num_train)])
    xs2_test = np.array([np.random.uniform(low=low, high=mid, size=useful_dim) if np.random.uniform() < xs2_low_prob_test else np.random.uniform(low=mid, high=high, size=useful_dim) for i in xrange(num_test)])

    xs_same_train = np.random.uniform(low=low, high=high, size=(num_train, same_dim))
    xs_same_test = np.random.uniform(low=low, high=high, size=(num_test, same_dim))

    xs_diff_low_prob_train = 0.9
    xs_diff_low_prob_test = 0.1

    xs_diff_train = np.array([np.random.uniform(low=low, high=mid, size=diff_dim) if np.random.uniform() < xs_diff_low_prob_train else np.random.uniform(low=mid, high=high, size=diff_dim) for i in xrange(num_train)])
    xs_diff_test = np.array([np.random.uniform(low=low, high=mid, size=diff_dim) if np.random.uniform() < xs_diff_low_prob_test else np.random.uniform(low=mid, high=high, size=diff_dim) for i in xrange(num_test)])

    corner_val = 1.
#    slope = corner_val / (useful_dim * 2)

    def f_helper_1(x):
        if x > mid:
            return x-mid - 0.25
        else:
            return mid-x - 0.25

#    alpha = 0.7
#    alpha = 0.2
#    alpha = 1.0
    def f_1(x):
        return alpha * corner_val * sum(map(f_helper_1, x)) / len(x)

    def f_helper_2(x):
        if x > mid:
            return x-mid - 0.25
        else:
            return mid-x - 0.25

    def f_2(x):
        return corner_val * sum(map(f_helper_2, x)) / len(x)

    ys_train_pre = np.array(map(f_1, xs1_train)) + np.array(map(f_2, xs2_train))
    ys_test_pre = np.array(map(f_1, xs1_test)) + np.array(map(f_2, xs2_test))


    noise_sd = .01

    ys_train = ys_train_pre + np.random.normal(loc=0., scale=noise_sd, size=num_train)
    ys_test = ys_test_pre + np.random.normal(loc=0., scale=noise_sd, size=num_test)

    xs_train = np.concatenate((xs1_train, xs2_train, xs_same_train, xs_diff_train), axis=1)
    xs_test = np.concatenate((xs1_test, xs2_test, xs_same_test, xs_diff_test), axis=1)

    return xs_train, xs_test, ys_train, ys_test
