import numpy as np
import scipy
import kernels

def logreg_data(x_dim, u_dim, v_dim, num_train, num_test):
    # x is generated based on u coordinates.  v basis coordinates determines y 
    y_dim = 1
    u_perp_dim = x_dim - u_dim
    u_bases = np.eye(x_dim)[:,u_perp_dim:]
    if u_perp_dim > 0:
        u_perp_bases = np.eye(x_dim)[:,0:u_perp_dim]
        v_perp_dim = x_dim - v_dim
    v_start_dim = 1
    v_bases = np.eye(x_dim)[:,v_start_dim:v_start_dim+v_dim]

    # define train u distribution
    p_u_train_dist = scipy.stats.multivariate_normal(mean=0.*np.ones(shape=u_dim), cov=np.eye(u_dim)*1.)
    p_u_train_pdf = p_u_train_dist.pdf
    p_u_train_sample = p_u_train_dist.rvs

    # define test u distribution
    p_u_test_dist = scipy.stats.multivariate_normal(mean=-0.*np.ones(shape=u_dim), cov=np.eye(u_dim)*1.)
    p_u_test_pdf = p_u_test_dist.pdf
    p_u_test_sample = p_u_test_dist.rvs

    # define shared u_perp distribution
    if u_perp_dim > 0:
        #    p_u_perp_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_perp_dim), cov=np.eye(u_perp_dim)*0.049)
        p_u_perp_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_perp_dim))
        p_u_perp_pdf = p_u_perp_dist.pdf
        p_u_perp_sample = p_u_perp_dist.rvs

    def f_v(v): 
        if 1*np.ones(v_dim).dot(v) > 0:
            return 500
        else:
            return -500

    logistic = lambda x: 1. / (1. + np.exp(-x))
    p_y_given_v_sample = lambda v: int(np.random.random() < logistic(f_v(v)))

    # generate data
    expand = lambda s: s.reshape(len(s),1) if len(s.shape) == 1 else s
    us_train = expand(np.array([p_u_train_sample() for i in xrange(num_train)]))
    us_test = expand(np.array([p_u_test_sample() for i in xrange(num_test)]))
    if u_perp_dim > 0:
        us_perp_train = expand(np.array([p_u_perp_sample() for i in xrange(num_train)]))
        us_perp_test = expand(np.array([p_u_perp_sample() for i in xrange(num_test)]))
    xs_train = np.dot(us_train, u_bases.T)
    xs_test = np.dot(us_test, u_bases.T)
    if u_perp_dim > 0:
        xs_train += np.dot(us_perp_train, u_perp_bases.T)
        xs_test += np.dot(us_perp_test, u_perp_bases.T)
    vs_train = np.dot(xs_train, v_bases)
    vs_test = np.dot(xs_test, v_bases)
    ys_train = np.array([p_y_given_v_sample(v) for v in vs_train])
    ys_test = np.array([p_y_given_v_sample(v) for v in vs_test])

    return xs_train, xs_test, ys_train, ys_test

def why_weighting_is_important_data(x_dim, num_train, num_test):

    u_dim = 1
    v_dim = 2
    the_u_base = np.zeros(x_dim)
    the_u_base[0] = 1.
    the_u_base[1] = 1.
    u_bases = kernels.ortho(np.array([the_u_base]).T)
    the_u_perp_base = np.zeros(x_dim)
    the_u_perp_base[0] = -1.
    the_u_perp_base[1] = 1.
    u_perp_bases = kernels.ortho(np.array([the_u_perp_base]).T)
    u_remaining_dim = x_dim - 2
    if u_remaining_dim > 0:
        u_remaining_bases = np.eye(x_dim)[:,2:2+u_remaining_dim]
    v_bases = np.zeros((x_dim,2))
    v_bases[0,0] = 1.
    v_bases[1,1] = 1.

    def p_u_train_sample():
        if np.random.uniform() < 0.15:
            return scipy.stats.uniform(loc=0, scale=2**.5).rvs()
        else:
            return scipy.stats.uniform(loc=2**.5, scale=8.**.5 - 2**.5).rvs()

    p_u_test_dist = scipy.stats.uniform(loc=0., scale=2.**.5)
    p_u_test_pdf = p_u_test_dist.pdf
    p_u_test_sample = p_u_test_dist.rvs

    p_u_perp_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=1), cov=np.eye(1)*0.049)
    p_u_perp_pdf = p_u_perp_dist.pdf
    p_u_perp_sample = p_u_perp_dist.rvs

    if u_remaining_dim > 0:
        p_u_remaining_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_remaining_dim), cov=np.eye(u_remaining_dim)*0.049)
        p_u_remaining_pdf = p_u_remaining_dist.pdf
        p_u_remaining_sample = p_u_remaining_dist.rvs
    
    small_var = 0.00002
    big_var = 0.00002
    y_dim = 1
    f_v1 = lambda v1: 1.*v1 if v1 < 1. else 1.
    p_y1_noise_sample = lambda v1: scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*small_var).rvs() if v1 < 1. else scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*big_var).rvs()
    p_y1_given_v1_sample = lambda v1: f_v1(v1) + p_y1_noise_sample(v1)
    f_v2 = lambda v2: 1.*v2 if v2 >= 1. else 1.
    p_y2_noise_sample = lambda v2: scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*big_var).rvs() if v2 < 1. else scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*small_var).rvs()
    p_y2_given_v2_sample = lambda v2: f_v2(v2) + p_y2_noise_sample(v2)

    # generate data
    expand = lambda s: s.reshape(len(s),1) if len(s.shape) == 1 else s
    us_train = expand(np.array([p_u_train_sample() for i in xrange(num_train)]))
    us_test = expand(np.array([p_u_test_sample() for i in xrange(num_test)]))
    us_perp_train = expand(np.array([p_u_perp_sample() for i in xrange(num_train)]))
    us_perp_test = expand(np.array([p_u_perp_sample() for i in xrange(num_test)]))
    if u_remaining_dim > 0:
        us_remaining_train = expand(np.array([p_u_remaining_sample() for i in xrange(num_train)]))
        us_remaining_test = expand(np.array([p_u_remaining_sample() for i in xrange(num_test)]))
    xs_train = np.dot(us_train, u_bases.T)
    xs_test = np.dot(us_test, u_bases.T)
    xs_train += np.dot(us_perp_train, u_perp_bases.T)
    xs_test += np.dot(us_perp_test, u_perp_bases.T)
    if u_remaining_dim > 0:
        xs_train += np.dot(us_remaining_train, u_remaining_bases.T)
        xs_test += np.dot(us_remaining_test, u_remaining_bases.T)
    
    v1_base = v_bases[:,0]
    v1s_train = np.dot(xs_train, v1_base)
    v1s_test = np.dot(xs_test, v1_base)
    v2_base = v_bases[:,1]
    v2s_train = np.dot(xs_train, v2_base)
    v2s_test = np.dot(xs_test, v2_base)
    y1s_train = np.array([p_y1_given_v1_sample(v1) for v1 in v1s_train])
    y1s_test = np.array([p_y1_given_v1_sample(v1) for v1 in v1s_test])
    y2s_train = np.array([p_y2_given_v2_sample(v2) for v2 in v2s_train])
    y2s_test = np.array([p_y2_given_v2_sample(v2) for v2 in v2s_test])
    ys_train = y1s_train + y2s_train
    ys_test = y1s_test + y2s_test

    return xs_train, xs_test, ys_train, ys_test
