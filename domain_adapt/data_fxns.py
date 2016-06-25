import numpy as np
from collections import namedtuple
import itertools

def twin_peaks_data(sign, noise_dim):
    
    if sign == 0:
        case = np.random.choice([-1,1])
    else:
        case = sign

    if case == 1:
        if np.random.uniform() < 0.5:
            r = np.random.uniform(low=0.,high=np.pi)
            s = np.random.uniform(low=0.,high=np.pi)
        else:
            r = np.random.uniform(low=-np.pi,high=0.)
            s = np.random.uniform(low=-np.pi,high=0.)
    else:
        if np.random.uniform() < 0.5:
            r = np.random.uniform(low=0.,high=np.pi)
            s = np.random.uniform(low=-np.pi,high=0)
        else:
            r = np.random.uniform(low=-np.pi,high=0.)
            s = np.random.uniform(low=0.,high=np.pi)

    return np.concatenate((np.array([r,s,np.sin(r)*np.tanh(s)]), np.random.uniform(low=-np.pi,high=np.pi,size=noise_dim)))

def fig_2_train_test_data(N_train, N_test, z_1_prob_train, z_1_prob_test, noise_dim):
    """
    returns train data, weights, test data
    """
    z_0_prob_train = 1. - z_1_prob_train
    p_z_sample_train = lambda : int(np.random.uniform() < z_1_prob_train)

    p_x_given_z_sample = lambda z: twin_peaks_data(1 if z==1 else -1, noise_dim)
    p_y_given_x_z_sample = lambda x,z: x[z]**2
    
    zs_train = np.array([p_z_sample_train() for i in xrange(N_train)])
    xs_train = np.array([p_x_given_z_sample(z) for z in zs_train])
    ys_train = np.array([p_y_given_x_z_sample(x,z) for (x,z) in itertools.izip(xs_train,zs_train)])

    z_0_prob_test = 1. - z_1_prob_test
    p_z_sample_test = lambda : int(np.random.uniform() < z_1_prob_test)

    zs_test = np.array([p_z_sample_test() for i in xrange(N_test)])
    xs_test = np.array([p_x_given_z_sample(z) for z in zs_test])
    ys_test = np.array([p_y_given_x_z_sample(x,z) for (x,z) in itertools.izip(xs_test,zs_test)])

    ws_train = np.array([z_0_prob_test / z_0_prob_train if z == 0 else z_1_prob_test / z_1_prob_train for z in zs_train])

    return xs_train, ys_train, ws_train, xs_test, ys_test

def pca(xs, k, normalize=True):
    xs = xs - xs.mean(axis=0)
    if normalize:
        xs = xs / np.std(xs,axis=0)
    eig_vals, eig_vecs = np.linalg.eig(xs.T.dot(xs))
    sorted_eigvals, sorted_eigvecs = map(np.array,zip(*sorted(zip(eig_vals, eig_vecs.T),key=lambda (eig_val,eig_vec):np.abs(eig_val),reverse=True)))
    keep_eigvecs = sorted_eigvecs[0:k,:]
#    print sorted_eigvecs
    return (xs.dot(keep_eigvecs.T))#.dot(keep_eigvecs)

def sample_data(xs, ys, sample_f):
    N = xs.shape[0]
    us = pca(xs,1,True)[:,0]
    training_prob = np.array(map(sample_f,us))
    test_prob = 1. - training_prob
    in_training = np.random.uniform(size=N) < training_prob
    in_test = np.logical_not(in_training)
    ws = test_prob / training_prob
    return xs[in_training,:],ys[in_training],ws[in_training],xs[in_test,:],ys[in_test]
    

###
# below, some fxns that return split data along with true weights
###


def split_whitewine_data():

    import data.mushroom.mushroom.fxns as uci_data
    
    xs, ys = uci_data.whitewine_data()

    keep = 1000
    
    xs = xs[0:keep,:]
    ys = ys[0:keep]
    
    us = pca(xs,1,True)[:,0]
    high = np.percentile(us,95)
    low = np.percentile(us,0.1)

    xs = xs-xs.mean(axis=0)
    xs = xs / xs.std(axis=0)
    
    def p_s(u):
        return max(min((1/(high-low)*(u-low),0.8)),0.2)
    
    return sample_data(xs,ys,p_s)
