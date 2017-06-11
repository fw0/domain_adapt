import numpy as np
import scipy
import kernels
import python_utils.python_utils.basic as basic
import matplotlib.pyplot as plt
import domain_adapt.domain_adapt.new.utils as utils
import pdb

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

def why_weighting_is_important_data(x_dim, num_train, num_test, proportion=0.5, axis_boundaries=np.cumsum(np.arange(7))):

    verbose = False
    
    u_dim = 1
    #v_dim = 2
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

#    def p_u_train_sample():
#        if np.random.uniform() < proportion:
#            return scipy.stats.uniform(loc=0, scale=2**.5).rvs()
#        else:
#            return scipy.stats.uniform(loc=2**.5, scale=8.**.5 - 2**.5).rvs()

    num_sections = 3
#    axis_boundaries = np.arange(0, (num_sections*2)+1)
    #axis_boundaries = np.array([0,1,3,6,10,15,21])
    assert len(axis_boundaries) % 2 == 1
    boundaries = axis_boundaries * (2**0.5)

#    train_us = np.random.uniform(low=0., =boundaries[-1], size=num_train)

    def boundary_sample(left_boundaries, right_boundaries, num):
        if verbose: print left_boundaries, right_boundaries
        assert len(right_boundaries) == len(left_boundaries)
        widths = right_boundaries - left_boundaries
        if verbose: print widths, 'widths'
        if verbose: print np.cumsum(widths), 'cumsum'
        support_width = np.sum(widths)
        if verbose: print support_width, 'support_width'
        support_coordinates = np.random.uniform(low=0., high=support_width, size=num)
        #print support_coordinates[0:20], 'support_coordinates'
        support_segment_paddings = np.array([0.])
        support_segment_paddings = np.append(support_segment_paddings,np.cumsum(widths))
        if verbose: print support_segment_paddings, 'support_segment_paddings'
        gts = support_coordinates[:,np.newaxis] > support_segment_paddings[0:-1]
#        lts = support_coordinates[:,np.newaxis] <= support_segment_paddings + widths[0]
        lts = support_coordinates[:,np.newaxis] <= support_segment_paddings[1:]
        #print gts[0:20]
        #print lts[0:20]
        #print gts.astype(int) + lts.astype(int)
        in_segments = (gts.astype(int) + lts.astype(int)) == 2
        #print in_segments
        #print gts[0:20]
        #print lts[0:20]
        #print (gts.astype(int) + lts.astype(int))[0:20]
        assert np.all(np.sum(in_segments, axis=1) == np.ones(num))
        which_segment = np.argmax(in_segments, axis=1)
#        which_segment = np.argmax(support_coordinates[:,np.newaxis] > support_segment_paddings, axis=1)
        #print which_segment[0:20], 'which_segment'
        offset = support_coordinates - support_segment_paddings[which_segment]
        #print offset[0:20], 'offset'
        samples = left_boundaries[which_segment] + offset
        if verbose:
            fig, ax = plt.subplots()
            ax.hist(samples,bins=10)
            basic.display_fig_inline(fig)
        return samples
    

    #p_u_test_dist = scipy.stats.uniform(loc=0., scale=2.**.5)
    #p_u_test_pdf = p_u_test_dist.pdf
    #p_u_test_sample = p_u_test_dist.rvs

    perp_noise = 0.08
    p_u_perp_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=1), cov=np.eye(1)*perp_noise)
    p_u_perp_pdf = p_u_perp_dist.pdf
    p_u_perp_sample = p_u_perp_dist.rvs

    #remaining_noise = 0.049
    remaining_noise = 1.049
    if u_remaining_dim > 0:
        p_u_remaining_train_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_remaining_dim), cov=np.eye(u_remaining_dim)*remaining_noise)
        p_u_remaining_train_pdf = p_u_remaining_train_dist.pdf
        p_u_remaining_train_sample = p_u_remaining_train_dist.rvs

        p_u_remaining_test_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_remaining_dim), cov=np.eye(u_remaining_dim)*remaining_noise)
        p_u_remaining_test_pdf = p_u_remaining_test_dist.pdf
        p_u_remaining_test_sample = p_u_remaining_test_dist.rvs
    
    small_var = 0.00002
    big_var = 0.00002
    y_dim = 1
    #f_v1 = lambda v1: 1.*v1 if v1 < 1. else 1.
    v1_constant_right_axis_boundaries = axis_boundaries[2::2]
    v1_constant_left_axis_boundaries = axis_boundaries[1:-1:2]
    if verbose: print v1_constant_right_axis_boundaries, 'v1_constant_right_axis_boundaries'
    if verbose: print v1_constant_left_axis_boundaries, 'v1_constant_left_axis_boundaries'
#    pdb.set_trace()
    v1_sloped_right_axis_boundaries = axis_boundaries[1:-1:2]
    v1_sloped_left_axis_boundaries = axis_boundaries[0:-2:2]
    def f_v1(v1):

        if v1 >= v1_constant_right_axis_boundaries[-1]:
            return (v1_sloped_right_axis_boundaries - v1_sloped_left_axis_boundaries).sum() + v1 - v1_sloped_right_axis_boundaries[-1]
        if v1 < v1_sloped_left_axis_boundaries[0]:
            return 0

        
        rs = v1 < v1_constant_right_axis_boundaries
        ls = v1 >= v1_constant_left_axis_boundaries
#        print np.array([rs,ls])
#        pdb.set_trace()
        in_segment = np.all(np.array([rs,ls]), axis=0)
        if np.sum(in_segment) > 0:
#            assert False

            try:
                assert np.sum(in_segment) == 1
            except:
                print v1
                print rs, v1_constant_right_axis_boundarie
                print ls, v1_constant_left_axis_boundaries
                print in_segment
                pdb.set_trace()
            which_segment = np.argmax(in_segment)
            return (v1_sloped_right_axis_boundaries-v1_sloped_left_axis_boundaries)[0:which_segment+1].sum()
        else:
            rs = v1 < v1_sloped_right_axis_boundaries
            ls = v1 >= v1_sloped_left_axis_boundaries
            in_segment = np.all(np.array([rs,ls]), axis=0)
#            print rs, v1_sloped_right_axis_boundaries
#            print ls, v1_sloped_left_axis_boundaries
#            print in_segment
#            print v1
            try:
                assert np.sum(in_segment) == 1
            except:
                print v1 
                print rs, v1_sloped_right_axis_boundaries
                print ls, v1_sloped_left_axis_boundaries
                print in_segment
                pdb.set_trace()
            which_segment = np.argmax(in_segment)
            return (v1_sloped_right_axis_boundaries - v1_sloped_left_axis_boundaries)[0:which_segment].sum() + v1 - v1_sloped_left_axis_boundaries[which_segment]
        
    p_y1_noise_sample = lambda v1: scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*small_var).rvs() if v1 < 1. else scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*big_var).rvs()
    p_y1_given_v1_sample = lambda v1: f_v1(v1) + p_y1_noise_sample(v1)
    #f_v2 = lambda v2: 1.*v2 if v2 >= 1. else 1.
    v2_constant_right_axis_boundaries = axis_boundaries[1:-1:2]
    v2_constant_left_axis_boundaries = axis_boundaries[0:-2:2]
    v2_sloped_right_axis_boundaries = axis_boundaries[2::2]
    v2_sloped_left_axis_boundaries = axis_boundaries[1:-1:2]
    def f_v2(v2):

        if v2 < v2_constant_left_axis_boundaries[0]:
            return v2 - v2_constant_left_axis_boundaries[0]
        if v2 >= v2_sloped_right_axis_boundaries[-1]:
            return (v2_sloped_right_axis_boundaries - v2_sloped_left_axis_boundaries).sum()
        
        rs = v2 < v2_constant_right_axis_boundaries
        ls = v2 >= v2_constant_left_axis_boundaries
        in_segment = np.all(np.array([rs,ls]), axis=0)
        if np.sum(in_segment) > 0:
            try:
                assert np.sum(in_segment) == 1
            except:
                print v2
                print rs, v2_constant_right_axis_boundaries
                print ls, v2_constant_left_axis_boundaries
                print in_segment
                pdb.set_trace()
            which_segment = np.argmax(in_segment)
            #print v2, which_segment, 'ASDF', in_segment
            return (v2_sloped_right_axis_boundaries - v2_sloped_left_axis_boundaries)[0:which_segment].sum()
        else:
            rs = v2 < v2_sloped_right_axis_boundaries
            ls = v2 >= v2_sloped_left_axis_boundaries
            in_segment = np.all(np.array([rs,ls]), axis=0)
#            print rs
#            print ls
#            print in_segment
            try:
                assert np.sum(in_segment) == 1
            except:
                print v2
                print rs, v2_sloped_right_axis_boundaries
                print ls, v2_sloped_left_axis_boundaries
                print in_segment
                pdb.set_trace()
            which_segment = np.argmax(in_segment)
            return (v2_sloped_right_axis_boundaries - v2_sloped_left_axis_boundaries)[0:which_segment].sum() + v2 - v2_sloped_left_axis_boundaries[which_segment]
    p_y2_noise_sample = lambda v2: scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*big_var).rvs() if v2 < 1. else scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*small_var).rvs()
    p_y2_given_v2_sample = lambda v2: f_v2(v2) + p_y2_noise_sample(v2)

    # generate data
    expand = lambda s: s.reshape(len(s),1) if len(s.shape) == 1 else s
    #us_train = expand(us_train)
    #us_test = expand(us_test)
    num_train_overlap = int(proportion * num_train)
    num_train_useless = num_train - num_train_overlap
    us_train_overlap = boundary_sample(boundaries[0:-2:2], boundaries[1:-1:2], num_train_overlap)
    us_train_useless = boundary_sample(boundaries[1:-1:2], boundaries[2::2], num_train_useless)
    if verbose:
        fig,ax = plt.subplots()
        ax.hist(us_train_overlap)
        ax.hist(us_train_useless)
        basic.display_fig_inline(fig)
    #pdb.set_trace()
    us_train = expand(np.concatenate((us_train_overlap, us_train_useless), axis=0))
    if verbose:
        fig,ax = plt.subplots()
        ax.hist(us_train[:,0])
        ax.set_title('all')
        basic.display_fig_inline(fig)
    
    us_test = expand(boundary_sample(boundaries[0:-2:2], boundaries[1:-1:2], num_test))
    #us_train = expand(np.array([p_u_train_sample() for i in xrange(num_train)]))
    #us_test = expand(np.array([p_u_test_sample() for i in xrange(num_test)]))
    us_perp_train = expand(np.array([p_u_perp_sample() for i in xrange(num_train)]))
    us_perp_test = expand(np.array([p_u_perp_sample() for i in xrange(num_test)]))
    if u_remaining_dim > 0:
        us_remaining_train = expand(np.array([p_u_remaining_train_sample() for i in xrange(num_train)]))
        us_remaining_test = expand(np.array([p_u_remaining_test_sample() for i in xrange(num_test)]))
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
    if verbose:
        fig,ax = plt.subplots()
        ax.scatter(v1s_train,y1s_train)
        basic.display_fig_inline(fig)
    y1s_test = np.array([p_y1_given_v1_sample(v1) for v1 in v1s_test])
    y2s_train = np.array([p_y2_given_v2_sample(v2) for v2 in v2s_train])
    if verbose:
        fig,ax = plt.subplots()
        ax.scatter(v2s_train,y2s_train)
        basic.display_fig_inline(fig)
    y2s_test = np.array([p_y2_given_v2_sample(v2) for v2 in v2s_test])
    ys_train = y1s_train + y2s_train
    ys_test = y1s_test + y2s_test

    return xs_train, xs_test, ys_train-1., ys_test-1.


def v_data(x_dim, num_train, num_test):

    def f(x):
        if x > 0.:
            y = x
        else:
            y = -x
        return y

    scale = 0.1

#    xs_train = np.random.normal(loc=-1., scale=0.8, size=(num_train,1))
    xs_train = np.random.uniform(low=-2., high=2., size=(num_train,))
    xs_test = np.random.normal(loc=1., scale=0.8, size=(num_test,))
    ys_train = np.array([f(x) + np.random.normal(scale=scale) for x in xs_train])
    ys_test = np.array([f(x) + np.random.normal(scale=scale) for x in xs_test])

    xs_train = np.concatenate((xs_train.reshape(num_train,1), np.random.normal(size=(num_train,x_dim-1))), axis=1)
    xs_test = np.concatenate((xs_test.reshape(num_test,1), np.random.normal(size=(num_test,x_dim-1))), axis=1)

    return xs_train, xs_test, ys_train, ys_test

    
class data(object):

    train_color = 'r'
    test_color = 'b'
    
    @classmethod
    def plot_ys_v_xs(cls, plot_dim, xs_train, xs_test, ys_train, ys_test, ax=None, B=None):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel('x_%d' % plot_dim)
        ax.set_ylabel('y')
        if not (B is None):
            if len(B.shape) == 1:
                #B = utils.ortho(B.reshape((len(B),1)))
                B = B.reshape((len(B),1))
            xs_train = np.dot(xs_train, B)
            xs_test = np.dot(xs_test, B)
        s = 1
        alpha = 0.3
        ax.scatter(xs_train[:,plot_dim], ys_train, color=cls.train_color, label='train',s=s, alpha=alpha)
        ax.scatter(xs_test[:,plot_dim], ys_test, color=cls.test_color, label='test',s=s, alpha=alpha)
        try:
            ax.set_title('%d %s' % (plot_dim, B))
        except:
            pass
        ax.legend()
#        fig.tight_layout()
        if not (fig is None):
            basic.display_fig_inline(fig)

    @classmethod
    def plot_us(cls, B, xs_train, xs_test, ax=None, normed=False):
        fig = None
        if ax is None:
            fig, ax = plt.subplots()
        ax.set_xlabel('B\'x')
        s = 1.
        alpha = 0.5
        ax.scatter(np.dot(xs_train, B), np.zeros(len(xs_train))-1, color=cls.train_color, label='train', s=s, alpha=alpha)
        ax.scatter(np.dot(xs_test, B), np.zeros(len(xs_test))-1, color=cls.test_color, label='test', s=s, alpha=alpha)
        alpha = 0.5
        bins = 20
        ax.hist(np.dot(xs_train, B), color=cls.train_color, alpha=alpha, bins=bins, normed=normed)
        ax.hist(np.dot(xs_test, B), color=cls.test_color, alpha=alpha, bins=bins, normed=normed)
        ax.legend()
        if not fig is None:
            fig.tight_layout()
        if not (fig is None):
            basic.display_fig_inline(fig)

class random_data(data):

    def __init__(self, x_dim, shift=1.):
        self.x_dim, self.shift = x_dim, shift

        # define train distribution
        self.train_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=self.x_dim))
        self.train_pdf = self.train_dist.pdf
        self.train_sample = self.train_dist.rvs

        # define test distribution
        self.test_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=self.x_dim) + self.shift)
        self.test_pdf = self.test_dist.pdf
        self.test_sample = self.test_dist.rvs
        
    def sample(self, num_train, num_test):
        xs_train = np.array([self.train_sample() for i in xrange(num_train)])
        xs_test = np.array([self.test_sample() for i in xrange(num_test)])
        ys_train = np.random.normal(size=num_train)
        ys_test = np.random.normal(size=num_test)
        return xs_train, xs_test, ys_train, ys_test

    
def better_v_data(x_dim, num_train, num_test, proportion=0.5, axis_boundaries=np.cumsum(np.arange(7))):

    verbose = False
    
    u_dim = 1
    #v_dim = 2
    the_u_base = np.zeros(x_dim)
    the_u_base[0] = 1.
    the_u_base[1] = 0.
    u_bases = kernels.ortho(np.array([the_u_base]).T)
    the_u_perp_base = np.zeros(x_dim)
    the_u_perp_base[0] = 0.
    the_u_perp_base[1] = 1.
    u_perp_bases = kernels.ortho(np.array([the_u_perp_base]).T)
    u_remaining_dim = x_dim - 2
    if u_remaining_dim > 0:
        u_remaining_bases = np.eye(x_dim)[:,2:2+u_remaining_dim]
    v_bases = np.zeros((x_dim,1))
    v_bases[0,0] = 1.
#    v_bases[1,1] = 1.

#    def p_u_train_sample():
#        if np.random.uniform() < proportion:
#            return scipy.stats.uniform(loc=0, scale=2**.5).rvs()
#        else:
#            return scipy.stats.uniform(loc=2**.5, scale=8.**.5 - 2**.5).rvs()

    num_sections = 3
#    axis_boundaries = np.arange(0, (num_sections*2)+1)
    #axis_boundaries = np.array([0,1,3,6,10,15,21])
    assert len(axis_boundaries) % 2 == 1
#    boundaries = axis_boundaries * (2**0.5)

    boundaries = axis_boundaries
#    train_us = np.random.uniform(low=0., =boundaries[-1], size=num_train)

    def boundary_sample(left_boundaries, right_boundaries, num):
        if verbose: print left_boundaries, right_boundaries
        assert len(right_boundaries) == len(left_boundaries)
        widths = right_boundaries - left_boundaries
        if verbose: print widths, 'widths'
        if verbose: print np.cumsum(widths), 'cumsum'
        support_width = np.sum(widths)
        if verbose: print support_width, 'support_width'
        support_coordinates = np.random.uniform(low=0., high=support_width, size=num)
        #print support_coordinates[0:20], 'support_coordinates'
        support_segment_paddings = np.array([0.])
        support_segment_paddings = np.append(support_segment_paddings,np.cumsum(widths))
        if verbose: print support_segment_paddings, 'support_segment_paddings'
        gts = support_coordinates[:,np.newaxis] > support_segment_paddings[0:-1]
#        lts = support_coordinates[:,np.newaxis] <= support_segment_paddings + widths[0]
        lts = support_coordinates[:,np.newaxis] <= support_segment_paddings[1:]
        #print gts[0:20]
        #print lts[0:20]
        #print gts.astype(int) + lts.astype(int)
        in_segments = (gts.astype(int) + lts.astype(int)) == 2
        #print in_segments
        #print gts[0:20]
        #print lts[0:20]
        #print (gts.astype(int) + lts.astype(int))[0:20]
        assert np.all(np.sum(in_segments, axis=1) == np.ones(num))
        which_segment = np.argmax(in_segments, axis=1)
#        which_segment = np.argmax(support_coordinates[:,np.newaxis] > support_segment_paddings, axis=1)
        #print which_segment[0:20], 'which_segment'
        offset = support_coordinates - support_segment_paddings[which_segment]
        #print offset[0:20], 'offset'
        samples = left_boundaries[which_segment] + offset
        if verbose:
            fig, ax = plt.subplots()
            ax.hist(samples,bins=10)
            basic.display_fig_inline(fig)
        return samples
    

    #p_u_test_dist = scipy.stats.uniform(loc=0., scale=2.**.5)
    #p_u_test_pdf = p_u_test_dist.pdf
    #p_u_test_sample = p_u_test_dist.rvs

    perp_noise = 0.08
    p_u_perp_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=1), cov=np.eye(1)*perp_noise)
    p_u_perp_pdf = p_u_perp_dist.pdf
    p_u_perp_sample = p_u_perp_dist.rvs

    #remaining_noise = 0.049
    remaining_noise = 1.049
    if u_remaining_dim > 0:
        p_u_remaining_train_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_remaining_dim), cov=np.eye(u_remaining_dim)*remaining_noise)
        p_u_remaining_train_pdf = p_u_remaining_train_dist.pdf
        p_u_remaining_train_sample = p_u_remaining_train_dist.rvs

        p_u_remaining_test_dist = scipy.stats.multivariate_normal(mean=np.zeros(shape=u_remaining_dim), cov=np.eye(u_remaining_dim)*remaining_noise)
        p_u_remaining_test_pdf = p_u_remaining_test_dist.pdf
        p_u_remaining_test_sample = p_u_remaining_test_dist.rvs
    
    small_var = 0.00002
    big_var = 0.00002
    y_dim = 1
    right_boundaries = boundaries[1:]
    left_boundaries = boundaries[0:-1]
    #f_v1 = lambda v1: 1.*v1 if v1 < 1. else 1.
#    v1_constant_right_axis_boundaries = axis_boundaries[2::2]
#    v1_constant_left_axis_boundaries = axis_boundaries[1:-1:2]
#    if verbose: print v1_constant_right_axis_boundaries, 'v1_constant_right_axis_boundaries'
#    if verbose: print v1_constant_left_axis_boundaries, 'v1_constant_left_axis_boundaries'
#    pdb.set_trace()
#    v1_sloped_right_axis_boundaries = axis_boundaries[1:-1:2]
#    v1_sloped_left_axis_boundaries = axis_boundaries[0:-2:2]
    def f_v1(v1):

        up = None
        
        if v1 >= boundaries[-1]:
            up = True
        if v1 <= boundaries[0]:
            up = False

        
        rs = v1 < right_boundaries
        ls = v1 >= left_boundaries
#        print np.array([rs,ls])
#        pdb.set_trace()
        in_segment = np.all(np.array([rs,ls]), axis=0)
        if np.sum(in_segment) > 0:
#            assert False

            try:
                assert np.sum(in_segment) == 1
            except:
                print v1
                print rs, v1_constant_right_axis_boundarie
                print ls, v1_constant_left_axis_boundaries
                print in_segment
                pdb.set_trace()
            which_segment = np.argmax(in_segment)
            up = ((which_segment % 2) == 0)
            if up:
                return 1. * v1
            else:
                return -1. * v1


        
    p_y1_noise_sample = lambda v1: scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*small_var).rvs() if v1 < 1. else scipy.stats.multivariate_normal(mean=np.zeros(shape=y_dim),cov=np.eye(y_dim)*big_var).rvs()
    p_y1_given_v1_sample = lambda v1: f_v1(v1) + p_y1_noise_sample(v1)


    # generate data
    expand = lambda s: s.reshape(len(s),1) if len(s.shape) == 1 else s
    #us_train = expand(us_train)
    #us_test = expand(us_test)
    num_train_overlap = int(proportion * num_train)
    num_train_useless = num_train - num_train_overlap
    us_train_overlap = boundary_sample(boundaries[0:-2:2], boundaries[1:-1:2], num_train_overlap)
    us_train_useless = boundary_sample(boundaries[1:-1:2], boundaries[2::2], num_train_useless)
    if verbose:
        fig,ax = plt.subplots()
        ax.hist(us_train_overlap)
        ax.hist(us_train_useless)
        basic.display_fig_inline(fig)
    #pdb.set_trace()
    us_train = expand(np.concatenate((us_train_overlap, us_train_useless), axis=0))
    if verbose:
        fig,ax = plt.subplots()
        ax.hist(us_train[:,0])
        ax.set_title('all')
        basic.display_fig_inline(fig)
    
    us_test = expand(boundary_sample(boundaries[0:-2:2], boundaries[1:-1:2], num_test))
    #us_train = expand(np.array([p_u_train_sample() for i in xrange(num_train)]))
    #us_test = expand(np.array([p_u_test_sample() for i in xrange(num_test)]))
    us_perp_train = expand(np.array([p_u_perp_sample() for i in xrange(num_train)]))
    us_perp_test = expand(np.array([p_u_perp_sample() for i in xrange(num_test)]))
    if u_remaining_dim > 0:
        us_remaining_train = expand(np.array([p_u_remaining_train_sample() for i in xrange(num_train)]))
        us_remaining_test = expand(np.array([p_u_remaining_test_sample() for i in xrange(num_test)]))
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
#    v2_base = v_bases[:,1]
#    v2s_train = np.dot(xs_train, v2_base)
#    v2s_test = np.dot(xs_test, v2_base)
    y1s_train = np.array([p_y1_given_v1_sample(v1) for v1 in v1s_train])
    if verbose:
        fig,ax = plt.subplots()
        ax.scatter(v1s_train,y1s_train)
        basic.display_fig_inline(fig)
    y1s_test = np.array([p_y1_given_v1_sample(v1) for v1 in v1s_test])
#    y2s_train = np.array([p_y2_given_v2_sample(v2) for v2 in v2s_train])
#    if verbose:
#        fig,ax = plt.subplots()
#        ax.scatter(v2s_train,y2s_train)
#        basic.display_fig_inline(fig)
#    y2s_test = np.array([p_y2_given_v2_sample(v2) for v2 in v2s_test])
#    ys_train = y1s_train + y2s_train
#    ys_test = y1s_test + y2s_test
    ys_train = y1s_train
    ys_test = y1s_test

    return xs_train, xs_test, ys_train, ys_test
