import numpy as np
import pdb

def cholesky_LDL(M):
    assert M.shape[0] == M.shape[1]
    N = M.shape[0]
    M_orig = M
    M = M.copy()
    pivots = np.zeros(N, dtype=int)

    pivot = True
    
    for i in xrange(N):

        if pivot:
            pivots[i] = np.argmax(np.diag(M)[i:]) + i
            M[:,[i,pivots[i]]] = M[:,[pivots[i],i]]
            M[[i,pivots[i]],:] = M[[pivots[i],i],:]

        for s in xrange(i+1,N):
            for t in xrange(i+1,N):
                M[s,t] = M[s,t] - (M[i,t] * M[s,i] / M[i,i])

        for k in xrange(i+1,N):
            M[k,i] = M[k,i] / M[i,i]


            
    L = np.eye(N)
    for i in xrange(N):
        for j in xrange(i):
            L[i,j] = M[i,j]

    D = np.diag(np.diag(M))
            
    print 'M_orig'
    print M_orig
    print 'G'
    print np.dot(L,D**0.5)
    print 'approx'
    print np.dot(L,np.dot(D,L.T))
    print pivots


def cholesky_LL(M):
    assert M.shape[0] == M.shape[1]
    N = M.shape[0]
    M_orig = M
    M = M.copy()
    pivots = np.zeros(N, dtype=int)

    print 'M_orig'
    print M
    
    pivot = True

    shift = 0.0

    stop = 2
    
    for i in xrange(N):

        print 'iter'
        print i

        if i == stop:
#            assert False
            break
        
        print 'candidates'
        print np.diag(M)
        
        if pivot:
            pivots[i] = np.argmax(np.diag(M)[i:]) + i
            M[:,[i,pivots[i]]] = M[:,[pivots[i],i]]
            M[[i,pivots[i]],:] = M[[pivots[i],i],:]

            M_orig[:,[i,pivots[i]]] = M_orig[:,[pivots[i],i]]
            M_orig[[i,pivots[i]],:] = M_orig[[pivots[i],i],:]

        print 'pivot old'
        print M[i,i]

        M[i,i] -= shift

        if M[i,i] < 0:
            #assert False
            break
            
        for s in xrange(i+1,N):
            for t in xrange(i+1,N):
                M[s,t] = M[s,t] - (M[i,t] * M[s,i] / M[i,i])

        for k in xrange(i+1,N):
            M[k,i] = M[k,i] / (M[i,i]**0.5)

        M[i,i] = M[i,i]**0.5

        print M

    
    print M

    R = N
    
    L = np.zeros((N,N))
    for i in xrange(N):
        for j in xrange(min(i+1, R)):
            L[i,j] = M[i,j]

    print 'M_orig'
    print M_orig
    print 'G'
    print L
    print 'approx'
    print np.dot(L,L.T)
    print 'residual'
    print M_orig - np.dot(L,L.T)
    print 'approx eig'
    print np.linalg.eig(np.dot(L,L.T))
    print 'approx with correct diag eig'
    approx = np.dot(L,L.T)
    print approx
    for i in xrange(N):
        approx[i,i] = 1.#M_orig[i,i]
    print np.linalg.eig(approx)
    print pivots
    
