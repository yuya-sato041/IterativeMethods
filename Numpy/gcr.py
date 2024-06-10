import numpy as np

from .common import Timer

# The Conjugate Gradient Method(CG) function
def gcr(A, b, n, maxiter, x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial value x0
    if x0:
        x = x0
    else:
        x = np.zeros(n, T)

    #-- residual
    residual = np.zeros(maxiter+1, T)
    
    #-- timer
    timer = Timer()
    timer.start()

    #-- initial value calculation
    r = b - np.dot(A, x)
    p = np.zeros((maxiter+1, n), T)
    p[0] = np.copy(r)
    ap = np.zeros((maxiter+1, n), T)
    ap[0] = np.dot(A, p[0])
    bnorm = np.linalg.norm(b)

    ap2 = np.zeros(maxiter, T)
    
    #-- iteration
    for i in range(maxiter):
        #-- apennd resd to list
        res = np.linalg.norm(r) / bnorm
        residual[i] = res
        
        #-- convergence check
        if res <= tol:
            break
        
        #-- omega
        omega = np.dot(ap[i], r)
        
        #-- sigma
        sigma = np.dot(ap[i], ap[i])
        ap2[i] = sigma

        #-- alpha                    
        alpha = omega / sigma

        #-- x
        x += alpha * p[i]

        #-- residual
        r -= alpha * ap[i]

        #-- beta
        ar = np.dot(A, r)
        beta = -np.dot(ap[:i+1], ar) / ap2[:i+1]
        beta = beta.reshape((i+1, 1))
        
        #-- p
        bp = beta * p[:i+1]
        p[i+1] = r + np.sum(bp, axis=0)
        
        #-- ap
        bap = beta * ap[:i+1]
        ap[i+1] = ar + np.sum(bap, axis=0)
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "GCR",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info