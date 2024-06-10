import numpy as np

from .common import Timer

# The Conjugate Gradient Method(CG) function
def cg(A, b, n, maxiter, x0=None, tol=1e-10, T=np.float64) -> tuple:
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
    p = np.copy(r)
    rr = np.dot(r, r)
    bnorm = np.linalg.norm(b)
    

    #-- iteration
    for i in range(maxiter):

        #-- apennd resd to list
        res = np.linalg.norm(r) / bnorm
        residual[i] = res
        #-- convergence check
        if res <= tol:
            break  
        
        #-- ap
        ap =  np.dot(A, p)
        
        #-- sgma
        sigma = np.dot(p, ap)

        #-- alpha                    
        alpha = rr / sigma

        #-- x
        x += alpha * p

        #-- residual
        r -= alpha * ap
        
        #-- rr
        old_rr = np.copy(rr)
        rr = np.dot(r, r)

        #-- beta
        beta = rr / old_rr

        #-- d
        p = r + beta * p
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "CG",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info