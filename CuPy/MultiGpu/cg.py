import numpy as np
import cupy as cp

from ..common import Timer, MultiGpu

# The Conjugate Gradient Method(CG) function
def cg(A, b, n, maxiter, x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial
    if x0:
        x = cp.array(x0)
    else:
        x = cp.zeros(n, dtype=T)
    b = cp.array(b)
    MultiGpu.init()
    MultiGpu.alloc(A, b, T)

    #-- residual
    residual = cp.zeros(maxiter+1, T)
    
    #-- timer
    timer = Timer()
    timer.start()

    #-- initial value calculation
    r = b - MultiGpu.dot(A, x)
    p = cp.copy(r)
    rr = cp.dot(r, r)
    bnorm = cp.linalg.norm(b)

    #-- iteration
    for i in range(maxiter):

         #-- apennd resd to list
        res = cp.linalg.norm(r) / bnorm
        residual[i] = res
        #-- convergence check
        if res <= tol:
            break  

        #-- ap
        ap =  MultiGpu.dot(A, p)
        
        #-- sgma
        sigma = cp.dot(p, ap)
        
        #-- alpha                    
        alpha = rr / sigma

        #-- x
        x += alpha * p

        #-- residual
        r -= alpha * ap  
        
        #-- rr
        old_rr = cp.copy(rr)
        rr = cp.dot(r, r)

        #-- beta
        beta = rr / old_rr

        #-- d
        p = r + beta * p
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "CG + GPU",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info