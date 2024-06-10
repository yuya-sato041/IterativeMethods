import cupy as cp
import numpy as np

from ..common import Timer

# The Conjugate Gradient Method(CG) function
def gcr(A, b, n, maxiter, x0=None, tol=1e-10, T=cp.float64) -> tuple:
    #-- initial value
    if x0:
        x = cp.array(x0, T)
    else:
        x = cp.zeros(n, T)
    A = cp.array(A, T)
    b = cp.array(b, T)

    #-- residual
    residual = cp.zeros(maxiter+1, T)
    
    #-- timer
    timer = Timer()
    timer.start()

    #-- initial value calculation

    r = b - cp.dot(A, x)
    p = cp.zeros((maxiter+1, n), T)
    p[0] = cp.copy(r)
    ap = cp.zeros((maxiter+1, n), T)
    ap[0] = cp.dot(A, p[0])
    bnorm = cp.linalg.norm(b)
    
    ap2 = cp.zeros(maxiter, T)

    #-- iteration
    for i in range(maxiter):
        #-- apennd resd to list
        res = cp.linalg.norm(r) / bnorm
        residual[i] = res

        #-- convergence check
        if res <= tol:
            break       
        
        #-- omega
        omega = cp.dot(ap[i], r)
        
        #-- sigma
        sigma = cp.dot(ap[i], ap[i])
        ap2[i] = sigma

        #-- alpha            
        alpha = omega / sigma

        #-- x
        x += alpha * p[i]
        
        #-- residual
        r -= alpha * ap[i]  

        #-- beta
        ar = cp.dot(A, r)
        beta = -cp.dot(ap[:i+1], ar) / ap2[:i+1]
        beta = beta.reshape((i+1, 1))
        
        #-- p
        bp = beta * p[:i+1]
        p[i+1] = r + cp.sum(bp, axis=0)
        
        #-- ap
        bap = beta * ap[:i+1]
        ap[i+1] = ar + cp.sum(bap, axis=0)
    
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "GCR + GPU",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info