import cupy as cp
import numpy as np

from ..common import Timer

# The Conjugate Gradient Method(CG) function
def vpgcr(A, b, n, maxiter, method, inner_iter, inner_tol=1e-1, x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial value x0
    if x0:
        x = x0
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
    p[0] = method(A, r, n, inner_iter, tol=inner_tol)[0]
    q = cp.zeros((maxiter+1, n), T)
    q[0] = cp.dot(A, p[0])
    bnorm = cp.linalg.norm(b)

    q2 = cp.zeros(maxiter, T)
    
    inner_iter_n = cp.zeros(maxiter, np.int32)
    inner_lres = cp.zeros(maxiter, T)
    
    #-- iteration
    for i in range(maxiter):
        #-- apennd resd to list
        res = cp.linalg.norm(r) / bnorm
        residual[i] = res
        
        #-- convergence check
        if res <= tol:
            break
        
        #-- omega
        omega = cp.dot(q[i], r)
        
        #-- sigma
        sigma = cp.dot(q[i], q[i])
        q2[i] = sigma

        #-- alpha                    
        alpha = omega / sigma

        #-- x
        x += alpha * p[i]

        #-- residual
        r -= alpha * q[i]

        #-- z
        z, info = method(A, r, n, inner_iter, tol=inner_tol)
        inner_iter_n[i] = info["iterations"]
        inner_lres[i] = info["residual"][-1]

        #-- beta
        az = cp.dot(A, z)
        beta = -cp.dot(q[:i+1], az) / q2[:i+1]
        beta = beta.reshape((i+1, 1))
        
        #-- p
        bp = beta * p[:i+1]
        p[i+1] = z + cp.sum(bp, axis=0)
        
        #-- ap
        bq = beta * q[:i+1]
        q[i+1] = az + cp.sum(bq, axis=0)
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "VPGCR+GPU",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i,
            "inner iterations" : inner_iter_n[:i],
            "inner last residual" : inner_lres[:i]}
    
    return x, info