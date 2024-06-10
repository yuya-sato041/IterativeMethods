import numpy as np

from .common import Timer

# The Conjugate Gradient Method(CG) function
def vpcg(A, b, n, maxiter, method, inner_iter, inner_tol=1e-1, x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial value x0
    if x0:
        x = x0
    else:
        x = np.zeros(n, T)

    #-- residual
    residual = np.zeros(maxiter+1, T)
    
    inner_iter_n = np.zeros(maxiter, np.int32)
    inner_lres = np.zeros(maxiter, T)
    
    #-- timer
    timer = Timer()
    timer.start()

    #-- initial value calculation
    r = b - np.dot(A, x)
    z = method(A, r, n, inner_iter, tol=inner_tol)[0]
    p = np.copy(z)
    rz = np.dot(r, z)
    bnorm = np.linalg.norm(b)
    

    #-- iteration
    for i in range(maxiter):

        #-- apennd resd to list
        res = np.linalg.norm(r) / bnorm
        print(res)
        residual[i] = res
        #-- convergence check
        if res <= tol:
            break  
        
        #-- ap
        ap =  np.dot(A, p)
        
        #-- sgma
        sigma = np.dot(p, ap)

        #-- alpha                    
        alpha = rz / sigma

        #-- x
        x += alpha * p

        #-- residual
        r -= alpha * ap
        
        z, info = method(A, r, n, inner_iter, tol=inner_tol)
        inner_iter_n[i] = info["iterations"]
        inner_lres[i] = info["residual"][-1]
        
        #-- rr
        old_rz = np.copy(rz)
        rz = np.dot(r, z)

        #-- beta
        beta = rz / old_rz

        #-- d
        p = z + beta * p
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "VPCG",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i,
            "inner iterations" : inner_iter_n[:i],
            "inner last residual" : inner_lres[:i]}
    
    return x, info