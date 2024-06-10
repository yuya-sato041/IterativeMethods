import cupy as cp
import numpy as np

from ..common import Timer

# The Conjugate Gradient Method(CG) function
def vpgmres(A, b, n, maxiter, method, inner_iter, res_his=True, inner_tol=1e-1, x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial value
    if x0:
        x0 = cp.array(x0, T)
    else:
        x0 = cp.zeros(n, T)
    A = cp.array(A, T)
    b = cp.array(b, T)

    #-- residual
    residual = cp.zeros(maxiter+1, T)
    
    #-- timer
    timer = Timer()
    timer.start()

    bnorm = cp.linalg.norm(b)

    #-- initial value calculation
    r = b - cp.dot(A, x0)
    beta0 = cp.linalg.norm(r)
    
    v = cp.zeros((maxiter+1, n), T)
    v[0] = r / beta0
    H = cp.zeros((maxiter + 1, maxiter), T)
    Z = cp.zeros((maxiter+1, n), T)
    
    res = beta0/bnorm
    residual[0] = res
    
    inner_iter_n = cp.zeros(maxiter, cp.int32)
    inner_lres = cp.zeros(maxiter, T)
    
    for i in range(maxiter):
        Z[i], info = method(A, v[i], n, inner_iter, tol=inner_tol)
        inner_iter_n[i] = info["iterations"]
        inner_lres[i] = info["residual"][-1]

        w = cp.dot(A, Z[i])
        for j in range(i+1):
            H[j, i] = cp.dot(w, v[j])
            w = w - H[j, i]*v[j]
        
        H[i+1, i] = cp.linalg.norm(w)
        
        #-- convergence check

    
        if res_his:
            e_1 = cp.zeros(i+2, T)
            e_1[0] = 1
            eta = beta0*e_1
            gamma = H[:i + 2, :i+1]
            y = cp.linalg.lstsq(gamma, eta, rcond=None)[0]
            x = x0 + cp.sum(Z[:i+1] * y.reshape((i+1, 1)), axis=0)
            r = b - cp.dot(A, x)
            beta = cp.linalg.norm(r)
            res = beta / bnorm
            residual[i+1] = res
            if res <= tol:
                break
        else:
            if H[i+1, i] <= tol:
                m = i
                e_1 = cp.zeros(m+2, T)
                e_1[0] = 1
                eta = beta0*e_1
                gamma = H[:m + 2, :m+1]
                y = cp.linalg.lstsq(gamma, eta, rcond=None)[0]
                x = x0 + cp.sum(Z[:i+1] * y.reshape((i+1, 1)), axis=0)
                break
        
        v[i+1] = w/H[i+1, i]
        
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "VPGMRES + GPU",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i+1,
            "inner iterations" : inner_iter_n[:i+1],
            "inner last residual" : inner_lres[:i+1]}
    
    return x, info