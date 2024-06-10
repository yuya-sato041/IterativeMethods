import numpy as np

from .common import Timer

# The Conjugate Gradient Method(CG) function
def vpgmres(A, b, n, maxiter, method, inner_iter, res_his=True, inner_tol=1e-1, x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial value x0
    if x0:
        x0 = x0
    else:
        x0 = np.zeros(n, T)

    #-- residual
    residual = np.zeros(maxiter+1, T)
    
    #-- timer
    timer = Timer()
    timer.start()

    bnorm = np.linalg.norm(b)

    #-- initial value calculation
    r = b - np.dot(A, x0)
    beta0 = np.linalg.norm(r)
    
    v = np.zeros((maxiter+1, n), T)
    v[0] = r / beta0
    H = np.zeros((maxiter+1, maxiter), T)
    Z = np.zeros((maxiter+1, n), T)
    
    res = beta0 / bnorm
    residual[0] = res
    
    inner_iter_n = np.zeros(maxiter, np.int32)
    inner_lres = np.zeros(maxiter, T)
    
    for i in range(maxiter):
        Z[i], info = method(A, v[i], n, inner_iter, tol=inner_tol)
        inner_iter_n[i] = info["iterations"]
        inner_lres[i] = info["residual"][-1]

        w = np.dot(A, Z[i])
        for j in range(i+1):
            H[j, i] = np.dot(w, v[j])
            w = w - H[j, i]*v[j]
        
        H[i+1, i] = np.linalg.norm(w)
    
        if res_his:
            e_1 = np.zeros(i+2, T)
            e_1[0] = 1
            eta = beta0*e_1
            gamma = H[:i + 2, :i+1]
            y = np.linalg.lstsq(gamma, eta, rcond=None)[0]
            x = x0 + np.sum(Z[:i+1] * y.reshape((i+1, 1)), axis=0)
            r = b - np.dot(A, x)
            beta = np.linalg.norm(r)
            res = beta / bnorm
            residual[i+1] = res
            if res <= tol:
                break
            
        else:
            if H[i+1, i] <= tol:
                i
                e_1 = np.zeros(i+2, T)
                e_1[0] = 1
                eta = beta0*e_1
                gamma = H[:i + 2, :i+1]
                y = np.linalg.lstsq(gamma, eta, rcond=None)[0]
                x = x0 + np.sum(Z[:i+1] * y.reshape((i+1, 1)), axis=0)
                break
            
        
        v[i+1] = w/H[i+1, i]
        
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "VPGMRES",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i+1,
            "inner iterations" : inner_iter_n[:i+1],
            "inner last residual" : inner_lres[:i+1]}
    
    return x, info