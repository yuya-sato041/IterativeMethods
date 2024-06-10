# QRの並列化が必須

import numpy as np
import cupy as cp
import scipy
from .common import Timer

# The Conjugate Gradient Method(CG) function
def vrm(A : np.ndarray, b : np.ndarray, n : int, maxiter : int, ndata : int, nbound : int, method,x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- initial value
    if x0:
        x = cp.array(x0, T)
    else:
        x = cp.zeros(n, T)
    A = cp.array(A, T)
    b = cp.array(b, T)
	
	#-- timer
    timer = Timer()
    timer.start()

    #-- reduction process
    B = A[:ndata, :ndata]
    C = A[:ndata, ndata:]
    E = cp.identity(ndata)
    Q, R, p = scipy.linalg.qr(C, pivoting=True)
    Q = Q[:, :nbound]
    R = R[:nbound, :]
    
    P = cp.zeros((nbound, nbound))
    index = cp.arange(nbound)
    P[p, index] = 1

    F = cp.dot(Q, Q.T)
    U = E - F
    UT = U.T
    
    B_reduced = (UT) @ B @ U + F
    
    c = b[:ndata]
    d = b[ndata:]
    
    d_reduced = Q @ cp.linalg.pinv(R.T) @ P.T @ d
    c_reduced = UT @ (c - B @ d_reduced) + d_reduced
    
    x, info = method(B_reduced, c_reduced, ndata, maxiter, tol=tol)
    residual = info["residual"]
    i = info["iterations"]
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "VRM + GPU",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info