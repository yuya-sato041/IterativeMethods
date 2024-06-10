import numpy as np
import cupy as cp
import scipy
from ..common import Timer

# The Conjugate Gradient Method(CG) function
def ivrm(A : np.ndarray, b : np.ndarray, n : int, maxiter : int, ndata : int, nbound : int, method,x0=None, tol=1e-10, T=np.float64) -> tuple:
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

    CT = C.T
    alpha = C @ cp.linalg.pinv(CT @ C)

    F = alpha @ CT
    U = E - F
    UT = U.T
    
    B_reduced = (UT) @ B @ U + F
    
    c = b[:ndata]
    d = b[ndata:]
    
    d_reduced = alpha @ d
    c_reduced = UT @ (c - B @ d_reduced) + d_reduced
    
    
    x, info = method(B_reduced, c_reduced, ndata, maxiter, tol=tol)
    residual = info["residual"]
    i = info["iterations"]
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "iVRM + GPU",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info