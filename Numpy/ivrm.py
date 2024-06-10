import numpy as np
import scipy
from .common import Timer

# The Conjugate Gradient Method(CG) function
def ivrm(A : np.ndarray, b : np.ndarray, n : int, maxiter : int, ndata : int, nbound : int, method,x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- timer
    timer = Timer()
    timer.start()

    #-- reduction process
    B = A[:ndata, :ndata]
    C = A[:ndata, ndata:]
    E = np.identity(ndata)

    CT = C.T
    alpha = C @ np.linalg.pinv(CT @ C)

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
    info = {"method name" : "iVRM",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info