import numpy as np
import scipy
from .common import Timer

# The Conjugate Gradient Method(CG) function
def vrm(A : np.ndarray, b : np.ndarray, n : int, maxiter : int, ndata : int, nbound : int, method,x0=None, tol=1e-10, T=np.float64) -> tuple:
    #-- timer
    timer = Timer()
    timer.start()

    #-- reduction process
    B = A[:ndata, :ndata]
    C = A[:ndata, ndata:]
    E = np.identity(ndata)
    Q, R, p = scipy.linalg.qr(C, pivoting=True)
    Q = Q[:, :nbound]
    R = R[:nbound, :]
    
    P = np.zeros((nbound, nbound))
    index = np.arange(nbound)
    P[p, index] = 1

    F = np.dot(Q, Q.T)
    U = E - F
    UT = U.T
    
    B_reduced = (UT) @ B @ U + F
    
    c = b[:ndata]
    d = b[ndata:]
    
    d_reduced = Q @ np.linalg.pinv(R.T) @ P.T @ d
    c_reduced = UT @ (c - B @ d_reduced) + d_reduced
    
    x, info = method(B_reduced, c_reduced, ndata, maxiter, tol=tol)
    residual = info["residual"]
    i = info["iterations"]
    
    timer.end()
    #-- infomation of method name, residual ,time and iterarions
    info = {"method name" : "VRM",
            "residual" : residual[residual > 0],
            "time" : timer.get_time(),
            "iterations" : i}
    
    return x, info