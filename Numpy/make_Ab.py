import numpy as np
import scipy
from io import StringIO

# The funcition to make coefficient matrix A and constant vecotor from .txt files.
def make_Ab(path_matrix, path_vector=False, n=None,T= np.float64):
    if path_vector:
        with open(path_matrix) as f:
            ls_matrix = f.readlines()

        with open(path_vector) as f:
            ls_vector = f.readlines()

        a = [i.strip() for i in ls_matrix[1:]]
        b = [i.strip() for i in ls_vector[1:]]

        A = []
        for i in range(n):
            A.append(a[n*i : n*(i+1)])
            
        A = np.array(A, T)
        b = np.array(b, T)
        
        return A, b, np.array(ls_matrix[0].split(), np.int64)

    else:
        with open(path_matrix) as f:
            txt = f.read()
            
        A = scipy.io.mmread(StringIO(txt)).A.astype(T)
        n = A.shape[0]
        x = np.ones(n)
        b = np.dot(A, x)
        
        return A, b