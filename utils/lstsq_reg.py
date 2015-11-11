import numpy as np

"""
Performs tikhonov regularized least squares
I.e. solution of y = Ax for x

Parameters:
    A: NxM ndarray
    y: NxK ndarray
    mue: double
        regularization parameter
Returns:
    x: MxK array
"""
def lstsq_reg(A,Y,mue):
    Ac = A.copy()
    Yc = Y.copy()
    result = np.dot(np.dot(np.linalg.inv(
                            (Ac.T).dot(Ac)
                            +mue*np.eye(np.shape(Ac)[1])),
                        Ac.T),
                    Yc)
    return result

# Simple test example
if __name__ == '__main__':
    pass
