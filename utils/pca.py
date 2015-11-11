import numpy as np

"""
Performs principle-component-analysis on dataset

Parameters:
    data: NxM ndarray
        input data (trials x dimensions)
Returns:
    signals: NxM ndarray
        transformed signals
    PC: MxM ndarray
        principle components (each column is one)
    V: Mx1 ndarray
        variance explained by each dimension
"""
def pca(dat):
    data = dat.copy()
    (trials,dims) = np.shape(data)
    data -= np.mean(data,0)         # substract mean
    [u,S,PC] = np.linalg.svd(data)  # apply svd
    PC = PC.T                       # each column is now a PC
    V = S*S                         # compute explained variance
    signals = np.dot(data,PC)       # transform data to PC space
    return (signals, PC, V)

# Simple test example
if __name__ == '__main__':
    from pylab import *
    # Rotate a 3D-gaussian blob and use PCA to rotate back
    data = np.random.randn(200,3)
    data[:,0] *= 10         # make 1st dim more important
    data[:,2] *= 0.1        # make 3rd dim less important
    alpha = 315 *np.pi/180  # rotation angle in rad (counter-clockwise)
    rot_matrix = np.array([[np.cos(alpha), -np.sin(alpha)],
                           [np.sin(alpha),  np.cos(alpha)]])
    data[:,:2] = np.dot(data[:,:2],rot_matrix) # rotate first 2 dims
    scatter(data[:,0], data[:,1])
    axis('equal')
    title('before pca')
    
    [signals, PC, V] = pca(data)
    figure()
    scatter(signals[:,0], signals[:,1])
    axis('equal')
    title('after pca')  
    show()
