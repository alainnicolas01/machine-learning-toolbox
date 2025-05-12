import numpy as np

def my_softmax(z):
    ez = np.exp(z)
    sm = ez/np.sum(ez)
    return(sm)