import numpy as np


# UNQ_C4
# GRADED FUNCTION: predict

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def predict(X, w, b):
    """
    Predict whether the label is 0 or 1 using learned logistic
    regression parameters w

    Args:
      X : (ndarray Shape (m,n)) data, m examples by n features
      w : (ndarray Shape (n,))  values of parameters of the model
      b : (scalar)              value of bias parameter of the model

    Returns:
      p : (ndarray (m,)) The predictions for X using a threshold at 0.5
    """
    # number of training examples
    m, n = X.shape
    p = np.zeros(m)

    ### START CODE HERE ###
    for i in range(m):
        z_wb = 0
        for j in range(n):
            z_wb += X[i][j] * w[j]

        z_wb += b
        f_wb = sigmoid(z_wb)

        if f_wb >= 0.5:
            p[i] = 1
        else:
            p[i] = 0
    ### END CODE HERE ###
    return p
