import numpy as np

# 4 Compute Cost With Multiple Variables
def compute_cost(X, y, w, b):
    """
    Computes the cost function for both single-variable and multiple-variable linear regression.

    Args:
      X (ndarray (m, n)) : Feature matrix, where m = number of examples, n = number of features
      y (ndarray (m, ))  : Target values
      w (ndarray (n, ))  : Model parameters (weights)
      b (scalar)         : Model parameter (bias)

    Returns:
      cost (float): The computed cost function value
    """
    m = len(y)  # Number of training examples
    predictions = np.dot(X, w) + b  # Compute predictions for all samples
    errors = predictions - y  # Difference between predictions and actual values
    cost = np.sum(errors ** 2) / (2 * m)  # Mean Squared Error (MSE) cost function
    return cost
