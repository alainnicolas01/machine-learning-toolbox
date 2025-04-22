import numpy as np
from toolbox.cost_function import compute_cost

# Steigung berechnen
def compute_gradient(X, y, w, b):
    """
    Computes the gradient for linear regression (works for both single and multiple-variable cases).

    Args:
      X (ndarray (m, n)) : Feature data (m examples, n features)
      y (ndarray (m, ))  : Target values
      w (ndarray (n, ))  : Model parameters (weights)
      b (scalar)         : Model parameter (bias)

    Returns:
      dj_dw (ndarray (n, )): Gradient of cost w.r.t. w
      dj_db (scalar)       : Gradient of cost w.r.t. b
      Rückgabe:
      dj_dw (ndarray (n, )): Ableitung der Kostenfunktion bezüglich w [Steigung für Gewichte]
      dj_db (scalar)       : Ableitung der Kostenfunktion bezüglich b [Steigung für Bias]
    """
    m = len(y)
    predictions = np.dot(X, w) + b  # Compute predictions
    errors = predictions - y

    dj_dw = np.dot(X.T, errors) / m  # Compute gradient w.r.t. w
    dj_db = np.sum(errors) / m  # Compute gradient w.r.t. b

    return dj_dw, dj_db


# Führt den Steigungsabstieg (Gradient Descent) für die lineare Regression durch.
def gradient_descent_for_linear_regression(X, y, w_in, b_in, alpha, num_iters):
    """
    Performs batch gradient descent for linear regression (single or multiple variables).

    Args:
      X (ndarray (m, n)) : Feature data (m examples, n features)
      y (ndarray (m, ))  : Target values
      w_in (ndarray (n, )): Initial weights
      b_in (scalar)      : Initial bias
      alpha (float)      : Learning rate
      num_iters (int)    : Number of iterations

    Returns:
      w (ndarray (n, )): Updated weights
      b (scalar):       Updated bias
      J_history (list): History of cost function values per iteration
    """
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(X, y, w, b)

        # Update parameters
        w -= alpha * dj_dw
        b -= alpha * dj_db

        # Store cost for analysis
        J_history.append(compute_cost(X, y, w, b))

        # Print updates at intervals
        if i % (num_iters // 10) == 0:
            print(f"Iteration {i}: Cost {J_history[-1]:.4e}")

    return w, b, J_history
