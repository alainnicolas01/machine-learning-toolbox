import numpy as np

def compute_entropy(y):
    """
    Computes the entropy for a node.

    Args:
       y (ndarray): Numpy array of 0s and 1s, where 1 = edible, 0 = poisonous

    Returns:
        entropy (float): Entropy at that node
    """
    # Calculate the proportion of edible (1s)
    if len(y) == 0:
        return 0.0
    p = np.mean(y)

    # If all elements are the same, entropy is 0
    if p == 0 or p == 1:
        return 0.

    # Entropy formula for binary classification
    entropy = -p * np.log2(p) - (1 - p) * np.log2(1 - p)

    return entropy
