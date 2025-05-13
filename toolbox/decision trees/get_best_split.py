import numpy as np
from .compute_information_gain import compute_information_gain

def get_best_split(X, y, node_indices):
    """
    Returns the best feature to split the node on, based on information gain.

    Args:
        X (ndarray):            Feature matrix of shape (n_samples, n_features)
        y (array-like):         Target labels (0 or 1)
        node_indices (list):    Indices of the samples at the current node

    Returns:
        best_feature (int):     Index of the best feature for the split
    """
    num_features = X.shape[1]
    best_feature = -1
    max_info_gain = -1  # Initialize to a value lower than any possible gain

    if np.all(y[node_indices] == y[node_indices][0]):
        return -1  # No split needed for pure node
    ### START CODE HERE ###
    for feature in range(num_features):
        info_gain = compute_information_gain(X, y, node_indices, feature)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature = feature
    ### END CODE HERE ###

    return best_feature
