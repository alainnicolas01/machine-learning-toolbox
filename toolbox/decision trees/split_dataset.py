def split_dataset(X, node_indices, feature):
    """
    Splits the data at the given node into
    left and right branches based on the value of a feature.

    Args:
        X (ndarray):             Data matrix of shape (n_samples, n_features)
        node_indices (list):     Active sample indices at this node
        feature (int):           Feature index to split on

    Returns:
        left_indices (list):     Indices where feature == 1
        right_indices (list):    Indices where feature == 0
    """

    left_indices = []
    right_indices = []

    for i in node_indices:
        if X[i, feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)

    return left_indices, right_indices
