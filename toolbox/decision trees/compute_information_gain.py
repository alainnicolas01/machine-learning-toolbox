from .split_dataset import split_dataset
from .compute_entropy import compute_entropy

def compute_information_gain(X, y, node_indices, feature):
    """
    Compute the information gain of splitting the node on a given feature.

    Args:
        X (ndarray):            Data matrix of shape (n_samples, n_features)
        y (array-like):         Target values (0 or 1) for each sample
        node_indices (list):    Indices of samples being considered at this step
        feature (int):          Index of the feature to split on

    Returns:
        information_gain (float): The information gain from this split
    """
    # Split dataset into left/right based on feature
    left_indices, right_indices = split_dataset(X, node_indices, feature)

    # Get target values at current node and branches
    y_node = y[node_indices]
    y_left = y[left_indices]
    y_right = y[right_indices]

    # Compute number of samples
    n = len(y_node)
    n_left = len(y_left)
    n_right = len(y_right)

    # Avoid division by zero in case of invalid split
    if n == 0:
        return 0.0

    # Compute entropy at the current node
    entropy_node = compute_entropy(y_node)

    # Compute entropy of left and right children
    entropy_left = compute_entropy(y_left)
    entropy_right = compute_entropy(y_right)

    # Weighted average of child entropies
    weighted_avg_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

    # Information gain is the difference
    information_gain = entropy_node - weighted_avg_entropy

    return information_gain
