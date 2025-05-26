import numpy as np
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

digits = load_digits()
x_train = digits.data  # shape = (1797, 64)
y_train = digits.target

# 1. Daten skalieren -> relevant für PCA
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.reshape(x_train.shape[0], -1))

# 2. PCA anwenden (z. B. auf 2 Dimensionen)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_train_scaled)

def print_pca_info(pca_obj):
    print("Explained Variance Ratio:", pca_obj.explained_variance_ratio_)
    print("Singular Values:", pca_obj.singular_values_)
    print("Components (Eigenvectors):", pca_obj.components_.shape)

def get_optimal_components(pca_obj, threshold=0.95):
    cumulative_variance = np.cumsum(pca_obj.explained_variance_ratio_)
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    print(f"{threshold*100:.0f}% of variance is explained by {n_components} components.")
    return n_components
