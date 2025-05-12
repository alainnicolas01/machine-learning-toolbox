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
