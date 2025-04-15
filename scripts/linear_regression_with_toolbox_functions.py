import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from toolbox.gradient_descent_methods import gradient_descent_for_linear_regression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_california_housing

# Step 1: Load open-source dataset (California Housing Data from sklearn)
def fetch_housing_data():
    """
    Fetches the California Housing dataset from scikit-learn.
    Converts it into a Pandas DataFrame for easier manipulation.
    """
    data = fetch_california_housing()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target  # The target variable is the median house price.
    return df

# Step 2: Load the dataset
data = fetch_housing_data()

# Selecting relevant features for our regression model
X_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']  # Features influencing house prices
X_train = data[X_features].values  # Extract feature matrix
y_train = data['target'].values  # Extract target variable (house price)

# Step 3: Scale/normalize the training data
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Print the range of values before and after normalization
print(f"Peak to Peak range by column in Raw Data: {np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized Data: {np.ptp(X_norm, axis=0)}")

# Step 4: Initialize parameters and perform gradient descent
w_init = np.zeros(X_norm.shape[1])  # Initialize weights to zero
b_init = 0  # Initialize bias to zero
alpha = 0.01  # Learning rate
num_iters = 1000  # Number of iterations

# Run gradient descent
w_final, b_final, J_history = gradient_descent_for_linear_regression(X_norm, y_train, w_init, b_init, alpha, num_iters)

print(f"Final parameters after gradient descent: w: {w_final}, b: {b_final}")

# Step 5: Make predictions using the trained model
y_pred = np.dot(X_norm, w_final) + b_final  # Compute predictions manually

# Print first few predictions compared to actual values
print(f"Predictions on training set: {y_pred[:4]}")
print(f"Target values: {y_train[:4]}")


# Step 6: Create a figure with subplots for both plots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# First subplot: Target vs Prediction
for i in range(4):
    axs[0].scatter(X_train[:, i], y_train, label=f'Target - {X_features[i]}', alpha=0.6)
    axs[0].scatter(X_train[:, i], y_pred, label=f'Predict - {X_features[i]}', color='orange', alpha=0.6)
axs[0].set_xlabel("Features")
axs[0].set_ylabel("Price")
axs[0].set_title("Target vs Prediction using Gradient Descent")
axs[0].legend()

# Second subplot: Cost function vs Iterations
axs[1].plot(range(num_iters), J_history, label='Cost Function', color='b')
axs[1].set_xlabel('Iterations')
axs[1].set_ylabel('Cost')
axs[1].set_title('Cost Function vs Iterations (Gradient Descent Convergence)')
axs[1].legend()

# Display both plots
plt.tight_layout()
plt.show()