import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler

# Step 1: Load open-source dataset (Example: California Housing Data from sklearn)
from sklearn.datasets import fetch_california_housing

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

X_features = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']
X_train = data[X_features].values  # Extract feature matrix
y_train = data['target'].values  # Extract target variable (house price)

# Step 3: Scale/normalize the training data
# StandardScaler transforms data to have mean=0 and variance=1 (z-score normalization)
scaler = StandardScaler()
X_norm = scaler.fit_transform(X_train)

# Print the range of values before and after normalization
print(f"Peak to Peak range by column in Raw Data: {np.ptp(X_train, axis=0)}")
print(f"Peak to Peak range by column in Normalized Data: {np.ptp(X_norm, axis=0)}")



# Step 4: Create and fit the regression model
# Stochastic Gradient Descent (SGD) is used for training the linear model --> comes from SCIKIT
machine_learning_algo = SGDRegressor(max_iter=1000)  # Train for a maximum of 1000 iterations
machine_learning_algo.fit(X_norm, y_train)  # Fit the model using the normalized training data
print(machine_learning_algo)
print(f"Number of iterations completed: {machine_learning_algo.n_iter_}, Number of weight updates: {machine_learning_algo.t_}")



# Step 5: View model parameters
b_norm = machine_learning_algo.intercept_  # Intercept (bias term)
w_norm = machine_learning_algo.coef_  # Coefficients (weights for each feature)
print(f"Model parameters: w: {w_norm}, b: {b_norm}")

# Step 6: Make predictions using the trained model
# We make predictions using both the model's predict method and manual matrix multiplication
y_pred_sgd = machine_learning_algo.predict(X_norm)  # Using built-in predict function
y_pred = np.dot(X_norm, w_norm) + b_norm  # Manually computing predictions

# Check if both methods give the same predictions
print(f"Prediction using np.dot() and machine_learning_algo.predict match: {(y_pred == y_pred_sgd).all()}")

# Print first few predictions compared to actual values
print(f"Predictions on training set: {y_pred[:4]}")
print(f"Target values: {y_train[:4]}")

# Step 7: Plot results
# Scatter plots to visualize predictions vs actual house prices
fig, ax = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:, i], y_train, label='Target')  # Actual data points
    ax[i].set_xlabel(X_features[i])  # Feature name
    ax[i].scatter(X_train[:, i], y_pred, color='orange', label='Predict')  # Predicted values
ax[0].set_ylabel("Price")
ax[0].legend()
fig.suptitle("Target vs Prediction using z-score normalized model")
plt.show()