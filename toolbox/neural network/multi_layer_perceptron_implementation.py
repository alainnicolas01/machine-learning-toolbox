import copy
from itertools import islice
from typing import List, Tuple, Union, Iterable

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.preprocessing import LabelBinarizer
from tqdm.notebook import tqdm


class MLPClassifier(BaseEstimator):
    """ Multi-Layer Perceptron for Classifikation

        Args:
            num_nodes_per_layer: List with the number of nodes per layer
                 - including input-layer (features)
                 - no bias nodes should be included
                 - example: [10, 5, 2] -> 10 Features, 5 Nodes in 1st Hidden Layer, 2 Output Nodes

            lr: initial learning rate (float)

            alpha: L2 Regularization strength (float)

            batch_size: number of samples to use for one update
                - set to None for batch gradient descent (all samples)

            max_num_steps: max number of steps (gradient updates)

            weights_init_range: Tuple of two floats.
                - The interval on which the weights will be randomly initialized

            adaptive_lr_constant: Adpative learning rate constant
                - lr_new = lr * exp(-adaptive_lr_constant * step)
                - disable with 0.0

            min_lr: Minimum Learning Rate, relevant when using adaptive_lr_constant > 0.0

            random_seed: random seed for weights initialization (int)
    """

    def __init__(
            self,
            num_nodes_per_layer: List[int],
            lr: float = 0.01,
            alpha: float = 0.0,
            batch_size: Union[int, None] = None,
            max_num_steps: int = 1000,
            weights_init_range: Tuple[float, float] = (-0.7, 0.7),
            adaptive_lr_constant: float = 0.0,
            min_lr: float = 1e-4,
            random_seed: int = 123
    ):
        self.num_nodes_per_layer = num_nodes_per_layer
        self.lr = lr
        self.alpha = alpha
        self.batch_size = batch_size
        self.max_num_steps = max_num_steps
        self.weights_init_range = weights_init_range
        self.adaptive_lr_constant = adaptive_lr_constant
        self.min_lr = min_lr
        self.random_seed = random_seed
        self._initialize_weights()

    def _initialize_weights(self):
        """ Initialize Model-Parameters
        Example: num_nodes_per_layer = [3, 4, 2]
        W0.shape = (4,4)

        self.weights_ = [W0, W1, ..., Wn]
        W0 = 3 Input Features -> 4 Neuros from the Hidden Layer 1
        W1 = 4 Neurons of the Hidden Layer 1 to the 2 Output Neurons from the Output Layer
        W0 = [
                [w11, w12, w13, w14], -> Weights of the Input Feature 1 to the 4 Hidden Neurons
                [w21, w22, w23, w24], -> Weights of the Input Feature 2 to the 4 Hidden Neurons
                [w31, w32, w33, w34], -> Weights of the Input Feature 3 to the 4 Hidden Neurons
                [b1,  b2,  b3,  b4] -> Bias_weights for the 4 Hidden Neurons
             ]
        """

        np.random.seed(self.random_seed)
        self.weights_ = []
        for i in range(len(self.num_nodes_per_layer) - 1):
            input_size = self.num_nodes_per_layer[i] + 1  # Plus one because of the BIAS Adding
            output_size = self.num_nodes_per_layer[i + 1]
            low, high = self.weights_init_range
            W = np.random.uniform(low=low, high=high, size=(input_size, output_size))
            self.weights_.append(W)

    def forward(self, X: np.ndarray) -> List[np.ndarray]:
        """ forward pass - calculate all activations
            Args:
                X: Input-Activations [num_samples, num_features]

            Returns: List of Layer-Activations

            Example for X:
            2 Examples with each 3 features
            X = np.array([
                        [0.1, 0.2, 0.3], Example Data 1
                        [0.5, 0.6, 0.7] Example Data 2
                        ])

            Example of activations:
            num_nodes_per_layer = [3, 4, 2] and X from above

            activations = [
                            array([[0.1, 0.2, 0.3],
                                    [0.5, 0.6, 0.7]]), Input

                            array([[0.55, 0.48, 0.67, 0.62],
                                    [0.73, 0.59, 0.84, 0.77]]), Hidden Layer Versteckte Schicht

                            array([[0.61, 0.39],
                                    [0.22, 0.78]]) Softmax-Output
                            ]
        """
        activations = [X]
        A = X
        for i, W in enumerate(self.weights_):  # runs through every Wn
            A = np.hstack([A, np.ones(
                (A.shape[0], 1))])  # Adding the Bias -> Adds to every row at the right a 1
            Z = A @ W  # Matrix Multiplication
            if i < len(self.weights_) - 1:
                A = self.sigmoid(Z)
            else:
                A = self.softmax(Z)
            activations.append(A)
        return activations

    def backward(self, activations: List[np.ndarray], delta_upstream: np.ndarray) -> List[
        np.ndarray]:
        """ backward pass / backpropagation - Calculate partial derivatives of model weights
            Args:
                activations: List of Layer Activations, Output of forward()

                delta_upstream: Matrix with partial derivatives of Loss functions wrt.
                    logits of last layer[num_samples, num_outputs]

            Returns: List of Layer-Gradients
            activations = [A0, A1, ..., An] etc.

        """
        gradients = []
        for i in reversed(range(len(
                self.weights_))):  # backwards from output layer towards input layer weights
            A_prev = np.hstack([activations[i], np.ones((activations[i].shape[0],
                                                         1))])  # gets corresponding Activation, for example, A0 if we are looking at W0

            # Compute the gradient of the weight matrix:
            # - First term: standard gradient from the chain rule (input.T @ delta)
            # - Second term: L2 regularization (applied to all weights except the bias row, which remains unregularized)
            # The result is the full weight gradient including regularization
            dW = A_prev.T @ delta_upstream + self.alpha * np.vstack(
                [self.weights_[i][:-1], np.zeros((1, self.weights_[i].shape[1]))])
            gradients.insert(0, dW)

            if i > 0:
                delta_upstream = (delta_upstream @ self.weights_[i][:-1].T) * (
                        activations[i] * (1 - activations[i]))  # sigmoid'
        return gradients

    def fit(
            self,
            X: np.ndarray, y: np.ndarray,
            X_val: Union[np.ndarray, None] = None,
            y_val: Union[np.ndarray, None] = None,
            validate_after_every_num_steps: int = 20) -> None:
        """ Fit Model Parameters
            Args:
                X, y: Training-Data
                X_val, y_val: (Optional) Validation-Data
                validate_after_every_num_steps: (Optional) Validation after every xth step
        """
        num_classes = np.unique(y).shape[0]
        lb = LabelBinarizer()
        y_encoded = lb.fit_transform(y)
        if y_val.shape[1] == 1:
            y_encoded = np.hstack([1 - y_encoded, y_encoded])

        for step in tqdm(range(self.max_num_steps)):
            lr = max(self.lr * np.exp(-self.adaptive_lr_constant * step), self.min_lr)
            batch_generator = self.generate_batches(X, y_encoded, self.batch_size)
            X_batch, y_batch = next(batch_generator)

            activations = self.forward(X_batch)
            y_pred = activations[-1]
            delta = self.calculate_gradient_of_cost_function(y_batch, y_pred)
            gradients = self.backward(activations, delta)

            for i in range(len(self.weights_)):
                self.weights_[i] -= lr * gradients[i]

            if X_val is not None and y_val is not None and step % validate_after_every_num_steps == 0:
                acc = self.score(X_val, y_val)
                print(f"Step {step}: validation accuracy {acc: .4f}")

        return self

    def calculate_cost(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """ Calculate Cross-Entropy cost """
        m = y_true.shape[0]
        eps = 1e-12  # Small constant to prevent log(0)
        cross_entropy = -np.sum(y_true * np.log(y_pred + eps)) / m
        l2_reg = self.alpha * sum(np.sum(W[:-1] ** 2) for W in self.weights_)
        return cross_entropy + l2_reg

    def calculate_gradient_of_cost_function(self, y_true: np.ndarray,
                                            y_pred: np.ndarray) -> np.ndarray:
        """ Calculate gradient of cost function w.r.t logits dJ/dZ """
        return (y_pred - y_true) / y_true.shape[0]

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """ Softmax along last dimension: x is of shape [num_samples, num_classes] """
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """ Sigmoid activation function """
        return 1 / (1 + np.exp(-x))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """ All class probabilities per sample
            -> input shape  [num_samples, num_features]
            -> output shape [num_samples, num_classes]
        """
        return self.forward(X)[-1]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """ Return most likely class per sample
           -> input shape  [num_samples, num_features]
           -> output shape [num_samples, 1]
        """
        return np.argmax(self.predict_proba(X), axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """ Calculate Accuracy """
        y_pred = self.predict(X)
        return np.mean(y_pred == y)

    def generate_batches(
            self, X: np.ndarray, y: np.ndarray,
            batch_size: Union[int, None],
            shuffle: bool = True) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
        """ Generate batches of data [BATCH_SIZE, NUM_FEATURES]"""

        # Batch Gradient Descent
        if not batch_size:
            batch_size = X.shape[0]
            while True:
                yield X, y

        # Flexible Batch-Size that works with batch_size != None
        while True:
            # randomly shuffle input data
            shuffled_indices = np.arange(X.shape[0])
            if shuffle:
                np.random.shuffle(shuffled_indices)

            # iterate over batches
            iter_indices = np.arange(X.shape[0])
            slices = islice(iter_indices, 0, None, batch_size)

            # select batches
            for start_index in slices:
                indices_to_select = shuffled_indices[start_index: start_index + batch_size]
                yield X[indices_to_select, :], y[indices_to_select, :]

    def grad_check(
            self, X: np.ndarray, y: np.ndarray,
            epsilon: float = 0.0001,
            decimal: int = 3,
            verbose: bool = False) -> None:
        """Compare the gradient with finite differences around current point in parameter space
            Args:
                X, y: Data
                epsilon: step to take around current parameters
                decimal: verify equality if approximatino is within +- 1e-<decimal>
        """
        # calculate gradient
        activations = self.forward(X)
        y_hat = activations[-1]
        delta_upstream = self.calculate_gradient_of_cost_function(y, y_hat)
        gradients = self.backward(activations, delta_upstream)

        # approximate gradient
        gradients_approx = copy.deepcopy(gradients)
        original_layers = copy.deepcopy(self.weights_)

        # Iterate over each parameter of the network
        for i, weights in enumerate(self.weights_):
            for j, _ in enumerate(weights.flat):
                # generate copy of original parameters for modification
                modified_layers = copy.deepcopy(original_layers)

                # assign modified layers for use in other methods
                self.weights_ = modified_layers

                # J(theta + epsilon)
                modified_layers[i].flat[j] += epsilon
                y_hat = self.predict_proba(X)
                cost_plus_epsilon = self.calculate_cost(y, y_hat)

                # J(theta - epsilon)
                modified_layers[i].flat[j] -= 2 * epsilon
                y_hat = self.predict_proba(X)
                cost_minus_epsilon = self.calculate_cost(y, y_hat)

                # Approx gradient with:
                # (J(theta + epsilon) - J(theta - epsilon)) / (2 * epsilon)
                grad_approx = (
                        (cost_plus_epsilon - cost_minus_epsilon) /
                        (2 * epsilon))

                gradients_approx[i].flat[j] = grad_approx

        # reset layers attribute
        self.weights_ = copy.deepcopy(original_layers)

        # normalize gradients
        gradients_approx = [x / np.linalg.norm(x) for x in gradients_approx]
        gradients = [x / np.linalg.norm(x) for x in gradients]

        if verbose:
            print('approx : ', gradients_approx)
            print('calc : ', gradients)

        for approx, calculated in zip(gradients_approx, gradients):
            np.testing.assert_array_almost_equal(approx, calculated, decimal=decimal)
        print(f"Gradients within +- 1e-{decimal} of approximated gradient!")
