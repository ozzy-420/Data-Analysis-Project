import logging
import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
                             roc_auc_score, mean_squared_error, r2_score)

# Configure logging
logger = logging.getLogger(__name__)


def add_bias_term(X: np.ndarray) -> np.ndarray:
    """
    Adds a bias term (column of ones) to the feature matrix.

    Args:
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Feature matrix with bias term added.
    """
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)


class NumpyLinearRegressionClosedForm:
    """
    Linear Regression implementation using the closed-form solution.
    """

    def __init__(self):
        self.theta = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'NumpyLinearRegressionClosedForm':
        """
        Fits the linear regression model using the closed-form solution.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.

        Returns:
            self: Fitted model.
        """
        X_b = add_bias_term(X)
        try:
            self.theta = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
        except np.linalg.LinAlgError:
            logger.warning("Matrix X_b.T @ X_b is singular. Using pseudo-inverse.")
            self.theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts target values using the fitted model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted values.
        """
        X_b = add_bias_term(X)
        return X_b @ self.theta


def run_numpy_linear_regression_closed_form(X_train: np.ndarray, y_train: np.ndarray,
                                            X_val: np.ndarray, y_val: np.ndarray,
                                            X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """
    Trains and evaluates the linear regression model using the closed-form solution.

    Args:
        X_train, X_val, X_test (np.ndarray): Feature matrices for training, validation, and testing.
        y_train, y_val, y_test (np.ndarray): Target vectors for training, validation, and testing.

    Returns:
        dict: Evaluation metrics for each dataset split.
    """
    logger.info("Running NumPy Linear Regression (Closed Form)...")

    model = NumpyLinearRegressionClosedForm()
    model.fit(X_train, y_train)

    results = {}
    for split_name, X_split, y_split_true in [("train", X_train, y_train), ("val", X_val, y_val),
                                              ("test", X_test, y_test)]:
        y_pred = model.predict(X_split)
        metrics = {
            "MSE": mean_squared_error(y_split_true, y_pred),
            "R2": r2_score(y_split_true, y_pred)
        }
        results[split_name] = metrics
        logger.info(f"Linear Regression ({split_name}) - MSE: {metrics['MSE']:.4f}, R2: {metrics['R2']:.4f}")
    return results


class NumpyLogisticRegressionGD:
    """
    Logistic Regression implementation using gradient descent.
    """

    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, tol=1e-4, verbose=False):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.tol = tol
        self.verbose = verbose
        self.theta = None
        self.cost_history = []
        self.test_cost_history = []

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Computes the sigmoid function.

        Args:
            z (np.ndarray): Input array.

        Returns:
            np.ndarray: Sigmoid output.
        """
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _compute_cost(self, X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
        """
        Computes the logistic regression cost function.

        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Target vector.
            theta (np.ndarray): Model parameters.

        Returns:
            float: Cost value.
        """
        m = len(y)
        h = self._sigmoid(X @ theta)
        epsilon = 1e-5  # To prevent log(0)
        cost = -(1 / m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
        return cost

    def fit(self, X: np.ndarray, y: np.ndarray, X_val: np.ndarray = None,
            y_val: np.ndarray = None) -> 'NumpyLogisticRegressionGD':
        X_b = add_bias_term(X)
        n_samples, n_features = X_b.shape
        self.theta = np.zeros(n_features)
        self.cost_history = []
        self.test_cost_history = []

        prev_cost = float('inf')

        for iteration in range(self.n_iterations):
            indices = np.random.permutation(n_samples)
            X_b_shuffled = X_b[indices]
            y_shuffled = y[indices]

            for i in range(0, n_samples, self.batch_size):
                X_batch = X_b_shuffled[i:i + self.batch_size]
                y_batch = y_shuffled[i:i + self.batch_size]

                if X_batch.shape[0] == 0:
                    continue

                h = self._sigmoid(X_batch @ self.theta)
                gradient = (1 / len(y_batch)) * X_batch.T @ (h - y_batch)
                self.theta -= self.learning_rate * gradient

            current_cost = self._compute_cost(X_b, y, self.theta)
            self.cost_history.append(current_cost)

            if X_val is not None and y_val is not None:
                X_val_b = add_bias_term(X_val)
                val_cost = self._compute_cost(X_val_b, y_val, self.theta)
                self.test_cost_history.append(val_cost)

            if abs(prev_cost - current_cost) < self.tol:
                logger.info(f"Converged at iteration {iteration + 1}, Cost: {current_cost:.6f}")
                break
            prev_cost = current_cost

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts probabilities using the fitted model.

        Args:
            X (np.ndarray): Feature matrix.

        Returns:
            np.ndarray: Predicted probabilities.
        """
        X_b = add_bias_term(X)
        return self._sigmoid(X_b @ self.theta)

    def predict(self, X: np.ndarray, threshold=0.5) -> np.ndarray:
        """
        Predicts binary labels using the fitted model.

        Args:
            X (np.ndarray): Feature matrix.
            threshold (float): Decision threshold.

        Returns:
            np.ndarray: Predicted labels.
        """
        return (self.predict_proba(X) >= threshold).astype(int)


def run_numpy_logistic_regression_gd(X_train: np.ndarray, y_train: np.ndarray,
                                     X_val: np.ndarray, y_val: np.ndarray,
                                     X_test: np.ndarray, y_test: np.ndarray) -> tuple:
    """
    Trains and evaluates the logistic regression model using gradient descent.

    Args:
        X_train, X_val, X_test (np.ndarray): Feature matrices for training, validation, and testing.
        y_train, y_val, y_test (np.ndarray): Target vectors for training, validation, and testing.

    Returns:
        tuple: (Evaluation metrics dict, cost history list)
    """
    logger.info("Running NumPy Logistic Regression (Gradient Descent)...")

    model = NumpyLogisticRegressionGD(learning_rate=0.1, n_iterations=500, batch_size=64, verbose=False, tol=1e-5)
    model.fit(X_train, y_train)

    results = {}
    for split_name, X_split, y_split_true in [("train", X_train, y_train), ("val", X_val, y_val),
                                              ("test", X_test, y_test)]:
        y_pred = model.predict(X_split)
        y_proba = model.predict_proba(X_split)

        metrics = {
            "Accuracy": accuracy_score(y_split_true, y_pred),
            "F1": f1_score(y_split_true, y_pred, zero_division=0),
            "Precision": precision_score(y_split_true, y_pred, zero_division=0),
            "Recall": recall_score(y_split_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_split_true, y_proba) if len(np.unique(y_split_true)) > 1 else "N/A"
        }
        results[split_name] = metrics
        logger.info(
            f"Logistic Regression ({split_name}) - Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, ROC_AUC: {metrics['ROC_AUC'] if isinstance(metrics['ROC_AUC'], float) else 'N/A'}")

    return results, model.cost_history, model.test_cost_history
