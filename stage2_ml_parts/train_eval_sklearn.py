import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Configure logging
logger = logging.getLogger(__name__)


def evaluate_model(model, X, y_true, is_regression=False):
    """
    Evaluates a model and returns performance metrics.

    Args:
        model: Trained model to evaluate.
        X: Features for evaluation.
        y_true: True labels or target values.
        is_regression (bool): Flag indicating if the task is regression.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    logger.debug("Evaluating model...")
    y_pred = model.predict(X)

    if is_regression:
        metrics = {
            "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
        logger.debug(f"Regression metrics: {metrics}")
    else:
        y_proba = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else None
        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_score(y_true, y_proba) if y_proba is not None else "N/A"
        }
        logger.debug(f"Classification metrics: {metrics}")
    return metrics


def train_and_evaluate_sklearn_model(pipeline, X_train, y_train, X_val, y_val, X_test, y_test, is_regression=False):
    """
    Trains and evaluates a Sklearn model on training, validation, and test datasets.

    Args:
        pipeline: Sklearn pipeline containing preprocessing and model.
        X_train: Training features.
        y_train: Training labels or target values.
        X_val: Validation features.
        y_val: Validation labels or target values.
        X_test: Test features.
        y_test: Test labels or target values.
        is_regression (bool): Flag indicating if the task is regression.

    Returns:
        tuple: Metrics for training, validation, and test datasets.
    """
    logger.debug("Training model...")
    pipeline.fit(X_train, y_train)
    logger.debug("Model training completed.")

    logger.debug("Evaluating model...")
    logger.debug("Evaluating model on training data...")
    metrics_train = evaluate_model(pipeline, X_train, y_train, is_regression)

    logger.debug("Evaluating model on validation data...")
    metrics_val = evaluate_model(pipeline, X_val, y_val, is_regression)

    logger.debug("Evaluating model on test data...")
    metrics_test = evaluate_model(pipeline, X_test, y_test, is_regression)

    logger.debug("Evaluation completed.")
    return metrics_train, metrics_val, metrics_test
