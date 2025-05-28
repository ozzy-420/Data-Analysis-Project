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
    if X.shape[0] == 0:  # Handle empty X
        logger.warning("Attempting to evaluate on empty data. Returning N/A metrics.")
        if is_regression:
            return {"MSE": "N/A", "R2": "N/A"}
        else:
            return {"Accuracy": "N/A", "F1": "N/A", "Precision": "N/A", "Recall": "N/A", "ROC_AUC": "N/A"}

    y_pred = model.predict(X)

    if is_regression:
        metrics = {
            "MSE": mean_squared_error(y_true, y_pred),
            "R2": r2_score(y_true, y_pred)
        }
        logger.debug(f"Regression metrics: {metrics}")
    else:
        roc_auc_val = "N/A"
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X)[:, 1]
                # Check if y_true has more than one class
                if len(np.unique(y_true)) > 1:
                    roc_auc_val = roc_auc_score(y_true, y_proba)
                else:
                    logger.warning(f"Only one class present in y_true for predict_proba. ROC AUC is 'N/A'.")
            except Exception as e:
                logger.warning(f"Could not calculate predict_proba or ROC_AUC: {e}")
        else:
            logger.warning("Model does not have predict_proba method. ROC_AUC will be 'N/A'.")

        metrics = {
            "Accuracy": accuracy_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred, zero_division=0),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "ROC_AUC": roc_auc_val
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
    if X_train.shape[0] == 0:
        logger.error("Training data is empty. Skipping training and evaluation.")
        empty_metrics = {"Accuracy": "N/A", "F1": "N/A", "Precision": "N/A", "Recall": "N/A", "ROC_AUC": "N/A"}
        if is_regression:
            empty_metrics = {"MSE": "N/A", "R2": "N/A"}
        return empty_metrics, empty_metrics, empty_metrics

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
