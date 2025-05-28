import pandas as pd
import numpy as np


def initialize_results_df():
    """
    Initializes an empty DataFrame to store model evaluation results.

    Returns:
        pd.DataFrame: Empty DataFrame with predefined columns for results.
    """
    columns = [
        "Model",
        "Train Acc/MSE", "Val Acc/MSE", "Test Acc/MSE",
        "Train F1/R2", "Val F1/R2", "Test F1/R2",
        "Train Precision", "Val Precision", "Test Precision",
        "Train Recall", "Val Recall", "Test Recall",
        "Train ROC_AUC", "Val ROC_AUC", "Test ROC_AUC",
        "Notes"
    ]
    return pd.DataFrame(columns=columns)


def add_results_to_df(df, model_name, metrics_train, metrics_val, metrics_test, is_regression=False, notes=""):
    """
    Adds the evaluation results of a model to the DataFrame.

    Args:
        df (pd.DataFrame): DataFrame to store results.
        model_name (str): Name of the model.
        metrics_train (dict): Metrics for the training set.
        metrics_val (dict): Metrics for the validation set.
        metrics_test (dict): Metrics for the test set.
        is_regression (bool): Flag indicating if the task is regression.
        notes (str): Additional notes about the model.

    Returns:
        pd.DataFrame: Updated DataFrame with the new results.
    """
    if is_regression:
        new_row = {
            "Model": model_name,
            "Train Acc/MSE": metrics_train.get("MSE", np.nan),
            "Val Acc/MSE": metrics_val.get("MSE", np.nan),
            "Test Acc/MSE": metrics_test.get("MSE", np.nan),
            "Train F1/R2": metrics_train.get("R2", np.nan),
            "Val F1/R2": metrics_val.get("R2", np.nan),
            "Test F1/R2": metrics_test.get("R2", np.nan),
            "Train Precision": np.nan, "Val Precision": np.nan, "Test Precision": np.nan,
            "Train Recall": np.nan, "Val Recall": np.nan, "Test Recall": np.nan,
            "Train ROC_AUC": np.nan, "Val ROC_AUC": np.nan, "Test ROC_AUC": np.nan,
            "Notes": notes
        }
    else:
        new_row = {
            "Model": model_name,
            "Train Acc/MSE": metrics_train.get("Accuracy", np.nan),
            "Val Acc/MSE": metrics_val.get("Accuracy", np.nan),
            "Test Acc/MSE": metrics_test.get("Accuracy", np.nan),
            "Train F1/R2": metrics_train.get("F1", np.nan),
            "Val F1/R2": metrics_val.get("F1", np.nan),
            "Test F1/R2": metrics_test.get("F1", np.nan),
            "Train Precision": metrics_train.get("Precision", np.nan),
            "Val Precision": metrics_val.get("Precision", np.nan),
            "Test Precision": metrics_test.get("Precision", np.nan),
            "Train Recall": metrics_train.get("Recall", np.nan),
            "Val Recall": metrics_val.get("Recall", np.nan),
            "Test Recall": metrics_test.get("Recall", np.nan),
            "Train ROC_AUC": metrics_train.get("ROC_AUC", np.nan) if isinstance(metrics_train.get("ROC_AUC"), (int, float)) else np.nan,
            "Val ROC_AUC": metrics_val.get("ROC_AUC", np.nan) if isinstance(metrics_val.get("ROC_AUC"), (int, float)) else np.nan,
            "Test ROC_AUC": metrics_test.get("ROC_AUC", np.nan) if isinstance(metrics_test.get("ROC_AUC"), (int, float)) else np.nan,
            "Notes": notes
        }

    # Initialize the DataFrame if it is None or empty
    if df is None or df.empty:
        df = initialize_results_df()

    # Add the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def display_results(df):
    """
    Displays the results DataFrame in a formatted way.

    Args:
        df (pd.DataFrame): DataFrame containing model evaluation results.
    """
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(float_format="%.4f"))


def add_bias_term(X):
    """
    Adds a column of ones (bias term) to the feature matrix.

    Args:
        X (np.ndarray): Feature matrix.

    Returns:
        np.ndarray: Feature matrix with bias term added.
    """
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
