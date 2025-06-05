import pandas as pd
import numpy as np


def initialize_results_df():
    """Initializes an empty DataFrame to store model evaluation results."""
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
    """Adds the evaluation results of a model to the DataFrame."""
    def get_metric(metrics_dict, key, default_val=np.nan):
        val = metrics_dict.get(key, default_val)
        if isinstance(val, str) and val == "N/A":
            return np.nan
        return val

    if is_regression:
        new_row = {
            "Model": model_name,
            "Train Acc/MSE": get_metric(metrics_train, "MSE"),
            "Val Acc/MSE": get_metric(metrics_val, "MSE"),
            "Test Acc/MSE": get_metric(metrics_test, "MSE"),
            "Train F1/R2": get_metric(metrics_train, "R2"),
            "Val F1/R2": get_metric(metrics_val, "R2"),
            "Test F1/R2": get_metric(metrics_test, "R2"),
            "Train Precision": np.nan, "Val Precision": np.nan, "Test Precision": np.nan,
            "Train Recall": np.nan, "Val Recall": np.nan, "Test Recall": np.nan,
            "Train ROC_AUC": np.nan, "Val ROC_AUC": np.nan, "Test ROC_AUC": np.nan,
            "Notes": notes
        }
    else:
        new_row = {
            "Model": model_name,
            "Train Acc/MSE": get_metric(metrics_train, "Accuracy"),
            "Val Acc/MSE": get_metric(metrics_val, "Accuracy"),
            "Test Acc/MSE": get_metric(metrics_test, "Accuracy"),
            "Train F1/R2": get_metric(metrics_train, "F1"),
            "Val F1/R2": get_metric(metrics_val, "F1"),
            "Test F1/R2": get_metric(metrics_test, "F1"),
            "Train Precision": get_metric(metrics_train, "Precision"),
            "Val Precision": get_metric(metrics_val, "Precision"),
            "Test Precision": get_metric(metrics_test, "Precision"),
            "Train Recall": get_metric(metrics_train, "Recall"),
            "Val Recall": get_metric(metrics_val, "Recall"),
            "Test Recall": get_metric(metrics_test, "Recall"),
            "Train ROC_AUC": get_metric(metrics_train, "ROC_AUC"),
            "Val ROC_AUC": get_metric(metrics_val, "ROC_AUC"),
            "Test ROC_AUC": get_metric(metrics_test, "ROC_AUC"),
            "Notes": notes
        }

    # Initialize the DataFrame if it is None or empty
    if df is None or df.empty:
        df = initialize_results_df()

    # Add the new row to the DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    return df


def display_results(df):
    """Displays the results DataFrame in a formatted way."""
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print(df.to_string(float_format="%.4f"))


def add_bias_term(X):
    """Adds a column of ones (bias term) to the feature matrix."""
    return np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
