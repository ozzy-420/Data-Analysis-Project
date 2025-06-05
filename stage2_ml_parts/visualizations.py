import os.path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)
OUTPUT_PATH = os.path.abspath(__file__)
OUTPUT_PATH = os.path.dirname(OUTPUT_PATH)
OUTPUT_PATH = os.path.dirname(OUTPUT_PATH)
OUTPUT_PATH = os.path.join(OUTPUT_PATH, "output", "stage2")


def _filename_to_path(filename):
    """Converts a filename to a full path based on the current working directory."""
    return os.path.join(OUTPUT_PATH, filename)


def plot_model_comparison_metrics(results_df, metric_key, title, filename):
    """Generates and saves a bar chart comparing models based on a specific metric."""
    if results_df.empty or metric_key not in results_df.columns:
        logger.warning(f"Cannot generate plot '{title}'. DataFrame is empty or metric key '{metric_key}' not found.")
        return

    plot_df = results_df.dropna(subset=[metric_key])
    if plot_df.empty:
        logger.warning(f"No valid data to plot for metric '{metric_key}' in '{title}'. All values might be NaN.")
        return

    plt.figure(figsize=(12, 7))
    bars = plt.bar(plot_df['Model'], plot_df[metric_key], color=plt.cm.viridis(np.linspace(0, 1, len(plot_df))))

    plt.xlabel("Model")
    plt.ylabel(
        metric_key.replace("Test ", "").replace(" Acc/MSE", " Accuracy/MSE").replace(" F1/R2", " F1 Score/R2 Score"))
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval):
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plt.ylim()[1], f'{yval:.3f}', ha='center',
                     va='bottom', fontsize=8)

    try:
        plt.savefig(_filename_to_path(filename))
        logger.info(f"Saved plot: {_filename_to_path(filename)}")
    except Exception as e:
        logger.error(f"Failed to save plot {_filename_to_path(filename)}: {e}")
    plt.close()


def plot_training_times(times_dict, title, filename):
    """Generates and saves a bar chart for training times."""
    if not times_dict:
        logger.warning(f"Cannot generate plot '{title}'. Times dictionary is empty.")
        return

    names = list(times_dict.keys())
    values = list(times_dict.values())

    plt.figure(figsize=(8, 6))
    bars = plt.bar(names, values, color=['skyblue', 'lightcoral'])
    plt.xlabel("Configuration")
    plt.ylabel("Time (seconds)")
    plt.title(title)
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()

    for bar in bars:
        yval = bar.get_height()
        if pd.notna(yval):
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.01 * plt.ylim()[1], f'{yval:.2f}s', ha='center',
                     va='bottom')

    try:
        plt.savefig(_filename_to_path(filename))
        logger.info(f"Saved plot: {_filename_to_path(filename)}")
    except Exception as e:
        logger.error(f"Failed to save plot {_filename_to_path(filename)}: {e}")
    plt.close()


def plot_numpy_cost_history(cost_history, title, filename):
    """Generates and saves a line plot of the cost history for NumPy GD."""
    if not cost_history:
        logger.warning(f"Cannot generate plot '{title}'. Cost history is empty.")
        return

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cost_history) + 1), cost_history, marker='o', linestyle='-')
    plt.xlabel("Epoch/Iteration")
    plt.ylabel("Cost")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(_filename_to_path(filename))
        logger.info(f"Saved plot: {_filename_to_path(filename)}")
    except Exception as e:
        logger.error(f"Failed to save plot {_filename_to_path(filename)}: {e}")
    plt.close()


def plot_pytorch_train_val_curves(history, model_name_suffix="", filename_prefix="pytorch_training"):
    """Generates and saves plots for PyTorch training/validation loss and validation accuracy.
    filename_prefix (str): Prefix for the saved plot filenames."""
    if not history or not history.get('train_loss'):
        logger.warning(
            f"Cannot generate PyTorch training curves for {model_name_suffix}. History is empty or incomplete.")
        return

    epochs_range = range(1, len(history['train_loss']) + 1)

    # Plot Loss
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, history['train_loss'], label='Training Loss')
    if 'val_loss' in history and history['val_loss']:
        plt.plot(epochs_range, history['val_loss'], label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss (BCEWithLogitsLoss)")
    plt.title(f"Training and Validation Loss{model_name_suffix}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.savefig(_filename_to_path(f"{filename_prefix}_loss_curves.png"))
        logger.info(f"Saved plot: {_filename_to_path(f'{filename_prefix}_loss_curves.png')}")
    except Exception as e:
        logger.error(f"Failed to save plot {_filename_to_path(f'{filename_prefix}_loss_curves.png')}: {e}")
    plt.close()

    # Plot Accuracy
    if 'val_accuracy' in history and history['val_accuracy']:
        val_accuracy_numeric = [x for x in history['val_accuracy'] if isinstance(x, (int, float))]
        if val_accuracy_numeric:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range[:len(val_accuracy_numeric)], val_accuracy_numeric, label='Validation Accuracy',
                     color='green')
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title(f"Validation Accuracy{model_name_suffix}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            try:
                plt.savefig(_filename_to_path(f"{filename_prefix}_accuracy_curves.png"))
                logger.info(f"Saved plot: {_filename_to_path(f'{filename_prefix}_accuracy_curves.png')}")
            except Exception as e:
                logger.error(f"Failed to save plot {_filename_to_path(f'{filename_prefix}_accuracy_curves.png')}: {e}")
            plt.close()
        else:
            logger.warning(f"No valid numeric validation accuracy data to plot for {model_name_suffix}.")

    # Plot ROC AUC
    if 'val_roc_auc' in history and history['val_roc_auc']:
        val_roc_auc_numeric = [x for x in history['val_roc_auc'] if isinstance(x, (int, float))]
        if val_roc_auc_numeric:
            plt.figure(figsize=(10, 6))
            plt.plot(epochs_range[:len(val_roc_auc_numeric)], val_roc_auc_numeric, label='Validation ROC AUC',
                     color='purple')
            plt.xlabel("Epoch")
            plt.ylabel("ROC AUC")
            plt.title(f"Validation ROC AUC{model_name_suffix}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            try:
                plt.savefig(_filename_to_path(f"{filename_prefix}_roc_auc_curves.png"))
                logger.info(f"Saved plot: {_filename_to_path(f'{filename_prefix}_roc_auc_curves.png')}")
            except Exception as e:
                logger.error(f"Failed to save plot {_filename_to_path(f'{filename_prefix}_roc_auc_curves.png')}: {e}")
            plt.close()
        else:
            logger.warning(f"No valid numeric validation ROC AUC data to plot for {model_name_suffix}.")
