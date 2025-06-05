import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)


def plot_learning_curves_sklearn(estimator, title, X, y, cv, output_dir, ylim=None, n_jobs=-1,
                                 train_sizes=np.linspace(.1, 1.0, 5)):
    """Generates learning curves for a scikit-learn estimator."""
    logger.info(f"Generating learning curves for: {title}")
    plt.figure(figsize=(10, 6))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (Accuracy)")
    plt.title(title)
    plt.grid(True)

    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy', shuffle=True
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.plot(train_sizes_abs, train_scores_mean, label='Training Score', color='r')
        plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color='r')
        plt.plot(train_sizes_abs, test_scores_mean, label='Validation Score', color='g')
        plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color='g')
        plt.legend(loc="best")

        filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(":", "") + ".png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Learning curves saved to {plot_path}")

    except Exception as e:
        logger.error(f"Error generating learning curves for {title}: {e}", exc_info=True)


def plot_cost_history(train_costs, test_costs=None, output_path=None):
    """Plots the cost history for gradient descent."""

    logger.info("Generating cost history plot.")
    plt.figure(figsize=(10, 6))
    plt.plot(train_costs, label='Training Cost', color='r')
    if test_costs is not None:
        plt.plot(test_costs, label='Validation/Test Cost', color='g')
    plt.xlabel("Epoch")
    plt.ylabel("Cost")
    plt.title("Cost vs Epoch (Gradient Descent)")
    plt.legend(loc="best")
    plt.grid(True)

    if output_path:
        plt.savefig(output_path)
        plt.close()
        logger.info(f"Cost history plot saved to {output_path}")
    else:
        plt.show()
        logger.info("Cost history plot displayed.")
