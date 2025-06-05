import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import logging
from sklearn.model_selection import learning_curve

logger = logging.getLogger(__name__)


def plot_learning_curves_sklearn(estimator, title, X, y, cv, output_dir, ylim=None, n_jobs=-1,
                                 train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generates learning curves for a scikit-learn estimator.

    Args:
        estimator: Scikit-learn model or pipeline.
        title (str): Title for the plot.
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray or pd.Series): Target vector.
        cv: Cross-validation strategy.
        output_dir (str): Directory to save the plot..
        ylim (tuple, optional): Limits for the y-axis.
        n_jobs (int, optional): Number of jobs for parallel processing.
        train_sizes (np.ndarray, optional): Training sizes for learning curve.

    Returns:
        None
    """
    logger.info(f"Generating learning curves for: {title}")
    plt.figure(figsize=(10, 6))
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score (Accuracy)")
    plt.grid(True)

    try:
        train_sizes_abs, train_scores, test_scores = learning_curve(
            estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring='accuracy', shuffle=True
        )
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        plt.fill_between(train_sizes_abs, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes_abs, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes_abs, train_scores_mean, 'o-', color="r", label="Training score")
        plt.plot(train_sizes_abs, test_scores_mean, 'o-', color="g", label="Cross-validation score")
        plt.legend(loc="best")
        plt.title(title)

        filename = title.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_").replace(":", "") + ".png"
        plot_path = os.path.join(output_dir, filename)
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Learning curves saved to {plot_path}")

        # Interpretation hints
        final_train_score = train_scores_mean[-1]
        final_cv_score = test_scores_mean[-1]
        gap = final_train_score - final_cv_score

        if final_cv_score < 0.6 and final_train_score < 0.65:
            logger.info(
                f"Potential Underfitting: Both training ({final_train_score:.3f}) "
                f"and CV ({final_cv_score:.3f}) scores are low. "
                "Model may be too simple. Consider increasing complexity "
                "(e.g., more features, polynomial features, more powerful model).")
        elif gap > 0.15 and final_train_score > 0.8:
            logger.info(
                f"Potential Overfitting: High training score ({final_train_score:.3f}) "
                f"but lower CV score ({final_cv_score:.3f}), significant gap ({gap:.3f}). "
                "Model may be too complex. Consider regularization, more data, or feature selection.")
        elif final_cv_score > 0.7 and gap < 0.1:
            logger.info(
                f"Good Fit: CV score ({final_cv_score:.3f}) is reasonably high and close to "
                f"training score ({final_train_score:.3f}). Gap is small ({gap:.3f}).")
        else:
            logger.info(
                f"Scores: Training={final_train_score:.3f}, CV={final_cv_score:.3f}, Gap={gap:.3f}. "
                f"Requires careful interpretation.")
        logger.info(f"--- End Learning Curve Analysis ---")

    except Exception as e:
        logger.error(f"Error generating learning curves for {title}: {e}", exc_info=True)


def plot_cost_history(train_costs, test_costs=None, output_path=None):
    """
    Plots the cost history for gradient descent.

    Args:
        train_costs (list or np.ndarray): Training cost values over epochs.
        test_costs (list or np.ndarray, optional): Validation/Test cost values over epochs.
        output_path (str, optional): Path to save the plot.

    Returns:
        None
    """

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
