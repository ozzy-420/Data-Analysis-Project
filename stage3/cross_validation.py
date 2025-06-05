import numpy as np
import logging
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score


def perform_sklearn_cross_validation(pipeline, X, y, n_splits, scoring, logger, output_dir=None):
    """
    Performs cross-validation for a scikit-learn pipeline.

    Args:
        pipeline (Pipeline): Scikit-learn pipeline to evaluate.
        X (np.ndarray or pd.DataFrame): Feature matrix.
        y (np.ndarray or pd.Series): Target vector.
        n_splits (int): Number of folds for cross-validation.
        scoring (str): Scoring metric for evaluation.
        logger (logging.Logger): Logger instance for logging.
        output_dir (str, optional): Directory to save results (if applicable).

    Returns:
        None
    """
    logger.info(f"Performing {n_splits}-fold cross-validation for scikit-learn model '{pipeline.steps[-1][0]}'...")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)

    try:
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        logger.info(f"Cross-validation scores ({scoring}): {scores}")
        logger.info(f"Mean CV score: {mean_score:.4f} (+/- {std_score:.4f})")

        # Analyze score consistency
        logger.info("--- CV Score Consistency Analysis ---")
        if std_score > 0.05 and std_score > 0.1 * mean_score:
            logger.warning("Scores across folds vary significantly. This might indicate instability, "
                           "high model variance, or heterogeneous data splits.")
        else:
            logger.info("Scores across folds are relatively consistent.")

    except Exception as e:
        logger.error(f"Error during scikit-learn cross-validation: {e}", exc_info=True)


def perform_numpy_cross_validation(model_class, X_processed, y_values, n_splits, model_params, logger, output_dir=None):
    """
    Performs cross-validation for a custom NumPy-based model.

    Args:
        model_class (class): Custom NumPy model class.
        X_processed (np.ndarray): Preprocessed feature matrix.
        y_values (np.ndarray): Target vector.
        n_splits (int): Number of folds for cross-validation.
        model_params (dict): Parameters for initializing the model.
        logger (logging.Logger): Logger instance for logging.
        output_dir (str, optional): Directory to save results (if applicable).

    Returns:
        None
    """
    logger.info(f"Performing {n_splits}-fold cross-validation for NumPy model '{model_class.__name__}'...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    if X_processed.shape[0] == 0:
        logger.error("Cannot perform NumPy CV: Processed data is empty.")
        return

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_processed, y_values)):
        logger.debug(f"NumPy CV - Fold {fold + 1}/{n_splits}")
        X_train_fold, X_val_fold = X_processed[train_idx], X_processed[val_idx]
        y_train_fold, y_val_fold = y_values[train_idx], y_values[val_idx]

        if X_train_fold.shape[0] == 0 or X_val_fold.shape[0] == 0:
            logger.warning(f"Skipping NumPy CV fold {fold + 1} due to empty train/validation split.")
            continue
        if len(np.unique(y_train_fold)) < 2:
            logger.warning(f"Skipping NumPy CV fold {fold + 1} due to single class in y_train_fold.")
            continue

        try:
            model_instance = model_class(**model_params)
            model_instance.fit(X_train_fold, y_train_fold)
            y_pred_fold = model_instance.predict(X_val_fold)
            score = accuracy_score(y_val_fold, y_pred_fold)
            fold_scores.append(score)
            logger.debug(f"NumPy CV - Fold {fold + 1} score: {score:.4f}")
        except Exception as e:
            logger.error(f"Error during NumPy CV for fold {fold + 1}: {e}", exc_info=True)
            fold_scores.append(np.nan)

    if fold_scores:
        valid_scores = [s for s in fold_scores if not np.isnan(s)]
        if valid_scores:
            mean_score_np = np.mean(valid_scores)
            std_score_np = np.std(valid_scores)
            logger.info(f"NumPy Model Cross-validation scores: {valid_scores}")
            logger.info(f"Mean CV score (NumPy Model): {mean_score_np:.4f} (+/- {std_score_np:.4f})")
            if std_score_np > 0.05 and std_score_np > 0.1 * mean_score_np:
                logger.warning("Scores across folds vary significantly for NumPy model.")
            else:
                logger.info("Scores across folds are relatively consistent for NumPy model.")
        else:
            logger.error("No valid scores obtained from NumPy CV.")
    else:
        logger.error("NumPy CV could not be completed for any fold.")
