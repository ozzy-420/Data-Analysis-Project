from stage1_data_analysis.data_loader import load_data
from stage2_ml_parts.data_preparation import prepare_data_for_ml, MAPPING_DICT
from stage2_ml_parts.build_pipelines import create_preprocessor, get_sklearn_models_with_pipelines
from stage2_ml_parts.train_eval_sklearn import train_and_evaluate_sklearn_model
from stage2_ml_parts.numpy_implementations import (
    run_numpy_logistic_regression_gd
)
from stage2_ml_parts.pytorch_implementations import run_pytorch_logistic_regression
from stage2_ml_parts.reporting import initialize_results_df, add_results_to_df, display_results

import logging
from datetime import datetime


# Default paths and parameters
DEFAULT_DATA_SOURCE = "data/UCI_Credit_Card.csv"
TARGET_COLUMN = 'default.payment.next.month'
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting Stage 2: Machine Learning Models")

    # Initialize results DataFrame
    results_df = initialize_results_df()

    # --- Data Preparation ---
    logger.info("Step 1: Preparing data...")
    raw_data = load_data(DEFAULT_DATA_SOURCE)
    if raw_data.columns[0].lower() == 'id' or raw_data.columns[0] == '':
        raw_data = raw_data.iloc[:, 1:]

    X_train, X_val, X_test, y_train, y_val, y_test, numerical_features, categorical_features = \
        prepare_data_for_ml(raw_data, TARGET_COLUMN, MAPPING_DICT)

    logger.debug(
        f"Data prepared. Dataset sizes: Training X: {X_train.shape}, Validation X: {X_val.shape}, Test X: {X_test.shape}")

    # --- Create Preprocessor ---
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    # --- Stage 3.0: Scikit-learn Models ---
    logger.info("Step 2: Training and evaluating Scikit-learn models (Evaluation 3.0)")
    sklearn_models_with_pipelines = get_sklearn_models_with_pipelines(preprocessor)

    for model_name, pipeline in sklearn_models_with_pipelines.items():
        logger.info(f"Training model: {model_name}")
        metrics_train, metrics_val, metrics_test = train_and_evaluate_sklearn_model(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test
        )
        results_df = add_results_to_df(results_df, model_name, metrics_train, metrics_val, metrics_test)

    # --- Stage 4.0: NumPy Implementations ---
    logger.info("Step 3: NumPy Implementations (Evaluation 4.0)")

    # 4.0a: Linear Regression (Closed Form)
    logger.debug("Running NumPy implementation: Linear Regression (Closed Form)")
    X_train_processed_for_numpy = preprocessor.transform(X_train)
    X_val_processed_for_numpy = preprocessor.transform(X_val)
    X_test_processed_for_numpy = preprocessor.transform(X_test)

    logger.warning(
        "NOTE: NumPy Linear Regression (Closed Form) requires specific data preparation (different target).")

    # 4.0b: Logistic Regression (Gradient Descent)
    logger.debug("Running NumPy implementation: Logistic Regression (Gradient Descent)")
    metrics_numpy_log_reg = run_numpy_logistic_regression_gd(
        X_train_processed_for_numpy, y_train.values,
        X_val_processed_for_numpy, y_val.values,
        X_test_processed_for_numpy, y_test.values,
        num_features=X_train_processed_for_numpy.shape[1]
    )
    if metrics_numpy_log_reg:
        results_df = add_results_to_df(results_df, "NumPy Logistic Regression (GD)",
                                       metrics_numpy_log_reg['train'],
                                       metrics_numpy_log_reg['val'],
                                       metrics_numpy_log_reg['test'])

    # --- Stage 5.5: PyTorch Implementation ---
    logger.info("Step 4: PyTorch Implementation (Evaluation 5.5)")
    logger.debug("Running PyTorch implementation: Logistic Regression")
    metrics_pytorch_log_reg_cpu, metrics_pytorch_log_reg_gpu, time_cpu, time_gpu = run_pytorch_logistic_regression(
        X_train_processed_for_numpy, y_train.values,
        X_val_processed_for_numpy, y_val.values,
        X_test_processed_for_numpy, y_test.values,
        input_dim=X_train_processed_for_numpy.shape[1]
    )
    if metrics_pytorch_log_reg_cpu:
        results_df = add_results_to_df(results_df, "PyTorch Logistic Regression (CPU)",
                                       metrics_pytorch_log_reg_cpu['train'],
                                       metrics_pytorch_log_reg_cpu['val'],
                                       metrics_pytorch_log_reg_cpu['test'],
                                       notes=f"Train time: {time_cpu:.2f}s")
    if metrics_pytorch_log_reg_gpu:
        results_df = add_results_to_df(results_df, "PyTorch Logistic Regression (GPU)",
                                       metrics_pytorch_log_reg_gpu['train'],
                                       metrics_pytorch_log_reg_gpu['val'],
                                       metrics_pytorch_log_reg_gpu['test'],
                                       notes=f"Train time: {time_gpu:.2f}s (GPU availability might vary)")

    # --- Display Results ---
    logger.info("Final evaluation results:")
    display_results(results_df)


if __name__ == "__main__":
    start = datetime.now()
    main()
    logger.info(f"Total process duration: {datetime.now() - start}")