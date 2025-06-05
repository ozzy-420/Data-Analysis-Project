import logging
from datetime import datetime
import pandas as pd
from stage1_data_analysis.data_loader import load_data
from stage2_ml_parts.data_preparation import prepare_data_for_ml, MAPPING_DICT
from stage2_ml_parts.build_pipelines import create_preprocessor, get_sklearn_models_with_pipelines
from stage2_ml_parts.train_eval_sklearn import train_and_evaluate_sklearn_model
from stage2_ml_parts.numpy_implementations import run_numpy_logistic_regression_gd
from stage2_ml_parts.pytorch_implementations import run_pytorch_logistic_regression
from stage2_ml_parts.reporting import initialize_results_df, add_results_to_df, display_results
from stage2_ml_parts.visualizations import (
    plot_model_comparison_metrics,
    plot_training_times,
    plot_numpy_cost_history,
    plot_pytorch_train_val_curves
)

# Configure logging
logger = logging.getLogger(__name__)

# Default paths and parameters
DEFAULT_DATA_SOURCE = "../data/UCI_Credit_Card.csv"
TARGET_COLUMN = 'default.payment.next.month'


def main():
    logger.info("Starting Stage 2: Machine Learning Models")

    # Initialize results DataFrame
    results_df = initialize_results_df()
    cost_history_numpy_log_reg = None
    history_pytorch_cpu = None
    history_pytorch_gpu = None
    time_cpu_pytorch = float('nan')
    time_gpu_pytorch = float('nan')

    # --- Data Preparation ---
    logger.info("Step 1: Preparing data...")
    raw_data = load_data(DEFAULT_DATA_SOURCE)
    if raw_data.columns[0].lower() == 'id' or raw_data.columns[0] == '':
        raw_data = raw_data.iloc[:, 1:]

    X_train, X_val, X_test, y_train, y_val, y_test, numerical_features, categorical_features = \
        prepare_data_for_ml(raw_data, TARGET_COLUMN, MAPPING_DICT)

    logger.debug(f"Data prepared. Dataset sizes: Training X: {X_train.shape}, Validation X: {X_val.shape}, Test X: {X_test.shape}")

    # --- Create Preprocessor ---
    preprocessor = create_preprocessor(numerical_features, categorical_features)

    # --- Stage 3.0: Scikit-learn Models ---
    logger.info("Step 2: Training and evaluating Scikit-learn models")
    sklearn_models_with_pipelines = get_sklearn_models_with_pipelines(preprocessor)

    for model_name, pipeline in sklearn_models_with_pipelines.items():
        logger.info(f"Training model: {model_name}")
        metrics_train, metrics_val, metrics_test = train_and_evaluate_sklearn_model(
            pipeline, X_train, y_train, X_val, y_val, X_test, y_test
        )
        results_df = add_results_to_df(results_df, model_name, metrics_train, metrics_val, metrics_test)

    # --- Stage 4.0: NumPy Implementations ---
    logger.info("Step 3: NumPy Implementations")

    # Logistic Regression (Gradient Descent)
    logger.debug("Running NumPy implementation: Logistic Regression (Gradient Descent)")
    X_train_processed = preprocessor.transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    numpy_log_reg_output = run_numpy_logistic_regression_gd(
        X_train_processed, y_train.values,
        X_val_processed, y_val.values,
        X_test_processed, y_test.values,
    )
    if numpy_log_reg_output:
        metrics_numpy_log_reg, cost_history_numpy_log_reg = numpy_log_reg_output
        results_df = add_results_to_df(results_df, "NumPy Logistic Regression (GD)",
                                       metrics_numpy_log_reg['train'],
                                       metrics_numpy_log_reg['val'],
                                       metrics_numpy_log_reg['test'])

    # --- Stage 5.5: PyTorch Implementation ---
    logger.info("Step 4: PyTorch Implementation")
    pytorch_output = run_pytorch_logistic_regression(
        X_train_processed, y_train.values,
        X_val_processed, y_val.values,
        X_test_processed, y_test.values,
        input_dim=X_train_processed.shape[1]
    )
    metrics_pytorch_log_reg_cpu, metrics_pytorch_log_reg_gpu, \
        time_cpu_pytorch, time_gpu_pytorch, \
        history_pytorch_cpu, history_pytorch_gpu = pytorch_output

    if metrics_pytorch_log_reg_cpu:
        results_df = add_results_to_df(results_df, "PyTorch Logistic Regression (CPU)",
                                       metrics_pytorch_log_reg_cpu['train'],
                                       metrics_pytorch_log_reg_cpu['val'],
                                       metrics_pytorch_log_reg_cpu['test'],
                                       notes=f"Train time: {time_cpu_pytorch:.2f}s")
    if metrics_pytorch_log_reg_gpu:
        results_df = add_results_to_df(results_df, "PyTorch Logistic Regression (GPU)",
                                       metrics_pytorch_log_reg_gpu['train'],
                                       metrics_pytorch_log_reg_gpu['val'],
                                       metrics_pytorch_log_reg_gpu['test'],
                                       notes=f"Train time: {time_gpu_pytorch:.2f}s")

    # --- Display Results ---
    logger.info("Final evaluation results:")
    display_results(results_df)

    # --- Generate Visualizations ---
    logger.info("Generating report visualizations...")
    try:
        if not results_df.empty:
            plot_model_comparison_metrics(results_df, metric_key='Test Acc/MSE',
                                          title='Model Comparison (Test Accuracy/MSE)',
                                          filename='../output/stage2/model_comparison_test_acc_mse.png')
            plot_model_comparison_metrics(results_df, metric_key='Test F1/R2',
                                          title='Model Comparison (Test F1/R2)',
                                          filename='../output/stage2/model_comparison_test_f1_r2.png')
            if 'Test ROC_AUC' in results_df.columns and results_df['Test ROC_AUC'].notna().any():
                plot_model_comparison_metrics(results_df, metric_key='Test ROC_AUC',
                                              title='Model Comparison (Test ROC AUC)',
                                              filename='../output/stage2/model_comparison_test_roc_auc.png')

        pytorch_times_for_plot = {}
        if "PyTorch Logistic Regression (CPU)" in results_df["Model"].values and not pd.isna(time_cpu_pytorch):
            pytorch_times_for_plot["PyTorch CPU"] = time_cpu_pytorch
        if "PyTorch Logistic Regression (GPU)" in results_df["Model"].values and not pd.isna(time_gpu_pytorch):
            pytorch_times_for_plot["PyTorch GPU"] = time_gpu_pytorch

        if pytorch_times_for_plot:
            plot_training_times(pytorch_times_for_plot, title="PyTorch Logistic Regression Training Times",
                                filename="../output/stage2/pytorch_training_times.png")

        if cost_history_numpy_log_reg:
            plot_numpy_cost_history(cost_history_numpy_log_reg, title="NumPy Logistic Regression (GD) - Cost History",
                                    filename="../output/stage2/numpy_lr_gd_cost_history.png")

        if history_pytorch_cpu:
            plot_pytorch_train_val_curves(history_pytorch_cpu, model_name_suffix=" (CPU)",
                                          filename_prefix="pytorch_lr_cpu")
        if history_pytorch_gpu:
            plot_pytorch_train_val_curves(history_pytorch_gpu, model_name_suffix=" (GPU)",
                                          filename_prefix="pytorch_lr_gpu")

        logger.info("Visualizations saved successfully.")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    start = datetime.now()
    main()
    logger.info(f"Total process duration: {datetime.now() - start}")