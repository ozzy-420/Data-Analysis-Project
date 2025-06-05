import logging
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from stage1_data_analysis.data_loader import load_data
from stage2_ml_parts.data_preparation import prepare_data_for_ml, MAPPING_DICT
from stage2_ml_parts.build_pipelines import create_preprocessor
from stage2_ml_parts.numpy_implementations import NumpyLogisticRegressionGD
from stage2_ml_parts.numpy_implementations import run_numpy_logistic_regression_gd
from stage3.cross_validation import perform_sklearn_cross_validation, perform_numpy_cross_validation
from stage3.error_analysis import plot_learning_curves_sklearn, plot_cost_history

logger = logging.getLogger(__name__)

DEFAULT_DATA_SOURCE = "../data/UCI_Credit_Card.csv"
TARGET_COLUMN = 'default.payment.next.month'
OUTPUT_DIR_STAGE3 = "../output/stage3"
os.makedirs(OUTPUT_DIR_STAGE3, exist_ok=True)

logger.info("Starting Stage 3: Model Evaluation and Tuning")

# --- Data Loading and Preparation ---
logger.info("Step 0: Loading and preparing data...")

raw_data = load_data(DEFAULT_DATA_SOURCE)
if raw_data.columns[0].lower() == 'id' or raw_data.columns[0] == '':
    raw_data = raw_data.iloc[:, 1:]

X_train_full, X_val_full, X_test_full, y_train_full, y_val_full, y_test_full, \
    numerical_features, categorical_features = prepare_data_for_ml(raw_data, TARGET_COLUMN, MAPPING_DICT)

X_train_val_combined = pd.concat([X_train_full, X_val_full], ignore_index=True)
y_train_val_combined = pd.concat([y_train_full, y_val_full], ignore_index=True)

preprocessor = create_preprocessor(numerical_features, categorical_features)
logger.info("Data loading and preparation complete.")

logger.info("--- Task 1: Cross-validation ---")
model_log_reg_cv = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', LogisticRegression(solver='liblinear', max_iter=1000))
])
perform_sklearn_cross_validation(
    model_log_reg_cv, X_train_val_combined.copy(), y_train_val_combined.copy(),
    n_splits=3, scoring='accuracy', logger=logger, output_dir=OUTPUT_DIR_STAGE3
)

logger.info("Preparing data for NumPy CV...")
preprocessor_cv_numpy = create_preprocessor(numerical_features, categorical_features)
X_train_val_processed_np = preprocessor_cv_numpy.fit_transform(X_train_val_combined.copy())
y_train_val_combined_np = y_train_val_combined.copy().values

perform_numpy_cross_validation(
    model_class=NumpyLogisticRegressionGD,
    X_processed=X_train_val_processed_np,
    y_values=y_train_val_combined_np,
    n_splits=3,
    model_params={'learning_rate': 0.01, 'n_iterations': 100, 'batch_size': 64, 'verbose': False, 'tol': 1e-4},
    logger=logger,
    output_dir=OUTPUT_DIR_STAGE3
)
logger.info("Cross-validation task complete.")

# --- Task 2: Convergence and Error Analysis ---
logger.info("--- Task 2: Convergence and Error Analysis ---")
log_reg_basic_sklearn = LogisticRegression(solver='liblinear', random_state=42, max_iter=1000)

pipeline_lc_simple = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', log_reg_basic_sklearn)
])
plot_learning_curves_sklearn(
    pipeline_lc_simple, "Learning Curves (Logistic Regression - Simple)",
    X_train_val_combined.copy(), y_train_val_combined.copy(),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    output_dir=OUTPUT_DIR_STAGE3
)

# Fit preprocessor first
preprocessor.fit(X_train_val_combined)

# THEN access fitted components
num_imputer = preprocessor.named_transformers_['num'].named_steps['imputer']
num_scaler = preprocessor.named_transformers_['num'].named_steps['scaler']
cat_transformer = preprocessor.named_transformers_['cat']

numerical_transformer_poly = Pipeline(steps=[
    ('imputer', num_imputer),
    ('poly', PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)),
    ('scaler', num_scaler)
])
preprocessor_poly_controlled = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer_poly, numerical_features),
        ('cat', cat_transformer, categorical_features)
    ],
    remainder='passthrough'
)
pipeline_lc_poly_controlled = Pipeline([
    ('preprocessor', preprocessor_poly_controlled),
    ('classifier', log_reg_basic_sklearn)
])
plot_learning_curves_sklearn(
    pipeline_lc_poly_controlled, "Learning Curves (Logistic Regression - Poly Degree 2 on Numeric)",
    X_train_val_combined.copy(), y_train_val_combined.copy(),
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    output_dir=OUTPUT_DIR_STAGE3)

logger.info("Running NumPy Logistic Regression GD and plotting cost curves...")

# Przetworzenie danych
preprocessor_numpy = create_preprocessor(numerical_features, categorical_features)
X_train_np = preprocessor_numpy.fit_transform(X_train_full)
X_test_np = preprocessor_numpy.transform(X_test_full)
X_val_np = preprocessor_numpy.transform(X_val_full)

y_train_np = y_train_full.values
y_test_np = y_test_full.values
y_val_np = y_val_full.values

# Trening modelu i pobranie historii kosztów
_, cost_history, test_cost_history = run_numpy_logistic_regression_gd(
    X_train_np, y_train_np,
    X_val_np, y_val_np,
    X_test_np, y_test_np
)

# Wykres funkcji kosztu
plot_cost_history(
    train_costs=cost_history,
    test_costs=test_cost_history,
    output_path=os.path.join(OUTPUT_DIR_STAGE3, "cost_curve_numpy_logreg.png"),
)

logger.info("Convergence and error analysis task complete.")
