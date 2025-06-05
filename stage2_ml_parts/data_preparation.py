import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from utils.utils import get_mapped
from utils.config import MAPPING

logger = logging.getLogger(__name__)


def prepare_data_for_ml(raw_data: pd.DataFrame, target_column: str, mapping_dict: dict = MAPPING):
    """Prepares data for machine learning by mapping values, splitting datasets, and identifying feature types."""
    logger.info("Starting data preparation.")

    if raw_data.columns[0].lower() == 'id' or raw_data.columns[0] == '':
        raw_data = raw_data.iloc[:, 1:]

    df = get_mapped(raw_data, mapping=mapping_dict)
    X, y = df.drop(columns=[target_column]), df[target_column]

    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42,
                                                      stratify=y_train_val)

    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    for col in mapping_dict.keys():
        if col in X_train.columns:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            categorical_features.append(col)
            numerical_features = [f for f in numerical_features if f != col]

    pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    for col in pay_cols:
        if col in numerical_features:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            categorical_features.append(col)
            numerical_features.remove(col)

    missing_cols = set(X_train.columns) - set(numerical_features) - set(categorical_features)
    for col in missing_cols:
        if X_train[col].nunique() < 20:
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            categorical_features.append(col)
        else:
            numerical_features.append(col)

    overlap = set(numerical_features) & set(categorical_features)
    for col in overlap:
        numerical_features.remove(col)

    return X_train, X_val, X_test, y_train, y_val, y_test, sorted(numerical_features), sorted(categorical_features)
