import pandas as pd
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

MAPPING_DICT = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {
        0: "Unknown", 1: "Graduate", 2: "University",
        3: "High School", 4: "Others", 5: "Unknown", 6: "Unknown"
    },
    "MARRIAGE": {0: "Unknown", 1: "Married", 2: "Single", 3: "Others"}
}


def apply_custom_mapping(data: pd.DataFrame, mapping_dict: dict) -> pd.DataFrame:
    """
    Maps categorical values in columns based on the provided dictionary.

    Args:
        data (pd.DataFrame): Input DataFrame.
        mapping_dict (dict): Dictionary specifying mappings for categorical columns.

    Returns:
        pd.DataFrame: DataFrame with mapped values.
    """
    logger.info("Applying custom mappings to categorical columns.")
    df = data.copy()
    for column, mapping in mapping_dict.items():
        if column in df.columns:
            logger.debug(f"Mapping column '{column}' using provided dictionary.")
            df[column] = df[column].map(mapping).fillna(df[column])
            # Ensure column type is object after mapping
            if pd.api.types.is_numeric_dtype(data[column]):
                if any(isinstance(v, str) for v in mapping.values()):
                    df[column] = df[column].astype(str)
    return df


def prepare_data_for_ml(raw_data: pd.DataFrame, target_column: str, mapping_dict: dict):
    """
    Prepares data for machine learning by mapping values, splitting datasets, and identifying feature types.

    Args:
        raw_data (pd.DataFrame): Raw input data.
        target_column (str): Name of the target column.
        mapping_dict (dict): Dictionary for mapping categorical values.

    Returns:
        tuple: Split datasets (X_train, X_val, X_test, y_train, y_val, y_test) and feature lists (numerical, categorical).
    """
    logger.info("Starting data preparation for machine learning.")

    # Apply custom mappings
    df = apply_custom_mapping(raw_data, mapping_dict)

    # Define features (X) and target (y)
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Split data into training+validation and test sets
    logger.debug("Splitting data into training+validation and test sets.")
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split training+validation set into training and validation sets
    logger.debug("Splitting training+validation set into training and validation sets.")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    # Identify numerical and categorical features
    logger.debug("Identifying numerical and categorical features.")
    numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    # Ensure mapped columns are treated as categorical if they became objects
    for col in mapping_dict.keys():
        if col in X_train.columns and col not in categorical_features:
            categorical_features.append(col)
            if col in numerical_features:
                numerical_features.remove(col)
        if col in categorical_features and X_train[col].dtype != 'object':
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)

    # Handle potential categorical features stored as numeric
    pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    potential_cats_numeric = [col for col in pay_cols if col in X_train.columns]

    for col in potential_cats_numeric:
        if col in numerical_features:
            logger.debug(f"Converting column '{col}' from numeric to categorical.")
            X_train[col] = X_train[col].astype(str)
            X_val[col] = X_val[col].astype(str)
            X_test[col] = X_test[col].astype(str)
            categorical_features.append(col)
            numerical_features.remove(col)

    # Re-check for unclassified columns
    identified_cols = set(numerical_features) | set(categorical_features)
    all_X_cols = set(X_train.columns)
    missing_cols = all_X_cols - identified_cols

    if missing_cols:
        logger.warning(f"Unclassified columns detected: {missing_cols}. Attempting to classify.")
        for col in missing_cols:
            if X_train[col].dtype == 'object' or X_train[col].nunique() < 20:
                logger.info(f"Column '{col}' classified as categorical.")
                X_train[col] = X_train[col].astype(str)
                X_val[col] = X_val[col].astype(str)
                X_test[col] = X_test[col].astype(str)
                categorical_features.append(col)
            else:
                logger.info(f"Column '{col}' classified as numerical.")
                numerical_features.append(col)

    # Ensure no overlap between numerical and categorical features
    overlap = set(numerical_features) & set(categorical_features)
    if overlap:
        logger.error(f"Overlap detected between numerical and categorical features: {overlap}")
        for col_overlap in overlap:
            numerical_features.remove(col_overlap)
        logger.warning(f"Overlap resolved. Updated numerical features: {sorted(numerical_features)}")

    logger.info(f"Final Numerical features: {sorted(numerical_features)}")
    logger.info(f"Final Categorical features: {sorted(categorical_features)}")

    return X_train, X_val, X_test, y_train, y_val, y_test, sorted(numerical_features), sorted(categorical_features)
