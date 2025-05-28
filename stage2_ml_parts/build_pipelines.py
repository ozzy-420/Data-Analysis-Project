import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


logger = logging.getLogger(__name__)


def create_preprocessor(numerical_features, categorical_features):
    """Creates a Sklearn preprocessor for data transformation."""
    logger.info("Creating preprocessing pipeline for numerical and categorical features.")

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    logger.debug("Numerical transformer created: Imputer (median) and StandardScaler.")

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    logger.debug("Categorical transformer created: Imputer (most frequent) and OneHotEncoder.")

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )
    logger.debug("ColumnTransformer created with numerical and categorical pipelines.")
    return preprocessor


def get_sklearn_models_with_pipelines(preprocessor):
    """Returns a dictionary of Sklearn models integrated with preprocessing pipelines."""
    logger.info("Creating pipelines for Sklearn models.")

    models = {
        "Logistic Regression": LogisticRegression(solver='liblinear', random_state=42, max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "SVC": SVC(probability=True, random_state=42)
    }

    pipelines = {}
    for name, model in models.items():
        pipelines[name] = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        logger.debug(f"Pipeline created for model: {name}.")
    return pipelines
