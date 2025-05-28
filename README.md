# Comprehensive Data Analysis and Machine Learning Pipeline

## Project Overview

This project provides a robust pipeline for end-to-end data analysis and machine learning model development. It encompasses two main stages:

1.  **Stage 1: Exploratory Data Analysis (EDA) & Visualization**: Focuses on understanding the dataset through statistical summaries and a variety of visualizations.
2.  **Stage 2: Machine Learning Model Development**: Implements a full ML workflow, including data preprocessing, training of multiple model types (Scikit-learn, custom NumPy, and PyTorch), rigorous evaluation, and comparative reporting.

The primary goal is to predict client default payments based on the [UCI Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?select=UCI_Credit_Card.csv).

## Key Features

**Stage 1: Data Analysis**
*   **Versatile Data Loading**: Supports CSV, Excel, JSON, and Parquet file formats.
*   **Descriptive Statistics**: Computes mean, median, standard deviation, percentiles, missing values, and class proportions.
*   **Data Visualization**: Generates key plots for data exploration:
    *   Boxplots (e.g., Education vs. Credit Limit)
    *   Violin plots (e.g., Gender & Credit Limit vs. Default)
    *   Histograms (e.g., Credit Limit Distribution)
    *   Correlation Heatmaps
    *   Regression plots (e.g., Age vs. Credit Limit)

**Stage 2: Machine Learning**
*   **Structured Data Preparation**:
    *   Customizable mapping of categorical values.
    *   Stratified train-validation-test split (60%/20%/20%).
    *   Automatic identification of numerical and categorical features.
*   **Advanced Preprocessing Pipelines**:
    *   Utilizes `sklearn.pipeline.Pipeline` and `sklearn.compose.ColumnTransformer`.
    *   Handles missing values using `SimpleImputer` (median for numerical, most frequent for categorical).
    *   Scales numerical features using `StandardScaler`.
    *   Encodes categorical features using `OneHotEncoder`.
*   **Diverse Model Implementation & Training**:
    *   **Scikit-learn**: Logistic Regression, Decision Tree Classifier, Support Vector Classifier (SVC).
    *   **NumPy (Custom Implementation)**: Logistic Regression with Mini-Batch Gradient Descent. *(A template for Linear Regression with closed-form solution is provided for regression tasks).*
    *   **PyTorch (Custom Implementation)**: Logistic Regression, demonstrating training on both CPU and GPU (if available), with performance comparison.
*   **Comprehensive Model Evaluation**:
    *   Metrics: Accuracy, F1-score, Precision, Recall, ROC AUC.
    *   Evaluation performed on training, validation, and test sets to assess generalization and overfitting.
*   **Centralized Reporting**:
    *   Consolidated table summarizing performance metrics for all models across all datasets.
    *   Includes training times for PyTorch CPU/GPU comparison.
*   **Robust Logging**:
    *   Detailed logging of operations, warnings, and errors to both console and `app.log` file.
    *   Configurable logging levels.

## Prerequisites

*   Python 3.8+
*   PIP (Python package installer)

## Installation

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/your-username/your-repository-name.git # Replace with your actual repository URL
    cd your-repository-name
    ```

2.  **Create and Activate a Virtual Environment (Recommended)**:
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Download Dataset**:
    *   Download the [UCI Credit Card Clients Dataset](https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset?select=UCI_Credit_Card.csv).
    *   Place the `UCI_Credit_Card.csv` file into the `data/` directory in the project root.

## How to Run

The project is divided into two main executable scripts corresponding to its stages.

### Stage 1: Data Analysis & Visualization

This script performs initial data loading, computes statistics, and generates key visualizations.

**Command**:
```bash
python data_analysis_main.py