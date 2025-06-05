import logging
import os
import matplotlib
import threading
from stage1_data_analysis.data_loader import load_data
from stage1_data_analysis.statistics_computations import compute_statistics, save_statistics
from stage1_data_analysis.visualisations_generator import (
    create_boxplot, create_violinplot, create_histogram, create_regression_plot, create_heatmap
)

# Constants
DATA_SOURCE = "../data/UCI_Credit_Card.csv"
OUTPUT_DIR = "../output/stage1"
PLOT_CONFIG = {
    "dpi": 300,
    "bbox_inches": "tight",
    "palette": "coolwarm"
}
MAPPING = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate", 2: "University", 3: "High",
                  4: "Other", 5: "Unknown", 0: "Unknown",
                  6: "Unknown"}
}

# Set Matplotlib backend
matplotlib.use('TkAgg')
logger = logging.getLogger(__name__)


def apply_mapping(data, mapping):
    """Maps values in columns based on the provided mapping dictionary."""
    for column, map_dict in mapping.items():
        if column in data.columns:
            data[column] = data[column].map(map_dict).fillna(data[column])


def generate_statistics(data, output_file):
    """Computes and saves statistics to a CSV file."""
    statistics = compute_statistics(data)
    save_statistics(statistics, output_file=output_file)


logger.info("Starting data visualization and statistics generation.")

os.makedirs(OUTPUT_DIR, exist_ok=True)

output_files = {
    "boxplot": os.path.join(OUTPUT_DIR, "boxplot.png"),
    "violinplot": os.path.join(OUTPUT_DIR, "violinplot.png"),
    "histogram_limit_bal": os.path.join(OUTPUT_DIR, "histogram_limit_bal.png"),
    "histogram_age": os.path.join(OUTPUT_DIR, "histogram_age.png"),
    "heatmap": os.path.join(OUTPUT_DIR, "heatmap_correlation.png"),
    "regression": os.path.join(OUTPUT_DIR, "regression_plot.png"),
    "statistics": os.path.join(OUTPUT_DIR, "statistics.csv")
}

logger.info(f"Loading data from {DATA_SOURCE}.")
data = load_data(DATA_SOURCE)

logger.info("Applying mapping to data.")
apply_mapping(data, MAPPING)

logger.info("Starting statistics computation in a separate thread.")
statistics_thread = threading.Thread(target=generate_statistics,
                                     args=(data, output_files["statistics"]))
statistics_thread.start()

logger.info("Generating visualizations.")
create_heatmap(data, output_files["heatmap"], PLOT_CONFIG)
create_boxplot(data, output_files["boxplot"],
               PLOT_CONFIG, x="EDUCATION", y="LIMIT_BAL")
create_violinplot(data, output_files["violinplot"], PLOT_CONFIG, x="SEX", y="LIMIT_BAL")
create_histogram(data, output_files["histogram_limit_bal"],
                 PLOT_CONFIG, x="LIMIT_BAL", hue="SEX")
create_regression_plot(data, output_files["regression"],
                       PLOT_CONFIG, x="AGE", y="LIMIT_BAL")

statistics_thread.join()
logger.info(f"Key visualizations and statistics saved to: {OUTPUT_DIR}")
