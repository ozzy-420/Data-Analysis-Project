import logging
import os
import matplotlib
import threading
from utils.utils import get_mapped
from utils.config import STAGE1_OUTPUT_DIR, DATA_SOURCE_PATH
from stage1_data_analysis.data_loader import load_data
from stage1_data_analysis.statistics_computations import compute_statistics
from stage1_data_analysis.visualisations_generator import (
    create_boxplot, create_violinplot, create_histogram, create_regression_plot, create_heatmap
)

# Set Matplotlib backend
matplotlib.use('TkAgg')
logger = logging.getLogger(__name__)

logger.info("Starting data visualization and statistics generation.")

output_files = {
    "boxplot": os.path.join(STAGE1_OUTPUT_DIR, "boxplot.png"),
    "violinplot": os.path.join(STAGE1_OUTPUT_DIR, "violinplot.png"),
    "histogram_limit_bal": os.path.join(STAGE1_OUTPUT_DIR, "histogram_limit_bal.png"),
    "histogram_age": os.path.join(STAGE1_OUTPUT_DIR, "histogram_age.png"),
    "heatmap": os.path.join(STAGE1_OUTPUT_DIR, "heatmap_correlation.png"),
    "regression": os.path.join(STAGE1_OUTPUT_DIR, "regression_plot.png"),
    "statistics": os.path.join(STAGE1_OUTPUT_DIR, "statistics.csv")
}

logger.info(f"Loading data from {DATA_SOURCE_PATH}.")
data = load_data(DATA_SOURCE_PATH)

logger.info("Applying mapping to data.")
data = get_mapped(data)

logger.info("Starting statistics computation in a separate thread.")
statistics_thread = threading.Thread(target=compute_statistics,
                                     args=(data, output_files["statistics"]))
statistics_thread.start()

logger.info("Generating visualizations.")
create_heatmap(data, output_files["heatmap"])
create_boxplot(data, output_files["boxplot"],
               x="EDUCATION", y="LIMIT_BAL")
create_violinplot(data, output_files["violinplot"], x="SEX", y="LIMIT_BAL")
create_histogram(data, output_files["histogram_limit_bal"],
                 x="LIMIT_BAL", hue="SEX")
create_regression_plot(data, output_files["regression"],
                       x="AGE", y="LIMIT_BAL")

statistics_thread.join()
logger.info(f"Key visualizations and statistics saved to: {STAGE1_OUTPUT_DIR}")
