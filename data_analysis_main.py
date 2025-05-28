import os
import matplotlib
from stage1_data_analysis.data_loader import load_data
from stage1_data_analysis.statistics_computations import compute_statistics, save_statistics
from stage1_data_analysis.visualisations_generator import (
    create_boxplot, create_violinplot, create_histogram, create_regression_plot, create_heatmap
)
import threading

# Set Matplotlib backend
matplotlib.use('TkAgg')

PLOT_CONFIG = {
    "dpi": 300,
    "bbox_inches": "tight",
    "palette": "coolwarm"
}


def apply_mapping(data, mapping):
    """Maps values in columns based on the provided mapping dictionary."""
    for column, map_dict in mapping.items():
        if column in data.columns:
            data[column] = data[column].map(map_dict).fillna(data[column])


def generate_statistics(data, output_file):
    """Computes and saves statistics to a CSV file."""
    statistics = compute_statistics(data)
    save_statistics(statistics, output_file=output_file)


def main(data_source, output_dir=None, plot_config=None, mapping=None):
    """Generates selected key visualizations for the dataset."""

    if data_source is None:
        raise ValueError("Data source is required")

    if plot_config is None:
        plot_config = PLOT_CONFIG

    data_filename = os.path.splitext(os.path.basename(data_source))[0]
    if output_dir is None:
        output_dir = f"output/{data_filename}_key_visuals"
    os.makedirs(output_dir, exist_ok=True)

    output_files = {
        "boxplot": os.path.join(output_dir, "boxplot.png"),
        "violinplot": os.path.join(output_dir, "violinplot.png"),
        "histogram_limit_bal": os.path.join(output_dir, "histogram_limit_bal.png"),
        "histogram_age": os.path.join(output_dir, "histogram_age.png"),
        "heatmap": os.path.join(output_dir, "heatmap_correlation.png"),
        "regression": os.path.join(output_dir, "regression_plot.png"),
        "statistics": os.path.join(output_dir, "statistics.csv")
    }

    data = load_data(data_source)
    if mapping:
        apply_mapping(data, mapping)

    # Compute statistics in a separate thread
    statistics_thread = threading.Thread(target=generate_statistics, args=(data, output_files["statistics"]))
    statistics_thread.start()

    # Key visualizations
    create_heatmap(data, output_files["heatmap"], plot_config)  # Overall correlations
    create_boxplot(data, output_files["boxplot"], plot_config, x="EDUCATION", y="LIMIT_BAL")  # Education vs. Credit Limit & Default
    create_violinplot(data, output_files["violinplot"], plot_config, x="SEX", y="LIMIT_BAL")  # Gender & Credit Limit vs. Default
    create_histogram(data, output_files["histogram_limit_bal"], plot_config, x="LIMIT_BAL", hue="SEX")  # Credit Limit Distribution by Default
    create_regression_plot(data, output_files["regression"], plot_config, x="AGE", y="LIMIT_BAL")  # Age vs. Credit Limit with Default

    statistics_thread.join()

    print(f"✅ Key visualizations saved to: {output_dir}")


if __name__ == "__main__":
    mapping = {"SEX": {1: "Male", 2: "Female"}, # (1=graduate school, 2=university, 3=high school, 4=others, 5=unknown, 6=unknown)
                   "EDUCATION" : {1 : "Graduate", 2 : "University", 3: "High", 4: "Other", 5: "Unknown", 0: "Unknown", 6: "Unknown"}}

    main(data_source="data/UCI_Credit_Card.csv", mapping=mapping)
