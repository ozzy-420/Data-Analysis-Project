import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

matplotlib.use('TkAgg')


def create_boxplot(data, output_file, plot_config, x=None, y=None, hue=None):
    """Generates and saves a boxplot."""
    sns.catplot(data=data, x=x, y=y, hue=hue, kind="box")
    plt.savefig(output_file, dpi=plot_config["dpi"], bbox_inches=plot_config["bbox_inches"])
    plt.close()


def create_violinplot(data, output_file, plot_config, x=None, y=None, hue=None):
    """Generates and saves a violin plot."""
    sns.catplot(data=data, x=x, y=y, hue=hue, kind="violin")
    plt.savefig(output_file, dpi=plot_config["dpi"], bbox_inches=plot_config["bbox_inches"])
    plt.close()


def create_histogram(data, output_file, plot_config, x=None, hue=None):
    """Generates and saves a histogram."""
    sns.histplot(data=data, x=x, kde=True, hue=hue)
    plt.savefig(output_file, dpi=plot_config["dpi"], bbox_inches=plot_config["bbox_inches"])
    plt.close()


def create_regression_plot(data, output_file, plot_config, x=None, y=None, hue=None):
    """Generates and saves a regression plot."""
    sns.lmplot(data=data, x=x, y=y, hue=hue)
    plt.savefig(output_file, dpi=plot_config["dpi"], bbox_inches=plot_config["bbox_inches"])
    plt.close()


def create_heatmap(data, output_file, plot_config):
    """Generates and saves a correlation heatmap using only numeric columns."""
    numeric_data = data.select_dtypes(include=["float", "int"])  # Keep only numeric columns
    corr_matrix = numeric_data.corr()  # Compute correlations
    sns.heatmap(corr_matrix, annot=True, cmap=plot_config["palette"], fmt=".2f")
    plt.savefig(output_file, dpi=plot_config["dpi"], bbox_inches=plot_config["bbox_inches"])
    plt.close()



def create_visualizations(data, output_files, plot_config, x=None, y=None, hue=None):
    """Generates and saves all visualizations."""
    create_boxplot(data, output_files["boxplot"], plot_config, x, y, hue)
    create_violinplot(data, output_files["violinplot"], plot_config, x, y, hue)
    create_histogram(data, output_files["histogram_limit_bal"], plot_config, y, hue)
    create_histogram(data, output_files["histogram_age"], plot_config, x, hue)
    create_regression_plot(data, output_files["regression"], plot_config, x, y, hue)
    create_heatmap(data, output_files["heatmap"], plot_config)

    print("✅ All visualizations have been saved!")
