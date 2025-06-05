import logging

import pandas as pd

logger = logging.getLogger(__name__)


def compute_statistics(data):
    logger.debug("Computing statistics...")
    stats = {}

    for column in data.columns:
        if data[column].dtype in ['int64', 'float64']:
            stats[column] = {
                'mean': data[column].mean(),
                'median': data[column].median(),
                'min': data[column].min(),
                'max': data[column].max(),
                'std_dev': data[column].std(),
                '5th_percentile': data[column].quantile(0.05),
                '95th_percentile': data[column].quantile(0.95),
                'missing_values': data[column].isna().sum()
            }
        else:
            stats[column] = {
                'unique_classes': data[column].nunique(),
                'missing_values': data[column].isna().sum(),
                'class_proportions': data[column].value_counts(normalize=True).to_dict()
            }
    logger.debug("Statistics computed")
    return stats


def save_statistics(stats, output_file="data/statistics.csv"):
    logger.debug(f"Saving statistics to {output_file}...")
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(output_file)
    logger.info(f"Statistics saved to file {output_file}")
