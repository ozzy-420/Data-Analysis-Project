import pandas as pd


# Function computes statistics of a data
def compute_statistics(data):
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

    return stats


# Function saves statistics to the output file
def save_statistics(stats, output_file="data/statistics.csv"):
    stats_df = pd.DataFrame(stats).T
    stats_df.to_csv(output_file)
    print(f"Statystyki zapisano do {output_file}")
