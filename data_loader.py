import pandas as pd
import os


def load_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == ".csv":
        return pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        return pd.read_excel(file_path)
    elif file_extension == ".json":
        return pd.read_json(file_path)
    elif file_extension == ".parquet":
        return pd.read_parquet(file_path)

    raise ValueError("File extension " + str(file_extension) + " not supported")

