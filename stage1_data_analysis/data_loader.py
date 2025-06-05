import logging

import pandas as pd
import os

logger = logging.getLogger(__name__)


def load_data(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()
    logger.debug(f"Loading data from extension {file_extension}")
    data = None

    if file_extension == ".csv":
        data = pd.read_csv(file_path)
    elif file_extension in [".xls", ".xlsx"]:
        data = pd.read_excel(file_path)
    elif file_extension == ".json":
        data = pd.read_json(file_path)
    elif file_extension == ".parquet":
        data = pd.read_parquet(file_path)

    if data is not None:
        logger.debug("Data loaded successfully!")
        return data

    raise ValueError("File extension " + str(file_extension) + " not supported")
