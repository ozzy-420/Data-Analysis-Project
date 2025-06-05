from utils.config import MAPPING


def get_mapped(data, mapping=MAPPING):
    """Maps values in columns based on the provided mapping dictionary."""
    new_data = data.copy()
    for column, map_dict in mapping.items():
        if column in new_data.columns:
            new_data[column] = new_data[column].map(map_dict).fillna(new_data[column])

    return new_data
