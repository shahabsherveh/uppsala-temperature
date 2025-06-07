import pandas as pd
import numpy as np


def read_data(file_path):
    """
    Read the data from the specified file path.
    """
    df = pd.read_csv(
        file_path,
        sep=r"\s+",
    )
    df.columns = ["year", "month", "day", "temp", "temp_corrected", "data_source"]
    return df


def preprocess_data(df, resample, test_size):
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df.set_index("date", inplace=True)
    ts = df.temp_corrected.resample(resample).mean()
    train = ts[:test_size].copy()
    test = ts[-test_size:].copy()
    return train, test
