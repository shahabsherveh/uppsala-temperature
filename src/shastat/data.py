import pandas as pd
import numpy as np
from .utils import BaseConfig


class DataProcessingConfig(BaseConfig):
    """
    Configuration class for data processing.
    """

    def __init__(
        self,
        resample_freq: str = "ME",
        start_year: str | None = None,
        end_year: str | None = None,
    ):
        self.resample_freq = resample_freq
        self.start_year = start_year
        self.end_year = end_year


def read_data(
    file_path,
    sep=r"\s+",
    columns=["year", "month", "day", "temp", "temp_corrected", "data_source"],
):
    """
    Read the data from the specified file path.
    """
    df = pd.read_csv(
        file_path,
        sep=sep,
        names=columns,
    )
    return df


def preprocess_data(
    df, resample_freq, start_year: str | None = None, end_year: str | None = None
):
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df.set_index("date", inplace=True)
    ts = df.temp_corrected.resample(resample_freq).mean()
    return ts[start_year:end_year]
