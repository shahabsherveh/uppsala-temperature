import pandas as pd
import numpy as np
from typing import Literal
import pdb


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


def get_smhi_data(
    freq: str = "h",
    parameter: int = 1,
    station: int = 97510,
    period: Literal[
        "corrected-archive", "latest-month", "latest-day", "latest-hour"
    ] = "corrected-archive",
    skiprows: int = 10,
) -> pd.DataFrame:
    """
    Read data from the SMHI API.
    """
    url = f"https://opendata-download-metobs.smhi.se/api/version/latest/parameter/{parameter}/station/{station}/period/{period}/data.csv"
    df = pd.read_csv(url, sep=";", skiprows=skiprows, usecols=range(4))
    df["timestamp"] = pd.to_datetime(
        df["Datum"] + " " + df["Tid (UTC)"],
    )
    df_new = pd.DataFrame(
        pd.date_range(start=df.timestamp.min(), end=df.timestamp.max(), freq=freq),
        columns=["timestamp"],
    )
    df_new = df_new.join(df.set_index("timestamp"), on="timestamp")

    return df_new


def get_smhi_forecast(
    freq: str = "h",
    parameter: int = 1,
    station: int = 97510,
    period: Literal[
        "corrected-archive", "latest-month", "latest-day", "latest-hour"
    ] = "corrected-archive",
) -> pd.DataFrame:
    """
    Read data from the SMHI API.
    """
    url = f"https://opendata-download-metobs.smhi.se/api/version/latest/parameter/{parameter}/station/{station}/period/{period}/data.csv"
    df = pd.read_csv(url, sep=";", skiprows=10, usecols=range(4))
    df["timestamp"] = pd.to_datetime(
        df["Datum"] + " " + df["Tid (UTC)"],
    )
    df_new = pd.DataFrame(
        pd.date_range(start=df.timestamp.min(), end=df.timestamp.max(), freq=freq),
        columns=["timestamp"],
    )
    df_new = df_new.join(df.set_index("timestamp"), on="timestamp")

    return df_new


def preprocess_smhi_data(
    df: pd.DataFrame, target_col="Lufttemperatur", freq="d", agg="mean"
) -> pd.Series:
    """
    Preprocess the SMHI data.
    """
    df_copy = df.copy()
    df_copy["timestamp"] = pd.to_datetime(df_copy.Datum + " " + df_copy["Tid (UTC)"])
    df_copy.drop(columns=["Datum", "Tid (UTC)"], inplace=True)
    df_copy.set_index("timestamp", inplace=True)
    ts = df_copy[target_col].astype(float)
    ts_final = pd.Series(
        index=pd.date_range(
            start=df_copy.index.min(), end=df_copy.index.max(), freq="h"
        ),
    )
    ts_final[ts.index] = ts
    return getattr(ts_final.resample(freq), agg)()


def preprocess_data(
    df, resample_freq, start_year: str | None = None, end_year: str | None = None
):
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    df.set_index("date", inplace=True)
    ts = df.temp_corrected.resample(resample_freq).mean()
    return ts[start_year:end_year]
