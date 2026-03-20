import pandas as pd
import numpy as np
from pathlib import Path


def read_sheet(file_path: str, file_name: str, sheet_name: str) -> pd.DataFrame:
    """Read a DataFrame from a specific sheet in an Excel file."""
    full_path = Path(file_path) / file_name
    return pd.read_excel(full_path, sheet_name=sheet_name, index_col=0, parse_dates=True)


def get_zscore(
    df: pd.DataFrame,
    series_name: str,
    start_date: str = None,
    end_date: str = None,
) -> pd.Series:
    """
    Extract a named time series, trim leading zeros, and return its z-score.

    Args:
        df:          DataFrame with a DatetimeIndex and named columns.
        series_name: Column name of the time series to extract.
        start_date:  Optional start date (inclusive), e.g. "2020-01-01".
        end_date:    Optional end date (inclusive), e.g. "2023-12-31".

    Returns:
        pd.Series of z-scores with the same DatetimeIndex.
    """
    series = df[series_name]

    # Trim leading zeros
    first_nonzero = series.ne(0).idxmax()
    series = series.loc[first_nonzero:]

    # Optional date slicing
    if start_date:
        series = series.loc[start_date:]
    if end_date:
        series = series.loc[:end_date]

    mean, std = series.mean(), series.std()
    if std == 0:
        raise ValueError(f"Series '{series_name}' has zero variance; z-score is undefined.")

    return (series - mean) / std


# --- Example usage ---
if __name__ == "__main__":
    df = read_sheet(
        file_path="/path/to/folder",
        file_name="data.xlsx",
        sheet_name="Sheet1",
    )

    z = get_zscore(
        df,
        series_name="Revenue",
        start_date="2021-01-01",   # optional
        end_date="2023-12-31",     # optional
    )

    print(z)
