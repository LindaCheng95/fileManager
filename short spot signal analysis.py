import pandas as pd
import numpy as np

def analyze_trading_signals(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Analyzes trading signals by looking forward 20 trading days.
    
    For each row where signal == +1:
      - Finds min of 'low' in the next `window` trading days
      - Finds max of 'high' in the next `window` trading days
      - Returns +1 if abs(min_low) > abs(max_high), else -1
    
    Args:
        df: DataFrame with columns 'low', 'high', 'signal'
        window: Number of trading days to look forward (default 20)
    
    Returns:
        Original DataFrame with 3 new columns:
          - 'future_min_low'   : min of 'low' over next `window` days (NaN if signal != +1)
          - 'future_max_high'  : max of 'high' over next `window` days (NaN if signal != +1)
          - 'dominant_move'    : +1 if abs(future_min_low) > abs(future_max_high), else -1 (NaN if signal != +1)
    """
    df = df.copy()

    df['future_min_low']  = np.nan
    df['future_max_high'] = np.nan
    df['dominant_move']   = np.nan

    signal_idx = df.index[df['signal'] == 1]

    for idx in signal_idx:
        loc = df.index.get_loc(idx)
        start = loc + 1
        end   = start + window

        if start >= len(df):
            continue

        future_slice = df.iloc[start:end]

        min_low  = future_slice['low'].min()
        max_high = future_slice['high'].max()

        df.at[idx, 'future_min_low']  = min_low
        df.at[idx, 'future_max_high'] = max_high
        df.at[idx, 'dominant_move']   = 1 if abs(min_low) > abs(max_high) else -1

    return df