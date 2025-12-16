import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import pandas_ta as ta

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the data by replacing infinities with NaN and dropping rows with NaN."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)
    return df


def generate_labels(df: pd.DataFrame, horizon=5, atr_mult=1.5, threshold = 0.2) -> pd.DataFrame:
    df = df.copy()
    df['future_close'] = df['Close'].shift(-horizon)
    df['future_ret'] = (df['future_close'] - df['Close']) / df['Close']

    # Consensus of indicators
    consensus = (
        0.25 * np.sign(df.get('ema_diff_9_21', 0)) +
        0.20 * np.sign(df.get('macd_hist', 0)) +
        0.15 * np.sign(df.get('rsi_norm', 0)) +
        0.15 * df.get('supertrend_dir', 0) +
        0.10 * np.sign(df.get('cci_norm', 0)) +
        0.10 * np.sign(df.get('obv_slope', 0)) +
        0.05 * np.sign(df.get('mfi_norm', 0))
    )
    df['consensus_score'] = consensus.clip(-1,1)

    atr_threshold = df['atr_pct'].rolling(20).mean() * atr_mult
    df['label_prob'] = np.tanh(df['future_ret'] / (atr_threshold + 1e-6)) * df['consensus_score']

    buy_cond = df['label_prob'] > threshold
    sell_cond = df['label_prob'] < -threshold
    df['label'] = 0
    df.loc[buy_cond, 'label'] = 1
    df.loc[sell_cond, 'label'] = -1

    df.dropna(subset=['label_prob'], inplace=True)
    return df



def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    df: pandas DataFrame with columns ['open','high','low','close','volume']
    Returns: df with features for Core 12 indicators ready for ML/backtesting
    """

    df = df.copy()

    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df['Open'] = pd.to_numeric(df['Open'], errors='coerce')
    df['High'] = pd.to_numeric(df['High'], errors='coerce')
    df['Low'] = pd.to_numeric(df['Low'], errors='coerce')
    df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce')

    df = clean_data(df)
    return df